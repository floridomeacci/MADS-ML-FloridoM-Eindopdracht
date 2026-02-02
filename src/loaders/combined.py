"""
Combined Embeddings Loader
--------------------------
Loads and concatenates Dutch + Roberta embeddings (1536-dim = 768 + 768).
Caches the merged dataset to avoid re-processing every run.
"""

from pathlib import Path
from torch.utils.data import DataLoader
from vectormesh.components.padding import FixedPadding
from datasets import Dataset
import torch
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorizers import load_cached_embeddings, prepare_multimodal_datasets, filter_to_rare_classes, balance_classes

# Configuration  
HIDDEN_SIZE = 1536  # 768 + 768 concatenated
DUTCH_PATTERN = "*legal_dutch*"
ROBERTA_PATTERN = "*legal_bert*"
DUTCH_COL = "legal_dutch"
ROBERTA_COL = "legal_bert"

# Cache paths for combined datasets
CACHE_DIR = Path("data/input/combined_cache")
TRAIN_CACHE = CACHE_DIR / "train"
VALID_CACHE = CACHE_DIR / "valid"

# Rare classes (15 classes with <100 samples)
RARE_CLASSES = [8, 10, 13, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31]


def _load_or_create_combined_cache(num_classes=32):
    """Load combined dataset from cache, or create and save it."""
    
    # Check if cache exists
    if TRAIN_CACHE.exists() and VALID_CACHE.exists():
        print(">>> Loading COMBINED dataset from cache (skipping conversion)...")
        train_ds = Dataset.load_from_disk(str(TRAIN_CACHE))
        valid_ds = Dataset.load_from_disk(str(VALID_CACHE))
        print(f"    Loaded {len(train_ds)} train, {len(valid_ds)} valid samples")
        return train_ds, valid_ds
    
    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(">>> Creating combined cache (one-time operation)...")
    data_path = Path("data/input")
    
    # Load cached embeddings from both sources
    dutch_train_cache, dutch_valid_cache = load_cached_embeddings(data_path, pattern=DUTCH_PATTERN)
    roberta_train_cache, roberta_valid_cache = load_cached_embeddings(data_path, pattern=ROBERTA_PATTERN)
    
    # Prepare multimodal datasets (merges both embedding types)
    train_ds, valid_ds = prepare_multimodal_datasets(
        dutch_train_cache, roberta_train_cache,
        dutch_valid_cache, roberta_valid_cache,
        num_classes=num_classes
    )
    
    # Save to cache
    print(">>> Saving combined dataset to cache...")
    train_ds.save_to_disk(str(TRAIN_CACHE))
    valid_ds.save_to_disk(str(VALID_CACHE))
    print(f"    Saved to {CACHE_DIR}")
    
    return train_ds, valid_ds


class CollateConcatenated:
    """Collate function that concatenates two embedding sources into one tensor."""
    
    def __init__(self, vec1_col: str, vec2_col: str, target_col: str, padder):
        self.vec1_col = vec1_col
        self.vec2_col = vec2_col
        self.target_col = target_col
        self.padder = padder
    
    def __call__(self, batch):
        embeddings1 = [item[self.vec1_col] for item in batch]
        embeddings2 = [item[self.vec2_col] for item in batch]
        X1 = self.padder(embeddings1)  # [batch, chunks, 768]
        X2 = self.padder(embeddings2)  # [batch, chunks, 768]
        X = torch.cat([X1, X2], dim=-1)  # [batch, chunks, 1536]
        y = torch.stack([item[self.target_col] for item in batch]).float()
        return X, y


def get_loaders(batch_size=32, max_chunks=30, num_classes=32):
    """
    Get train and validation loaders for combined embeddings.
    Uses cached combined dataset (created once, reused on subsequent runs).
    
    Returns:
        train_loader, valid_loader, hidden_size (1536)
    """
    # Load from cache or create it
    train_ds, valid_ds = _load_or_create_combined_cache(num_classes=num_classes)
    
    # Create collate function that concatenates embeddings
    collate_fn = CollateConcatenated(
        vec1_col=DUTCH_COL, 
        vec2_col=ROBERTA_COL, 
        target_col="onehot", 
        padder=FixedPadding(max_chunks)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return train_loader, valid_loader, HIDDEN_SIZE


def get_balanced_class_loaders(batch_size=32, max_chunks=30, seed=42):
    """
    Get train and validation loaders with all classes balanced to minimum count.
    Uses the cached combined dataset but applies balancing.
    
    Returns:
        train_loader, valid_loader, hidden_size (1536)
    """
    from collections import defaultdict
    import random
    
    random.seed(seed)
    
    # Load from cache (this already has both embedding columns)
    train_ds, valid_ds = _load_or_create_combined_cache(num_classes=32)
    
    # Balance training set
    def count_class_samples(dataset):
        counts = defaultdict(int)
        for example in dataset:
            # onehot is already there, find which classes are 1
            onehot = example['onehot']
            for i, val in enumerate(onehot):
                if val > 0.5:
                    counts[i] += 1
        return counts
    
    def balance_dataset(dataset, target_count, num_classes=32):
        class_to_indices = defaultdict(list)
        for idx, example in enumerate(dataset):
            onehot = example['onehot']
            for i, val in enumerate(onehot):
                if val > 0.5:
                    class_to_indices[i].append(idx)
        
        selected_indices = set()
        for class_id in range(num_classes):
            indices = class_to_indices.get(class_id, [])
            if len(indices) > 0:
                n_select = min(target_count, len(indices))
                selected = random.sample(indices, n_select)
                selected_indices.update(selected)
        
        return dataset.select(sorted(list(selected_indices)))
    
    print("Balancing combined training set...")
    train_counts = count_class_samples(train_ds)
    min_count = min(train_counts.values())
    print(f"Class sample counts: min={min_count}, max={max(train_counts.values())}")
    print(f"Balancing all classes to {min_count} samples each")
    
    train_ds = balance_dataset(train_ds, min_count)
    print(f"Balanced: {len(train_ds)} train, {len(valid_ds)} valid")
    
    # Create collate function that concatenates embeddings
    collate_fn = CollateConcatenated(
        vec1_col=DUTCH_COL, 
        vec2_col=ROBERTA_COL, 
        target_col="onehot", 
        padder=FixedPadding(max_chunks)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return train_loader, valid_loader, HIDDEN_SIZE


def get_rare_class_loaders(batch_size=32, max_chunks=30):
    """
    Get train and validation loaders for rare classes only (15 classes).
    
    Returns:
        train_loader, valid_loader, hidden_size (1536)
    """
    data_path = Path("data/input")
    
    # Load cached embeddings from both sources
    dutch_train_cache, dutch_valid_cache = load_cached_embeddings(data_path, pattern=DUTCH_PATTERN)
    roberta_train_cache, roberta_valid_cache = load_cached_embeddings(data_path, pattern=ROBERTA_PATTERN)
    
    # Filter to rare classes
    dutch_train_rare, dutch_valid_rare = filter_to_rare_classes(
        dutch_train_cache, dutch_valid_cache, 
        rare_classes=RARE_CLASSES, 
        num_classes=len(RARE_CLASSES)
    )
    roberta_train_rare, roberta_valid_rare = filter_to_rare_classes(
        roberta_train_cache, roberta_valid_cache, 
        rare_classes=RARE_CLASSES, 
        num_classes=len(RARE_CLASSES)
    )
    
    # Combine embeddings into single dataset
    def add_roberta_column(dutch_ds, roberta_ds):
        """Add roberta embeddings as a column to dutch dataset"""
        roberta_embs = [row.tolist() if hasattr(row, 'tolist') else list(row) 
                       for row in roberta_ds[DUTCH_COL]]
        return dutch_ds.add_column(ROBERTA_COL, roberta_embs)
    
    train_ds = add_roberta_column(dutch_train_rare, roberta_train_rare)
    valid_ds = add_roberta_column(dutch_valid_rare, roberta_valid_rare)
    
    # Create collate function that concatenates embeddings
    collate_fn = CollateConcatenated(
        vec1_col=DUTCH_COL, 
        vec2_col=ROBERTA_COL, 
        target_col="onehot", 
        padder=FixedPadding(max_chunks)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return train_loader, valid_loader, HIDDEN_SIZE
