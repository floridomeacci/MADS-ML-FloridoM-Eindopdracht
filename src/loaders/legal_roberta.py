"""
Roberta Embeddings Loader
-------------------------
Loads and prepares Legal-BERT embeddings (768-dim).
Model: Gerwin/legal-bert-dutch-english
"""

from pathlib import Path
from torch.utils.data import DataLoader
from vectormesh.data.dataset import Collate
from vectormesh.components.padding import FixedPadding

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectorizers import load_cached_embeddings, prepare_datasets, filter_to_rare_classes, balance_classes

# Configuration
HIDDEN_SIZE = 768
EMBEDDING_COL = "legal_dutch"  # Column name in the cache (Roberta uses same column name)
PATTERN = "*legal_bert*"

# Rare classes (15 classes with <100 samples)
RARE_CLASSES = [8, 10, 13, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 31]


def get_loaders(batch_size=32, max_chunks=30, num_classes=32):
    """
    Get train and validation loaders for Roberta embeddings.
    
    Returns:
        train_loader, valid_loader, hidden_size (768)
    """
    data_path = Path("data/input")
    
    # Load cached embeddings
    train_cache, valid_cache = load_cached_embeddings(data_path, pattern=PATTERN)
    train_ds, valid_ds = prepare_datasets(train_cache, valid_cache, num_classes=num_classes)
    
    # Create collate function
    collate_fn = Collate(
        embedding_col=EMBEDDING_COL, 
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
        train_loader, valid_loader, hidden_size (768)
    """
    data_path = Path("data/input")
    
    # Load cached embeddings
    train_cache, valid_cache = load_cached_embeddings(data_path, pattern=PATTERN)
    
    # Filter to rare classes
    train_ds, valid_ds = filter_to_rare_classes(
        train_cache, valid_cache, 
        rare_classes=RARE_CLASSES, 
        num_classes=len(RARE_CLASSES)
    )
    
    # Create collate function
    collate_fn = Collate(
        embedding_col=EMBEDDING_COL, 
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
    
    Returns:
        train_loader, valid_loader, hidden_size (768)
    """
    data_path = Path("data/input")
    
    # Load cached embeddings
    train_cache, valid_cache = load_cached_embeddings(data_path, pattern=PATTERN)
    
    # Balance all classes to minimum count
    train_ds, valid_ds = balance_classes(
        train_cache, valid_cache, 
        num_classes=32,
        seed=seed
    )
    
    # Create collate function
    collate_fn = Collate(
        embedding_col=EMBEDDING_COL, 
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
