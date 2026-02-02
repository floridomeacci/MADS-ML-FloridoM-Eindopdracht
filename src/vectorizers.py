"""
Vectorizer Management Module

This module handles loading and managing pre-cached vector embeddings
for legal text classification.

Key Functions:
- load_cached_embeddings: Load Legal BERT and Legal Dutch embeddings
- prepare_datasets: Apply one-hot encoding and prepare for training
- create_dataloaders: Set up DataLoader with proper collation
"""

from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import DataLoader

from vectormesh.data import Collate, OneHot
from vectormesh.data.dataset import CollateParallel
from vectormesh.data.cache import VectorCache
from vectormesh.components import FixedPadding

# Base paths for data organization
DATA_INPUT_PATH = Path("data/input")
DATA_OUTPUT_PATH = Path("data/output")


def load_cached_embeddings(
    artefacts_dir: Path = DATA_INPUT_PATH,
    pattern: str = "*bert*"
) -> Tuple[VectorCache, VectorCache]:
    """
    Load pre-cached vector embeddings for training and validation.
    
    The embeddings are pre-computed using Legal BERT models to avoid
    redundant processing. This significantly speeds up experimentation.
    
    Args:
        artefacts_dir: Path to the directory containing cached embeddings (default: data/input)
        pattern: Glob pattern to match cache directories
        
    Returns:
        Tuple of (training_cache, validation_cache)
        
    Example:
        >>> train_cache, valid_cache = load_cached_embeddings()
    """
    train_path = next(artefacts_dir.glob(f"{pattern}train/"))
    valid_path = next(artefacts_dir.glob(f"{pattern}valid/"))
    
    train_cache = VectorCache.load(path=train_path)
    valid_cache = VectorCache.load(path=valid_path)
    
    print(f"Loaded training cache: {train_path.name}")
    print(f"  Documents: {len(train_cache.dataset)}")
    print(f"Loaded validation cache: {valid_path.name}")
    print(f"  Documents: {len(valid_cache.dataset)}")
    
    return train_cache, valid_cache


def prepare_datasets(
    train_cache: VectorCache,
    valid_cache: VectorCache,
    num_classes: int = 32,
    label_col: str = "labels",
    target_col: str = "onehot"
):
    """
    Prepare datasets with one-hot encoded labels.
    
    Multi-label classification requires one-hot encoding where each
    document can have multiple active labels (legal facts).
    
    Args:
        train_cache: Training vector cache
        valid_cache: Validation vector cache
        num_classes: Number of legal fact categories
        label_col: Column name containing label lists
        target_col: Column name for one-hot encoded targets
        
    Returns:
        Tuple of (train_dataset, valid_dataset)
    """
    onehot = OneHot(
        num_classes=num_classes,
        label_col=label_col,
        target_col=target_col
    )
    
    train_dataset = train_cache.dataset.map(onehot)
    valid_dataset = valid_cache.dataset.map(onehot)
    
    print(f"Prepared {len(train_dataset)} training samples")
    print(f"Prepared {len(valid_dataset)} validation samples")
    
    return train_dataset, valid_dataset


def create_dataloaders(
    train_dataset,
    valid_dataset,
    embedding_col: str = "legal_dutch",
    target_col: str = "onehot",
    batch_size: int = 32,
    max_chunks: int = 30,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders with proper collation and padding.
    
    Documents are split into chunks and padded to fixed length.
    The collate function handles batching of variable-length sequences.
    
    Args:
        train_dataset: Training dataset with embeddings
        valid_dataset: Validation dataset with embeddings
        embedding_col: Column name containing vector embeddings
        target_col: Column name containing targets
        batch_size: Number of samples per batch
        max_chunks: Maximum number of chunks to pad to
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, valid_loader)
        
    Note:
        max_chunks=30 is chosen based on document length distribution.
        Longer documents are truncated, shorter ones are padded.
    """
    collate_fn = Collate(
        embedding_col=embedding_col,
        target_col=target_col,
        padder=FixedPadding(max_chunks=max_chunks)
    )
    
    # Optional deterministic shuffling for fairness across datasets
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    def _seed_worker(worker_id: int):
        if seed is None:
            return
        # Ensures each worker has a deterministic but distinct seed
        worker_seed = seed + worker_id
        import random
        import numpy as np
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if seed is not None else None,
    )
    
    print(f"Created dataloaders:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(valid_loader)}")
    
    return train_loader, valid_loader


# ========================= MULTIMODAL HELPERS ===============================
def _find_embedding_col(cache: VectorCache, preferred: Optional[List[str]] = None) -> str:
    """Heuristic to find the vector column name inside a cache dataset.

    Different caches may store the embedding column under different names
    (e.g., "legal_dutch" for both models). This helper picks a sensible one.
    """
    cols = list(cache.dataset.column_names)
    if preferred is None:
        preferred = [
            "legal_dutch",
            "legal_bert",
            "legal_dutch_english",
            "embeddings",
            "vectors",
        ]
    for name in preferred:
        if name in cols:
            return name
    # Fallback: choose the first non-target/text column
    exclude = {"text", "labels", "onehot"}
    for name in cols:
        if name not in exclude:
            return name
    raise ValueError(f"No embedding column found. Available columns: {cols}")


def prepare_multimodal_datasets(
    dutch_cache: VectorCache,
    roberta_cache: VectorCache,
    dutch_valid: VectorCache,
    roberta_valid: VectorCache,
    num_classes: int = 32,
    target_col: str = "onehot",
    dutch_col: str = "legal_dutch",
    roberta_col: str = "legal_bert",
):
    """Create datasets that contain BOTH embedding columns + one-hot targets.

    We take the Dutch cache as the base and add the Roberta vectors as a new
    column (and the same for the valid split). Assumes both caches were built
    from the same split so ordering/length matches.
    """
    import datasets.config
    
    onehot = OneHot(num_classes=num_classes, label_col="labels", target_col=target_col)

    # Resolve actual source column names inside each cache
    dutch_src = _find_embedding_col(dutch_cache)
    rob_src = _find_embedding_col(roberta_cache)

    # Disable fingerprinting to avoid slow hashing of large tensors
    old_caching = datasets.config.HF_DATASETS_CACHE
    datasets.config.HF_DATASETS_CACHE = None
    
    try:
        # Train: start from Dutch cache so we preserve its embedding column
        train_ds = dutch_cache.dataset.map(onehot, load_from_cache_file=False, desc="OneHot train")
        # If the Dutch source column name differs from dutch_col, ensure we expose it under dutch_col
        if dutch_src != dutch_col:
            if dutch_col in train_ds.column_names:
                pass  # already present
            else:
                train_ds = train_ds.add_column(dutch_col, train_ds[dutch_src])
        # Add Roberta vectors under a distinct name - convert to list format
        print(f"Converting Roberta train embeddings...")
        roberta_data = [row.tolist() if isinstance(row, torch.Tensor) else row for row in roberta_cache.dataset[rob_src]]
        # Ensure same length
        if len(roberta_data) != len(train_ds):
            raise ValueError(f"Roberta train has {len(roberta_data)} rows but Dutch has {len(train_ds)}")
        train_ds = train_ds.add_column(roberta_col, roberta_data)

        # Valid: same process
        valid_ds = dutch_valid.dataset.map(onehot, load_from_cache_file=False, desc="OneHot valid")
        if dutch_src != dutch_col:
            if dutch_col not in valid_ds.column_names:
                valid_ds = valid_ds.add_column(dutch_col, valid_ds[dutch_src])
        print(f"Converting Roberta valid embeddings...")
        roberta_valid_data = [row.tolist() if isinstance(row, torch.Tensor) else row for row in roberta_valid.dataset[rob_src]]
        # Ensure same length
        if len(roberta_valid_data) != len(valid_ds):
            raise ValueError(f"Roberta valid has {len(roberta_valid_data)} rows but Dutch has {len(valid_ds)}")
        valid_ds = valid_ds.add_column(roberta_col, roberta_valid_data)
    finally:
        datasets.config.HF_DATASETS_CACHE = old_caching

    return train_ds, valid_ds


def create_multimodal_dataloaders(
    train_dataset,
    valid_dataset,
    dutch_col: str = "legal_dutch",
    roberta_col: str = "legal_bert",
    target_col: str = "onehot",
    batch_size: int = 32,
    max_chunks: int = 30,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders that return ((emb_dutch, emb_roberta), targets).

    Uses CollateParallel so each batch contains two embedding tensors with the
    same padding scheme, ready for a Parallel → Concatenate pipeline.
    """
    collate_fn = CollateParallel(
        vec1_col=dutch_col,
        vec2_col=roberta_col,
        target_col=target_col,
        padder=FixedPadding(max_chunks=max_chunks),
    )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    def _seed_worker(worker_id: int):
        if seed is None:
            return
        worker_seed = seed + worker_id
        import random
        import numpy as np
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if seed is not None else None,
    )

    print("Created multimodal dataloaders:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(valid_loader)}")

    return train_loader, valid_loader


# ========================= RARE CLASS FILTERING ===============================
def filter_to_rare_classes(
    train_cache: VectorCache,
    valid_cache: VectorCache,
    rare_classes: list[int],
    num_classes: int = 15,
) -> Tuple:
    """
    Filter datasets to only samples containing target rare classes and remap labels.
    
    Args:
        train_cache: Training cache
        valid_cache: Validation cache
        rare_classes: List of rare class IDs to keep (e.g., [8, 10, 13, ...])
        num_classes: Number of classes in filtered dataset (len(rare_classes))
        
    Returns:
        Tuple of (filtered_train_ds, filtered_valid_ds) with 'onehot' column
    """
    def filter_fn(example):
        """Keep only samples that have at least one rare class label"""
        labels = example['labels']
        return any(c in labels for c in rare_classes)
    
    def remap_and_onehot_fn(example):
        """Convert from 32-class labels to one-hot vector of rare classes only"""
        old_labels = example['labels']
        # Create one-hot vector: 1 if rare class present, 0 otherwise
        onehot = torch.tensor([1.0 if c in old_labels else 0.0 for c in rare_classes])
        example['onehot'] = onehot
        return example
    
    # Filter and remap training set
    print(f"Filtering training set to {len(rare_classes)} rare classes...")
    train_ds = train_cache.dataset.filter(filter_fn)
    train_ds = train_ds.map(remap_and_onehot_fn)
    
    # Filter and remap validation set
    print(f"Filtering validation set to {len(rare_classes)} rare classes...")
    valid_ds = valid_cache.dataset.filter(filter_fn)
    valid_ds = valid_ds.map(remap_and_onehot_fn)
    
    print(f"Filtered: {len(train_ds)} train, {len(valid_ds)} valid")
    
    return train_ds, valid_ds


def balance_classes(
    train_cache: VectorCache,
    valid_cache: VectorCache,
    num_classes: int = 32,
    seed: int = 42,
) -> Tuple:
    """
    Downsample all classes to the minimum class count for balanced training.
    
    Multi-label aware: A document can belong to multiple classes. We count how many
    documents contain each class, find the minimum, then sample documents such that
    each class appears roughly the same number of times.
    
    Args:
        train_cache: Training cache
        valid_cache: Validation cache  
        num_classes: Total number of classes (32)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (balanced_train_ds, balanced_valid_ds) with 'onehot' column
    """
    import random
    from collections import defaultdict
    
    random.seed(seed)
    
    def count_class_samples(dataset):
        """Count how many samples contain each class"""
        counts = {i: 0 for i in range(num_classes)}  # Initialize all classes to 0
        for example in dataset:
            for label in example['labels']:
                label_int = int(label)  # Convert tensor to int
                if label_int < num_classes:
                    counts[label_int] += 1
        return counts
    
    def balance_dataset(dataset, target_count):
        """Downsample to target_count per class"""
        # Group sample indices by which classes they contain
        class_to_indices = defaultdict(list)
        for idx, example in enumerate(dataset):
            for label in example['labels']:
                label_int = int(label)  # Convert tensor to int
                if label_int < num_classes:
                    class_to_indices[label_int].append(idx)
        
        # For each class, randomly select target_count indices
        selected_indices = set()
        for class_id in range(num_classes):
            indices = class_to_indices.get(class_id, [])
            if len(indices) > 0:
                # Sample min(target_count, available) indices
                n_select = min(target_count, len(indices))
                selected = random.sample(indices, n_select)
                selected_indices.update(selected)
        
        # Return dataset filtered to selected indices
        selected_list = sorted(list(selected_indices))
        return dataset.select(selected_list)
    
    def add_onehot(example):
        """Add one-hot encoding for all 32 classes"""
        labels = example['labels']
        onehot = torch.tensor([1.0 if c in labels else 0.0 for c in range(num_classes)])
        example['onehot'] = onehot
        return example
    
    # Count samples per class in training set
    print("Counting samples per class...")
    train_counts = count_class_samples(train_cache.dataset)
    
    # Only consider classes that actually have samples
    present_counts = {k: v for k, v in train_counts.items() if v > 0}
    min_count = min(present_counts.values())
    max_count = max(present_counts.values())
    
    print(f"Classes with samples: {len(present_counts)}/{num_classes}")
    print(f"Class sample counts: min={min_count}, max={max_count}")
    print(f"Balancing all classes to {min_count} samples each")
    
    # Show distribution
    print("Per-class counts:", dict(sorted(train_counts.items())))
    
    # Balance training set
    print("Balancing training set...")
    train_ds = balance_dataset(train_cache.dataset, min_count)
    train_ds = train_ds.map(add_onehot)
    
    # For validation, we don't balance - keep all samples but add onehot
    print("Preparing validation set (unbalanced)...")
    valid_ds = valid_cache.dataset.map(add_onehot)
    
    print(f"Balanced: {len(train_ds)} train, {len(valid_ds)} valid")
    
    return train_ds, valid_ds
