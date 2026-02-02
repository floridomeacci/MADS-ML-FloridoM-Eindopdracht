"""
Experiment 10: Adaptive Oversampling
====================================
Balance training by oversampling rare classes:
- Count documents per class
- Compute oversample factor so each class has ~equal representation
- Rare classes get duplicated more times

Goal: Equal representation at the sample level across all classes.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
import json
from datetime import datetime


def count_class_distribution(labels_list: list) -> dict:
    """Count documents per class."""
    counts = Counter()
    for labels in labels_list:
        for class_id in labels:
            counts[class_id] += 1
    return dict(counts)


def compute_oversample_factors(class_counts: dict, target_per_class: int = 3000) -> dict:
    """
    Compute how many times to duplicate each class's samples.
    """
    factors = {}
    
    max_count = max(class_counts.values())
    
    print(f"\nClass distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        # How many times to duplicate to reach target?
        factor = max(1, int(np.ceil(target_per_class / count)))
        factors[class_id] = factor
        print(f"  Class {class_id:2d}: {count:5d} docs × {factor:2d} = ~{count * factor:6d}")
    
    return factors


class SimpleClassifier(nn.Module):
    """Simple MLP classifier."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def run(
    embedding_type: str = 'dutch',
    target_per_class: int = 3000,
    epochs: int = 25,
    hidden_dim: int = 256,
    batch_size: int = 32,
    lr: float = 0.001
):
    """Run Experiment 10: Adaptive Oversampling."""
    
    # Load data
    data_path = Path('data/input')
    
    if embedding_type == 'dutch':
        train_path = next(data_path.glob('*legal_dutch*train/'))
        valid_path = next(data_path.glob('*legal_dutch*valid/'))
        emb_col = 'legal_dutch'
        emb_col2 = None
    elif embedding_type == 'roberta':
        train_path = next(data_path.glob('*legal_bert*train/'))
        valid_path = next(data_path.glob('*legal_bert*valid/'))
        emb_col = 'legal_dutch'  # Same column name in both datasets
        emb_col2 = None
    else:  # combined
        # Load both and concatenate
        dutch_train = load_from_disk(str(next(data_path.glob('*legal_dutch*train/'))))
        dutch_valid = load_from_disk(str(next(data_path.glob('*legal_dutch*valid/'))))
        roberta_train = load_from_disk(str(next(data_path.glob('*legal_bert*train/'))))
        roberta_valid = load_from_disk(str(next(data_path.glob('*legal_bert*valid/'))))
        emb_col = 'legal_dutch'
        emb_col2 = 'legal_dutch'  # Will use roberta dataset for this
    
    if embedding_type == 'combined':
        train_ds = dutch_train
        valid_ds = dutch_valid
        train_ds_roberta = roberta_train
        valid_ds_roberta = roberta_valid
    else:
        train_ds = load_from_disk(str(train_path))
        valid_ds = load_from_disk(str(valid_path))
        train_ds_roberta = None
        valid_ds_roberta = None
    
    # Count class distribution and compute oversample factors
    class_counts = count_class_distribution(train_ds['labels'])
    oversample_factors = compute_oversample_factors(class_counts, target_per_class)
    
    # Process training data with oversampling
    train_features = []
    train_labels = []
    
    # Group samples by their primary class (for oversampling)
    class_samples = {c: [] for c in range(32)}
    
    for i, sample in enumerate(train_ds):
        embeddings = np.array(sample[emb_col])
        labels = sample['labels']
        
        # Mean pool the chunks
        feature = np.mean(embeddings, axis=0)
        
        # If combined, concatenate with roberta embeddings
        if embedding_type == 'combined' and train_ds_roberta is not None:
            roberta_emb = np.array(train_ds_roberta[i][emb_col])
            roberta_feature = np.mean(roberta_emb, axis=0)
            feature = np.concatenate([feature, roberta_feature])
        
        # Multi-hot labels
        label_vec = np.zeros(32, dtype=np.float32)
        for c in labels:
            if c < 32:
                label_vec[c] = 1.0
        
        # Add to each class it belongs to
        for c in labels:
            if c < 32:
                class_samples[c].append((feature, label_vec))
    
    # Now oversample
    final_counts = Counter()
    for class_id in range(32):
        samples = class_samples[class_id]
        if not samples:
            continue
        
        factor = oversample_factors.get(class_id, 1)
        
        # Add samples multiple times
        for _ in range(factor):
            for feat, lab in samples:
                train_features.append(feat)
                train_labels.append(lab)
                for c in range(32):
                    if lab[c] > 0:
                        final_counts[c] += 1
    
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    
    # Process validation data (no oversampling)
    valid_features = []
    valid_labels = []
    
    for i, sample in enumerate(valid_ds):
        embeddings = np.array(sample[emb_col])
        labels = sample['labels']
        
        feature = np.mean(embeddings, axis=0)
        
        # If combined, concatenate with roberta embeddings
        if embedding_type == 'combined' and valid_ds_roberta is not None:
            roberta_emb = np.array(valid_ds_roberta[i][emb_col])
            roberta_feature = np.mean(roberta_emb, axis=0)
            feature = np.concatenate([feature, roberta_feature])
        
        valid_features.append(feature)
        
        label_vec = np.zeros(32, dtype=np.float32)
        for c in labels:
            if c < 32:
                label_vec[c] = 1.0
        valid_labels.append(label_vec)
    
    X_valid = np.array(valid_features)
    y_valid = np.array(valid_labels)
    
    # Create model and train
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = SimpleClassifier(X_train.shape[1], hidden_dim, 32).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_valid_t = torch.FloatTensor(X_valid).to(device)
    y_valid_t = torch.FloatTensor(y_valid).to(device)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle
        perm = torch.randperm(len(X_train_t))
        X_shuffled = X_train_t[perm]
        y_shuffled = y_train_t[perm]
        
        train_loss = 0
        n_batches = 0
        for i in range(0, len(X_shuffled), batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_valid_t)
            val_loss = criterion(val_out, y_valid_t).item()
        
        train_loss /= n_batches
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            break
    
    # Load best model and evaluate
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        preds = torch.sigmoid(model(X_valid_t)).cpu().numpy()
    
    # Calculate metrics
    y_pred = (preds > 0.5).astype(int)
    y_true = y_valid
    
    from sklearn.metrics import f1_score
    
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Per-class F1
    class_f1 = []
    classes_detected = 0
    for c in range(32):
        f1 = f1_score(y_true[:, c], y_pred[:, c], zero_division=0)
        class_f1.append(f1)
        if f1 > 0:
            classes_detected += 1
    
    # Save results
    output_dir = Path(f'data/output/{embedding_type}/10_adaptive_oversample')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment': '10_adaptive_oversample',
        'embedding_type': embedding_type,
        'target_per_class': target_per_class,
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'classes_detected': classes_detected,
        'class_f1': {f'class_{i}': float(f) for i, f in enumerate(class_f1)},
        'oversample_factors': {str(k): v for k, v in oversample_factors.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    class_f1_dict = {i: class_f1[i] for i in range(32)}
    
    return results, class_f1_dict, macro_f1


if __name__ == '__main__':
    # For standalone testing
    run(embedding_type='dutch', epochs=5)
