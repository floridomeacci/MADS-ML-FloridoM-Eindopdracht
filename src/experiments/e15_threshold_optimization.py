"""
Experiment 15: Threshold Optimization
=====================================
Use the best E12 model but optimize thresholds per-class on validation set.
This separates model training from threshold selection.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
import json
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score

# Import visualization module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualize import generate_heatmap


TARGET = 3000
STRUGGLING_CLASSES = {5, 8, 21, 23, 24, 26, 30}


class BestClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.215),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.215),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def count_class_distribution(labels_list):
    counts = Counter()
    for labels in labels_list:
        for c in labels:
            counts[c] += 1
    return dict(counts)


def compute_oversample_factors(class_counts):
    factors = {}
    for class_id, count in class_counts.items():
        factors[class_id] = max(1, int(np.ceil(TARGET / count)))
    return factors


def prepare_data():
    data_path = Path('data/input')
    
    dutch_train = load_from_disk(str(next(data_path.glob('*legal_dutch*train/'))))
    dutch_valid = load_from_disk(str(next(data_path.glob('*legal_dutch*valid/'))))
    roberta_train = load_from_disk(str(next(data_path.glob('*legal_bert*train/'))))
    roberta_valid = load_from_disk(str(next(data_path.glob('*legal_bert*valid/'))))
    
    class_counts = count_class_distribution(dutch_train['labels'])
    oversample_factors = compute_oversample_factors(class_counts)
    
    class_samples = {c: [] for c in range(32)}
    
    for i in range(len(dutch_train)):
        dutch_emb = np.array(dutch_train[i]['legal_dutch'])
        roberta_emb = np.array(roberta_train[i]['legal_dutch'])
        labels = dutch_train[i]['labels']
        
        dutch_feat = np.mean(dutch_emb, axis=0)
        roberta_feat = np.mean(roberta_emb, axis=0)
        feature = np.concatenate([dutch_feat, roberta_feat])
        
        label_vec = np.zeros(32, dtype=np.float32)
        for c in labels:
            if c < 32:
                label_vec[c] = 1.0
        
        for c in labels:
            if c < 32:
                class_samples[c].append((feature, label_vec))
    
    train_features, train_labels = [], []
    for class_id in range(32):
        samples = class_samples[class_id]
        if not samples:
            continue
        factor = oversample_factors.get(class_id, 1)
        for _ in range(factor):
            for feat, lab in samples:
                train_features.append(feat)
                train_labels.append(lab)
    
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    
    valid_features, valid_labels = [], []
    for i in range(len(dutch_valid)):
        dutch_emb = np.array(dutch_valid[i]['legal_dutch'])
        roberta_emb = np.array(roberta_valid[i]['legal_dutch'])
        labels = dutch_valid[i]['labels']
        
        dutch_feat = np.mean(dutch_emb, axis=0)
        roberta_feat = np.mean(roberta_emb, axis=0)
        feature = np.concatenate([dutch_feat, roberta_feat])
        valid_features.append(feature)
        
        label_vec = np.zeros(32, dtype=np.float32)
        for c in labels:
            if c < 32:
                label_vec[c] = 1.0
        valid_labels.append(label_vec)
    
    X_valid = np.array(valid_features)
    y_valid = np.array(valid_labels)
    
    return X_train, y_train, X_valid, y_valid, class_counts


def optimize_thresholds(probs: np.ndarray, y_true: np.ndarray, class_id: int) -> tuple:
    """Find optimal threshold for a single class."""
    best_f1 = 0
    best_threshold = 0.5
    
    # Try thresholds from 0.1 to 0.9
    for threshold in np.arange(0.05, 0.95, 0.05):
        preds = (probs[:, class_id] > threshold).astype(int)
        f1 = f1_score(y_true[:, class_id], preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def run(epochs=100, patience=15):
    """Train model then optimize per-class thresholds."""
    
    X_train, y_train, X_valid, y_valid, class_counts = prepare_data()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = BestClassifier(input_dim=X_train.shape[1]).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000301, weight_decay=0.00010524)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_valid_t = torch.FloatTensor(X_valid).to(device)
    y_valid_t = torch.FloatTensor(y_valid).to(device)
    
    best_macro_f1 = 0
    patience_counter = 0
    best_state = None
    batch_size = 16
    
    for epoch in range(epochs):
        model.train()
        
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
        
        train_loss /= n_batches
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_valid_t)
            val_loss = criterion(val_out, y_valid_t).item()
            
            preds = torch.sigmoid(val_out).cpu().numpy()
            y_pred = (preds > 0.5).astype(int)
            
            macro_f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            best_state = model.state_dict().copy()
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or patience_counter > 0:
            print(f"Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"macro_f1={macro_f1:.4f}" +
                  (f" (patience {patience_counter}/{patience})" if patience_counter > 0 else " ⭐"))
        
        scheduler.step()
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best model
    print(f"\n--- Step 2: Threshold Optimization (best epoch: {best_epoch}) ---")
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        probs = torch.sigmoid(model(X_valid_t)).cpu().numpy()
    
    # Standard evaluation with 0.5 threshold
    y_pred_standard = (probs > 0.5).astype(int)
    macro_f1_standard = f1_score(y_valid, y_pred_standard, average='macro', zero_division=0)
    print(f"\nStandard (threshold=0.5): Macro F1 = {macro_f1_standard:.4f}")
    
    # Compute class F1 for standard thresholds
    class_f1_standard = {}
    for c in range(1, 32):
        class_f1_standard[c] = f1_score(y_valid[:, c], y_pred_standard[:, c], zero_division=0)
    
    # Optimize thresholds per class
    print("\nOptimizing per-class thresholds...")
    optimal_thresholds = {}
    
    for c in range(1, 32):  # Skip class 0
        threshold, f1 = optimize_thresholds(probs, y_valid, c)
        optimal_thresholds[c] = threshold
        
        # Only print struggling classes
        if c in STRUGGLING_CLASSES:
            standard_f1 = f1_score(y_valid[:, c], y_pred_standard[:, c], zero_division=0)
            print(f"  Class {c:2d}: threshold {0.5:.2f}→{threshold:.2f}, F1 {standard_f1:.4f}→{f1:.4f}")
    
    # Apply optimized thresholds
    y_pred_optimized = np.zeros_like(probs, dtype=int)
    for c in range(32):
        threshold = optimal_thresholds.get(c, 0.5)
        y_pred_optimized[:, c] = (probs[:, c] > threshold).astype(int)
    
    macro_f1_optimized = f1_score(y_valid, y_pred_optimized, average='macro', zero_division=0)
    micro_f1_optimized = f1_score(y_valid, y_pred_optimized, average='micro', zero_division=0)
    
    # Per-class results
    class_f1 = {}
    classes_detected = 0
    
    for c in range(1, 32):
        f1 = f1_score(y_valid[:, c], y_pred_optimized[:, c], zero_division=0)
        class_f1[c] = f1
        if f1 > 0:
            classes_detected += 1
    
    # Save results
    output_dir = Path('data/output/combined/15_threshold_optimization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment': '15_threshold_optimization',
        'macro_f1_standard': float(macro_f1_standard),
        'macro_f1_optimized': float(macro_f1_optimized),
        'micro_f1': float(micro_f1_optimized),
        'classes_detected': classes_detected,
        'optimal_thresholds': {f'class_{k}': v for k, v in optimal_thresholds.items()},
        'class_f1': {f'class_{k}': v for k, v in class_f1.items()},
        'best_epoch': best_epoch,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(best_state, output_dir / 'model.pt')
    
    # Create heatmap_results dict comparing standard vs optimized (with class F1 dicts)
    heatmap_results = {
        f"Standard 0.5 (F1={macro_f1_standard:.4f})": class_f1_standard,
        f"Optimized (F1={macro_f1_optimized:.4f})": class_f1
    }
    
    return heatmap_results, class_f1, macro_f1_optimized


if __name__ == '__main__':
    run(epochs=100, patience=15)
