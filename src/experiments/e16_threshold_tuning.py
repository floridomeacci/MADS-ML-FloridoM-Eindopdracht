"""
Experiment 16: Threshold Hyperparameter Tuning
==============================================
Systematically tune thresholds per class with fast 5-epoch runs.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
import json
from datetime import datetime
from sklearn.metrics import f1_score
from itertools import product

# Import visualization module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualize import generate_heatmap


TARGET = 3000
STRUGGLING_CLASSES = [5, 8, 21, 23, 24, 26, 30]
THRESHOLD_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


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


def train_model(X_train, y_train, X_valid, y_valid, epochs=5):
    """Train model for specified epochs and return probabilities."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = BestClassifier(input_dim=X_train.shape[1]).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000301, weight_decay=0.00010524)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_valid_t = torch.FloatTensor(X_valid).to(device)
    
    batch_size = 16
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        X_shuffled = X_train_t[perm]
        y_shuffled = y_train_t[perm]
        
        for i in range(0, len(X_shuffled), batch_size):
            batch_x = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_valid_t)).cpu().numpy()
    
    return probs, model.state_dict()


def find_best_threshold_per_class(probs, y_valid, class_id):
    """Find best threshold for a single class."""
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in THRESHOLD_CANDIDATES:
        preds = (probs[:, class_id] > threshold).astype(int)
        f1 = f1_score(y_valid[:, class_id], preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def run_all():
    """Run E16: Multiple training runs with threshold optimization."""
    
    # Load data once
    X_train, y_train, X_valid, y_valid, class_counts = prepare_data()
    
    # Run multiple training runs and ensemble
    n_runs = 5
    all_probs = []
    run_f1s = []  # Track individual run F1 scores
    
    for run_idx in range(n_runs):
        probs, _ = train_model(X_train, y_train, X_valid, y_valid, epochs=5)
        all_probs.append(probs)
        # Compute F1 for this run
        run_pred = (probs > 0.5).astype(int)
        run_f1 = f1_score(y_valid, run_pred, average='macro', zero_division=0)
        run_f1s.append(run_f1)
        print(f"  Run {run_idx+1} Macro F1: {run_f1:.4f}")
    
    # Average probabilities (ensemble)
    ensemble_probs = np.mean(all_probs, axis=0)
    
    # Find optimal threshold per class
    optimal_thresholds = {}
    
    for c in range(1, 32):
        threshold, f1 = find_best_threshold_per_class(ensemble_probs, y_valid, c)
        optimal_thresholds[c] = threshold
    
    # Apply optimized thresholds
    y_pred_optimized = np.zeros_like(ensemble_probs, dtype=int)
    for c in range(32):
        threshold = optimal_thresholds.get(c, 0.5)
        y_pred_optimized[:, c] = (ensemble_probs[:, c] > threshold).astype(int)
    
    # Final evaluation
    macro_f1_std = f1_score(y_valid, (ensemble_probs > 0.5).astype(int), average='macro', zero_division=0)
    macro_f1_opt = f1_score(y_valid, y_pred_optimized, average='macro', zero_division=0)
    micro_f1_opt = f1_score(y_valid, y_pred_optimized, average='micro', zero_division=0)
    
    # Per-class results
    class_f1 = {}
    classes_detected = 0
    
    for c in range(1, 32):
        f1 = f1_score(y_valid[:, c], y_pred_optimized[:, c], zero_division=0)
        class_f1[c] = f1
        if f1 > 0:
            classes_detected += 1
    
    # Save results
    output_dir = Path('data/output/combined/16_threshold_tuning')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment': '16_threshold_tuning',
        'n_runs': n_runs,
        'epochs_per_run': 5,
        'macro_f1_standard': float(macro_f1_std),
        'macro_f1_optimized': float(macro_f1_opt),
        'micro_f1': float(micro_f1_opt),
        'classes_detected': classes_detected,
        'optimal_thresholds': {f'class_{k}': v for k, v in optimal_thresholds.items()},
        'class_f1': {f'class_{k}': v for k, v in class_f1.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Compute class F1 for standard thresholds
    y_pred_std = (ensemble_probs > 0.5).astype(int)
    class_f1_std = {}
    for c in range(1, 32):
        class_f1_std[c] = f1_score(y_valid[:, c], y_pred_std[:, c], zero_division=0)
    
    # Create heatmap_results dict: standard vs optimized (with class F1 dicts)
    heatmap_results = {
        f"Standard 0.5 (F1={macro_f1_std:.4f})": class_f1_std,
        f"Optimized (F1={macro_f1_opt:.4f})": class_f1
    }
    
    return heatmap_results, class_f1, macro_f1_opt


# Alias for main.py consistency
def run(**kwargs):
    """Alias for run_all() for main.py compatibility."""
    return run_all()


if __name__ == '__main__':
    run_all()
