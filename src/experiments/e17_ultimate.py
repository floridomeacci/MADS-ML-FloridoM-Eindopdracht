"""
Experiment 17: Ultimate Model - Best Settings + Threshold Optimization
======================================================================
Combines:
- Best architecture from E12: hidden=[128,128], dropout=0.215, batchnorm=True
- Best optimizer: Adam lr=0.000301, weight_decay=0.00010524, StepLR
- Ensemble of 5 models
- Optimized thresholds from E16
- 100 epochs with early stopping (patience=15)
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
STRUGGLING_CLASSES = [5, 8, 21, 23, 24, 26, 30]

# Optimal thresholds from E16
OPTIMAL_THRESHOLDS = {
    1: 0.50, 2: 0.15, 3: 0.30, 4: 0.45, 5: 0.20,
    6: 0.55, 7: 0.15, 8: 0.60, 9: 0.65, 10: 0.50,
    11: 0.45, 12: 0.45, 13: 0.85, 14: 0.45, 15: 0.30,
    16: 0.10, 17: 0.25, 18: 0.25, 19: 0.15, 20: 0.70,
    21: 0.30, 22: 0.35, 23: 0.20, 24: 0.35, 25: 0.25,
    26: 0.60, 27: 0.75, 28: 0.20, 29: 0.30, 30: 0.60,
    31: 0.05
}

THRESHOLD_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


class BestClassifier(nn.Module):
    """Best architecture from E11/E12."""
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


def apply_thresholds(probs, thresholds):
    """Apply per-class thresholds."""
    preds = np.zeros_like(probs, dtype=int)
    for c in range(probs.shape[1]):
        threshold = thresholds.get(c, 0.5)
        preds[:, c] = (probs[:, c] > threshold).astype(int)
    return preds


def find_best_threshold(probs, y_valid, class_id):
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


def train_single_model(X_train, y_train, X_valid, y_valid, model_id, epochs=100, patience=15):
    """Train a single model with early stopping."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = BestClassifier(input_dim=X_train.shape[1]).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000301, weight_decay=0.00010524)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_valid_t = torch.FloatTensor(X_valid).to(device)
    y_valid_t = torch.FloatTensor(y_valid).to(device)
    
    batch_size = 16
    best_macro_f1 = 0
    patience_counter = 0
    best_state = None
    best_epoch = 0
    
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
        scheduler.step()
        
        # Validation with optimized thresholds
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_valid_t)).cpu().numpy()
        
        y_pred = apply_thresholds(probs, OPTIMAL_THRESHOLDS)
        macro_f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            best_state = model.state_dict().copy()
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or patience_counter > 0:
            print(f"  Model {model_id} Epoch {epoch:3d}: loss={train_loss:.4f}, macro_f1={macro_f1:.4f}" +
                  (f" (patience {patience_counter}/{patience})" if patience_counter > 0 else " ⭐"))
        
        if patience_counter >= patience:
            print(f"  Model {model_id} early stop at epoch {epoch}, best={best_epoch}")
            break
    
    # Load best state
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(model(X_valid_t)).cpu().numpy()
    
    return final_probs, best_state, best_epoch, best_macro_f1


def run():
    """Run E17: Ultimate model with ensemble and threshold optimization."""
    
    # Load data
    X_train, y_train, X_valid, y_valid, class_counts = prepare_data()
    
    # Train ensemble
    n_models = 5
    all_probs = []
    all_best_epochs = []
    all_best_f1s = []
    all_class_f1s = []  # Track class-level F1 for each model
    
    for i in range(n_models):
        probs, state, best_epoch, best_f1 = train_single_model(
            X_train, y_train, X_valid, y_valid, 
            model_id=i+1, epochs=100, patience=15
        )
        all_probs.append(probs)
        all_best_epochs.append(best_epoch)
        all_best_f1s.append(best_f1)
        
        # Compute class-level F1 for this model
        y_pred_model = (probs > 0.5).astype(int)
        model_class_f1 = {}
        for c in range(1, 32):
            model_class_f1[c] = f1_score(y_valid[:, c], y_pred_model[:, c], zero_division=0)
        all_class_f1s.append(model_class_f1)
    
    # Ensemble averaging
    ensemble_probs = np.mean(all_probs, axis=0)
    
    # Re-optimize thresholds on ensemble
    final_thresholds = {}
    for c in range(1, 32):
        threshold, f1 = find_best_threshold(ensemble_probs, y_valid, c)
        final_thresholds[c] = threshold
    
    # Apply final thresholds
    y_pred_final = apply_thresholds(ensemble_probs, final_thresholds)
    
    # Final metrics
    macro_f1_std = f1_score(y_valid, (ensemble_probs > 0.5).astype(int), average='macro', zero_division=0)
    macro_f1_opt = f1_score(y_valid, y_pred_final, average='macro', zero_division=0)
    micro_f1 = f1_score(y_valid, y_pred_final, average='micro', zero_division=0)
    
    # Compute per-class F1
    class_f1 = {}
    classes_detected = 0
    for c in range(1, 32):
        f1 = f1_score(y_valid[:, c], y_pred_final[:, c], zero_division=0)
        class_f1[c] = f1
        if f1 > 0:
            classes_detected += 1
    
    # Save results
    output_dir = Path('data/output/combined/17_ultimate')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment': '17_ultimate',
        'n_models': n_models,
        'epochs': 100,
        'patience': 15,
        'avg_best_epoch': float(np.mean(all_best_epochs)),
        'macro_f1_standard': float(macro_f1_std),
        'macro_f1_optimized': float(macro_f1_opt),
        'micro_f1': float(micro_f1),
        'classes_detected': classes_detected,
        'final_thresholds': {f'class_{k}': v for k, v in final_thresholds.items()},
        'class_f1': {f'class_{k}': v for k, v in class_f1.items()},
        'model_best_epochs': all_best_epochs,
        'model_best_f1s': all_best_f1s,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save ensemble probabilities for analysis
    np.save(output_dir / 'ensemble_probs.npy', ensemble_probs)
    
    # Create heatmap_results dict with individual models and ensemble (with class-level F1)
    heatmap_results = {}
    for i, (f1, model_class_f1) in enumerate(zip(all_best_f1s, all_class_f1s)):
        heatmap_results[f"Model {i+1} (F1={f1:.4f})"] = model_class_f1
    heatmap_results[f"Ensemble (F1={macro_f1_opt:.4f})"] = class_f1
    
    return heatmap_results, class_f1, macro_f1_opt


if __name__ == '__main__':
    run()
