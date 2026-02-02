"""
Experiment 14: Focal Loss + Adaptive Thresholds
================================================
Problem: Class 23 co-occurs with dominant classes 17 (31.5%) and 12 (24.1%).
The model learns to predict 17/12 and ignores 23.

Solution:
1. Focal loss - focuses on hard examples
2. Adaptive thresholds - lower threshold for struggling classes (0.3 instead of 0.5)
3. Class-weighted loss - penalize missing struggling classes more
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
import json
from datetime import datetime
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualize import generate_heatmap


# Classes that co-occur with dominant classes
STRUGGLING_CLASSES = {5, 8, 21, 23, 24, 26, 30}

# Adaptive thresholds - lower for classes that get "stolen"
# More aggressive thresholds based on E14 results
THRESHOLDS = {c: 0.5 for c in range(32)}  # Default
THRESHOLDS[5] = 0.35   # Improved in E14, keep similar
THRESHOLDS[8] = 0.25   # Got worse, lower more
THRESHOLDS[21] = 0.25  # Got worse, lower more
THRESHOLDS[23] = 0.15  # Breakthrough! Lower even more
THRESHOLDS[24] = 0.35  # Similar
THRESHOLDS[26] = 0.15  # Got much worse, lower significantly
THRESHOLDS[30] = 0.25  # Got worse, lower more

TARGET = 3000


class ClassWeightedFocalLoss(nn.Module):
    """Focal loss with per-class weights for struggling classes."""
    
    def __init__(self, class_weights, alpha=1.0, gamma=2.0):
        super().__init__()
        self.register_buffer('class_weights', class_weights)
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class weights
        loss = self.alpha * focal_weight * bce * self.class_weights
        
        return loss.mean()


def apply_adaptive_thresholds(probs: np.ndarray) -> np.ndarray:
    """Apply per-class thresholds instead of fixed 0.5."""
    preds = np.zeros_like(probs, dtype=int)
    for c in range(probs.shape[1]):
        threshold = THRESHOLDS.get(c, 0.5)
        preds[:, c] = (probs[:, c] > threshold).astype(int)
    return preds


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


def prepare_data():
    """Load data with standard oversampling."""
    data_path = Path('data/input')
    
    dutch_train = load_from_disk(str(next(data_path.glob('*legal_dutch*train/'))))
    dutch_valid = load_from_disk(str(next(data_path.glob('*legal_dutch*valid/'))))
    roberta_train = load_from_disk(str(next(data_path.glob('*legal_bert*train/'))))
    roberta_valid = load_from_disk(str(next(data_path.glob('*legal_bert*valid/'))))
    
    class_counts = count_class_distribution(dutch_train['labels'])
    oversample_factors = compute_oversample_factors(class_counts)
    
    # Process training data
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
    
    # Validation data
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


def run(gamma=2.0, struggling_weight=5.0, epochs=100, patience=15):
    """Run E14 with focal loss + adaptive thresholds."""
    
    X_train, y_train, X_valid, y_valid, class_counts = prepare_data()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = BestClassifier(input_dim=X_train.shape[1]).to(device)
    
    # Create class weights - higher for struggling classes
    class_weights = torch.ones(32)
    for c in STRUGGLING_CLASSES:
        class_weights[c] = struggling_weight
    class_weights = class_weights.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000301, weight_decay=0.00010524)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = ClassWeightedFocalLoss(class_weights, alpha=1.0, gamma=gamma)
    
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
            
            probs = torch.sigmoid(val_out).cpu().numpy()
            y_pred = apply_adaptive_thresholds(probs)  # Use adaptive thresholds
            
            macro_f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
            
            struggling_f1s = [f1_score(y_valid[:, c], y_pred[:, c], zero_division=0) 
                            for c in STRUGGLING_CLASSES]
            avg_struggling = np.mean(struggling_f1s)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            best_state = model.state_dict().copy()
            best_epoch = epoch
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or patience_counter > 0:
            print(f"Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"macro_f1={macro_f1:.4f}, struggling={avg_struggling:.4f}" +
                  (f" (patience {patience_counter}/{patience})" if patience_counter > 0 else " ⭐"))
        
        scheduler.step()
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Final evaluation
    print(f"\n--- Final Evaluation (best epoch: {best_epoch}) ---")
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        probs = torch.sigmoid(model(X_valid_t)).cpu().numpy()
    
    y_pred = apply_adaptive_thresholds(probs)  # Use adaptive thresholds
    
    macro_f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_valid, y_pred, average='micro', zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: Experiment 14 (Focal + Adaptive Thresholds)")
    class_f1 = {}
    classes_detected = 0
    
    # Compute per-class F1
    for c in range(1, 32):
        f1 = f1_score(y_valid[:, c], y_pred[:, c], zero_division=0)
        class_f1[c] = f1
        if f1 > 0:
            classes_detected += 1
    
    # Save
    output_dir = Path('data/output/combined/14_focal_loss')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment': '14_focal_loss',
        'gamma': gamma,
        'struggling_weight': struggling_weight,
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'classes_detected': classes_detected,
        'class_f1': {f'class_{k}': v for k, v in class_f1.items()},
        'best_epoch': best_epoch,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(best_state, output_dir / 'model.pt')
    
    return results, class_f1, macro_f1


if __name__ == '__main__':
    run(gamma=2.0, struggling_weight=5.0, epochs=100, patience=15)
