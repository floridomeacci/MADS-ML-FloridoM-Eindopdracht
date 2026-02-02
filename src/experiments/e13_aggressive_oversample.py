"""
Experiment 13: Aggressive Oversampling for Struggling Classes
=============================================================
Target classes: 5, 8, 21, 23, 24, 26, 30

Strategy: Oversample these classes to 6000+ samples while keeping
the base target at 3000 for others.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
import json
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualize import generate_heatmap


# Struggling classes that need aggressive oversampling
STRUGGLING_CLASSES = {5, 8, 21, 23, 24, 26, 30}
AGGRESSIVE_TARGET = 6000  # Double the normal target
NORMAL_TARGET = 3000


def count_class_distribution(labels_list: list) -> dict:
    """Count documents per class."""
    counts = Counter()
    for labels in labels_list:
        for class_id in labels:
            counts[class_id] += 1
    return dict(counts)


def compute_oversample_factors(class_counts: dict) -> dict:
    """Compute oversample factors with aggressive targeting for struggling classes."""
    factors = {}
    
    for class_id in class_counts.keys():
        count = class_counts[class_id]
        
        # Use higher target for struggling classes
        if class_id in STRUGGLING_CLASSES:
            target = AGGRESSIVE_TARGET
        else:
            target = NORMAL_TARGET
        
        factor = max(1, int(np.ceil(target / count)))
        factors[class_id] = factor
    
    return factors


class BestClassifier(nn.Module):
    """Best classifier from E11/E12."""
    
    def __init__(self, input_dim: int, num_classes: int = 32):
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
    """Load and prepare combined data with aggressive oversampling."""
    data_path = Path('data/input')
    
    # Load both datasets
    dutch_train = load_from_disk(str(next(data_path.glob('*legal_dutch*train/'))))
    dutch_valid = load_from_disk(str(next(data_path.glob('*legal_dutch*valid/'))))
    roberta_train = load_from_disk(str(next(data_path.glob('*legal_bert*train/'))))
    roberta_valid = load_from_disk(str(next(data_path.glob('*legal_bert*valid/'))))
    
    # Count class distribution
    class_counts = count_class_distribution(dutch_train['labels'])
    oversample_factors = compute_oversample_factors(class_counts)
    
    print("\n--- Oversample Factors ---")
    print("Struggling classes (target 6000):")
    for c in sorted(STRUGGLING_CLASSES):
        count = class_counts.get(c, 0)
        factor = oversample_factors.get(c, 1)
        print(f"  Class {c:2d}: {count:4d} docs × {factor:3d} = ~{count * factor}")
    
    print("\nOther classes (target 3000):")
    for c in sorted(class_counts.keys()):
        if c not in STRUGGLING_CLASSES and c > 0:
            count = class_counts[c]
            factor = oversample_factors.get(c, 1)
            if count < 500:  # Only show smaller classes
                print(f"  Class {c:2d}: {count:4d} docs × {factor:3d} = ~{count * factor}")
    
    # Process training data
    class_samples = {c: [] for c in range(32)}
    
    for i in range(len(dutch_train)):
        dutch_emb = np.array(dutch_train[i]['legal_dutch'])
        roberta_emb = np.array(roberta_train[i]['legal_dutch'])
        labels = dutch_train[i]['labels']
        
        # Mean pool and concatenate
        dutch_feat = np.mean(dutch_emb, axis=0)
        roberta_feat = np.mean(roberta_emb, axis=0)
        feature = np.concatenate([dutch_feat, roberta_feat])
        
        # Multi-hot labels
        label_vec = np.zeros(32, dtype=np.float32)
        for c in labels:
            if c < 32:
                label_vec[c] = 1.0
        
        for c in labels:
            if c < 32:
                class_samples[c].append((feature, label_vec))
    
    # Oversample
    train_features, train_labels = [], []
    final_counts = Counter()
    
    for class_id in range(32):
        samples = class_samples[class_id]
        if not samples:
            continue
        factor = oversample_factors.get(class_id, 1)
        for _ in range(factor):
            for feat, lab in samples:
                train_features.append(feat)
                train_labels.append(lab)
                for c in range(32):
                    if lab[c] > 0:
                        final_counts[c] += 1
    
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    
    print(f"\n--- Final Distribution for Struggling Classes ---")
    for c in sorted(STRUGGLING_CLASSES):
        orig = class_counts.get(c, 0)
        final = final_counts.get(c, 0)
        print(f"  Class {c:2d}: {orig:4d} → {final:6d} (×{final/orig:.1f})" if orig > 0 else f"  Class {c:2d}: 0")
    
    # Process validation data
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
    
    return X_train, y_train, X_valid, y_valid, class_counts, final_counts


def run(epochs: int = 100, patience: int = 15):
    """Run E13 with aggressive oversampling for struggling classes."""
    
    # Prepare data
    X_train, y_train, X_valid, y_valid, class_counts, final_counts = prepare_data()
    
    # Create model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = BestClassifier(input_dim=X_train.shape[1]).to(device)
    
    # Same optimizer settings as E12
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000301, weight_decay=0.00010524)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_valid_t = torch.FloatTensor(X_valid).to(device)
    y_valid_t = torch.FloatTensor(y_valid).to(device)
    
    # Training loop
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
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_valid_t)
            val_loss = criterion(val_out, y_valid_t).item()
            
            preds = torch.sigmoid(val_out).cpu().numpy()
            y_pred = (preds > 0.5).astype(int)
            
            from sklearn.metrics import f1_score
            macro_f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
            
            # Calculate F1 for struggling classes specifically
            struggling_f1s = []
            for c in STRUGGLING_CLASSES:
                f1 = f1_score(y_valid[:, c], y_pred[:, c], zero_division=0)
                struggling_f1s.append(f1)
            avg_struggling_f1 = np.mean(struggling_f1s)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            best_state = model.state_dict().copy()
            best_epoch = epoch
        else:
            patience_counter += 1
        
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0 or patience_counter > 0:
            print(f"Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"macro_f1={macro_f1:.4f}, struggling_avg={avg_struggling_f1:.4f}" +
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
        preds = torch.sigmoid(model(X_valid_t)).cpu().numpy()
    
    y_pred = (preds > 0.5).astype(int)
    
    macro_f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_valid, y_pred, average='micro', zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: Experiment 13")
    print(f"{'='*60}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    
    # Per-class F1 with focus on struggling classes
    class_f1 = {}
    classes_detected = 0
    
    print(f"\n--- Struggling Classes Performance ---")
    for c in sorted(STRUGGLING_CLASSES):
        f1 = f1_score(y_valid[:, c], y_pred[:, c], zero_division=0)
        class_f1[c] = f1
        if f1 > 0:
            classes_detected += 1
    
    # Also compute F1 for all classes
    for c in range(1, 32):
        if c not in class_f1:
            f1 = f1_score(y_valid[:, c], y_pred[:, c], zero_division=0)
            class_f1[c] = f1
            if f1 > 0:
                classes_detected += 1
    
    # Save results
    output_dir = Path('data/output/combined/13_aggressive_oversample')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment': '13_aggressive_oversample',
        'struggling_classes': list(STRUGGLING_CLASSES),
        'aggressive_target': AGGRESSIVE_TARGET,
        'normal_target': NORMAL_TARGET,
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
    run(epochs=100, patience=15)
