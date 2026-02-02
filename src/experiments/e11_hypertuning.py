"""
Experiment 11: Hyperparameter Tuning with Adaptive Oversampling
================================================================
Uses E10's adaptive oversampling as base, then applies multiple
hyperparameter tuning techniques to find optimal configuration.

Techniques:
1. Grid Search - Exhaustive search over key parameters
2. Random Search - Random sampling from parameter distributions
3. Bayesian Optimization (Hyperopt TPE) - Smart search based on previous results

All runs use Combined embeddings (proven best in E10).
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
import json
from datetime import datetime
from itertools import product
import random
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualize import generate_heatmap


def count_class_distribution(labels_list: list) -> dict:
    """Count documents per class."""
    counts = Counter()
    for labels in labels_list:
        for class_id in labels:
            counts[class_id] += 1
    return dict(counts)


def compute_oversample_factors(class_counts: dict, target_per_class: int = 3000) -> dict:
    """Compute how many times to duplicate each class's samples."""
    factors = {}
    for class_id in class_counts.keys():
        count = class_counts[class_id]
        factor = max(1, int(np.ceil(target_per_class / count)))
        factors[class_id] = factor
    return factors


class FlexibleClassifier(nn.Module):
    """Flexible MLP with configurable architecture."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list = [256, 256],
        num_classes: int = 32, 
        dropout: float = 0.3,
        use_batchnorm: bool = False
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def prepare_data(target_per_class: int = 3000):
    """Load and prepare combined data with oversampling."""
    data_path = Path('data/input')
    
    # Load both datasets
    dutch_train = load_from_disk(str(next(data_path.glob('*legal_dutch*train/'))))
    dutch_valid = load_from_disk(str(next(data_path.glob('*legal_dutch*valid/'))))
    roberta_train = load_from_disk(str(next(data_path.glob('*legal_bert*train/'))))
    roberta_valid = load_from_disk(str(next(data_path.glob('*legal_bert*valid/'))))
    
    # Count class distribution
    class_counts = count_class_distribution(dutch_train['labels'])
    oversample_factors = compute_oversample_factors(class_counts, target_per_class)
    
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
    
    return X_train, y_train, X_valid, y_valid, class_counts


def train_model(
    X_train, y_train, X_valid, y_valid,
    hidden_dims=[256, 256],
    dropout=0.3,
    lr=0.001,
    batch_size=32,
    epochs=5,
    use_batchnorm=False,
    weight_decay=0.0,
    lr_scheduler=None,
    verbose=False
):
    """Train model with given hyperparameters and return metrics."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = FlexibleClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_batchnorm=use_batchnorm
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    # Optional LR scheduler
    scheduler = None
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_valid_t = torch.FloatTensor(X_valid).to(device)
    y_valid_t = torch.FloatTensor(y_valid).to(device)
    
    best_loss = float('inf')
    best_state = None
    
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
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_valid_t), y_valid_t).item()
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
        
        if scheduler:
            if lr_scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        if verbose:
            print(f"  Epoch {epoch}: val_loss={val_loss:.4f}")
    
    # Evaluate with best model
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        preds = torch.sigmoid(model(X_valid_t)).cpu().numpy()
    
    from sklearn.metrics import f1_score
    y_pred = (preds > 0.5).astype(int)
    
    macro_f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_valid, y_pred, average='micro', zero_division=0)
    
    # Per-class F1
    class_f1 = {}
    classes_detected = 0
    for c in range(32):
        f1 = f1_score(y_valid[:, c], y_pred[:, c], zero_division=0)
        class_f1[c] = f1
        if f1 > 0:
            classes_detected += 1
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'classes_detected': classes_detected,
        'class_f1': class_f1,
        'val_loss': best_loss
    }


def grid_search(X_train, y_train, X_valid, y_valid, epochs=5):
    """Grid search over key hyperparameters."""
    print("\n" + "="*60)
    print("GRID SEARCH")
    print("="*60)
    
    # Define grid
    param_grid = {
        'hidden_dims': [[128, 128], [256, 256], [512, 256], [256, 128, 64]],
        'dropout': [0.2, 0.3, 0.4],
        'lr': [0.001, 0.0005]
    }
    
    results = []
    total = len(param_grid['hidden_dims']) * len(param_grid['dropout']) * len(param_grid['lr'])
    
    print(f"Total combinations: {total}")
    
    for i, (hidden, drop, lr) in enumerate(product(
        param_grid['hidden_dims'], 
        param_grid['dropout'], 
        param_grid['lr']
    )):
        print(f"\n[{i+1}/{total}] hidden={hidden}, dropout={drop}, lr={lr}")
        
        metrics = train_model(
            X_train, y_train, X_valid, y_valid,
            hidden_dims=hidden,
            dropout=drop,
            lr=lr,
            epochs=epochs
        )
        
        results.append({
            'params': {'hidden_dims': hidden, 'dropout': drop, 'lr': lr},
            **metrics
        })
        
        print(f"  → Macro F1: {metrics['macro_f1']:.4f}, Classes: {metrics['classes_detected']}/32")
    
    # Find best
    best = max(results, key=lambda x: x['macro_f1'])
    print(f"\n✓ Best Grid Search: Macro F1={best['macro_f1']:.4f}")
    print(f"  Params: {best['params']}")
    
    return results, best


def random_search(X_train, y_train, X_valid, y_valid, n_trials=15, epochs=5):
    """Random search with parameter sampling."""
    print("\n" + "="*60)
    print("RANDOM SEARCH")
    print("="*60)
    
    results = []
    
    for trial in range(n_trials):
        # Sample hyperparameters
        n_layers = random.choice([2, 3, 4])
        hidden_dims = [random.choice([64, 128, 256, 512]) for _ in range(n_layers)]
        dropout = random.uniform(0.1, 0.5)
        lr = 10 ** random.uniform(-4, -2)  # Log-uniform
        weight_decay = 10 ** random.uniform(-6, -3)
        batch_size = random.choice([16, 32, 64])
        use_batchnorm = random.choice([True, False])
        lr_scheduler = random.choice([None, 'step', 'plateau'])
        
        params = {
            'hidden_dims': hidden_dims,
            'dropout': round(dropout, 3),
            'lr': round(lr, 6),
            'weight_decay': round(weight_decay, 8),
            'batch_size': batch_size,
            'use_batchnorm': use_batchnorm,
            'lr_scheduler': lr_scheduler
        }
        
        print(f"\n[{trial+1}/{n_trials}] {params}")
        
        metrics = train_model(
            X_train, y_train, X_valid, y_valid,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            use_batchnorm=use_batchnorm,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler
        )
        
        results.append({'params': params, **metrics})
        print(f"  → Macro F1: {metrics['macro_f1']:.4f}, Classes: {metrics['classes_detected']}/32")
    
    best = max(results, key=lambda x: x['macro_f1'])
    print(f"\n✓ Best Random Search: Macro F1={best['macro_f1']:.4f}")
    print(f"  Params: {best['params']}")
    
    return results, best


def bayesian_search(X_train, y_train, X_valid, y_valid, n_trials=20, epochs=5):
    """Bayesian optimization using Hyperopt TPE."""
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION (Hyperopt TPE)")
    print("="*60)
    
    try:
        from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    except ImportError:
        print("Hyperopt not installed. Skipping Bayesian search.")
        return [], None
    
    # Define search space
    space = {
        'n_layers': hp.choice('n_layers', [2, 3, 4]),
        'layer_size': hp.choice('layer_size', [64, 128, 256, 512]),
        'dropout': hp.uniform('dropout', 0.1, 0.5),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
        'use_batchnorm': hp.choice('use_batchnorm', [True, False])
    }
    
    results = []
    
    def objective(params):
        hidden_dims = [params['layer_size']] * params['n_layers']
        
        metrics = train_model(
            X_train, y_train, X_valid, y_valid,
            hidden_dims=hidden_dims,
            dropout=params['dropout'],
            lr=params['lr'],
            batch_size=params['batch_size'],
            epochs=epochs,
            use_batchnorm=params['use_batchnorm'],
            weight_decay=params['weight_decay']
        )
        
        results.append({
            'params': {
                'hidden_dims': hidden_dims,
                'dropout': round(params['dropout'], 3),
                'lr': round(params['lr'], 6),
                'weight_decay': round(params['weight_decay'], 8),
                'batch_size': params['batch_size'],
                'use_batchnorm': params['use_batchnorm']
            },
            **metrics
        })
        
        print(f"  Trial {len(results)}: Macro F1={metrics['macro_f1']:.4f}, Classes={metrics['classes_detected']}")
        
        # Minimize negative macro F1
        return {'loss': -metrics['macro_f1'], 'status': STATUS_OK}
    
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        verbose=False
    )
    
    best = max(results, key=lambda x: x['macro_f1'])
    print(f"\n✓ Best Bayesian Search: Macro F1={best['macro_f1']:.4f}")
    print(f"  Params: {best['params']}")
    
    return results, best


def run(epochs=5, target_per_class=3000):
    """Run all hyperparameter tuning techniques."""
    
    # Prepare data once
    X_train, y_train, X_valid, y_valid, class_counts = prepare_data(target_per_class)
    
    all_results = {}
    
    # 1. Grid Search
    grid_results, grid_best = grid_search(X_train, y_train, X_valid, y_valid, epochs)
    all_results['grid_search'] = {'trials': grid_results, 'best': grid_best}
    
    # 2. Random Search
    random_results, random_best = random_search(X_train, y_train, X_valid, y_valid, n_trials=15, epochs=epochs)
    all_results['random_search'] = {'trials': random_results, 'best': random_best}
    
    # 3. Bayesian Optimization
    bayes_results, bayes_best = bayesian_search(X_train, y_train, X_valid, y_valid, n_trials=20, epochs=epochs)
    all_results['bayesian_search'] = {'trials': bayes_results, 'best': bayes_best}
    
    # Summary - find overall best
    summary = []
    if grid_best:
        summary.append(('Grid Search', grid_best['macro_f1'], grid_best['classes_detected'], grid_best['params']))
    if random_best:
        summary.append(('Random Search', random_best['macro_f1'], random_best['classes_detected'], random_best['params']))
    if bayes_best:
        summary.append(('Bayesian (TPE)', bayes_best['macro_f1'], bayes_best['classes_detected'], bayes_best['params']))
    
    # Overall best
    overall_best = max(summary, key=lambda x: x[1])
    
    # Save results
    output_dir = Path('data/output/combined/11_hypertuning')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert class_f1 dicts for JSON serialization
    def serialize_results(results_list):
        serialized = []
        for r in results_list:
            r_copy = r.copy()
            if 'class_f1' in r_copy:
                r_copy['class_f1'] = {str(k): v for k, v in r_copy['class_f1'].items()}
            if 'params' in r_copy and 'hidden_dims' in r_copy['params']:
                r_copy['params']['hidden_dims'] = list(r_copy['params']['hidden_dims'])
            serialized.append(r_copy)
        return serialized
    
    save_data = {
        'experiment': '11_hypertuning',
        'embedding_type': 'combined',
        'epochs_per_trial': epochs,
        'timestamp': datetime.now().isoformat(),
        'grid_search': serialize_results(grid_results),
        'random_search': serialize_results(random_results),
        'bayesian_search': serialize_results(bayes_results) if bayes_results else [],
        'summary': {
            'grid_best_macro_f1': grid_best['macro_f1'] if grid_best else None,
            'random_best_macro_f1': random_best['macro_f1'] if random_best else None,
            'bayesian_best_macro_f1': bayes_best['macro_f1'] if bayes_best else None,
            'overall_best': {
                'method': overall_best[0],
                'macro_f1': overall_best[1],
                'classes_detected': overall_best[2]
            }
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Build multi-result dict for heatmap comparison
    heatmap_results = {}
    if grid_best and grid_best.get('class_f1'):
        heatmap_results[f"Grid (F1={grid_best['macro_f1']:.2f})"] = grid_best['class_f1']
    if random_best and random_best.get('class_f1'):
        heatmap_results[f"Random (F1={random_best['macro_f1']:.2f})"] = random_best['class_f1']
    if bayes_best and bayes_best.get('class_f1'):
        heatmap_results[f"Bayesian (F1={bayes_best['macro_f1']:.2f})"] = bayes_best['class_f1']
    
    # Best overall for main return
    best_method_results = None
    if overall_best[0] == 'Grid Search' and grid_best:
        best_method_results = grid_best
    elif overall_best[0] == 'Random Search' and random_best:
        best_method_results = random_best
    elif overall_best[0] == 'Bayesian (TPE)' and bayes_best:
        best_method_results = bayes_best
    
    class_f1 = best_method_results.get('class_f1', {}) if best_method_results else {}
    macro_f1 = overall_best[1]
    
    # Return heatmap_results as first item for multi-heatmap
    return heatmap_results, class_f1, macro_f1


if __name__ == '__main__':
    run(epochs=5)
