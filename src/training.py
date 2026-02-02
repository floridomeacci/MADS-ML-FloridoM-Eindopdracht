"""
Training and Evaluation Module

This module handles model training, evaluation, and experiment tracking
using MLflow for reproducibility.

Key Functions:
- train_model: Main training loop with MLflow logging
- evaluate_model: Compute metrics on validation set
- run_experiment: Complete experiment workflow
"""

from pathlib import Path
from typing import Dict, Optional
import random
import numpy as np

import torch
import torch.optim as optim
import mlflow
from mltrainer import Trainer, TrainerSettings, ReportTypes

from vectormesh.data.vectorizers import detect_device


def set_seed(seed: int = 42) -> None:
    """Set all relevant RNG seeds for reproducibility across experiments.

    This seeds Python's random, NumPy, and PyTorch (CPU/CUDA/MPS where
    applicable). Some MPS ops can still be non-deterministic, but this
    removes most run-to-run variance so different datasets use the same
    initialization and batch orders.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA/MPS safe calls (no-ops if backend not present)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can be restrictive on MPS; leave default but seed anyway
    # torch.use_deterministic_algorithms(False)


def train_model(
    model,
    train_loader,
    valid_loader,
    experiment_name: str,
    epochs: int = 15,
    learning_rate: float = 0.001,
    log_dir: Optional[Path] = None
) -> Trainer:
    """
    Train a model using the mltrainer framework.
    
    This function sets up training with:
    - Binary cross-entropy loss (for multi-label classification)
    - Adam optimizer
    - Learning rate scheduler (ReduceLROnPlateau)
    - TensorBoard logging
    - Early stopping
    
    Args:
        model: VectorMesh pipeline to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        experiment_name: Name for the experiment (MLflow)
        epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        log_dir: Directory for TensorBoard logs
        
    Returns:
        Trained Trainer object
        
    Note:
        Uses BCEWithLogitsLoss which combines sigmoid + BCE.
        Model should output raw logits, not probabilities.
    """
    if log_dir is None:
        log_dir = Path("experiments") / experiment_name
    
    device = detect_device()
    print(f"Training on device: {device}")
    
    # Configure training settings
    settings = TrainerSettings(
        epochs=epochs,
        metrics=[],  # Metrics disabled due to compatibility
        logdir=log_dir,
        train_steps=len(train_loader),
        valid_steps=len(valid_loader),
        reporttypes=[ReportTypes.TENSORBOARD],
    )
    
    # Binary cross-entropy for multi-label classification
    # Combines sigmoid activation with BCE loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=train_loader,
        validdataloader=valid_loader,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
    )
    
    # Log hyperparameters to MLflow
    mlflow.log_params({
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": train_loader.batch_size,
        "device": device,
        "model_type": model.__class__.__name__,
    })
    
    # Train the model
    print(f"\nStarting training for {epochs} epochs...")
    trainer.loop()
    
    print("\nTraining completed!")
    return trainer


def evaluate_model(
    model,
    data_loader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate model performance on a dataset.
    
    Computes validation loss and could be extended with additional
    metrics like F1 score, precision, recall for multi-label classification.
    
    Args:
        model: Trained model to evaluate
        data_loader: DataLoader with evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary of metric name -> value
        
    Note:
        Model is set to eval() mode to disable dropout and batch norm.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Infer num_classes from first batch
    num_classes = None
    per_class_tp = None
    per_class_fp = None
    per_class_fn = None
    per_class_tn = None
    
    # Multi-label metrics
    total_hamming = 0.0
    exact_matches = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            embeddings, targets = batch
            # Handle tuple inputs from CollateParallel
            if isinstance(embeddings, (list, tuple)):
                embeddings = tuple(e.to(device) for e in embeddings)
                outputs = model(embeddings)
            else:
                embeddings = embeddings.to(device)
                outputs = model(embeddings)
            targets = targets.to(device)
            
            # Initialize num_classes from first batch
            if num_classes is None:
                num_classes = targets.shape[1]
                per_class_tp = torch.zeros(num_classes)
                per_class_fp = torch.zeros(num_classes)
                per_class_fn = torch.zeros(num_classes)
                per_class_tn = torch.zeros(num_classes)
            
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            
            # Predictions with threshold 0.5
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            # Per-class TP, FP, FN, TN for F1 calculation
            for class_idx in range(num_classes):
                pred_class = predictions[:, class_idx]
                true_class = targets[:, class_idx]
                
                per_class_tp[class_idx] += ((pred_class == 1) & (true_class == 1)).sum().item()
                per_class_fp[class_idx] += ((pred_class == 1) & (true_class == 0)).sum().item()
                per_class_fn[class_idx] += ((pred_class == 0) & (true_class == 1)).sum().item()
                per_class_tn[class_idx] += ((pred_class == 0) & (true_class == 0)).sum().item()
            
            # Hamming loss (fraction of wrong labels)
            total_hamming += (predictions != targets).float().mean().item()
            
            # Subset accuracy (exact match - all labels correct)
            exact_matches += (predictions == targets).all(dim=1).sum().item()
            total_samples += targets.size(0)
    
    avg_loss = total_loss / len(data_loader)
    hamming_loss = total_hamming / len(data_loader)
    subset_accuracy = exact_matches / total_samples
    
    # Calculate per-class F1 scores and accuracies
    per_class_f1 = []
    per_class_acc = []
    
    for i in range(num_classes):
        tp = per_class_tp[i]
        fp = per_class_fp[i]
        fn = per_class_fn[i]
        tn = per_class_tn[i]
        
        # F1 score
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        per_class_f1.append(f1.item())
        
        # Accuracy per class
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        per_class_acc.append(acc.item())
    
    # Macro-averaged F1 (average of per-class F1s)
    macro_f1 = sum(per_class_f1) / len(per_class_f1)
    
    # Micro-averaged F1 (compute from total TP/FP/FN)
    total_tp = per_class_tp.sum()
    total_fp = per_class_fp.sum()
    total_fn = per_class_fn.sum()
    micro_precision = total_tp / (total_tp + total_fp + 1e-8)
    micro_recall = total_tp / (total_tp + total_fn + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    
    results = {
        "validation_loss": avg_loss,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1.item(),
        "hamming_loss": hamming_loss,
        "subset_accuracy": subset_accuracy,
    }
    
    # Add per-class F1 scores
    for i, f1 in enumerate(per_class_f1):
        results[f"class_{i}_f1"] = f1
    
    return results


def run_experiment(
    model_name: str,
    model_fn,
    train_loader,
    valid_loader,
    output_dir: str = "experiments",
    model_params: Optional[Dict] = None,
    training_params: Optional[Dict] = None,
    seed: int = 42,
):
    """
    Run a complete training experiment with MLflow tracking.
    
    This orchestrates:
    1. Model creation
    2. Training with logging
    3. Evaluation
    4. Result tracking in MLflow
    
    Args:
        model_name: Name of the model architecture
        model_fn: Function that creates the model
        train_loader: Training data loader
        valid_loader: Validation data loader
        model_params: Parameters for model creation
        training_params: Parameters for training
        
    Example:
        >>> run_experiment(
        ...     model_name="baseline",
        ...     model_fn=create_baseline_model,
        ...     train_loader=train_loader,
        ...     valid_loader=valid_loader,
        ...     model_params={"hidden_size": 768, "out_size": 32},
        ...     training_params={"epochs": 50, "learning_rate": 0.001}
        ... )
    """
    if model_params is None:
        model_params = {}
    if training_params is None:
        training_params = {}
    
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*60}")
        print(f"Experiment: {model_name}")
        print(f"{'='*60}\n")
        
        # Log model configuration
        mlflow.log_params(model_params)
        mlflow.log_param("seed", seed)
        
        # Create model
        print("Creating model...")
        # Ensure same initialization & dataloader shuffles across datasets
        set_seed(seed)
        model = model_fn(**model_params)
        print(f"Model architecture:\n{model}\n")
        
        # Train model
        log_dir = Path(output_dir) / model_name
        trainer = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            experiment_name=model_name,
            log_dir=log_dir,
            **training_params
        )
        
        # Evaluate on validation set
        print("\nEvaluating model...")
        metrics = evaluate_model(
            model=model,
            data_loader=valid_loader,
            device=detect_device()
        )
        
        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            print(f"{metric_name}: {value:.4f}")
        
        # Save results to JSON file in the log directory
        import json
        timestamped_dir = sorted([d for d in log_dir.iterdir() if d.is_dir()])[-1]
        results_file = timestamped_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Experiment {model_name} completed!")
        print(f"{'='*60}\n")
        
        return trainer, metrics
