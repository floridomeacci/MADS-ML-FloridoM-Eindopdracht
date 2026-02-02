"""
Experiment 01: Baseline Mean Aggregation

ARCHITECTURE:
- MeanAggregator: Simple averaging over document chunks
- NeuralNet: 768 → 32 with dropout

HYPOTHESIS:
Baseline to establish minimum performance. Mean pooling loses positional
information but provides stable gradients and fast training.

NOTE: Main.py orchestrates running this on all 3 embeddings and creates the comparison heatmap.
"""

import mlflow
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import create_baseline_model
from src.training import run_experiment


def run(train_loader, valid_loader, output_dir="experiments", experiment_name="classification", seed: int = 42, hidden_size: int = 768):
    """Run Baseline Mean Aggregation experiment."""
    mlflow.set_tracking_uri(f"file:./{output_dir}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    trainer, metrics = run_experiment(
        model_name="01_baseline_mean_aggregation",
        model_fn=create_baseline_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=output_dir,
        model_params={"hidden_size": hidden_size, "out_size": 32},
        training_params={"epochs": 1, "learning_rate": 0.001},
        seed=seed,
    )
    
    return trainer, metrics
