"""
Experiment 05: Mixture of Experts (Top-2)

ARCHITECTURE:
- MeanAggregator: Average pooling over chunks
- MoE: 4 expert networks with top-2 sparse routing
- Each expert: NeuralNet(768→32)

HYPOTHESIS:
Different experts can specialize in different legal document types or
categories. Top-2 routing provides model capacity while maintaining
computational efficiency through sparsity.

Works with any embedding type - output directory determines where results are saved.
"""

import mlflow
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import create_moe_model
from src.training import run_experiment


def run(train_loader, valid_loader, output_dir="experiments", experiment_name="classification", seed: int = 42, hidden_size: int = 768):
    """Run Mixture of Experts experiment."""
    mlflow.set_tracking_uri(f"file:./{output_dir}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    trainer, metrics = run_experiment(
        model_name="05_mixture_of_experts_top2",
        model_fn=create_moe_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=output_dir,
        model_params={"hidden_size": hidden_size, "out_size": 32},
        training_params={"epochs": 1, "learning_rate": 0.001},
        seed=seed,
    )
    
    return trainer, metrics
