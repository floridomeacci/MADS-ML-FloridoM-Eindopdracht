"""
Experiment 02: Attention Mechanism

ARCHITECTURE:
- AttentionAggregator: Learnable attention weights over chunks
- NeuralNet: 768 → 32 with dropout

HYPOTHESIS:
Attention allows the model to focus on important document sections,
potentially outperforming simple mean pooling for legal texts.

Works with any embedding type - output directory determines where results are saved.
"""

import mlflow
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import create_attention_model
from src.training import run_experiment


def run(train_loader, valid_loader, output_dir="experiments", experiment_name="classification", seed: int = 42, hidden_size: int = 768):
    """Run Attention Mechanism experiment."""
    mlflow.set_tracking_uri(f"file:./{output_dir}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    trainer, metrics = run_experiment(
        model_name="02_attention_mechanism",
        model_fn=create_attention_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=output_dir,
        model_params={"hidden_size": hidden_size, "out_size": 32},
        training_params={"epochs": 1, "learning_rate": 0.001},
        seed=seed,
    )
    
    return trainer, metrics
