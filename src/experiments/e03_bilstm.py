"""
Experiment 03: BiLSTM Sequential Processing

ARCHITECTURE:
- RNNAggregator: Bidirectional LSTM processes chunks sequentially
- NeuralNet: 768 → 32 with dropout

HYPOTHESIS:
BiLSTM captures sequential dependencies between document chunks,
which may be important for legal documents where argument structure matters.

Works with any embedding type - output directory determines where results are saved.
"""

import mlflow
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import create_rnn_model
from src.training import run_experiment


def run(train_loader, valid_loader, output_dir="experiments", experiment_name="classification", seed: int = 42, hidden_size: int = 768):
    """Run BiLSTM Sequential experiment."""
    mlflow.set_tracking_uri(f"file:./{output_dir}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    trainer, metrics = run_experiment(
        model_name="03_bilstm_sequential",
        model_fn=create_rnn_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=output_dir,
        model_params={"hidden_size": hidden_size, "out_size": 32},
        training_params={"epochs": 1, "learning_rate": 0.001},
        seed=seed,
    )
    
    return trainer, metrics
