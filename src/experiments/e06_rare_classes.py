"""
Experiment 06: Rare Classes Focus (BiLSTM)

ARCHITECTURE:
- RNNAggregator: Bidirectional LSTM processes chunks sequentially
- NeuralNet: 768 → 15 (only rare classes) with dropout

HYPOTHESIS:
Training exclusively on 15 rare classes (8,10,13,18,19,20,21,23,24,25,26,28,29,30,31)
to determine if poor performance is due to class interference or insufficient data.

DATA FILTERING (happens in loader layer, NOT here):
- main.py calls loader.get_rare_class_loaders() instead of get_loaders()
- get_rare_class_loaders() calls filter_to_rare_classes() in vectorizers.py
- filter_to_rare_classes() does:
  1. Filters samples: keeps only docs with at least one rare class label
  2. Remaps labels: creates 15-dim one-hot (position 0=class 8, position 1=class 10, etc.)
  
Result: 15,532 → 1,362 train samples, 1,942 → 163 valid samples

This experiment expects PRE-FILTERED data loaders to be passed in.
Works with any embedding type - output directory determines where results are saved.
"""

import mlflow
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import create_rnn_model
from src.training import run_experiment


def run(train_loader, valid_loader, output_dir="experiments", experiment_name="classification", seed: int = 42, hidden_size: int = 768):
    """Run Rare Classes BiLSTM experiment."""
    mlflow.set_tracking_uri(f"file:./{output_dir}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    trainer, metrics = run_experiment(
        model_name="06_bilstm_rare_classes",
        model_fn=create_rnn_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=output_dir,
        model_params={"hidden_size": hidden_size, "out_size": 15},
        training_params={"epochs": 5, "learning_rate": 0.001},
        seed=seed,
    )
    
    return trainer, metrics
