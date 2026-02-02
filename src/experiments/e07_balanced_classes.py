"""
Experiment 07: Balanced Classes (BiLSTM)

ARCHITECTURE:
- RNNAggregator: Bidirectional LSTM processes chunks sequentially
- NeuralNet: 768 → 32 (all classes) with dropout

HYPOTHESIS:
Downsample all 32 classes to the same number of samples (the minimum class count).
This tests whether class imbalance causes poor rare class performance, or if it's 
simply insufficient data for those classes.

DATA BALANCING (happens in loader layer, NOT here):
- main.py calls loader.get_balanced_class_loaders() instead of get_loaders()
- get_balanced_class_loaders() calls balance_classes() in vectorizers.py
- balance_classes() does:
  1. Counts samples per class
  2. Finds the minimum count
  3. Downsamples all classes to that minimum
  4. Creates standard 32-dim one-hot labels

Result: All classes have equal representation (limited by smallest class)

This experiment expects PRE-BALANCED data loaders to be passed in.
Works with any embedding type - output directory determines where results are saved.
"""

import mlflow
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import create_rnn_model
from src.training import run_experiment


def run(train_loader, valid_loader, output_dir="experiments", experiment_name="classification", seed: int = 42, hidden_size: int = 768):
    """Run Balanced Classes BiLSTM experiment."""
    mlflow.set_tracking_uri(f"file:./{output_dir}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    trainer, metrics = run_experiment(
        model_name="07_bilstm_balanced_classes",
        model_fn=create_rnn_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=output_dir,
        model_params={"hidden_size": hidden_size, "out_size": 32},
        training_params={"epochs": 5, "learning_rate": 0.001},
        seed=seed,
    )
    
    return trainer, metrics
