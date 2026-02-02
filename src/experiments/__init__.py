"""
Experiments Package

Individual experiment modules that work with any embedding type.
Output directory parameter controls where results are saved.

Available experiments:
- e01_baseline_mean: Baseline with mean aggregation
- e02_attention: Attention mechanism
- e03_bilstm: Bidirectional LSTM
- e04_highway: Highway network with gating
- e05_moe: Mixture of Experts
- e06_rare_classes: BiLSTM on 15 rare classes only
- e07_balanced_classes: BiLSTM with all classes balanced to minimum
- e08_pca_trigram: PCA dimensionality reduction + trigram features
- e09_advanced_pca_trigram: Advanced PCA + trigram with skip connections, 25 epochs
"""

from . import (
    e01_baseline_mean,
    e02_attention,
    e03_bilstm,
    e04_highway,
    e05_moe,
    e06_rare_classes,
    e07_balanced_classes,
    e08_pca_trigram,
    e09_advanced_pca_trigram,
)

__all__ = [
    "e01_baseline_mean",
    "e02_attention",
    "e03_bilstm",
    "e04_highway",
    "e05_moe",
    "e06_rare_classes",
    "e07_balanced_classes",
    "e08_pca_trigram",
    "e09_advanced_pca_trigram",
]

