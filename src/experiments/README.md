# Experiments

## E01 - Baseline Mean Aggregation
Simple baseline that averages all chunk embeddings before classification.

## E02 - Attention Mechanism
Learnable attention weights over document chunks.

## E03 - BiLSTM Sequential
Bidirectional LSTM to capture sequential dependencies in documents.

## E04 - Highway Network
Highway connections with gating for better gradient flow.

## E05 - Mixture of Experts (MoE)
Gated ensemble of expert networks with top-k routing.

## E06 - Rare Classes
BiLSTM trained only on 15 rare classes to test data sufficiency.

## E07 - Balanced Classes
Class-balanced sampling during training.

## E08 - PCA Trigram
PCA dimensionality reduction with trigram features.

## E09 - Advanced PCA Trigram
Extended PCA approach with additional feature engineering.

## E10 - Adaptive Oversampling
Oversample each class to target 3000 samples using combined embeddings.

## E11 - Hyperparameter Tuning
Grid search vs random search to find optimal architecture settings.

## E12 - Final Model
Train best architecture for 100 epochs with early stopping.

## E13 - Aggressive Oversampling
10x oversample struggling classes (failed - made performance worse).

## E14 - Focal Loss
Focal loss with adaptive per-class thresholds.

## E15 - Threshold Optimization
Optimize classification threshold per class on validation set.

## E16 - Threshold Tuning
Ensemble of 5 models with systematic threshold tuning.

## E17 - Ultimate Model
Combines best architecture, ensemble, and threshold optimization.
