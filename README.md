# VectorMesh - Dutch Legal Document Classification

Multi-label classification of Dutch legal documents (aktes) into 31 legal fact categories using neural embeddings and threshold optimization.

## Results

**Best Model (E17 Ultimate)**: Macro F1 = **0.7661**, 31/31 classes detected

| Experiment | Macro F1 | Classes | Key Technique |
|------------|----------|---------|---------------|
| E10 | 0.65 | 30/31 | Adaptive oversampling |
| E11 | 0.71 | 30/31 | Random hyperparameter search |
| E12 | 0.72 | 30/31 | 100 epochs + early stopping |
| E14 | 0.70 | 31/31 | Focal loss + adaptive thresholds |
| E15 | 0.75 | 31/31 | Per-class threshold optimization |
| E16 | 0.76 | 31/31 | Ensemble of 5 models |
| **E17** | **0.77** | **31/31** | **Ultimate: ensemble + thresholds** |

## Quick Start

```bash
# Run the best model (E17)
python main.py -e e17

# List all experiments
python main.py --list

# Run a specific experiment
python main.py -e e10 --epochs 25
```

## Project Structure

```
FJM Examen Opdracht/
├── main.py                 # Entry point - run experiments
├── RAPPORT.md              # Full experiment documentation (Dutch)
├── README.md               # This file
│
├── data/
│   ├── input/              # Cached embeddings (Dutch + RoBERTa)
│   └── output/             # Experiment results, models, metrics
│
├── src/
│   ├── experiments/        # E01-E17 experiment implementations
│   ├── loaders/            # Data loading (dutch, roberta, combined)
│   ├── models.py           # Model architectures
│   ├── training.py         # Training utilities
│   ├── vectorizers.py      # Embedding loaders
│   └── visualize.py        # Heatmap generation
│
└── visuals/                # Generated heatmaps and plots
```

## Experiments Overview

### Phase 1: Architecture Exploration (E01-E09)
- **E01**: Baseline mean aggregation
- **E02**: Attention mechanism
- **E03**: BiLSTM sequential
- **E04**: Highway network
- **E05**: Mixture of Experts
- **E06**: Rare classes focus
- **E07**: Balanced classes
- **E08-E09**: PCA + trigram features

### Phase 2: Oversampling & Tuning (E10-E13)
- **E10**: Adaptive oversampling (target 3000 per class)
- **E11**: Hyperparameter tuning (Grid vs Random search)
- **E12**: Final model with 100 epochs
- **E13**: Aggressive oversampling (failed experiment)

### Phase 3: Threshold Optimization (E14-E17)
- **E14**: Focal loss + adaptive thresholds (Class 23 first detected!)
- **E15**: Per-class threshold optimization
- **E16**: Ensemble + threshold tuning
- **E17**: Ultimate model combining all techniques

## Key Findings

### The Class 23 Problem
Class 23 always had F1=0 despite oversampling. **Solution**: Lower the classification threshold from 0.50 to 0.20. The model WAS learning Class 23 but outputting low probabilities because it co-occurs with dominant classes (17, 12).

### Threshold Optimization > Oversampling
Per-class threshold optimization (+4.7% Macro F1) was more effective than aggressive oversampling (which actually made things worse).

### Best Architecture
```python
hidden = [128, 128]
dropout = 0.215
batchnorm = True
optimizer = Adam(lr=0.000301, weight_decay=0.00010524)
scheduler = StepLR(step_size=2, gamma=0.5)
```

## Embeddings

Two pre-trained legal language models:
- **Dutch**: `joelniklaus/legal-dutch-roberta-base` (768-dim)
- **RoBERTa**: `Gerwin/legal-bert-dutch-english` (768-dim)
- **Combined**: Concatenated (1536-dim) - used in E10+

## Usage

### Run an Experiment
```python
from src.experiments import e17_ultimate
results = e17_ultimate.run()
```

### Generate Heatmaps
```python
from src.visualize import generate_heatmap, generate_comparison_heatmap

# Single experiment
generate_heatmap('E17', class_f1_dict, macro_f1=0.7661)

# Compare multiple experiments
generate_comparison_heatmap({
    'E12': e12_f1,
    'E16': e16_f1,
    'E17': e17_f1
})
```

### Load Data
```python
from src.loaders import combined
train_loader, valid_loader = combined.get_loaders(batch_size=32)
```

## Requirements

- Python 3.12+
- PyTorch 2.0+
- datasets
- scikit-learn
- matplotlib, seaborn

## Documentation

- [RAPPORT.md](RAPPORT.md) - Full experiment documentation (Dutch)
- [src/experiments/README.md](src/experiments/README.md) - Experiment details

## Author

Florido Meacci - HU University of Applied Sciences
