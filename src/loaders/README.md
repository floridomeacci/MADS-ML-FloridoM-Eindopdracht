# Loaders

Data loading utilities for cached embeddings.

## Files

### `legal_dutch.py`
Loads Dutch embeddings from `joelniklaus/legal-dutch-roberta-base` (768-dim).

### `legal_roberta.py`
Loads RoBERTa embeddings from `Gerwin/legal-bert-dutch-english` (768-dim).

### `combined.py`
Loads and concatenates both Dutch and RoBERTa embeddings (1536-dim).

## Usage

```python
from src.loaders import combined

train_loader, valid_loader = combined.get_loaders(batch_size=32)
```

Each loader provides:
- `get_loaders(batch_size, max_chunks, num_classes)` - Returns train and validation DataLoaders
- Automatic padding to `max_chunks` (default: 30)
- One-hot encoded labels for multi-label classification
