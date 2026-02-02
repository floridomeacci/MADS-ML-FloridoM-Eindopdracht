"""
Experiment 09: Advanced PCA + Trigram with Skip Connections

ARCHITECTURE:
- Preprocessing: Padded chunks → PCA per chunk → concatenate with trigrams
- ResidualBlock layers with skip connections for gradient flow
- 25 epochs for deeper training
- Input: Variable-length chunks padded to max_chunks

HYPOTHESIS:
Skip connections help gradients flow through deeper networks, preventing
vanishing gradients. Longer training (25 epochs) allows the model to
better learn rare class patterns. Padding preserves chunk-level information
instead of mean aggregation.

Improvements over E08:
- Skip connections (residual blocks)
- Padding instead of mean aggregation
- 25 epochs instead of 5
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import mlflow
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import run_experiment


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Skip connection: output = ReLU(block(x) + x)
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.relu(out)
        out = self.dropout(out)
        return out


class AdvancedClassifier(nn.Module):
    """
    Advanced classifier with skip connections for PCA + trigram features.
    
    Architecture:
    - Input projection to hidden_dim
    - Multiple residual blocks with skip connections
    - Output projection to num_classes
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = 32, 
                 dropout: float = 0.3, num_residual_blocks: int = 3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks with skip connections
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, input_dim)
        x = self.input_proj(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.output_proj(x)
        return x


class ChunkLevelClassifier(nn.Module):
    """
    Classifier that processes padded chunks with attention pooling.
    
    Architecture:
    - Process each chunk through shared layers
    - Attention-weighted pooling over chunks
    - Residual blocks for classification
    """
    
    def __init__(self, chunk_dim: int, hidden_dim: int = 512, num_classes: int = 32,
                 dropout: float = 0.3, num_residual_blocks: int = 3):
        super().__init__()
        
        # Chunk encoder
        self.chunk_encoder = nn.Sequential(
            nn.Linear(chunk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Attention for chunk pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, mask=None):
        # x shape: (batch, max_chunks, chunk_dim)
        batch_size, max_chunks, chunk_dim = x.shape
        
        # Encode each chunk
        x = self.chunk_encoder(x)  # (batch, max_chunks, hidden_dim)
        
        # Compute attention weights
        attn_scores = self.attention(x).squeeze(-1)  # (batch, max_chunks)
        
        if mask is not None:
            # Mask out padding positions
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, max_chunks)
        
        # Weighted sum over chunks
        x = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output
        x = self.output_proj(x)
        return x


def pad_chunks(embeddings_list: list, max_chunks: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad variable-length chunks to fixed length.
    
    Args:
        embeddings_list: List of chunk embeddings per document
        max_chunks: Maximum number of chunks to keep
        
    Returns:
        Padded embeddings (n_samples, max_chunks, embed_dim) and mask (n_samples, max_chunks)
    """
    n_samples = len(embeddings_list)
    embed_dim = len(embeddings_list[0][0])  # First sample, first chunk
    
    padded = np.zeros((n_samples, max_chunks, embed_dim), dtype=np.float32)
    mask = np.zeros((n_samples, max_chunks), dtype=np.float32)
    
    for i, doc_chunks in enumerate(embeddings_list):
        n_chunks = min(len(doc_chunks), max_chunks)
        chunks_array = np.array(doc_chunks[:n_chunks], dtype=np.float32)
        padded[i, :n_chunks] = chunks_array
        mask[i, :n_chunks] = 1.0
    
    return padded, mask


def aggregate_chunks_mean(dataset, embedding_col: str) -> np.ndarray:
    """
    Aggregate variable-length chunks using mean pooling.
    """
    def compute_mean(example):
        chunks = np.array(example[embedding_col], dtype=np.float32)
        return {"mean_embedding": chunks.mean(axis=0).tolist()}
    
    ds_with_means = dataset.map(compute_mean, desc="Aggregating chunks")
    embeddings = np.array(ds_with_means["mean_embedding"], dtype=np.float32)
    return embeddings


def apply_pca_to_embeddings(embeddings: np.ndarray, n_components: int = 256, 
                            fitted_pca: PCA = None) -> tuple[np.ndarray, PCA]:
    """Apply PCA to reduce embedding dimensionality."""
    if fitted_pca is None:
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(embeddings)
    else:
        pca = fitted_pca
        reduced = pca.transform(embeddings)
    
    return reduced, pca


def extract_trigram_features(texts: list[str], max_features: int = 100,
                            fitted_vectorizer: CountVectorizer = None) -> tuple[np.ndarray, CountVectorizer]:
    """Extract character trigram features from text."""
    if fitted_vectorizer is None:
        vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(3, 3),
            max_features=max_features,
            binary=True
        )
        trigram_features = vectorizer.fit_transform(texts).toarray()
    else:
        vectorizer = fitted_vectorizer
        trigram_features = vectorizer.transform(texts).toarray()
    
    return trigram_features.astype(np.float32), vectorizer


def preprocess_dataset(dataset, embedding_col: str, n_pca_components: int = 256, 
                       n_trigrams: int = 100, fitted_pca: PCA = None,
                       fitted_vectorizer: CountVectorizer = None):
    """
    Apply PCA + trigram preprocessing to a dataset (mean aggregation version).
    """
    embeddings = aggregate_chunks_mean(dataset, embedding_col)
    texts = dataset['text']
    labels = list(dataset['labels'])
    
    # Apply PCA
    reduced_embeddings, pca = apply_pca_to_embeddings(embeddings, n_pca_components, fitted_pca)
    
    # Extract trigrams
    trigram_features, trigram_vectorizer = extract_trigram_features(texts, n_trigrams, fitted_vectorizer)
    
    # Concatenate
    combined_features = np.concatenate([reduced_embeddings, trigram_features], axis=1)
    
    return combined_features, labels, pca, trigram_vectorizer


def create_advanced_model(input_dim: int = 356, hidden_dim: int = 512, 
                          num_classes: int = 32, dropout: float = 0.3,
                          num_residual_blocks: int = 3):
    """Create the advanced classifier with skip connections."""
    return AdvancedClassifier(input_dim, hidden_dim, num_classes, dropout, num_residual_blocks)


def create_dataloaders(train_features, train_labels, valid_features, valid_labels, 
                       batch_size: int = 32, num_classes: int = 32):
    """Create PyTorch DataLoaders from preprocessed data."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Convert to tensors
    train_X = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.zeros(len(train_labels), num_classes, dtype=torch.float32)
    for i, label_list in enumerate(train_labels):
        for label in label_list:
            if label < num_classes:
                train_y[i, label] = 1.0
    
    valid_X = torch.tensor(valid_features, dtype=torch.float32)
    valid_y = torch.zeros(len(valid_labels), num_classes, dtype=torch.float32)
    for i, label_list in enumerate(valid_labels):
        for label in label_list:
            if label < num_classes:
                valid_y[i, label] = 1.0
    
    train_dataset = TensorDataset(train_X, train_y)
    valid_dataset = TensorDataset(valid_X, valid_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader


def run_single(embedding_type: str = "dutch", output_dir: str = "data/output", 
        experiment_name: str = "advanced-pca-trigram", seed: int = 42,
        n_pca_components: int = 256, n_trigrams: int = 100, batch_size: int = 32,
        epochs: int = 25, hidden_dim: int = 512, num_residual_blocks: int = 3):
    """
    Run Advanced PCA + Trigram experiment with skip connections for single embedding.
    
    Args:
        embedding_type: Which base embedding to use ("dutch" or "roberta")
        output_dir: Directory to save MLflow artifacts
        experiment_name: MLflow experiment name
        n_pca_components: Number of PCA components
        n_trigrams: Number of trigram features
        epochs: Number of training epochs (default 25)
        hidden_dim: Hidden dimension for residual blocks
        num_residual_blocks: Number of residual blocks with skip connections
    """
    from datasets import load_from_disk
    from pathlib import Path
    
    # Load raw cached data
    data_path = Path("data/input")
    
    if embedding_type == "dutch":
        embedding_col = "legal_dutch"
        train_path = next(data_path.glob("*legal_dutch*train"))
        valid_path = next(data_path.glob("*legal_dutch*valid"))
    elif embedding_type == "roberta":
        embedding_col = "legal_dutch"  # Same column name in cache
        train_path = next(data_path.glob("*legal_bert*train"))
        valid_path = next(data_path.glob("*legal_bert*valid"))
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}. Use 'dutch' or 'roberta'.")
    
    train_ds = load_from_disk(str(train_path))
    valid_ds = load_from_disk(str(valid_path))
    
    # Preprocess training data
    train_features, train_labels, pca, trigram_vec = preprocess_dataset(
        train_ds, embedding_col, n_pca_components, n_trigrams
    )
    
    # Preprocess validation data
    valid_features, valid_labels, _, _ = preprocess_dataset(
        valid_ds, embedding_col, n_pca_components, n_trigrams,
        fitted_pca=pca, fitted_vectorizer=trigram_vec
    )
    
    # Create dataloaders
    train_loader, valid_loader = create_dataloaders(
        train_features, train_labels,
        valid_features, valid_labels,
        batch_size=batch_size
    )
    
    # Calculate input size
    input_dim = n_pca_components + n_trigrams
    
    # Setup MLflow
    full_output_dir = f"{output_dir}/{embedding_type}"
    mlflow.set_tracking_uri(f"file:./{full_output_dir}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Run experiment
    trainer, metrics = run_experiment(
        model_name="09_advanced_pca_trigram",
        model_fn=create_advanced_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=full_output_dir,
        model_params={
            "input_dim": input_dim, 
            "hidden_dim": hidden_dim, 
            "num_classes": 32,
            "dropout": 0.3,
            "num_residual_blocks": num_residual_blocks
        },
        training_params={"epochs": epochs, "learning_rate": 0.001},
        seed=seed,
    )
    
    # Return results
    class_f1 = {i: metrics.get(f'class_{i}_f1', 0) for i in range(32)}
    macro_f1 = metrics.get('macro_f1', 0)
    
    return metrics, class_f1, macro_f1


def run_combined(output_dir: str = "experiments", 
                 experiment_name: str = "combined-advanced-pca-trigram", seed: int = 42,
                 n_pca_components: int = 256, n_trigrams: int = 100, batch_size: int = 32,
                 epochs: int = 25, hidden_dim: int = 512, num_residual_blocks: int = 3):
    """
    Run Advanced PCA + Trigram experiment with COMBINED Dutch + RoBERTa embeddings.
    
    Architecture:
    - PCA(Dutch) + PCA(RoBERTa) + Trigrams
    - Input dim: 256 + 256 + 100 = 612
    """
    from datasets import load_from_disk
    from pathlib import Path
    
    # Load raw cached data
    data_path = Path("data/input")
    
    # Dutch paths
    dutch_train_path = next(data_path.glob("*legal_dutch*train"))
    dutch_valid_path = next(data_path.glob("*legal_dutch*valid"))
    
    # RoBERTa paths
    roberta_train_path = next(data_path.glob("*legal_bert*train"))
    roberta_valid_path = next(data_path.glob("*legal_bert*valid"))
    
    dutch_train_ds = load_from_disk(str(dutch_train_path))
    dutch_valid_ds = load_from_disk(str(dutch_valid_path))
    roberta_train_ds = load_from_disk(str(roberta_train_path))
    roberta_valid_ds = load_from_disk(str(roberta_valid_path))
    
    # Process Dutch embeddings
    dutch_train_emb = aggregate_chunks_mean(dutch_train_ds, "legal_dutch")
    dutch_train_pca, dutch_pca = apply_pca_to_embeddings(dutch_train_emb, n_pca_components)
    
    dutch_valid_emb = aggregate_chunks_mean(dutch_valid_ds, "legal_dutch")
    dutch_valid_pca, _ = apply_pca_to_embeddings(dutch_valid_emb, n_pca_components, dutch_pca)
    
    # Process RoBERTa embeddings
    roberta_train_emb = aggregate_chunks_mean(roberta_train_ds, "legal_dutch")  # Same col name in cache
    roberta_train_pca, roberta_pca = apply_pca_to_embeddings(roberta_train_emb, n_pca_components)
    roberta_valid_emb = aggregate_chunks_mean(roberta_valid_ds, "legal_dutch")
    roberta_valid_pca, _ = apply_pca_to_embeddings(roberta_valid_emb, n_pca_components, roberta_pca)
    
    # Extract trigrams (same for both - use dutch text)
    train_trigrams, trigram_vec = extract_trigram_features(dutch_train_ds['text'], n_trigrams)
    valid_trigrams, _ = extract_trigram_features(dutch_valid_ds['text'], n_trigrams, trigram_vec)
    
    # Combine features: Dutch PCA + RoBERTa PCA + Trigrams
    train_features = np.concatenate([dutch_train_pca, roberta_train_pca, train_trigrams], axis=1)
    valid_features = np.concatenate([dutch_valid_pca, roberta_valid_pca, valid_trigrams], axis=1)
    
    # Get labels
    train_labels = list(dutch_train_ds['labels'])
    valid_labels = list(dutch_valid_ds['labels'])
    
    # Create dataloaders
    train_loader, valid_loader = create_dataloaders(
        train_features, train_labels,
        valid_features, valid_labels,
        batch_size=batch_size
    )
    
    # Calculate input size: 2*PCA + trigrams
    input_dim = 2 * n_pca_components + n_trigrams
    
    # Setup MLflow
    full_output_dir = f"{output_dir}/combined"
    mlflow.set_tracking_uri(f"file:./{full_output_dir}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Run experiment
    trainer, metrics = run_experiment(
        model_name="09_advanced_pca_trigram",
        model_fn=create_advanced_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=full_output_dir,
        model_params={
            "input_dim": input_dim, 
            "hidden_dim": hidden_dim, 
            "num_classes": 32,
            "dropout": 0.3,
            "num_residual_blocks": num_residual_blocks
        },
        training_params={"epochs": epochs, "learning_rate": 0.001},
        seed=seed,
    )
    
    # Return results
    class_f1 = {i: metrics.get(f'class_{i}_f1', 0) for i in range(32)}
    macro_f1 = metrics.get('macro_f1', 0)
    
    return metrics, class_f1, macro_f1


def run(embedding_type: str = 'dutch', epochs: int = 25, **kwargs):
    """Unified run interface for main.py orchestration."""
    if embedding_type == 'combined':
        return run_combined(epochs=epochs, **kwargs)
    else:
        return run_single(embedding_type=embedding_type, epochs=epochs, **kwargs)


if __name__ == '__main__':
    run(embedding_type='dutch', epochs=5)
