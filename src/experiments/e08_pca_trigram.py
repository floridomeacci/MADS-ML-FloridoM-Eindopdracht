"""
Experiment 08: PCA + Trigram Preprocessing

ARCHITECTURE:
- Preprocessing: Mean aggregation over chunks → PCA → concatenate with trigrams
- Simple feedforward classifier on combined features
- Input: PCA-reduced embedding (256) + trigram features (100) = 356 dims

HYPOTHESIS:
PCA can reduce noise in embeddings while preserving signal.
Character trigrams capture morphological patterns in Dutch legal text
that transformer embeddings might miss (e.g., prefixes, suffixes).

Aggregating chunks first simplifies the architecture while still benefiting
from preprocessing.
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


def aggregate_chunks(dataset, embedding_col: str) -> np.ndarray:
    """
    Aggregate variable-length chunks using mean pooling.
    
    Args:
        dataset: HuggingFace dataset
        embedding_col: Name of the embedding column
        
    Returns:
        Aggregated embeddings of shape (n_samples, embedding_dim)
    """
    # Use dataset.map for efficient processing
    def compute_mean(example):
        chunks = np.array(example[embedding_col], dtype=np.float32)
        return {"mean_embedding": chunks.mean(axis=0).tolist()}
    
    ds_with_means = dataset.map(compute_mean, desc="Aggregating chunks")
    
    # Extract all mean embeddings at once
    embeddings = np.array(ds_with_means["mean_embedding"], dtype=np.float32)
    return embeddings


def apply_pca_to_embeddings(embeddings: np.ndarray, n_components: int = 256, 
                            fitted_pca: PCA = None) -> tuple[np.ndarray, PCA]:
    """
    Apply PCA to reduce embedding dimensionality.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        n_components: Target dimensionality
        fitted_pca: Pre-fitted PCA for validation set (None for training)
        
    Returns:
        Reduced embeddings and PCA model
    """
    if fitted_pca is None:
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(embeddings)
    else:
        pca = fitted_pca
        reduced = pca.transform(embeddings)
    
    return reduced, pca


def extract_trigram_features(texts: list[str], max_features: int = 100,
                            fitted_vectorizer: CountVectorizer = None) -> tuple[np.ndarray, CountVectorizer]:
    """
    Extract character trigram features from text.
    
    Args:
        texts: List of document texts
        max_features: Number of top trigrams to keep
        fitted_vectorizer: Pre-fitted vectorizer for validation set
        
    Returns:
        Trigram feature matrix and vectorizer
    """
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
    Apply PCA + trigram preprocessing to a dataset.
    
    Args:
        dataset: HuggingFace dataset with embeddings and text
        embedding_col: Name of embedding column (e.g., 'legal_dutch')
        n_pca_components: PCA target dimensions
        n_trigrams: Number of trigram features
        fitted_pca: Pre-fitted PCA model (for validation)
        fitted_vectorizer: Pre-fitted trigram vectorizer (for validation)
        
    Returns:
        Preprocessed features, labels, PCA model, and vectorizer
    """
    # Aggregate variable-length chunks using mean pooling
    embeddings = aggregate_chunks(dataset, embedding_col)
    texts = dataset['text']
    labels = list(dataset['labels'])
    
    # Apply PCA
    reduced_embeddings, pca = apply_pca_to_embeddings(embeddings, n_pca_components, fitted_pca)
    
    # Extract trigrams
    trigram_features, trigram_vectorizer = extract_trigram_features(texts, n_trigrams, fitted_vectorizer)
    
    # Concatenate: (n_samples, n_pca_components + n_trigrams)
    combined_features = np.concatenate([reduced_embeddings, trigram_features], axis=1)
    
    return combined_features, labels, pca, trigram_vectorizer


class SimpleClassifier(nn.Module):
    """Simple feedforward classifier for preprocessed features."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def create_pca_trigram_model(input_dim: int = 356, hidden_dim: int = 256, 
                              num_classes: int = 32, dropout: float = 0.3):
    """Create the simple classifier for PCA + trigram features."""
    return SimpleClassifier(input_dim, hidden_dim, num_classes, dropout)


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


def run(embedding_type: str = "dutch", output_dir: str = "experiments", 
        experiment_name: str = "pca-trigram", seed: int = 42,
        n_pca_components: int = 256, n_trigrams: int = 100, batch_size: int = 32):
    """
    Run PCA + Trigram preprocessing experiment.
    
    Args:
        embedding_type: Which base embedding to use ("dutch" or "roberta")
        output_dir: Directory to save MLflow artifacts
        experiment_name: MLflow experiment name
        n_pca_components: Number of PCA components
        n_trigrams: Number of trigram features
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
        # Note: RoBERTa dataset also uses 'legal_dutch' column (naming inconsistency in cache)
        embedding_col = "legal_dutch"
        train_path = next(data_path.glob("*legal_bert*train"))
        valid_path = next(data_path.glob("*legal_bert*valid"))
    elif embedding_type == "combined":
        # For combined: load both, apply PCA separately, concatenate
        dutch_train = load_from_disk(str(next(data_path.glob("*legal_dutch*train"))))
        dutch_valid = load_from_disk(str(next(data_path.glob("*legal_dutch*valid"))))
        roberta_train = load_from_disk(str(next(data_path.glob("*legal_bert*train"))))
        roberta_valid = load_from_disk(str(next(data_path.glob("*legal_bert*valid"))))
        
        # Preprocess Dutch (with trigrams)
        dutch_train_feat, train_labels, pca_dutch, trigram_vec = preprocess_dataset(
            dutch_train, "legal_dutch", n_pca_components, n_trigrams
        )
        dutch_valid_feat, valid_labels, _, _ = preprocess_dataset(
            dutch_valid, "legal_dutch", n_pca_components, n_trigrams,
            fitted_pca=pca_dutch, fitted_vectorizer=trigram_vec
        )
        
        # Preprocess RoBERTa (PCA only, no trigrams)
        roberta_train_emb = aggregate_chunks(roberta_train, "legal_dutch")
        roberta_valid_emb = aggregate_chunks(roberta_valid, "legal_dutch")
        roberta_train_pca, pca_roberta = apply_pca_to_embeddings(roberta_train_emb, n_pca_components)
        roberta_valid_pca, _ = apply_pca_to_embeddings(roberta_valid_emb, n_pca_components, pca_roberta)
        
        # Concatenate: dutch_pca+trigrams + roberta_pca
        train_features = np.concatenate([dutch_train_feat, roberta_train_pca], axis=1)
        valid_features = np.concatenate([dutch_valid_feat, roberta_valid_pca], axis=1)
        
        # Create dataloaders
        train_loader, valid_loader = create_dataloaders(
            train_features, train_labels,
            valid_features, valid_labels,
            batch_size=batch_size
        )
        
        # Combined input: dutch_pca + trigrams + roberta_pca
        input_dim = n_pca_components + n_trigrams + n_pca_components
        
        # Setup MLflow
        full_output_dir = f"{output_dir}/{embedding_type}"
        mlflow.set_tracking_uri(f"file:./{full_output_dir}/mlruns")
        mlflow.set_experiment(experiment_name)
        
        # Run experiment
        trainer, metrics = run_experiment(
            model_name="08_pca_trigram",
            model_fn=create_pca_trigram_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            output_dir=full_output_dir,
            model_params={"input_dim": input_dim, "hidden_dim": 256, "num_classes": 32},
            training_params={"epochs": 5, "learning_rate": 0.001},
            seed=seed,
        )
        
        class_f1 = {i: metrics.get(f'class_{i}_f1', 0) for i in range(32)}
        macro_f1 = metrics.get('macro_f1', 0)
        return metrics, class_f1, macro_f1
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}. Use 'dutch', 'roberta', or 'combined'.")
    
    train_ds = load_from_disk(str(train_path))
    valid_ds = load_from_disk(str(valid_path))
    
    # Preprocess training data (fit PCA and trigram vectorizer)
    train_features, train_labels, pca, trigram_vec = preprocess_dataset(
        train_ds, embedding_col, n_pca_components, n_trigrams
    )
    
    # Preprocess validation data (using fitted PCA and vectorizer)
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
    
    # Calculate input size (PCA dims + trigram dims)
    input_dim = n_pca_components + n_trigrams
    
    # Setup MLflow
    full_output_dir = f"{output_dir}/{embedding_type}"
    mlflow.set_tracking_uri(f"file:./{full_output_dir}/mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Run experiment with simple classifier
    trainer, metrics = run_experiment(
        model_name="08_pca_trigram",
        model_fn=create_pca_trigram_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        output_dir=full_output_dir,
        model_params={"input_dim": input_dim, "hidden_dim": 256, "num_classes": 32},
        training_params={"epochs": 5, "learning_rate": 0.001},
        seed=seed,
    )
    
    # Return consistent format for main.py orchestration
    class_f1 = {i: metrics.get(f'class_{i}_f1', 0) for i in range(32)}
    macro_f1 = metrics.get('macro_f1', 0)
    
    return metrics, class_f1, macro_f1


if __name__ == '__main__':
    run(embedding_type='dutch')
