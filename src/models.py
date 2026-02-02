"""
Model Architecture Definitions

This module contains different pipeline architectures for legal text
classification, ranging from simple baselines to advanced MoE systems.

Architecture Progression:
1. Baseline: MeanAggregator + NeuralNet
2. Multi-Modal: Parallel pipelines with multiple features
3. Advanced: MoE, Highway networks, Attention aggregation
"""

from vectormesh.components import (
    MeanAggregator,
    NeuralNet,
    Serial,
    Parallel,
    Concatenate2D,
    AttentionAggregator,
    RNNAggregator,
    Skip,
    Highway,
    Projection,
)
from vectormesh.components.gating import MoE


def create_baseline_model(hidden_size: int = 768, out_size: int = 32):
    """
    Create a simple baseline model.
    
    Architecture:
    - Input: (batch, chunks, dim) - chunked document embeddings
    - MeanAggregator: Average pooling over chunks → (batch, dim)
    - NeuralNet: Two-layer MLP with dropout → (batch, out_size)
    
    This serves as the baseline to compare against more complex architectures.
    Simple averaging works surprisingly well for document classification.
    
    Args:
        hidden_size: Dimension of input embeddings (768 for BERT)
        out_size: Number of output classes (32 legal facts)
        
    Returns:
        Serial pipeline ready for training
    """
    return Serial([
        MeanAggregator(),  # (batch chunks dim) -> (batch dim)
        NeuralNet(hidden_size=hidden_size, out_size=out_size)
    ])


def create_attention_model(hidden_size: int = 768, out_size: int = 32):
    """
    Create model with learnable attention aggregation.
    
    Architecture:
    - Input: (batch, chunks, dim)
    - AttentionAggregator: Learned attention weights over chunks
    - NeuralNet: Classification head
    
    Attention allows the model to focus on important chunks (e.g., specific
    legal clauses) while downweighting less relevant sections.
    
    Args:
        hidden_size: Dimension of input embeddings
        out_size: Number of output classes
        
    Returns:
        Serial pipeline with attention mechanism
    """
    return Serial([
        AttentionAggregator(hidden_size=hidden_size),
        NeuralNet(hidden_size=hidden_size, out_size=out_size)
    ])


def create_rnn_model(hidden_size: int = 768, out_size: int = 32):
    """
    Create model with sequential (GRU) aggregation.
    
    Architecture:
    - Input: (batch, chunks, dim)
    - RNNAggregator: GRU processes chunks sequentially
    - NeuralNet: Classification head
    
    RNN aggregation captures sequential dependencies between document
    chunks, useful when order matters (narrative structure in legal texts).
    
    Args:
        hidden_size: Dimension of input embeddings
        out_size: Number of output classes
        
    Returns:
        Serial pipeline with RNN aggregation
    """
    return Serial([
        RNNAggregator(hidden_size=hidden_size),
        NeuralNet(hidden_size=hidden_size, out_size=out_size)
    ])


def create_highway_model(hidden_size: int = 768, out_size: int = 32):
    """
    Create model with Highway network gating.
    
    Architecture:
    - Input: (batch, chunks, dim)
    - MeanAggregator: Average pooling
    - Highway: Gated transformation with skip connection
    - NeuralNet: Classification head
    
    Highway networks use learned gates to control information flow.
    The gate decides how much of the transformed vs original representation
    to pass through, enabling deeper architectures.
    
    Args:
        hidden_size: Dimension of input embeddings
        out_size: Number of output classes
        
    Returns:
        Serial pipeline with Highway gating
    """
    return Serial([
        MeanAggregator(),
        Highway(
            transform=NeuralNet(hidden_size=hidden_size, out_size=hidden_size),
            hidden_size=hidden_size
        ),
        NeuralNet(hidden_size=hidden_size, out_size=out_size)
    ])


def create_moe_model(
    hidden_size: int = 768,
    out_size: int = 32,
    num_experts: int = 4,
    top_k: int = 2
):
    """
    Create Mixture of Experts model with sparse gating.
    
    Architecture:
    - Input: (batch, chunks, dim)
    - MeanAggregator: Average pooling
    - MoE: Multiple expert networks with top-k routing
    
    The MoE layer contains multiple expert networks (neural nets).
    For each sample, a gating network selects the top-k most relevant
    experts. This allows specialization - different experts can learn
    different types of legal reasoning.
    
    Sparsely-gated MoE scales model capacity without proportionally
    increasing computation, as only k experts are active per sample.
    
    Args:
        hidden_size: Dimension of input embeddings
        out_size: Number of output classes
        num_experts: Number of expert networks
        top_k: Number of experts to activate per sample
        
    Returns:
        Serial pipeline with MoE layer
        
    Note:
        Typically num_experts=4-8 and top_k=2 works well.
        Higher values increase capacity but also computation.
    """
    experts = [
        NeuralNet(hidden_size=hidden_size, out_size=out_size)
        for _ in range(num_experts)
    ]
    
    moe = MoE(
        experts=experts,
        hidden_size=hidden_size,
        out_size=out_size,
        top_k=top_k
    )
    
    return Serial([MeanAggregator(), moe])


def create_multimodal_model(
    bert_dim: int = 768,
    regex_dim: int = 123,
    out_size: int = 32
):
    """
    Create multi-modal model combining embeddings and regex features.
    
    Architecture:
    - Input: ((batch chunks bert_dim), (batch regex_dim))
    - Parallel processing:
      * Branch 1: MeanAggregator + NeuralNet for embeddings
      * Branch 2: NeuralNet for regex features
    - Concatenate outputs
    - Final NeuralNet for classification
    
    This combines semantic understanding (from BERT) with structural
    patterns (from regex). Legal documents have both semantic content
    and formal structure (article references, citations).
    
    Args:
        bert_dim: Dimension of BERT embeddings (typically 768)
        regex_dim: Dimension of regex feature vector
        out_size: Number of output classes
        
    Returns:
        Serial pipeline with parallel processing
        
    Note:
        Requires CollateParallel for proper data loading.
    """
    parallel = Parallel([
        # Branch 1: Process 3D BERT embeddings
        Serial([
            MeanAggregator(),
            NeuralNet(hidden_size=bert_dim, out_size=32)
        ]),
        # Branch 2: Process 1D regex features
        Serial([
            NeuralNet(hidden_size=regex_dim, out_size=32)
        ])
    ])
    
    return Serial([
        parallel,
        Concatenate2D(),  # Combine both branches
        NeuralNet(hidden_size=64, out_size=out_size)
    ])
