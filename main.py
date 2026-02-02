"""
Main Entry Point - ORCHESTRATOR
================================

Main controls:
1. Which experiments to run
2. Loops through ALL 3 embedding types (dutch, roberta, combined)
3. Collects results from each
4. Creates comparison heatmap showing all 3

EXPERIMENTS:
    e01-e09: Use loaders (mltrainer-based)
    e10-e17: Self-contained (direct numpy/torch)

USAGE:
    python main.py -e e01              # Run E01 on all 3 embeddings
    python main.py -e e10 --epochs 5   # Run E10 on all 3 embeddings with 5 epochs
    python main.py --list              # List all experiments
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visualize import generate_comparison_heatmap

# Import experiments
from experiments import (
    e01_baseline_mean,
    e02_attention,
    e03_bilstm,
    e04_highway,
    e05_moe,
    e06_rare_classes,
    e07_balanced_classes,
    e08_pca_trigram,
    e09_advanced_pca_trigram,
    e10_adaptive_chunks,
    e11_hypertuning,
    e12_final_model,
    e13_aggressive_oversample,
    e14_focal_loss,
    e15_threshold_optimization,
    e16_threshold_tuning,
    e17_ultimate,
)

# Import loaders
from loaders import legal_dutch, legal_roberta, combined

EXPERIMENTS = {
    'e01': ('Baseline Mean', e01_baseline_mean, 'loader'),
    'e02': ('Attention', e02_attention, 'loader'),
    'e03': ('BiLSTM', e03_bilstm, 'loader'),
    'e04': ('Highway', e04_highway, 'loader'),
    'e05': ('MoE', e05_moe, 'loader'),
    'e06': ('Rare Classes', e06_rare_classes, 'loader'),
    'e07': ('Balanced', e07_balanced_classes, 'loader'),
    'e08': ('PCA Trigram', e08_pca_trigram, 'self'),
    'e09': ('Adv PCA', e09_advanced_pca_trigram, 'self'),
    'e10': ('Adaptive Chunks', e10_adaptive_chunks, 'self'),
    'e11': ('Hypertuning', e11_hypertuning, 'single'),
    'e12': ('Final Model', e12_final_model, 'single'),
    'e13': ('Aggressive OS', e13_aggressive_oversample, 'single'),
    'e14': ('Focal Loss', e14_focal_loss, 'single'),
    'e15': ('Threshold Opt', e15_threshold_optimization, 'single'),
    'e16': ('Threshold Tune', e16_threshold_tuning, 'single'),
    'e17': ('Ultimate', e17_ultimate, 'single'),
}

LOADERS = {
    'dutch': legal_dutch,
    'roberta': legal_roberta,
    'combined': combined,
}


def list_experiments():
    """List all available experiments."""
    print("\n" + "=" * 60)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 60)
    for key, (name, _, exp_type) in EXPERIMENTS.items():
        print(f"  {key}: {name} [{exp_type}]")
    print("=" * 60 + "\n")


def run_experiment(experiment_id: str, epochs: int = 5):
    """
    Run experiment on ALL 3 embedding types and create comparison heatmap.
    
    This is the main orchestrator - it:
    1. Loops through dutch, roberta, combined
    2. Runs the experiment for each
    3. Collects F1 scores
    4. Creates ONE comparison heatmap with all 3
    
    For 'single' experiments (e11-e17), runs once with internal logic.
    """
    if experiment_id not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment_id}")
        list_experiments()
        return
    
    name, module, exp_type = EXPERIMENTS[experiment_id]
    
    print("\n" + "=" * 70)
    print(f"RUNNING {experiment_id.upper()}: {name}")
    
    # Single-run experiments (e11-e17) - don't loop through embeddings
    if exp_type == 'single':
        print(f"Mode: Single run (combined embeddings) | Epochs: {epochs}")
        print("=" * 70)
        
        try:
            results, class_f1, macro_f1 = module.run(epochs=epochs)
        except TypeError:
            # Some don't take epochs
            results, class_f1, macro_f1 = module.run()
        
        print(f"\n✓ {experiment_id.upper()} COMPLETE - Macro F1 = {macro_f1:.4f}")
        
        # Check if results is a dict with multiple runs (like E11 with grid/random/bayesian)
        if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            # Multiple internal runs - use results directly for heatmap
            heatmap_data = results
            print(f"  Multi-run heatmap: {len(heatmap_data)} variants")
        else:
            # Single run - create single-entry heatmap
            label = f"{experiment_id.upper()} (F1={macro_f1:.2f})"
            heatmap_data = {label: class_f1}
        
        generate_comparison_heatmap(
            heatmap_data,
            title=f'{experiment_id.upper()} {name}',
            filename=f'heatmap_{experiment_id}.png'
        )
        return heatmap_data
    
    print(f"Embeddings: dutch, roberta, combined | Epochs: {epochs}")
    print("=" * 70)
    
    all_results = {}
    
    for emb_type in ['dutch', 'roberta', 'combined']:
        print(f"\n{'─' * 50}")
        print(f"  {emb_type.upper()} EMBEDDINGS")
        print(f"{'─' * 50}")
        
        if exp_type == 'loader':
            # Loader-based experiments (e01-e07)
            loader_module = LOADERS[emb_type]
            
            # E06 and E07 need special loaders
            if experiment_id == 'e06':
                train_loader, valid_loader, hidden_size = loader_module.get_rare_class_loaders()
            elif experiment_id == 'e07':
                train_loader, valid_loader, hidden_size = loader_module.get_balanced_class_loaders()
            else:
                train_loader, valid_loader, hidden_size = loader_module.get_loaders()
            
            _, metrics = module.run(
                train_loader=train_loader,
                valid_loader=valid_loader,
                output_dir=f"data/output/{emb_type}",
                experiment_name=f"{emb_type}-classification",
                hidden_size=hidden_size,
            )
            
            # Extract class F1 scores
            class_f1 = {i: metrics.get(f'class_{i}_f1', 0.0) for i in range(32)}
            macro_f1 = metrics.get('macro_f1', 0.0)
            
        else:
            # Self-contained experiments (e08-e17)
            # Each experiment handles its own parameters internally
            try:
                # Try with both parameters
                results, class_f1, macro_f1 = module.run(
                    embedding_type=emb_type,
                    epochs=epochs
                )
            except TypeError:
                # Fall back to just embedding_type (e08, e09 don't take epochs)
                results, class_f1, macro_f1 = module.run(
                    embedding_type=emb_type
                )
        
        # Store results for comparison heatmap
        label = f"{experiment_id.upper()}_{emb_type} (F1={macro_f1:.2f})"
        all_results[label] = class_f1
        
        print(f"  → {emb_type}: Macro F1 = {macro_f1:.4f}")
    
    # Generate comparison heatmap with all 3
    print(f"\n{'─' * 50}")
    print("  GENERATING COMPARISON HEATMAP")
    print(f"{'─' * 50}")
    
    generate_comparison_heatmap(
        all_results,
        title=f'{experiment_id.upper()} {name}: Dutch vs RoBERTa vs Combined',
        filename=f'heatmap_{experiment_id}_comparison.png'
    )
    
    print(f"\n{'=' * 70}")
    print(f"✓ {experiment_id.upper()} COMPLETE - All 3 embeddings done!")
    print(f"{'=' * 70}\n")
    
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='VectorMesh Experiment Runner')
    parser.add_argument('--experiment', '-e', type=str, default='e01',
                        help='Experiment to run (e01-e17)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all experiments')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    run_experiment(args.experiment, epochs=args.epochs)


if __name__ == '__main__':
    main()
