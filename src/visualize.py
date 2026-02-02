"""
Unified Visualization Module
============================
Generates heatmaps for experiment results.
Call from any experiment to create a heatmap.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json


# All 31 classes (class 0 doesn't exist)
ALL_CLASSES = list(range(1, 32))
STRUGGLING_CLASSES = [5, 8, 21, 23, 24, 26, 30]

# Output directory for visuals
VISUALS_DIR = Path(__file__).parent.parent / 'data' / 'visuals'
VISUALS_DIR.mkdir(exist_ok=True)


def generate_heatmap(
    experiment_name: str,
    class_f1: dict,
    macro_f1: float = None,
    micro_f1: float = None,
    compare_with: dict = None,
    save: bool = True,
    show: bool = False
) -> str:
    """
    Generate a heatmap for experiment results.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'E17_ultimate')
        class_f1: Dict of {class_id: f1_score} for all classes
        macro_f1: Overall macro F1 score
        micro_f1: Overall micro F1 score
        compare_with: Optional dict of {experiment_name: {class_id: f1}} for comparison
        save: Whether to save the heatmap to file
        show: Whether to display the heatmap
        
    Returns:
        Path to saved heatmap file
    """
    # Prepare data
    f1_values = [class_f1.get(c, 0) for c in ALL_CLASSES]
    
    if compare_with:
        # Comparison heatmap
        experiments = list(compare_with.keys()) + [experiment_name]
        matrix = []
        for exp_name in compare_with.keys():
            exp_f1 = compare_with[exp_name]
            matrix.append([exp_f1.get(c, 0) for c in ALL_CLASSES])
        matrix.append(f1_values)
        matrix = np.array(matrix)
        
        fig, ax = plt.subplots(figsize=(18, 2 + len(experiments)))
        sns.heatmap(
            matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=[f'{c}' for c in ALL_CLASSES],
            yticklabels=experiments,
            vmin=0, vmax=1,
            cbar_kws={'label': 'F1 Score'},
            ax=ax,
            annot_kws={'size': 8}
        )
    else:
        # Single experiment heatmap
        matrix = np.array([f1_values])
        
        fig, ax = plt.subplots(figsize=(18, 3))
        sns.heatmap(
            matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=[f'{c}' for c in ALL_CLASSES],
            yticklabels=[experiment_name],
            vmin=0, vmax=1,
            cbar_kws={'label': 'F1 Score'},
            ax=ax,
            annot_kws={'size': 10}
        )
    
    # Title
    title = f'{experiment_name}: All Classes F1 Scores'
    if macro_f1:
        title += f' (Macro F1 = {macro_f1:.4f})'
    if micro_f1:
        title += f' | Micro F1 = {micro_f1:.4f}'
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Class')
    
    # Mark struggling classes
    for i, c in enumerate(ALL_CLASSES):
        if c in STRUGGLING_CLASSES:
            ax.axvline(x=i, color='orange', linewidth=0.5, alpha=0.5)
            ax.axvline(x=i+1, color='orange', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    if save:
        filename = f'heatmap_{experiment_name.lower().replace(" ", "_")}.png'
        filepath = VISUALS_DIR / filename
        plt.savefig(filepath, dpi=150)
        print(f'Saved heatmap: {filepath}')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(filepath) if save else None


def generate_comparison_heatmap(
    experiments_data: dict,
    title: str = 'Experiment Comparison',
    filename: str = 'heatmap_comparison.png',
    show: bool = False
) -> str:
    """
    Generate a comparison heatmap for multiple experiments.
    
    Args:
        experiments_data: Dict of {experiment_name: {class_id: f1_score}}
        title: Title for the heatmap
        filename: Output filename
        show: Whether to display
        
    Returns:
        Path to saved file
    """
    experiments = list(experiments_data.keys())
    matrix = []
    
    for exp_name in experiments:
        exp_f1 = experiments_data[exp_name]
        matrix.append([exp_f1.get(c, 0) for c in ALL_CLASSES])
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(18, 2 + len(experiments)))
    sns.heatmap(
        matrix, annot=True, fmt='.2f', cmap='RdYlGn',
        xticklabels=[f'{c}' for c in ALL_CLASSES],
        yticklabels=experiments,
        vmin=0, vmax=1,
        cbar_kws={'label': 'F1 Score'},
        ax=ax,
        annot_kws={'size': 8}
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Class')
    plt.tight_layout()
    
    filepath = VISUALS_DIR / filename
    plt.savefig(filepath, dpi=150)
    print(f'Saved comparison heatmap: {filepath}')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(filepath)


def generate_struggling_classes_heatmap(
    experiments_data: dict,
    title: str = 'Struggling Classes Progress',
    filename: str = 'heatmap_struggling.png',
    show: bool = False
) -> str:
    """
    Generate heatmap focused on struggling classes only.
    
    Args:
        experiments_data: Dict of {experiment_name: {class_id: f1_score}}
        
    Returns:
        Path to saved file
    """
    experiments = list(experiments_data.keys())
    matrix = []
    
    for exp_name in experiments:
        exp_f1 = experiments_data[exp_name]
        matrix.append([exp_f1.get(c, 0) for c in STRUGGLING_CLASSES])
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(10, 2 + len(experiments)))
    sns.heatmap(
        matrix, annot=True, fmt='.2f', cmap='RdYlGn',
        xticklabels=[f'Class {c}' for c in STRUGGLING_CLASSES],
        yticklabels=experiments,
        vmin=0, vmax=1,
        cbar_kws={'label': 'F1 Score'},
        ax=ax,
        annot_kws={'size': 10}
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Struggling Class')
    plt.tight_layout()
    
    filepath = VISUALS_DIR / filename
    plt.savefig(filepath, dpi=150)
    print(f'Saved struggling classes heatmap: {filepath}')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(filepath)


def load_experiment_results(experiment_dir: str) -> dict:
    """Load results.json from an experiment directory."""
    results_path = Path(experiment_dir) / 'results.json'
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        # Convert class_f1 keys from 'class_1' to 1
        class_f1 = {}
        for k, v in data.get('class_f1', {}).items():
            class_id = int(k.replace('class_', ''))
            class_f1[class_id] = v
        return {
            'class_f1': class_f1,
            'macro_f1': data.get('macro_f1_optimized', data.get('macro_f1', 0)),
            'micro_f1': data.get('micro_f1', 0)
        }
    return None


# Historical data for comparison (hardcoded for convenience)
HISTORICAL_DATA = {
    'E12': {
        1: 0.9153, 2: 0.9189, 3: 0.9189, 4: 0.9667, 5: 0.5333,
        6: 0.9870, 7: 0.8780, 8: 0.4000, 9: 0.7273, 10: 0.7619,
        11: 0.8533, 12: 0.6950, 13: 0.5455, 14: 0.7045, 15: 0.8824,
        16: 0.9785, 17: 0.9772, 18: 0.8571, 19: 0.9091, 20: 0.7059,
        21: 0.4762, 22: 1.0000, 23: 0.0000, 24: 0.6000, 25: 0.9091,
        26: 0.1667, 27: 0.7586, 28: 0.8333, 29: 0.7857, 30: 0.5600,
        31: 0.7500
    },
    'E16': {
        1: 0.9333, 2: 0.9756, 3: 0.9189, 4: 0.9587, 5: 0.6154,
        6: 0.9889, 7: 0.8980, 8: 0.3582, 9: 0.7735, 10: 0.8462,
        11: 0.8961, 12: 0.6992, 13: 0.6250, 14: 0.7619, 15: 0.8947,
        16: 0.9814, 17: 0.9791, 18: 0.7778, 19: 1.0000, 20: 0.7500,
        21: 0.5455, 22: 1.0000, 23: 0.1818, 24: 0.7273, 25: 0.9091,
        26: 0.1667, 27: 0.8000, 28: 0.8571, 29: 0.8333, 30: 0.7500,
        31: 0.9000
    },
    'E17': {
        1: 0.9333, 2: 1.0000, 3: 0.9189, 4: 0.9667, 5: 0.6047,
        6: 0.9889, 7: 0.9011, 8: 0.3662, 9: 0.7598, 10: 0.8387,
        11: 0.8820, 12: 0.7050, 13: 0.6250, 14: 0.7805, 15: 0.9167,
        16: 0.9811, 17: 0.9790, 18: 0.8571, 19: 1.0000, 20: 0.7500,
        21: 0.5833, 22: 1.0000, 23: 0.2000, 24: 0.6667, 25: 1.0000,
        26: 0.1667, 27: 0.8148, 28: 0.8571, 29: 0.8333, 30: 0.7500,
        31: 0.8889
    }
}


if __name__ == '__main__':
    # Example: Generate comparison heatmap for E12, E16, E17
    print("Generating example heatmaps...")
    
    generate_comparison_heatmap(
        HISTORICAL_DATA,
        title='Experiment Comparison: E12 → E16 → E17',
        filename='heatmap_comparison_e12_e16_e17.png'
    )
    
    generate_struggling_classes_heatmap(
        HISTORICAL_DATA,
        title='Struggling Classes: E12 → E16 → E17',
        filename='heatmap_struggling_e12_e16_e17.png'
    )
    
    print("Done!")
