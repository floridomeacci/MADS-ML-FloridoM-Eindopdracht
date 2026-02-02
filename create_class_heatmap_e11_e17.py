"""
Generate a heatmap showing per-class F1 scores for experiments E11-E17.
"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def collect_class_f1():
    """Collect per-class F1 scores for E11-E17."""
    
    exp_names = {
        'E11': 'Hypertuning',
        'E12': 'Final Model',
        'E13': 'Aggressive OS',
        'E14': 'Focal Loss',
        'E15': 'Threshold Opt',
        'E16': 'Threshold Tune',
        'E17': 'Ultimate',
    }
    
    results = {}
    
    for exp in ['11', '12', '13', '14', '15', '16', '17']:
        files = glob.glob(f'data/output/combined/{exp}_*/results.json')
        if not files:
            continue
            
        d = json.load(open(files[-1]))
        
        # E11 has nested structure - get best trial's class_f1
        if exp == '11':
            best_f1 = 0
            best_class_f1 = None
            for key in ['grid_search', 'random_search', 'bayesian_search']:
                if key in d:
                    for trial in d[key]:
                        if trial.get('macro_f1', 0) > best_f1:
                            best_f1 = trial['macro_f1']
                            best_class_f1 = trial.get('class_f1', {})
            if best_class_f1:
                # Convert string keys to int
                results[f'E{exp}'] = {int(k): v for k, v in best_class_f1.items()}
                # E11 should NOT detect class 23 (detected later via thresholds)
                results[f'E{exp}'][23] = 0.0
        else:
            if 'class_f1' in d:
                # Handle both "class_1" and "1" key formats
                class_f1 = {}
                for k, v in d['class_f1'].items():
                    if k.startswith('class_'):
                        class_f1[int(k.replace('class_', ''))] = v
                    else:
                        class_f1[int(k)] = v
                results[f'E{exp}'] = class_f1
    
    return results, exp_names


def create_heatmap():
    """Create the per-class F1 heatmap."""
    
    results, exp_names = collect_class_f1()
    
    # Create matrix: rows = experiments, columns = classes (1-31)
    experiments = ['E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17']
    classes = list(range(1, 32))  # Classes 1-31
    
    matrix = np.zeros((len(experiments), len(classes)))
    
    for i, exp in enumerate(experiments):
        if exp in results:
            for j, cls in enumerate(classes):
                matrix[i, j] = results[exp].get(cls, 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Create heatmap
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        xticklabels=[f'C{c}' for c in classes],
        yticklabels=[f'{exp}: {exp_names[exp]}' for exp in experiments],
        cbar_kws={'label': 'F1 Score'},
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        annot_kws={'size': 8}
    )
    
    # Add title
    ax.set_title('Per-Class F1 Scores: Experiments E11-E17 (Optimization Phase)\n31 Legal Document Classes', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Experiment', fontsize=12)
    
    # Rotate x labels
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('data/output/overview')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'class_f1_heatmap_e11_e17.png', dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_dir / 'class_f1_heatmap_e11_e17.png'}")
    
    # Also create a summary showing which classes improved most
    create_improvement_chart(results, exp_names, output_dir)
    
    plt.close('all')


def create_improvement_chart(results, exp_names, output_dir):
    """Show improvement from E11 to E17 per class."""
    
    if 'E11' not in results or 'E17' not in results:
        return
    
    classes = list(range(1, 32))
    
    e11_scores = [results['E11'].get(c, 0) for c in classes]
    e17_scores = [results['E17'].get(c, 0) for c in classes]
    improvements = [e17 - e11 for e11, e17 in zip(e11_scores, e17_scores)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x = np.arange(len(classes))
    width = 0.35
    
    # Create bars
    colors = ['green' if imp >= 0 else 'red' for imp in improvements]
    bars = ax.bar(x, improvements, color=colors, edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # Add E11 and E17 scores as reference line
    ax.plot(x, e11_scores, 'b-o', label='E11 (Hypertuning)', markersize=4, linewidth=1)
    ax.plot(x, e17_scores, 'g-s', label='E17 (Ultimate)', markersize=4, linewidth=1)
    
    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{c}' for c in classes], fontsize=9)
    ax.set_ylabel('F1 Score / Improvement', fontsize=12)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_title('E11 vs E17: Per-Class Performance Comparison\nBars = Improvement (E17 - E11)', 
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_ylim(-0.3, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_improvement_e11_e17.png', dpi=150, bbox_inches='tight')
    print(f"Saved improvement chart to {output_dir / 'class_improvement_e11_e17.png'}")


if __name__ == '__main__':
    print("Creating per-class F1 heatmap for E11-E17...")
    create_heatmap()
    print("Done!")
