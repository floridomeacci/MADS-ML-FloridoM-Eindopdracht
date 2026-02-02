"""
Generate a comprehensive heatmap overview of all experiments E01-E17.
"""
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def collect_results():
    """Collect all experiment results."""
    results = {}
    
    # E01-E07: Multi-embedding experiments (dutch, roberta, combined)
    for exp in ['01', '02', '03', '04', '05', '06', '07']:
        for emb in ['dutch', 'roberta', 'combined']:
            pattern = f'data/output/{emb}/{exp}_*/'
            dirs = glob.glob(pattern)
            if dirs:
                latest_dir = sorted(dirs)[-1]
                subdirs = sorted(glob.glob(f'{latest_dir}*/results.json'))
                if subdirs:
                    f = subdirs[-1]
                else:
                    f = f'{latest_dir}results.json'
                
                if os.path.exists(f):
                    d = json.load(open(f))
                    macro = d.get('macro_f1', 0)
                    key = f'E{exp}_{emb}'
                    results[key] = {'macro_f1': macro, 'embedding': emb, 'experiment': f'E{exp}'}
    
    # E08-E10: Self-contained with embedding types
    for exp in ['08', '09', '10']:
        for emb in ['dutch', 'roberta', 'combined']:
            # Check direct results.json first
            files = glob.glob(f'data/output/{emb}/{exp}_*/results.json')
            if not files:
                # Check subdirectories
                files = glob.glob(f'data/output/{emb}/{exp}_*/*/results.json')
            if files:
                d = json.load(open(files[-1]))
                macro = d.get('macro_f1', 0)
                key = f'E{exp}_{emb}'
                results[key] = {'macro_f1': macro, 'embedding': emb, 'experiment': f'E{exp}'}
    
    # E11-E17: Combined only
    for exp in ['11', '12', '13', '14', '15', '16', '17']:
        files = glob.glob(f'data/output/combined/{exp}_*/results.json')
        if files:
            d = json.load(open(files[-1]))
            
            # E11 has nested structure - find best from all search methods
            if exp == '11':
                macro = 0
                for key in ['grid_search', 'random_search', 'bayesian_search']:
                    if key in d:
                        for trial in d[key]:
                            if trial.get('macro_f1', 0) > macro:
                                macro = trial['macro_f1']
            else:
                macro = d.get('macro_f1', d.get('macro_f1_optimized', 0))
            
            key = f'E{exp}_combined'
            results[key] = {'macro_f1': macro, 'embedding': 'combined', 'experiment': f'E{exp}'}
    
    return results

def create_heatmap(results):
    """Create a comprehensive heatmap."""
    
    # Experiment names and descriptions
    exp_names = {
        'E01': 'Baseline Mean',
        'E02': 'Attention',
        'E03': 'BiLSTM',
        'E04': 'Highway',
        'E05': 'MoE',
        'E06': 'Rare Classes',
        'E07': 'Balanced',
        'E08': 'PCA Trigram',
        'E09': 'Adv PCA',
        'E10': 'Adaptive OS',
        'E11': 'Hypertuning',
        'E12': 'Final Model',
        'E13': 'Aggressive OS',
        'E14': 'Focal Loss',
        'E15': 'Threshold Opt',
        'E16': 'Threshold Tune',
        'E17': 'Ultimate',
    }
    
    experiments = [f'E{i:02d}' for i in range(1, 18)]
    embeddings = ['dutch', 'roberta', 'combined']
    
    # Create matrix
    matrix = np.full((len(experiments), len(embeddings)), np.nan)
    
    for i, exp in enumerate(experiments):
        for j, emb in enumerate(embeddings):
            key = f'{exp}_{emb}'
            if key in results:
                matrix[i, j] = results[key]['macro_f1']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # Create heatmap with custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Mask NaN values
    mask = np.isnan(matrix)
    
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        mask=mask,
        xticklabels=['Dutch', 'RoBERTa', 'Combined'],
        yticklabels=[f'{exp}: {exp_names[exp]}' for exp in experiments],
        cbar_kws={'label': 'Macro F1 Score'},
        vmin=0,
        vmax=0.8,
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    
    # Add title
    ax.set_title('VectorMesh Experiments Overview\nMacro F1 Scores by Embedding Type', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # Add colored backgrounds for experiment groups
    # E01-E05: Architecture experiments (light blue)
    # E06-E07: Data balancing (light green)
    # E08-E10: Feature engineering (light yellow)
    # E11-E17: Optimization (light purple)
    
    ax.axhline(y=5, color='black', linewidth=2)
    ax.axhline(y=7, color='black', linewidth=2)
    ax.axhline(y=10, color='black', linewidth=2)
    
    # Add group labels on the right
    ax.text(3.3, 2.5, 'Architecture', fontsize=10, va='center', fontweight='bold', color='#1f77b4')
    ax.text(3.3, 6, 'Data Balance', fontsize=10, va='center', fontweight='bold', color='#2ca02c')
    ax.text(3.3, 8.5, 'Features', fontsize=10, va='center', fontweight='bold', color='#ff7f0e')
    ax.text(3.3, 13.5, 'Optimization', fontsize=10, va='center', fontweight='bold', color='#9467bd')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('data/output/overview')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'experiments_overview_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_dir / 'experiments_overview_heatmap.png'}")
    
    # Also create a summary bar chart
    create_summary_chart(results, exp_names, output_dir)
    
    plt.close('all')  # Close figures instead of showing

def create_summary_chart(results, exp_names, output_dir):
    """Create a grouped bar chart showing all embeddings per experiment."""
    
    experiments = [f'E{i:02d}' for i in range(1, 18)]
    embeddings = ['dutch', 'roberta', 'combined']
    
    # Collect scores for each embedding
    scores = {emb: [] for emb in embeddings}
    
    for exp in experiments:
        for emb in embeddings:
            key = f'{exp}_{emb}'
            if key in results:
                scores[emb].append(results[key]['macro_f1'])
            else:
                scores[emb].append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Bar positions
    x = np.arange(len(experiments))
    width = 0.25
    
    # Colors
    colors = {'dutch': '#1f77b4', 'roberta': '#ff7f0e', 'combined': '#2ca02c'}
    
    # Create grouped bars
    bars_dutch = ax.bar(x - width, scores['dutch'], width, label='Dutch', color=colors['dutch'], edgecolor='black', linewidth=0.5)
    bars_roberta = ax.bar(x, scores['roberta'], width, label='RoBERTa', color=colors['roberta'], edgecolor='black', linewidth=0.5)
    bars_combined = ax.bar(x + width, scores['combined'], width, label='Combined', color=colors['combined'], edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars (only for combined to avoid clutter)
    for bar, score in zip(bars_combined, scores['combined']):
        if score > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels([f'{exp}\n{exp_names[exp]}' for exp in experiments], rotation=45, ha='right')
    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_title('Macro F1 Score by Experiment and Embedding Type', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.85)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.7, color='darkgreen', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add vertical lines to separate experiment groups
    ax.axvline(x=4.5, color='black', linestyle='-', alpha=0.3, linewidth=2)  # After E05
    ax.axvline(x=6.5, color='black', linestyle='-', alpha=0.3, linewidth=2)  # After E07
    ax.axvline(x=9.5, color='black', linestyle='-', alpha=0.3, linewidth=2)  # After E10
    
    # Add group labels
    ax.text(2, 0.82, 'Architecture', ha='center', fontsize=10, fontweight='bold', color='#1f77b4')
    ax.text(5.5, 0.82, 'Balance', ha='center', fontsize=10, fontweight='bold', color='#2ca02c')
    ax.text(8, 0.82, 'Features', ha='center', fontsize=10, fontweight='bold', color='#ff7f0e')
    ax.text(13, 0.82, 'Optimization', ha='center', fontsize=10, fontweight='bold', color='#9467bd')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experiments_summary_barchart.png', dpi=150, bbox_inches='tight')
    print(f"Saved bar chart to {output_dir / 'experiments_summary_barchart.png'}")


if __name__ == '__main__':
    print("Collecting experiment results...")
    results = collect_results()
    
    print("\nResults collected:")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v['macro_f1']:.4f}")
    
    print("\nGenerating heatmap...")
    create_heatmap(results)
