"""
Generate comprehensive, publication-quality plots for PEFT Benchmarking results.

Plots generated:
 0. Summary Dashboard (Comprehensive 4-panel executive overview)
 1. Accuracy by method (grouped bar chart with error bars across seeds)
 2. Accuracy vs peak VRAM (scatter plot with Pareto trade-off)
 3. Accuracy vs fine-tuning time (scatter plot showing training speed vs performance)
 4. Peak VRAM by method (bar chart comparing memory consumption & savings)
 5. Trainable parameters (% of model) (logarithmic bar chart with parameter counts)
 6. Checkpoint size (logarithmic bar chart showing storage footprint reduction)
 7. Training loss curves (SST-2 for every method across training progression)
 8. Confusion matrices (SST-2, MRPC, RTE comparisons for BERT-base across methods)
 9. Accuracy vs trainable parameters (scatter plot with logarithmic x-axis)
10. Training time vs peak VRAM (2D compute vs memory operational envelope)
11. Seed variability (strip/box/error plot showing performance variance across random seeds)
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Set publication-ready Matplotlib aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.8

# Color palette for methods
METHOD_COLORS = {
    'Full Fine-Tuning': '#2563EB',   # Blue
    'LoRA':             '#10B981',   # Emerald Green
    'AdaLoRA':          '#F59E0B',   # Amber / Orange
    'Prefix Tuning':    '#8B5CF6',   # Purple
    'IA³':              '#EC4899',   # Pink / Rose
    'full':             '#2563EB',
    'lora':             '#10B981',
    'adalora':          '#F59E0B',
    'prefix':           '#8B5CF6',
    'ia3':              '#EC4899',
}

METHOD_ORDER = ['full', 'lora', 'adalora', 'prefix', 'ia3']
METHOD_LABELS = {
    'full': 'Full Fine-Tuning',
    'lora': 'LoRA',
    'adalora': 'AdaLoRA',
    'prefix': 'Prefix Tuning',
    'ia3': 'IA³',
}

MODEL_LABELS = {
    'bert-base-uncased': 'BERT-base',
    'distilbert-base-uncased': 'DistilBERT',
}

DATASET_LABELS = {
    'sst2': 'SST-2',
    'mrpc': 'MRPC',
    'rte': 'RTE',
}

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def load_all_metrics(results_dir=RESULTS_DIR):
    """Load all metrics.json files into a consolidated pandas DataFrame."""
    metrics_files = sorted(glob.glob(os.path.join(results_dir, '**', 'metrics.json'), recursive=True))
    records = []
    
    for mf in metrics_files:
        rel = os.path.relpath(mf, results_dir)
        parts = rel.split(os.sep)
        if len(parts) < 5:
            continue
        model, dataset, method, seed = parts[0], parts[1], parts[2], parts[3]
        
        with open(mf, 'r') as f:
            d = json.load(f)
            
        off = d.get('official_metrics', {})
        diag = d.get('diagnostic_metrics', {})
        eff = d.get('efficiency_metrics', {})
        param = d.get('parameter_metrics', {})
        
        records.append({
            'file_path': mf,
            'model': model,
            'model_name': MODEL_LABELS.get(model, model),
            'dataset': dataset,
            'dataset_name': DATASET_LABELS.get(dataset, dataset.upper()),
            'method': method,
            'method_name': METHOD_LABELS.get(method, method),
            'seed': seed,
            'accuracy': off.get('accuracy', 0.0),
            'accuracy_pct': off.get('accuracy', 0.0) * 100.0,
            'precision': diag.get('precision', 0.0),
            'recall': diag.get('recall', 0.0),
            'macro_f1': diag.get('macro_f1', 0.0),
            'ece': diag.get('expected_calibration_error', 0.0),
            'brier': diag.get('brier_score', 0.0),
            'confusion_matrix': diag.get('confusion_matrix'),
            'training_time_s': eff.get('training_time_seconds', 0.0),
            'training_time_m': eff.get('training_time_seconds', 0.0) / 60.0,
            'train_samples_per_s': eff.get('train_samples_per_second', 0.0),
            'inference_latency_ms': eff.get('inference_latency_ms_per_sample', 0.0),
            'peak_vram_mb': eff.get('peak_vram_mb', 0.0),
            'peak_vram_gb': eff.get('peak_vram_mb', 0.0) / 1024.0,
            'avg_vram_mb': eff.get('avg_vram_mb', 0.0),
            'gpu_util_pct': eff.get('gpu_utilization_pct', 0.0),
            'analytical_flops': eff.get('approximate_analytical_flops_estimate', 0.0),
            'trainable_params': param.get('trainable_parameters', 0),
            'total_params': param.get('total_parameters', 0),
            'pct_trainable': param.get('pct_trainable', 0.0),
            'checkpoint_size_mb': param.get('checkpoint_size_mb', 0.0),
        })
        
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} experiment runs across {df['model'].nunique()} models, {df['dataset'].nunique()} datasets, {df['method'].nunique()} methods.")
    return df


def plot_00_summary_dashboard(df, results_dir=RESULTS_DIR, output_dir=PLOTS_DIR):
    """Plot 0: Comprehensive 4-panel executive summary dashboard."""
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22)
    
    # 1. Top-Left: Accuracy Comparison on SST-2 & MRPC (BERT-base)
    ax1 = fig.add_subplot(gs[0, 0])
    df_bert = df[df['model'] == 'bert-base-uncased']
    methods_bert = [m for m in METHOD_ORDER if m in df_bert['method'].unique()]
    datasets = ['sst2', 'mrpc', 'rte']
    x = np.arange(len(datasets))
    width = 0.15
    
    for m_idx, method in enumerate(methods_bert):
        means = [df_bert[(df_bert['dataset'] == ds) & (df_bert['method'] == method)]['accuracy_pct'].mean() for ds in datasets]
        stds = [df_bert[(df_bert['dataset'] == ds) & (df_bert['method'] == method)]['accuracy_pct'].std() for ds in datasets]
        pos = x - (len(methods_bert)*width)/2 + (m_idx + 0.5)*width
        ax1.bar(pos, means, width, yerr=stds, capsize=3, label=METHOD_LABELS[method],
                color=METHOD_COLORS[method], edgecolor='#1e293b', alpha=0.9)
        
    ax1.set_xticks(x)
    ax1.set_xticklabels(['SST-2 (Sentiment)', 'MRPC (Paraphrase)', 'RTE (Entailment)'], fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('A: Benchmark Accuracy Across GLUE Tasks (BERT-base)', pad=10)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, fontsize=8.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Top-Right: Memory Footprint (Peak VRAM in MB)
    ax2 = fig.add_subplot(gs[0, 1])
    sst2_bert = df_bert[df_bert['dataset'] == 'sst2'].groupby('method')['peak_vram_mb'].mean()
    m_names = [METHOD_LABELS[m] for m in methods_bert if m in sst2_bert]
    m_vrams = [sst2_bert[m] for m in methods_bert if m in sst2_bert]
    m_cols = [METHOD_COLORS[m] for m in methods_bert if m in sst2_bert]
    
    bars = ax2.barh(np.arange(len(m_names)), m_vrams, color=m_cols, edgecolor='#1e293b', height=0.55, alpha=0.9)
    ax2.set_yticks(np.arange(len(m_names)))
    ax2.set_yticklabels(m_names, fontweight='bold')
    ax2.set_xlabel('Peak GPU VRAM Footprint (MB)', fontweight='bold')
    ax2.set_title('B: GPU Memory Footprint During SST-2 Training (BERT-base)', pad=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    full_vram = sst2_bert.get('full', 1482)
    for bar, val in zip(bars, m_vrams):
        red = (1 - (val / full_vram)) * 100
        text_str = f"{val:.0f} MB" if red <= 0 else f"{val:.0f} MB (-{red:.1f}%)"
        ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, text_str,
                 va='center', fontweight='bold', fontsize=8.5)
    ax2.set_xlim(0, max(m_vrams) * 1.25)
    
    # 3. Bottom-Left: Checkpoint Size Reduction (Log scale)
    ax3 = fig.add_subplot(gs[1, 0])
    ckpt_df = df_bert.drop_duplicates(subset=['method']).set_index('method')
    m_ckpts = [ckpt_df.loc[m, 'checkpoint_size_mb'] for m in methods_bert if m in ckpt_df.index]
    bars3 = ax3.bar(np.arange(len(m_names)), m_ckpts, color=m_cols, edgecolor='#1e293b', width=0.55, alpha=0.9)
    ax3.set_yscale('log')
    ax3.set_xticks(np.arange(len(m_names)))
    ax3.set_xticklabels(m_names, rotation=20, ha='right', fontweight='bold')
    ax3.set_ylabel('Checkpoint Size (MB, Log Scale)', fontweight='bold')
    ax3.set_title('C: Fine-Tuned Checkpoint Storage Footprint', pad=10)
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    full_ckpt = ckpt_df.loc['full', 'checkpoint_size_mb'] if 'full' in ckpt_df.index else 4180
    for bar, val in zip(bars3, m_ckpts):
        red = (1 - (val / full_ckpt)) * 100
        text_str = f"{val:.1f}MB" if red <= 0 else f"{val:.1f}MB\n(-{red:.1f}%)"
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3, text_str,
                 ha='center', va='bottom', fontweight='bold', fontsize=8)
    ax3.set_ylim(1, 15000)
    
    # 4. Bottom-Right: Accuracy vs Trainable Parameters Pareto Frontier
    ax4 = fig.add_subplot(gs[1, 1])
    task_markers = {'sst2': 'o', 'mrpc': '^', 'rte': 's'}
    for _, row in df_bert.groupby(['dataset', 'method']).agg({'accuracy_pct': 'mean', 'pct_trainable': 'mean'}).reset_index().iterrows():
        ds = row['dataset']
        m = row['method']
        acc = row['accuracy_pct']
        pct = row['pct_trainable']
        ax4.scatter(pct, acc, color=METHOD_COLORS[m], marker=task_markers[ds], s=90,
                    edgecolor='#0f172a', linewidth=1.1, alpha=0.95, zorder=5)
        
    ax4.set_xscale('log')
    ax4.set_xlabel('Trainable Parameters (% of Pretrained Model, Log Scale)', fontweight='bold')
    ax4.set_ylabel('Evaluation Accuracy (%)', fontweight='bold')
    ax4.set_title('D: Accuracy vs. Parameter Budget (Logarithmic X-Axis)', pad=10)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_xlim(0.015, 250)
    ax4.set_ylim(35, 100)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Legend for task markers
    task_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8, label='SST-2'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8, label='MRPC'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8, label='RTE')
    ]
    ax4.legend(handles=task_patches, loc='lower right', frameon=True, fontsize=8.5)
    
    plt.suptitle('Parameter-Efficient Fine-Tuning (PEFT) Benchmark: Executive Summary', y=0.98, fontsize=16, fontweight='bold')
    out_path = os.path.join(output_dir, '00_peft_benchmark_summary_dashboard.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_01_accuracy_by_method(df, output_dir=PLOTS_DIR):
    """Plot 1: Accuracy by Method across models and datasets (with error bars)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    datasets = ['sst2', 'mrpc', 'rte']
    models = ['bert-base-uncased', 'distilbert-base-uncased']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        df_model = df[df['model'] == model]
        methods_present = [m for m in METHOD_ORDER if m in df_model['method'].unique()]
        
        x = np.arange(len(datasets))
        total_width = 0.76
        n_methods = len(methods_present)
        width = total_width / n_methods
        
        for m_idx, method in enumerate(methods_present):
            means = []
            stds = []
            for ds in datasets:
                subset = df_model[(df_model['dataset'] == ds) & (df_model['method'] == method)]
                if not subset.empty:
                    means.append(subset['accuracy_pct'].mean())
                    stds.append(subset['accuracy_pct'].std())
                else:
                    means.append(0)
                    stds.append(0)
                    
            pos = x - (total_width / 2) + (m_idx + 0.5) * width
            bars = ax.bar(pos, means, width, yerr=stds, capsize=3.5,
                          label=METHOD_LABELS.get(method, method),
                          color=METHOD_COLORS.get(method, '#888888'),
                          edgecolor='#1e293b', linewidth=0.8, alpha=0.9)
            
            for bar, mean_val in zip(bars, means):
                if mean_val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.2,
                            f"{mean_val:.1f}%", ha='center', va='bottom', fontsize=7.5, rotation=0, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], fontsize=11, fontweight='bold')
        ax.set_title(f"Backbone: {MODEL_LABELS.get(model, model)}", pad=12)
        ax.set_xlabel("GLUE Benchmark Task", labelpad=8)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(0, 105)
        if idx == 0:
            ax.set_ylabel("Evaluation Accuracy (%)", labelpad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=5, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10)
    
    plt.suptitle("Accuracy by Method Across GLUE Tasks and Model Architectures", y=1.12, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '01_accuracy_by_method.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_02_accuracy_vs_peak_vram(df, output_dir=PLOTS_DIR):
    """Plot 2: Accuracy vs Peak VRAM (Scatter & Pareto Trade-off)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), sharey=True)
    datasets = ['sst2', 'mrpc', 'rte']
    markers = {'bert-base-uncased': 'o', 'distilbert-base-uncased': 's'}
    
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        df_ds = df[df['dataset'] == ds]
        
        grouped = df_ds.groupby(['model', 'method']).agg({
            'accuracy_pct': ['mean', 'std'],
            'peak_vram_mb': ['mean', 'std']
        }).reset_index()
        
        grouped = grouped.sort_values(by=[('peak_vram_mb', 'mean')])
        
        offset_toggle = 1
        for _, row in grouped.iterrows():
            model = row[('model', '')]
            method = row[('method', '')]
            acc_mean = row[('accuracy_pct', 'mean')]
            acc_std = row[('accuracy_pct', 'std')]
            vram_mean = row[('peak_vram_mb', 'mean')]
            vram_std = row[('peak_vram_mb', 'std')]
            
            color = METHOD_COLORS.get(method, '#888888')
            marker = markers.get(model, 'o')
            
            ax.errorbar(vram_mean, acc_mean, xerr=vram_std, yerr=acc_std,
                        fmt=marker, color=color, markersize=9.5, capsize=3,
                        markeredgecolor='#0f172a', markeredgewidth=1.0, alpha=0.9, zorder=5)
            
            y_off = 10 if offset_toggle > 0 else -16
            offset_toggle *= -1
            if ds == 'mrpc' and acc_mean < 70:
                if method == 'lora':
                    y_off = 9
                elif method == 'adalora':
                    y_off = -15
                elif method == 'ia3':
                    y_off = 9
            
            m_short = METHOD_LABELS.get(method, method)
            b_short = 'BERT' if 'bert-' in model else 'Distil'
            ax.annotate(f"{m_short} ({b_short})\n{acc_mean:.1f}% | {vram_mean:.0f}MB",
                        (vram_mean, acc_mean),
                        textcoords="offset points", xytext=(0, y_off),
                        ha='center', fontsize=7.5, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=color, alpha=0.85, lw=0.8),
                        zorder=6)
            
        ax.set_title(f"Task: {DATASET_LABELS[ds]}", pad=10)
        ax.set_xlabel("Peak GPU VRAM Footprint (MB)", labelpad=8)
        if idx == 0:
            ax.set_ylabel("Evaluation Accuracy (%)", labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(32, 102)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    method_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS[m],
                                 markeredgecolor='#0f172a', markersize=8.5, label=METHOD_LABELS[m])
                      for m in METHOD_ORDER]
    model_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8.5, label='BERT-base (Circle)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8.5, label='DistilBERT (Square)')
    ]
    
    fig.legend(handles=method_patches + model_patches, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=7, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)
    
    plt.suptitle("Accuracy vs. Peak GPU VRAM Footprint Across Methods and Tasks", y=1.16, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '02_accuracy_vs_peak_vram.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_03_accuracy_vs_fine_tuning_time(df, output_dir=PLOTS_DIR):
    """Plot 3: Accuracy vs Fine-Tuning Time (Scatter plot)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), sharey=True)
    datasets = ['sst2', 'mrpc', 'rte']
    markers = {'bert-base-uncased': 'o', 'distilbert-base-uncased': 's'}
    
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        df_ds = df[df['dataset'] == ds]
        
        grouped = df_ds.groupby(['model', 'method']).agg({
            'accuracy_pct': ['mean', 'std'],
            'training_time_s': ['mean', 'std']
        }).reset_index()
        
        offset_toggle = 1
        for _, row in grouped.iterrows():
            model = row[('model', '')]
            method = row[('method', '')]
            acc_mean = row[('accuracy_pct', 'mean')]
            acc_std = row[('accuracy_pct', 'std')]
            time_mean = row[('training_time_s', 'mean')]
            time_std = row[('training_time_s', 'std')]
            
            color = METHOD_COLORS.get(method, '#888888')
            marker = markers.get(model, 'o')
            
            ax.errorbar(time_mean, acc_mean, xerr=time_std, yerr=acc_std,
                        fmt=marker, color=color, markersize=9.5, capsize=3,
                        markeredgecolor='#0f172a', markeredgewidth=1.0, alpha=0.9, zorder=5)
            
            y_off = 10 if offset_toggle > 0 else -16
            offset_toggle *= -1
            m_short = METHOD_LABELS.get(method, method)
            b_short = 'BERT' if 'bert-' in model else 'Distil'
            
            ax.annotate(f"{m_short} ({b_short})\n{acc_mean:.1f}% | {time_mean:.0f}s",
                        (time_mean, acc_mean),
                        textcoords="offset points", xytext=(0, y_off),
                        ha='center', fontsize=7.5, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=color, alpha=0.85, lw=0.8),
                        zorder=6)
            
        ax.set_title(f"Task: {DATASET_LABELS[ds]}", pad=10)
        ax.set_xlabel("Fine-Tuning Time (Seconds)", labelpad=8)
        if idx == 0:
            ax.set_ylabel("Evaluation Accuracy (%)", labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(32, 102)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    method_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS[m],
                                 markeredgecolor='#0f172a', markersize=8.5, label=METHOD_LABELS[m])
                      for m in METHOD_ORDER]
    model_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8.5, label='BERT-base'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8.5, label='DistilBERT')
    ]
    
    fig.legend(handles=method_patches + model_patches, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=7, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)
    
    plt.suptitle("Accuracy vs. Fine-Tuning Time Across GLUE Tasks", y=1.16, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '03_accuracy_vs_finetuning_time.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_04_peak_vram_by_method(df, output_dir=PLOTS_DIR):
    """Plot 4: Peak VRAM by Method (Grouped Bar Chart with VRAM savings)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    models = ['bert-base-uncased', 'distilbert-base-uncased']
    datasets = ['sst2', 'mrpc', 'rte']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        df_model = df[df['model'] == model]
        methods_present = [m for m in METHOD_ORDER if m in df_model['method'].unique()]
        
        x = np.arange(len(datasets))
        total_width = 0.76
        n_methods = len(methods_present)
        width = total_width / n_methods
        
        for m_idx, method in enumerate(methods_present):
            means = []
            stds = []
            for ds in datasets:
                subset = df_model[(df_model['dataset'] == ds) & (df_model['method'] == method)]
                if not subset.empty:
                    means.append(subset['peak_vram_mb'].mean())
                    stds.append(subset['peak_vram_mb'].std())
                else:
                    means.append(0)
                    stds.append(0)
                    
            pos = x - (total_width / 2) + (m_idx + 0.5) * width
            bars = ax.bar(pos, means, width, yerr=stds, capsize=3.5,
                          label=METHOD_LABELS.get(method, method),
                          color=METHOD_COLORS.get(method, '#888888'),
                          edgecolor='#1e293b', linewidth=0.8, alpha=0.9)
            
            for bar, mean_val in zip(bars, means):
                if mean_val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 45,
                            f"{mean_val:.0f}\nMB", ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], fontsize=11, fontweight='bold')
        ax.set_title(f"Backbone: {MODEL_LABELS.get(model, model)}", pad=12)
        ax.set_xlabel("GLUE Benchmark Task", labelpad=8)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(0, 2550)
        if idx == 0:
            ax.set_ylabel("Peak GPU Memory (VRAM MB)", labelpad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=5, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10)
    
    plt.suptitle("Peak GPU VRAM Consumption by Method Across Architectures", y=1.12, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '04_peak_vram_by_method.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_05_trainable_parameters_pct(df, output_dir=PLOTS_DIR):
    """Plot 5: Trainable Parameters (% of Model) (Logarithmic Bar Chart)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    models = ['bert-base-uncased', 'distilbert-base-uncased']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        df_model = df[df['model'] == model].drop_duplicates(subset=['method'])
        df_model = df_model.sort_values(by='pct_trainable', ascending=True)
        
        methods = df_model['method'].tolist()
        pcts = df_model['pct_trainable'].tolist()
        counts = df_model['trainable_params'].tolist()
        total_p = df_model['total_params'].iloc[0] if not df_model.empty else 1
        
        colors = [METHOD_COLORS.get(m, '#888888') for m in methods]
        labels = [METHOD_LABELS.get(m, m) for m in methods]
        
        y_pos = np.arange(len(methods))
        bars = ax.barh(y_pos, pcts, color=colors, edgecolor='#1e293b', linewidth=0.8, alpha=0.9, height=0.55)
        
        ax.set_xscale('log')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_xlabel("Trainable Parameters (% of Pretrained Model, Log Scale)", labelpad=8)
        ax.set_title(f"{MODEL_LABELS.get(model, model)} (Total: {total_p/1e6:.1f}M params)", pad=12)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.set_xlim(0.015, 300)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for bar, pct, cnt in zip(bars, pcts, counts):
            if cnt >= 1e6:
                cnt_str = f"{cnt/1e6:.2f}M"
            elif cnt >= 1e3:
                cnt_str = f"{cnt/1e3:.1f}K"
            else:
                cnt_str = f"{cnt}"
            ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height()/2,
                    f"{pct:.3f}% ({cnt_str} trainable)",
                    va='center', ha='left', fontsize=8.5, fontweight='bold', color='#0f172a')
            
    plt.suptitle("Trainable Parameter Efficiency: Percentage and Absolute Weight Counts", y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '05_trainable_parameters_pct.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_06_checkpoint_size(df, output_dir=PLOTS_DIR):
    """Plot 6: Checkpoint Size (MB) (Logarithmic Bar Chart)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    models = ['bert-base-uncased', 'distilbert-base-uncased']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        df_model = df[df['model'] == model].drop_duplicates(subset=['method'])
        df_model = df_model.sort_values(by='checkpoint_size_mb', ascending=True)
        
        methods = df_model['method'].tolist()
        sizes = df_model['checkpoint_size_mb'].tolist()
        colors = [METHOD_COLORS.get(m, '#888888') for m in methods]
        labels = [METHOD_LABELS.get(m, m) for m in methods]
        
        full_size = df_model[df_model['method'] == 'full']['checkpoint_size_mb'].values
        full_val = full_size[0] if len(full_size) > 0 else max(sizes)
        
        y_pos = np.arange(len(methods))
        bars = ax.barh(y_pos, sizes, color=colors, edgecolor='#1e293b', linewidth=0.8, alpha=0.9, height=0.55)
        
        ax.set_xscale('log')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_xlabel("Checkpoint Disk Footprint (MB, Log Scale)", labelpad=8)
        ax.set_title(f"Backbone: {MODEL_LABELS.get(model, model)}", pad=12)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.set_xlim(2.0, 15000)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for bar, size_mb in zip(bars, sizes):
            reduction = (1 - (size_mb / full_val)) * 100
            if reduction > 0.1:
                label_text = f"{size_mb:.1f} MB (-{reduction:.1f}%)"
            else:
                label_text = f"{size_mb:.1f} MB (Baseline Full)"
            ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height()/2,
                    label_text, va='center', ha='left', fontsize=8.5, fontweight='bold', color='#0f172a')
            
    plt.suptitle("Fine-Tuned Checkpoint Storage Footprint Comparison", y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '06_checkpoint_size.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_07_training_loss_curves_sst2(results_dir=RESULTS_DIR, output_dir=PLOTS_DIR):
    """Plot 7: Training Loss Curves on SST-2 for Every Method (BERT-base and DistilBERT)."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True)
    models = ['bert-base-uncased', 'distilbert-base-uncased']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_dir = os.path.join(results_dir, model, 'sst2')
        methods = [m for m in METHOD_ORDER if os.path.exists(os.path.join(model_dir, m))]
        
        for method in methods:
            color = METHOD_COLORS.get(method, '#888888')
            label = METHOD_LABELS.get(method, method)
            seed_logs = glob.glob(os.path.join(model_dir, method, '*', 'train_log.csv'))
            
            dfs = []
            for slog in seed_logs:
                df_log = pd.read_csv(slog).dropna(subset=['loss', 'epoch'])
                if not df_log.empty:
                    df_log['loss_smooth'] = df_log['loss'].rolling(window=7, min_periods=1).mean()
                    dfs.append(df_log)
                    
            if not dfs:
                continue
                
            common_epochs = np.linspace(0.01, 3.0, 150)
            interp_losses = []
            
            for df_l in dfs:
                interp = np.interp(common_epochs, df_l['epoch'], df_l['loss_smooth'])
                interp_losses.append(interp)
                ax.plot(df_l['epoch'], df_l['loss_smooth'], color=color, alpha=0.18, linewidth=0.9)
                
            interp_arr = np.array(interp_losses)
            mean_loss = np.mean(interp_arr, axis=0)
            std_loss = np.std(interp_arr, axis=0)
            
            ax.plot(common_epochs, mean_loss, color=color, linewidth=2.4, label=label)
            ax.fill_between(common_epochs, np.maximum(0, mean_loss - std_loss), mean_loss + std_loss,
                            color=color, alpha=0.18)
            
        ax.set_title(f"Backbone: {MODEL_LABELS.get(model, model)}", pad=12)
        ax.set_xlabel("Training Epoch (0 to 3.0)", labelpad=8)
        if idx == 0:
            ax.set_ylabel("Cross-Entropy Training Loss", labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 2.05)
        ax.set_xlim(0, 3.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5, loc='upper right')
        
    plt.suptitle("SST-2 Training Loss Trajectory Across PEFT Methods (Mean ± 1 Std across Seeds)", y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '07_training_loss_curves_sst2.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_08_confusion_matrices_bert(results_dir=RESULTS_DIR, output_dir=PLOTS_DIR):
    """Plot 8: Confusion Matrices for BERT-base across SST-2, MRPC, RTE and all PEFT methods."""
    datasets = ['sst2', 'mrpc', 'rte']
    methods = ['full', 'lora', 'adalora', 'prefix', 'ia3']
    
    class_labels = {
        'sst2': ['Negative', 'Positive'],
        'mrpc': ['Not Paraphrase', 'Paraphrase'],
        'rte': ['Entailment', 'Not Entailment'],
    }
    
    fig, axes = plt.subplots(len(datasets), len(methods), figsize=(17.5, 10.5))
    
    for row_idx, ds in enumerate(datasets):
        c_labels = class_labels[ds]
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            
            files = sorted(glob.glob(os.path.join(results_dir, 'bert-base-uncased', ds, method, '*', 'metrics.json')))
            if not files:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue
                
            cms = []
            accs = []
            for fpath in files:
                with open(fpath, 'r') as f:
                    d = json.load(f)
                cms.append(np.array(d.get('diagnostic_metrics', {}).get('confusion_matrix', [[0, 0], [0, 0]])))
                accs.append(d.get('official_metrics', {}).get('accuracy', 0.0) * 100.0)
                
            mean_cm = np.mean(cms, axis=0)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            
            row_sums = mean_cm.sum(axis=1, keepdims=True)
            norm_cm = np.divide(mean_cm, row_sums, out=np.zeros_like(mean_cm, dtype=float), where=row_sums != 0)
            
            im = ax.imshow(norm_cm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
            
            for i in range(2):
                for j in range(2):
                    count_val = mean_cm[i, j]
                    pct_val = norm_cm[i, j] * 100.0
                    cell_text = f"{count_val:.0f}\n({pct_val:.1f}%)"
                    text_color = "white" if norm_cm[i, j] > 0.55 else "black"
                    ax.text(j, i, cell_text, ha="center", va="center", color=text_color,
                            fontsize=9.5, fontweight='bold')
                    
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            
            if row_idx == len(datasets) - 1:
                ax.set_xticklabels(c_labels, fontsize=8.5, rotation=15, ha='right')
                ax.set_xlabel("Predicted Label", fontsize=9.5, labelpad=4)
            else:
                ax.set_xticklabels([])
                
            if col_idx == 0:
                ax.set_yticklabels(c_labels, fontsize=8.5)
                ax.set_ylabel(f"{DATASET_LABELS[ds]}\nTrue Label", fontsize=10, fontweight='bold', labelpad=6)
            else:
                ax.set_yticklabels([])
                
            method_badge = METHOD_LABELS.get(method, method)
            ax.set_title(f"{method_badge}\nAcc: {mean_acc:.1f}% ± {std_acc:.1f}%", fontsize=9.5, pad=6,
                         color=METHOD_COLORS.get(method, '#333333'), fontweight='bold')
            
    plt.suptitle("Confusion Matrix Analysis for BERT-base across GLUE Tasks and PEFT Methods", y=0.99, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '08_confusion_matrices_bert.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_09_accuracy_vs_trainable_params_log(df, output_dir=PLOTS_DIR):
    """Plot 9: Accuracy vs Trainable Parameters (% with logarithmic X-axis)."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True)
    models = ['bert-base-uncased', 'distilbert-base-uncased']
    task_markers = {'sst2': 'o', 'mrpc': '^', 'rte': 's'}
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        df_model = df[df['model'] == model]
        
        grouped = df_model.groupby(['dataset', 'method']).agg({
            'accuracy_pct': ['mean', 'std'],
            'pct_trainable': 'mean',
            'trainable_params': 'mean'
        }).reset_index()
        
        offset_toggle = 1
        for _, row in grouped.iterrows():
            ds = row[('dataset', '')]
            method = row[('method', '')]
            acc_mean = row[('accuracy_pct', 'mean')]
            acc_std = row[('accuracy_pct', 'std')]
            pct_train = row[('pct_trainable', 'mean')]
            cnt_train = row[('trainable_params', 'mean')]
            
            color = METHOD_COLORS.get(method, '#888888')
            marker = task_markers.get(ds, 'o')
            
            ax.errorbar(pct_train, acc_mean, yerr=acc_std,
                        fmt=marker, color=color, markersize=9.5, capsize=3,
                        markeredgecolor='#0f172a', markeredgewidth=1.0, alpha=0.9, zorder=5)
            
            y_off = 9 if offset_toggle > 0 else -15
            offset_toggle *= -1
            if ds == 'mrpc' and acc_mean < 70:
                y_off = -14 if method in ['lora', 'adalora'] else 9
                
            ax.annotate(f"{METHOD_LABELS.get(method, method)} [{DATASET_LABELS[ds]}]",
                        (pct_train, acc_mean),
                        textcoords="offset points", xytext=(0, y_off),
                        ha='center', fontsize=7.5, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=color, alpha=0.85, lw=0.8),
                        zorder=6)
            
        ax.set_xscale('log')
        ax.set_title(f"Backbone: {MODEL_LABELS.get(model, model)}", pad=12)
        ax.set_xlabel("Trainable Parameters (% of Total Model, Logarithmic Scale)", labelpad=8)
        if idx == 0:
            ax.set_ylabel("Evaluation Accuracy (%)", labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0.015, 250)
        ax.set_ylim(32, 102)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    method_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS[m],
                                 markeredgecolor='#0f172a', markersize=8.5, label=METHOD_LABELS[m])
                      for m in METHOD_ORDER]
    task_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8.5, label='SST-2 (Circle)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8.5, label='MRPC (Triangle)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8.5, label='RTE (Square)')
    ]
    
    fig.legend(handles=method_patches + task_patches, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)
    
    plt.suptitle("Accuracy vs. Parameter Adaptation Budget (Logarithmic X-Axis)", y=1.15, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '09_accuracy_vs_trainable_params_log.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_10_training_time_vs_peak_vram(df, output_dir=PLOTS_DIR):
    """Plot 10: Training Time vs Peak VRAM (2D Operational Resource Envelope)."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    models = ['bert-base-uncased', 'distilbert-base-uncased']
    task_markers = {'sst2': 'o', 'mrpc': '^', 'rte': 's'}
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        df_model = df[df['model'] == model]
        
        grouped = df_model.groupby(['dataset', 'method']).agg({
            'training_time_s': ['mean', 'std'],
            'peak_vram_mb': ['mean', 'std']
        }).reset_index()
        
        offset_toggle = 1
        for _, row in grouped.iterrows():
            ds = row[('dataset', '')]
            method = row[('method', '')]
            time_mean = row[('training_time_s', 'mean')]
            time_std = row[('training_time_s', 'std')]
            vram_mean = row[('peak_vram_mb', 'mean')]
            vram_std = row[('peak_vram_mb', 'std')]
            
            color = METHOD_COLORS.get(method, '#888888')
            marker = task_markers.get(ds, 'o')
            
            ax.errorbar(time_mean, vram_mean, xerr=time_std, yerr=vram_std,
                        fmt=marker, color=color, markersize=9.5, capsize=3,
                        markeredgecolor='#0f172a', markeredgewidth=1.0, alpha=0.9, zorder=5)
            
            y_off = 9 if offset_toggle > 0 else -15
            offset_toggle *= -1
            
            ax.annotate(f"{METHOD_LABELS.get(method, method)}\n[{DATASET_LABELS[ds]}]",
                        (time_mean, vram_mean),
                        textcoords="offset points", xytext=(0, y_off),
                        ha='center', fontsize=7.5, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=color, alpha=0.85, lw=0.8),
                        zorder=6)
            
        ax.set_title(f"Backbone: {MODEL_LABELS.get(model, model)}", pad=12)
        ax.set_xlabel("Total Fine-Tuning Duration (Seconds)", labelpad=8)
        ax.set_ylabel("Peak GPU VRAM Footprint (MB)", labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    method_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS[m],
                                 markeredgecolor='#0f172a', markersize=8.5, label=METHOD_LABELS[m])
                      for m in METHOD_ORDER]
    task_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8.5, label='SST-2 (Circle)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8.5, label='MRPC (Triangle)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8.5, label='RTE (Square)')
    ]
    
    fig.legend(handles=method_patches + task_patches, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)
    
    plt.suptitle("Hardware Resource Envelope: Training Time vs. Peak GPU Memory", y=1.15, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '10_training_time_vs_peak_vram.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_11_seed_variability(df, output_dir=PLOTS_DIR):
    """Plot 11: Seed Variability (Strip/Jitter & Box plot of Accuracy across seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True)
    models = ['bert-base-uncased', 'distilbert-base-uncased']
    datasets = ['sst2', 'mrpc', 'rte']
    seed_markers = {'seed42': 'o', 'seed43': 's', 'seed44': '^'}
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        df_model = df[df['model'] == model]
        methods_present = [m for m in METHOD_ORDER if m in df_model['method'].unique()]
        
        x_ticks = []
        x_labels = []
        x_pos_curr = 0
        
        for ds in datasets:
            for m_idx, method in enumerate(methods_present):
                subset = df_model[(df_model['dataset'] == ds) & (df_model['method'] == method)]
                if subset.empty:
                    continue
                    
                color = METHOD_COLORS.get(method, '#888888')
                accs = subset['accuracy_pct'].tolist()
                
                # Box plot summary
                bp = ax.boxplot(accs, positions=[x_pos_curr], widths=0.45,
                                patch_artist=True, showmeans=True,
                                meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor=color, markersize=5),
                                medianprops=dict(color='#0f172a', linewidth=1.5),
                                boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, linewidth=1.2),
                                whiskerprops=dict(color=color, linewidth=1.2),
                                capprops=dict(color=color, linewidth=1.2))
                
                # Plot individual seed points
                for _, srow in subset.iterrows():
                    s = srow['seed']
                    val = srow['accuracy_pct']
                    marker = seed_markers.get(s, 'o')
                    ax.scatter(x_pos_curr, val, color=color, edgecolor='#0f172a',
                               marker=marker, s=38, zorder=4, alpha=0.9)
                    
                x_ticks.append(x_pos_curr)
                x_labels.append(f"{METHOD_LABELS.get(method, method)[:4]}")
                x_pos_curr += 1
                
            x_pos_curr += 0.85  # gap between datasets
            
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8.5, fontweight='bold')
        ax.set_title(f"Backbone: {MODEL_LABELS.get(model, model)}", pad=12)
        if idx == 0:
            ax.set_ylabel("Evaluation Accuracy (%)", labelpad=8)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(32, 102)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Calculate centers for SST-2, MRPC, RTE
        ds_len = len(methods_present)
        c_sst2 = (ds_len - 1) / 2
        c_mrpc = ds_len + 0.85 + (ds_len - 1) / 2
        c_rte = 2 * (ds_len + 0.85) + (ds_len - 1) / 2
        
        for c_pos, ds_name in zip([c_sst2, c_mrpc, c_rte], ['SST-2', 'MRPC', 'RTE']):
            ax.text(c_pos, 98.5, ds_name, ha='center', va='top', fontsize=10.5, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f1f5f9", ec="#cbd5e1", alpha=0.9))

    seed_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=7.5, label='Seed 42 (Circle)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=7.5, label='Seed 43 (Square)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=7.5, label='Seed 44 (Triangle)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=7.5, label='Seed Mean (Diamond)')
    ]
    
    fig.legend(handles=seed_patches, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=4, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9.5)
    
    plt.suptitle("Seed Variability & Performance Stability Across 3 Random Seeds (42, 43, 44)", y=1.12, fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, '11_seed_variability.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("=" * 60)
    print("Starting Plot Generation for PEFT Benchmarking Results...")
    print("=" * 60)
    
    df = load_all_metrics(RESULTS_DIR)
    
    print("\n0. Generating: Executive Summary Dashboard...")
    plot_00_summary_dashboard(df, RESULTS_DIR, PLOTS_DIR)
    
    print("\n1. Generating: Accuracy by Method...")
    plot_01_accuracy_by_method(df, PLOTS_DIR)
    
    print("\n2. Generating: Accuracy vs Peak VRAM...")
    plot_02_accuracy_vs_peak_vram(df, PLOTS_DIR)
    
    print("\n3. Generating: Accuracy vs Fine-Tuning Time...")
    plot_03_accuracy_vs_fine_tuning_time(df, PLOTS_DIR)
    
    print("\n4. Generating: Peak VRAM by Method...")
    plot_04_peak_vram_by_method(df, PLOTS_DIR)
    
    print("\n5. Generating: Trainable Parameters (% of Model)...")
    plot_05_trainable_parameters_pct(df, PLOTS_DIR)
    
    print("\n6. Generating: Checkpoint Size...")
    plot_06_checkpoint_size(df, PLOTS_DIR)
    
    print("\n7. Generating: Training Loss Curves (SST-2)...")
    plot_07_training_loss_curves_sst2(RESULTS_DIR, PLOTS_DIR)
    
    print("\n8. Generating: Confusion Matrices (BERT-base)...")
    plot_08_confusion_matrices_bert(RESULTS_DIR, PLOTS_DIR)
    
    print("\n9. Generating: Accuracy vs Trainable Parameters (Log X)...")
    plot_09_accuracy_vs_trainable_params_log(df, PLOTS_DIR)
    
    print("\n10. Generating: Training Time vs Peak VRAM...")
    plot_10_training_time_vs_peak_vram(df, PLOTS_DIR)
    
    print("\n11. Generating: Seed Variability...")
    plot_11_seed_variability(df, PLOTS_DIR)
    
    print("\n" + "=" * 60)
    print("All Plots Successfully Generated in:", PLOTS_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
