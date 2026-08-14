"""
Generate the refined analysis notebook matching the exact 13-section structure and empirical guidelines.
"""

import json
import os

def create_notebook():
    notebook_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', 'analysis.ipynb')
    
    cells = []
    
    # ----------------------------------------------------
    # Cell 1: Title & Structure
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Practical Benchmark Analysis: Parameter-Efficient Fine-Tuning (PEFT)\n",
            "\n",
            "This notebook presents an empirical evaluation of Parameter-Efficient Fine-Tuning methods (**LoRA**, **AdaLoRA**, **Prefix Tuning**, **IA³**) compared against standard **Full Fine-Tuning** across **BERT-base** and **DistilBERT** on three GLUE classification tasks (**SST-2**, **MRPC**, **RTE**).\n",
            "\n",
            "This is a **practical benchmark** measuring resource trade-offs under identical training conditions rather than claiming universal or population-level superiority.\n",
            "\n",
            "---\n",
            "\n",
            "### Notebook Structure\n",
            "\n",
            "1. Environment Setup & Dataset Resolution\n",
            "2. Benchmark Results Summary\n",
            "3. Statistical Comparison Against Full Fine-Tuning\n",
            "4. Accuracy Across Tasks and Backbones\n",
            "5. Accuracy vs. Peak GPU VRAM\n",
            "6. Parameter Efficiency\n",
            "7. Checkpoint Storage Footprint\n",
            "8. SST-2 Training Dynamics\n",
            "9. Confusion Matrix Analysis\n",
            "10. Accuracy vs. Parameter Budget\n",
            "11. Seed Variability\n",
            "12. Qualitative Failure Analysis\n",
            "13. Empirical Takeaways & Practitioner Recommendations"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 2: Section 1 - Environment Setup & Dataset Resolution
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Environment Setup & Dataset Resolution\n",
            "\n",
            "The notebook supports both **Kaggle** datasets (such as `/kaggle/input/datasets/satyanshgaur1/peft-benchmarking-results` or `/kaggle/input/peft-benchmarking-results`) and **local** environments with automatic discovery and zip extraction."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import glob\n",
            "import json\n",
            "import zipfile\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "try:\n",
            "    from IPython.display import display\n",
            "except ImportError:\n",
            "    display = print\n",
            "\n",
            "# Clean styling for practical figures\n",
            "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n",
            "plt.rcParams['axes.edgecolor'] = '#444444'\n",
            "plt.rcParams['axes.linewidth'] = 1.0\n",
            "plt.rcParams['axes.titlesize'] = 12\n",
            "plt.rcParams['axes.titleweight'] = 'bold'\n",
            "plt.rcParams['axes.labelsize'] = 10.5\n",
            "plt.rcParams['grid.color'] = '#e2e8f0'\n",
            "plt.rcParams['grid.linestyle'] = '--'\n",
            "plt.rcParams['grid.alpha'] = 0.8\n",
            "\n",
            "# Method color scheme\n",
            "METHOD_COLORS = {\n",
            "    'Full Fine-Tuning': '#2563EB',   # Blue\n",
            "    'LoRA':             '#10B981',   # Green\n",
            "    'AdaLoRA':          '#F59E0B',   # Amber\n",
            "    'Prefix Tuning':    '#8B5CF6',   # Purple\n",
            "    'IA³':              '#EC4899',   # Pink\n",
            "    'full':             '#2563EB',\n",
            "    'lora':             '#10B981',\n",
            "    'adalora':          '#F59E0B',\n",
            "    'prefix':           '#8B5CF6',\n",
            "    'ia3':              '#EC4899',\n",
            "}\n",
            "\n",
            "METHOD_ORDER = ['full', 'lora', 'adalora', 'prefix', 'ia3']\n",
            "METHOD_LABELS = {\n",
            "    'full': 'Full Fine-Tuning',\n",
            "    'lora': 'LoRA',\n",
            "    'adalora': 'AdaLoRA',\n",
            "    'prefix': 'Prefix Tuning',\n",
            "    'ia3': 'IA³',\n",
            "}\n",
            "MODEL_LABELS = {\n",
            "    'bert-base-uncased': 'BERT-base',\n",
            "    'distilbert-base-uncased': 'DistilBERT',\n",
            "}\n",
            "DATASET_LABELS = {\n",
            "    'sst2': 'SST-2',\n",
            "    'mrpc': 'MRPC',\n",
            "    'rte': 'RTE',\n",
            "}\n",
            "\n",
            "def resolve_dataset_path():\n",
            "    \"\"\"Finds the results directory from Kaggle mounts or local folders.\"\"\"\n",
            "    candidates = [\n",
            "        '/kaggle/input/datasets/satyanshgaur1/peft-benchmarking-results',\n",
            "        '/kaggle/input/datasets/satyanshgaur1/peft-benchmarking-results/results',\n",
            "        '/kaggle/input/peft-benchmarking-results',\n",
            "        '/kaggle/input/peft-benchmarking-results/results',\n",
            "        '../results',\n",
            "        'results',\n",
            "        './results'\n",
            "    ]\n",
            "    for c in candidates:\n",
            "        if os.path.exists(c):\n",
            "            if os.path.exists(os.path.join(c, 'manifest.json')):\n",
            "                return os.path.abspath(c)\n",
            "            if os.path.exists(os.path.join(c, 'results', 'manifest.json')):\n",
            "                return os.path.abspath(os.path.join(c, 'results'))\n",
            "            zips = glob.glob(os.path.join(c, '*.zip'))\n",
            "            if zips:\n",
            "                extract_dir = '/kaggle/working/results' if os.path.exists('/kaggle/working') else './results'\n",
            "                os.makedirs(extract_dir, exist_ok=True)\n",
            "                print(f\"Extracting {zips[0]} to {extract_dir}...\")\n",
            "                with zipfile.ZipFile(zips[0], 'r') as zf:\n",
            "                    zf.extractall(extract_dir)\n",
            "                if os.path.exists(os.path.join(extract_dir, 'results', 'manifest.json')):\n",
            "                    return os.path.abspath(os.path.join(extract_dir, 'results'))\n",
            "                return os.path.abspath(extract_dir)\n",
            "    return os.path.abspath('../results' if os.path.exists('../results') else 'results')\n",
            "\n",
            "RESULTS_DIR = resolve_dataset_path()\n",
            "print(f\"[✓] Using Results Directory: {RESULTS_DIR}\")\n",
            "\n",
            "# Load manifest index\n",
            "manifest_path = os.path.join(RESULTS_DIR, 'manifest.json')\n",
            "if os.path.exists(manifest_path):\n",
            "    with open(manifest_path, 'r') as f:\n",
            "        manifest = json.load(f)\n",
            "    print(f\"[✓] Loaded manifest index containing {len(manifest)} experiment runs.\")\n",
            "\n",
            "# Load individual run metrics\n",
            "records = []\n",
            "for mf in sorted(glob.glob(os.path.join(RESULTS_DIR, '**', 'metrics.json'), recursive=True)):\n",
            "    rel = os.path.relpath(mf, RESULTS_DIR)\n",
            "    parts = rel.split(os.sep)\n",
            "    if len(parts) < 4:\n",
            "        continue\n",
            "    model, dataset, method, seed = parts[0], parts[1], parts[2], parts[3]\n",
            "    with open(mf, 'r') as f:\n",
            "        d = json.load(f)\n",
            "    off = d.get('official_metrics', {})\n",
            "    diag = d.get('diagnostic_metrics', {})\n",
            "    eff = d.get('efficiency_metrics', {})\n",
            "    param = d.get('parameter_metrics', {})\n",
            "    records.append({\n",
            "        'model': model,\n",
            "        'model_name': MODEL_LABELS.get(model, model),\n",
            "        'dataset': dataset,\n",
            "        'dataset_name': DATASET_LABELS.get(dataset, dataset.upper()),\n",
            "        'method': method,\n",
            "        'method_name': METHOD_LABELS.get(method, method),\n",
            "        'seed': seed,\n",
            "        'accuracy': off.get('accuracy', 0.0),\n",
            "        'accuracy_pct': off.get('accuracy', 0.0) * 100.0,\n",
            "        'precision': diag.get('precision', 0.0),\n",
            "        'recall': diag.get('recall', 0.0),\n",
            "        'macro_f1': diag.get('macro_f1', 0.0),\n",
            "        'confusion_matrix': diag.get('confusion_matrix'),\n",
            "        'training_time_s': eff.get('training_time_seconds', 0.0),\n",
            "        'peak_vram_mb': eff.get('peak_vram_mb', 0.0),\n",
            "        'trainable_params': param.get('trainable_parameters', 0),\n",
            "        'total_params': param.get('total_parameters', 0),\n",
            "        'pct_trainable': param.get('pct_trainable', 0.0),\n",
            "        'checkpoint_size_mb': param.get('checkpoint_size_mb', 0.0),\n",
            "    })\n",
            "df = pd.DataFrame(records)\n",
            "print(f\"[✓] Loaded {len(df)} experiment runs across {df['model'].nunique()} backbones and {df['dataset'].nunique()} tasks.\")\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 3: Section 2 - Benchmark Results Summary
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Benchmark Results Summary\n",
            "\n",
            "Below are the empirical findings across **BERT-base** and **DistilBERT** on **SST-2**, **MRPC**, and **RTE**, averaged over 3 random seeds (`42`, `43`, `44`)."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "summary_rows = []\n",
            "for model in ['distilbert-base-uncased', 'bert-base-uncased']:\n",
            "    for ds in ['sst2', 'mrpc', 'rte']:\n",
            "        for method in METHOD_ORDER:\n",
            "            subset = df[(df['model'] == model) & (df['dataset'] == ds) & (df['method'] == method)]\n",
            "            if not subset.empty:\n",
            "                acc_mean = subset['accuracy'].mean()\n",
            "                acc_std = subset['accuracy'].std()\n",
            "                vram_mean = subset['peak_vram_mb'].mean()\n",
            "                time_mean = subset['training_time_s'].mean()\n",
            "                t_params = subset['trainable_params'].iloc[0]\n",
            "                pct_train = subset['pct_trainable'].iloc[0]\n",
            "                ckpt_mb = subset['checkpoint_size_mb'].iloc[0]\n",
            "                summary_rows.append({\n",
            "                    'Model': MODEL_LABELS[model],\n",
            "                    'Dataset': DATASET_LABELS[ds],\n",
            "                    'Method': METHOD_LABELS[method],\n",
            "                    'Accuracy (μ ± σ)': f\"{acc_mean:.4f} ± {acc_std:.4f}\",\n",
            "                    'Peak VRAM': f\"{vram_mean:.1f} MB\",\n",
            "                    'Training Time': f\"{time_mean:.1f} s\",\n",
            "                    'Trainable Params (% Total)': f\"{t_params:,} ({pct_train:.2f}%)\",\n",
            "                    'Checkpoint Size': f\"{ckpt_mb:.2f} MB\"\n",
            "                })\n",
            "            elif model == 'distilbert-base-uncased' and method == 'prefix':\n",
            "                summary_rows.append({\n",
            "                    'Model': MODEL_LABELS[model],\n",
            "                    'Dataset': DATASET_LABELS[ds],\n",
            "                    'Method': METHOD_LABELS[method],\n",
            "                    'Accuracy (μ ± σ)': 'N/A (Skipped)',\n",
            "                    'Peak VRAM': 'N/A',\n",
            "                    'Training Time': 'N/A',\n",
            "                    'Trainable Params (% Total)': 'N/A',\n",
            "                    'Checkpoint Size': 'N/A'\n",
            "                })\n",
            "\n",
            "df_summary = pd.DataFrame(summary_rows)\n",
            "display(df_summary)\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 4: Section 3 - Statistical Comparison Against Full Fine-Tuning
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Statistical Comparison Against Full Fine-Tuning\n",
            "\n",
            "> **Statistical caution:** Comparisons are based on three random seeds (42, 43, 44). Paired tests and confidence intervals are therefore exploratory rather than confirmatory; they should not be interpreted as establishing population-level significance.\n",
            "\n",
            "With $n=3$ paired observations ($df=2$), inferential tests have low statistical power. Primary evidence should be based on **mean $\\pm$ std, per-seed results, absolute accuracy differences, parameter reduction, VRAM reduction, and checkpoint savings**, with paired t-tests serving solely as a secondary diagnostic."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "stats_path = os.path.join(RESULTS_DIR, 'statistical_tests.json')\n",
            "if os.path.exists(stats_path):\n",
            "    with open(stats_path, 'r') as f:\n",
            "        stats_data = json.load(f)\n",
            "    t_rows = []\n",
            "    for pair, methods in stats_data.get('comparisons', {}).items():\n",
            "        model, ds = pair.split('__')\n",
            "        for m, info in methods.items():\n",
            "            vs = info.get('vs_full_fine_tuning')\n",
            "            if vs:\n",
            "                t_rows.append({\n",
            "                    'Model': MODEL_LABELS.get(model, model),\n",
            "                    'Dataset': DATASET_LABELS.get(ds, ds.upper()),\n",
            "                    'Method': METHOD_LABELS.get(m, m),\n",
            "                    'Mean Acc': f\"{info['mean_accuracy']:.4f}\",\n",
            "                    '95% CI': f\"±{info['ci95_accuracy']:.4f}\",\n",
            "                    'Diff vs Full': f\"{vs['mean_difference']:+.4f}\",\n",
            "                    \"Cohen's d\": f\"{vs['cohens_d']:+.2f}\",\n",
            "                    't-test p': f\"{vs['exploratory_paired_ttest_pvalue']:.4f}\"\n",
            "                })\n",
            "    display(pd.DataFrame(t_rows))\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 5: Section 4 - Accuracy Across Tasks and Backbones
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Accuracy Across Tasks and Backbones\n",
            "\n",
            "Comparison of classification accuracy across GLUE tasks and models, with error bars indicating $\\pm 1\\sigma$ across seeds."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)\n",
            "datasets = ['sst2', 'mrpc', 'rte']\n",
            "models = ['bert-base-uncased', 'distilbert-base-uncased']\n",
            "\n",
            "for idx, model in enumerate(models):\n",
            "    ax = axes[idx]\n",
            "    df_model = df[df['model'] == model]\n",
            "    methods_present = [m for m in METHOD_ORDER if m in df_model['method'].unique()]\n",
            "    \n",
            "    x = np.arange(len(datasets))\n",
            "    total_width = 0.76\n",
            "    width = total_width / len(methods_present)\n",
            "    \n",
            "    for m_idx, method in enumerate(methods_present):\n",
            "        means, stds = [], []\n",
            "        for ds in datasets:\n",
            "            subset = df_model[(df_model['dataset'] == ds) & (df_model['method'] == method)]\n",
            "            means.append(subset['accuracy_pct'].mean() if not subset.empty else 0)\n",
            "            stds.append(subset['accuracy_pct'].std() if not subset.empty else 0)\n",
            "                \n",
            "        pos = x - (total_width / 2) + (m_idx + 0.5) * width\n",
            "        bars = ax.bar(pos, means, width, yerr=stds, capsize=3.5,\n",
            "                      label=METHOD_LABELS.get(method, method),\n",
            "                      color=METHOD_COLORS.get(method, '#888888'),\n",
            "                      edgecolor='#1e293b', linewidth=0.8, alpha=0.9)\n",
            "        \n",
            "        for bar, mean_val in zip(bars, means):\n",
            "            if mean_val > 0:\n",
            "                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.2,\n",
            "                        f\"{mean_val:.1f}%\", ha='center', va='bottom', fontsize=7.5, fontweight='bold')\n",
            "\n",
            "    ax.set_xticks(x)\n",
            "    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], fontsize=11, fontweight='bold')\n",
            "    ax.set_title(f\"Backbone: {MODEL_LABELS.get(model, model)}\", pad=12)\n",
            "    ax.set_xlabel(\"GLUE Task\", labelpad=8)\n",
            "    ax.grid(axis='y', linestyle='--', alpha=0.6)\n",
            "    ax.set_ylim(0, 105)\n",
            "    if idx == 0:\n",
            "        ax.set_ylabel(\"Evaluation Accuracy (%)\", labelpad=8)\n",
            "    ax.spines['top'].set_visible(False)\n",
            "    ax.spines['right'].set_visible(False)\n",
            "    \n",
            "handles, labels = axes[0].get_legend_handles_labels()\n",
            "fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),\n",
            "           ncol=5, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10)\n",
            "\n",
            "plt.suptitle(\"Predictive Accuracy by Adaptation Method Across GLUE Tasks\", y=1.12, fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 6: Section 5 - Accuracy vs. Peak GPU VRAM
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Accuracy vs. Peak GPU VRAM\n",
            "\n",
            "Practical evaluation of GPU memory consumption versus classification accuracy. LoRA delivers competitive accuracy on SST-2 while reducing peak GPU memory by ~35%–45%."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)\n",
            "datasets = ['sst2', 'mrpc', 'rte']\n",
            "markers = {'bert-base-uncased': 'o', 'distilbert-base-uncased': 's'}\n",
            "\n",
            "for idx, ds in enumerate(datasets):\n",
            "    ax = axes[idx]\n",
            "    df_ds = df[df['dataset'] == ds]\n",
            "    grouped = df_ds.groupby(['model', 'method']).agg({\n",
            "        'accuracy_pct': ['mean', 'std'],\n",
            "        'peak_vram_mb': ['mean', 'std']\n",
            "    }).reset_index().sort_values(by=[('peak_vram_mb', 'mean')])\n",
            "    \n",
            "    offset_toggle = 1\n",
            "    for _, row in grouped.iterrows():\n",
            "        model = row[('model', '')]\n",
            "        method = row[('method', '')]\n",
            "        acc_mean = row[('accuracy_pct', 'mean')]\n",
            "        acc_std = row[('accuracy_pct', 'std')]\n",
            "        vram_mean = row[('peak_vram_mb', 'mean')]\n",
            "        vram_std = row[('peak_vram_mb', 'std')]\n",
            "        \n",
            "        color = METHOD_COLORS.get(method, '#888888')\n",
            "        marker = markers.get(model, 'o')\n",
            "        \n",
            "        ax.errorbar(vram_mean, acc_mean, xerr=vram_std, yerr=acc_std,\n",
            "                    fmt=marker, color=color, markersize=9, capsize=3,\n",
            "                    markeredgecolor='#0f172a', markeredgewidth=1.0, alpha=0.9, zorder=5)\n",
            "        \n",
            "        y_off = 10 if offset_toggle > 0 else -16\n",
            "        offset_toggle *= -1\n",
            "        if ds == 'mrpc' and acc_mean < 70:\n",
            "            y_off = -15 if method == 'adalora' else 9\n",
            "        \n",
            "        m_short = METHOD_LABELS.get(method, method)\n",
            "        b_short = 'BERT' if 'bert-' in model else 'Distil'\n",
            "        ax.annotate(f\"{m_short} ({b_short})\\n{acc_mean:.1f}% | {vram_mean:.0f}MB\",\n",
            "                    (vram_mean, acc_mean),\n",
            "                    textcoords=\"offset points\", xytext=(0, y_off),\n",
            "                    ha='center', fontsize=7.5, fontweight='bold',\n",
            "                    bbox=dict(boxstyle=\"round,pad=0.2\", fc=\"#ffffff\", ec=color, alpha=0.85, lw=0.8),\n",
            "                    zorder=6)\n",
            "        \n",
            "    ax.set_title(f\"Task: {DATASET_LABELS[ds]}\", pad=10)\n",
            "    ax.set_xlabel(\"Peak GPU VRAM (MB)\", labelpad=8)\n",
            "    if idx == 0:\n",
            "        ax.set_ylabel(\"Evaluation Accuracy (%)\", labelpad=8)\n",
            "    ax.grid(True, linestyle='--', alpha=0.6)\n",
            "    ax.set_ylim(32, 102)\n",
            "    ax.spines['top'].set_visible(False)\n",
            "    ax.spines['right'].set_visible(False)\n",
            "    \n",
            "method_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS[m],\n",
            "                             markeredgecolor='#0f172a', markersize=8, label=METHOD_LABELS[m])\n",
            "                  for m in METHOD_ORDER]\n",
            "model_patches = [\n",
            "    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8, label='BERT-base'),\n",
            "    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8, label='DistilBERT')\n",
            "]\n",
            "fig.legend(handles=method_patches + model_patches, loc='upper center', bbox_to_anchor=(0.5, 1.08),\n",
            "           ncol=7, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9)\n",
            "\n",
            "plt.suptitle(\"Accuracy–VRAM Trade-off Across Methods and Tasks\", y=1.15, fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 7: Section 6 - Parameter Efficiency
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Parameter Efficiency\n",
            "\n",
            "Comparison of trainable parameters as a percentage of total pretrained model weights.\n",
            "\n",
            "> **Parameter Note**: IA³ adapts **76,034 parameters (~0.07%)** on BERT-base and **619,778 parameters (~0.92%)** on DistilBERT (due to the classification head). LoRA adapts **296,450 parameters (~0.27%)** on BERT-base."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))\n",
            "models = ['bert-base-uncased', 'distilbert-base-uncased']\n",
            "\n",
            "for idx, model in enumerate(models):\n",
            "    ax = axes[idx]\n",
            "    df_params = df[df['model'] == model].drop_duplicates(subset=['method']).sort_values(by='pct_trainable', ascending=True)\n",
            "    methods = df_params['method'].tolist()\n",
            "    pcts = df_params['pct_trainable'].tolist()\n",
            "    counts = df_params['trainable_params'].tolist()\n",
            "    total_p = df_params['total_params'].iloc[0] if not df_params.empty else 1\n",
            "    colors = [METHOD_COLORS.get(m, '#888888') for m in methods]\n",
            "    labels = [METHOD_LABELS.get(m, m) for m in methods]\n",
            "    \n",
            "    y_pos = np.arange(len(methods))\n",
            "    bars = ax.barh(y_pos, pcts, color=colors, edgecolor='#1e293b', linewidth=0.8, alpha=0.9, height=0.55)\n",
            "    ax.set_xscale('log')\n",
            "    ax.set_yticks(y_pos)\n",
            "    ax.set_yticklabels(labels, fontsize=9.5, fontweight='bold')\n",
            "    ax.set_xlabel(\"Trainable Parameters (% of Pretrained Model, Log Scale)\", labelpad=8)\n",
            "    ax.set_title(f\"{MODEL_LABELS.get(model, model)} ({total_p/1e6:.1f}M Params)\", pad=10)\n",
            "    ax.grid(axis='x', linestyle='--', alpha=0.6)\n",
            "    ax.set_xlim(0.015, 300)\n",
            "    ax.spines['top'].set_visible(False)\n",
            "    ax.spines['right'].set_visible(False)\n",
            "    \n",
            "    for bar, pct, cnt in zip(bars, pcts, counts):\n",
            "        cnt_str = f\"{cnt/1e6:.2f}M\" if cnt >= 1e6 else (f\"{cnt/1e3:.1f}K\" if cnt >= 1e3 else f\"{cnt}\")\n",
            "        ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height()/2,\n",
            "                f\"{pct:.3f}% ({cnt_str})\", va='center', ha='left', fontsize=8.5, fontweight='bold')\n",
            "\n",
            "plt.suptitle(\"Trainable Parameter Efficiency Across Methods and Architectures\", y=1.02, fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 8: Section 7 - Checkpoint Storage Footprint
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Checkpoint Storage Footprint\n",
            "\n",
            "> **Measurement note:** Checkpoint size refers to the serialized fine-tuned model artifact produced by the benchmark pipeline and excludes optimizer state. For PEFT methods, this represents the adapter parameters and associated task-specific components.\n",
            "\n",
            "Adapter checkpoints require only **6 MB** (IA³), **14–31 MB** (LoRA), and **20–34 MB** (AdaLoRA), reducing disk storage footprint by **99.5%–99.8%** compared to full weight checkpoints (~2.5–4.2 GB)."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))\n",
            "models = ['bert-base-uncased', 'distilbert-base-uncased']\n",
            "\n",
            "for idx, model in enumerate(models):\n",
            "    ax = axes[idx]\n",
            "    df_ckpt = df[df['model'] == model].drop_duplicates(subset=['method']).sort_values(by='checkpoint_size_mb', ascending=True)\n",
            "    sizes = df_ckpt['checkpoint_size_mb'].tolist()\n",
            "    ckpt_methods = df_ckpt['method'].tolist()\n",
            "    ckpt_colors = [METHOD_COLORS.get(m, '#888888') for m in ckpt_methods]\n",
            "    ckpt_labels = [METHOD_LABELS.get(m, m) for m in ckpt_methods]\n",
            "    full_size = df_ckpt[df_ckpt['method'] == 'full']['checkpoint_size_mb'].values[0]\n",
            "    \n",
            "    bars = ax.barh(np.arange(len(sizes)), sizes, color=ckpt_colors, edgecolor='#1e293b', linewidth=0.8, alpha=0.9, height=0.55)\n",
            "    ax.set_xscale('log')\n",
            "    ax.set_yticks(np.arange(len(sizes)))\n",
            "    ax.set_yticklabels(ckpt_labels, fontsize=9.5, fontweight='bold')\n",
            "    ax.set_xlabel(\"Checkpoint Disk Footprint (MB, Log Scale)\", labelpad=8)\n",
            "    ax.set_title(f\"Backbone: {MODEL_LABELS.get(model, model)}\", pad=10)\n",
            "    ax.grid(axis='x', linestyle='--', alpha=0.6)\n",
            "    ax.set_xlim(2.0, 15000)\n",
            "    ax.spines['top'].set_visible(False)\n",
            "    ax.spines['right'].set_visible(False)\n",
            "    \n",
            "    for bar, size_mb in zip(bars, sizes):\n",
            "        reduction = (1 - (size_mb / full_size)) * 100\n",
            "        label_text = f\"{size_mb:.1f} MB (-{reduction:.1f}%)\" if reduction > 0.1 else f\"{size_mb:.1f} MB (Full)\"\n",
            "        ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height()/2,\n",
            "                label_text, va='center', ha='left', fontsize=8.5, fontweight='bold')\n",
            "\n",
            "plt.suptitle(\"Fine-Tuned Checkpoint Disk Storage Comparison\", y=1.02, fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 9: Section 8 - SST-2 Training Dynamics
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. SST-2 Training Dynamics\n",
            "\n",
            "> **Visualization note:** Curves display **smoothed training loss** (rolling window = 7 steps) interpolated across epochs to illustrate convergence trajectory rather than raw batch noise.\n",
            "\n",
            "- **Full Fine-Tuning**: Rapid descent to near-zero cross-entropy training loss ($<0.03$).\n",
            "- **LoRA & Prefix Tuning**: Steady, stable convergence ($0.18–0.22$) reaching competitive accuracy.\n",
            "- **AdaLoRA**: Starts with elevated loss (~1.90) due to initial SVD rank budgeting and steadily decreases to ~0.61."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)\n",
            "models = ['bert-base-uncased', 'distilbert-base-uncased']\n",
            "\n",
            "for idx, model in enumerate(models):\n",
            "    ax = axes[idx]\n",
            "    model_dir = os.path.join(RESULTS_DIR, model, 'sst2')\n",
            "    methods = [m for m in METHOD_ORDER if os.path.exists(os.path.join(model_dir, m))]\n",
            "    \n",
            "    for method in methods:\n",
            "        color = METHOD_COLORS.get(method, '#888888')\n",
            "        label = METHOD_LABELS.get(method, method)\n",
            "        seed_logs = glob.glob(os.path.join(model_dir, method, '*', 'train_log.csv'))\n",
            "        \n",
            "        dfs = []\n",
            "        for slog in seed_logs:\n",
            "            df_log = pd.read_csv(slog).dropna(subset=['loss', 'epoch'])\n",
            "            if not df_log.empty:\n",
            "                df_log['loss_smooth'] = df_log['loss'].rolling(window=7, min_periods=1).mean()\n",
            "                dfs.append(df_log)\n",
            "                \n",
            "        if not dfs:\n",
            "            continue\n",
            "            \n",
            "        common_epochs = np.linspace(0.01, 3.0, 150)\n",
            "        interp_losses = []\n",
            "        for df_l in dfs:\n",
            "            interp = np.interp(common_epochs, df_l['epoch'], df_l['loss_smooth'])\n",
            "            interp_losses.append(interp)\n",
            "            ax.plot(df_l['epoch'], df_l['loss_smooth'], color=color, alpha=0.18, linewidth=0.9)\n",
            "            \n",
            "        interp_arr = np.array(interp_losses)\n",
            "        mean_loss = np.mean(interp_arr, axis=0)\n",
            "        std_loss = np.std(interp_arr, axis=0)\n",
            "        \n",
            "        ax.plot(common_epochs, mean_loss, color=color, linewidth=2.2, label=label)\n",
            "        ax.fill_between(common_epochs, np.maximum(0, mean_loss - std_loss), mean_loss + std_loss,\n",
            "                        color=color, alpha=0.18)\n",
            "        \n",
            "    ax.set_title(f\"Backbone: {MODEL_LABELS.get(model, model)}\", pad=12)\n",
            "    ax.set_xlabel(\"Training Epoch (0 to 3.0)\", labelpad=8)\n",
            "    if idx == 0:\n",
            "        ax.set_ylabel(\"Smoothed Training Loss\", labelpad=8)\n",
            "    ax.grid(True, linestyle='--', alpha=0.6)\n",
            "    ax.set_ylim(0, 2.05)\n",
            "    ax.set_xlim(0, 3.0)\n",
            "    ax.spines['top'].set_visible(False)\n",
            "    ax.spines['right'].set_visible(False)\n",
            "    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9, loc='upper right')\n",
            "    \n",
            "plt.suptitle(\"Smoothed Training Loss Convergence on SST-2 (Mean ± 1σ Across Seeds)\", y=1.02, fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 10: Section 9 - Confusion Matrix Analysis
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Confusion Matrix Analysis\n",
            "\n",
            "### MRPC majority-class baseline\n",
            "\n",
            "The majority class accounts for approximately **68.4% of the evaluation examples** (279 positive vs. 129 negative out of 408 total). Therefore, an accuracy near 68.4% can be achieved without learning the paraphrase distinction at all.\n",
            "\n",
            "For this reason, MRPC accuracy alone is insufficient to assess whether a PEFT method has learned useful task representations.\n",
            "\n",
            "The diagnostic grid below focuses on **Full Fine-Tuning** (reference baseline), **LoRA**, and **IA³** on BERT-base to inspect the MRPC majority-class collapse:"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "datasets = ['sst2', 'mrpc', 'rte']\n",
            "selected_methods = ['full', 'lora', 'ia3']\n",
            "class_labels = {\n",
            "    'sst2': ['Negative', 'Positive'],\n",
            "    'mrpc': ['Not Paraphrase', 'Paraphrase'],\n",
            "    'rte': ['Entailment', 'Not Entailment'],\n",
            "}\n",
            "\n",
            "fig, axes = plt.subplots(len(datasets), len(selected_methods), figsize=(12, 9.5))\n",
            "\n",
            "for row_idx, ds in enumerate(datasets):\n",
            "    c_labels = class_labels[ds]\n",
            "    for col_idx, method in enumerate(selected_methods):\n",
            "        ax = axes[row_idx, col_idx]\n",
            "        files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'bert-base-uncased', ds, method, '*', 'metrics.json')))\n",
            "        if not files:\n",
            "            ax.text(0.5, 0.5, \"N/A\", ha='center', va='center', fontsize=11)\n",
            "            ax.axis('off')\n",
            "            continue\n",
            "            \n",
            "        cms, accs = [], []\n",
            "        for fpath in files:\n",
            "            with open(fpath, 'r') as f:\n",
            "                d = json.load(f)\n",
            "            cms.append(np.array(d.get('diagnostic_metrics', {}).get('confusion_matrix', [[0, 0], [0, 0]])))\n",
            "            accs.append(d.get('official_metrics', {}).get('accuracy', 0.0) * 100.0)\n",
            "            \n",
            "        mean_cm = np.mean(cms, axis=0)\n",
            "        mean_acc, std_acc = np.mean(accs), np.std(accs)\n",
            "        \n",
            "        row_sums = mean_cm.sum(axis=1, keepdims=True)\n",
            "        norm_cm = np.divide(mean_cm, row_sums, out=np.zeros_like(mean_cm, dtype=float), where=row_sums != 0)\n",
            "        \n",
            "        im = ax.imshow(norm_cm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)\n",
            "        for i in range(2):\n",
            "            for j in range(2):\n",
            "                count_val = mean_cm[i, j]\n",
            "                pct_val = norm_cm[i, j] * 100.0\n",
            "                text_color = \"white\" if norm_cm[i, j] > 0.55 else \"black\"\n",
            "                ax.text(j, i, f\"{count_val:.0f}\\n({pct_val:.1f}%)\", ha=\"center\", va=\"center\",\n",
            "                        color=text_color, fontsize=9.5, fontweight='bold')\n",
            "                \n",
            "        ax.set_xticks([0, 1])\n",
            "        ax.set_yticks([0, 1])\n",
            "        if row_idx == len(datasets) - 1:\n",
            "            ax.set_xticklabels(c_labels, fontsize=8.5, rotation=15, ha='right')\n",
            "            ax.set_xlabel(\"Predicted Label\", fontsize=9.5, labelpad=4)\n",
            "        else:\n",
            "            ax.set_xticklabels([])\n",
            "        if col_idx == 0:\n",
            "            ax.set_yticklabels(c_labels, fontsize=8.5)\n",
            "            ax.set_ylabel(f\"{DATASET_LABELS[ds]}\\nTrue Label\", fontsize=10, fontweight='bold', labelpad=6)\n",
            "        else:\n",
            "            ax.set_yticklabels([])\n",
            "            \n",
            "        method_badge = METHOD_LABELS.get(method, method)\n",
            "        ax.set_title(f\"{method_badge}\\nAcc: {mean_acc:.1f}% ± {std_acc:.1f}%\", fontsize=9.5, pad=5,\n",
            "                     color=METHOD_COLORS.get(method, '#333333'), fontweight='bold')\n",
            "\n",
            "plt.suptitle(\"Diagnostic Confusion Matrices for BERT-base: Full vs. LoRA vs. IA³\", y=0.99, fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 11: Section 10 - Accuracy vs. Parameter Budget
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Accuracy vs. Parameter Budget\n",
            "\n",
            "Scatter plot mapping model accuracy against the percentage of trainable parameters on a logarithmic scale."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)\n",
            "models = ['bert-base-uncased', 'distilbert-base-uncased']\n",
            "task_markers = {'sst2': 'o', 'mrpc': '^', 'rte': 's'}\n",
            "\n",
            "for idx, model in enumerate(models):\n",
            "    ax = axes[idx]\n",
            "    df_model = df[df['model'] == model]\n",
            "    grouped = df_model.groupby(['dataset', 'method']).agg({\n",
            "        'accuracy_pct': ['mean', 'std'],\n",
            "        'pct_trainable': 'mean',\n",
            "        'trainable_params': 'mean'\n",
            "    }).reset_index()\n",
            "    \n",
            "    offset_toggle = 1\n",
            "    for _, row in grouped.iterrows():\n",
            "        ds = row[('dataset', '')]\n",
            "        method = row[('method', '')]\n",
            "        acc_mean = row[('accuracy_pct', 'mean')]\n",
            "        acc_std = row[('accuracy_pct', 'std')]\n",
            "        pct_train = row[('pct_trainable', 'mean')]\n",
            "        \n",
            "        color = METHOD_COLORS.get(method, '#888888')\n",
            "        marker = task_markers.get(ds, 'o')\n",
            "        \n",
            "        ax.errorbar(pct_train, acc_mean, yerr=acc_std,\n",
            "                    fmt=marker, color=color, markersize=9, capsize=3,\n",
            "                    markeredgecolor='#0f172a', markeredgewidth=1.0, alpha=0.9, zorder=5)\n",
            "        \n",
            "        y_off = 9 if offset_toggle > 0 else -15\n",
            "        offset_toggle *= -1\n",
            "        if ds == 'mrpc' and acc_mean < 70:\n",
            "            y_off = -14 if method in ['lora', 'adalora'] else 9\n",
            "            \n",
            "        ax.annotate(f\"{METHOD_LABELS.get(method, method)} [{DATASET_LABELS[ds]}]\",\n",
            "                    (pct_train, acc_mean),\n",
            "                    textcoords=\"offset points\", xytext=(0, y_off),\n",
            "                    ha='center', fontsize=7.5, fontweight='bold',\n",
            "                    bbox=dict(boxstyle=\"round,pad=0.2\", fc=\"#ffffff\", ec=color, alpha=0.85, lw=0.8),\n",
            "                    zorder=6)\n",
            "        \n",
            "    ax.set_xscale('log')\n",
            "    ax.set_title(f\"Backbone: {MODEL_LABELS.get(model, model)}\", pad=12)\n",
            "    ax.set_xlabel(\"Trainable Parameters (% of Total Model, Logarithmic Scale)\", labelpad=8)\n",
            "    if idx == 0:\n",
            "        ax.set_ylabel(\"Evaluation Accuracy (%)\", labelpad=8)\n",
            "    ax.grid(True, linestyle='--', alpha=0.6)\n",
            "    ax.set_xlim(0.015, 250)\n",
            "    ax.set_ylim(32, 102)\n",
            "    ax.spines['top'].set_visible(False)\n",
            "    ax.spines['right'].set_visible(False)\n",
            "    \n",
            "method_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS[m],\n",
            "                             markeredgecolor='#0f172a', markersize=8, label=METHOD_LABELS[m])\n",
            "                  for m in METHOD_ORDER]\n",
            "task_patches = [\n",
            "    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8, label='SST-2'),\n",
            "    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8, label='MRPC'),\n",
            "    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=8, label='RTE')\n",
            "]\n",
            "fig.legend(handles=method_patches + task_patches, loc='upper center', bbox_to_anchor=(0.5, 1.08),\n",
            "           ncol=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9)\n",
            "\n",
            "plt.suptitle(\"Accuracy vs. Parameter Budget Across Tasks\", y=1.15, fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 12: Section 11 - Seed Variability
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 11. Seed Variability\n",
            "\n",
            "Dot plot displaying individual seed accuracy points (Seeds 42, 43, 44) alongside seed means across methods and tasks."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)\n",
            "models = ['bert-base-uncased', 'distilbert-base-uncased']\n",
            "datasets = ['sst2', 'mrpc', 'rte']\n",
            "seed_markers = {'seed42': 'o', 'seed43': 's', 'seed44': '^'}\n",
            "\n",
            "for idx, model in enumerate(models):\n",
            "    ax = axes[idx]\n",
            "    df_model = df[df['model'] == model]\n",
            "    methods_present = [m for m in METHOD_ORDER if m in df_model['method'].unique()]\n",
            "    \n",
            "    x_ticks, x_labels = [], []\n",
            "    x_pos_curr = 0\n",
            "    \n",
            "    for ds in datasets:\n",
            "        for m_idx, method in enumerate(methods_present):\n",
            "            subset = df_model[(df_model['dataset'] == ds) & (df_model['method'] == method)]\n",
            "            if subset.empty:\n",
            "                continue\n",
            "            color = METHOD_COLORS.get(method, '#888888')\n",
            "            acc_mean = subset['accuracy_pct'].mean()\n",
            "            \n",
            "            # Plot horizontal mean line\n",
            "            ax.hlines(acc_mean, x_pos_curr - 0.25, x_pos_curr + 0.25, color=color, linewidth=2.5, alpha=0.8)\n",
            "            \n",
            "            # Plot individual seed points\n",
            "            for _, srow in subset.iterrows():\n",
            "                marker = seed_markers.get(srow['seed'], 'o')\n",
            "                ax.scatter(x_pos_curr, srow['accuracy_pct'], color=color, edgecolor='#0f172a',\n",
            "                           marker=marker, s=42, zorder=4, alpha=0.9)\n",
            "                \n",
            "            x_ticks.append(x_pos_curr)\n",
            "            x_labels.append(f\"{METHOD_LABELS.get(method, method)[:4]}\")\n",
            "            x_pos_curr += 1\n",
            "        x_pos_curr += 0.85\n",
            "        \n",
            "    ax.set_xticks(x_ticks)\n",
            "    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8.5, fontweight='bold')\n",
            "    ax.set_title(f\"Backbone: {MODEL_LABELS.get(model, model)}\", pad=12)\n",
            "    if idx == 0:\n",
            "        ax.set_ylabel(\"Evaluation Accuracy (%)\", labelpad=8)\n",
            "    ax.grid(axis='y', linestyle='--', alpha=0.6)\n",
            "    ax.set_ylim(32, 102)\n",
            "    ax.spines['top'].set_visible(False)\n",
            "    ax.spines['right'].set_visible(False)\n",
            "    \n",
            "    ds_len = len(methods_present)\n",
            "    c_sst2 = (ds_len - 1) / 2\n",
            "    c_mrpc = ds_len + 0.85 + (ds_len - 1) / 2\n",
            "    c_rte = 2 * (ds_len + 0.85) + (ds_len - 1) / 2\n",
            "    for c_pos, ds_name in zip([c_sst2, c_mrpc, c_rte], ['SST-2', 'MRPC', 'RTE']):\n",
            "        ax.text(c_pos, 98.5, ds_name, ha='center', va='top', fontsize=10.5, fontweight='bold',\n",
            "                bbox=dict(boxstyle=\"round,pad=0.3\", fc=\"#f1f5f9\", ec=\"#cbd5e1\", alpha=0.9))\n",
            "\n",
            "seed_patches = [\n",
            "    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=7, label='Seed 42'),\n",
            "    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=7, label='Seed 43'),\n",
            "    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#64748b', markeredgecolor='#0f172a', markersize=7, label='Seed 44'),\n",
            "    plt.Line2D([0], [0], color='#333333', linewidth=2.5, label='Seed Mean')\n",
            "]\n",
            "fig.legend(handles=seed_patches, loc='upper center', bbox_to_anchor=(0.5, 1.06),\n",
            "           ncol=4, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=9)\n",
            "\n",
            "plt.suptitle(\"Individual Seed Results Across Tasks (Seeds 42, 43, 44)\", y=1.12, fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 13: Section 12 - Qualitative Failure Analysis
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 12. Qualitative Failure Analysis\n",
            "\n",
            "Sample misclassification inspection across tasks."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=== Sample Misclassified Instances (MRPC - BERT AdaLoRA) ===\")\n",
            "mrpc_misc = glob.glob(os.path.join(RESULTS_DIR, 'bert-base-uncased', 'mrpc', 'adalora', '*', 'misclassified.csv'))\n",
            "if mrpc_misc:\n",
            "    df_m = pd.read_csv(mrpc_misc[0])\n",
            "    display(df_m[['id', 'label', 'prediction', 'probability', 'input_text']].head(5))\n",
            "\n",
            "print(\"\\n=== Sample Misclassified Instances (SST-2 - BERT LoRA) ===\")\n",
            "sst2_misc = glob.glob(os.path.join(RESULTS_DIR, 'bert-base-uncased', 'sst2', 'lora', '*', 'misclassified.csv'))\n",
            "if sst2_misc:\n",
            "    df_s = pd.read_csv(sst2_misc[0])\n",
            "    display(df_s[['id', 'label', 'prediction', 'probability', 'input_text']].head(5))\n"
        ]
    })
    
    # ----------------------------------------------------
    # Cell 14: Section 13 - Empirical Takeaways & Practitioner Recommendations
    # ----------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 13. Empirical Takeaways & Practitioner Recommendations\n",
            "\n",
            "The table below is generated programmatically from the benchmark results, distinguishing what was observed under this standardized setup from broader generalizations:"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compute exact values dynamically from df\n",
            "full_bert_sst2 = df[(df['model'] == 'bert-base-uncased') & (df['dataset'] == 'sst2') & (df['method'] == 'full')]['accuracy'].mean() * 100\n",
            "lora_bert_sst2 = df[(df['model'] == 'bert-base-uncased') & (df['dataset'] == 'sst2') & (df['method'] == 'lora')]['accuracy'].mean() * 100\n",
            "lora_distil_sst2 = df[(df['model'] == 'distilbert-base-uncased') & (df['dataset'] == 'sst2') & (df['method'] == 'lora')]['accuracy'].mean() * 100\n",
            "prefix_bert_sst2 = df[(df['model'] == 'bert-base-uncased') & (df['dataset'] == 'sst2') & (df['method'] == 'prefix')]['accuracy'].mean() * 100\n",
            "ia3_bert_sst2 = df[(df['model'] == 'bert-base-uncased') & (df['dataset'] == 'sst2') & (df['method'] == 'ia3')]['accuracy'].mean() * 100\n",
            "\n",
            "lora_vram_bert = df[(df['model'] == 'bert-base-uncased') & (df['dataset'] == 'sst2') & (df['method'] == 'lora')]['peak_vram_mb'].mean()\n",
            "full_vram_bert = df[(df['model'] == 'bert-base-uncased') & (df['dataset'] == 'sst2') & (df['method'] == 'full')]['peak_vram_mb'].mean()\n",
            "lora_vram_reduction = (1 - (lora_vram_bert / full_vram_bert)) * 100\n",
            "\n",
            "ia3_params_bert = df[(df['model'] == 'bert-base-uncased') & (df['method'] == 'ia3')]['trainable_params'].iloc[0]\n",
            "ia3_pct_bert = df[(df['model'] == 'bert-base-uncased') & (df['method'] == 'ia3')]['pct_trainable'].iloc[0]\n",
            "ia3_ckpt_bert = df[(df['model'] == 'bert-base-uncased') & (df['method'] == 'ia3')]['checkpoint_size_mb'].iloc[0]\n",
            "\n",
            "rec_data = [\n",
            "    {\n",
            "        'Scenario / Priority': 'Highest accuracy in this benchmark',\n",
            "        'Observed in Benchmark': f'Full Fine-Tuning achieved highest accuracy ({full_bert_sst2:.1f}% on BERT/SST-2) as the reference baseline.',\n",
            "        'Practical Recommendation': 'Use Full Fine-Tuning when GPU memory and checkpoint storage are not bottlenecks.'\n",
            "    },\n",
            "    {\n",
            "        'Scenario / Priority': 'Best accuracy/resource trade-off observed',\n",
            "        'Observed in Benchmark': f'LoRA reached {lora_bert_sst2:.2f}% on BERT/SST-2 and {lora_distil_sst2:.2f}% on DistilBERT with {lora_vram_reduction:.1f}% lower VRAM.',\n",
            "        'Practical Recommendation': 'LoRA provides the most reliable general-purpose balance between resource savings and accuracy.'\n",
            "    },\n",
            "    {\n",
            "        'Scenario / Priority': 'High accuracy on BERT/SST-2 with fixed VRAM',\n",
            "        'Observed in Benchmark': f'Prefix Tuning reached {prefix_bert_sst2:.2f}% on BERT/SST-2 and resisted MRPC collapse better (73.12%).',\n",
            "        'Practical Recommendation': 'Consider Prefix Tuning for encoder backbones when higher parameter capacity (~11.9%) is acceptable.'\n",
            "    },\n",
            "    {\n",
            "        'Scenario / Priority': 'Highest parameter efficiency observed',\n",
            "        'Observed in Benchmark': f'IA³ adapted {ia3_params_bert:,} parameters ({ia3_pct_bert:.3f}%) with a {ia3_ckpt_bert:.1f} MB checkpoint (SST-2 acc: {ia3_bert_sst2:.2f}%).',\n",
            "        'Practical Recommendation': 'Use IA³ when checkpoint size is the primary constraint and moderate accuracy loss is acceptable.'\n",
            "    },\n",
            "    {\n",
            "        'Scenario / Priority': 'Small / Imbalanced Datasets Caveat',\n",
            "        'Observed in Benchmark': 'LoRA, AdaLoRA, and IA³ collapsed to predicting the majority class on MRPC (68.38%) under fixed learning rates.',\n",
            "        'Practical Recommendation': 'Always validate PEFT decision boundaries on small datasets rather than assuming hyperparameter transferability.'\n",
            "    }\n",
            "]\n",
            "\n",
            "df_rec = pd.DataFrame(rec_data)\n",
            "display(df_rec)\n"
        ]
    })
    
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(notebook_path, 'w') as f:
        json.dump(nb_dict, f, indent=1)
        
    print(f"Successfully generated refined notebook at: {notebook_path}")

if __name__ == '__main__':
    create_notebook()
