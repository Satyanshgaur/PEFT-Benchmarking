"""Orchestrator script for executing full benchmark matrix, per-backbone Pareto frontiers, and statistical tests."""

import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import argparse
import numpy as np
import scipy.stats as stats
import subprocess


def is_run_completed(results_root: str, model: str, dataset: str, method: str, seed: int) -> bool:
    """Check if experiment output folder contains completed metrics.json and run.json."""
    model_folder = model.replace("/", "_")
    run_dir = os.path.join(results_root, model_folder, dataset, method, f"seed{seed}")
    metrics_path = os.path.join(run_dir, "metrics.json")
    run_path = os.path.join(run_dir, "run.json")
    return os.path.exists(metrics_path) and os.path.exists(run_path)


def compute_per_backbone_pareto_frontiers(results_root: str) -> dict:
    """Compute Pareto-optimal methods per backbone model family."""
    manifest_path = os.path.join(results_root, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Group completed runs by (model, dataset)
    groups = {}
    for entry in manifest:
        if entry.get("status") != "completed":
            continue
        key = (entry["model"], entry["dataset"])
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)

    pareto_results = {
        "schema_version": "1.0.0",
        "description": "Per-backbone Pareto frontier analysis comparing Accuracy vs Peak VRAM and Training Time",
        "backbones": {}
    }

    for (model, dataset), entries in groups.items():
        method_stats = {}
        for entry in entries:
            m = entry["method"]
            if m not in method_stats:
                method_stats[m] = {"accs": [], "vrams": [], "times": []}
            acc = list(entry["official_metrics"].values())[0]
            method_stats[m]["accs"].append(acc)
            method_stats[m]["vrams"].append(entry["peak_vram_mb"])
            method_stats[m]["times"].append(entry["training_time_seconds"])

        agg = []
        for m, s in method_stats.items():
            mean_acc = float(np.mean(s["accs"]))
            mean_vram = float(np.mean(s["vrams"]))
            mean_time = float(np.mean(s["times"]))
            agg.append({
                "method": m,
                "mean_accuracy": mean_acc,
                "mean_peak_vram_mb": mean_vram,
                "mean_training_time_seconds": mean_time
            })

        for i, item_i in enumerate(agg):
            dominated = False
            for j, item_j in enumerate(agg):
                if i == j:
                    continue
                if (
                    item_j["mean_accuracy"] >= item_i["mean_accuracy"]
                    and item_j["mean_peak_vram_mb"] <= item_i["mean_peak_vram_mb"]
                    and item_j["mean_training_time_seconds"] <= item_i["mean_training_time_seconds"]
                    and (
                        item_j["mean_accuracy"] > item_i["mean_accuracy"]
                        or item_j["mean_peak_vram_mb"] < item_i["mean_peak_vram_mb"]
                        or item_j["mean_training_time_seconds"] < item_i["mean_training_time_seconds"]
                    )
                ):
                    dominated = True
                    break
            item_i["pareto_dominated"] = dominated
            item_i["pareto_optimal"] = not dominated

        group_key = f"{model}__{dataset}"
        pareto_results["backbones"][group_key] = agg

    with open(os.path.join(results_root, "pareto.json"), "w", encoding="utf-8") as f:
        json.dump(pareto_results, f, indent=2)

    return pareto_results


def compute_statistical_tests(results_root: str) -> dict:
    """Compute effect sizes, 95% CIs, and exploratory hypothesis tests vs Full Fine-Tuning."""
    manifest_path = os.path.join(results_root, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    groups = {}
    for entry in manifest:
        if entry.get("status") != "completed":
            continue
        key = (entry["model"], entry["dataset"], entry["method"])
        if key not in groups:
            groups[key] = []
        acc = list(entry["official_metrics"].values())[0]
        groups[key].append((entry["seed"], acc))

    stats_results = {
        "schema_version": "1.0.0",
        "methodology_note": (
            "With 3 random seeds, hypothesis tests are reported as exploratory diagnostics. "
            "Primary statistical evaluation emphasizes effect sizes (Cohen's d) and 95% confidence intervals."
        ),
        "comparisons": {}
    }

    model_datasets = set((k[0], k[1]) for k in groups.keys())

    for model, dataset in model_datasets:
        full_key = (model, dataset, "full")
        if full_key not in groups:
            continue
        
        full_runs = sorted(groups[full_key], key=lambda x: x[0])
        full_accs = np.array([x[1] for x in full_runs])

        pair_key = f"{model}__{dataset}"
        stats_results["comparisons"][pair_key] = {}

        methods = set(k[2] for k in groups.keys() if k[0] == model and k[1] == dataset)

        for m in methods:
            m_key = (model, dataset, m)
            m_runs = sorted(groups[m_key], key=lambda x: x[0])
            m_accs = np.array([x[1] for x in m_runs])

            mean_acc = float(np.mean(m_accs))
            std_acc = float(np.std(m_accs, ddof=1)) if len(m_accs) > 1 else 0.0
            sem = std_acc / np.sqrt(len(m_accs)) if len(m_accs) > 0 else 0.0
            ci95 = float(sem * 1.96)

            comp = {
                "n_seeds": len(m_accs),
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "ci95_accuracy": ci95,
            }

            if m != "full" and len(m_accs) == len(full_accs) and len(m_accs) >= 2:
                diff = m_accs - full_accs
                mean_diff = np.mean(diff)
                std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 1e-6
                cohens_d = float(mean_diff / (std_diff + 1e-8))

                try:
                    ttest_res = stats.ttest_rel(m_accs, full_accs)
                    p_ttest = float(ttest_res.pvalue)
                except Exception:
                    p_ttest = None

                try:
                    wilc_res = stats.wilcoxon(diff)
                    p_wilcoxon = float(wilc_res.pvalue)
                except Exception:
                    p_wilcoxon = None

                comp["vs_full_fine_tuning"] = {
                    "mean_difference": float(mean_diff),
                    "cohens_d": cohens_d,
                    "exploratory_paired_ttest_pvalue": p_ttest,
                    "exploratory_wilcoxon_pvalue": p_wilcoxon
                }

            stats_results["comparisons"][pair_key][m] = comp

    with open(os.path.join(results_root, "statistical_tests.json"), "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=2)

    return stats_results


def main():
    parser = argparse.ArgumentParser(description="Orchestrate full PEFT benchmark matrix execution.")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-epoch dry run for full matrix validation")
    parser.add_argument("--force", action="store_true", help="Re-run existing completed benchmark runs")
    parser.add_argument("--global_config", type=str, default="config/global.yaml", help="Path to global.yaml")
    parser.add_argument("--results_root", type=str, default="results", help="Results root directory")

    args = parser.parse_args()

    with open(args.global_config, "r", encoding="utf-8") as f:
        global_cfg = yaml.safe_load(f)

    models = ["distilbert-base-uncased", "bert-base-uncased"]
    datasets = ["sst2", "mrpc", "rte"]
    methods = ["full", "lora", "adalora", "prefix", "ia3"]
    seeds = global_cfg.get("seeds", [42, 43, 44])

    total_experiments = len(models) * len(datasets) * len(methods) * len(seeds)
    print(f"🚀 Starting PEFT benchmark matrix: {total_experiments} total runs ({len(models)} models x {len(datasets)} datasets x {len(methods)} methods x {len(seeds)} seeds)")

    completed_count = 0

    for model in models:
        for dataset in datasets:
            for method in methods:
                for seed in seeds:
                    completed_count += 1
                    if not args.force and is_run_completed(args.results_root, model, dataset, method, seed):
                        print(f"⏩ [{completed_count}/{total_experiments}] Skipping existing run: {model} | {dataset} | {method} | seed {seed}")
                        continue

                    print(f"\n▶️ [{completed_count}/{total_experiments}] Executing: model={model}, dataset={dataset}, method={method}, seed={seed}")
                    
                    cmd = [
                        sys.executable, "scripts/run_single_benchmark.py",
                        "--model", model,
                        "--dataset", dataset,
                        "--method", method,
                        "--seed", str(seed),
                        "--results_root", args.results_root
                    ]
                    if args.dry_run:
                        cmd.append("--dry-run")

                    res = subprocess.run(cmd)
                    if res.returncode != 0:
                        print(f"❌ Error executing run {model}/{dataset}/{method}/seed{seed}")

    # Aggregate post-run results
    print("\n📊 Computing per-backbone Pareto frontiers...")
    compute_per_backbone_pareto_frontiers(args.results_root)

    print("📈 Computing multi-seed statistical summaries and effect sizes...")
    compute_statistical_tests(args.results_root)

    print("\n🎉 Full matrix benchmark execution complete! Multi-run summary saved to results/pareto.json and results/statistical_tests.json")


if __name__ == "__main__":
    main()
