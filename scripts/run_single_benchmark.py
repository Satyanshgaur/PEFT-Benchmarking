"""Single experiment CLI runner executing a specific (model, dataset, method, seed) benchmark."""

import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import shutil
import argparse
import random
import subprocess
import numpy as np
import torch
import transformers

from src.utils.metadata import create_run_metadata
from src.utils.sanity_checks import run_sanity_checks
from src.data.loader import load_and_preprocess_glue_dataset
from src.models.peft_factory import create_benchmark_model
from src.trainer.benchmark_trainer import run_benchmark_experiment


def make_json_serializable(obj):
    """Recursively convert sets, numpy types, and non-serializable objects into standard Python types."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, "value"):  # Handle Enums like TaskType
        return str(obj.value)
    return obj


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed across Python, NumPy, PyTorch, and Transformers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def update_manifest(results_root: str, experiment_entry: dict):
    """Update or append entry to results/manifest.json."""
    manifest_path = os.path.join(results_root, "manifest.json")
    manifest_data = []

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = []

    updated = False
    for i, entry in enumerate(manifest_data):
        if (
            entry.get("model") == experiment_entry["model"]
            and entry.get("dataset") == experiment_entry["dataset"]
            and entry.get("method") == experiment_entry["method"]
            and entry.get("seed") == experiment_entry["seed"]
        ):
            manifest_data[i] = experiment_entry
            updated = True
            break

    if not updated:
        manifest_data.append(experiment_entry)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(manifest_data), f, indent=2)


def push_to_github(commit_msg: str):
    """Automatically stage, commit, and push results to GitHub."""
    try:
        subprocess.run(["git", "add", "."], check=True)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            subprocess.run(["git", "push"], check=True)
            print(f"⬆️ Successfully pushed benchmark results to GitHub: '{commit_msg}'")
        else:
            print("ℹ️ No uncommitted changes detected for GitHub push.")
    except Exception as e:
        print(f"⚠️ Git push notice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run a single PEFT benchmark experiment.")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Base model identifier")
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "mrpc", "rte"], help="GLUE dataset name")
    parser.add_argument("--method", type=str, default="lora", choices=["full", "lora", "adalora", "prefix", "ia3"], help="Adaptation method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch mini-batch dry run for validation")
    parser.add_argument("--no-push", action="store_true", help="Skip automatic git push after benchmark")
    parser.add_argument("--global_config", type=str, default="config/global.yaml", help="Path to global.yaml")
    parser.add_argument("--method_config", type=str, default="config/method_configs.yaml", help="Path to method_configs.yaml")
    parser.add_argument("--results_root", type=str, default="results", help="Results root directory")

    args = parser.parse_args()

    # Load configuration files
    with open(args.global_config, "r", encoding="utf-8") as f:
        global_cfg = yaml.safe_load(f)

    with open(args.method_config, "r", encoding="utf-8") as f:
        method_cfg_all = yaml.safe_load(f)

    method_specific_cfg = method_cfg_all.get(args.method, {})

    if args.epochs is not None:
        global_cfg["training"]["epochs"] = args.epochs

    if args.dry_run:
        global_cfg["training"]["epochs"] = 1

    # Set random seed
    is_deterministic = global_cfg.get("deterministic", {}).get("seed_all", True)
    set_seed(args.seed, deterministic=is_deterministic)

    # Set up experiment output directory
    model_folder = args.model.replace("/", "_")
    output_dir = os.path.join(args.results_root, model_folder, args.dataset, args.method, f"seed{args.seed}")
    config_dir = os.path.join(output_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # Save exact per-experiment configs for self-contained reproducibility
    with open(os.path.join(config_dir, "global.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(global_cfg, f)

    with open(os.path.join(config_dir, "method.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(method_specific_cfg, f)

    # Load dataset & tokenizer
    max_len = global_cfg["training"].get("max_seq_length", 128)
    tokenized_dataset, tokenizer, dataset_meta, raw_val_texts = load_and_preprocess_glue_dataset(
        dataset_name=args.dataset,
        tokenizer_name_or_path=args.model,
        max_seq_length=max_len
    )

    if args.dry_run:
        tokenized_dataset["train"] = tokenized_dataset["train"].select(range(min(64, len(tokenized_dataset["train"]))))
        tokenized_dataset["validation"] = tokenized_dataset["validation"].select(range(min(32, len(tokenized_dataset["validation"]))))
        raw_val_texts = raw_val_texts[:len(tokenized_dataset["validation"])]

    # Create model & PEFT adapter
    num_labels = dataset_meta["num_labels"]
    model, adapter_config_dict, param_stats = create_benchmark_model(
        model_name_or_path=args.model,
        method_name=args.method,
        method_config=method_specific_cfg,
        num_labels=num_labels
    )

    # Build run metadata
    command_str = " ".join(sys.argv)
    run_metadata = create_run_metadata(
        seed=args.seed,
        command=command_str,
        tokenizer_name=tokenizer.name_or_path,
        tokenizer_revision=getattr(tokenizer, "tokenizer_file", "main") or "main",
        vocab_size=len(tokenizer),
        max_seq_length=max_len,
        dataset_info=dataset_meta
    )
    exp_id = run_metadata["experiment_id"]

    # Execute training & evaluation
    metrics_dict, predictions_df, misclassified_df, train_log_df = run_benchmark_experiment(
        model=model,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        global_config=global_cfg,
        method_name=args.method,
        dataset_name=args.dataset,
        output_dir=output_dir,
        raw_val_texts=raw_val_texts
    )
    metrics_dict["experiment_id"] = exp_id

    # Save run.json
    with open(os.path.join(output_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(run_metadata), f, indent=2)

    # Save metrics.json
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(metrics_dict), f, indent=2)

    # Save adapter_config.json if PEFT method
    if adapter_config_dict:
        with open(os.path.join(output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(adapter_config_dict), f, indent=2)

    # Save predictions.csv
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # Save misclassified.csv
    misclassified_df.to_csv(os.path.join(output_dir, "misclassified.csv"), index=False)

    # Save train_log.csv
    train_log_df.to_csv(os.path.join(output_dir, "train_log.csv"), index=False)

    # Execute sanity checks
    sanity_results = run_sanity_checks(
        method=args.method,
        total_params=param_stats["total_parameters"],
        trainable_params=param_stats["trainable_parameters"],
        peak_vram_mb=metrics_dict["efficiency_metrics"]["peak_vram_mb"],
        avg_vram_mb=metrics_dict["efficiency_metrics"]["avg_vram_mb"],
        official_metrics=metrics_dict["official_metrics"],
        predictions_df=predictions_df,
        expected_val_len=len(tokenized_dataset["validation"]),
        checkpoint_size_mb=metrics_dict["parameter_metrics"]["checkpoint_size_mb"]
    )

    # Update manifest.json
    manifest_entry = {
        "schema_version": "1.0.0",
        "experiment_id": exp_id,
        "status": "completed",
        "model": args.model,
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "official_metrics": metrics_dict["official_metrics"],
        "peak_vram_mb": metrics_dict["efficiency_metrics"]["peak_vram_mb"],
        "training_time_seconds": metrics_dict["efficiency_metrics"]["training_time_seconds"],
        "output_dir": output_dir
    }
    update_manifest(args.results_root, manifest_entry)

    print(f"✅ Experiment completed! UUID: {exp_id} | Output: {output_dir}")

    # Push results to GitHub unless --no-push specified
    if not args.no_push:
        commit_msg = f"benchmark: completed {args.model} {args.dataset} {args.method} seed{args.seed} [id:{exp_id}]"
        push_to_github(commit_msg)


if __name__ == "__main__":
    main()
