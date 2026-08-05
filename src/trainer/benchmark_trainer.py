"""Benchmark trainer orchestrating fine-tuning, loss logging, inference latency profiling, and resource monitoring."""

import os
import time
import shutil
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)

from src.utils.tracking import ResourceTracker
from src.utils.metrics import evaluate_predictions
from src.utils.flops import estimate_analytical_flops


class MetricsLoggerCallback(TrainerCallback):
    """Callback to record step/epoch training loss, validation loss, and learning rate for train_log.csv."""

    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.start_time = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        elapsed = time.perf_counter() - self.start_time
        entry = {
            "epoch": float(logs.get("epoch", state.epoch or 0.0)),
            "step": int(state.global_step),
            "loss": float(logs.get("loss", 0.0)) if "loss" in logs else None,
            "eval_loss": float(logs.get("eval_loss", 0.0)) if "eval_loss" in logs else None,
            "learning_rate": float(logs.get("learning_rate", 0.0)) if "learning_rate" in logs else None,
            "elapsed_seconds": float(elapsed)
        }
        self.logs.append(entry)


def get_directory_size_mb(directory_path: str) -> float:
    """Compute total disk size of a directory in Megabytes."""
    if not os.path.exists(directory_path):
        return 0.0
    total_bytes = 0
    for root, _, files in os.walk(directory_path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return float(total_bytes / (1024 * 1024))


def run_benchmark_experiment(
    model: torch.nn.Module,
    tokenizer: Any,
    tokenized_dataset: Any,
    global_config: Dict[str, Any],
    method_name: str,
    dataset_name: str,
    output_dir: str,
    raw_val_texts: List[str] = None
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute training, resource profiling, evaluation, failure analysis, and checkpoint saving.

    Args:
        model: PyTorch model or PEFT model.
        tokenizer: Pretrained tokenizer.
        tokenized_dataset: HuggingFace dataset containing 'train' and 'validation'.
        global_config: Global config dict from global.yaml.
        method_name: One of 'full', 'lora', 'adalora', 'prefix', 'ia3'.
        dataset_name: One of 'sst2', 'mrpc', 'rte'.
        output_dir: Target experiment output directory (results/<model>/<dataset>/<method>/seed<seed>/).
        raw_val_texts: Raw text strings for failure analysis.

    Returns:
        Tuple of (metrics_dict, predictions_df, misclassified_df, train_log_df).
    """
    train_cfg = global_config["training"]
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = float(train_cfg["learning_rate"])
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    polling_interval_ms = global_config.get("hardware_tracking", {}).get("polling_interval_ms", 500)

    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none",
        fp16=train_cfg.get("fp16", False) and torch.cuda.is_available(),
        disable_tqdm=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metrics_callback = MetricsLoggerCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[metrics_callback]
    )

    # Initialize resource tracker
    resource_tracker = ResourceTracker(polling_interval_ms=polling_interval_ms)

    # --- 1. TRAIN STAGE ---
    resource_tracker.start()
    train_result = trainer.train()
    training_hardware_stats = resource_tracker.stop()

    training_time_sec = train_result.metrics.get("train_runtime", training_hardware_stats["elapsed_seconds"])
    train_samples_per_sec = train_result.metrics.get("train_samples_per_second", 0.0)
    train_steps_per_sec = train_result.metrics.get("train_steps_per_second", 0.0)

    # --- 2. INFERENCE LATENCY & EVALUATION STAGE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    eval_dataloader = trainer.get_eval_dataloader()
    all_logits = []
    all_labels = []

    # Warmup pass for accurate latency measurement
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
            if i >= 2:
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_start_time = time.perf_counter()
    with torch.no_grad():
        for batch in eval_dataloader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_end_time = time.perf_counter()

    infer_total_sec = infer_end_time - infer_start_time
    total_val_samples = len(tokenized_dataset["validation"])
    latency_ms_per_sample = (infer_total_sec * 1000.0) / total_val_samples if total_val_samples > 0 else 0.0
    throughput_samples_per_sec = total_val_samples / infer_total_sec if infer_total_sec > 0 else 0.0

    logits_matrix = np.concatenate(all_logits, axis=0)
    labels_vector = np.concatenate(all_labels, axis=0)

    # Compute official, diagnostic, calibration, and failure analysis metrics
    metrics_dict, predictions_df, misclassified_df = evaluate_predictions(
        logits=logits_matrix,
        labels=labels_vector,
        dataset_name=dataset_name,
        raw_texts=raw_val_texts
    )

    # --- 3. PARAMETER & FLOPs ESTIMATION ---
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct_trainable = (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0

    seq_len = train_cfg.get("max_seq_length", 128)
    num_train_samples = len(tokenized_dataset["train"])
    flops_info = estimate_analytical_flops(
        total_params=total_params,
        trainable_params=trainable_params,
        seq_length=seq_len,
        num_samples=num_train_samples,
        num_epochs=epochs,
        is_full_ft=(method_name == "full")
    )

    # --- 4. CHECKPOINT SIZE ---
    trainer.save_model(checkpoint_dir)
    checkpoint_size_mb = get_directory_size_mb(checkpoint_dir)

    # Assemble full efficiency metrics
    efficiency_metrics = {
        "training_time_seconds": float(training_time_sec),
        "train_samples_per_second": float(train_samples_per_sec),
        "train_steps_per_second": float(train_steps_per_sec),
        "inference_latency_ms_per_sample": float(latency_ms_per_sample),
        "inference_throughput_samples_per_second": float(throughput_samples_per_sec),
        "peak_vram_mb": float(training_hardware_stats["peak_vram_mb"]),
        "avg_vram_mb": float(training_hardware_stats["avg_vram_mb"]),
        "cpu_ram_mb": float(training_hardware_stats["avg_cpu_ram_mb"]),
        "gpu_utilization_pct": float(training_hardware_stats["gpu_utilization_pct"]),
        "cpu_utilization_pct": float(training_hardware_stats["cpu_utilization_pct"]),
        "polling_interval_ms": int(training_hardware_stats["polling_interval_ms"]),
        "measurement_backend": training_hardware_stats["measurement_backend"],
        "approximate_analytical_flops_estimate": float(flops_info["approximate_analytical_flops_estimate"]),
        "flops_per_sample": float(flops_info["flops_per_sample"]),
        "estimation_methodology": flops_info["estimation_methodology"]
    }

    parameter_metrics = {
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "pct_trainable": float(pct_trainable),
        "checkpoint_size_mb": float(checkpoint_size_mb)
    }

    metrics_dict["schema_version"] = "1.0.0"
    metrics_dict["efficiency_metrics"] = efficiency_metrics
    metrics_dict["parameter_metrics"] = parameter_metrics

    # Train log dataframe
    train_log_df = pd.DataFrame(metrics_callback.logs)

    return metrics_dict, predictions_df, misclassified_df, train_log_df
