"""Unified model initialization factory for Full Fine-Tuning, LoRA, AdaLoRA, Prefix Tuning, and IA3."""

from typing import Dict, Any, Tuple
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from peft import (
    get_peft_model,
    LoraConfig,
    AdaLoraConfig,
    PrefixTuningConfig,
    IA3Config,
    TaskType
)


def resolve_target_modules(model: nn.Module, method_name: str, requested_modules: list) -> list:
    """Resolve model-specific layer names for BERT vs DistilBERT."""
    model_type = getattr(model.config, "model_type", "").lower()
    
    if method_name in ["lora", "adalora"]:
        if "distilbert" in model_type:
            return ["q_lin", "v_lin"]
        else:
            return ["query", "value"]
            
    elif method_name == "ia3":
        if "distilbert" in model_type:
            return ["k_lin", "v_lin", "lin2"]
        else:
            return ["key", "value", "dense"]
            
    return requested_modules


def create_benchmark_model(
    model_name_or_path: str,
    method_name: str,
    method_config: Dict[str, Any],
    num_labels: int = 2
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Initialize sequence classification model and apply PEFT adapter if specified.

    Args:
        model_name_or_path: Hugging Face model identifier (e.g. 'bert-base-uncased').
        method_name: One of 'full', 'lora', 'adalora', 'prefix', 'ia3'.
        method_config: Hyperparameter dictionary for the specified method.
        num_labels: Number of classification target classes.

    Returns:
        Tuple of (model, adapter_config_dict, parameter_statistics_dict).
    """
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels
    )

    adapter_config_dict: Dict[str, Any] = {}

    if method_name == "full":
        model = base_model
        for param in model.parameters():
            param.requires_grad = True

    elif method_name == "lora":
        target_mods = resolve_target_modules(base_model, "lora", method_config.get("target_modules", []))
        peft_config = LoraConfig(
            r=method_config.get("r", 8),
            lora_alpha=method_config.get("lora_alpha", 16),
            lora_dropout=method_config.get("lora_dropout", 0.1),
            target_modules=target_mods,
            bias=method_config.get("bias", "none"),
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(base_model, peft_config)
        adapter_config_dict = peft_config.to_dict()

    elif method_name == "adalora":
        target_mods = resolve_target_modules(base_model, "adalora", method_config.get("target_modules", []))
        peft_config = AdaLoraConfig(
            init_r=method_config.get("init_r", 12),
            target_r=method_config.get("target_r", 8),
            lora_alpha=method_config.get("lora_alpha", 16),
            lora_dropout=method_config.get("lora_dropout", 0.1),
            target_modules=target_mods,
            total_step=method_config.get("total_step", 10000),
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(base_model, peft_config)
        adapter_config_dict = peft_config.to_dict()

    elif method_name == "prefix":
        hidden_size = getattr(base_model.config, "hidden_size", getattr(base_model.config, "dim", 768))
        num_layers = getattr(base_model.config, "num_hidden_layers", getattr(base_model.config, "n_layers", 12))
        num_heads = getattr(base_model.config, "num_attention_heads", getattr(base_model.config, "n_heads", 12))
        peft_config = PrefixTuningConfig(
            num_virtual_tokens=method_config.get("num_virtual_tokens", 20),
            prefix_projection=method_config.get("prefix_projection", True),
            encoder_hidden_size=hidden_size,
            token_dim=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(base_model, peft_config)
        adapter_config_dict = peft_config.to_dict()

    elif method_name == "ia3":
        target_mods = resolve_target_modules(base_model, "ia3", method_config.get("target_modules", []))
        model_type = getattr(base_model.config, "model_type", "").lower()
        ffn_mods = ["lin2"] if "distilbert" in model_type else ["dense"]
        
        peft_config = IA3Config(
            target_modules=target_mods,
            feedforward_modules=ffn_mods,
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(base_model, peft_config)
        adapter_config_dict = peft_config.to_dict()

    else:
        raise ValueError(f"Unsupported adaptation method: '{method_name}'")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct_trainable = (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0

    param_stats = {
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "pct_trainable": float(pct_trainable)
    }

    return model, adapter_config_dict, param_stats
