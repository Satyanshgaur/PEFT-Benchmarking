"""Analytical FLOPs estimation module for transformer fine-tuning benchmark."""

from typing import Dict, Any


def estimate_analytical_flops(
    total_params: int,
    trainable_params: int,
    seq_length: int,
    num_samples: int,
    num_epochs: int,
    is_full_ft: bool = False
) -> Dict[str, Any]:
    """Calculate approximate analytical training FLOPs estimate.

    Methodology:
    - Standard forward pass requires ~2 FLOPs per parameter per token (1 multiply + 1 add).
    - Backward pass for active gradient propagation requires:
        - 2 FLOPs per parameter for gradient w.r.t inputs/activations (for all layers back to target)
        - 2 FLOPs per trainable parameter for gradient w.r.t parameters
    - For Full Fine-Tuning: 6 * N_total * tokens (2 for forward, 4 for backward).
    - For PEFT: ~4 * N_total * tokens (forward + input grad backprop) + 2 * N_trainable * tokens (param grad).

    Args:
        total_params: Total parameters in the model.
        trainable_params: Number of updated trainable parameters.
        seq_length: Average sequence length (tokens per sample).
        num_samples: Total number of training samples per epoch.
        num_epochs: Number of training epochs.
        is_full_ft: Whether method is Full Fine-Tuning.

    Returns:
        Dictionary containing total analytical FLOPs estimate and per-sample FLOPs.
    """
    tokens_per_pass = seq_length * num_samples * num_epochs

    if is_full_ft:
        flops_per_token = 6 * total_params
    else:
        flops_per_token = (4 * total_params) + (2 * trainable_params)

    total_flops = flops_per_token * tokens_per_pass
    flops_per_sample = flops_per_token * seq_length

    return {
        "approximate_analytical_flops_estimate": float(total_flops),
        "flops_per_sample": float(flops_per_sample),
        "total_tokens_processed": int(tokens_per_pass),
        "estimation_methodology": (
            "Analytical estimate based on 2 FLOPs/param/token for forward pass and "
            "gradient backpropagation FLOPs weighted by trainable vs frozen weights."
        )
    }
