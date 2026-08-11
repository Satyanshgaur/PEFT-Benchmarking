"""Automated sanity checks and assertions to detect benchmark implementation bugs early."""

from typing import Dict, Any
import pandas as pd


def run_sanity_checks(
    method: str,
    total_params: int,
    trainable_params: int,
    peak_vram_mb: float,
    avg_vram_mb: float,
    official_metrics: Dict[str, float],
    predictions_df: pd.DataFrame,
    expected_val_len: int,
    checkpoint_size_mb: float = 0.0
) -> Dict[str, Any]:
    """Execute rigorous assertion suite on experiment outputs.

    Raises AssertionError if any sanity check fails.
    Returns dictionary summarizing sanity check statuses.
    """
    pct_trainable = (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0

    # 1. Parameter sanity checks
    if method == "full":
        assert trainable_params == total_params, (
            f"Full fine-tuning trainable parameters ({trainable_params}) must equal total parameters ({total_params})."
        )
    else:
        assert pct_trainable < 15.0, (
            f"PEFT method '{method}' parameter percentage ({pct_trainable:.2f}%) exceeds expected threshold (< 15.0%)."
        )

    # 2. Memory sanity check
    assert peak_vram_mb >= avg_vram_mb, (
        f"Peak GPU memory ({peak_vram_mb:.2f} MB) cannot be less than average memory ({avg_vram_mb:.2f} MB)."
    )

    # 3. Predictive metric range check
    for metric_name, val in official_metrics.items():
        assert 0.0 <= val <= 1.0, (
            f"Metric '{metric_name}' value ({val}) lies outside valid probability range [0.0, 1.0]."
        )

    # 4. Predictions dataframe count check
    assert len(predictions_df) == expected_val_len, (
        f"Predictions count ({len(predictions_df)}) does not match validation dataset length ({expected_val_len})."
    )

    return {
        "sanity_checks_passed": True,
        "method": method,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "pct_trainable": float(pct_trainable),
        "predictions_validated": len(predictions_df)
    }
