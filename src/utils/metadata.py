"""System, hardware, runtime, and tokenizer metadata logger for run.json."""

import sys
import uuid
import datetime
import subprocess
from typing import Dict, Any, Optional

import torch
import transformers


def get_git_commit() -> str:
    """Retrieve current git commit hash, or fallback to 'unknown'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def get_driver_version() -> str:
    """Retrieve NVIDIA driver version if available."""
    try:
        import pynvml
        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
        return version
    except Exception:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            return output
        except Exception:
            return "N/A (CPU or Driver query failed)"


def create_run_metadata(
    seed: int,
    command: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    vocab_size: int,
    max_seq_length: int,
    dataset_info: Dict[str, Any],
    experiment_id: Optional[str] = None
) -> Dict[str, Any]:
    """Build complete run.json dictionary conforming to schema v1.0.0."""
    exp_id = experiment_id or str(uuid.uuid4())[:8]
    
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"
    
    metadata = {
        "schema_version": "1.0.0",
        "experiment_id": exp_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command,
        "seed": seed,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": cuda_ver,
        "gpu_name": gpu_name,
        "driver_version": get_driver_version(),
        "tokenizer": {
            "name": tokenizer_name,
            "revision": tokenizer_revision,
            "vocab_size": vocab_size,
            "max_seq_length": max_seq_length
        },
        "dataset": dataset_info,
        "git_commit": get_git_commit()
    }
    
    return metadata
