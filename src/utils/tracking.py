"""Hardware resource and execution time tracking module (VRAM, CPU RAM, GPU/CPU %, latency)."""

import time
import threading
from typing import Dict, Any, List

import torch
import psutil

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


class ResourceTracker:
    """Monitors peak/avg GPU VRAM, CPU RAM, GPU/CPU utilization %, and execution timing."""

    def __init__(self, polling_interval_ms: int = 500):
        self.polling_interval_sec = polling_interval_ms / 1000.0
        self.keep_polling = False
        self.poll_thread: threading.Thread = None

        self.vram_samples_mb: List[float] = []
        self.ram_samples_mb: List[float] = []
        self.gpu_util_samples_pct: List[float] = []
        self.cpu_util_samples_pct: List[float] = []

        self.start_time: float = 0.0
        self.end_time: float = 0.0

        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.nvml_handle = None
        else:
            self.nvml_handle = None

        self.process = psutil.Process()

    def _poll_loop(self):
        """Background sampling loop running every polling_interval_sec."""
        while self.keep_polling:
            # CPU RAM
            try:
                ram_mb = self.process.memory_info().rss / (1024 * 1024)
                self.ram_samples_mb.append(ram_mb)
            except Exception:
                pass

            # CPU Utilization %
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                self.cpu_util_samples_pct.append(cpu_pct)
            except Exception:
                pass

            # GPU VRAM & Utilization via NVML or PyTorch
            if torch.cuda.is_available():
                try:
                    vram_bytes = torch.cuda.memory_allocated(0)
                    self.vram_samples_mb.append(vram_bytes / (1024 * 1024))
                except Exception:
                    pass

                if self.nvml_handle is not None:
                    try:
                        rates = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                        self.gpu_util_samples_pct.append(float(rates.gpu))
                    except Exception:
                        pass

            time.sleep(self.polling_interval_sec)

    def start(self):
        """Reset PyTorch memory stats and start background tracker."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.vram_samples_mb.clear()
        self.ram_samples_mb.clear()
        self.gpu_util_samples_pct.clear()
        self.cpu_util_samples_pct.clear()

        self.keep_polling = True
        self.start_time = time.perf_counter()
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop background tracker and summarize hardware metrics."""
        self.end_time = time.perf_counter()
        self.keep_polling = False
        if self.poll_thread is not None and self.poll_thread.is_alive():
            self.poll_thread.join(timeout=1.0)

        elapsed_sec = self.end_time - self.start_time

        # Peak VRAM
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
        elif self.vram_samples_mb:
            peak_vram_mb = max(self.vram_samples_mb)
        else:
            peak_vram_mb = 0.0

        avg_vram_mb = (sum(self.vram_samples_mb) / len(self.vram_samples_mb)) if self.vram_samples_mb else peak_vram_mb
        avg_ram_mb = (sum(self.ram_samples_mb) / len(self.ram_samples_mb)) if self.ram_samples_mb else 0.0
        avg_gpu_util = (sum(self.gpu_util_samples_pct) / len(self.gpu_util_samples_pct)) if self.gpu_util_samples_pct else 0.0
        avg_cpu_util = (sum(self.cpu_util_samples_pct) / len(self.cpu_util_samples_pct)) if self.cpu_util_samples_pct else 0.0

        return {
            "elapsed_seconds": float(elapsed_sec),
            "peak_vram_mb": float(peak_vram_mb),
            "avg_vram_mb": float(avg_vram_mb),
            "avg_cpu_ram_mb": float(avg_ram_mb),
            "gpu_utilization_pct": float(avg_gpu_util),
            "cpu_utilization_pct": float(avg_cpu_util),
            "polling_interval_ms": int(self.polling_interval_sec * 1000),
            "measurement_backend": "nvml_psutil" if HAS_PYNVML else "pytorch_psutil"
        }
