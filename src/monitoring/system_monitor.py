"""Live system resource monitoring for LocalRAG.

Collects real-time hardware telemetry:
- CPU utilization percentage via psutil
- System RAM used, total, and percentage via psutil
- GPU utilization percentage, VRAM used/total/percentage, and temperature via nvidia-smi

Designed to be lightweight, non-blocking, and fault-tolerant.
Does not launch secondary inference processes or disrupt active models.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

import psutil

logger = logging.getLogger(__name__)


def get_cpu_metrics() -> dict[str, Any]:
    """
    Get current system CPU utilization.
    
    Uses interval=None for non-blocking instantaneous measurement since last call.
    """
    try:
        pct = psutil.cpu_percent(interval=None)
        return {
            "utilization_pct": float(pct),
            "available": True,
        }
    except Exception as exc:
        logger.warning("Failed to collect CPU metrics: %s", exc)
        return {
            "utilization_pct": 0.0,
            "available": False,
            "error": str(exc),
        }


def get_ram_metrics() -> dict[str, Any]:
    """
    Get system virtual memory (RAM) usage.
    """
    try:
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        return {
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "utilization_pct": round(float(mem.percent), 1),
            "available": True,
        }
    except Exception as exc:
        logger.warning("Failed to collect RAM metrics: %s", exc)
        return {
            "used_gb": 0.0,
            "total_gb": 0.0,
            "utilization_pct": 0.0,
            "available": False,
            "error": str(exc),
        }


def parse_nvidia_smi_output(raw_output: str) -> list[dict[str, Any]]:
    """
    Parse CSV output from nvidia-smi query command.
    
    Expected format:
        index, name, utilization.gpu, memory.used, memory.total, temperature.gpu
    Example:
        0, NVIDIA GeForce RTX 4050 Laptop GPU, 5, 2442, 6141, 45
    """
    gpus: list[dict[str, Any]] = []
    if not raw_output or not raw_output.strip():
        return gpus

    for line in raw_output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue

        try:
            gpu_idx = int(parts[0])
        except (ValueError, TypeError):
            continue

        name = parts[1]

        # utilization.gpu (percentage or N/A)
        try:
            gpu_util = float(parts[2])
        except (ValueError, TypeError):
            gpu_util = None

        # memory.used (MB)
        try:
            vram_used_mb = float(parts[3])
        except (ValueError, TypeError):
            vram_used_mb = 0.0

        # memory.total (MB)
        try:
            vram_total_mb = float(parts[4])
        except (ValueError, TypeError):
            vram_total_mb = 0.0

        vram_pct = (
            round((vram_used_mb / vram_total_mb) * 100.0, 1)
            if vram_total_mb > 0
            else 0.0
        )

        # temperature.gpu (Celsius or N/A)
        try:
            temp_c = float(parts[5])
        except (ValueError, TypeError):
            temp_c = None

        vram_used_gb = round(vram_used_mb / 1024.0, 2)
        vram_total_gb = round(vram_total_mb / 1024.0, 2)

        gpus.append({
            "index": gpu_idx,
            "name": name,
            "utilization_pct": gpu_util,
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
            "vram_used_gb": vram_used_gb,
            "vram_total_gb": vram_total_gb,
            "vram_pct": vram_pct,
            "temperature_c": temp_c,
            "available": True,
        })

    return gpus


def get_gpu_metrics(timeout_seconds: float = 1.5) -> dict[str, Any] | None:
    """
    Retrieve live NVIDIA GPU telemetry using nvidia-smi.
    
    Robust against missing binaries, driver failures, timeouts, or non-NVIDIA hosts.
    Never throws an unhandled exception.
    """
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None

    try:
        proc = subprocess.run(
            [
                binary,
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if proc.returncode != 0:
            logger.debug("nvidia-smi returned non-zero code %d: %s", proc.returncode, proc.stderr)
            return None

        gpus = parse_nvidia_smi_output(proc.stdout)
        if not gpus:
            return None

        # Return primary GPU (index 0 or first reported)
        return gpus[0]

    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi execution timed out after %.1fs", timeout_seconds)
        return None
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("Failed to query GPU via nvidia-smi: %s", exc)
        return None


def get_system_snapshot() -> dict[str, Any]:
    """
    Collect a unified system hardware snapshot.
    """
    return {
        "cpu": get_cpu_metrics(),
        "ram": get_ram_metrics(),
        "gpu": get_gpu_metrics(),
    }


def format_metrics_for_display(
    snapshot: dict[str, Any] | None = None,
) -> tuple[str, str, str, str, str]:
    """
    Format system telemetry snapshot into user-friendly dashboard display strings:
    
    Returns:
        (cpu_str, ram_str, gpu_util_str, gpu_vram_str, gpu_temp_str)
    """
    if snapshot is None:
        snapshot = get_system_snapshot()

    cpu = snapshot.get("cpu", {})
    ram = snapshot.get("ram", {})
    gpu = snapshot.get("gpu")

    # 1. CPU utilization
    if cpu.get("available", False):
        cpu_str = f"{cpu['utilization_pct']:.1f}%"
    else:
        cpu_str = "N/A"

    # 2. System RAM
    if ram.get("available", False):
        ram_str = f"{ram['used_gb']:.1f} / {ram['total_gb']:.1f} GB ({ram['utilization_pct']:.1f}%)"
    else:
        ram_str = "N/A"

    # 3. GPU Metrics
    if gpu and gpu.get("available", False):
        # GPU Utilization
        util = gpu.get("utilization_pct")
        gpu_name = gpu.get("name", "GPU")
        # Shorten name if lengthy, e.g. "RTX 4050 Laptop"
        short_name = gpu_name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        if util is not None:
            gpu_util_str = f"{util:.0f}% ({short_name})"
        else:
            gpu_util_str = f"N/A ({short_name})"

        # GPU VRAM
        vram_used = gpu.get("vram_used_gb", 0.0)
        vram_total = gpu.get("vram_total_gb", 0.0)
        vram_pct = gpu.get("vram_pct", 0.0)
        gpu_vram_str = f"{vram_used:.1f} / {vram_total:.1f} GB ({vram_pct:.1f}%)"

        # GPU Temperature
        temp = gpu.get("temperature_c")
        if temp is not None:
            gpu_temp_str = f"{temp:.0f} °C"
        else:
            gpu_temp_str = "N/A"
    else:
        gpu_util_str = "Unavailable"
        gpu_vram_str = "Unavailable"
        gpu_temp_str = "N/A"

    return cpu_str, ram_str, gpu_util_str, gpu_vram_str, gpu_temp_str

