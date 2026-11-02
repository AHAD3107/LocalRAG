"""Unit tests for the live system resource monitor module.

Tests CPU, RAM, and GPU telemetry extraction, robust parsing of nvidia-smi outputs,
graceful handling of missing binaries and timeouts, and display formatting.
Does not require a physical GPU or live hardware dependencies.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.system_monitor import (
    format_metrics_for_display,
    get_cpu_metrics,
    get_gpu_metrics,
    get_ram_metrics,
    get_system_snapshot,
    parse_nvidia_smi_output,
)


def test_get_cpu_metrics():
    metrics = get_cpu_metrics()
    assert isinstance(metrics, dict)
    assert metrics["available"] is True
    assert isinstance(metrics["utilization_pct"], float)
    assert 0.0 <= metrics["utilization_pct"] <= 100.0


def test_get_ram_metrics():
    metrics = get_ram_metrics()
    assert isinstance(metrics, dict)
    assert metrics["available"] is True
    assert metrics["used_gb"] > 0
    assert metrics["total_gb"] > 0
    assert metrics["used_gb"] <= metrics["total_gb"]
    assert 0.0 <= metrics["utilization_pct"] <= 100.0


def test_parse_nvidia_smi_output_standard():
    raw = "0, NVIDIA GeForce RTX 4050 Laptop GPU, 12, 2442, 6141, 46\n"
    gpus = parse_nvidia_smi_output(raw)
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu["index"] == 0
    assert gpu["name"] == "NVIDIA GeForce RTX 4050 Laptop GPU"
    assert gpu["utilization_pct"] == 12.0
    assert gpu["vram_used_mb"] == 2442.0
    assert gpu["vram_total_mb"] == 6141.0
    assert gpu["vram_used_gb"] == 2.38
    assert gpu["vram_total_gb"] == 6.0
    assert gpu["vram_pct"] == 39.8
    assert gpu["temperature_c"] == 46.0
    assert gpu["available"] is True


def test_parse_nvidia_smi_output_multiple_gpus():
    raw = (
        "0, NVIDIA RTX 4050 Laptop GPU, 5, 1024, 6144, 42\n"
        "1, NVIDIA RTX 4090, 85, 18432, 24576, 68\n"
    )
    gpus = parse_nvidia_smi_output(raw)
    assert len(gpus) == 2
    assert gpus[0]["index"] == 0
    assert gpus[0]["name"] == "NVIDIA RTX 4050 Laptop GPU"
    assert gpus[0]["utilization_pct"] == 5.0
    assert gpus[1]["index"] == 1
    assert gpus[1]["name"] == "NVIDIA RTX 4090"
    assert gpus[1]["utilization_pct"] == 85.0
    assert gpus[1]["temperature_c"] == 68.0


def test_parse_nvidia_smi_output_malformed_and_missing_values():
    raw = (
        "\n"
        "not_a_valid_line\n"
        "0, GPU Without Numbers, [Not Supported], N/A, 4096, [Unknown]\n"
        "short, line, here\n"
    )
    gpus = parse_nvidia_smi_output(raw)
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu["index"] == 0
    assert gpu["name"] == "GPU Without Numbers"
    assert gpu["utilization_pct"] is None
    assert gpu["vram_used_mb"] == 0.0
    assert gpu["vram_total_mb"] == 4096.0
    assert gpu["temperature_c"] is None


def test_parse_nvidia_smi_output_empty():
    assert parse_nvidia_smi_output("") == []
    assert parse_nvidia_smi_output("   \n\t  ") == []


def test_get_gpu_metrics_missing_binary(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    assert get_gpu_metrics() is None


def test_get_gpu_metrics_nonzero_exit():
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "NVIDIA-SMI has failed because it couldn't communicate with the driver."
    with patch("shutil.which", return_value="nvidia-smi"):
        with patch("subprocess.run", return_value=mock_proc):
            assert get_gpu_metrics() is None


def test_get_gpu_metrics_timeout():
    with patch("shutil.which", return_value="nvidia-smi"):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=1.5)):
            assert get_gpu_metrics() is None


def test_get_gpu_metrics_file_not_found():
    with patch("shutil.which", return_value="nvidia-smi"):
        with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
            assert get_gpu_metrics() is None


def test_format_metrics_for_display_with_gpu():
    snapshot = {
        "cpu": {"utilization_pct": 25.5, "available": True},
        "ram": {"used_gb": 12.3, "total_gb": 32.0, "utilization_pct": 38.4, "available": True},
        "gpu": {
            "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "utilization_pct": 14.0,
            "vram_used_gb": 2.1,
            "vram_total_gb": 6.0,
            "vram_pct": 35.0,
            "temperature_c": 48.0,
            "available": True,
        },
    }
    cpu_str, ram_str, gpu_util_str, gpu_vram_str, gpu_temp_str = format_metrics_for_display(snapshot)
    assert cpu_str == "25.5%"
    assert ram_str == "12.3 / 32.0 GB (38.4%)"
    assert "14%" in gpu_util_str
    assert "RTX 4050" in gpu_util_str
    assert gpu_vram_str == "2.1 / 6.0 GB (35.0%)"
    assert gpu_temp_str == "48 °C"


def test_format_metrics_for_display_without_gpu():
    snapshot = {
        "cpu": {"utilization_pct": 10.0, "available": True},
        "ram": {"used_gb": 8.0, "total_gb": 16.0, "utilization_pct": 50.0, "available": True},
        "gpu": None,
    }
    cpu_str, ram_str, gpu_util_str, gpu_vram_str, gpu_temp_str = format_metrics_for_display(snapshot)
    assert cpu_str == "10.0%"
    assert ram_str == "8.0 / 16.0 GB (50.0%)"
    assert gpu_util_str == "Unavailable"
    assert gpu_vram_str == "Unavailable"
    assert gpu_temp_str == "N/A"


def test_format_metrics_for_display_with_failures():
    snapshot = {
        "cpu": {"available": False},
        "ram": {"available": False},
        "gpu": None,
    }
    cpu_str, ram_str, gpu_util_str, gpu_vram_str, gpu_temp_str = format_metrics_for_display(snapshot)
    assert cpu_str == "N/A"
    assert ram_str == "N/A"
    assert gpu_util_str == "Unavailable"
    assert gpu_vram_str == "Unavailable"
    assert gpu_temp_str == "N/A"


def test_get_system_snapshot_structure():
    snapshot = get_system_snapshot()
    assert "cpu" in snapshot
    assert "ram" in snapshot
    assert "gpu" in snapshot
    assert "utilization_pct" in snapshot["cpu"]
    assert "used_gb" in snapshot["ram"]
