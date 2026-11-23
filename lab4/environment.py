"""Параметри середовища для пункту 5: модель CPU, ОС, версії бібліотек, цілочисельні інструкції."""

import platform
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from numpy._core._multiarray_umath import __cpu_features__

# Інструкції, від яких залежить швидкість INT8 (пункт 6): VNNI на x86, dot product на ARM.
INT8_FEATURES = ["AVX2", "AVX512F", "AVX512VNNI", "ASIMD", "ASIMDDP"]


def cpu_name():
    """Назва процесора: на Windows — з реєстру, на macOS — sysctl, на Linux — /proc/cpuinfo."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True).strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def describe(threads, batch_size):
    return {
        "cpu": cpu_name(),
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "cpu_features": {name: bool(__cpu_features__.get(name, False)) for name in INT8_FEATURES},
        "threads": threads,
        "batch_size": batch_size,
    }
