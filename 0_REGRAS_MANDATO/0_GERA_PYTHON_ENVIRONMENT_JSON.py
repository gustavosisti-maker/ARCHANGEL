from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_JSON = ROOT / "0_REGRAS_MANDATO" / "BASE_JSON"
OUT_PATH = BASE_JSON / "ARCHANGEL_PYTHON_ENVIRONMENT.json"
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "_CACHE" / "cupy_kernel_cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "_CACHE" / "numba_cache"))

FUTURE_HARDWARE_READINESS = {
    "status": "planned_not_installed",
    "design_goal": "Keep CUDA and CPU scaling runtime-detected, not hard-coded to the current machine.",
    "planned_motherboard": {
        "name": "ASUS ROG Strix X870-A Gaming WiFi",
        "platform": "AMD AM5 / X870 / DDR5 / PCIe 5.0",
    },
    "planned_cpu": {
        "name": "AMD Ryzen 9 9950X3D",
        "cores": 16,
        "threads": 32,
        "migration_note": "Use hardware profile after installation before raising CPU workers.",
    },
    "planned_gpu": {
        "name": "GeForce RTX 5080",
        "memory_gb": 16,
        "nvidia_architecture": "Blackwell",
        "cuda_compute_capability": "12.0",
        "migration_note": "Current CUDA backend selection should automatically use the new GPU after driver/library validation.",
    },
}

PACKAGES = [
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "joblib",
    "numba",
    "fastparquet",
    "polars",
    "duckdb",
    "dask",
    "distributed",
    "ray",
    "optuna",
    "skopt",
    "statsmodels",
    "xgboost",
    "lightgbm",
    "catboost",
    "ta",
    "talib",
    "pynvml",
    "tqdm",
    "rich",
    "matplotlib",
    "seaborn",
    "torch",
    "torchvision",
    "torchaudio",
    "cupy",
    "tensorflow",
]


def package_status(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "installed")
        return {"installed": True, "version": str(version)}
    except Exception as exc:
        return {
            "installed": False,
            "version": None,
            "import_error": f"{type(exc).__name__}: {exc}",
        }


def nvidia_smi() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mb": float(parts[2]),
                    "memory_free_mb": float(parts[3]),
                    "compute_capability": parts[4],
                }
            )
    return {"available": True, "gpu_count": len(gpus), "gpus": gpus}


def torch_cuda_status() -> dict:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        status = {
            "installed": True,
            "version": str(torch.__version__),
            "cuda_available": available,
            "cuda_version": str(getattr(torch.version, "cuda", None)),
            "device_count": int(torch.cuda.device_count()),
            "devices": [],
            "test_sum": None,
        }
        if available:
            status["devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            x = torch.arange(10, device="cuda", dtype=torch.float32)
            status["test_sum"] = float(x.sum().item())
        return status
    except Exception as exc:
        return {"installed": False, "cuda_available": False, "error": f"{type(exc).__name__}: {exc}"}


def cupy_cuda_status() -> dict:
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        a = cp.arange(10, dtype=cp.float32)
        return {
            "installed": True,
            "version": str(cp.__version__),
            "cuda_available": count > 0,
            "device_count": count,
            "test_sum": float(cp.sum(a).get()),
        }
    except Exception as exc:
        return {"installed": False, "cuda_available": False, "error": f"{type(exc).__name__}: {exc}"}


def numba_cuda_status() -> dict:
    try:
        from numba import cuda

        return {
            "installed": True,
            "cuda_available": bool(cuda.is_available()),
        }
    except Exception as exc:
        return {"installed": False, "cuda_available": False, "error": f"{type(exc).__name__}: {exc}"}


def pip_check() -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    BASE_JSON.mkdir(parents=True, exist_ok=True)
    packages = {name: package_status(name) for name in PACKAGES}
    torch_status = torch_cuda_status()
    cupy_status = cupy_cuda_status()
    numba_status = numba_cuda_status()
    gpu = nvidia_smi()

    cuda_ready = bool(torch_status.get("cuda_available") and cupy_status.get("cuda_available"))

    data = {
        "schema_version": "ARCHANGEL_PYTHON_ENVIRONMENT_1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "name": "ARCHANGEL",
            "layer": "PYTHON_CUDA_ENVIRONMENT",
            "script": Path(__file__).name,
        },
        "paths": {
            "project_root": str(ROOT),
            "base_json_dir": str(BASE_JSON),
            "python_executable": sys.executable,
            "python_scripts_dir": str(Path(sys.executable).with_name("Scripts")),
            "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
            "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        },
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "packages": packages,
        "cuda": {
            "ready_for_code_migration": cuda_ready,
            "nvidia_smi": gpu,
            "torch": torch_status,
            "cupy": cupy_status,
            "numba": numba_status,
            "tensorflow": packages["tensorflow"],
            "notes": [
                "PyTorch and CuPy are the primary CUDA paths currently validated for ARCHANGEL.",
                "TensorFlow is not installed because no compatible wheel was found for this Python/Windows combination during setup.",
                "torchaudio is not required for ARCHANGEL financial ML workloads.",
            ],
        },
        "future_hardware_readiness": FUTURE_HARDWARE_READINESS,
        "pip_check": pip_check(),
        "summary": {
            "core_stack_ready": all(
                packages[name]["installed"]
                for name in ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "joblib"]
            ),
            "ml_stack_ready": all(
                packages[name]["installed"]
                for name in ["xgboost", "lightgbm", "catboost", "optuna", "statsmodels"]
            ),
            "parallel_stack_ready": all(
                packages[name]["installed"] for name in ["dask", "distributed", "ray", "polars", "duckdb"]
            ),
            "cuda_ready_for_pytorch": bool(torch_status.get("cuda_available")),
            "cuda_ready_for_cupy": bool(cupy_status.get("cuda_available")),
            "cuda_ready_for_numba": bool(numba_status.get("cuda_available")),
            "cuda_ready_for_tensorflow": False,
            "recommended_next_step": "Start CUDA migration with isolated optional backends for PyTorch/CuPy, preserving CPU fallback.",
        },
    }

    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON salvo em: {OUT_PATH}")


if __name__ == "__main__":
    main()
