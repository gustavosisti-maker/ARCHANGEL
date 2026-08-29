# -*- coding: utf-8 -*-
"""ARCHANGEL v1 - walk-forward ML baseline.

Treina um baseline NumPy sem dependencias externas de ML. O objetivo desta
etapa e validar a governanca: features permitidas, purging/embargo, previsoes
fora da amostra e registro de experimentos.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RULES_DIR = ROOT_DIR / "0_REGRAS_MANDATO"
BASE_JSON_DIR = RULES_DIR / "BASE_JSON"

DATASETS_JSON_PATH = BASE_JSON_DIR / "05_DATASETS_ML_LATEST.json"
WALK_FORWARD_JSON_PATH = BASE_JSON_DIR / "06_WALK_FORWARD_LATEST.json"
MACHINE_PROFILE_PATH = BASE_JSON_DIR / "00_01_MACHINE_PROFILE_LATEST.json"
PYTHON_ENVIRONMENT_PATH = BASE_JSON_DIR / "00_02_PYTHON_ENVIRONMENT_LATEST.json"

EXPERIMENTS_DIR = ROOT_DIR / "6_EXPERIMENTS"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
MODELS_DIR = EXPERIMENTS_DIR / "models"
PREDICTIONS_DIR = EXPERIMENTS_DIR / "predictions"
LOGS_DIR = EXPERIMENTS_DIR / "_logs"
REGISTRY_PATH = EXPERIMENTS_DIR / "experiments.sqlite"
RUN_REPORT_LATEST_PATH = BASE_JSON_DIR / "06_WALK_FORWARD_LATEST.json"

SCRIPT_NAME = "06_WALK_FORWARD_TRAINING.py"
SCHEMA_VERSION = "ARCHANGEL_WALK_FORWARD_1.0"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

TIMESTAMP_COL = "timestamp_utc_ms"
DATETIME_COL = "DateTime"
DEFAULT_TARGET = "label_dir_h20_thr25bps"
DEFAULT_MODEL = "auto_logistic_binary"
DEFAULT_MAX_FEATURES = 80
DEFAULT_MAX_TRAIN_ROWS = 200_000
DEFAULT_MAX_TEST_ROWS = 50_000

FUTURE_HARDWARE_READINESS = {
    "current_gpu_policy": "Use CUDA opportunistically through PyTorch/CuPy and keep CPU fallback active.",
    "planned_cpu": {
        "name": "AMD Ryzen 9 9950X3D",
        "cores": 16,
        "threads": 32,
        "migration_note": "Increase CPU workers after hardware profile confirms thermals and RAM bandwidth.",
    },
    "planned_gpu": {
        "name": "GeForce RTX 5080",
        "memory_gb": 16,
        "nvidia_architecture": "Blackwell",
        "cuda_compute_capability": "12.0",
        "migration_note": "Backend selection is capability/runtime based, so no code path should hard-code GTX 1660 Ti.",
    },
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, required: bool = True) -> Dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"JSON obrigatorio nao encontrado: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_python_environment() -> Dict[str, Any]:
    return load_json(PYTHON_ENVIRONMENT_PATH, required=False)


def resolve_training_backend(args: argparse.Namespace) -> Dict[str, Any]:
    python_env = load_python_environment()
    cuda = python_env.get("cuda", {}) if isinstance(python_env, dict) else {}
    torch_status = cuda.get("torch", {}) if isinstance(cuda, dict) else {}
    nvidia = cuda.get("nvidia_smi", {}) if isinstance(cuda, dict) else {}

    requested_model = str(args.model or DEFAULT_MODEL)
    requested_device = str(args.device or "auto").lower()
    torch_cuda_available = bool(torch_status.get("cuda_available"))
    cuda_ready = bool(cuda.get("ready_for_code_migration")) or torch_cuda_available

    if requested_device == "cpu" or requested_model.startswith("numpy"):
        backend = "numpy_cpu"
        device = "cpu"
        reason = "CPU solicitado explicitamente ou modelo NumPy selecionado."
    elif requested_device == "cuda" and not torch_cuda_available:
        backend = "numpy_cpu"
        device = "cpu"
        reason = "CUDA foi solicitado, mas torch.cuda nao esta disponivel; usando fallback CPU."
    elif requested_device in {"auto", "cuda"} and cuda_ready and torch_cuda_available:
        backend = "torch_cuda"
        device = "cuda"
        reason = "CUDA validado no ambiente Python; usando PyTorch na GPU."
    else:
        backend = "numpy_cpu"
        device = "cpu"
        reason = "CUDA nao validado; usando baseline NumPy CPU."

    return {
        "requested_model": requested_model,
        "requested_device": requested_device,
        "resolved_backend": backend,
        "resolved_device": device,
        "reason": reason,
        "torch_cuda_available": torch_cuda_available,
        "torch_version": torch_status.get("version"),
        "torch_cuda_version": torch_status.get("cuda_version"),
        "nvidia_smi": nvidia,
        "python_environment_path": str(PYTHON_ENVIRONMENT_PATH),
    }


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def safe_token(value: Any) -> str:
    text = str(value or "unknown")
    text = text.replace("\\", "_").replace("/", "_").replace(":", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", text)


def short_hash(*values: Any) -> str:
    raw = "|".join(str(value) for value in values)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]


def parse_csv_filter(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    if value.strip().lower() in {"", "all", "*"}:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def infer_horizon_bars(label_name: str, default: int = 20) -> int:
    match = re.search(r"_h(\d+)", str(label_name))
    return int(match.group(1)) if match else default


def timeframe_to_ms(timeframe: str, fallback_ms: int = 60_000) -> int:
    text = str(timeframe or "").strip().lower()
    try:
        if text.endswith("min"):
            return int(text[:-3]) * 60_000
        if text.endswith("h"):
            return int(text[:-1]) * 3_600_000
        if text.endswith("d"):
            return int(text[:-1]) * 86_400_000
        if text.endswith("s"):
            return int(text[:-1]) * 1_000
    except Exception:
        return fallback_ms
    return fallback_ms


def candidate_features(allowed: list[str], max_features: int) -> list[str]:
    priority_tokens = (
        "xasset_",
        "ret_log",
        "rv_std",
        "atr_",
        "bb_",
        "rsi_",
        "ema_",
        "volume_z",
        "dollar_volume",
        "efficiency",
        "vol_regime",
        "breakout",
        "cost_",
    )
    selected = [col for col in allowed if any(tok in col for tok in priority_tokens)]
    selected += [col for col in allowed if col not in selected]
    return selected[: max(1, int(max_features))]


def select_datasets(manifest: Dict[str, Any], args: argparse.Namespace) -> list[Dict[str, Any]]:
    assets = parse_csv_filter(args.assets)
    sources = parse_csv_filter(args.sources)
    timeframes = parse_csv_filter(args.timeframes)
    datasets = [item for item in manifest.get("datasets", []) if item.get("status") == "OK"]
    out = []
    for item in datasets:
        if assets is not None and str(item.get("asset")) not in assets:
            continue
        if sources is not None and str(item.get("source")) not in sources:
            continue
        if timeframes is not None and str(item.get("timeframe")) not in timeframes:
            continue
        if args.min_trainable_rows and int(item.get("trainable_rows") or 0) < args.min_trainable_rows:
            continue
        out.append(item)
    out = sorted(out, key=lambda x: (str(x.get("source")), str(x.get("asset")), str(x.get("timeframe"))))
    if args.limit is not None:
        out = out[: max(0, args.limit)]
    return out


def make_row_windows(n_rows: int, train_rows: int, test_rows: int, step_rows: int) -> list[Dict[str, int]]:
    windows = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_rows
        test_start = train_end
        test_end = min(test_start + test_rows, n_rows)
        if test_end - test_start < max(50, int(test_rows * 0.25)):
            break
        windows.append({
            "window_id": len(windows) + 1,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        start += step_rows
        if test_end >= n_rows:
            break
    return windows


def apply_purge_embargo(
    train_idx: np.ndarray,
    test_start_ts: int,
    test_end_ts: int,
    purge_ms: int,
    embargo_ms: int,
    timestamps: np.ndarray,
) -> np.ndarray:
    train_ts = timestamps[train_idx]
    keep = train_ts < (test_start_ts - purge_ms)
    keep &= ~((train_ts >= test_start_ts) & (train_ts <= test_end_ts + embargo_ms))
    return train_idx[keep]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def prepare_matrix(df: pd.DataFrame, feature_cols: list[str], stats: Optional[Dict[str, np.ndarray]] = None) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).astype("float64")
    if stats is None:
        med = x.median(axis=0, skipna=True).fillna(0.0).to_numpy(dtype="float64")
        arr = x.fillna(pd.Series(med, index=feature_cols)).to_numpy(dtype="float64")
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        stats = {"median": med, "mean": mean, "std": std}
    else:
        arr = x.fillna(pd.Series(stats["median"], index=feature_cols)).to_numpy(dtype="float64")
    arr = (arr - stats["mean"]) / stats["std"]
    arr = np.clip(arr, -8.0, 8.0)
    return arr.astype("float32"), stats


def fit_logistic_numpy(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    epochs: int,
    lr: float,
    l2: float,
) -> Dict[str, np.ndarray]:
    n_rows, n_cols = x.shape
    w = np.zeros(n_cols, dtype="float64")
    b = 0.0
    if sample_weight is None:
        sw = np.ones(n_rows, dtype="float64")
    else:
        sw = sample_weight.astype("float64")
    sw = sw / max(float(sw.mean()), 1e-12)

    for _ in range(max(1, int(epochs))):
        p = sigmoid(x @ w + b)
        err = (p - y) * sw
        grad_w = (x.T @ err) / max(1, n_rows) + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return {"coef": w.astype("float32"), "intercept": np.array([b], dtype="float32")}


def predict_logistic(model: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return sigmoid(x @ model["coef"].astype("float64") + float(model["intercept"][0]))


def fit_predict_logistic_torch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray],
    x_test: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    device: str,
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    import torch

    torch_device = torch.device(device)
    x_t = torch.as_tensor(x_train, dtype=torch.float32, device=torch_device)
    y_t = torch.as_tensor(y_train, dtype=torch.float32, device=torch_device)
    if sample_weight is None:
        sw_t = torch.ones_like(y_t)
    else:
        sw_t = torch.as_tensor(sample_weight, dtype=torch.float32, device=torch_device)
    sw_t = sw_t / torch.clamp(sw_t.mean(), min=1e-12)

    w = torch.zeros(x_t.shape[1], dtype=torch.float32, device=torch_device)
    b = torch.zeros((), dtype=torch.float32, device=torch_device)

    for _ in range(max(1, int(epochs))):
        logits = torch.clamp(x_t.matmul(w) + b, min=-35.0, max=35.0)
        p = torch.sigmoid(logits)
        err = (p - y_t) * sw_t
        grad_w = x_t.T.matmul(err) / max(1, x_t.shape[0]) + float(l2) * w
        grad_b = err.mean()
        w = w - float(lr) * grad_w
        b = b - float(lr) * grad_b

    x_eval = torch.as_tensor(x_test, dtype=torch.float32, device=torch_device)
    with torch.no_grad():
        proba = torch.sigmoid(torch.clamp(x_eval.matmul(w) + b, min=-35.0, max=35.0))
        proba_np = proba.detach().cpu().numpy().astype("float32")
        model = {
            "coef": w.detach().cpu().numpy().astype("float32"),
            "intercept": np.array([float(b.detach().cpu().item())], dtype="float32"),
        }
    return model, proba_np


def fit_predict_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray],
    x_test: np.ndarray,
    args: argparse.Namespace,
    backend_plan: Dict[str, Any],
) -> tuple[Dict[str, np.ndarray], np.ndarray, str, Optional[str]]:
    if backend_plan.get("resolved_backend") == "torch_cuda":
        try:
            model, proba = fit_predict_logistic_torch(
                x_train=x_train,
                y_train=y_train,
                sample_weight=sample_weight,
                x_test=x_test,
                epochs=args.epochs,
                lr=args.learning_rate,
                l2=args.l2,
                device=str(backend_plan.get("resolved_device") or "cuda"),
            )
            return model, proba, "torch_cuda", None
        except Exception as exc:
            if not args.cuda_fallback_cpu:
                raise
            model = fit_logistic_numpy(
                x=x_train,
                y=y_train,
                sample_weight=sample_weight,
                epochs=args.epochs,
                lr=args.learning_rate,
                l2=args.l2,
            )
            proba = predict_logistic(model, x_test)
            return model, proba, "numpy_cpu_fallback", f"{type(exc).__name__}: {exc}"

    model = fit_logistic_numpy(
        x=x_train,
        y=y_train,
        sample_weight=sample_weight,
        epochs=args.epochs,
        lr=args.learning_rate,
        l2=args.l2,
    )
    proba = predict_logistic(model, x_test)
    return model, proba, "numpy_cpu", None


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    pred = (proba >= threshold).astype("int8")
    y = y_true.astype("int8")
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    total = max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "pred_positive_rate": float(pred.mean()) if len(pred) else None,
        "accuracy": float((pred == y).sum() / total),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision_pos": float(precision),
        "recall_pos": float(recall),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def strategy_proxy_metrics(df: pd.DataFrame, proba: np.ndarray, target_horizon: int) -> Dict[str, Any]:
    ret_col = f"label_fwd_ret_net_long_h{target_horizon}"
    if ret_col not in df.columns:
        return {"available": False}
    signal = (proba >= 0.55).astype("float64")
    ret = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    pnl = signal * ret
    active = signal > 0
    if pnl.size == 0:
        return {"available": True, "trades": 0}
    equity = np.cumprod(1.0 + np.clip(pnl, -0.99, 10.0))
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return {
        "available": True,
        "trades": int(active.sum()),
        "mean_return_per_bar": float(pnl.mean()),
        "mean_return_when_active": float(pnl[active].mean()) if active.any() else 0.0,
        "hit_rate_when_active": float((pnl[active] > 0).mean()) if active.any() else None,
        "total_return_proxy": float(equity[-1] - 1.0),
        "max_drawdown_proxy": float(dd.min()),
    }


def save_model_npz(path: Path, model: Dict[str, np.ndarray], stats: Dict[str, np.ndarray], feature_cols: list[str]) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        coef=model["coef"],
        intercept=model["intercept"],
        median=stats["median"],
        mean=stats["mean"],
        std=stats["std"],
        feature_cols=np.array(feature_cols, dtype=object),
    )


def init_registry() -> None:
    ensure_dir(REGISTRY_PATH.parent)
    with sqlite3.connect(REGISTRY_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                created_at_utc TEXT,
                run_id TEXT,
                dataset_path TEXT,
                target_col TEXT,
                model_type TEXT,
                status TEXT,
                metrics_json TEXT,
                report_path TEXT
            )
            """
        )
        conn.commit()


def register_experiment(record: Dict[str, Any]) -> None:
    init_registry()
    with sqlite3.connect(REGISTRY_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                experiment_id, created_at_utc, run_id, dataset_path, target_col,
                model_type, status, metrics_json, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["experiment_id"],
                now_utc_iso(),
                RUN_ID,
                record.get("dataset_path"),
                record.get("target_col"),
                record.get("model_type"),
                record.get("status"),
                json.dumps(record.get("metrics", {}), ensure_ascii=False, default=str),
                record.get("report_path"),
            ),
        )
        conn.commit()


def train_one_dataset(item: Dict[str, Any], args: argparse.Namespace, backend_plan: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    dataset_path = Path(str(item["output_path"]))
    target_col = args.target or item.get("target_default") or DEFAULT_TARGET
    target_horizon = infer_horizon_bars(target_col)
    tf_ms = timeframe_to_ms(str(item.get("timeframe")))
    purge_ms = int(args.purge_bars if args.purge_bars is not None else target_horizon) * tf_ms
    embargo_ms = int(args.embargo_bars if args.embargo_bars is not None else target_horizon) * tf_ms

    allowed = item.get("allowed_feature_columns") or []
    feature_cols = candidate_features([str(col) for col in allowed], args.max_features)
    ret_col = f"label_fwd_ret_net_long_h{target_horizon}"
    read_cols = list(dict.fromkeys([
        TIMESTAMP_COL,
        DATETIME_COL,
        target_col,
        "is_trainable",
        "sample_weight",
        ret_col,
        *feature_cols,
    ]))

    experiment_id = short_hash(RUN_ID, dataset_path, target_col, args.model)
    result: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": "PENDING",
        "dataset_path": str(dataset_path),
        "asset": item.get("asset"),
        "symbol": item.get("symbol"),
        "source": item.get("source"),
        "timeframe": item.get("timeframe"),
        "target_col": target_col,
        "model_type": args.model,
        "backend_plan": backend_plan,
        "feature_columns": feature_cols,
        "purge_bars": int(args.purge_bars if args.purge_bars is not None else target_horizon),
        "embargo_bars": int(args.embargo_bars if args.embargo_bars is not None else target_horizon),
    }

    try:
        df = pd.read_parquet(dataset_path, columns=[col for col in read_cols if col], engine="pyarrow")
        if target_col not in df.columns:
            raise ValueError(f"Target ausente no dataset: {target_col}")

        df[TIMESTAMP_COL] = pd.to_numeric(df[TIMESTAMP_COL], errors="coerce")
        df = df.dropna(subset=[TIMESTAMP_COL, target_col]).sort_values(TIMESTAMP_COL, kind="mergesort")
        if "is_trainable" in df.columns:
            df = df[df["is_trainable"].astype(bool)].copy()

        y_raw = pd.to_numeric(df[target_col], errors="coerce")
        df = df[y_raw.notna()].copy()
        y_raw = y_raw[y_raw.notna()]
        y_bin = (y_raw.to_numpy(dtype="float64") > 0).astype("int8")
        df["__target_binary"] = y_bin

        if len(df) < args.train_rows + max(50, args.test_rows // 4):
            raise ValueError(f"Poucas linhas para walk-forward: {len(df)}")

        windows = make_row_windows(len(df), args.train_rows, args.test_rows, args.step_rows)
        if args.max_windows is not None:
            windows = windows[: max(0, args.max_windows)]
        if not windows:
            raise ValueError("Nenhuma janela walk-forward criada.")

        timestamps = df[TIMESTAMP_COL].to_numpy(dtype="int64")
        predictions = []
        window_metrics = []
        backend_usage: Dict[str, int] = {}
        cuda_fallback_errors: list[str] = []

        for window in windows:
            train_idx = np.arange(window["train_start"], window["train_end"])
            test_idx = np.arange(window["test_start"], window["test_end"])
            test_start_ts = int(timestamps[test_idx[0]])
            test_end_ts = int(timestamps[test_idx[-1]])
            train_idx = apply_purge_embargo(train_idx, test_start_ts, test_end_ts, purge_ms, embargo_ms, timestamps)
            if len(train_idx) < max(100, args.min_train_rows_after_purge):
                continue

            if len(train_idx) > args.max_train_rows:
                train_idx = train_idx[-args.max_train_rows:]
            if len(test_idx) > args.max_test_rows:
                test_idx = test_idx[:args.max_test_rows]

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            x_train, stats = prepare_matrix(train_df, feature_cols)
            x_test, _ = prepare_matrix(test_df, feature_cols, stats)
            y_train = train_df["__target_binary"].to_numpy(dtype="float64")
            y_test = test_df["__target_binary"].to_numpy(dtype="int8")
            sw = train_df["sample_weight"].to_numpy(dtype="float64") if "sample_weight" in train_df.columns else None

            model, proba, backend_used, fallback_error = fit_predict_logistic(
                x_train=x_train,
                y_train=y_train,
                sample_weight=sw,
                x_test=x_test,
                args=args,
                backend_plan=backend_plan,
            )
            backend_usage[backend_used] = backend_usage.get(backend_used, 0) + 1
            if fallback_error:
                cuda_fallback_errors.append(fallback_error)

            metrics = classification_metrics(y_test, proba, threshold=args.threshold)
            proxy = strategy_proxy_metrics(test_df, proba, target_horizon)
            metrics.update({
                "window_id": window["window_id"],
                "backend_used": backend_used,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_start_ts": int(timestamps[train_idx[0]]),
                "train_end_ts": int(timestamps[train_idx[-1]]),
                "test_start_ts": test_start_ts,
                "test_end_ts": test_end_ts,
                "strategy_proxy": proxy,
            })
            window_metrics.append(metrics)

            pred_df = pd.DataFrame({
                "experiment_id": experiment_id,
                "window_id": window["window_id"],
                TIMESTAMP_COL: test_df[TIMESTAMP_COL].to_numpy(dtype="int64"),
                "actual_label": y_raw.iloc[test_idx].to_numpy(dtype="float64"),
                "actual_binary": y_test,
                "proba_long": proba.astype("float32"),
                "pred_binary": (proba >= args.threshold).astype("int8"),
            })
            if DATETIME_COL in test_df.columns:
                pred_df[DATETIME_COL] = test_df[DATETIME_COL].to_numpy()
            predictions.append(pred_df)

            model_path = MODELS_DIR / f"{experiment_id}_w{window['window_id']:03d}.npz"
            save_model_npz(model_path, model, stats, feature_cols)

        if not window_metrics:
            raise ValueError("Todas as janelas foram descartadas apos purging/embargo.")

        predictions_df = pd.concat(predictions, ignore_index=True)
        pred_path = PREDICTIONS_DIR / f"{experiment_id}_predictions.parquet"
        ensure_dir(pred_path.parent)
        predictions_df.to_parquet(pred_path, engine="pyarrow", compression="zstd", index=False)

        summary_metrics = {
            "windows": len(window_metrics),
            "avg_accuracy": float(np.mean([m["accuracy"] for m in window_metrics])),
            "avg_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in window_metrics])),
            "avg_precision_pos": float(np.mean([m["precision_pos"] for m in window_metrics])),
            "avg_recall_pos": float(np.mean([m["recall_pos"] for m in window_metrics])),
            "positive_windows_accuracy_gt_50pct": int(sum(m["accuracy"] > 0.50 for m in window_metrics)),
            "total_oos_rows": int(sum(m["test_rows"] for m in window_metrics)),
        }
        proxy_returns = [
            m.get("strategy_proxy", {}).get("total_return_proxy")
            for m in window_metrics
            if m.get("strategy_proxy", {}).get("available")
        ]
        if proxy_returns:
            summary_metrics["avg_total_return_proxy"] = float(np.mean(proxy_returns))

        result.update({
            "status": "OK",
            "rows_loaded": int(len(df)),
            "feature_columns_count": len(feature_cols),
            "window_metrics": window_metrics,
            "metrics": summary_metrics,
            "backend_usage": backend_usage,
            "cuda_fallback_errors": cuda_fallback_errors[:10],
            "predictions_path": str(pred_path),
        })

    except Exception as exc:
        result.update({"status": "ERROR", "error": str(exc), "metrics": {}})

    result["elapsed_seconds"] = round(time.time() - started, 6)
    return result


def build_machine_summary() -> Dict[str, Any]:
    profile = load_json(MACHINE_PROFILE_PATH, required=False)
    python_env = load_json(PYTHON_ENVIRONMENT_PATH, required=False)
    packages = profile.get("software_environment", {}).get("python_packages", {})
    ai_compute = profile.get("ai_compute_profile", {})
    return {
        "python_packages": {
            name: {
                "installed": cfg.get("installed"),
                "version": cfg.get("version"),
            }
            for name, cfg in packages.items()
            if name in {"numpy", "pandas", "pyarrow", "sklearn", "lightgbm", "xgboost", "torch", "numba", "duckdb", "polars"}
        },
        "ai_compute_profile": ai_compute,
        "python_environment_summary": python_env.get("summary", {}) if isinstance(python_env, dict) else {},
        "python_cuda": python_env.get("cuda", {}) if isinstance(python_env, dict) else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward ML baseline ARCHANGEL.")
    parser.add_argument("--assets", default="BTC,ETH,SOL,BNB,XRP")
    parser.add_argument("--sources", default="binance_spot")
    parser.add_argument("--timeframes", default="5min,15min,1h")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-cuda-fallback-cpu", dest="cuda_fallback_cpu", action="store_false")
    parser.set_defaults(cuda_fallback_cpu=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-trainable-rows", type=int, default=600)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--train-rows", type=int, default=700)
    parser.add_argument("--test-rows", type=int, default=200)
    parser.add_argument("--step-rows", type=int, default=200)
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument("--max-test-rows", type=int, default=DEFAULT_MAX_TEST_ROWS)
    parser.add_argument("--min-train-rows-after-purge", type=int, default=300)
    parser.add_argument("--purge-bars", type=int)
    parser.add_argument("--embargo-bars", type=int)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    backend_plan = resolve_training_backend(args)

    for path in [EXPERIMENTS_DIR, CONFIGS_DIR, RESULTS_DIR, MODELS_DIR, PREDICTIONS_DIR, LOGS_DIR]:
        ensure_dir(path)

    manifest = load_json(DATASETS_JSON_PATH)
    selected = select_datasets(manifest, args)
    print("=" * 100)
    print("ARCHANGEL v1 | 6_WALK_FORWARD | BACKEND AUTO CPU/CUDA")
    print("=" * 100)
    print(f"[RUN_ID] {RUN_ID}")
    print(f"[DATASETS_JSON] {DATASETS_JSON_PATH}")
    print(f"[SELECTED] {len(selected)}")
    print(f"[MODEL] {args.model}")
    print(f"[BACKEND] {backend_plan['resolved_backend']} | device={backend_plan['resolved_device']}")
    print(f"[TARGET] {args.target}")
    print(f"[MAX_FEATURES] {args.max_features}")
    print("=" * 100)

    results = []
    for idx, item in enumerate(selected, start=1):
        result = train_one_dataset(item, args, backend_plan)
        results.append(result)
        register_experiment({
            **result,
            "report_path": str(WALK_FORWARD_JSON_PATH),
        })
        m = result.get("metrics") or {}
        print(
            f"[PROGRESSO] {idx}/{len(selected)} | {result.get('status')} | "
            f"{result.get('source')} | {result.get('asset')} | {result.get('timeframe')} | "
            f"windows={m.get('windows')} | bal_acc={m.get('avg_balanced_accuracy')} | "
            f"{result.get('elapsed_seconds')}s"
            + (f" | error={result.get('error')}" if result.get("error") else "")
        )

    ok = [item for item in results if item.get("status") == "OK"]
    errors = [item for item in results if item.get("status") != "OK"]
    backend_usage_total: Dict[str, int] = {}
    for item in ok:
        for backend, count in (item.get("backend_usage") or {}).items():
            backend_usage_total[str(backend)] = backend_usage_total.get(str(backend), 0) + int(count)
    run_report_path = LOGS_DIR / f"6_WALK_FORWARD_RUN_REPORT_{RUN_ID}.json"
    run_report_base_json_path = RUN_REPORT_LATEST_PATH
    payload = {
        "schema_version": SCHEMA_VERSION,
        "system": {
            "name": "ARCHANGEL",
            "version": "v1",
            "layer": "6_WALK_FORWARD",
            "script": SCRIPT_NAME,
            "run_id": RUN_ID,
            "generated_at_utc": now_utc_iso(),
        },
        "paths": {
            "root_dir": str(ROOT_DIR),
            "datasets_json_path": str(DATASETS_JSON_PATH),
            "python_environment_path": str(PYTHON_ENVIRONMENT_PATH),
            "experiments_dir": str(EXPERIMENTS_DIR),
            "registry_path": str(REGISTRY_PATH),
            "walk_forward_json_path": str(WALK_FORWARD_JSON_PATH),
            "run_report_path": str(run_report_path),
            "run_report_base_json_path": str(run_report_base_json_path),
            "run_report_latest_path": str(RUN_REPORT_LATEST_PATH),
        },
        "policy": {
            "model_allowed_feature_source": "05_DATASETS_ML_LATEST.allowed_feature_columns",
            "forbidden_prefixes": ["label_", "meta_", "quality_"],
            "purging_embargo": "Train rows inside target horizon before test start are removed; test interval is never used in train.",
            "oos_rule": "Only predictions on walk-forward test windows are saved.",
        },
        "config": vars(args),
        "backend_plan": backend_plan,
        "future_hardware_readiness": FUTURE_HARDWARE_READINESS,
        "machine_summary": build_machine_summary(),
        "summary": {
            "datasets_selected": len(selected),
            "experiments_ok": len(ok),
            "experiments_error": len(errors),
            "backend_usage": backend_usage_total,
            "cuda_enabled_for_this_run": bool(backend_usage_total.get("torch_cuda")),
            "cuda_fallback_windows": int(backend_usage_total.get("numpy_cpu_fallback", 0)),
            "avg_balanced_accuracy": (
                float(np.mean([r["metrics"]["avg_balanced_accuracy"] for r in ok])) if ok else None
            ),
            "total_oos_rows": int(sum(r.get("metrics", {}).get("total_oos_rows", 0) or 0 for r in ok)),
        },
        "experiments": results,
    }
    write_json_atomic(run_report_path, payload)
    write_json_atomic(run_report_base_json_path, payload)
    write_json_atomic(RUN_REPORT_LATEST_PATH, payload)
    write_json_atomic(WALK_FORWARD_JSON_PATH, payload)

    print("=" * 100)
    print(f"[OK] {len(ok)}")
    print(f"[ERROS] {len(errors)}")
    print(f"[JSON WALK_FORWARD] {WALK_FORWARD_JSON_PATH}")
    print(f"[REGISTRY] {REGISTRY_PATH}")
    print("=" * 100)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
