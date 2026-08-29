# -*- coding: utf-8 -*-
"""
Benchmark controlado CPU vs CUDA para a etapa 3 de features.

Este script nao reprocessa nem sobrescreve os Parquets oficiais. Ele usa
amostras reais das series mais lentas do ultimo run da etapa 3 e grava um
relatorio JSON em BASE_JSON para decisao de migracao CUDA.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RULES_DIR = ROOT_DIR / "0_REGRAS_MANDATO"
BASE_JSON_DIR = RULES_DIR / "BASE_JSON"
FEATURES_SCRIPT_PATH = RULES_DIR / "03_GERA_FEATURES.py"
LATEST_FEATURES_REPORT_PATH = BASE_JSON_DIR / "03_FEATURES_RUN_REPORT_LATEST.json"
PYTHON_ENVIRONMENT_PATH = BASE_JSON_DIR / "00_02_PYTHON_ENVIRONMENT_LATEST.json"
BENCHMARK_LATEST_PATH = BASE_JSON_DIR / "03_01_FEATURES_CUDA_BENCHMARK_LATEST.json"

SCHEMA_VERSION = "ARCHANGEL_FEATURES_CUDA_BENCHMARK_1.0"
DEFAULT_ROW_CAP = 250_000
DEFAULT_LIMIT = 2
NUMERICAL_ATOL = 1e-6
COMPARE_COLUMNS = (
    "feat_log_close",
    "feat_ret_log_1",
    "feat_ret_log_5",
    "feat_ret_log_20",
    "feat_ret_simple_1",
    "feat_roll_ret_sum_20",
    "feat_roll_ret_mean_20",
    "feat_rv_std_20",
    "feat_rv_abs_mean_20",
    "feat_rv_quadratic_20",
    "feat_sma_20",
    "feat_close_sma_20_ratio",
    "feat_bb_z_20",
    "feat_bb_width_20",
    "feat_volume_z_20",
    "feat_volume_ma_ratio_20",
    "feat_absret_volume_corr_20",
)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def phase_seconds(result: Dict[str, Any], phase_name: str) -> float:
    for item in result.get("phase_timings") or []:
        if item.get("phase") == phase_name:
            try:
                return float(item.get("elapsed_seconds") or 0.0)
            except Exception:
                return 0.0
    return 0.0


def family_seconds(result: Dict[str, Any], family_name: str) -> float:
    family = result.get("feature_family_benchmark") or {}
    item = family.get(family_name) if isinstance(family, dict) else None
    if not isinstance(item, dict):
        return 0.0
    try:
        return float(item.get("elapsed_seconds") or item.get("total_elapsed_seconds") or 0.0)
    except Exception:
        return 0.0


def select_candidates(report: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    candidates = []
    for result in report.get("results") or []:
        if not isinstance(result, dict) or result.get("status") != "OK":
            continue
        input_path = Path(str(result.get("input_path") or ""))
        if not input_path.is_file():
            continue
        candidates.append({
            "asset": result.get("asset"),
            "symbol": result.get("symbol"),
            "source": result.get("source"),
            "timeframe": result.get("timeframe"),
            "input_path": str(input_path),
            "latest_run_rows": result.get("prepared_rows") or result.get("raw_rows"),
            "latest_generate_all_features_seconds": phase_seconds(result, "generate_all_features"),
            "latest_write_parquet_seconds": phase_seconds(result, "write_parquet"),
            "latest_returns_seconds": family_seconds(result, "returns"),
            "latest_volume_seconds": family_seconds(result, "volume"),
            "latest_volatility_seconds": family_seconds(result, "volatility"),
        })

    candidates.sort(
        key=lambda item: float(item.get("latest_generate_all_features_seconds") or 0.0),
        reverse=True,
    )
    return candidates[:limit]


def load_features_module(mode: str):
    os.environ["ARCHANGEL_FEATURE_CUDA_MODE"] = mode
    os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT_DIR / "_CACHE" / "cupy_kernel_cache"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT_DIR / "_CACHE" / "numba_cache"))

    module_name = f"archangel_features_{mode}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, FEATURES_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar {FEATURES_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backend_from_steps(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    for step in steps or []:
        if step.get("family") == "__compute_backend__":
            return dict(step)
    return {}


def vector_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(step) for step in steps or [] if step.get("family") == "__cuda_vector_core__"]


def rolling_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(step) for step in steps or [] if step.get("family") == "__cuda_rolling__"]


def compare_outputs(cpu_df: pd.DataFrame, cuda_df: pd.DataFrame) -> Dict[str, Any]:
    comparisons = {}
    worst = 0.0
    for column in COMPARE_COLUMNS:
        if column not in cpu_df.columns or column not in cuda_df.columns:
            continue
        cpu_values = cpu_df[column].to_numpy(dtype="float64", copy=False)
        cuda_values = cuda_df[column].to_numpy(dtype="float64", copy=False)
        if len(cpu_values) != len(cuda_values):
            comparisons[column] = {"status": "LENGTH_MISMATCH"}
            continue
        diff = np.abs(cpu_values - cuda_values)
        if np.all(np.isnan(diff)):
            max_abs_diff = 0.0
        else:
            max_abs_diff = float(np.nanmax(diff))
        worst = max(worst, max_abs_diff)
        comparisons[column] = {
            "status": "OK",
            "max_abs_diff": max_abs_diff,
        }
    return {
        "status": "OK" if worst <= NUMERICAL_ATOL else "CHECK",
        "max_abs_diff": worst,
        "absolute_tolerance": NUMERICAL_ATOL,
        "columns": comparisons,
    }


def run_generate(module, prepared: pd.DataFrame, timeframe: Optional[str]) -> Dict[str, Any]:
    gc.collect()
    start = time.perf_counter()
    output_df, metadata, steps = module.generate_all_features(prepared.copy(), timeframe=timeframe)
    elapsed = time.perf_counter() - start
    feature_columns = [col for col in output_df.columns if str(col).startswith("feat_")]
    return {
        "elapsed_seconds": round(elapsed, 6),
        "output_df": output_df,
        "metadata_count": len(metadata),
        "feature_columns_count": len(feature_columns),
        "compute_backend": backend_from_steps(steps),
        "cuda_vector_steps": vector_steps(steps),
        "cuda_rolling_steps": rolling_steps(steps),
    }


def run_candidate(candidate: Dict[str, Any], row_cap: int) -> Dict[str, Any]:
    cpu_module = load_features_module("cpu")
    cuda_module = load_features_module("cuda")

    input_path = Path(candidate["input_path"])
    raw_df = cpu_module.read_ohlcv_parquet_fast(input_path)
    source_rows = int(len(raw_df))
    sampled_df = raw_df.tail(row_cap).copy() if row_cap and len(raw_df) > row_cap else raw_df.copy()
    sampled_rows = int(len(sampled_df))

    prepared_df = cpu_module.prepare_ohlcv(sampled_df)
    timeframe = candidate.get("timeframe")

    cpu_result = run_generate(cpu_module, prepared_df, timeframe)
    cuda_result = run_generate(cuda_module, prepared_df, timeframe)

    compare = compare_outputs(cpu_result["output_df"], cuda_result["output_df"])
    cpu_elapsed = float(cpu_result["elapsed_seconds"])
    cuda_elapsed = float(cuda_result["elapsed_seconds"])
    speedup = (cpu_elapsed / cuda_elapsed) if cuda_elapsed > 0 else None

    del raw_df, sampled_df, prepared_df
    del cpu_result["output_df"], cuda_result["output_df"]
    gc.collect()

    return {
        **candidate,
        "source_rows": source_rows,
        "sampled_rows": sampled_rows,
        "row_cap": row_cap,
        "cpu": cpu_result,
        "cuda": cuda_result,
        "comparison": compare,
        "speedup_cuda_vs_cpu": round(speedup, 6) if speedup is not None else None,
        "cuda_faster": bool(speedup is not None and speedup > 1.0),
    }


def recommendation(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "Benchmark sem resultados; verificar entradas e relatorio da etapa 3."

    valid = [item for item in results if item.get("status") == "OK"]
    if not valid:
        return "Benchmark falhou nas series testadas; manter etapa 3 em CPU ate corrigir os erros."

    speedups = [
        float(item.get("speedup_cuda_vs_cpu") or 0.0)
        for item in valid
        if item.get("speedup_cuda_vs_cpu") is not None
    ]
    cuda_used = any(
        ((item.get("cuda") or {}).get("compute_backend") or {}).get("resolved_backend") == "cupy_cuda"
        for item in valid
    )
    numerical_ok = all((item.get("comparison") or {}).get("status") == "OK" for item in valid)

    if not cuda_used:
        return "CUDA nao foi acionada no benchmark; revisar ambiente CuPy antes de medir ganho."
    if not numerical_ok:
        return "CUDA acionou, mas a equivalencia numerica precisa ser revisada antes de ampliar a migracao."
    if speedups and max(speedups) > 1.10:
        return (
            "CUDA ficou mais rapida nas rolling windows migradas e preservou equivalencia numerica; "
            "manter fallback CPU e testar run completo apenas com limite conservador de workers."
        )
    return (
        "CUDA esta correta numericamente, mas o ganho ainda e pequeno nesta GPU; "
        "manter auto conservador e usar CUDA em smoke tests ou em GPU com mais memoria."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CPU vs CUDA da etapa 3.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Numero de series mais lentas a testar.")
    parser.add_argument("--row-cap", type=int, default=DEFAULT_ROW_CAP, help="Maximo de linhas recentes por serie.")
    args = parser.parse_args()

    started_at = now_iso()
    latest_report = load_json(LATEST_FEATURES_REPORT_PATH)
    python_env = load_json(PYTHON_ENVIRONMENT_PATH)
    candidates = select_candidates(latest_report, max(1, int(args.limit)))

    results = []
    for candidate in candidates:
        try:
            result = run_candidate(candidate, max(1, int(args.row_cap)))
            result["status"] = "OK"
        except Exception as exc:
            result = {
                **candidate,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            }
        results.append(result)
        atomic_write_json(
            BENCHMARK_LATEST_PATH,
            build_payload(started_at, results, python_env, partial=True, row_cap=max(1, int(args.row_cap)), limit=max(1, int(args.limit))),
        )

    payload = build_payload(
        started_at,
        results,
        python_env,
        partial=False,
        row_cap=max(1, int(args.row_cap)),
        limit=max(1, int(args.limit)),
    )
    atomic_write_json(BENCHMARK_LATEST_PATH, payload)

    summary = payload.get("summary") or {}
    print("[FEATURES_CUDA_BENCHMARK]", json.dumps(summary, ensure_ascii=False, default=str))
    print(f"[OK] JSON salvo em: {BENCHMARK_LATEST_PATH}")
    return 0 if summary.get("series_error", 0) == 0 else 1


def build_payload(
    started_at: str,
    results: List[Dict[str, Any]],
    python_env: Dict[str, Any],
    partial: bool,
    row_cap: int,
    limit: int,
) -> Dict[str, Any]:
    finished_at = now_iso()
    ok_results = [item for item in results if item.get("status") == "OK"]
    cpu_total = sum(float((item.get("cpu") or {}).get("elapsed_seconds") or 0.0) for item in ok_results)
    cuda_total = sum(float((item.get("cuda") or {}).get("elapsed_seconds") or 0.0) for item in ok_results)
    speedup = (cpu_total / cuda_total) if cuda_total > 0 else None
    cuda_backend_count: Dict[str, int] = {}
    cuda_rolling_family_count: Dict[str, int] = {}
    cuda_rolling_accelerated_operations = 0
    cuda_rolling_fallback_operations = 0
    for item in ok_results:
        backend = ((item.get("cuda") or {}).get("compute_backend") or {}).get("resolved_backend") or "UNKNOWN"
        cuda_backend_count[backend] = cuda_backend_count.get(backend, 0) + 1
        for step in (item.get("cuda") or {}).get("cuda_rolling_steps") or []:
            if not isinstance(step, dict):
                continue
            family = str(step.get("source_family") or "unknown")
            cuda_rolling_family_count[family] = cuda_rolling_family_count.get(family, 0) + 1
            cuda_rolling_accelerated_operations += int(step.get("accelerated_count") or 0)
            cuda_rolling_fallback_operations += int(step.get("fallback_count") or 0)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": finished_at,
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "partial": partial,
        "system": {
            "name": "ARCHANGEL",
            "version": "v1",
            "layer": "FEATURES_CUDA_BENCHMARK",
            "script": Path(__file__).name,
            "generated_at": finished_at,
        },
        "paths": {
            "benchmark_latest_path": str(BENCHMARK_LATEST_PATH),
            "features_script_path": str(FEATURES_SCRIPT_PATH),
            "latest_features_report_path": str(LATEST_FEATURES_REPORT_PATH),
            "python_environment_path": str(PYTHON_ENVIRONMENT_PATH),
        },
        "config": {
            "row_cap": row_cap,
            "limit": limit,
            "candidate_selection": "series_ok ordenadas por maior tempo generate_all_features no ultimo run da etapa 3",
            "compare_columns": list(COMPARE_COLUMNS),
            "numerical_absolute_tolerance": NUMERICAL_ATOL,
            "cpu_mode": "ARCHANGEL_FEATURE_CUDA_MODE=cpu",
            "cuda_mode": "ARCHANGEL_FEATURE_CUDA_MODE=cuda",
            "note": "Benchmark isolado: nao grava features oficiais nem altera manifestos da etapa 3.",
        },
        "summary": {
            "status": "PARTIAL" if partial else ("OK" if len(ok_results) == len(results) else "CHECK"),
            "series_tested": len(results),
            "series_ok": len(ok_results),
            "series_error": len(results) - len(ok_results),
            "cpu_total_seconds": round(cpu_total, 6),
            "cuda_total_seconds": round(cuda_total, 6),
            "speedup_cuda_vs_cpu_total": round(speedup, 6) if speedup is not None else None,
            "cuda_faster_count": sum(1 for item in ok_results if item.get("cuda_faster")),
            "cuda_slower_or_equal_count": sum(1 for item in ok_results if not item.get("cuda_faster")),
            "cuda_backend_count": cuda_backend_count,
            "cuda_rolling_family_count": dict(sorted(cuda_rolling_family_count.items())),
            "cuda_rolling_accelerated_operations": int(cuda_rolling_accelerated_operations),
            "cuda_rolling_fallback_operations": int(cuda_rolling_fallback_operations),
            "numerical_comparison_ok": all((item.get("comparison") or {}).get("status") == "OK" for item in ok_results) if ok_results else False,
            "recommendation": recommendation(results),
        },
        "python_cuda_summary": (python_env.get("summary") if isinstance(python_env, dict) else {}),
        "started_at": started_at,
        "finished_at": finished_at,
        "results": results,
    }


if __name__ == "__main__":
    raise SystemExit(main())
