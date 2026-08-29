# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_NAME = "08_EXPERIMENT_REGISTRY.py"
SCHEMA_VERSION = "ARCHANGEL_EXPERIMENT_REGISTRY_2.0"

RULES_DIR = Path(__file__).resolve().parent
ROOT_DIR = RULES_DIR.parent
BASE_JSON_DIR = RULES_DIR / "BASE_JSON"

REGISTRY_DIR = ROOT_DIR / "8_EXPERIMENT_REGISTRY"
LOGS_DIR = REGISTRY_DIR / "_logs"
PARQUET_DIR = REGISTRY_DIR / "REGISTRY_PARQUET"
JSONL_DIR = REGISTRY_DIR / "REGISTRY_JSONL"
SQLITE_DIR = REGISTRY_DIR / "REGISTRY_SQLITE"

FEATURES_JSON_PATH = BASE_JSON_DIR / "03_FEATURES_CATALOG_LATEST.json"
LABELS_JSON_PATH = BASE_JSON_DIR / "04_LABELS_CATALOG_LATEST.json"
DATASETS_JSON_PATH = BASE_JSON_DIR / "05_DATASETS_ML_LATEST.json"
WALK_FORWARD_JSON_PATH = BASE_JSON_DIR / "06_WALK_FORWARD_LATEST.json"
BACKTEST_JSON_PATH = BASE_JSON_DIR / "07_BACKTEST_PORTFOLIO_LATEST.json"
BACKTEST_VALIDATION_JSON_PATH = BASE_JSON_DIR / "07_BACKTEST_VALIDATION_LATEST.json"
BACKTEST_STRESS_JSON_PATH = BASE_JSON_DIR / "07_BACKTEST_STRESS_LATEST.json"
BACKTEST_PARAM_SEARCH_JSON_PATH = BASE_JSON_DIR / "07_BACKTEST_PARAM_SEARCH_LATEST.json"
COST_MODEL_JSON_PATH = BASE_JSON_DIR / "00_COST_MODEL.json"
PYTHON_ENV_JSON_PATH = BASE_JSON_DIR / "00_02_PYTHON_ENVIRONMENT_LATEST.json"
MACHINE_PROFILE_JSON_PATH = BASE_JSON_DIR / "00_01_MACHINE_PROFILE_LATEST.json"
AI_CONTEXT_INDEX_PATH = BASE_JSON_DIR / "99_AI_CONTEXT_INDEX_LATEST.json"

REGISTRY_JSON_PATH = BASE_JSON_DIR / "08_EXPERIMENT_REGISTRY_LATEST.json"
REGISTRY_LATEST_JSON_PATH = BASE_JSON_DIR / "08_EXPERIMENT_REGISTRY_LATEST.json"
RUN_REPORT_LATEST_PATH = BASE_JSON_DIR / "08_EXPERIMENT_REGISTRY_LATEST.json"
VALIDATION_LATEST_PATH = BASE_JSON_DIR / "08_EXPERIMENT_REGISTRY_VALIDATION_LATEST.json"

REGISTRY_JSONL_PATH = JSONL_DIR / "8_EXPERIMENT_REGISTRY.jsonl"
REGISTRY_PARQUET_PATH = PARQUET_DIR / "8_EXPERIMENT_REGISTRY_LATEST.parquet"
REGISTRY_SQLITE_PATH = SQLITE_DIR / "8_EXPERIMENT_REGISTRY.sqlite"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def run_id_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    for path in [BASE_JSON_DIR, REGISTRY_DIR, LOGS_DIR, PARQUET_DIR, JSONL_DIR, SQLITE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any | None = None) -> Any:
    if default is None:
        default = {}
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        path.write_text(tmp_path.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            tmp_path.unlink()
        except OSError:
            pass


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def stable_hash(value: Any, length: int = 16) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def file_fingerprint(path_value: str | None, sample_bytes: int = 65536) -> dict[str, Any]:
    if not path_value:
        return {"path": None, "exists": False}
    path = Path(path_value)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            head = handle.read(sample_bytes)
            digest.update(head)
            if stat.st_size > sample_bytes:
                handle.seek(max(0, stat.st_size - sample_bytes))
                digest.update(handle.read(sample_bytes))
    except OSError as exc:
        return {
            "path": str(path),
            "exists": True,
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "hash_error": str(exc),
        }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "sha256_head_tail": digest.hexdigest(),
        "hash_policy": "sha256(head+tail) for large artifacts; enough for fast lineage, not legal-grade full-file attestation.",
    }


def manifest_summary(name: str, path: Path) -> dict[str, Any]:
    data = load_json(path, default={})
    system = data.get("system", {}) if isinstance(data, dict) else {}
    return {
        "name": name,
        "path": str(path),
        "exists": path.is_file(),
        "schema_version": data.get("schema_version") if isinstance(data, dict) else None,
        "run_id": data.get("run_id") or system.get("run_id") if isinstance(data, dict) else None,
        "generated_at_utc": data.get("generated_at_utc") or system.get("generated_at_utc") or system.get("generated_at") if isinstance(data, dict) else None,
        "summary": data.get("summary") if isinstance(data, dict) else None,
        "fingerprint": file_fingerprint(str(path)),
    }


def series_key(item: dict[str, Any]) -> str:
    return "|".join(str(item.get(part) or "").lower() for part in ["source", "asset", "symbol", "timeframe"])


def build_dataset_index(datasets_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {series_key(item): item for item in datasets_json.get("datasets", []) if isinstance(item, dict)}


def build_backtest_index(backtest_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("experiment_id")): item
        for item in backtest_json.get("experiments", [])
        if isinstance(item, dict) and item.get("experiment_id")
    }


def find_model_files(experiment_id: str) -> list[str]:
    model_dir = ROOT_DIR / "6_EXPERIMENTS" / "models"
    if not model_dir.is_dir():
        return []
    return [str(path) for path in sorted(model_dir.glob(f"{experiment_id}_w*.npz"))]


def artifact_record(kind: str, path: str | None) -> dict[str, Any]:
    fp = file_fingerprint(path)
    return {
        "kind": kind,
        "path": fp.get("path"),
        "exists": fp.get("exists"),
        "size_bytes": fp.get("size_bytes"),
        "mtime_utc": fp.get("mtime_utc"),
        "fingerprint": fp.get("sha256_head_tail"),
    }


def flatten_for_table(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            flat[key] = value
    return flat


def build_registry_rows(run_id: str, manifests: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = build_dataset_index(manifests["datasets"])
    backtests = build_backtest_index(manifests["backtest"])
    wf_experiments = manifests["walk_forward"].get("experiments", [])
    backtest_summary = manifests["backtest"].get("summary", {})
    backtest_config = manifests["backtest"].get("config", {})
    validation_summary = manifests["backtest_validation"].get("summary", {})
    stress_summary = manifests["backtest_stress"].get("summary", {})
    param_search_summary = manifests["backtest_param_search"].get("summary", {})
    cost_model_summary = manifests["cost_model"].get("summary", {})

    rows: list[dict[str, Any]] = []
    for experiment in wf_experiments:
        if not isinstance(experiment, dict):
            continue
        dataset = datasets.get(series_key(experiment), {})
        backtest = backtests.get(str(experiment.get("experiment_id")), {})
        model_files = find_model_files(str(experiment.get("experiment_id")))
        feature_columns = experiment.get("feature_columns", [])
        artifact_paths = {
            "feature_path": dataset.get("feature_path"),
            "label_path": dataset.get("label_path"),
            "dataset_path": experiment.get("dataset_path") or dataset.get("output_path"),
            "predictions_path": experiment.get("predictions_path"),
            "backtest_trades_path": backtest.get("trades_path"),
            "backtest_equity_path": backtest.get("equity_path"),
            "model_files": model_files,
        }
        artifact_fingerprints = {
            name: file_fingerprint(path)
            for name, path in artifact_paths.items()
            if isinstance(path, str) or path is None
        }
        artifact_fingerprints["model_files"] = [file_fingerprint(path) for path in model_files]

        wf_metrics = experiment.get("metrics", {}) if isinstance(experiment.get("metrics"), dict) else {}
        bt_metrics = backtest.get("metrics", {}) if isinstance(backtest.get("metrics"), dict) else {}
        row = {
            "registry_run_id": run_id,
            "registry_generated_at_utc": utc_now_iso(),
            "registry_schema_version": SCHEMA_VERSION,
            "experiment_id": experiment.get("experiment_id"),
            "registry_id": stable_hash(
                {
                    "experiment_id": experiment.get("experiment_id"),
                    "dataset_path": artifact_paths["dataset_path"],
                    "predictions_path": artifact_paths["predictions_path"],
                    "target_col": experiment.get("target_col"),
                    "model_type": experiment.get("model_type"),
                    "backtest_config_hash": stable_hash(backtest_config),
                }
            ),
            "asset": experiment.get("asset"),
            "symbol": experiment.get("symbol"),
            "source": experiment.get("source"),
            "timeframe": experiment.get("timeframe"),
            "target_col": experiment.get("target_col"),
            "horizon_bars": backtest.get("horizon_bars"),
            "dataset_status": dataset.get("status"),
            "dataset_quality_status": dataset.get("quality_status"),
            "dataset_ml_gate": dataset.get("quality_ml_gate"),
            "dataset_trainable_rows": dataset.get("trainable_rows"),
            "dataset_total_rows": dataset.get("rows"),
            "dataset_allowed_feature_columns_count": dataset.get("allowed_feature_columns_count"),
            "feature_columns_count": len(feature_columns) if isinstance(feature_columns, list) else experiment.get("feature_columns_count"),
            "feature_columns_hash": stable_hash(feature_columns),
            "forbidden_feature_policy": "label_, meta_ and quality_ are forbidden as model inputs unless explicitly allowed by dataset manifest.",
            "model_type": experiment.get("model_type"),
            "model_backend": (experiment.get("backend_plan") or {}).get("resolved_backend") if isinstance(experiment.get("backend_plan"), dict) else None,
            "model_device": (experiment.get("backend_plan") or {}).get("resolved_device") if isinstance(experiment.get("backend_plan"), dict) else None,
            "model_files_count": len(model_files),
            "walk_forward_status": experiment.get("status"),
            "walk_forward_windows": wf_metrics.get("windows"),
            "walk_forward_avg_accuracy": wf_metrics.get("avg_accuracy"),
            "walk_forward_avg_balanced_accuracy": wf_metrics.get("avg_balanced_accuracy"),
            "walk_forward_total_oos_rows": wf_metrics.get("total_oos_rows"),
            "backtest_status": backtest.get("status"),
            "backtest_trades": bt_metrics.get("total_trades"),
            "backtest_total_return": bt_metrics.get("total_return"),
            "backtest_cagr": bt_metrics.get("cagr"),
            "backtest_max_drawdown": bt_metrics.get("max_drawdown"),
            "backtest_win_rate": bt_metrics.get("win_rate"),
            "backtest_profit_factor": bt_metrics.get("profit_factor"),
            "backtest_research_status": bt_metrics.get("status"),
            "portfolio_run_id": backtest_summary.get("run_id"),
            "portfolio_total_return": backtest_summary.get("portfolio_total_return"),
            "portfolio_cagr": backtest_summary.get("portfolio_cagr"),
            "portfolio_max_drawdown": backtest_summary.get("portfolio_max_drawdown"),
            "portfolio_research_status": backtest_summary.get("portfolio_research_status"),
            "approval_status": "NOT_AN_APPROVAL_ENGINE",
            "target_annual_return_min": backtest_summary.get("target_annual_return_min"),
            "reference_drawdown_limit": backtest_summary.get("reference_drawdown_limit"),
            "cost_model_schema": manifests["cost_model"].get("schema_version"),
            "cost_model_summary": cost_model_summary,
            "risk_config_hash": stable_hash({key: backtest_config.get(key) for key in sorted(backtest_config) if "risk" in key or "exposure" in key or "drawdown" in key or "leverage" in key or "position" in key}),
            "backtest_config_hash": stable_hash(backtest_config),
            "validation_status": validation_summary.get("status"),
            "validation_checks_failed": validation_summary.get("checks_failed"),
            "stress_status": stress_summary.get("status"),
            "stress_worst_scenario": stress_summary.get("worst_scenario"),
            "stress_worst_total_return": stress_summary.get("worst_total_return"),
            "stress_worst_max_drawdown": stress_summary.get("worst_max_drawdown"),
            "param_search_status": param_search_summary.get("status"),
            "param_search_candidates": param_search_summary.get("candidates_evaluated"),
            "param_search_best_score": param_search_summary.get("best_score"),
            "param_search_passes_research_references": param_search_summary.get("passes_research_references"),
            "feature_path": artifact_paths["feature_path"],
            "label_path": artifact_paths["label_path"],
            "dataset_path": artifact_paths["dataset_path"],
            "predictions_path": artifact_paths["predictions_path"],
            "backtest_trades_path": artifact_paths["backtest_trades_path"],
            "backtest_equity_path": artifact_paths["backtest_equity_path"],
            "model_files": model_files,
            "artifact_fingerprints": artifact_fingerprints,
            "lineage_hash": stable_hash(
                {
                    "features_manifest": manifest_summary("features", FEATURES_JSON_PATH).get("fingerprint"),
                    "labels_manifest": manifest_summary("labels", LABELS_JSON_PATH).get("fingerprint"),
                    "datasets_manifest": manifest_summary("datasets", DATASETS_JSON_PATH).get("fingerprint"),
                    "walk_forward_manifest": manifest_summary("walk_forward", WALK_FORWARD_JSON_PATH).get("fingerprint"),
                    "backtest_manifest": manifest_summary("backtest", BACKTEST_JSON_PATH).get("fingerprint"),
                    "cost_model": manifest_summary("cost_model", COST_MODEL_JSON_PATH).get("fingerprint"),
                    "experiment_id": experiment.get("experiment_id"),
                }
            ),
            "status": classify_registry_status(dataset, experiment, backtest, validation_summary, stress_summary),
        }
        rows.append(row)
    return rows


def classify_registry_status(
    dataset: dict[str, Any],
    experiment: dict[str, Any],
    backtest: dict[str, Any],
    validation_summary: dict[str, Any],
    stress_summary: dict[str, Any],
) -> str:
    if dataset.get("quality_ml_gate") == "ML_BLOCKED":
        return "BLOCKED_DATA_QUALITY"
    if experiment.get("status") != "OK":
        return "BLOCKED_WALK_FORWARD"
    if backtest and backtest.get("status") != "OK":
        return "NEEDS_BACKTEST_REVIEW"
    if validation_summary.get("status") not in {None, "PASS"}:
        return "NEEDS_VALIDATION_REVIEW"
    if stress_summary.get("status") not in {None, "OK"}:
        return "NEEDS_STRESS_REVIEW"
    return "RESEARCH_REGISTERED"


def write_registry_sqlite(rows: list[dict[str, Any]], manifests_index: list[dict[str, Any]]) -> None:
    flat_rows = [flatten_for_table(row) for row in rows]
    artifact_rows = []
    for row in rows:
        for kind in ["feature_path", "label_path", "dataset_path", "predictions_path", "backtest_trades_path", "backtest_equity_path"]:
            artifact = artifact_record(kind, row.get(kind))
            artifact["registry_id"] = row.get("registry_id")
            artifact["experiment_id"] = row.get("experiment_id")
            artifact_rows.append(artifact)
        for model_path in row.get("model_files", []):
            artifact = artifact_record("model_file", model_path)
            artifact["registry_id"] = row.get("registry_id")
            artifact["experiment_id"] = row.get("experiment_id")
            artifact_rows.append(artifact)

    manifest_rows = [flatten_for_table(item) for item in manifests_index]
    with sqlite3.connect(REGISTRY_SQLITE_PATH) as conn:
        pd.DataFrame(flat_rows).to_sql("experiments", conn, if_exists="replace", index=False)
        pd.DataFrame(artifact_rows).to_sql("artifacts", conn, if_exists="replace", index=False)
        pd.DataFrame(manifest_rows).to_sql("manifests", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_registry_id ON experiments(registry_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_symbol_timeframe ON experiments(symbol, timeframe)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_experiment_id ON artifacts(experiment_id)")
        conn.commit()


def append_registry_jsonl(rows: list[dict[str, Any]]) -> None:
    REGISTRY_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_JSONL_PATH.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def validate_registry(rows: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add("registry_has_rows", len(rows) > 0, f"rows={len(rows)}")
    ids = [row.get("registry_id") for row in rows]
    add("registry_ids_unique", len(ids) == len(set(ids)), f"unique={len(set(ids))} total={len(ids)}")
    exp_ids = [row.get("experiment_id") for row in rows]
    add("experiment_ids_present", all(bool(item) for item in exp_ids), f"missing={sum(1 for item in exp_ids if not item)}")
    add("parquet_exists", REGISTRY_PARQUET_PATH.is_file(), str(REGISTRY_PARQUET_PATH))
    add("sqlite_exists", REGISTRY_SQLITE_PATH.is_file(), str(REGISTRY_SQLITE_PATH))
    add("jsonl_exists", REGISTRY_JSONL_PATH.is_file(), str(REGISTRY_JSONL_PATH))
    add("base_json_exists", REGISTRY_JSON_PATH.is_file(), str(REGISTRY_JSON_PATH))
    missing_dataset = sum(1 for row in rows if not file_fingerprint(row.get("dataset_path")).get("exists"))
    missing_pred = sum(1 for row in rows if not file_fingerprint(row.get("predictions_path")).get("exists"))
    add("dataset_artifacts_exist", missing_dataset == 0, f"missing_dataset={missing_dataset}")
    add("prediction_artifacts_exist", missing_pred == 0, f"missing_predictions={missing_pred}")
    add("approval_separated", all(row.get("approval_status") == "NOT_AN_APPROVAL_ENGINE" for row in rows), "registry does not approve execution")

    if REGISTRY_SQLITE_PATH.is_file():
        try:
            with sqlite3.connect(REGISTRY_SQLITE_PATH) as conn:
                sqlite_count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            add("sqlite_row_count_matches", sqlite_count == len(rows), f"sqlite={sqlite_count} rows={len(rows)}")
        except Exception as exc:
            add("sqlite_row_count_matches", False, str(exc))

    if REGISTRY_PARQUET_PATH.is_file():
        try:
            parquet_count = len(pd.read_parquet(REGISTRY_PARQUET_PATH))
            add("parquet_row_count_matches", parquet_count == len(rows), f"parquet={parquet_count} rows={len(rows)}")
        except Exception as exc:
            add("parquet_row_count_matches", False, str(exc))

    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": "ARCHANGEL_EXPERIMENT_REGISTRY_VALIDATION_1.0",
        "system": {
            "name": "ARCHANGEL",
            "version": "v1",
            "layer": "8_EXPERIMENT_REGISTRY_VALIDATION",
            "script": SCRIPT_NAME,
            "run_id": payload["system"]["run_id"],
            "generated_at_utc": utc_now_iso(),
        },
        "summary": {
            "status": "PASS" if passed else "FAIL",
            "checks_total": len(checks),
            "checks_passed": sum(1 for item in checks if item["passed"]),
            "checks_failed": sum(1 for item in checks if not item["passed"]),
            "registry_rows": len(rows),
        },
        "checks": checks,
    }


def update_ai_context_index(summary: dict[str, Any]) -> None:
    if not AI_CONTEXT_INDEX_PATH.is_file():
        return
    try:
        index = load_json(AI_CONTEXT_INDEX_PATH, default={})
        if not isinstance(index, dict):
            return
        files = index.setdefault("files", [])
        existing = {item.get("file"): item for item in files if isinstance(item, dict)}
        for file_name, role, hint in [
            ("08_EXPERIMENT_REGISTRY_LATEST.json", "registry formal de experimentos e telemetria", "Use para rastrear dataset, features, labels, modelo, custos, risco, backtest, stress, validacao, status e caminhos de outputs."),
            ("08_EXPERIMENT_REGISTRY_VALIDATION_LATEST.json", "validacao do registry formal", "Use para verificar unicidade, artefatos e persistencia em JSON/Parquet/SQLite."),
        ]:
            path = BASE_JSON_DIR / file_name
            data = load_json(path, default={}) if path.is_file() else {}
            existing[file_name] = {
                "file": file_name,
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "schema_version": data.get("schema_version") if isinstance(data, dict) else None,
                "run_id": (data.get("system") or {}).get("run_id") if isinstance(data, dict) else None,
                "generated_at": (data.get("system") or {}).get("generated_at_utc") if isinstance(data, dict) else None,
                "role": role,
                "ai_reading_hint": hint,
                "summary": data.get("summary", summary) if isinstance(data, dict) else summary,
            }
        index["files"] = list(existing.values())
        order = index.setdefault("ai_reading_order", [])
        for name in [
            "08_EXPERIMENT_REGISTRY_LATEST.json",
            "08_EXPERIMENT_REGISTRY_VALIDATION_LATEST.json",
        ]:
            if name not in order:
                order.append(name)
        root_summary = index.setdefault("summary", {})
        root_summary["experiment_registry_status"] = summary.get("status")
        root_summary["experiment_registry_rows"] = summary.get("registry_rows")
        root_summary["experiment_registry_unique_experiments"] = summary.get("unique_experiments")
        root_summary["experiment_registry_validation_status"] = summary.get("validation_status")
        root_summary["experiment_registry_sqlite_path"] = str(REGISTRY_SQLITE_PATH)
        write_json_atomic(AI_CONTEXT_INDEX_PATH, index)
    except Exception:
        return


def main() -> int:
    ensure_dirs()
    started = time.perf_counter()
    run_id = run_id_now()
    print("\nARCHANGEL v1 - ETAPA 8 EXPERIMENT REGISTRY")
    print("=" * 100)
    print(f"Run ID: {run_id}")
    print(f"Rules dir: {RULES_DIR}")
    print(f"Base JSON: {BASE_JSON_DIR}")
    print("=" * 100)

    manifest_paths = {
        "features": FEATURES_JSON_PATH,
        "labels": LABELS_JSON_PATH,
        "datasets": DATASETS_JSON_PATH,
        "walk_forward": WALK_FORWARD_JSON_PATH,
        "backtest": BACKTEST_JSON_PATH,
        "backtest_validation": BACKTEST_VALIDATION_JSON_PATH,
        "backtest_stress": BACKTEST_STRESS_JSON_PATH,
        "backtest_param_search": BACKTEST_PARAM_SEARCH_JSON_PATH,
        "cost_model": COST_MODEL_JSON_PATH,
        "python_environment": PYTHON_ENV_JSON_PATH,
        "machine_profile": MACHINE_PROFILE_JSON_PATH,
    }
    manifests = {name: load_json(path, default={}) for name, path in manifest_paths.items()}
    manifests_index = [manifest_summary(name, path) for name, path in manifest_paths.items()]
    rows = build_registry_rows(run_id, manifests)

    flat_rows = [flatten_for_table(row) for row in rows]
    pd.DataFrame(flat_rows).to_parquet(REGISTRY_PARQUET_PATH, index=False)
    append_registry_jsonl(rows)
    write_registry_sqlite(rows, manifests_index)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    missing_artifacts = {
        "feature_path": sum(1 for row in rows if not file_fingerprint(row.get("feature_path")).get("exists")),
        "label_path": sum(1 for row in rows if not file_fingerprint(row.get("label_path")).get("exists")),
        "dataset_path": sum(1 for row in rows if not file_fingerprint(row.get("dataset_path")).get("exists")),
        "predictions_path": sum(1 for row in rows if not file_fingerprint(row.get("predictions_path")).get("exists")),
        "backtest_trades_path": sum(1 for row in rows if not file_fingerprint(row.get("backtest_trades_path")).get("exists")),
        "backtest_equity_path": sum(1 for row in rows if not file_fingerprint(row.get("backtest_equity_path")).get("exists")),
    }

    summary = {
        "status": "OK",
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "registry_rows": len(rows),
        "unique_experiments": len({row.get("experiment_id") for row in rows}),
        "status_counts": status_counts,
        "missing_artifacts": missing_artifacts,
        "approval_status": "NOT_AN_APPROVAL_ENGINE",
        "portfolio_research_status": manifests["backtest"].get("summary", {}).get("portfolio_research_status"),
        "portfolio_total_return": manifests["backtest"].get("summary", {}).get("portfolio_total_return"),
        "portfolio_cagr": manifests["backtest"].get("summary", {}).get("portfolio_cagr"),
        "portfolio_max_drawdown": manifests["backtest"].get("summary", {}).get("portfolio_max_drawdown"),
        "target_annual_return_min": manifests["backtest"].get("summary", {}).get("target_annual_return_min"),
        "reference_drawdown_limit": manifests["backtest"].get("summary", {}).get("reference_drawdown_limit"),
        "validation_status": None,
        "elapsed_seconds": None,
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "system": {
            "name": "ARCHANGEL",
            "version": "v1",
            "layer": "8_EXPERIMENT_REGISTRY",
            "script": SCRIPT_NAME,
            "run_id": run_id,
            "generated_at_utc": summary["generated_at_utc"],
        },
        "paths": {
            "root_dir": str(ROOT_DIR),
            "rules_dir": str(RULES_DIR),
            "base_json_dir": str(BASE_JSON_DIR),
            "registry_dir": str(REGISTRY_DIR),
            "registry_json_path": str(REGISTRY_JSON_PATH),
            "registry_latest_json_path": str(REGISTRY_LATEST_JSON_PATH),
            "registry_run_report_latest_path": str(RUN_REPORT_LATEST_PATH),
            "registry_validation_latest_path": str(VALIDATION_LATEST_PATH),
            "registry_jsonl_path": str(REGISTRY_JSONL_PATH),
            "registry_parquet_path": str(REGISTRY_PARQUET_PATH),
            "registry_sqlite_path": str(REGISTRY_SQLITE_PATH),
        },
        "policy": {
            "purpose": "Registry formal para rastrear experimentos de pesquisa ponta a ponta.",
            "approval": "Este registry nao aprova execucao real nem testnet. Ele registra evidencia, linhagem e status.",
            "lineage": "Cada experimento vincula dataset, features, labels, previsoes, modelos, custos, risco, backtest, stress e validacao.",
            "fingerprints": "Artefatos recebem fingerprint rapido por tamanho, mtime e sha256 head/tail.",
            "ai_usage": "Codex deve comecar por este JSON quando precisar entender historico e rastreabilidade de experimentos.",
        },
        "summary": summary,
        "manifest_index": manifests_index,
        "registry_columns": list(flat_rows[0].keys()) if flat_rows else [],
        "experiments": rows,
        "top_experiments_by_balanced_accuracy": sorted(
            rows,
            key=lambda row: safe_float(row.get("walk_forward_avg_balanced_accuracy"), -1.0) or -1.0,
            reverse=True,
        )[:10],
        "top_experiments_by_backtest_total_return": sorted(
            rows,
            key=lambda row: safe_float(row.get("backtest_total_return"), -999.0) or -999.0,
            reverse=True,
        )[:10],
        "telemetry": {
            "elapsed_seconds": None,
            "process_id": os.getpid(),
            "rows_written_parquet": len(rows),
            "sqlite_tables": ["experiments", "artifacts", "manifests"],
        },
        "next_steps": [
            "Usar este registry como tabela mestre antes de ablação de features.",
            "Adicionar tags manuais de hipótese quando novas famílias de features forem testadas.",
            "Criar promotion gates separados se futuramente houver critérios formais para testnet.",
        ],
    }

    write_json_atomic(REGISTRY_JSON_PATH, payload)
    write_json_atomic(REGISTRY_LATEST_JSON_PATH, payload)
    run_report_path = LOGS_DIR / f"8_EXPERIMENT_REGISTRY_RUN_REPORT_{run_id}.json"
    write_json_atomic(run_report_path, payload)
    write_json_atomic(RUN_REPORT_LATEST_PATH, payload)

    validation = validate_registry(rows, payload)
    payload["summary"]["validation_status"] = validation["summary"]["status"]
    payload["summary"]["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    payload["telemetry"]["elapsed_seconds"] = payload["summary"]["elapsed_seconds"]
    write_json_atomic(VALIDATION_LATEST_PATH, validation)
    write_json_atomic(REGISTRY_JSON_PATH, payload)
    write_json_atomic(REGISTRY_LATEST_JSON_PATH, payload)
    write_json_atomic(run_report_path, payload)
    write_json_atomic(RUN_REPORT_LATEST_PATH, payload)
    update_ai_context_index(payload["summary"])

    print("\nRESUMO REGISTRY")
    print("-" * 100)
    print(f"Status: {payload['summary']['status']}")
    print(f"Experimentos registrados: {payload['summary']['registry_rows']}")
    print(f"Experimentos unicos: {payload['summary']['unique_experiments']}")
    print(f"Validacao: {payload['summary']['validation_status']}")
    print(f"Parquet: {REGISTRY_PARQUET_PATH}")
    print(f"SQLite: {REGISTRY_SQLITE_PATH}")
    print(f"JSON: {REGISTRY_JSON_PATH}")
    print("-" * 100)
    return 0 if validation["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
