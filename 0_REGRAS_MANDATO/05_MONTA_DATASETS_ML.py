# -*- coding: utf-8 -*-
"""ARCHANGEL v1 - montagem de datasets ML.

Junta features e labels por serie/timeframe, preserva metadados e grava um
manifesto com colunas permitidas para treino. Esta etapa nao treina modelos.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RULES_DIR = ROOT_DIR / "0_REGRAS_MANDATO"
BASE_JSON_DIR = RULES_DIR / "BASE_JSON"

FEATURES_JSON_PATH = BASE_JSON_DIR / "03_FEATURES_CATALOG_LATEST.json"
LABELS_JSON_PATH = BASE_JSON_DIR / "04_LABELS_CATALOG_LATEST.json"
COST_MODEL_PATH = BASE_JSON_DIR / "00_COST_MODEL.json"
DATA_QUALITY_REPORT_PATH = BASE_JSON_DIR / "02_01_DATA_QUALITY_REPORT_LATEST.json"

DATASETS_DIR = ROOT_DIR / "5_DATASETS_ML"
DATASETS_PARQUET_DIR = DATASETS_DIR / "DATASETS_PARQUET"
DATASETS_LOG_DIR = DATASETS_DIR / "_logs"
DATASETS_JSON_PATH = BASE_JSON_DIR / "05_DATASETS_ML_LATEST.json"
RUN_REPORT_LATEST_PATH = BASE_JSON_DIR / "05_DATASETS_ML_LATEST.json"

SCRIPT_NAME = "05_MONTA_DATASETS_ML.py"
SCHEMA_VERSION = "ARCHANGEL_DATASET_ML_1.1_FORMAL_ML_GATE"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

DATETIME_COL = "DateTime"
TIMESTAMP_COL = "timestamp_utc_ms"
TIMEZONE_LOCAL = "Asia/Dubai"
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "zstd"

DEFAULT_ASSETS = {"BTC", "ETH", "SOL", "BNB", "XRP"}
DEFAULT_TIMEFRAMES = {"5min", "15min", "1h"}
DEFAULT_SOURCES: Optional[set[str]] = None
DEFAULT_TARGET = "label_dir_h20_thr25bps"

FEATURE_PREFIXES = ("feat_", "regime_")
QUALITY_COLUMNS = {
    "quality_ml_row_eligible",
    "quality_is_feature_valid",
    "quality_is_warmup",
    "quality_data_usable_for_ml",
    "quality_feature_non_null_ratio",
}
META_COLUMNS = {
    DATETIME_COL,
    TIMESTAMP_COL,
    "feature_ready_timestamp_utc_ms",
    "meta_asset",
    "meta_symbol",
    "meta_source",
    "meta_timeframe",
    "meta_timezone",
    "meta_bar_timestamp_policy",
    "meta_feature_schema_version",
    "meta_feature_run_id",
}
BTC_XASSET_COLUMNS = (
    "feat_ret_log_1",
    "feat_ret_log_5",
    "feat_rv_std_20",
    "feat_rv_std_50",
)

ML_GATE_READY = "ML_READY"
ML_GATE_CAUTION_ACCEPTABLE = "ML_CAUTION_ACCEPTABLE"
ML_GATE_BLOCKED = "ML_BLOCKED"
ML_GATE_NOT_APPLICABLE = "ML_NOT_APPLICABLE"
ML_TRAINABLE_GATES = {ML_GATE_READY, ML_GATE_CAUTION_ACCEPTABLE}


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
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]


def parse_csv_filter(value: Optional[str], default: Optional[set[str]]) -> Optional[set[str]]:
    if value is None:
        return default
    if value.strip().lower() in {"", "all", "*"}:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def as_bool(value: Any, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "sim", "y"}:
        return True
    if text in {"0", "false", "no", "nao", "não", "n"}:
        return False
    return default


def build_quality_index(quality_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_series = quality_report.get("series", {}) if isinstance(quality_report, dict) else {}
    if isinstance(raw_series, dict):
        return {
            str(series_id): record
            for series_id, record in raw_series.items()
            if isinstance(record, dict)
        }
    if isinstance(raw_series, list):
        return {
            str(record.get("series_id")): record
            for record in raw_series
            if isinstance(record, dict) and record.get("series_id")
        }
    return {}


def root_cause_code(record: Dict[str, Any]) -> str:
    root = record.get("root_cause")
    if isinstance(root, dict):
        return str(root.get("code") or "UNKNOWN")
    return str(root or "UNKNOWN")


def classify_ml_gate(feature: Dict[str, Any], quality_record: Dict[str, Any]) -> Dict[str, Any]:
    quality_summary = feature.get("quality_summary") or {}
    post_audit = feature.get("post_audit") or {}

    raw_status = str(
        feature.get("quality_ml_status")
        or quality_record.get("ml_quality_status")
        or ""
    ).strip()

    usable = as_bool(
        feature.get("quality_ml_usable_for_broad_training"),
        default=as_bool(quality_record.get("ml_usable_for_broad_training"), default=True),
    )

    quality_status = str(
        feature.get("quality_status_normalized")
        or quality_record.get("status")
        or quality_summary.get("quality_status")
        or "UNKNOWN"
    )
    audit_status = str(post_audit.get("audit_status") or "UNKNOWN")
    root_code = root_cause_code(quality_record)

    if raw_status == "ML_READY":
        gate = ML_GATE_READY
        reason = "Relatorio de qualidade marcou a serie como pronta para ML amplo."
    elif raw_status == "ML_CAUTION":
        gate = ML_GATE_CAUTION_ACCEPTABLE if usable else ML_GATE_BLOCKED
        reason = (
            "WARNING classificado como aceitavel para ML amplo."
            if usable
            else "WARNING classificado como bloqueante para ML amplo."
        )
    elif raw_status == "ML_BLOCKED":
        gate = ML_GATE_BLOCKED
        usable = False
        reason = "Relatorio de qualidade marcou bloqueio para ML amplo."
    elif raw_status == "ML_NOT_APPLICABLE":
        gate = ML_GATE_NOT_APPLICABLE
        usable = False
        reason = "Serie nao aplicavel para treino ML."
    elif quality_status in {"PASS", "OK"} and audit_status in {"PASS", "OK"}:
        raw_status = "ML_READY"
        gate = ML_GATE_READY
        reason = "Sem status ML explicito; inferido como pronto por qualidade PASS."
    elif quality_status in {"FAIL", "ERROR"} or audit_status in {"FAIL", "ERROR"}:
        raw_status = "ML_BLOCKED"
        gate = ML_GATE_BLOCKED
        usable = False
        reason = "Sem status ML explicito; bloqueado por FAIL/ERROR de qualidade."
    else:
        raw_status = "ML_CAUTION"
        gate = ML_GATE_CAUTION_ACCEPTABLE
        usable = True
        reason = "Sem status ML explicito; WARNING mantido como cautela aceitavel."

    return {
        "quality_status": quality_status,
        "quality_ml_raw_status": raw_status,
        "quality_ml_gate": gate,
        "quality_ml_gate_reason": reason,
        "quality_ml_usable_for_broad_training": bool(usable and gate in ML_TRAINABLE_GATES),
        "quality_root_cause_code": root_code,
    }


def normalize_cost_config(cost_model: Dict[str, Any], source: str, symbol: str) -> Dict[str, float]:
    default = cost_model.get("default", {}) if isinstance(cost_model.get("default"), dict) else {}
    source_cfg = cost_model.get(str(source), {}) if isinstance(cost_model.get(str(source)), dict) else {}
    symbol_cfg = cost_model.get("symbols", {}).get(str(symbol), {}) if isinstance(cost_model.get("symbols"), dict) else {}

    merged: Dict[str, Any] = {**default, **source_cfg}
    if "slippage_bps_round_trip_override" in symbol_cfg:
        merged["slippage_bps_round_trip"] = symbol_cfg["slippage_bps_round_trip_override"]
    merged.update({k: v for k, v in symbol_cfg.items() if not k.endswith("_override")})

    return {
        "cost_fee_bps_round_trip": float(merged.get("fee_bps_round_trip", 20.0)),
        "cost_slippage_bps_round_trip": float(merged.get("slippage_bps_round_trip", 15.0)),
        "cost_funding_bps_per_day": float(merged.get("funding_bps_per_day", 0.0)),
        "cost_min_net_edge_bps": float(merged.get("min_net_edge_bps", 5.0)),
        "cost_spread_bps_one_way": float(merged.get("spread_bps_one_way", np.nan)),
        "cost_market_impact_bps_one_way": float(merged.get("market_impact_bps_one_way", np.nan)),
        "cost_liquidation_buffer_pct": float(merged.get("liquidation_buffer_pct", np.nan)),
    }


def datetime_dubai_naive_to_utc_ms(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt_utc = dt.dt.tz_convert("UTC")
        else:
            dt_utc = dt.dt.tz_localize(TIMEZONE_LOCAL).dt.tz_convert("UTC")
        values = dt_utc.astype("int64")
        median_abs = float(pd.to_numeric(values, errors="coerce").dropna().abs().median())
        if median_abs >= 1e17:
            return (values / 1_000_000).round().astype("Int64")
        if median_abs >= 1e14:
            return (values / 1_000).round().astype("Int64")
        if median_abs >= 1e11:
            return values.round().astype("Int64")
    except Exception:
        pass
    return pd.Series(pd.NA, index=series.index, dtype="Int64")


def ensure_timestamp_key(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = df.copy()
    if TIMESTAMP_COL in out.columns:
        out[TIMESTAMP_COL] = pd.to_numeric(out[TIMESTAMP_COL], errors="coerce").astype("Int64")
        if out[TIMESTAMP_COL].notna().mean() >= 0.90:
            return out

    if DATETIME_COL not in out.columns:
        raise ValueError(f"{TIMESTAMP_COL} e {DATETIME_COL} ausentes em {path}")

    out[DATETIME_COL] = pd.to_datetime(out[DATETIME_COL], errors="coerce")
    out[TIMESTAMP_COL] = datetime_dubai_naive_to_utc_ms(out[DATETIME_COL])
    return out


def get_ok_outputs(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    outputs = payload.get("series_outputs", [])
    if not isinstance(outputs, list):
        return []
    return [item for item in outputs if isinstance(item, dict) and item.get("status") == "OK"]


def pair_feature_label_outputs(
    features_json: Dict[str, Any],
    labels_json: Dict[str, Any],
) -> list[Dict[str, Any]]:
    feature_by_series = {
        str(item.get("series_id")): item
        for item in get_ok_outputs(features_json)
        if item.get("series_id") and item.get("output_path")
    }

    pairs = []
    for label in get_ok_outputs(labels_json):
        series_id = str(label.get("series_id") or "")
        feature = feature_by_series.get(series_id)
        if not feature:
            continue
        pairs.append({"feature": feature, "label": label})
    return pairs


def select_columns(path: Path, prefixes: Iterable[str], extras: Iterable[str]) -> list[str]:
    import pyarrow.parquet as pq

    columns = [str(column) for column in pq.ParquetFile(path).schema.names]
    selected = [column for column in columns if column in extras or column.startswith(tuple(prefixes))]
    return list(dict.fromkeys(selected))


def read_features(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    columns = select_columns(path, FEATURE_PREFIXES, META_COLUMNS | QUALITY_COLUMNS | {"Open", "High", "Low", "Close", "Volume"})
    df = pd.read_parquet(path, columns=columns, engine=PARQUET_ENGINE)
    feature_cols = [col for col in df.columns if col.startswith(FEATURE_PREFIXES)]
    return df, feature_cols, columns


def read_labels(path: Path) -> tuple[pd.DataFrame, list[str]]:
    columns = select_columns(path, ("label_",), {DATETIME_COL, TIMESTAMP_COL})
    df = pd.read_parquet(path, columns=columns, engine=PARQUET_ENGINE)
    label_cols = [col for col in df.columns if col.startswith("label_")]
    return df, label_cols


def read_btc_xasset(
    feature_index: Dict[tuple[str, str], Dict[str, Any]],
    source: str,
    timeframe: str,
    target_asset: str,
) -> Optional[pd.DataFrame]:
    if target_asset == "BTC":
        return None
    btc = feature_index.get((source, timeframe))
    if not btc:
        return None

    path = Path(str(btc.get("output_path")))
    if not path.exists():
        return None

    available = select_columns(path, (), {TIMESTAMP_COL, *BTC_XASSET_COLUMNS})
    cols = [col for col in available if col == TIMESTAMP_COL or col in BTC_XASSET_COLUMNS]
    if len(cols) <= 1:
        return None

    df = pd.read_parquet(path, columns=cols, engine=PARQUET_ENGINE)
    rename = {
        col: f"xasset_BTCUSDT_{col.replace('feat_', '')}"
        for col in df.columns
        if col != TIMESTAMP_COL
    }
    return df.rename(columns=rename)


def build_output_path(source: str, asset: str, symbol: str, timeframe: str, series_id: str) -> Path:
    out_dir = DATASETS_PARQUET_DIR / safe_token(source) / safe_token(asset) / safe_token(timeframe)
    ensure_dir(out_dir)
    digest = short_hash(series_id, source, asset, symbol, timeframe)
    return out_dir / f"dataset_{safe_token(symbol)}_{safe_token(source)}_{safe_token(timeframe)}_{digest}.parquet"


def add_ml_controls(
    df: pd.DataFrame,
    label_cols: list[str],
    target_col: str,
    quality_status: str,
    quality_ml_raw_status: str = "ML_CAUTION",
    quality_ml_gate: str = ML_GATE_CAUTION_ACCEPTABLE,
    quality_ml_gate_reason: str = "",
    quality_ml_usable_for_broad_training: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if "quality_ml_row_eligible" in out.columns:
        quality_ok = out["quality_ml_row_eligible"].astype(bool)
    elif "quality_is_feature_valid" in out.columns:
        quality_ok = out["quality_is_feature_valid"].astype(bool)
    else:
        quality_ok = pd.Series(True, index=out.index)
    trainable_gate = quality_ml_gate in ML_TRAINABLE_GATES
    quality_ok = quality_ok & bool(quality_ml_usable_for_broad_training) & trainable_gate

    if target_col in out.columns:
        target_ok = out[target_col].notna()
    else:
        target_ok = out[label_cols].notna().any(axis=1) if label_cols else pd.Series(False, index=out.index)

    out["sample_weight"] = np.where(quality_ok & target_ok, 1.0, 0.0).astype("float32")
    out["is_trainable"] = (quality_ok & target_ok).astype("bool")
    out["data_quality_flag"] = str(quality_status or "UNKNOWN")
    out["quality_ml_raw_status"] = str(quality_ml_raw_status or "ML_CAUTION")
    out["quality_ml_gate"] = str(quality_ml_gate or ML_GATE_CAUTION_ACCEPTABLE)
    out["quality_ml_gate_reason"] = str(quality_ml_gate_reason or "")
    out["quality_ml_usable_for_broad_training"] = bool(quality_ml_usable_for_broad_training)
    out["meta_dataset_schema_version"] = SCHEMA_VERSION
    out["meta_dataset_run_id"] = RUN_ID
    out["meta_dataset_generated_at_utc"] = now_utc_iso()
    out["meta_target_default"] = target_col if target_col in out.columns else None
    return out


def build_one_dataset(
    pair: Dict[str, Any],
    feature_index: Dict[tuple[str, str], Dict[str, Any]],
    quality_index: Dict[str, Dict[str, Any]],
    cost_model: Dict[str, Any],
    target_col: str,
) -> Dict[str, Any]:
    feature = pair["feature"]
    label = pair["label"]

    source = str(feature.get("source") or label.get("source") or "unknown")
    asset = str(feature.get("asset") or label.get("asset") or "unknown")
    symbol = str(feature.get("symbol") or label.get("symbol") or asset)
    timeframe = str(feature.get("timeframe") or label.get("timeframe") or "unknown")
    series_id = str(feature.get("series_id") or label.get("series_id") or "")

    feature_path = Path(str(feature.get("output_path")))
    label_path = Path(str(label.get("output_path")))
    output_path = build_output_path(source, asset, symbol, timeframe, series_id)

    started = time.time()
    result: Dict[str, Any] = {
        "status": "PENDING",
        "series_id": series_id,
        "asset": asset,
        "symbol": symbol,
        "source": source,
        "timeframe": timeframe,
        "feature_path": str(feature_path),
        "label_path": str(label_path),
        "output_path": str(output_path),
    }

    try:
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature parquet nao encontrado: {feature_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label parquet nao encontrado: {label_path}")

        features_df, feature_cols, feature_columns_read = read_features(feature_path)
        labels_df, label_cols = read_labels(label_path)

        features_df = ensure_timestamp_key(features_df, feature_path)
        labels_df = ensure_timestamp_key(labels_df, label_path)
        features_df = features_df.dropna(subset=[TIMESTAMP_COL]).drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
        labels_df = labels_df.dropna(subset=[TIMESTAMP_COL]).drop_duplicates(subset=[TIMESTAMP_COL], keep="last")

        dataset = features_df.merge(labels_df.drop(columns=[DATETIME_COL], errors="ignore"), on=TIMESTAMP_COL, how="inner")

        btc_xasset = read_btc_xasset(feature_index, source, timeframe, asset)
        xasset_cols: list[str] = []
        if btc_xasset is not None:
            btc_xasset[TIMESTAMP_COL] = pd.to_numeric(btc_xasset[TIMESTAMP_COL], errors="coerce").astype("Int64")
            dataset = dataset.merge(btc_xasset.dropna(subset=[TIMESTAMP_COL]), on=TIMESTAMP_COL, how="left")
            xasset_cols = [col for col in btc_xasset.columns if col.startswith("xasset_")]

        for col, value in normalize_cost_config(cost_model, source, symbol).items():
            dataset[col] = value

        ml_gate = classify_ml_gate(feature, quality_index.get(series_id, {}))
        dataset = add_ml_controls(
            dataset,
            label_cols,
            target_col,
            ml_gate["quality_status"],
            ml_gate["quality_ml_raw_status"],
            ml_gate["quality_ml_gate"],
            ml_gate["quality_ml_gate_reason"],
            ml_gate["quality_ml_usable_for_broad_training"],
        )

        allowed_feature_cols = [
            col for col in dataset.columns
            if col.startswith(("feat_", "regime_", "xasset_", "cost_"))
        ]

        forbidden_as_features = [
            col for col in dataset.columns
            if col.startswith(("label_", "meta_", "quality_")) or col in {DATETIME_COL, TIMESTAMP_COL}
        ]

        dataset = dataset.sort_values(TIMESTAMP_COL, kind="mergesort").reset_index(drop=True)
        dataset.to_parquet(output_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)

        result.update({
            "status": "OK",
            "rows": int(len(dataset)),
            "columns": int(len(dataset.columns)),
            "feature_columns_count": int(len(feature_cols)),
            "xasset_columns_count": int(len(xasset_cols)),
            "label_columns_count": int(len(label_cols)),
            "allowed_feature_columns_count": int(len(allowed_feature_cols)),
            "target_default": target_col if target_col in dataset.columns else None,
            "trainable_rows": int(dataset["is_trainable"].sum()),
            "quality_status": ml_gate["quality_status"],
            "quality_ml_raw_status": ml_gate["quality_ml_raw_status"],
            "quality_ml_gate": ml_gate["quality_ml_gate"],
            "quality_ml_gate_reason": ml_gate["quality_ml_gate_reason"],
            "quality_ml_usable_for_broad_training": ml_gate["quality_ml_usable_for_broad_training"],
            "quality_root_cause_code": ml_gate["quality_root_cause_code"],
            "first_timestamp_utc_ms": int(dataset[TIMESTAMP_COL].min()) if not dataset.empty else None,
            "last_timestamp_utc_ms": int(dataset[TIMESTAMP_COL].max()) if not dataset.empty else None,
            "allowed_feature_columns": allowed_feature_cols,
            "forbidden_as_features": forbidden_as_features,
            "feature_columns_read": feature_columns_read,
        })

    except Exception as exc:
        result.update({"status": "ERROR", "error": str(exc)})

    result["elapsed_seconds"] = round(time.time() - started, 6)
    return result


def build_feature_index(features_json: Dict[str, Any]) -> Dict[tuple[str, str], Dict[str, Any]]:
    index = {}
    for item in get_ok_outputs(features_json):
        if item.get("asset") == "BTC":
            index[(str(item.get("source")), str(item.get("timeframe")))] = item
    return index


def count_by(items: list[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Monta datasets ML ARCHANGEL a partir de features e labels.")
    parser.add_argument("--assets", help="CSV de assets, use all para todos.")
    parser.add_argument("--timeframes", help="CSV de timeframes, use all para todos.")
    parser.add_argument("--sources", help="CSV de sources, use all para todos.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target default registrado no dataset.")
    parser.add_argument("--limit", type=int, help="Limita quantidade de datasets para smoke tests.")
    args = parser.parse_args()

    ensure_dir(DATASETS_PARQUET_DIR)
    ensure_dir(DATASETS_LOG_DIR)

    features_json = load_json(FEATURES_JSON_PATH)
    labels_json = load_json(LABELS_JSON_PATH)
    cost_model = load_json(COST_MODEL_PATH, required=False)
    quality_report = load_json(DATA_QUALITY_REPORT_PATH, required=False)

    assets = parse_csv_filter(args.assets, DEFAULT_ASSETS)
    timeframes = parse_csv_filter(args.timeframes, DEFAULT_TIMEFRAMES)
    sources = parse_csv_filter(args.sources, DEFAULT_SOURCES)

    pairs = pair_feature_label_outputs(features_json, labels_json)
    selected = []
    for pair in pairs:
        feature = pair["feature"]
        if assets is not None and str(feature.get("asset")) not in assets:
            continue
        if timeframes is not None and str(feature.get("timeframe")) not in timeframes:
            continue
        if sources is not None and str(feature.get("source")) not in sources:
            continue
        selected.append(pair)

    selected = sorted(
        selected,
        key=lambda p: (
            str(p["feature"].get("source")),
            str(p["feature"].get("asset")),
            str(p["feature"].get("timeframe")),
            str(p["feature"].get("series_id")),
        ),
    )

    if args.limit is not None:
        selected = selected[: max(0, args.limit)]

    feature_index = build_feature_index(features_json)
    quality_index = build_quality_index(quality_report)

    print("=" * 100)
    print("ARCHANGEL v1 | 5_DATASETS_ML | MONTAGEM FEATURES + LABELS")
    print("=" * 100)
    print(f"[RUN_ID] {RUN_ID}")
    print(f"[FEATURES_JSON] {FEATURES_JSON_PATH}")
    print(f"[LABELS_JSON] {LABELS_JSON_PATH}")
    print(f"[COST_MODEL] {COST_MODEL_PATH}")
    print(f"[DATA_QUALITY_REPORT_EXISTS] {DATA_QUALITY_REPORT_PATH.exists()}")
    print(f"[QUALITY_REPORT_STATUS] {quality_report.get('status') if isinstance(quality_report, dict) else None}")
    print(f"[SELECTED] {len(selected)}")
    print(f"[ASSETS] {None if assets is None else sorted(assets)}")
    print(f"[TIMEFRAMES] {None if timeframes is None else sorted(timeframes)}")
    print(f"[SOURCES] {None if sources is None else sorted(sources)}")
    print("=" * 100)

    results = []
    for idx, pair in enumerate(selected, start=1):
        result = build_one_dataset(pair, feature_index, quality_index, cost_model, args.target)
        results.append(result)
        print(
            f"[PROGRESSO] {idx}/{len(selected)} | {result.get('status')} | "
            f"{result.get('source')} | {result.get('asset')} | {result.get('timeframe')} | "
            f"ml_gate={result.get('quality_ml_gate')} | "
            f"rows={result.get('rows')} | trainable={result.get('trainable_rows')} | "
            f"{result.get('elapsed_seconds')}s"
            + (f" | error={result.get('error')}" if result.get("error") else "")
        )

    ok = [item for item in results if item.get("status") == "OK"]
    errors = [item for item in results if item.get("status") != "OK"]
    run_report_path = DATASETS_LOG_DIR / f"5_DATASETS_ML_RUN_REPORT_{RUN_ID}.json"
    run_report_base_json_path = RUN_REPORT_LATEST_PATH

    payload = {
        "schema_version": SCHEMA_VERSION,
        "system": {
            "name": "ARCHANGEL",
            "version": "v1",
            "layer": "5_DATASETS_ML",
            "script": SCRIPT_NAME,
            "run_id": RUN_ID,
            "generated_at_utc": now_utc_iso(),
        },
        "paths": {
            "root_dir": str(ROOT_DIR),
            "features_json_path": str(FEATURES_JSON_PATH),
            "labels_json_path": str(LABELS_JSON_PATH),
            "cost_model_path": str(COST_MODEL_PATH),
            "data_quality_report_path": str(DATA_QUALITY_REPORT_PATH),
            "datasets_parquet_dir": str(DATASETS_PARQUET_DIR),
            "datasets_json_path": str(DATASETS_JSON_PATH),
            "run_report_path": str(run_report_path),
            "run_report_base_json_path": str(run_report_base_json_path),
            "run_report_latest_path": str(RUN_REPORT_LATEST_PATH),
        },
        "policy": {
            "join_key": TIMESTAMP_COL,
            "default_assets": sorted(DEFAULT_ASSETS),
            "default_timeframes": sorted(DEFAULT_TIMEFRAMES),
            "target_default_requested": args.target,
            "model_allowed_feature_prefixes": ["feat_", "regime_", "xasset_", "cost_"],
            "model_forbidden_prefixes": ["label_", "meta_", "quality_"],
            "anti_leakage_rule": "Modelos devem usar apenas allowed_feature_columns; label_ e meta_ nunca entram como features.",
            "cross_asset_rule": "xasset_* usa dados alinhados no mesmo timestamp_utc_ms, sem merge futuro.",
            "ml_quality_gate_rule": (
                "ML_READY e ML_CAUTION_ACCEPTABLE entram no treino; "
                "ML_BLOCKED e ML_NOT_APPLICABLE recebem is_trainable=False."
            ),
        },
        "summary": {
            "pairs_available": len(pairs),
            "pairs_selected": len(selected),
            "datasets_ok": len(ok),
            "datasets_error": len(errors),
            "total_rows": int(sum(item.get("rows", 0) or 0 for item in ok)),
            "total_trainable_rows": int(sum(item.get("trainable_rows", 0) or 0 for item in ok)),
            "ml_gate_counts": count_by(ok, "quality_ml_gate"),
            "ml_raw_status_counts": count_by(ok, "quality_ml_raw_status"),
            "ml_root_cause_counts": count_by(ok, "quality_root_cause_code"),
            "datasets_trainable_for_broad_ml": int(
                sum(1 for item in ok if item.get("quality_ml_gate") in ML_TRAINABLE_GATES)
            ),
            "datasets_blocked_for_broad_ml": int(
                sum(1 for item in ok if item.get("quality_ml_gate") == ML_GATE_BLOCKED)
            ),
        },
        "datasets": results,
    }

    write_json_atomic(run_report_path, payload)
    write_json_atomic(run_report_base_json_path, payload)
    write_json_atomic(RUN_REPORT_LATEST_PATH, payload)
    write_json_atomic(DATASETS_JSON_PATH, payload)

    print("=" * 100)
    print(f"[OK] {len(ok)}")
    print(f"[ERROS] {len(errors)}")
    print(f"[JSON DATASETS] {DATASETS_JSON_PATH}")
    print(f"[RUN REPORT] {run_report_path}")
    print("=" * 100)

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
