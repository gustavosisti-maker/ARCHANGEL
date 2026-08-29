# -*- coding: utf-8 -*-
"""Auditoria formal de qualidade da camada de dados ARCHANGEL.

Lê o MAPA_ATIVOS.json, valida os Parquets sem alterá-los e grava um contrato de
qualidade consumido pela etapa de features em BASE_JSON.
"""

from __future__ import annotations

import json
import os
import gc
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_JSON_DIR = PROJECT_ROOT / "0_REGRAS_MANDATO" / "BASE_JSON"
MAPA_ATIVOS_PATH = BASE_JSON_DIR / "MAPA_ATIVOS.json"
QUALITY_DIR = BASE_JSON_DIR
QUALITY_REPORT_PATH = QUALITY_DIR / "DATA_QUALITY_REPORT.json"
QUALITY_ROOT_CAUSE_REPORT_PATH = QUALITY_DIR / "DATA_QUALITY_ROOT_CAUSE_REPORT.json"
QUALITY_ROOT_CAUSE_CSV_PATH = PROJECT_ROOT / "2_BASES" / "_quality" / "DATA_QUALITY_ROOT_CAUSE_TRIAGE.csv"
QUALITY_CHUNKS_DIR = BASE_JSON_DIR / "DATA_QUALITY_CHUNKS"

SCHEMA_VERSION = "ARCHANGEL_DATA_QUALITY_REPORT_1.1_ROOT_CAUSE_ML_POLICY"
TIMEZONE_LOCAL = "Asia/Dubai"
DATETIME_COL = "DateTime"
TIMESTAMP_UTC_MS_COL = "timestamp_utc_ms"
PARQUET_ENGINE = "pyarrow"

MIN_ROWS_WARNING = 300
ML_MIN_ROWS_BY_KIND = {
    "ohlcv": 300,
    "funding_rate": 100,
    "open_interest": 100,
}
MAX_TIMESTAMP_DATETIME_DRIFT_MS = 1_000
GAP_WARNING_RATIO = 0.0
GAP_FAIL_RATIO = 0.01
ML_GAP_BLOCK_RATIO = 0.001
RETURN_OUTLIER_ABS_PCT = 0.50
ML_RETURN_OUTLIER_BLOCK_RATIO = 0.001
BATCH_SIZE_ROWS = 250_000


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_timestamp_to_utc_ms(series: pd.Series) -> pd.Series:
    """Normaliza timestamps Unix em s/ms/us/ns para UTC em milissegundos."""
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return pd.Series(pd.NA, index=series.index, dtype="Int64")

    median_abs = float(values.dropna().abs().median())
    if median_abs >= 1e17:
        normalized = (values / 1_000_000).round()
    elif median_abs >= 1e14:
        normalized = (values / 1_000).round()
    elif median_abs >= 1e11:
        normalized = values.round()
    elif median_abs >= 1e8:
        normalized = (values * 1_000).round()
    else:
        normalized = pd.Series(pd.NA, index=series.index)
    return normalized.astype("Int64")


def datetime_dubai_naive_to_utc_ms(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt_utc = dt.dt.tz_convert("UTC")
        else:
            dt_utc = dt.dt.tz_localize(TIMEZONE_LOCAL).dt.tz_convert("UTC")
        return normalize_timestamp_to_utc_ms(dt_utc.astype("int64"))
    except Exception:
        return pd.Series(pd.NA, index=series.index, dtype="Int64")


def infer_timeframe_ms(series: Dict[str, Any]) -> Optional[int]:
    """Infere o timeframe da estrutura física, que é mais confiável que gaps."""
    relative_path = str(series.get("file", {}).get("relative_path", "")).lower()
    name = str(series.get("file", {}).get("name", "")).lower()
    text = f"{relative_path} {name}".replace("_", " ")

    aliases = {
        "1 min": 60_000,
        "3 min": 180_000,
        "5 min": 300_000,
        "7 min": 420_000,
        "13 min": 780_000,
        "15 min": 900_000,
        "23 min": 1_380_000,
        "37 min": 2_220_000,
        "47 min": 2_820_000,
        "1 hour": 3_600_000,
        "4 hour": 14_400_000,
        "1 day": 86_400_000,
        "3 day": 259_200_000,
        "7 day": 604_800_000,
    }
    for label, value in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if label in text:
            return value

    timeframe = str(series.get("timeframe") or "")
    if timeframe.endswith("min") and timeframe[:-3].isdigit():
        return int(timeframe[:-3]) * 60_000
    if timeframe.endswith("h") and timeframe[:-1].isdigit():
        return int(timeframe[:-1]) * 3_600_000
    if timeframe.endswith("D") and timeframe[:-1].isdigit():
        return int(timeframe[:-1]) * 86_400_000
    return None


def ohlcv_column_map(columns: list[str]) -> Dict[str, str]:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    aliases = {
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c"),
        "volume": ("volume", "vol", "base_volume"),
    }
    return {
        canonical: next((normalized[name] for name in names if name in normalized), "")
        for canonical, names in aliases.items()
    }


def classify(errors: list[str], warnings: list[str]) -> str:
    if errors:
        return "FAIL"
    if warnings:
        return "WARNING"
    return "PASS"


def infer_root_cause(record: Dict[str, Any]) -> Dict[str, Any]:
    errors = [str(item) for item in record.get("errors", [])]
    warnings = [str(item) for item in record.get("warnings", [])]
    checks = record.get("checks", {}) if isinstance(record.get("checks"), dict) else {}
    text = " | ".join(errors + warnings).lower()

    if "não existe" in text:
        code = "MISSING_FILE"
        family = "filesystem"
        action = "Remover a série do manifesto ou reexecutar a ingestão para recriar o Parquet."
    elif "falha ao ler parquet" in text or "falha ao abrir parquet" in text:
        code = "PARQUET_READ_ERROR"
        family = "storage_format"
        action = "Regerar o Parquet a partir da fonte bruta."
    elif "parquet vazio" in text:
        code = "EMPTY_FILE"
        family = "data_volume"
        action = "Rebaixar a série para inelegível e reexecutar ingestão quando houver dados."
    elif "colunas ohlcv ausentes" in text:
        code = "OHLCV_SCHEMA_MISSING_COLUMNS"
        family = "schema"
        action = "Padronizar colunas OHLCV antes de liberar features."
    elif "timestamp_utc_ms" in text or "datetime" in text:
        code = "TIME_INDEX_CONTRACT"
        family = "time_index"
        action = "Reconstruir timestamp_utc_ms/DateTime e reauditar a série."
    elif "gaps" in text:
        code = "TIME_GAPS"
        family = "time_grid"
        action = "Reingestar a janela faltante ou manter bloqueada para ML amplo se o gap for material."
    elif "ohlc inválido" in text or "volume negativo" in text or "valores nulos" in text:
        code = "OHLCV_VALUE_INTEGRITY"
        family = "market_data_values"
        action = "Corrigir candles inválidos na origem ou excluir a janela comprometida."
    elif "histórico curto" in text:
        code = "SHORT_HISTORY"
        family = "data_volume"
        action = "Usar apenas para análise exploratória até acumular histórico suficiente."
    elif "retornos absolutos" in text:
        code = "RETURN_OUTLIERS"
        family = "market_data_values"
        action = "Inspecionar candles extremos e decidir entre correção, winsorização ou bloqueio."
    else:
        code = "NO_ISSUE_DETECTED" if record.get("status") == "PASS" else "UNCLASSIFIED"
        family = "none" if record.get("status") == "PASS" else "unknown"
        action = "Sem ação." if record.get("status") == "PASS" else "Inspecionar manualmente."

    return {
        "code": code,
        "family": family,
        "action": action,
        "gap_ratio": checks.get("gap_ratio"),
        "gap_count": checks.get("gap_count"),
        "rows": checks.get("rows"),
    }


def classify_warning_for_ml(warning: str, record: Dict[str, Any]) -> Dict[str, Any]:
    checks = record.get("checks", {}) if isinstance(record.get("checks"), dict) else {}
    dataset_kind = str(record.get("dataset_kind") or "unknown")
    rows = int(checks.get("rows") or 0)
    gap_ratio = checks.get("gap_ratio")
    return_outliers = int(checks.get("return_outlier_count") or 0)
    warning_lower = str(warning).lower()

    detail = {
        "warning": str(warning),
        "code": "WARNING_UNCLASSIFIED",
        "ml_status": "ML_CAUTION",
        "ml_blocking": False,
        "reason": "Alerta preservado para revisão, mas sem bloqueio automático definido.",
        "recommended_action": "Revisar manualmente antes de usar em experimentos amplos.",
    }

    if dataset_kind not in {"ohlcv", "funding_rate", "open_interest"}:
        detail.update({
            "code": "NON_MARKET_SERIES_NOT_APPLICABLE",
            "ml_status": "ML_NOT_APPLICABLE",
            "ml_blocking": False,
            "reason": "Série não é insumo temporal de mercado usado como base direta de ML.",
            "recommended_action": "Ignorar no gate de ML ou mapear explicitamente se virar feature auxiliar.",
        })
    elif "histórico curto" in warning_lower:
        min_rows = int(ML_MIN_ROWS_BY_KIND.get(dataset_kind, MIN_ROWS_WARNING))
        blocking = rows < min_rows
        detail.update({
            "code": "SHORT_HISTORY",
            "ml_status": "ML_BLOCKED" if blocking else "ML_CAUTION",
            "ml_blocking": blocking,
            "reason": f"{rows} linhas disponíveis; mínimo ML para {dataset_kind}: {min_rows}.",
            "recommended_action": "Bloquear treinamento amplo até acumular histórico ou reduzir escopo do experimento.",
        })
    elif "gaps temporais" in warning_lower:
        ratio = float(gap_ratio or 0.0)
        blocking = ratio > ML_GAP_BLOCK_RATIO
        detail.update({
            "code": "TIME_GAPS_MATERIAL" if blocking else "TIME_GAPS_SMALL",
            "ml_status": "ML_BLOCKED" if blocking else "ML_CAUTION",
            "ml_blocking": blocking,
            "reason": f"gap_ratio={ratio:.6f}; limite bloqueante ML={ML_GAP_BLOCK_RATIO:.6f}.",
            "recommended_action": (
                "Reingestar/preencher janela faltante antes de ML amplo."
                if blocking
                else "Aceitável com cautela; manter purging/embargo e registrar no experimento."
            ),
        })
    elif "retornos absolutos" in warning_lower:
        ratio = float(return_outliers / max(1, rows))
        blocking = ratio > ML_RETURN_OUTLIER_BLOCK_RATIO
        detail.update({
            "code": "RETURN_OUTLIERS_MATERIAL" if blocking else "RETURN_OUTLIERS_SMALL",
            "ml_status": "ML_BLOCKED" if blocking else "ML_CAUTION",
            "ml_blocking": blocking,
            "reason": f"return_outlier_ratio={ratio:.6f}; limite bloqueante ML={ML_RETURN_OUTLIER_BLOCK_RATIO:.6f}.",
            "recommended_action": (
                "Inspecionar candles extremos e corrigir/winsorizar antes de ML amplo."
                if blocking
                else "Aceitável com cautela após inspeção pontual dos candles extremos."
            ),
        })
    elif "datetime ausente" in warning_lower or "timestamp_utc_ms ausente" in warning_lower:
        blocking = dataset_kind == "ohlcv"
        detail.update({
            "code": "TIME_COLUMNS_MISSING",
            "ml_status": "ML_BLOCKED" if blocking else "ML_NOT_APPLICABLE",
            "ml_blocking": blocking,
            "reason": "Colunas temporais são obrigatórias para OHLCV e opcionais para séries não usadas como base.",
            "recommended_action": "Reconstruir as colunas temporais se a série for usada em features/ML.",
        })
    elif "nenhuma coluna temporal" in warning_lower:
        detail.update({
            "code": "NO_AUDITABLE_COLUMNS",
            "ml_status": "ML_NOT_APPLICABLE",
            "ml_blocking": False,
            "reason": "Não há colunas suficientes para auditoria temporal detalhada.",
            "recommended_action": "Ignorar no ML ou remodelar o arquivo se ele precisar virar série temporal.",
        })

    return detail


def enrich_quality_record_for_ml(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    errors = out.get("errors") if isinstance(out.get("errors"), list) else []
    warnings = out.get("warnings") if isinstance(out.get("warnings"), list) else []

    out["root_cause"] = infer_root_cause(out)
    warning_details = [classify_warning_for_ml(str(warning), out) for warning in warnings]
    out["warning_ml_classification"] = warning_details
    out["ml_blocking_reasons"] = [
        item for item in warning_details if item.get("ml_blocking")
    ]

    if errors:
        out["ml_quality_status"] = "ML_BLOCKED"
        out["ml_quality_reason"] = "Série com status FAIL na auditoria de dados."
    elif out.get("status") == "PASS":
        out["ml_quality_status"] = "ML_READY"
        out["ml_quality_reason"] = "Série sem erros nem warnings."
    elif out["ml_blocking_reasons"]:
        out["ml_quality_status"] = "ML_BLOCKED"
        out["ml_quality_reason"] = "WARNING classificado como bloqueante para ML amplo."
    elif any(item.get("ml_status") == "ML_NOT_APPLICABLE" for item in warning_details):
        out["ml_quality_status"] = "ML_NOT_APPLICABLE"
        out["ml_quality_reason"] = "Série fora do escopo direto do gate de ML."
    else:
        out["ml_quality_status"] = "ML_CAUTION"
        out["ml_quality_reason"] = "Warnings aceitáveis com registro e cautela experimental."

    out["ml_usable_for_broad_training"] = out["ml_quality_status"] in {"ML_READY", "ML_CAUTION"}
    return out


def audit_series(series: Dict[str, Any]) -> Dict[str, Any]:
    series_id = str(series.get("series_id") or "")
    dataset_kind = str(series.get("dataset_kind") or "unknown")
    path = Path(str(series.get("file", {}).get("absolute_path") or ""))
    errors: list[str] = []
    warnings: list[str] = []
    checks: Dict[str, Any] = {}

    record: Dict[str, Any] = {
        "series_id": series_id,
        "dataset_kind": dataset_kind,
        "source": series.get("source"),
        "asset": series.get("asset"),
        "symbol": series.get("symbol"),
        "timeframe": series.get("timeframe"),
        "path": str(path),
        "status": "FAIL",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }

    if not path.is_file():
        errors.append("Arquivo listado no MAPA_ATIVOS não existe.")
        return record

    try:
        df = pd.read_parquet(path, engine=PARQUET_ENGINE)
    except Exception as exc:
        errors.append(f"Falha ao ler Parquet: {exc}")
        return record

    checks["rows"] = int(len(df))
    checks["columns"] = [str(column) for column in df.columns]
    checks["column_count"] = int(len(df.columns))

    if df.empty:
        errors.append("Parquet vazio.")
        return record

    if len(df) < MIN_ROWS_WARNING:
        warnings.append(f"Histórico curto: {len(df)} linhas < {MIN_ROWS_WARNING}.")

    datetime_present = DATETIME_COL in df.columns
    timestamp_present = TIMESTAMP_UTC_MS_COL in df.columns
    checks["datetime_present"] = datetime_present
    checks["timestamp_utc_ms_present"] = timestamp_present

    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[us]")
    ts = pd.Series(pd.NA, index=df.index, dtype="Int64")

    if datetime_present:
        dt = pd.to_datetime(df[DATETIME_COL], errors="coerce")
        checks["datetime_null_count"] = int(dt.isna().sum())
        checks["datetime_duplicate_count"] = int(dt.duplicated().sum())
        checks["datetime_monotonic_increasing"] = bool(dt.dropna().is_monotonic_increasing)
        if dt.isna().any():
            errors.append("DateTime contém valores nulos ou inválidos.")
        if dt.duplicated().any():
            errors.append("DateTime contém duplicatas.")
        if not checks["datetime_monotonic_increasing"]:
            errors.append("DateTime não está em ordem crescente.")
    else:
        warnings.append("DateTime ausente.")

    if timestamp_present:
        ts = normalize_timestamp_to_utc_ms(df[TIMESTAMP_UTC_MS_COL])
        checks["timestamp_null_count"] = int(ts.isna().sum())
        checks["timestamp_duplicate_count"] = int(ts.duplicated().sum())
        checks["timestamp_monotonic_increasing"] = bool(ts.dropna().is_monotonic_increasing)
        if ts.isna().any():
            errors.append("timestamp_utc_ms contém valores nulos ou fora de escala.")
        if ts.duplicated().any():
            errors.append("timestamp_utc_ms contém duplicatas.")
        if not checks["timestamp_monotonic_increasing"]:
            errors.append("timestamp_utc_ms não está em ordem crescente.")
    elif dataset_kind == "ohlcv":
        errors.append("OHLCV sem timestamp_utc_ms.")
    else:
        warnings.append("timestamp_utc_ms ausente.")

    if datetime_present and timestamp_present:
        dt_ms = datetime_dubai_naive_to_utc_ms(df[DATETIME_COL])
        comparable = pd.DataFrame({"datetime_ms": dt_ms, "timestamp_ms": ts}).dropna()
        if comparable.empty:
            errors.append("Não há timestamps válidos para comparar DateTime e timestamp_utc_ms.")
        else:
            drift = (comparable["datetime_ms"].astype("int64") - comparable["timestamp_ms"].astype("int64")).abs()
            checks["timezone_max_abs_drift_ms"] = int(drift.max())
            checks["timezone_mean_abs_drift_ms"] = float(drift.mean())
            if int(drift.max()) > MAX_TIMESTAMP_DATETIME_DRIFT_MS:
                errors.append("DateTime Dubai e timestamp_utc_ms divergem além da tolerância.")

    expected_ms = infer_timeframe_ms(series)
    checks["expected_timeframe_ms"] = expected_ms
    if timestamp_present and expected_ms and ts.notna().sum() >= 2:
        deltas = ts.dropna().astype("int64").sort_values().diff().dropna()
        if not deltas.empty:
            tolerance = max(1_000, int(expected_ms * 0.001))
            irregular = (deltas - expected_ms).abs() > tolerance
            gaps = deltas > expected_ms * 1.5
            checks["median_delta_ms"] = float(deltas.median())
            checks["max_delta_ms"] = int(deltas.max())
            checks["irregular_delta_count"] = int(irregular.sum())
            checks["gap_count"] = int(gaps.sum())
            checks["gap_ratio"] = float(gaps.mean())
            if float(gaps.mean()) > GAP_FAIL_RATIO:
                errors.append("Proporção de gaps acima do limite de falha.")
            elif float(gaps.mean()) > GAP_WARNING_RATIO:
                warnings.append("Gaps temporais detectados.")

    if dataset_kind == "ohlcv":
        colmap = ohlcv_column_map(checks["columns"])
        missing = [name for name, column in colmap.items() if not column]
        checks["ohlcv_missing_columns"] = missing
        if missing:
            errors.append(f"Colunas OHLCV ausentes: {missing}")
        else:
            numeric = df[[colmap[name] for name in ("open", "high", "low", "close", "volume")]].apply(
                pd.to_numeric, errors="coerce"
            )
            numeric.columns = ["open", "high", "low", "close", "volume"]
            checks["ohlcv_null_count"] = int(numeric.isna().sum().sum())
            invalid = (
                (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
                | (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1))
                | (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1))
            )
            checks["invalid_ohlc_count"] = int(invalid.sum())
            checks["negative_volume_count"] = int((numeric["volume"] < 0).sum())
            if checks["ohlcv_null_count"]:
                errors.append("OHLCV contém valores nulos ou não numéricos.")
            if checks["invalid_ohlc_count"]:
                errors.append("OHLC inválido: preços não positivos ou fora do intervalo High/Low.")
            if checks["negative_volume_count"]:
                errors.append("Volume negativo detectado.")

            returns = numeric["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            outliers = returns.abs() > RETURN_OUTLIER_ABS_PCT
            checks["return_outlier_count"] = int(outliers.sum())
            if checks["return_outlier_count"]:
                warnings.append(
                    f"Retornos absolutos acima de {RETURN_OUTLIER_ABS_PCT:.0%} detectados."
                )

    if datetime_present:
        valid_dt = dt.dropna()
        if not valid_dt.empty:
            checks["start"] = str(valid_dt.min())
            checks["end"] = str(valid_dt.max())

    record["status"] = classify(errors, warnings)
    return record


def audit_series_streaming(series: Dict[str, Any]) -> Dict[str, Any]:
    """Audita um Parquet em lotes para limitar o consumo de memória."""
    series_id = str(series.get("series_id") or "")
    dataset_kind = str(series.get("dataset_kind") or "unknown")
    path = Path(str(series.get("file", {}).get("absolute_path") or ""))
    errors: list[str] = []
    warnings: list[str] = []
    checks: Dict[str, Any] = {}
    record: Dict[str, Any] = {
        "series_id": series_id,
        "dataset_kind": dataset_kind,
        "source": series.get("source"),
        "asset": series.get("asset"),
        "symbol": series.get("symbol"),
        "timeframe": series.get("timeframe"),
        "path": str(path),
        "status": "FAIL",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }

    if not path.is_file():
        errors.append("Arquivo listado no MAPA_ATIVOS não existe.")
        return record

    try:
        parquet = pq.ParquetFile(path)
        columns = [str(column) for column in parquet.schema_arrow.names]
    except Exception as exc:
        errors.append(f"Falha ao abrir Parquet: {exc}")
        return record

    checks["rows"] = int(parquet.metadata.num_rows)
    checks["columns"] = columns
    checks["column_count"] = len(columns)
    if checks["rows"] == 0:
        errors.append("Parquet vazio.")
        return record
    if checks["rows"] < MIN_ROWS_WARNING:
        warnings.append(f"Histórico curto: {checks['rows']} linhas < {MIN_ROWS_WARNING}.")

    datetime_present = DATETIME_COL in columns
    timestamp_present = TIMESTAMP_UTC_MS_COL in columns
    checks["datetime_present"] = datetime_present
    checks["timestamp_utc_ms_present"] = timestamp_present
    if not datetime_present:
        warnings.append("DateTime ausente.")
    if not timestamp_present and dataset_kind == "ohlcv":
        errors.append("OHLCV sem timestamp_utc_ms.")
    elif not timestamp_present:
        warnings.append("timestamp_utc_ms ausente.")

    colmap = ohlcv_column_map(columns) if dataset_kind == "ohlcv" else {}
    missing_ohlcv = [name for name, column in colmap.items() if not column]
    if dataset_kind == "ohlcv":
        checks["ohlcv_missing_columns"] = missing_ohlcv
        if missing_ohlcv:
            errors.append(f"Colunas OHLCV ausentes: {missing_ohlcv}")

    read_columns = [name for name in [DATETIME_COL, TIMESTAMP_UTC_MS_COL] if name in columns]
    if dataset_kind == "ohlcv" and not missing_ohlcv:
        read_columns.extend(colmap.values())
    read_columns = list(dict.fromkeys(read_columns))

    if not read_columns:
        warnings.append("Nenhuma coluna temporal ou OHLCV elegível para auditoria detalhada.")
        record["status"] = classify(errors, warnings)
        return record

    datetime_nulls = datetime_duplicates = datetime_out_of_order = 0
    timestamp_nulls = timestamp_duplicates = timestamp_out_of_order = 0
    timezone_drift_count = 0
    timezone_drift_total = 0
    timezone_drift_max = 0
    ohlcv_null_count = invalid_ohlc_count = negative_volume_count = 0
    return_outlier_count = 0
    delta_count = gap_count = irregular_delta_count = 0
    median_deltas: list[float] = []
    max_delta_ms: Optional[int] = None
    first_datetime: Optional[pd.Timestamp] = None
    last_datetime: Optional[pd.Timestamp] = None
    previous_datetime: Optional[pd.Timestamp] = None
    previous_timestamp: Optional[int] = None
    previous_close: Optional[float] = None
    expected_ms = infer_timeframe_ms(series)
    checks["expected_timeframe_ms"] = expected_ms

    try:
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE_ROWS, columns=read_columns):
            frame = batch.to_pandas()

            dt = None
            if datetime_present:
                dt = pd.to_datetime(frame[DATETIME_COL], errors="coerce")
                datetime_nulls += int(dt.isna().sum())
                valid_dt = dt.dropna()
                if not valid_dt.empty:
                    if first_datetime is None:
                        first_datetime = valid_dt.iloc[0]
                    last_datetime = valid_dt.iloc[-1]
                    dt_deltas = valid_dt.diff().dropna()
                    datetime_duplicates += int((dt_deltas == pd.Timedelta(0)).sum())
                    datetime_out_of_order += int((dt_deltas < pd.Timedelta(0)).sum())
                    if previous_datetime is not None:
                        boundary_delta = valid_dt.iloc[0] - previous_datetime
                        datetime_duplicates += int(boundary_delta == pd.Timedelta(0))
                        datetime_out_of_order += int(boundary_delta < pd.Timedelta(0))
                    previous_datetime = valid_dt.iloc[-1]

            ts = None
            if timestamp_present:
                ts = normalize_timestamp_to_utc_ms(frame[TIMESTAMP_UTC_MS_COL])
                timestamp_nulls += int(ts.isna().sum())
                valid_ts = ts.dropna().astype("int64")
                if not valid_ts.empty:
                    ts_deltas = valid_ts.diff().dropna()
                    timestamp_duplicates += int((ts_deltas == 0).sum())
                    timestamp_out_of_order += int((ts_deltas < 0).sum())
                    if previous_timestamp is not None:
                        boundary_delta = int(valid_ts.iloc[0]) - previous_timestamp
                        timestamp_duplicates += int(boundary_delta == 0)
                        timestamp_out_of_order += int(boundary_delta < 0)
                        ts_deltas = pd.concat([pd.Series([boundary_delta]), ts_deltas], ignore_index=True)
                    previous_timestamp = int(valid_ts.iloc[-1])

                    if expected_ms and not ts_deltas.empty:
                        positive_deltas = ts_deltas[ts_deltas > 0]
                        if not positive_deltas.empty:
                            tolerance = max(1_000, int(expected_ms * 0.001))
                            delta_count += int(len(positive_deltas))
                            gap_count += int((positive_deltas > expected_ms * 1.5).sum())
                            irregular_delta_count += int(((positive_deltas - expected_ms).abs() > tolerance).sum())
                            median_deltas.extend(float(value) for value in positive_deltas.iloc[::100])
                            batch_max = int(positive_deltas.max())
                            max_delta_ms = batch_max if max_delta_ms is None else max(max_delta_ms, batch_max)

            if dt is not None and ts is not None:
                comparable = pd.DataFrame({
                    "datetime_ms": datetime_dubai_naive_to_utc_ms(frame[DATETIME_COL]),
                    "timestamp_ms": ts,
                }).dropna()
                if not comparable.empty:
                    drift = (
                        comparable["datetime_ms"].astype("int64")
                        - comparable["timestamp_ms"].astype("int64")
                    ).abs()
                    timezone_drift_count += int(len(drift))
                    timezone_drift_total += int(drift.sum())
                    timezone_drift_max = max(timezone_drift_max, int(drift.max()))

            if dataset_kind == "ohlcv" and not missing_ohlcv:
                numeric = frame[[colmap[name] for name in ("open", "high", "low", "close", "volume")]].apply(
                    pd.to_numeric, errors="coerce"
                )
                numeric.columns = ["open", "high", "low", "close", "volume"]
                ohlcv_null_count += int(numeric.isna().sum().sum())
                invalid = (
                    (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
                    | (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1))
                    | (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1))
                )
                invalid_ohlc_count += int(invalid.sum())
                negative_volume_count += int((numeric["volume"] < 0).sum())

                close = numeric["close"]
                if previous_close is not None:
                    close = pd.concat([pd.Series([previous_close]), close], ignore_index=True)
                returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
                return_outlier_count += int((returns.abs() > RETURN_OUTLIER_ABS_PCT).sum())
                valid_close = numeric["close"].dropna()
                if not valid_close.empty:
                    previous_close = float(valid_close.iloc[-1])

            del frame
            gc.collect()

    except Exception as exc:
        errors.append(f"Falha durante a auditoria em lotes: {exc}")
        return record

    checks.update({
        "datetime_null_count": datetime_nulls,
        "datetime_duplicate_count": datetime_duplicates,
        "datetime_out_of_order_count": datetime_out_of_order,
        "datetime_monotonic_increasing": datetime_out_of_order == 0,
        "timestamp_null_count": timestamp_nulls,
        "timestamp_duplicate_count": timestamp_duplicates,
        "timestamp_out_of_order_count": timestamp_out_of_order,
        "timestamp_monotonic_increasing": timestamp_out_of_order == 0,
        "timezone_max_abs_drift_ms": timezone_drift_max if timezone_drift_count else None,
        "timezone_mean_abs_drift_ms": (
            float(timezone_drift_total / timezone_drift_count) if timezone_drift_count else None
        ),
        "median_delta_ms": float(np.median(median_deltas)) if median_deltas else None,
        "max_delta_ms": max_delta_ms,
        "irregular_delta_count": irregular_delta_count,
        "gap_count": gap_count,
        "gap_ratio": (float(gap_count / delta_count) if delta_count else None),
        "ohlcv_null_count": ohlcv_null_count,
        "invalid_ohlc_count": invalid_ohlc_count,
        "negative_volume_count": negative_volume_count,
        "return_outlier_count": return_outlier_count,
        "start": str(first_datetime) if first_datetime is not None else None,
        "end": str(last_datetime) if last_datetime is not None else None,
    })

    if datetime_nulls:
        errors.append("DateTime contém valores nulos ou inválidos.")
    if datetime_duplicates:
        errors.append("DateTime contém duplicatas.")
    if datetime_out_of_order:
        errors.append("DateTime não está em ordem crescente.")
    if timestamp_nulls:
        errors.append("timestamp_utc_ms contém valores nulos ou fora de escala.")
    if timestamp_duplicates:
        errors.append("timestamp_utc_ms contém duplicatas.")
    if timestamp_out_of_order:
        errors.append("timestamp_utc_ms não está em ordem crescente.")
    if timezone_drift_count and timezone_drift_max > MAX_TIMESTAMP_DATETIME_DRIFT_MS:
        errors.append("DateTime Dubai e timestamp_utc_ms divergem além da tolerância.")
    if checks["gap_ratio"] is not None and checks["gap_ratio"] > GAP_FAIL_RATIO:
        errors.append("Proporção de gaps acima do limite de falha.")
    elif checks["gap_ratio"] is not None and checks["gap_ratio"] > GAP_WARNING_RATIO:
        warnings.append("Gaps temporais detectados.")
    if ohlcv_null_count:
        errors.append("OHLCV contém valores nulos ou não numéricos.")
    if invalid_ohlc_count:
        errors.append("OHLC inválido: preços não positivos ou fora do intervalo High/Low.")
    if negative_volume_count:
        errors.append("Volume negativo detectado.")
    if return_outlier_count:
        warnings.append(f"Retornos absolutos acima de {RETURN_OUTLIER_ABS_PCT:.0%} detectados.")

    record["status"] = classify(errors, warnings)
    return record


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(temp_path, path)


def normalize_series_for_report(series: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(series_id): enrich_quality_record_for_ml(record)
        for series_id, record in series.items()
        if isinstance(record, dict)
    }


def build_report_payload(series: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    series = normalize_series_for_report(series)
    status_counts = Counter(record["status"] for record in series.values())
    kind_counts = Counter(record["dataset_kind"] for record in series.values())
    ohlcv_counts = Counter(
        record["status"] for record in series.values() if record["dataset_kind"] == "ohlcv"
    )
    ml_quality_counts = Counter(record.get("ml_quality_status") for record in series.values())
    root_cause_counts = Counter(
        (record.get("root_cause") or {}).get("code") for record in series.values()
    )
    fail_root_cause_counts = Counter(
        (record.get("root_cause") or {}).get("code")
        for record in series.values()
        if record.get("status") == "FAIL"
    )
    warning_code_counts = Counter(
        detail.get("code")
        for record in series.values()
        for detail in record.get("warning_ml_classification", [])
    )
    warning_ml_status_counts = Counter(
        detail.get("ml_status")
        for record in series.values()
        for detail in record.get("warning_ml_classification", [])
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "generated_at_utc": now_utc_iso(),
        "project_root": str(PROJECT_ROOT),
        "input_manifest": str(MAPA_ATIVOS_PATH),
        "output_path": str(QUALITY_REPORT_PATH),
        "time_policy": {
            "timezone_reference": TIMEZONE_LOCAL,
            "datetime_column": DATETIME_COL,
            "timestamp_utc_ms_column": TIMESTAMP_UTC_MS_COL,
            "max_timestamp_datetime_drift_ms": MAX_TIMESTAMP_DATETIME_DRIFT_MS,
        },
        "quality_policy": {
            "min_rows_warning": MIN_ROWS_WARNING,
            "gap_fail_ratio": GAP_FAIL_RATIO,
            "ml_min_rows_by_kind": ML_MIN_ROWS_BY_KIND,
            "ml_gap_block_ratio": ML_GAP_BLOCK_RATIO,
            "ml_return_outlier_block_ratio": ML_RETURN_OUTLIER_BLOCK_RATIO,
            "return_outlier_abs_pct": RETURN_OUTLIER_ABS_PCT,
            "ml_rule": (
                "Séries FAIL e WARNING classificado como ML_BLOCKED devem ser bloqueadas "
                "para treinamento amplo; ML_CAUTION pode entrar com registro explícito."
            ),
        },
        "summary": {
            "total_series": len(series),
            "status_counts": dict(sorted(status_counts.items())),
            "dataset_kind_counts": dict(sorted(kind_counts.items())),
            "ohlcv_status_counts": dict(sorted(ohlcv_counts.items())),
            "ml_quality_status_counts": dict(sorted(ml_quality_counts.items())),
            "root_cause_counts": dict(sorted(root_cause_counts.items())),
            "fail_root_cause_counts": dict(sorted(fail_root_cause_counts.items())),
            "warning_code_counts": dict(sorted(warning_code_counts.items())),
            "warning_ml_status_counts": dict(sorted(warning_ml_status_counts.items())),
            "ml_blocked_series_count": sum(
                1 for record in series.values() if record.get("ml_quality_status") == "ML_BLOCKED"
            ),
            "ml_caution_series_count": sum(
                1 for record in series.values() if record.get("ml_quality_status") == "ML_CAUTION"
            ),
            "ml_ready_series_count": sum(
                1 for record in series.values() if record.get("ml_quality_status") == "ML_READY"
            ),
        },
        "series": series,
    }


def build_root_cause_report_payload(full_report: Dict[str, Any]) -> Dict[str, Any]:
    series = full_report.get("series", {})
    if not isinstance(series, dict):
        series = {}

    triage_rows = []
    for record in series.values():
        if not isinstance(record, dict):
            continue
        if record.get("status") == "PASS" and record.get("ml_quality_status") == "ML_READY":
            continue
        root_cause = record.get("root_cause") or {}
        checks = record.get("checks") or {}
        triage_rows.append({
            "series_id": record.get("series_id"),
            "dataset_kind": record.get("dataset_kind"),
            "source": record.get("source"),
            "asset": record.get("asset"),
            "symbol": record.get("symbol"),
            "timeframe": record.get("timeframe"),
            "status": record.get("status"),
            "ml_quality_status": record.get("ml_quality_status"),
            "ml_usable_for_broad_training": record.get("ml_usable_for_broad_training"),
            "root_cause_code": root_cause.get("code"),
            "root_cause_family": root_cause.get("family"),
            "root_cause_action": root_cause.get("action"),
            "errors": record.get("errors", []),
            "warnings": record.get("warnings", []),
            "warning_ml_classification": record.get("warning_ml_classification", []),
            "rows": checks.get("rows"),
            "gap_ratio": checks.get("gap_ratio"),
            "gap_count": checks.get("gap_count"),
            "return_outlier_count": checks.get("return_outlier_count"),
            "start": checks.get("start"),
            "end": checks.get("end"),
            "path": record.get("path"),
        })

    triage_rows.sort(key=lambda item: (
        str(item.get("ml_quality_status") or ""),
        str(item.get("root_cause_code") or ""),
        str(item.get("source") or ""),
        str(item.get("asset") or ""),
        str(item.get("timeframe") or ""),
    ))

    return {
        "schema_version": "ARCHANGEL_DATA_QUALITY_ROOT_CAUSE_REPORT_1.0",
        "status": "COMPLETE",
        "generated_at_utc": full_report.get("generated_at_utc"),
        "source_report_path": str(QUALITY_REPORT_PATH),
        "summary": full_report.get("summary", {}),
        "triage_series_count": len(triage_rows),
        "triage": triage_rows,
    }


def write_root_cause_csv(payload: Dict[str, Any]) -> None:
    rows = payload.get("triage", [])
    if not isinstance(rows, list):
        rows = []

    fieldnames = [
        "series_id",
        "dataset_kind",
        "source",
        "asset",
        "symbol",
        "timeframe",
        "status",
        "ml_quality_status",
        "ml_usable_for_broad_training",
        "root_cause_code",
        "root_cause_family",
        "root_cause_action",
        "errors",
        "warnings",
        "rows",
        "gap_ratio",
        "gap_count",
        "return_outlier_count",
        "start",
        "end",
        "path",
    ]

    QUALITY_ROOT_CAUSE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUALITY_ROOT_CAUSE_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["errors"] = " | ".join(str(item) for item in out.get("errors", []))
            out["warnings"] = " | ".join(str(item) for item in out.get("warnings", []))
            writer.writerow(out)


def write_quality_reports(series: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    payload = build_report_payload(series)
    root_cause_payload = build_root_cause_report_payload(payload)
    write_json_atomic(QUALITY_REPORT_PATH, payload)
    write_json_atomic(QUALITY_ROOT_CAUSE_REPORT_PATH, root_cause_payload)
    write_root_cause_csv(root_cause_payload)
    return payload


def load_catalog() -> list[Dict[str, Any]]:
    if not MAPA_ATIVOS_PATH.is_file():
        raise FileNotFoundError(f"MAPA_ATIVOS não encontrado: {MAPA_ATIVOS_PATH}")

    with MAPA_ATIVOS_PATH.open("r", encoding="utf-8") as handle:
        mapa = json.load(handle)

    catalog = mapa.get("series_catalog", [])
    if not isinstance(catalog, list):
        raise ValueError("series_catalog inválido no MAPA_ATIVOS.json")
    return catalog


def audit_range(catalog: list[Dict[str, Any]], start: int, end: int) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    selected = catalog[start:end]
    for offset, item in enumerate(selected, start=start + 1):
        result = enrich_quality_record_for_ml(audit_series_streaming(item))
        series_id = result["series_id"]
        if not series_id:
            series_id = f"unidentified__{offset:04d}"
            result["series_id"] = series_id
            result["status"] = "FAIL"
            result["errors"].append("series_id ausente no MAPA_ATIVOS.")
        results[series_id] = result
        if offset == start + 1 or offset % 25 == 0 or offset == end:
            print(f"[QUALITY] {offset}/{len(catalog)} | {result['status']} | {series_id}")
    return results


def chunk_path(start: int, end: int) -> Path:
    return QUALITY_CHUNKS_DIR / f"DATA_QUALITY_CHUNK_{start:04d}_{end:04d}.json"


def save_chunk(catalog: list[Dict[str, Any]], start: int, end: int) -> Path:
    series = audit_range(catalog, start, end)
    path = chunk_path(start, end)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": now_utc_iso(),
        "project_root": str(PROJECT_ROOT),
        "range_start": start,
        "range_end_exclusive": end,
        "total_catalog_series": len(catalog),
        "series": series,
    }
    write_json_atomic(path, payload)
    print(f"[CHUNK DONE] {path}")
    return path


def merge_chunks(catalog: list[Dict[str, Any]]) -> None:
    series: Dict[str, Dict[str, Any]] = {}
    for path in sorted(QUALITY_CHUNKS_DIR.glob("DATA_QUALITY_CHUNK_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            chunk = json.load(handle)
        chunk_series = chunk.get("series", {})
        if not isinstance(chunk_series, dict):
            raise ValueError(f"Chunk inválido: {path}")
        series.update(chunk_series)

    expected_ids = {str(item.get("series_id") or "") for item in catalog}
    missing_ids = sorted(series_id for series_id in expected_ids if series_id and series_id not in series)
    if missing_ids:
        raise ValueError(f"Chunks incompletos: {len(missing_ids)} séries ausentes.")

    payload = write_quality_reports(series)
    print(f"[DONE] Relatório salvo em: {QUALITY_REPORT_PATH}")
    print(f"[DONE] Causa-raiz salva em: {QUALITY_ROOT_CAUSE_REPORT_PATH}")
    print(f"[DONE] CSV de triagem salvo em: {QUALITY_ROOT_CAUSE_CSV_PATH}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def enrich_existing_report() -> None:
    if not QUALITY_REPORT_PATH.is_file():
        raise FileNotFoundError(f"Relatório consolidado não encontrado: {QUALITY_REPORT_PATH}")

    with QUALITY_REPORT_PATH.open("r", encoding="utf-8") as handle:
        current = json.load(handle)

    series = current.get("series", {})
    if not isinstance(series, dict) or not series:
        raise ValueError("Relatório consolidado não contém um mapa 'series' válido.")

    payload = write_quality_reports(series)
    print(f"[DONE] Relatório enriquecido salvo em: {QUALITY_REPORT_PATH}")
    print(f"[DONE] Causa-raiz salva em: {QUALITY_ROOT_CAUSE_REPORT_PATH}")
    print(f"[DONE] CSV de triagem salvo em: {QUALITY_ROOT_CAUSE_CSV_PATH}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def main(
    start: Optional[int] = None,
    end: Optional[int] = None,
    merge: bool = False,
    enrich_existing: bool = False,
) -> None:
    if enrich_existing:
        enrich_existing_report()
        return

    catalog = load_catalog()

    if merge:
        merge_chunks(catalog)
        return

    if start is not None or end is not None:
        range_start = 0 if start is None else max(0, int(start))
        range_end = len(catalog) if end is None else min(len(catalog), int(end))
        if range_end <= range_start:
            raise ValueError("Intervalo de auditoria vazio ou inválido.")
        save_chunk(catalog, range_start, range_end)
        return

    series = audit_range(catalog, 0, len(catalog))
    payload = write_quality_reports(series)
    print(f"[DONE] Relatório salvo em: {QUALITY_REPORT_PATH}")
    print(f"[DONE] Causa-raiz salva em: {QUALITY_ROOT_CAUSE_REPORT_PATH}")
    print(f"[DONE] CSV de triagem salvo em: {QUALITY_ROOT_CAUSE_CSV_PATH}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoria formal de qualidade dos Parquets ARCHANGEL.")
    parser.add_argument("--start", type=int, help="Índice inicial inclusivo do bloco.")
    parser.add_argument("--end", type=int, help="Índice final exclusivo do bloco.")
    parser.add_argument("--merge", action="store_true", help="Consolida os blocos em DATA_QUALITY_REPORT.json.")
    parser.add_argument(
        "--enrich-existing",
        action="store_true",
        help="Reclassifica o DATA_QUALITY_REPORT.json existente com causa-raiz e política ML.",
    )
    args = parser.parse_args()
    main(start=args.start, end=args.end, merge=args.merge, enrich_existing=args.enrich_existing)
