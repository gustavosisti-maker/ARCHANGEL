# -*- coding: utf-8 -*-
"""
ARCHANGEL v1 - MAPA_ATIVOS.py

Arquivo:
    <PROJECT_ROOT>\\0_REGRAS_MANDATO\\1_MAPA_ATIVOS.py

Saída única:
    <PROJECT_ROOT>\\0_REGRAS_MANDATO\\BASE_JSON\\MAPA_ATIVOS.json

Objetivo:
    Mapear dinamicamente toda a base de dados local do ARCHANGEL para alimentar
    sistemas de IA, agentes de backtest, módulos de ML/DL e desenvolvimento
    do sistema de trading.

Princípio central:
    Este script NÃO usa lista fixa de ativos, símbolos ou periodicidades.

    Ele entra nos arquivos, lê metadados/dados e descobre:
        - Nome/código do ativo
        - Símbolo
        - Fonte/exchange
        - Tipo da série
        - Periodicidade real
        - Cobertura histórica
        - Tamanho
        - Colunas
        - Sanidade dos dados

Este script:
    - Não baixa dados
    - Não altera bases Parquet
    - Não recalcula séries
    - Apenas escaneia, audita e exporta um JSON completo e AI friendly

Requisitos:
    pip install pandas numpy pyarrow
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================

# A raiz é inferida do local deste script, sem depender da letra da unidade.
ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_JSON_DIR = ROOT_DIR / "0_REGRAS_MANDATO" / "BASE_JSON"

OUTPUT_JSON_NAME = "MAPA_ATIVOS.json"
OUTPUT_JSON_PATH = BASE_JSON_DIR / OUTPUT_JSON_NAME

BASE_DATA_DIR_CANDIDATES = [
    ROOT_DIR / "2_BASES",
    ROOT_DIR / "1_BASES",
]

TIMEZONE_LOCAL = "Asia/Dubai"
SCHEMA_VERSION = "ARCHANGEL_DYNAMIC_ASSET_DATA_MAP_AI_FRIENDLY_2.0"

SAMPLE_HASH_ROWS = 80
SAMPLE_GAP_ROWS = 5000

READ_FULL_PARQUET_FOR_AUDIT = True
ENABLE_QUALITY_CHECKS = True

# Fonte operacional atual: Binance only.
# Arquivos antigos de outras fontes podem permanecer no disco, mas não entram no
# series_catalog nem nos indexes usados pelas etapas seguintes.
SOURCE_INCLUDE_PREFIXES = tuple(
    x.strip().lower()
    for x in os.environ.get("ARCHANGEL_MAP_SOURCE_INCLUDE_PREFIXES", "binance").split(",")
    if x.strip()
)

INCLUDED_EXTENSIONS = {
    ".parquet",
    ".json",
    ".txt",
    ".csv",
    ".log",
    ".py",
}

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
}


# =============================================================================
# 2. UTILITÁRIOS GERAIS
# =============================================================================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_local_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_base_data_dir() -> Path:
    for candidate in BASE_DATA_DIR_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return BASE_DATA_DIR_CANDIDATES[0]


def safe_relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def file_size_bytes(path: Path) -> Optional[int]:
    try:
        if path.exists() and path.is_file():
            return int(path.stat().st_size)
        return None
    except Exception:
        return None


def file_size_mb(path: Path) -> Optional[float]:
    size = file_size_bytes(path)
    if size is None:
        return None
    return round(size / (1024 * 1024), 6)


def file_modified_at(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return None


def file_created_at(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_ctime).isoformat(timespec="seconds")
    except Exception:
        return None


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def normalize_path_lower(path: Path) -> str:
    return str(path).replace("/", "\\").lower()


def should_exclude_dir(path: Path) -> bool:
    return path.name in EXCLUDED_DIR_NAMES


def extension_of(path: Path) -> str:
    return path.suffix.lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_column_map(columns: List[str]) -> Dict[str, str]:
    """
    Mapeia colunas ignorando case.
    """
    return {str(c).lower(): str(c) for c in columns}


# =============================================================================
# 3. DESCOBERTA DINÂMICA DE DATASET
# =============================================================================

def infer_location_type(path: Path) -> str:
    p = normalize_path_lower(path)

    if "\\custom_timeframes\\" in p:
        return "custom_timeframes"

    if "\\processed\\" in p:
        return "processed"

    if "\\raw\\" in p:
        return "raw"

    if "\\metadata\\" in p:
        return "metadata"

    if "\\diagnostics\\" in p:
        return "diagnostics"

    if "\\_logs\\" in p:
        return "logs"

    return "unknown"


def infer_source_dynamic(path: Path, df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """
    Descobre fonte/exchange sem lista fixa.

    Prioridade:
        1. Colunas Source / Exchange / Venue
        2. Caminho
        3. Nome do arquivo
    """
    if df is not None and not df.empty:
        colmap = normalize_column_map([str(c) for c in df.columns])

        for key in ["source", "exchange", "venue", "market"]:
            if key in colmap:
                col = colmap[key]
                vals = df[col].dropna().astype(str).unique()
                if len(vals) > 0:
                    return str(vals[0]).strip()

    text = normalize_path_lower(path)

    known_tokens = [
        "binance_spot",
        "binance_futures",
        "binance",
        "bybit_linear",
        "bybit_inverse",
        "bybit",
        "okx",
        "kraken",
        "coinbase",
        "kucoin",
    ]

    for token in known_tokens:
        if token in text:
            return token

    # fallback: tenta capturar algo entre ohlcv\<source>\
    parts = Path(str(path)).parts
    lower_parts = [p.lower() for p in parts]

    for anchor in ["ohlcv", "funding_rate", "open_interest"]:
        if anchor in lower_parts:
            idx = lower_parts.index(anchor)
            if idx + 1 < len(parts):
                return parts[idx + 1]

    return None


def source_is_included(source: Optional[str]) -> bool:
    if not SOURCE_INCLUDE_PREFIXES:
        return True
    if not source:
        return False

    source_lower = str(source).strip().lower()
    return any(source_lower.startswith(prefix) for prefix in SOURCE_INCLUDE_PREFIXES)


def infer_dataset_kind_dynamic(path: Path, columns: List[str]) -> str:
    """
    Descobre tipo de dataset pelo conteúdo/colunas/caminho.
    """
    p = normalize_path_lower(path)
    cols_lower = {str(c).lower() for c in columns}

    if "funding" in p or "fundingrate" in cols_lower or "funding_rate" in cols_lower:
        return "funding_rate"

    if "open_interest" in p or "openinterest" in cols_lower or "open_interest" in cols_lower:
        return "open_interest"

    ohlcv_cols = {"datetime", "open", "high", "low", "close", "volume"}

    if ohlcv_cols.issubset(cols_lower):
        return "ohlcv"

    if "ohlcv" in p:
        return "ohlcv"

    if "datetime" in cols_lower and any(c in cols_lower for c in ["close", "price", "value"]):
        return "timeseries"

    if "datetime" in cols_lower:
        return "datetime_table"

    return "unknown"


def find_datetime_column(columns: List[str]) -> Optional[str]:
    candidates = [
        "DateTime",
        "datetime",
        "date_time",
        "timestamp",
        "time",
        "date",
        "OpenTime",
        "open_time",
    ]

    colmap = normalize_column_map(columns)

    for c in candidates:
        if c.lower() in colmap:
            return colmap[c.lower()]

    return None


def find_symbol_column(columns: List[str]) -> Optional[str]:
    candidates = [
        "Symbol",
        "symbol",
        "Ticker",
        "ticker",
        "Instrument",
        "instrument",
        "Pair",
        "pair",
    ]

    colmap = normalize_column_map(columns)

    for c in candidates:
        if c.lower() in colmap:
            return colmap[c.lower()]

    return None


def extract_symbol_from_text(text: str) -> Optional[str]:
    """
    Extrai símbolo genérico de strings.

    Exemplos detectados:
        BTCUSDT
        ETHUSDT
        SOL-USDT
        BTC_USDT
        BTCUSD
    """
    upper = text.upper()

    patterns = [
        r"([A-Z0-9]{2,15}USDT)",
        r"([A-Z0-9]{2,15}USD)",
        r"([A-Z0-9]{2,15}USDC)",
        r"([A-Z0-9]{2,15}BUSD)",
        r"([A-Z0-9]{2,15}[-_]USDT)",
        r"([A-Z0-9]{2,15}[-_]USD)",
        r"([A-Z0-9]{2,15}[-_]USDC)",
    ]

    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1).replace("-", "").replace("_", "")

    return None


def infer_symbol_dynamic(path: Path, df: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Descobre símbolo pelo conteúdo e pelo nome/caminho.
    """
    if df is not None and not df.empty:
        sym_col = find_symbol_column([str(c) for c in df.columns])

        if sym_col:
            vals = df[sym_col].dropna().astype(str).unique()
            if len(vals) > 0:
                return str(vals[0]).strip().upper().replace("-", "").replace("_", "")

    symbol_from_path = extract_symbol_from_text(str(path))

    if symbol_from_path:
        return symbol_from_path

    return None


def asset_from_symbol(symbol: Optional[str]) -> Optional[str]:
    """
    Deriva ativo base do símbolo sem lista fixa.
    """
    if not symbol:
        return None

    s = symbol.upper().replace("-", "").replace("_", "")

    quote_suffixes = [
        "USDT",
        "USDC",
        "BUSD",
        "USD",
        "BTC",
        "ETH",
        "EUR",
        "BRL",
    ]

    for quote in quote_suffixes:
        if s.endswith(quote) and len(s) > len(quote):
            return s[:-len(quote)]

    return s


def infer_asset_dynamic(path: Path, df: Optional[pd.DataFrame], symbol: Optional[str]) -> Optional[str]:
    """
    Descobre ativo sem lista fixa.

    Prioridade:
        1. Coluna Asset/BaseAsset
        2. Derivado do símbolo
        3. Pasta imediatamente acima ou tokens de arquivo
    """
    if df is not None and not df.empty:
        colmap = normalize_column_map([str(c) for c in df.columns])

        for key in ["asset", "baseasset", "base_asset", "coin", "currency"]:
            if key in colmap:
                col = colmap[key]
                vals = df[col].dropna().astype(str).unique()
                if len(vals) > 0:
                    return str(vals[0]).strip().upper()

    from_symbol = asset_from_symbol(symbol)

    if from_symbol:
        return from_symbol

    # fallback por caminho: procura pasta curta, geralmente asset
    parts = list(path.parts)

    ignore = {
        "ohlcv",
        "processed",
        "custom_timeframes",
        "raw",
        "metadata",
        "diagnostics",
        "_logs",
        "funding_rate",
        "open_interest",
        "binance",
        "binance_spot",
        "bybit",
        "bybit_linear",
    }

    for part in reversed(parts):
        clean = re.sub(r"[^A-Za-z0-9]", "", part).upper()

        if 2 <= len(clean) <= 12 and clean.lower() not in ignore and not clean.endswith("PARQUET"):
            if not re.search(r"\d", clean):
                return clean

    return None


def normalize_timeframe_from_seconds(seconds: Optional[float]) -> Optional[str]:
    """
    Converte periodicidade inferida em label amigável.

    Não depende de lista fixa. Cria labels conforme o delta real.
    """
    if seconds is None:
        return None

    try:
        sec = float(seconds)
    except Exception:
        return None

    if sec <= 0:
        return None

    minute = 60
    hour = 3600
    day = 86400

    if sec < hour and abs(sec % minute) < 1e-9:
        n = int(round(sec / minute))
        return f"{n}min"

    if sec < day and abs(sec % hour) < 1e-9:
        n = int(round(sec / hour))
        return f"{n}h"

    if abs(sec % day) < 1e-9:
        n = int(round(sec / day))
        return f"{n}D"

    return f"{int(round(sec))}s"


def infer_timeframe_from_standard_path(path: Path) -> Optional[str]:
    """
    Extrai timeframe de nomes/pastas ARCHANGEL padronizados antes de recorrer
    à inferência por deltas, que pode ser distorcida por gaps reais de mercado.
    """
    text = str(path).replace("\\", "/")
    candidates = [path.stem, path.parent.name, text]

    patterns = [
        r"(?i)(?:^|[_/\-])(?P<tf>\d+min|\d+h|\d+D)(?:$|[_/\-.])",
        r"(?i)(?:^|[_/\-])(?P<num>\d+)[_\-]?min(?:ute)?s?(?:$|[_/\-.])",
        r"(?i)(?:^|[_/\-])(?P<num>\d+)[_\-]?hour?s?(?:$|[_/\-.])",
        r"(?i)(?:^|[_/\-])(?P<num>\d+)[_\-]?day?s?(?:$|[_/\-.])",
    ]

    for candidate in candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate)
            if not match:
                continue

            if "tf" in match.groupdict() and match.group("tf"):
                tf = match.group("tf")
                unit = tf[-3:].lower() if tf.lower().endswith("min") else tf[-1].lower()
                number = re.sub(r"\D", "", tf)
            else:
                number = match.group("num")
                lower_pattern = pattern.lower()
                if "min" in lower_pattern:
                    unit = "min"
                elif "hour" in lower_pattern:
                    unit = "h"
                else:
                    unit = "d"

            if not number:
                continue

            if unit == "min":
                return f"{int(number)}min"
            if unit == "h":
                return f"{int(number)}h"
            if unit == "d":
                return f"{int(number)}D"

    return None


def infer_periodicity_from_datetime(df: Optional[pd.DataFrame], datetime_col: Optional[str]) -> Dict[str, Any]:
    """
    Infere periodicidade real da série a partir da coluna temporal.

    Não usa periodicidades pré-definidas.
    """
    result = {
        "datetime_column": datetime_col,
        "inferred_timeframe": None,
        "inferred_seconds_median": None,
        "inferred_seconds_mode": None,
        "inferred_timedelta_median": None,
        "inferred_timedelta_mode": None,
        "unique_deltas_seconds_sample": [],
        "is_regular_by_sample": None,
        "event_based_or_irregular": None,
        "confidence": "unknown",
    }

    if df is None or df.empty or not datetime_col or datetime_col not in df.columns:
        return result

    dt = pd.to_datetime(df[datetime_col], errors="coerce").dropna().sort_values()

    if len(dt) < 3:
        result["event_based_or_irregular"] = True
        result["confidence"] = "low_too_few_rows"
        return result

    if len(dt) > SAMPLE_GAP_ROWS:
        idx = np.linspace(0, len(dt) - 1, SAMPLE_GAP_ROWS).astype(int)
        dt_sample = dt.iloc[idx]
    else:
        dt_sample = dt

    diffs = dt_sample.diff().dropna()

    if diffs.empty:
        result["event_based_or_irregular"] = True
        result["confidence"] = "low_no_diffs"
        return result

    seconds = diffs.dt.total_seconds().astype(float)
    median_sec = float(seconds.median())

    try:
        mode_sec = float(seconds.mode().iloc[0])
    except Exception:
        mode_sec = median_sec

    unique_values = sorted(seconds.dropna().unique().tolist())
    unique_sample = unique_values[:30]

    # Regularidade: modo domina a amostra?
    mode_ratio = float((seconds == mode_sec).mean()) if len(seconds) else 0.0

    inferred_tf = normalize_timeframe_from_seconds(mode_sec)

    result["inferred_timeframe"] = inferred_tf
    result["inferred_seconds_median"] = median_sec
    result["inferred_seconds_mode"] = mode_sec
    result["inferred_timedelta_median"] = str(pd.Timedelta(seconds=median_sec))
    result["inferred_timedelta_mode"] = str(pd.Timedelta(seconds=mode_sec))
    result["unique_deltas_seconds_sample"] = unique_sample
    result["is_regular_by_sample"] = bool(mode_ratio >= 0.95)
    result["event_based_or_irregular"] = bool(mode_ratio < 0.80)
    result["confidence"] = (
        "high" if mode_ratio >= 0.95 else
        "medium" if mode_ratio >= 0.80 else
        "low_irregular_or_event_based"
    )

    return result


def build_series_id(
    dataset_kind: Optional[str],
    source: Optional[str],
    asset: Optional[str],
    symbol: Optional[str],
    timeframe: Optional[str],
    file_stem: str,
) -> str:
    base = "__".join([
        str(dataset_kind or "unknown"),
        str(source or "unknown"),
        str(asset or "unknown"),
        str(symbol or "unknown"),
        str(timeframe or "unknown"),
    ]).replace(" ", "_")

    # Hash curto evita colisão se dois arquivos forem parecidos.
    suffix = sha256_text(file_stem)[:8]

    return f"{base}__{suffix}"


def expected_columns_for_kind(dataset_kind: str) -> List[str]:
    if dataset_kind == "ohlcv":
        return ["DateTime", "Open", "High", "Low", "Close", "Volume"]

    if dataset_kind == "funding_rate":
        return ["DateTime", "Symbol", "FundingRate"]

    if dataset_kind == "open_interest":
        return ["DateTime", "Symbol", "OIInterval", "OpenInterest"]

    if dataset_kind in ["timeseries", "datetime_table"]:
        return ["DateTime"]

    return []


# =============================================================================
# 4. LEITURA E AUDITORIA DE PARQUET
# =============================================================================

def sample_hash_dataframe(df: pd.DataFrame, n: int = SAMPLE_HASH_ROWS) -> Optional[str]:
    if df is None or df.empty:
        return None

    try:
        sample = df.copy()

        if len(sample) > n:
            idx = sorted(set(np.linspace(0, len(sample) - 1, min(n, len(sample))).astype(int)))
            sample = sample.iloc[idx].copy()

        text = sample.to_csv(index=False)
        return sha256_text(text)

    except Exception:
        return None


def audit_dataframe_quality(
    df: Optional[pd.DataFrame],
    dataset_kind: str,
    datetime_col: Optional[str],
    periodicity: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Audita sanidade básica sem depender de timeframe pré-definido.
    """
    result = {
        "rows": 0,
        "columns": [],
        "column_count": 0,
        "start": None,
        "end": None,
        "duplicates_datetime": None,
        "null_datetime_count": None,
        "gap_count_sampled": None,
        "gap_ratio_sampled": None,
        "has_bad_ohlc": None,
        "negative_volume_count": None,
        "null_counts": {},
        "sample_hash": None,
        "status": "EMPTY",
        "warnings": [],
    }

    if df is None:
        result["status"] = "READ_ERROR_OR_EMPTY"
        result["warnings"].append("Não foi possível ler o Parquet ou o arquivo está vazio.")
        return result

    if df.empty:
        result["status"] = "EMPTY"
        result["warnings"].append("DataFrame vazio.")
        return result

    result["rows"] = int(len(df))
    result["columns"] = [str(c) for c in df.columns]
    result["column_count"] = int(len(df.columns))
    result["sample_hash"] = sample_hash_dataframe(df)

    result["null_counts"] = {
        str(c): int(df[c].isna().sum())
        for c in df.columns
    }

    if not datetime_col or datetime_col not in df.columns:
        result["status"] = "CHECK_WARNING"
        result["warnings"].append("Coluna temporal não identificada.")
        return result

    dt = pd.to_datetime(df[datetime_col], errors="coerce")
    null_dt = int(dt.isna().sum())

    result["null_datetime_count"] = null_dt

    valid_dt = dt.dropna().sort_values()

    if not valid_dt.empty:
        result["start"] = str(valid_dt.min())
        result["end"] = str(valid_dt.max())

    result["duplicates_datetime"] = int(dt.duplicated().sum())

    expected_seconds = periodicity.get("inferred_seconds_mode")

    if expected_seconds and len(valid_dt) >= 3 and not periodicity.get("event_based_or_irregular"):
        sample_dt = valid_dt

        if len(sample_dt) > SAMPLE_GAP_ROWS:
            idx = np.linspace(0, len(sample_dt) - 1, SAMPLE_GAP_ROWS).astype(int)
            sample_dt = sample_dt.iloc[idx]

        diffs = sample_dt.diff().dropna().dt.total_seconds()

        if not diffs.empty:
            gaps = diffs > float(expected_seconds) * 1.5
            result["gap_count_sampled"] = int(gaps.sum())
            result["gap_ratio_sampled"] = round(float(gaps.mean()), 8)

    if dataset_kind == "ohlcv":
        colmap = normalize_column_map([str(c) for c in df.columns])

        required_lower = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_lower if c not in colmap]

        if missing:
            result["warnings"].append(f"Colunas OHLCV ausentes: {missing}")
            result["has_bad_ohlc"] = True
        else:
            open_col = colmap["open"]
            high_col = colmap["high"]
            low_col = colmap["low"]
            close_col = colmap["close"]
            volume_col = colmap["volume"]

            x = df[[open_col, high_col, low_col, close_col, volume_col]].copy()

            for c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")

            bad = (
                (x[open_col] <= 0) |
                (x[high_col] <= 0) |
                (x[low_col] <= 0) |
                (x[close_col] <= 0) |
                (x[high_col] < x[[open_col, low_col, close_col]].max(axis=1)) |
                (x[low_col] > x[[open_col, high_col, close_col]].min(axis=1))
            )

            result["has_bad_ohlc"] = bool(bad.any())
            result["negative_volume_count"] = int((x[volume_col].fillna(0) < 0).sum())

            if result["has_bad_ohlc"]:
                result["warnings"].append("Foram detectadas inconsistências OHLC.")

            if result["negative_volume_count"] > 0:
                result["warnings"].append("Foram detectados volumes negativos.")

    if result["duplicates_datetime"] and result["duplicates_datetime"] > 0:
        result["warnings"].append("Existem timestamps duplicados.")

    if result["null_datetime_count"] and result["null_datetime_count"] > 0:
        result["warnings"].append("Existem timestamps nulos ou inválidos.")

    if result["gap_count_sampled"] is not None and result["gap_count_sampled"] > 0:
        result["warnings"].append("Gaps detectados na amostra temporal.")

    if periodicity.get("confidence") == "low_irregular_or_event_based":
        if dataset_kind in ["ohlcv", "open_interest"]:
            result["warnings"].append("Periodicidade irregular detectada em série que normalmente deveria ser regular.")

    if result["warnings"]:
        result["status"] = "CHECK_WARNING"
    else:
        result["status"] = "OK"

    return result


# =============================================================================
# 5. SCAN FÍSICO
# =============================================================================

def scan_files_and_directories(base_dir: Path) -> Dict[str, Any]:
    directories: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    extension_summary: Dict[str, Dict[str, Any]] = {}

    if not base_dir.exists():
        return {
            "directories": directories,
            "files": files,
            "errors": [
                {
                    "type": "BASE_DIR_NOT_FOUND",
                    "path": str(base_dir),
                }
            ],
            "extension_summary": {},
            "summary": {
                "total_directories": 0,
                "total_files": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0,
            },
        }

    for current_dir, dir_names, file_names in os.walk(base_dir):
        current_path = Path(current_dir)

        dir_names[:] = [
            d for d in dir_names
            if not should_exclude_dir(current_path / d)
        ]

        try:
            directories.append({
                "name": current_path.name,
                "absolute_path": str(current_path),
                "relative_path": safe_relative_path(current_path, base_dir),
                "location_type": infer_location_type(current_path),
                "created_at": file_created_at(current_path),
                "modified_at": file_modified_at(current_path),
            })
        except Exception as exc:
            errors.append({
                "type": "DIRECTORY_SCAN_ERROR",
                "path": str(current_path),
                "error": str(exc),
            })

        for file_name in file_names:
            file_path = current_path / file_name
            ext = extension_of(file_path)

            if INCLUDED_EXTENSIONS and ext not in INCLUDED_EXTENSIONS:
                continue

            try:
                size_b = file_size_bytes(file_path) or 0
                size_m = file_size_mb(file_path) or 0.0

                record = {
                    "name": file_path.name,
                    "stem": file_path.stem,
                    "extension": ext,
                    "absolute_path": str(file_path),
                    "relative_path": safe_relative_path(file_path, base_dir),
                    "parent_dir": str(file_path.parent),
                    "location_type": infer_location_type(file_path),
                    "size_bytes": size_b,
                    "size_mb": size_m,
                    "created_at": file_created_at(file_path),
                    "modified_at": file_modified_at(file_path),
                }

                files.append(record)

                if ext not in extension_summary:
                    extension_summary[ext] = {
                        "count": 0,
                        "total_size_bytes": 0,
                        "total_size_mb": 0.0,
                    }

                extension_summary[ext]["count"] += 1
                extension_summary[ext]["total_size_bytes"] += size_b

            except Exception as exc:
                errors.append({
                    "type": "FILE_SCAN_ERROR",
                    "path": str(file_path),
                    "error": str(exc),
                })

    total_size_bytes = sum(f.get("size_bytes", 0) or 0 for f in files)

    for _, item in extension_summary.items():
        item["total_size_mb"] = round(item["total_size_bytes"] / (1024 * 1024), 6)

    return {
        "directories": directories,
        "files": files,
        "errors": errors,
        "extension_summary": dict(sorted(extension_summary.items())),
        "summary": {
            "total_directories": len(directories),
            "total_files": len(files),
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 6),
            "total_parquet_files": sum(1 for f in files if f.get("extension") == ".parquet"),
            "total_json_files": sum(1 for f in files if f.get("extension") == ".json"),
            "total_txt_files": sum(1 for f in files if f.get("extension") == ".txt"),
            "total_csv_files": sum(1 for f in files if f.get("extension") == ".csv"),
            "total_log_files": sum(1 for f in files if f.get("extension") == ".log"),
            "total_py_files": sum(1 for f in files if f.get("extension") == ".py"),
            "total_errors": len(errors),
        },
    }


# =============================================================================
# 6. CATÁLOGO DINÂMICO DE SÉRIES
# =============================================================================

def build_series_record_from_parquet(file_record: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    path = Path(file_record["absolute_path"])

    df: Optional[pd.DataFrame] = None
    read_error = None
    columns: List[str] = []

    try:
        if READ_FULL_PARQUET_FOR_AUDIT:
            df = pd.read_parquet(path)
            columns = [str(c) for c in df.columns]
        else:
            df = pd.read_parquet(path)
            columns = [str(c) for c in df.columns]
    except Exception as exc:
        read_error = str(exc)
        df = None
        columns = []

    datetime_col = find_datetime_column(columns)
    dataset_kind = infer_dataset_kind_dynamic(path, columns)
    source = infer_source_dynamic(path, df)
    symbol = infer_symbol_dynamic(path, df)
    asset = infer_asset_dynamic(path, df, symbol)

    periodicity = infer_periodicity_from_datetime(df, datetime_col)
    timeframe_from_path = infer_timeframe_from_standard_path(path)
    timeframe = timeframe_from_path or periodicity.get("inferred_timeframe")
    timeframe_discovery_method = (
        "standard_path_or_filename"
        if timeframe_from_path
        else "inferred_from_DateTime_deltas"
    )

    # Funding costuma ser event-based. Mantém label mais honesto.
    if not timeframe_from_path and dataset_kind == "funding_rate" and periodicity.get("event_based_or_irregular"):
        timeframe = "event_based"
        timeframe_discovery_method = "event_based_from_irregular_datetime"

    location_type = infer_location_type(path)
    custom_timeframe = location_type == "custom_timeframes"

    quality = audit_dataframe_quality(
        df=df,
        dataset_kind=dataset_kind,
        datetime_col=datetime_col,
        periodicity=periodicity,
    ) if ENABLE_QUALITY_CHECKS else {
        "status": "NOT_CHECKED",
        "warnings": [],
    }

    expected_cols = expected_columns_for_kind(dataset_kind)
    columns_lower_to_original = normalize_column_map(columns)

    missing_expected_columns = []
    for c in expected_cols:
        if c.lower() not in columns_lower_to_original:
            missing_expected_columns.append(c)

    series_id = build_series_id(
        dataset_kind=dataset_kind,
        source=source,
        asset=asset,
        symbol=symbol,
        timeframe=timeframe,
        file_stem=path.stem,
    )

    record = {
        "series_id": series_id,
        "discovery_method": {
            "asset": "content_or_symbol_or_path_dynamic",
            "symbol": "content_or_path_regex_dynamic",
            "source": "content_or_path_dynamic",
            "timeframe": timeframe_discovery_method,
            "dataset_kind": "inferred_from_columns_and_path",
        },
        "dataset_kind": dataset_kind,
        "source": source,
        "asset": asset,
        "symbol": symbol,
        "timeframe": timeframe,
        "timeframe_from_path": timeframe_from_path,
        "custom_timeframe": custom_timeframe,
        "location_type": location_type,
        "file": {
            "name": path.name,
            "stem": path.stem,
            "extension": path.suffix.lower(),
            "absolute_path": str(path),
            "relative_path": safe_relative_path(path, base_dir),
            "parent_dir": str(path.parent),
            "size_bytes": file_record.get("size_bytes"),
            "size_mb": file_record.get("size_mb"),
            "created_at": file_record.get("created_at"),
            "modified_at": file_record.get("modified_at"),
            "exists": path.exists(),
            "read_error": read_error,
        },
        "schema": {
            "columns": columns,
            "column_count": len(columns),
            "datetime_column_detected": datetime_col,
            "expected_columns_by_detected_kind": expected_cols,
            "missing_expected_columns": missing_expected_columns,
        },
        "periodicity": periodicity,
        "quality": quality,
        "date_range": {
            "start": quality.get("start"),
            "end": quality.get("end"),
            "timezone_policy": "DateTime tratado como horário local de Dubai, naive, sem timezone attached",
            "timezone_reference": TIMEZONE_LOCAL,
        },
        "usage": {
            "preferred_format": "parquet",
            "python_read_instruction": f"pd.read_parquet(r'{str(path)}')",
            "datetime_column": datetime_col,
            "tags": [
                "trading_system",
                "backtesting",
                "walk_forward",
                "machine_learning",
                "deep_learning",
                "ai_agent_navigation",
            ],
        },
    }

    return record


def build_series_catalog(physical_inventory: Dict[str, Any], base_dir: Path) -> List[Dict[str, Any]]:
    files = physical_inventory.get("files", [])
    parquet_files = [f for f in files if f.get("extension") == ".parquet"]

    series: List[Dict[str, Any]] = []
    skipped_by_source = 0

    for idx, file_record in enumerate(parquet_files, start=1):
        path = Path(file_record["absolute_path"])
        source_from_path = infer_source_dynamic(path, None)

        if source_from_path and not source_is_included(source_from_path):
            skipped_by_source += 1
            print(
                f"[MAP SKIP SOURCE] Parquet {idx}/{len(parquet_files)} | "
                f"source={source_from_path} | {file_record.get('relative_path')}"
            )
            continue

        print(f"[MAP] Parquet {idx}/{len(parquet_files)} | {file_record.get('relative_path')}")

        try:
            record = build_series_record_from_parquet(file_record, base_dir)
            if not source_is_included(record.get("source")):
                skipped_by_source += 1
                print(
                    f"[MAP SKIP SOURCE] Parquet {idx}/{len(parquet_files)} | "
                    f"source={record.get('source')} | {file_record.get('relative_path')}"
                )
                continue

            series.append(record)
        except Exception as exc:
            series.append({
                "series_id": f"error__{idx}",
                "dataset_kind": "unknown",
                "source": None,
                "asset": None,
                "symbol": None,
                "timeframe": None,
                "status": "ERROR",
                "file": file_record,
                "error": str(exc),
            })

    if skipped_by_source:
        print(
            f"[MAP SOURCE FILTER] Ignorados {skipped_by_source} parquet(s) fora de "
            f"SOURCE_INCLUDE_PREFIXES={SOURCE_INCLUDE_PREFIXES}"
        )

    return series


# =============================================================================
# 7. ÍNDICES DINÂMICOS
# =============================================================================

def build_indexes(series_catalog: List[Dict[str, Any]], physical_inventory: Dict[str, Any]) -> Dict[str, Any]:
    by_id: Dict[str, Dict[str, Any]] = {}
    ids_by_asset: Dict[str, List[str]] = {}
    ids_by_symbol: Dict[str, List[str]] = {}
    ids_by_source: Dict[str, List[str]] = {}
    ids_by_timeframe: Dict[str, List[str]] = {}
    ids_by_dataset_kind: Dict[str, List[str]] = {}
    ids_by_status: Dict[str, List[str]] = {}
    ids_by_location_type: Dict[str, List[str]] = {}

    for item in series_catalog:
        sid = item.get("series_id")
        if not sid:
            continue

        by_id[sid] = item

        asset = str(item.get("asset") or "unknown")
        symbol = str(item.get("symbol") or "unknown")
        source = str(item.get("source") or "unknown")
        timeframe = str(item.get("timeframe") or "unknown")
        kind = str(item.get("dataset_kind") or "unknown")
        location = str(item.get("location_type") or "unknown")
        status = str(item.get("quality", {}).get("status") or item.get("status") or "unknown")

        ids_by_asset.setdefault(asset, []).append(sid)
        ids_by_symbol.setdefault(symbol, []).append(sid)
        ids_by_source.setdefault(source, []).append(sid)
        ids_by_timeframe.setdefault(timeframe, []).append(sid)
        ids_by_dataset_kind.setdefault(kind, []).append(sid)
        ids_by_status.setdefault(status, []).append(sid)
        ids_by_location_type.setdefault(location, []).append(sid)

    files_by_extension: Dict[str, List[str]] = {}
    files_by_location_type: Dict[str, List[str]] = {}

    for f in physical_inventory.get("files", []):
        ext = str(f.get("extension") or "unknown")
        loc = str(f.get("location_type") or "unknown")
        rel = str(f.get("relative_path") or f.get("absolute_path"))

        files_by_extension.setdefault(ext, []).append(rel)
        files_by_location_type.setdefault(loc, []).append(rel)

    return {
        "series_by_id": by_id,
        "series_ids_by_asset": ids_by_asset,
        "series_ids_by_symbol": ids_by_symbol,
        "series_ids_by_source": ids_by_source,
        "series_ids_by_timeframe": ids_by_timeframe,
        "series_ids_by_dataset_kind": ids_by_dataset_kind,
        "series_ids_by_status": ids_by_status,
        "series_ids_by_location_type": ids_by_location_type,
        "files_by_extension": files_by_extension,
        "files_by_location_type": files_by_location_type,
    }


# =============================================================================
# 8. RESUMOS DINÂMICOS
# =============================================================================

def build_discovered_universe(series_catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    assets = sorted({str(x.get("asset")) for x in series_catalog if x.get("asset")})
    symbols = sorted({str(x.get("symbol")) for x in series_catalog if x.get("symbol")})
    sources = sorted({str(x.get("source")) for x in series_catalog if x.get("source")})
    timeframes = sorted({str(x.get("timeframe")) for x in series_catalog if x.get("timeframe")})
    dataset_kinds = sorted({str(x.get("dataset_kind")) for x in series_catalog if x.get("dataset_kind")})

    return {
        "assets_discovered": assets,
        "symbols_discovered": symbols,
        "sources_discovered": sources,
        "timeframes_discovered": timeframes,
        "dataset_kinds_discovered": dataset_kinds,
        "counts": {
            "assets": len(assets),
            "symbols": len(symbols),
            "sources": len(sources),
            "timeframes": len(timeframes),
            "dataset_kinds": len(dataset_kinds),
        },
    }


def build_asset_matrix(series_catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Matriz 100% derivada dos dados encontrados.
    """
    matrix: Dict[str, Any] = {}

    for item in series_catalog:
        asset = item.get("asset") or "unknown"
        symbol = item.get("symbol") or "unknown"
        source = item.get("source") or "unknown"
        timeframe = item.get("timeframe") or "unknown"
        kind = item.get("dataset_kind") or "unknown"
        status = item.get("quality", {}).get("status") or "unknown"

        if asset not in matrix:
            matrix[asset] = {
                "symbols": {},
                "sources": {},
                "dataset_counts": {},
                "status_counts": {},
                "total_series_found": 0,
            }

        matrix[asset]["total_series_found"] += 1
        matrix[asset]["dataset_counts"][kind] = matrix[asset]["dataset_counts"].get(kind, 0) + 1
        matrix[asset]["status_counts"][status] = matrix[asset]["status_counts"].get(status, 0) + 1

        if symbol not in matrix[asset]["symbols"]:
            matrix[asset]["symbols"][symbol] = []

        matrix[asset]["symbols"][symbol].append(item.get("series_id"))

        if source not in matrix[asset]["sources"]:
            matrix[asset]["sources"][source] = {
                "timeframes": {},
                "dataset_kinds": {},
                "statuses": {},
            }

        matrix[asset]["sources"][source]["timeframes"].setdefault(timeframe, []).append(item.get("series_id"))
        matrix[asset]["sources"][source]["dataset_kinds"][kind] = (
            matrix[asset]["sources"][source]["dataset_kinds"].get(kind, 0) + 1
        )
        matrix[asset]["sources"][source]["statuses"][status] = (
            matrix[asset]["sources"][source]["statuses"].get(status, 0) + 1
        )

    return matrix


def build_summary(series_catalog: List[Dict[str, Any]], physical_inventory: Dict[str, Any]) -> Dict[str, Any]:
    total_series = len(series_catalog)

    def count_where(fn) -> int:
        return sum(1 for x in series_catalog if fn(x))

    total_size_mb_parquet = round(
        sum(
            safe_float(x.get("file", {}).get("size_mb"), 0.0) or 0.0
            for x in series_catalog
        ),
        6,
    )

    status_counts: Dict[str, int] = {}
    kind_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    timeframe_counts: Dict[str, int] = {}
    asset_counts: Dict[str, int] = {}

    for x in series_catalog:
        status = str(x.get("quality", {}).get("status") or x.get("status") or "unknown")
        kind = str(x.get("dataset_kind") or "unknown")
        source = str(x.get("source") or "unknown")
        timeframe = str(x.get("timeframe") or "unknown")
        asset = str(x.get("asset") or "unknown")

        status_counts[status] = status_counts.get(status, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
        asset_counts[asset] = asset_counts.get(asset, 0) + 1

    rows_total = sum(
        safe_int(x.get("quality", {}).get("rows"), 0) or 0
        for x in series_catalog
    )

    warning_series = [
        x.get("series_id")
        for x in series_catalog
        if str(x.get("quality", {}).get("status") or x.get("status")) != "OK"
    ]

    return {
        "total_series_parquet_mapped": total_series,
        "total_rows_detected_across_series": rows_total,
        "total_parquet_size_mb_detected": total_size_mb_parquet,
        "total_ohlcv_series": count_where(lambda x: x.get("dataset_kind") == "ohlcv"),
        "total_funding_rate_series": count_where(lambda x: x.get("dataset_kind") == "funding_rate"),
        "total_open_interest_series": count_where(lambda x: x.get("dataset_kind") == "open_interest"),
        "total_timeseries_or_datetime_table_series": count_where(
            lambda x: x.get("dataset_kind") in ["timeseries", "datetime_table"]
        ),
        "total_unknown_series": count_where(lambda x: x.get("dataset_kind") == "unknown"),
        "total_custom_timeframe_series": count_where(lambda x: bool(x.get("custom_timeframe"))),
        "status_counts": dict(sorted(status_counts.items())),
        "dataset_kind_counts": dict(sorted(kind_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "timeframe_counts": dict(sorted(timeframe_counts.items())),
        "asset_counts": dict(sorted(asset_counts.items())),
        "warning_series_count": len(warning_series),
        "warning_series_ids_first_300": warning_series[:300],
        "physical_inventory_summary": physical_inventory.get("summary", {}),
        "extension_summary": physical_inventory.get("extension_summary", {}),
    }


def build_ai_recommendations(summary: Dict[str, Any]) -> Dict[str, Any]:
    warnings = []

    if summary.get("warning_series_count", 0) > 0:
        warnings.append(
            "Existem séries com CHECK_WARNING ou erro. Revisar warning_series_ids_first_300 antes de uso produtivo."
        )

    if summary.get("total_unknown_series", 0) > 0:
        warnings.append(
            "Existem séries com dataset_kind unknown. Pode haver arquivos fora do padrão ou sem DateTime."
        )

    if summary.get("total_series_parquet_mapped", 0) == 0:
        warnings.append(
            "Nenhum Parquet foi mapeado. Verificar diretório de base."
        )

    return {
        "primary_use": (
            "Este JSON deve ser usado por agentes AI para localizar dados, selecionar ativos, "
            "entender cobertura histórica, escolher timeframes descobertos dinamicamente e evitar "
            "abrir todos os arquivos antes de planejar backtests e otimizações."
        ),
        "recommended_ai_reading_order": [
            "executive_summary_for_ai",
            "discovered_universe",
            "summary",
            "asset_matrix",
            "indexes.series_ids_by_asset",
            "indexes.series_ids_by_timeframe",
            "indexes.series_by_id.<series_id>.file.absolute_path",
            "series_catalog",
            "physical_inventory.files",
        ],
        "backtest_agent_guidance": [
            "Não assumir lista fixa de ativos ou timeframes; usar discovered_universe e indexes.",
            "Preferir séries com quality.status == OK.",
            "Usar usage.python_read_instruction para carregar Parquet.",
            "Filtrar por asset, source, timeframe e dataset_kind usando indexes.",
            "A periodicidade vem de periodicity.inferred_timeframe, calculada pelos deltas de DateTime.",
            "Evitar combinar séries com DateTime em zonas/políticas diferentes sem normalização.",
            "Considerar que DateTime está em Dubai local naive se a base seguir o padrão ARCHANGEL.",
            "Antes de otimização, revisar gap_count_sampled, duplicates_datetime e has_bad_ohlc.",
        ],
        "ml_dl_agent_guidance": [
            "Usar OHLCV como base primária de features quando dataset_kind == ohlcv.",
            "Usar funding_rate e open_interest como features auxiliares por alinhamento temporal.",
            "Evitar usar dados futuros no merge de features.",
            "Fazer joins com asof/backward ou resampling explícito para impedir lookahead bias.",
            "Usar apenas séries cuja periodicidade esteja clara ou tratar event_based separadamente.",
        ],
        "data_quality_warnings": warnings,
    }


# =============================================================================
# 9. PAYLOAD FINAL
# =============================================================================

def build_payload() -> Dict[str, Any]:
    base_data_dir = resolve_base_data_dir()

    print("=" * 90)
    print("ARCHANGEL | MAPA_ATIVOS | MAPEAMENTO DINÂMICO DA BASE")
    print("=" * 90)
    print(f"ROOT_DIR.........: {ROOT_DIR}")
    print(f"BASE_DATA_DIR....: {base_data_dir}")
    print(f"OUTPUT_JSON......: {OUTPUT_JSON_PATH}")
    print("-" * 90)

    physical_inventory = scan_files_and_directories(base_data_dir)
    series_catalog = build_series_catalog(physical_inventory, base_data_dir)

    discovered_universe = build_discovered_universe(series_catalog)
    indexes = build_indexes(series_catalog, physical_inventory)
    asset_matrix = build_asset_matrix(series_catalog)
    summary = build_summary(series_catalog, physical_inventory)
    ai_recommendations = build_ai_recommendations(summary)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": now_utc_iso(),
        "generated_at_local": now_local_iso(),
        "file_identity": {
            "script_name": "MAPA_ATIVOS.py",
            "output_file_name": OUTPUT_JSON_NAME,
            "output_file_path": str(OUTPUT_JSON_PATH),
            "root_dir": str(ROOT_DIR),
            "base_json_dir": str(BASE_JSON_DIR),
            "base_data_dir_detected": str(base_data_dir),
            "base_data_dir_candidates": [str(x) for x in BASE_DATA_DIR_CANDIDATES],
        },
        "design_policy": {
            "no_fixed_asset_list": True,
            "no_fixed_symbol_list": True,
            "no_fixed_timeframe_list": True,
            "source_scope_policy": "BINANCE_ONLY_CURRENTLY",
            "source_include_prefixes": list(SOURCE_INCLUDE_PREFIXES),
            "excluded_sources_note": (
                "Arquivos físicos de outras fontes podem existir no disco, mas são filtrados "
                "do series_catalog e dos indexes para não alimentar as etapas seguintes."
            ),
            "discovery_method": (
                "Ativos, símbolos, fontes, tipos de dados e periodicidades são descobertos "
                "dinamicamente lendo os arquivos e analisando colunas, conteúdo, caminhos e deltas de DateTime."
            ),
        },
        "purpose": (
            "Mapa completo e AI friendly da base de dados local do ARCHANGEL, "
            "incluindo arquivos, diretórios, séries, ativos descobertos, símbolos descobertos, "
            "fontes descobertas, periodicidade inferida, tamanhos, cobertura histórica e sanidade dos dados."
        ),
        "timezone_policy": {
            "timezone_reference": TIMEZONE_LOCAL,
            "datetime_policy_assumption": (
                "Quando a base segue o padrão ARCHANGEL, DateTime está salvo em horário local de Dubai, "
                "naive, sem timezone attached."
            ),
            "important_warning": (
                "Agentes AI devem verificar schema.datetime_column_detected e tratar DateTime explicitamente "
                "antes de misturar com UTC ou outras bases."
            ),
        },
        "executive_summary_for_ai": {
            "what_this_file_is": (
                "Índice mestre dinâmico da base local. Use este JSON para descobrir, sem listas manuais, "
                "quais ativos, símbolos, fontes, tipos de séries e periodicidades existem."
            ),
            "best_entry_points": [
                "discovered_universe",
                "summary",
                "asset_matrix",
                "indexes.series_ids_by_asset",
                "indexes.series_ids_by_timeframe",
                "indexes.series_by_id",
            ],
            "single_series_loading_pattern": (
                "series = data['indexes']['series_by_id'][series_id]; "
                "df = pd.read_parquet(series['file']['absolute_path'])"
            ),
            "quality_policy": (
                "Para backtests e ML, priorizar quality.status == OK. "
                "Séries CHECK_WARNING exigem inspeção antes de uso produtivo."
            ),
        },
        "discovered_universe": discovered_universe,
        "summary": summary,
        "ai_recommendations": ai_recommendations,
        "asset_matrix": asset_matrix,
        "indexes": indexes,
        "series_catalog": series_catalog,
        "physical_inventory": physical_inventory,
        "how_to_use": {
            "load_this_json": (
                "import json\n"
                "from pathlib import Path\n"
                f"path = Path(r'{str(OUTPUT_JSON_PATH)}')\n"
                "data = json.loads(path.read_text(encoding='utf-8'))"
            ),
            "list_discovered_assets": (
                "assets = data['discovered_universe']['assets_discovered']"
            ),
            "list_discovered_timeframes": (
                "timeframes = data['discovered_universe']['timeframes_discovered']"
            ),
            "load_one_series_example": (
                "series_id = next(iter(data['indexes']['series_by_id']))\n"
                "series = data['indexes']['series_by_id'][series_id]\n"
                "df = pd.read_parquet(series['file']['absolute_path'])"
            ),
            "filter_ok_ohlcv_dynamic_example": (
                "series = [s for s in data['series_catalog'] "
                "if s['dataset_kind']=='ohlcv' and s['quality']['status']=='OK']"
            ),
        },
    }

    return payload


# =============================================================================
# 10. SALVAMENTO
# =============================================================================

def save_json(payload: Dict[str, Any], output_path: Path) -> None:
    ensure_dir(output_path.parent)

    tmp_path = output_path.with_suffix(".tmp.json")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    os.replace(tmp_path, output_path)


# =============================================================================
# 11. MAIN
# =============================================================================

def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR não encontrado: {ROOT_DIR}")

    ensure_dir(BASE_JSON_DIR)

    payload = build_payload()
    save_json(payload, OUTPUT_JSON_PATH)

    print("-" * 90)
    print("MAPA_ATIVOS DINÂMICO CONCLUÍDO")
    print("-" * 90)
    print(f"JSON salvo em: {OUTPUT_JSON_PATH}")
    print("")
    print("Universo descoberto:")
    print(json.dumps(payload.get("discovered_universe", {}), ensure_ascii=False, indent=2))
    print("")
    print("Resumo:")
    print(json.dumps(payload.get("summary", {}), ensure_ascii=False, indent=2))
    print("=" * 90)


if __name__ == "__main__":
    main()
