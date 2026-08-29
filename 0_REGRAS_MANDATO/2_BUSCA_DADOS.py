# -*- coding: utf-8 -*-

"""
ARCHANGEL CRYPTO DATA ENGINE - BINANCE ONLY - OTIMIZADO
=======================================================

Objetivo:
    Baixar, atualizar, limpar, validar e salvar bases locais para:
    - Trading systems
    - Backtests
    - Machine Learning
    - Deep Learning

Inclui:
    - OHLCV Binance Spot
    - Séries customizadas: 3min, 7min, 13min, 23min, 37min, 47min, 3D, 7D
    - Timezone Dubai
    - timestamp_utc_ms universal interno
    - Bar timestamp policy = close_time
    - Salvamento Parquet em <PROJECT_ROOT>\\2_BASES
    - Atualização incremental
    - Checks por amostragem
    - Catálogo JSON
    - Diagnóstico Parquet
    - Paralelismo moderado por ativo/fonte
    - Custom timeframes gerados com uma única leitura do 1min

Política temporal:
    - DateTime é salvo como horário local Asia/Dubai, naive, sem timezone attached.
    - timestamp_utc_ms é salvo como referência universal UTC em milissegundos.
    - OHLCV usa timestamp de fechamento da barra: bar_timestamp_policy = close_time.
    - Features, labels, ML e backtest devem usar timestamp_utc_ms como chave temporal interna.

Integridade:
    - Escrita atômica com arquivo temporário
    - Deduplicação por timestamp_utc_ms
    - Overlap incremental
    - Limpeza OHLCV
    - Validação por amostragem
    - Lock no catálogo em modo paralelo
    - Nenhuma thread escreve no mesmo arquivo simultaneamente

Inventário AI:
    - O inventário completo deve ser gerado pelo MAPA_ATIVOS.py separado.
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, local

import requests
from requests.adapters import HTTPAdapter
import numpy as np
import pandas as pd


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "sim", "on"}


def env_csv_list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or list(default)


# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================

# A raiz é inferida do local deste script, sem depender da letra da unidade.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = str(PROJECT_ROOT / "2_BASES")
RULES_DIR = PROJECT_ROOT / "0_REGRAS_MANDATO"
BASE_JSON_DIR = RULES_DIR / "BASE_JSON"

SCRIPT_NAME = "2_BUSCA_DADOS.py"
SYSTEM_NAME = "ARCHANGEL"
SYSTEM_VERSION = "v1"
RUN_STATE_SCHEMA_VERSION = "ARCHANGEL_RUN_STATE_1.0"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

TIMEZONE_LOCAL = "Asia/Dubai"

# Política temporal ARCHANGEL
DATETIME_COL = "DateTime"
TIMESTAMP_UTC_MS_COL = "timestamp_utc_ms"

META_TIMEZONE = TIMEZONE_LOCAL
META_DATETIME_IS_NAIVE = True
META_TIMESTAMP_UTC_MS = True

# Crítico para evitar look-ahead em candles agregados.
# O timestamp do candle representa o momento em que a barra fecha.
BAR_TIMESTAMP_POLICY = "close_time"

SAVE_PARQUET = True
SAVE_CSV_DEBUG = False

REQUEST_TIMEOUT = 25
RETRY_ATTEMPTS = 4
RETRY_SLEEP_BASE = 1.5

DOWNLOAD_START_UTC = "2017-01-01 00:00:00"
DOWNLOAD_UNTIL_NOW = True
DOWNLOAD_END_UTC = None

INCREMENTAL_OVERLAP_BARS = 60
USE_INCREMENTAL_UPDATE = True

SAMPLE_CHECK_ROWS = 50

# Use None para produção completa.
# Para teste rápido, use por exemplo: 5000
MAX_TOTAL_BARS_PER_SERIES = None

# Mostra log a cada N batches.
PROGRESS_EVERY_N_BATCHES = 5


# =============================================================================
# PERFORMANCE / SEGURANÇA
# =============================================================================

ENABLE_PARALLEL_DOWNLOADS = True

# Paralelismo por ativo. Com Binance-only e 12 ativos, acima de 12 raramente ajuda.
MAX_WORKERS_DOWNLOAD = int(os.environ.get("ARCHANGEL_DOWNLOAD_WORKERS", "12"))

# Modo seguro: baixa todos os timeframes padrão diretamente da Binance.
# Timeframes customizados continuam sendo gerados localmente a partir do 1min.
BINANCE_NATIVE_TIMEFRAMES_TO_DOWNLOAD = env_csv_list(
    "ARCHANGEL_BINANCE_NATIVE_TIMEFRAMES",
    ["1min", "5min", "15min", "1h", "4h", "1D"],
)

# Gera todos os custom timeframes com uma única leitura do 1min.
ENABLE_FAST_CUSTOM_TIMEFRAMES = True

# O inventário completo agora deve ser feito pelo MAPA_ATIVOS.py.
GENERATE_AI_INVENTORY_IN_ENGINE = False

# Pausa entre batches HTTP. O retry/backoff segura 429/418; use 0.02-0.08 se houver rate limit.
HTTP_SLEEP_BETWEEN_BATCHES = float(os.environ.get("ARCHANGEL_HTTP_SLEEP_BETWEEN_BATCHES", "0"))
HTTP_POOL_CONNECTIONS = int(os.environ.get("ARCHANGEL_HTTP_POOL_CONNECTIONS", str(max(16, MAX_WORKERS_DOWNLOAD * 2))))
HTTP_POOL_MAXSIZE = int(os.environ.get("ARCHANGEL_HTTP_POOL_MAXSIZE", str(max(32, MAX_WORKERS_DOWNLOAD * 4))))

# Parquet rápido e estável.
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"


# =============================================================================
# TIMEFRAMES / ATIVOS
# =============================================================================

NATIVE_TIMEFRAMES = [
    "1min",
    "5min",
    "15min",
    "1h",
    "4h",
    "1D",
]

CUSTOM_TIMEFRAMES = [
    "3min",
    "7min",
    "13min",
    "23min",
    "37min",
    "47min",
    "3D",
    "7D",
]

RESAMPLE_FROM_1MIN_TIMEFRAMES = env_csv_list(
    "ARCHANGEL_RESAMPLE_FROM_1MIN_TIMEFRAMES",
    CUSTOM_TIMEFRAMES,
)

ENABLE_BINANCE_SPOT = True
ENABLE_BYBIT_LINEAR = False

ENABLE_BYBIT_FUNDING = False
ENABLE_BYBIT_OPEN_INTEREST = False

BYBIT_OPEN_INTEREST_TIMEFRAME = "1h"

SYMBOLS = {
    "BTC": {"binance_spot": "BTCUSDT"},
    "ETH": {"binance_spot": "ETHUSDT"},
    "SOL": {"binance_spot": "SOLUSDT"},
    "BNB": {"binance_spot": "BNBUSDT"},
    "DOGE": {"binance_spot": "DOGEUSDT"},
    "AVAX": {"binance_spot": "AVAXUSDT"},
    "LINK": {"binance_spot": "LINKUSDT"},
    "XRP": {"binance_spot": "XRPUSDT"},
    "LTC": {"binance_spot": "LTCUSDT"},
    "ADA": {"binance_spot": "ADAUSDT"},
    "DOT": {"binance_spot": "DOTUSDT"},
    "NEAR": {"binance_spot": "NEARUSDT"},
}


# =============================================================================
# MAPAS DE TIMEFRAME
# =============================================================================

BINANCE_INTERVAL_MAP = {
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1D": "1d",
    "1W": "1w",
    "1M": "1M",
}

BYBIT_INTERVAL_MAP = {
    "1min": "1",
    "3min": "3",
    "5min": "5",
    "15min": "15",
    "30min": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "1D": "D",
    "1W": "W",
    "1M": "M",
}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> None:
    Path(BASE_DATA_DIR).mkdir(parents=True, exist_ok=True)
    log_dir = Path(BASE_DATA_DIR) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"archangel_data_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logging.info(f"Log file: {log_file}")


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def format_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60

    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"

    return f"{m}m {s:02d}s"


def utc_now_ms() -> int:
    return int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)


def parse_utc_to_ms(value: Optional[str]) -> Optional[int]:
    if not value:
        return None

    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None

    return int(ts.timestamp() * 1000)


def ms_to_utc_str(ms: int) -> str:
    try:
        return str(pd.to_datetime(ms, unit="ms", utc=True))
    except Exception:
        return "N/A"


def ms_to_dubai_str(ms: int) -> str:
    try:
        return str(pd.to_datetime(ms, unit="ms", utc=True).tz_convert(TIMEZONE_LOCAL))
    except Exception:
        return "N/A"


def normalize_timestamp_to_utc_ms(series: pd.Series) -> pd.Series:
    """Normaliza timestamps Unix em s/ms/us/ns para UTC em milissegundos."""
    values = pd.to_numeric(series, errors="coerce")

    if values.dropna().empty:
        return pd.Series(pd.NA, index=series.index, dtype="Int64")

    median_abs = float(values.dropna().abs().median())

    if median_abs >= 1e17:      # nanossegundos
        normalized = (values / 1_000_000).round()
    elif median_abs >= 1e14:    # microssegundos
        normalized = (values / 1_000).round()
    elif median_abs >= 1e11:    # milissegundos
        normalized = values.round()
    elif median_abs >= 1e8:     # segundos
        normalized = (values * 1_000).round()
    else:
        normalized = pd.Series(pd.NA, index=series.index)

    return normalized.astype("Int64")


def utc_ms_to_dubai_naive(ms_series: pd.Series) -> pd.Series:
    """
    Converte timestamp UTC em milissegundos para DateTime Dubai naive.
    """
    dt_utc = pd.to_datetime(ms_series, unit="ms", utc=True, errors="coerce")
    return dt_utc.dt.tz_convert(TIMEZONE_LOCAL).dt.tz_localize(None)


def datetime_dubai_naive_to_utc_ms(series: pd.Series) -> pd.Series:
    """
    Converte DateTime Dubai naive para timestamp UTC em milissegundos.
    """
    dt = pd.to_datetime(series, errors="coerce")

    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt_utc = dt.dt.tz_convert("UTC")
        else:
            dt_utc = dt.dt.tz_localize(TIMEZONE_LOCAL).dt.tz_convert("UTC")

        # Pandas pode representar datetimes em ns, us ou ms. Normalizar pela
        # magnitude evita assumir incorretamente que o epoch interno é sempre ns.
        return normalize_timestamp_to_utc_ms(dt_utc.astype("int64"))

    except Exception:
        return pd.Series(pd.NA, index=series.index, dtype="Int64")


def add_time_columns_from_utc_ms(
    df: pd.DataFrame,
    timestamp_col: str,
    output_datetime_col: str = DATETIME_COL,
    output_timestamp_col: str = TIMESTAMP_UTC_MS_COL,
) -> pd.DataFrame:
    """
    Adiciona/normaliza:
        - timestamp_utc_ms: UTC universal
        - DateTime: Asia/Dubai naive

    timestamp_col deve estar em milissegundos UTC.
    """
    out = df.copy()

    out[output_timestamp_col] = normalize_timestamp_to_utc_ms(out[timestamp_col])
    out[output_datetime_col] = utc_ms_to_dubai_naive(out[output_timestamp_col])

    out = out.dropna(subset=[output_datetime_col, output_timestamp_col])
    out[output_timestamp_col] = out[output_timestamp_col].astype("int64")

    return out


def add_ohlcv_close_time_columns_from_open_ms(
    df: pd.DataFrame,
    open_time_col: str,
    timeframe: str,
    output_datetime_col: str = DATETIME_COL,
    output_timestamp_col: str = TIMESTAMP_UTC_MS_COL,
) -> pd.DataFrame:
    """
    Para OHLCV, cria timestamp de fechamento da barra.

    Política:
        BAR_TIMESTAMP_POLICY = close_time

    Se a exchange fornece OpenTime, usamos:
        close_boundary_ms = OpenTime + timeframe_ms

    Esse timestamp representa o momento em que o candle terminou e se tornou conhecido.
    """
    out = df.copy()

    open_ms = pd.to_numeric(out[open_time_col], errors="coerce")
    close_boundary_ms = open_ms + timeframe_to_ms(timeframe)

    out[output_timestamp_col] = close_boundary_ms.astype("Int64")
    out[output_datetime_col] = utc_ms_to_dubai_naive(out[output_timestamp_col])

    out = out.dropna(subset=[output_datetime_col, output_timestamp_col])
    out[output_timestamp_col] = out[output_timestamp_col].astype("int64")

    return out


def ensure_archangel_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que todo dataframe salvo pelo engine tenha:
        - DateTime
        - timestamp_utc_ms

    Se timestamp_utc_ms não existir mas DateTime existir, assume DateTime como Dubai naive.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    timestamp_is_usable = False

    if TIMESTAMP_UTC_MS_COL in out.columns:
        normalized_ts = normalize_timestamp_to_utc_ms(out[TIMESTAMP_UTC_MS_COL])
        timestamp_is_usable = bool(normalized_ts.notna().mean() >= 0.90)
        if timestamp_is_usable:
            out[TIMESTAMP_UTC_MS_COL] = normalized_ts

    if DATETIME_COL in out.columns:
        out[DATETIME_COL] = pd.to_datetime(out[DATETIME_COL], errors="coerce")

    if DATETIME_COL in out.columns and not timestamp_is_usable:
        rebuilt_ts = datetime_dubai_naive_to_utc_ms(out[DATETIME_COL])
        if rebuilt_ts.notna().mean() >= 0.90:
            out[TIMESTAMP_UTC_MS_COL] = rebuilt_ts
            timestamp_is_usable = True

    if DATETIME_COL not in out.columns and timestamp_is_usable:
        out[DATETIME_COL] = utc_ms_to_dubai_naive(out[TIMESTAMP_UTC_MS_COL])

    if DATETIME_COL in out.columns:
        out[DATETIME_COL] = pd.to_datetime(out[DATETIME_COL], errors="coerce")

    if TIMESTAMP_UTC_MS_COL in out.columns:
        out[TIMESTAMP_UTC_MS_COL] = normalize_timestamp_to_utc_ms(out[TIMESTAMP_UTC_MS_COL])

    if DATETIME_COL in out.columns and TIMESTAMP_UTC_MS_COL in out.columns:
        out = out.dropna(subset=[DATETIME_COL, TIMESTAMP_UTC_MS_COL])
        out[TIMESTAMP_UTC_MS_COL] = out[TIMESTAMP_UTC_MS_COL].astype("int64")

    return out


def get_time_policy_metadata() -> Dict[str, Any]:
    return {
        "meta_timezone": META_TIMEZONE,
        "meta_datetime_is_naive": META_DATETIME_IS_NAIVE,
        "meta_timestamp_utc_ms": META_TIMESTAMP_UTC_MS,
        "datetime_column": DATETIME_COL,
        "timestamp_utc_ms_column": TIMESTAMP_UTC_MS_COL,
        "bar_timestamp_policy": BAR_TIMESTAMP_POLICY,
        "datetime_policy": (
            "DateTime is stored as Asia/Dubai local naive for readability. "
            "timestamp_utc_ms is stored as UTC milliseconds and must be used as "
            "the universal internal time key for ML, labels, backtests and execution."
        ),
    }


def timeframe_to_pandas_freq(tf: str) -> str:
    tf = tf.strip()

    if tf.endswith("min"):
        n = int(tf.replace("min", ""))
        return f"{n}min"

    if tf.endswith("h"):
        n = int(tf.replace("h", ""))
        return f"{n}h"

    if tf.endswith("D"):
        n = tf.replace("D", "")
        n = int(n) if n else 1
        return f"{n}D"

    if tf.endswith("W"):
        n = tf.replace("W", "")
        n = int(n) if n else 1
        return f"{n}W"

    if tf.endswith("M"):
        return "MS"

    raise ValueError(f"Timeframe não suportado para pandas freq: {tf}")


def timeframe_to_ms(tf: str) -> int:
    freq = timeframe_to_pandas_freq(tf)

    if freq == "MS":
        return 30 * 24 * 60 * 60 * 1000

    return int(pd.Timedelta(freq).total_seconds() * 1000)


def timeframe_dirname(tf: str) -> str:
    return (
        tf.replace("min", "_min")
          .replace("h", "_hour")
          .replace("D", "_day")
          .replace("W", "_week")
          .replace("M", "_month")
    )


def source_dirname(source: str) -> str:
    return source.lower().replace(" ", "_")


def to_dubai_naive(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")

    try:
        if getattr(dt.dt, "tz", None) is not None:
            return dt.dt.tz_convert(TIMEZONE_LOCAL).dt.tz_localize(None)

        return dt

    except Exception:
        return pd.to_datetime(series, errors="coerce")


def normalize_datetime_dubai(df: pd.DataFrame, col: str = DATETIME_COL) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df

    out = df.copy()
    out[col] = to_dubai_naive(out[col])
    out = ensure_archangel_time_columns(out)

    if col in out.columns:
        out = out.dropna(subset=[col])

    if TIMESTAMP_UTC_MS_COL in out.columns:
        out = out.sort_values(TIMESTAMP_UTC_MS_COL).reset_index(drop=True)
    else:
        out = out.sort_values(col).reset_index(drop=True)

    return out


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = ensure_archangel_time_columns(df)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    required_drop = [DATETIME_COL, "Open", "High", "Low", "Close"]
    if TIMESTAMP_UTC_MS_COL in out.columns:
        required_drop.append(TIMESTAMP_UTC_MS_COL)

    out = out.dropna(subset=required_drop)

    out = out[
        (out["Open"] > 0) &
        (out["High"] > 0) &
        (out["Low"] > 0) &
        (out["Close"] > 0)
    ]

    out = out[
        (out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)) &
        (out["Low"] <= out[["Open", "Close", "High"]].min(axis=1))
    ]

    out = out[out["Volume"].fillna(0) >= 0]

    dedup_col = TIMESTAMP_UTC_MS_COL if TIMESTAMP_UTC_MS_COL in out.columns else DATETIME_COL
    out = out.drop_duplicates(subset=[dedup_col], keep="last")

    sort_col = TIMESTAMP_UTC_MS_COL if TIMESTAMP_UTC_MS_COL in out.columns else DATETIME_COL
    out = out.sort_values(sort_col).reset_index(drop=True)

    return out


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = [DATETIME_COL, TIMESTAMP_UTC_MS_COL, "Open", "High", "Low", "Close", "Volume"]

    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    out = df.copy()

    rename = {}
    for c in out.columns:
        cl = str(c).lower()

        if cl in ["datetime", "date", "time"]:
            rename[c] = DATETIME_COL
        elif cl in ["timestamp_utc_ms", "utc_timestamp_ms"]:
            rename[c] = TIMESTAMP_UTC_MS_COL
        elif cl == "timestamp":
            if TIMESTAMP_UTC_MS_COL not in out.columns:
                rename[c] = DATETIME_COL
        elif cl == "open":
            rename[c] = "Open"
        elif cl == "high":
            rename[c] = "High"
        elif cl == "low":
            rename[c] = "Low"
        elif cl == "close":
            rename[c] = "Close"
        elif cl in ["volume", "vol"]:
            rename[c] = "Volume"

    out = out.rename(columns=rename)

    required = [DATETIME_COL, "Open", "High", "Low", "Close", "Volume"]
    for c in required:
        if c not in out.columns:
            out[c] = np.nan

    out = ensure_archangel_time_columns(out)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = clean_ohlcv(out)

    for c in cols:
        if c not in out.columns:
            out[c] = np.nan

    return out[cols].copy()


def standardize_metric_df(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = normalize_datetime_dubai(df, DATETIME_COL)

    for c in value_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    dedup_col = TIMESTAMP_UTC_MS_COL if TIMESTAMP_UTC_MS_COL in out.columns else DATETIME_COL
    out = out.drop_duplicates(subset=[dedup_col], keep="last")

    sort_col = TIMESTAMP_UTC_MS_COL if TIMESTAMP_UTC_MS_COL in out.columns else DATETIME_COL
    out = out.sort_values(sort_col).reset_index(drop=True)

    return out


def df_hash_sample(df: pd.DataFrame, n: int = SAMPLE_CHECK_ROWS) -> str:
    if df is None or df.empty:
        return "EMPTY"

    sample = df.copy()

    if len(sample) > n:
        idx = sorted(set(
            list(np.linspace(0, len(sample) - 1, min(n, len(sample))).astype(int))
        ))
        sample = sample.iloc[idx].copy()

    text = sample.to_csv(index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_period(df: Optional[pd.DataFrame]) -> Tuple[Optional[str], Optional[str]]:
    if df is None or df.empty:
        return None, None

    if TIMESTAMP_UTC_MS_COL in df.columns:
        ts = pd.to_numeric(df[TIMESTAMP_UTC_MS_COL], errors="coerce").dropna()
        if ts.empty:
            return None, None

        start = ms_to_dubai_str(int(ts.min()))
        end = ms_to_dubai_str(int(ts.max()))
        return start, end

    if DATETIME_COL not in df.columns:
        return None, None

    dt = pd.to_datetime(df[DATETIME_COL], errors="coerce").dropna()
    if dt.empty:
        return None, None

    return str(dt.min()), str(dt.max())


def safe_read_parquet(path: str | Path) -> Optional[pd.DataFrame]:
    try:
        p = Path(path)

        if not p.exists():
            return None

        df = pd.read_parquet(p, engine=PARQUET_ENGINE)

        if df is None or df.empty:
            return None

        return ensure_archangel_time_columns(df)

    except Exception as e:
        logging.warning(f"Erro lendo parquet {path}: {e}")
        return None


def atomic_save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """
    Salvamento atômico.
    Segurança:
        1. Grava em .tmp.parquet.
        2. Apenas depois substitui o arquivo final.
        3. Se falhar no meio, o arquivo antigo permanece intacto.
    """
    p = Path(path)
    ensure_dir(p.parent)

    tmp = p.with_suffix(".tmp.parquet")

    out = ensure_archangel_time_columns(df)

    out.to_parquet(
        tmp,
        index=False,
        engine=PARQUET_ENGINE,
        compression=PARQUET_COMPRESSION,
    )

    os.replace(tmp, p)


def atomic_save_json(data: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)

    tmp = p.with_suffix(".tmp.json")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    os.replace(tmp, p)


def estimate_expected_batches(start_ms: int, end_ms: int, timeframe: str, limit_per_request: int) -> int:
    try:
        interval_ms = timeframe_to_ms(timeframe)
        total_bars = max(0, int((end_ms - start_ms) / interval_ms))
        return max(1, int(np.ceil(total_bars / limit_per_request)))
    except Exception:
        return 0


def file_size_mb(path: str | Path) -> Optional[float]:
    try:
        p = Path(path)

        if not p.exists() or not p.is_file():
            return None

        return round(p.stat().st_size / (1024 * 1024), 6)

    except Exception:
        return None


def file_modified_at(path: str | Path) -> Optional[str]:
    try:
        p = Path(path)

        if not p.exists():
            return None

        return datetime.fromtimestamp(p.stat().st_mtime).isoformat()

    except Exception:
        return None


def safe_relative_path(path: str | Path, base: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(path)


# =============================================================================
# PATH MANAGER
# =============================================================================

class PathManager:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.raw = self.base / "raw"
        self.processed = self.base / "processed"
        self.custom = self.base / "custom_timeframes"
        self.metadata = self.base / "metadata"
        self.diagnostics = self.base / "diagnostics"

        for p in [
            self.base,
            self.raw,
            self.processed,
            self.custom,
            self.metadata,
            self.diagnostics,
        ]:
            ensure_dir(p)

    def ohlcv_path(self, source: str, asset: str, symbol: str, timeframe: str, custom: bool = False) -> Path:
        root = self.custom if custom else self.processed

        return (
            root /
            "ohlcv" /
            source_dirname(source) /
            asset /
            timeframe_dirname(timeframe) /
            f"{asset}_{source}_{symbol}_{timeframe}.parquet"
        )

    def metric_path(self, metric: str, source: str, asset: str, symbol: str) -> Path:
        return (
            self.processed /
            metric /
            source_dirname(source) /
            asset /
            f"{asset}_{source}_{symbol}_{metric}.parquet"
        )

    def catalog_path(self) -> Path:
        return BASE_JSON_DIR / "CATALOGO_ARCHANGEL_SERIES.json"

    def run_state_path(self) -> Path:
        return BASE_JSON_DIR / "RUN_STATE.json"

    def diagnostics_path(self) -> Path:
        return self.diagnostics / f"DIAGNOSTICO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"


# =============================================================================
# HTTP CLIENT
# =============================================================================

class HttpClient:
    def __init__(self):
        self._local = local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=HTTP_POOL_CONNECTIONS,
                pool_maxsize=HTTP_POOL_MAXSIZE,
                max_retries=0,
                pool_block=False,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
        return session

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        last_exc = None

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                r = self._session().get(url, params=params, timeout=REQUEST_TIMEOUT)

                if r.status_code in [418, 429, 500, 502, 503, 504]:
                    wait = max(RETRY_SLEEP_BASE * attempt, 2.0)
                    logging.warning(
                        f"Rate/Server issue {r.status_code} | "
                        f"attempt={attempt}/{RETRY_ATTEMPTS} | sleep={wait:.1f}s | "
                        f"url={url} | params={params}"
                    )
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                return r.json()

            except Exception as e:
                last_exc = e
                wait = RETRY_SLEEP_BASE * attempt
                logging.warning(
                    f"HTTP fail attempt {attempt}/{RETRY_ATTEMPTS}: {e} | "
                    f"sleep={wait:.1f}s | url={url} | params={params}"
                )
                time.sleep(wait)

        raise RuntimeError(f"HTTP failed after retries: {url} | {last_exc}")


# =============================================================================
# BINANCE CLIENT
# =============================================================================

class BinanceClient:
    BASE = "https://api.binance.com"
    LIMIT = 1000

    def __init__(self, http: HttpClient):
        self.http = http

    def fetch_ohlcv(self, asset: str, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        if timeframe not in BINANCE_INTERVAL_MAP:
            raise ValueError(f"Binance não suporta timeframe nativo: {timeframe}")

        interval = BINANCE_INTERVAL_MAP[timeframe]
        interval_ms = timeframe_to_ms(timeframe)

        rows = []
        current = start_ms
        total = 0
        batch_no = 0
        t0 = time.time()

        estimated_batches = estimate_expected_batches(start_ms, end_ms, timeframe, self.LIMIT)

        logging.info(
            f"[BINANCE START] {asset} | {symbol} | {timeframe} | "
            f"from_utc={ms_to_utc_str(start_ms)} | to_utc={ms_to_utc_str(end_ms)} | "
            f"bar_timestamp_policy={BAR_TIMESTAMP_POLICY} | "
            f"estimated_batches≈{estimated_batches}"
        )

        while current <= end_ms:
            if MAX_TOTAL_BARS_PER_SERIES is not None and total >= MAX_TOTAL_BARS_PER_SERIES:
                logging.warning(
                    f"[BINANCE LIMIT] {asset} | {symbol} | {timeframe} | "
                    f"MAX_TOTAL_BARS_PER_SERIES atingido: {MAX_TOTAL_BARS_PER_SERIES}"
                )
                break

            limit = self.LIMIT

            if MAX_TOTAL_BARS_PER_SERIES is not None:
                limit = min(limit, MAX_TOTAL_BARS_PER_SERIES - total)

            batch_no += 1

            data = self.http.get_json(
                f"{self.BASE}/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": limit,
                }
            )

            if not data:
                logging.info(f"[BINANCE EMPTY] {asset} | {symbol} | {timeframe} | batch={batch_no}")
                break

            rows.extend(data)
            total += len(data)

            newest_open = int(data[-1][0])
            newest_close_boundary = newest_open + interval_ms
            next_start = newest_open + interval_ms

            if batch_no == 1 or batch_no % PROGRESS_EVERY_N_BATCHES == 0:
                pct = 0.0

                if end_ms > start_ms:
                    pct = min(100.0, ((newest_open - start_ms) / (end_ms - start_ms)) * 100.0)

                logging.info(
                    f"[BINANCE PROGRESS] {asset} | {symbol} | {timeframe} | "
                    f"batch={batch_no}/{estimated_batches} | rows={total:,} | "
                    f"latest_open_utc={ms_to_utc_str(newest_open)} | "
                    f"latest_close_utc={ms_to_utc_str(newest_close_boundary)} | "
                    f"latest_close_dubai={ms_to_dubai_str(newest_close_boundary)} | "
                    f"progress≈{pct:.2f}% | elapsed={format_seconds(time.time() - t0)}"
                )

            if next_start <= current:
                logging.warning(f"[BINANCE STOP] next_start <= current | {asset} | {symbol} | {timeframe}")
                break

            current = next_start

            if len(data) < limit:
                logging.info(
                    f"[BINANCE DONE] Último batch menor que limit | "
                    f"{asset} | {symbol} | {timeframe} | batch_rows={len(data)}"
                )
                break

            time.sleep(HTTP_SLEEP_BETWEEN_BATCHES)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "OpenTime", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteVolume", "Trades",
            "TakerBuyBaseVolume", "TakerBuyQuoteVolume", "Ignore"
        ])

        df = add_ohlcv_close_time_columns_from_open_ms(
            df=df,
            open_time_col="OpenTime",
            timeframe=timeframe,
        )

        out = standardize_ohlcv(
            df[[DATETIME_COL, TIMESTAMP_UTC_MS_COL, "Open", "High", "Low", "Close", "Volume"]]
        )

        logging.info(
            f"[BINANCE FINISH] {asset} | {symbol} | {timeframe} | "
            f"rows_clean={len(out):,} | elapsed={format_seconds(time.time() - t0)}"
        )

        return out


# =============================================================================
# BYBIT CLIENT
# =============================================================================

class BybitClient:
    BASE = "https://api.bybit.com"
    KLINE_LIMIT = 1000
    FUNDING_LIMIT = 200
    OI_LIMIT = 200

    def __init__(self, http: HttpClient):
        self.http = http

    def fetch_ohlcv(self, asset: str, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        if timeframe not in BYBIT_INTERVAL_MAP:
            raise ValueError(f"Bybit não suporta timeframe nativo: {timeframe}")

        interval = BYBIT_INTERVAL_MAP[timeframe]
        interval_ms = timeframe_to_ms(timeframe)

        rows = []
        current = start_ms
        total = 0
        batch_no = 0
        t0 = time.time()

        estimated_batches = estimate_expected_batches(start_ms, end_ms, timeframe, self.KLINE_LIMIT)

        logging.info(
            f"[BYBIT KLINE START] {asset} | {symbol} | {timeframe} | "
            f"from_utc={ms_to_utc_str(start_ms)} | to_utc={ms_to_utc_str(end_ms)} | "
            f"bar_timestamp_policy={BAR_TIMESTAMP_POLICY} | "
            f"estimated_batches≈{estimated_batches}"
        )

        while current <= end_ms:
            if MAX_TOTAL_BARS_PER_SERIES is not None and total >= MAX_TOTAL_BARS_PER_SERIES:
                logging.warning(
                    f"[BYBIT KLINE LIMIT] {asset} | {symbol} | {timeframe} | "
                    f"MAX_TOTAL_BARS_PER_SERIES atingido: {MAX_TOTAL_BARS_PER_SERIES}"
                )
                break

            batch_no += 1

            data = self.http.get_json(
                f"{self.BASE}/v5/market/kline",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "start": current,
                    "end": end_ms,
                    "limit": self.KLINE_LIMIT,
                }
            )

            result = data.get("result", {})
            batch = result.get("list", [])

            if not batch:
                logging.info(f"[BYBIT KLINE EMPTY] {asset} | {symbol} | {timeframe} | batch={batch_no}")
                break

            batch = sorted(batch, key=lambda x: int(x[0]))
            rows.extend(batch)
            total += len(batch)

            newest_open = int(batch[-1][0])
            newest_close_boundary = newest_open + interval_ms
            next_start = newest_open + interval_ms

            if batch_no == 1 or batch_no % PROGRESS_EVERY_N_BATCHES == 0:
                pct = 0.0

                if end_ms > start_ms:
                    pct = min(100.0, ((newest_open - start_ms) / (end_ms - start_ms)) * 100.0)

                logging.info(
                    f"[BYBIT KLINE PROGRESS] {asset} | {symbol} | {timeframe} | "
                    f"batch={batch_no}/{estimated_batches} | rows={total:,} | "
                    f"latest_open_utc={ms_to_utc_str(newest_open)} | "
                    f"latest_close_utc={ms_to_utc_str(newest_close_boundary)} | "
                    f"latest_close_dubai={ms_to_dubai_str(newest_close_boundary)} | "
                    f"progress≈{pct:.2f}% | elapsed={format_seconds(time.time() - t0)}"
                )

            if next_start <= current:
                logging.warning(f"[BYBIT KLINE STOP] next_start <= current | {asset} | {symbol} | {timeframe}")
                break

            current = next_start

            if len(batch) < self.KLINE_LIMIT:
                logging.info(
                    f"[BYBIT KLINE DONE] Último batch menor que limit | "
                    f"{asset} | {symbol} | {timeframe} | batch_rows={len(batch)}"
                )
                break

            time.sleep(HTTP_SLEEP_BETWEEN_BATCHES)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "OpenTime", "Open", "High", "Low", "Close", "Volume", "Turnover"
        ])

        df = add_ohlcv_close_time_columns_from_open_ms(
            df=df,
            open_time_col="OpenTime",
            timeframe=timeframe,
        )

        out = standardize_ohlcv(
            df[[DATETIME_COL, TIMESTAMP_UTC_MS_COL, "Open", "High", "Low", "Close", "Volume"]]
        )

        logging.info(
            f"[BYBIT KLINE FINISH] {asset} | {symbol} | {timeframe} | "
            f"rows_clean={len(out):,} | elapsed={format_seconds(time.time() - t0)}"
        )

        return out

    def fetch_funding_rate(self, asset: str, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        rows = []
        current = start_ms
        batch_no = 0
        t0 = time.time()

        logging.info(
            f"[BYBIT FUNDING START] {asset} | {symbol} | "
            f"from_utc={ms_to_utc_str(start_ms)} | to_utc={ms_to_utc_str(end_ms)}"
        )

        while current <= end_ms:
            batch_no += 1

            data = self.http.get_json(
                f"{self.BASE}/v5/market/funding/history",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": self.FUNDING_LIMIT,
                }
            )

            result = data.get("result", {})
            batch = result.get("list", [])

            if not batch:
                logging.info(f"[BYBIT FUNDING EMPTY] {asset} | {symbol} | batch={batch_no}")
                break

            batch = sorted(batch, key=lambda x: int(x["fundingRateTimestamp"]))
            rows.extend(batch)

            newest = int(batch[-1]["fundingRateTimestamp"])
            next_start = newest + 1

            if batch_no == 1 or batch_no % PROGRESS_EVERY_N_BATCHES == 0:
                pct = 0.0

                if end_ms > start_ms:
                    pct = min(100.0, ((newest - start_ms) / (end_ms - start_ms)) * 100.0)

                logging.info(
                    f"[BYBIT FUNDING PROGRESS] {asset} | {symbol} | "
                    f"batch={batch_no} | rows={len(rows):,} | latest_utc={ms_to_utc_str(newest)} | "
                    f"progress≈{pct:.2f}% | elapsed={format_seconds(time.time() - t0)}"
                )

            if next_start <= current:
                break

            current = next_start

            if len(batch) < self.FUNDING_LIMIT:
                break

            time.sleep(HTTP_SLEEP_BETWEEN_BATCHES)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        df = add_time_columns_from_utc_ms(
            df=df,
            timestamp_col="fundingRateTimestamp",
        )

        df["FundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df["Symbol"] = symbol

        out = standardize_metric_df(
            df[[DATETIME_COL, TIMESTAMP_UTC_MS_COL, "Symbol", "FundingRate"]],
            ["FundingRate"]
        )

        logging.info(
            f"[BYBIT FUNDING FINISH] {asset} | {symbol} | "
            f"rows_clean={len(out):,} | elapsed={format_seconds(time.time() - t0)}"
        )

        return out

    def fetch_open_interest(
        self,
        asset: str,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int
    ) -> pd.DataFrame:

        interval_map = {
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1h": "1h",
            "4h": "4h",
            "1D": "1d",
        }

        oi_interval = interval_map.get(timeframe, "1h")

        rows = []
        current = start_ms
        batch_no = 0
        t0 = time.time()

        logging.info(
            f"[BYBIT OI START] {asset} | {symbol} | interval={oi_interval} | "
            f"from_utc={ms_to_utc_str(start_ms)} | to_utc={ms_to_utc_str(end_ms)}"
        )

        while current <= end_ms:
            batch_no += 1

            data = self.http.get_json(
                f"{self.BASE}/v5/market/open-interest",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "intervalTime": oi_interval,
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": self.OI_LIMIT,
                }
            )

            result = data.get("result", {})
            batch = result.get("list", [])

            if not batch:
                logging.info(f"[BYBIT OI EMPTY] {asset} | {symbol} | batch={batch_no}")
                break

            batch = sorted(batch, key=lambda x: int(x["timestamp"]))
            rows.extend(batch)

            newest = int(batch[-1]["timestamp"])
            next_start = newest + 1

            if batch_no == 1 or batch_no % PROGRESS_EVERY_N_BATCHES == 0:
                pct = 0.0

                if end_ms > start_ms:
                    pct = min(100.0, ((newest - start_ms) / (end_ms - start_ms)) * 100.0)

                logging.info(
                    f"[BYBIT OI PROGRESS] {asset} | {symbol} | interval={oi_interval} | "
                    f"batch={batch_no} | rows={len(rows):,} | latest_utc={ms_to_utc_str(newest)} | "
                    f"progress≈{pct:.2f}% | elapsed={format_seconds(time.time() - t0)}"
                )

            if next_start <= current:
                break

            current = next_start

            if len(batch) < self.OI_LIMIT:
                break

            time.sleep(HTTP_SLEEP_BETWEEN_BATCHES)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        df = add_time_columns_from_utc_ms(
            df=df,
            timestamp_col="timestamp",
        )

        df["OpenInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["Symbol"] = symbol
        df["OIInterval"] = oi_interval

        out = standardize_metric_df(
            df[[DATETIME_COL, TIMESTAMP_UTC_MS_COL, "Symbol", "OIInterval", "OpenInterest"]],
            ["OpenInterest"]
        )

        logging.info(
            f"[BYBIT OI FINISH] {asset} | {symbol} | "
            f"rows_clean={len(out):,} | elapsed={format_seconds(time.time() - t0)}"
        )

        return out


# =============================================================================
# DATA ENGINE
# =============================================================================

class ArchangelCryptoDataEngine:
    def __init__(self):
        self.paths = PathManager(BASE_DATA_DIR)
        self.http = HttpClient()

        self.binance = BinanceClient(self.http)
        self.bybit = BybitClient(self.http)

        self.catalog_records: List[Dict[str, Any]] = []
        self.catalog_lock = Lock()

        self.run_state = self._load_run_state()

    # -------------------------------------------------------------------------
    # THREAD SAFE CATALOG
    # -------------------------------------------------------------------------

    def add_catalog_record(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.update(get_time_policy_metadata())

        with self.catalog_lock:
            self.catalog_records.append(record)

    # -------------------------------------------------------------------------
    # STATE
    # -------------------------------------------------------------------------

    def _load_run_state(self) -> Dict[str, Any]:
        p = self.paths.run_state_path()

        if not p.exists():
            return {"run_count": 0, "last_run": None}

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            return {"run_count": 0, "last_run": None}
        except Exception:
            return {"run_count": 0, "last_run": None}

    def _save_run_state(self) -> None:
        previous_run_count = int(self.run_state.get("run_count", 0) or 0)
        run_count = previous_run_count + 1
        last_run = datetime.now().isoformat()
        time_policy = get_time_policy_metadata()
        records_snapshot = list(self.catalog_records)

        status_counts: Dict[str, int] = {}
        dataset_kind_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        asset_counts: Dict[str, int] = {}
        timeframe_counts: Dict[str, int] = {}
        for record in records_snapshot:
            for target, key in [
                (status_counts, record.get("status")),
                (dataset_kind_counts, record.get("dataset_kind")),
                (source_counts, record.get("source")),
                (asset_counts, record.get("asset")),
                (timeframe_counts, record.get("timeframe")),
            ]:
                label = str(key or "UNKNOWN")
                target[label] = target.get(label, 0) + 1

        self.run_state = {
            "schema_version": RUN_STATE_SCHEMA_VERSION,
            "system": {
                "name": SYSTEM_NAME,
                "version": SYSTEM_VERSION,
                "layer": "2_BUSCA_DADOS",
                "script": SCRIPT_NAME,
                "run_id": RUN_ID,
                "generated_at": last_run,
            },
            "paths": {
                "project_root": str(PROJECT_ROOT),
                "base_data_dir": BASE_DATA_DIR,
                "base_json_dir": str(BASE_JSON_DIR),
                "run_state_path": str(self.paths.run_state_path()),
                "catalog_path": str(self.paths.catalog_path()),
            },
            "summary": {
                "run_count": run_count,
                "previous_run_count": previous_run_count,
                "last_run": last_run,
                "catalog_records_in_current_run": len(records_snapshot),
                "status_counts": dict(sorted(status_counts.items())),
                "dataset_kind_counts": dict(sorted(dataset_kind_counts.items())),
                "source_counts": dict(sorted(source_counts.items())),
                "asset_counts": dict(sorted(asset_counts.items())),
                "timeframe_counts": dict(sorted(timeframe_counts.items())),
                "warnings": sum(1 for r in records_snapshot if r.get("status") != "OK"),
            },
            "policy": {
                "use_incremental_update": USE_INCREMENTAL_UPDATE,
                "download_start_utc": DOWNLOAD_START_UTC,
                "download_until_now": DOWNLOAD_UNTIL_NOW,
                "download_end_utc": DOWNLOAD_END_UTC,
                "incremental_overlap_bars": INCREMENTAL_OVERLAP_BARS,
                "max_total_bars_per_series": MAX_TOTAL_BARS_PER_SERIES,
                "bar_timestamp_policy": BAR_TIMESTAMP_POLICY,
            },
            "time_policy": time_policy,
            "run_count": run_count,
            "last_run": last_run,
        }

        atomic_save_json(self.run_state, self.paths.run_state_path())

    # -------------------------------------------------------------------------
    # UPDATE WINDOW
    # -------------------------------------------------------------------------

    def _get_start_end_for_update(self, existing_df: Optional[pd.DataFrame], timeframe: str) -> Tuple[int, int]:
        end_ms = utc_now_ms() if DOWNLOAD_UNTIL_NOW else parse_utc_to_ms(DOWNLOAD_END_UTC)

        if end_ms is None:
            end_ms = utc_now_ms()

        full_start_ms = parse_utc_to_ms(DOWNLOAD_START_UTC)

        if full_start_ms is None:
            raise ValueError(f"DOWNLOAD_START_UTC inválido: {DOWNLOAD_START_UTC}")

        if not USE_INCREMENTAL_UPDATE:
            return full_start_ms, end_ms

        if existing_df is None or existing_df.empty:
            return full_start_ms, end_ms

        existing_df = ensure_archangel_time_columns(existing_df)

        if TIMESTAMP_UTC_MS_COL in existing_df.columns:
            ts = pd.to_numeric(existing_df[TIMESTAMP_UTC_MS_COL], errors="coerce").dropna()

            if ts.empty:
                return full_start_ms, end_ms

            last_ms = int(ts.max())

        elif DATETIME_COL in existing_df.columns:
            dt = pd.to_datetime(existing_df[DATETIME_COL], errors="coerce").dropna()

            if dt.empty:
                return full_start_ms, end_ms

            last_local = pd.Timestamp(dt.max()).tz_localize(TIMEZONE_LOCAL).tz_convert("UTC")
            last_ms = int(last_local.timestamp() * 1000)

        else:
            return full_start_ms, end_ms

        overlap_ms = timeframe_to_ms(timeframe) * INCREMENTAL_OVERLAP_BARS
        start_ms = max(full_start_ms, last_ms - overlap_ms)

        return start_ms, end_ms

    # -------------------------------------------------------------------------
    # LOAD / MERGE
    # -------------------------------------------------------------------------

    def load_existing(self, path: Path, kind: str) -> Optional[pd.DataFrame]:
        df = safe_read_parquet(path)

        if df is None:
            return None

        if kind == "ohlcv":
            return standardize_ohlcv(df)

        return normalize_datetime_dubai(df, DATETIME_COL)

    def merge_timeseries(self, old_df: Optional[pd.DataFrame], new_df: pd.DataFrame, kind: str) -> pd.DataFrame:
        if new_df is None or new_df.empty:
            return old_df if old_df is not None else pd.DataFrame()

        if old_df is None or old_df.empty:
            merged = new_df.copy()
        else:
            merged = pd.concat([old_df, new_df], ignore_index=True)

        merged = ensure_archangel_time_columns(merged)

        dedup_col = TIMESTAMP_UTC_MS_COL if TIMESTAMP_UTC_MS_COL in merged.columns else DATETIME_COL
        sort_col = TIMESTAMP_UTC_MS_COL if TIMESTAMP_UTC_MS_COL in merged.columns else DATETIME_COL

        if dedup_col in merged.columns:
            merged = merged.drop_duplicates(subset=[dedup_col], keep="last")
            merged = merged.sort_values(sort_col).reset_index(drop=True)

        if kind == "ohlcv":
            merged = standardize_ohlcv(merged)
        else:
            merged = normalize_datetime_dubai(merged, DATETIME_COL)
            dedup_col = TIMESTAMP_UTC_MS_COL if TIMESTAMP_UTC_MS_COL in merged.columns else DATETIME_COL
            sort_col = TIMESTAMP_UTC_MS_COL if TIMESTAMP_UTC_MS_COL in merged.columns else DATETIME_COL
            merged = merged.drop_duplicates(subset=[dedup_col], keep="last")
            merged = merged.sort_values(sort_col).reset_index(drop=True)

        return merged

    # -------------------------------------------------------------------------
    # RESAMPLE
    # -------------------------------------------------------------------------

    def resample_ohlcv(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV com política close_time.

        Importante:
            Como DateTime representa o fechamento da barra 1min,
            o resample usa:
                label="right"
                closed="right"

            Assim, um candle 23min marcado como 10:23 representa dados
            conhecidos até 10:23, evitando decisão em 10:00 com dados futuros.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        freq = timeframe_to_pandas_freq(target_tf)

        x = standardize_ohlcv(df)
        x = x.set_index(DATETIME_COL).sort_index()

        out = x.resample(freq, label="right", closed="right").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })

        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        out = out.reset_index()

        out = ensure_archangel_time_columns(out)

        return standardize_ohlcv(out)

    # -------------------------------------------------------------------------
    # CHECKS
    # -------------------------------------------------------------------------

    def integrity_check(self, df: pd.DataFrame, timeframe: str, kind: str) -> Dict[str, Any]:
        result = {
            "rows": 0,
            "start": None,
            "end": None,
            "duplicates": 0,
            "gap_count_sampled": 0,
            "has_bad_ohlc": False,
            "sample_hash": None,
            "status": "EMPTY",
            "time_policy": get_time_policy_metadata(),
        }

        if df is None or df.empty or DATETIME_COL not in df.columns:
            return result

        x = ensure_archangel_time_columns(df)
        x[DATETIME_COL] = pd.to_datetime(x[DATETIME_COL], errors="coerce")
        x = x.dropna(subset=[DATETIME_COL])

        sort_col = TIMESTAMP_UTC_MS_COL if TIMESTAMP_UTC_MS_COL in x.columns else DATETIME_COL
        x = x.sort_values(sort_col)

        result["rows"] = int(len(x))
        result["start"], result["end"] = extract_period(x)

        dedup_col = TIMESTAMP_UTC_MS_COL if TIMESTAMP_UTC_MS_COL in x.columns else DATETIME_COL
        result["duplicates"] = int(x.duplicated(subset=[dedup_col]).sum())
        result["sample_hash"] = df_hash_sample(x)

        try:
            freq_str = timeframe_to_pandas_freq(timeframe)

            if freq_str != "MS":
                freq_ms = timeframe_to_ms(timeframe)

                if TIMESTAMP_UTC_MS_COL in x.columns:
                    ts = pd.to_numeric(x[TIMESTAMP_UTC_MS_COL], errors="coerce").dropna().sort_values()

                    if len(ts) > SAMPLE_CHECK_ROWS:
                        start_idx = random.randint(0, max(0, len(ts) - SAMPLE_CHECK_ROWS))
                        ts_sample = ts.iloc[start_idx:start_idx + SAMPLE_CHECK_ROWS]
                    else:
                        ts_sample = ts

                    diffs = ts_sample.diff().dropna()
                    result["gap_count_sampled"] = int((diffs > freq_ms).sum())

                else:
                    freq = pd.Timedelta(freq_str)
                    dt = x[DATETIME_COL].dropna().sort_values()

                    if len(dt) > SAMPLE_CHECK_ROWS:
                        start_idx = random.randint(0, max(0, len(dt) - SAMPLE_CHECK_ROWS))
                        dt_sample = dt.iloc[start_idx:start_idx + SAMPLE_CHECK_ROWS]
                    else:
                        dt_sample = dt

                    diffs = dt_sample.diff().dropna()
                    result["gap_count_sampled"] = int((diffs > freq).sum())
            else:
                result["gap_count_sampled"] = -1

        except Exception:
            result["gap_count_sampled"] = -1

        if kind == "ohlcv":
            try:
                bad = (
                    (x["Open"] <= 0) |
                    (x["High"] <= 0) |
                    (x["Low"] <= 0) |
                    (x["Close"] <= 0) |
                    (x["High"] < x[["Open", "Low", "Close"]].max(axis=1)) |
                    (x["Low"] > x[["Open", "High", "Close"]].min(axis=1)) |
                    (x["Volume"].fillna(0) < 0)
                )

                result["has_bad_ohlc"] = bool(bad.any())

            except Exception:
                result["has_bad_ohlc"] = True

        if result["duplicates"] == 0 and not result["has_bad_ohlc"] and result["gap_count_sampled"] in [0, -1]:
            result["status"] = "OK"
        else:
            result["status"] = "CHECK_WARNING"

        return result

    # -------------------------------------------------------------------------
    # PROCESS OHLCV
    # -------------------------------------------------------------------------

    def process_ohlcv_source(self, source: str, asset: str, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        t0 = time.time()

        path = self.paths.ohlcv_path(source, asset, symbol, timeframe, custom=False)
        existing = self.load_existing(path, "ohlcv")
        start_ms, end_ms = self._get_start_end_for_update(existing, timeframe)

        old_rows = 0 if existing is None else len(existing)
        old_start, old_end = extract_period(existing)

        logging.info(
            f"[JOB START] OHLCV | {source} | {asset} | {symbol} | {timeframe} | "
            f"old_rows={old_rows:,} | old_start={old_start} | old_end={old_end} | "
            f"download_from_utc={ms_to_utc_str(start_ms)} | "
            f"bar_timestamp_policy={BAR_TIMESTAMP_POLICY}"
        )

        if source == "binance_spot":
            df_new = self.binance.fetch_ohlcv(asset, symbol, timeframe, start_ms, end_ms)
        elif source == "bybit_linear":
            df_new = self.bybit.fetch_ohlcv(asset, symbol, timeframe, start_ms, end_ms)
        else:
            raise ValueError(f"Fonte OHLCV desconhecida: {source}")

        merged = self.merge_timeseries(existing, df_new, "ohlcv")

        if SAVE_PARQUET:
            atomic_save_parquet(merged, path)

        if SAVE_CSV_DEBUG:
            csv_path = str(path).replace(".parquet", ".csv")
            merged.to_csv(csv_path, index=False)

        check = self.integrity_check(merged, timeframe, "ohlcv")

        self.add_catalog_record({
            "dataset_kind": "ohlcv",
            "source": source,
            "asset": asset,
            "symbol": symbol,
            "timeframe": timeframe,
            "custom_timeframe": False,
            "path": str(path),
            "old_rows": old_rows,
            "new_rows": 0 if df_new is None else len(df_new),
            "rows": check["rows"],
            "start": check["start"],
            "end": check["end"],
            "status": check["status"],
            "duplicates": check["duplicates"],
            "gap_count_sampled": check["gap_count_sampled"],
            "has_bad_ohlc": check["has_bad_ohlc"],
            "sample_hash": check["sample_hash"],
            "timezone": TIMEZONE_LOCAL,
        })

        logging.info(
            f"[JOB FINISH] OHLCV | {source} | {asset} | {symbol} | {timeframe} | "
            f"old_rows={old_rows:,} | new_rows={0 if df_new is None else len(df_new):,} | "
            f"final_rows={len(merged):,} | status={check['status']} | "
            f"saved={path} | elapsed={format_seconds(time.time() - t0)}"
        )

        return merged

    def process_custom_ohlcv(self, source: str, asset: str, symbol: str, target_tf: str) -> None:
        """
        Modo antigo: lê 1min para cada custom timeframe.
        Mantido como fallback.
        """
        t0 = time.time()

        base_tf = "1min"
        base_path = self.paths.ohlcv_path(source, asset, symbol, base_tf, custom=False)
        base_df = self.load_existing(base_path, "ohlcv")

        logging.info(
            f"[CUSTOM START] {source} | {asset} | {symbol} | {base_tf} -> {target_tf} | "
            f"bar_timestamp_policy={BAR_TIMESTAMP_POLICY}"
        )

        if base_df is None or base_df.empty:
            logging.warning(f"[CUSTOM SKIP] Sem base 1min | {source} | {asset} | {symbol} | {target_tf}")
            return

        custom_df = self.resample_ohlcv(base_df, target_tf)
        out_path = self.paths.ohlcv_path(source, asset, symbol, target_tf, custom=True)

        if SAVE_PARQUET:
            atomic_save_parquet(custom_df, out_path)

        check = self.integrity_check(custom_df, target_tf, "ohlcv")

        self.add_catalog_record({
            "dataset_kind": "ohlcv",
            "source": source,
            "asset": asset,
            "symbol": symbol,
            "timeframe": target_tf,
            "custom_timeframe": True,
            "created_from": base_tf,
            "path": str(out_path),
            "rows": check["rows"],
            "start": check["start"],
            "end": check["end"],
            "status": check["status"],
            "duplicates": check["duplicates"],
            "gap_count_sampled": check["gap_count_sampled"],
            "has_bad_ohlc": check["has_bad_ohlc"],
            "sample_hash": check["sample_hash"],
            "timezone": TIMEZONE_LOCAL,
        })

        logging.info(
            f"[CUSTOM FINISH] {source} | {asset} | {symbol} | {base_tf} -> {target_tf} | "
            f"rows={len(custom_df):,} | status={check['status']} | "
            f"saved={out_path} | elapsed={format_seconds(time.time() - t0)}"
        )

    def process_all_custom_ohlcv_fast(
        self,
        source: str,
        asset: str,
        symbol: str,
        target_timeframes: Optional[List[str]] = None,
        base_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Modo otimizado:
            - Lê a base 1min apenas uma vez.
            - Gera todos os custom timeframes.
            - Mantém validação e escrita atômica.
            - Usa timestamp de fechamento da barra.
        """
        t0 = time.time()

        base_tf = "1min"
        base_path = self.paths.ohlcv_path(source, asset, symbol, base_tf, custom=False)
        if base_df is None:
            base_df = self.load_existing(base_path, "ohlcv")

        target_timeframes = list(target_timeframes or CUSTOM_TIMEFRAMES)

        logging.info(
            f"[CUSTOM FAST START] {source} | {asset} | {symbol} | "
            f"base={base_tf} | target_count={len(target_timeframes)} | "
            f"bar_timestamp_policy={BAR_TIMESTAMP_POLICY}"
        )

        if base_df is None or base_df.empty:
            logging.warning(
                f"[CUSTOM FAST SKIP] Sem base 1min | {source} | {asset} | {symbol}"
            )
            return

        for target_tf in target_timeframes:
            tf_t0 = time.time()

            try:
                custom_df = self.resample_ohlcv(base_df, target_tf)
                is_native_target = target_tf in NATIVE_TIMEFRAMES
                out_path = self.paths.ohlcv_path(
                    source,
                    asset,
                    symbol,
                    target_tf,
                    custom=not is_native_target,
                )

                if SAVE_PARQUET:
                    atomic_save_parquet(custom_df, out_path)

                check = self.integrity_check(custom_df, target_tf, "ohlcv")

                self.add_catalog_record({
                    "dataset_kind": "ohlcv",
                    "source": source,
                    "asset": asset,
                    "symbol": symbol,
                    "timeframe": target_tf,
                    "custom_timeframe": not is_native_target,
                    "created_from": base_tf,
                    "generated_locally_from_1min": True,
                    "path": str(out_path),
                    "rows": check["rows"],
                    "start": check["start"],
                    "end": check["end"],
                    "status": check["status"],
                    "duplicates": check["duplicates"],
                    "gap_count_sampled": check["gap_count_sampled"],
                    "has_bad_ohlc": check["has_bad_ohlc"],
                    "sample_hash": check["sample_hash"],
                    "timezone": TIMEZONE_LOCAL,
                })

                logging.info(
                    f"[CUSTOM FAST FINISH] {source} | {asset} | {symbol} | "
                    f"{base_tf} -> {target_tf} | rows={len(custom_df):,} | "
                    f"status={check['status']} | saved={out_path} | "
                    f"elapsed={format_seconds(time.time() - tf_t0)}"
                )

            except Exception as e:
                logging.exception(
                    f"[CUSTOM FAST ERROR] {source} | {asset} | {symbol} | {target_tf}: {e}"
                )

        logging.info(
            f"[CUSTOM FAST DONE] {source} | {asset} | {symbol} | "
            f"elapsed={format_seconds(time.time() - t0)}"
        )

    # -------------------------------------------------------------------------
    # PROCESS FUNDING / OI
    # -------------------------------------------------------------------------

    def process_bybit_funding(self, asset: str, symbol: str) -> None:
        t0 = time.time()

        source = "bybit_linear"
        path = self.paths.metric_path("funding_rate", source, asset, symbol)
        existing = self.load_existing(path, "funding_rate")
        start_ms, end_ms = self._get_start_end_for_update(existing, "1h")

        old_rows = 0 if existing is None else len(existing)

        logging.info(
            f"[JOB START] FUNDING | Bybit | {asset} | {symbol} | "
            f"old_rows={old_rows:,} | download_from_utc={ms_to_utc_str(start_ms)}"
        )

        df_new = self.bybit.fetch_funding_rate(asset, symbol, start_ms, end_ms)
        merged = self.merge_timeseries(existing, df_new, "funding_rate")

        if SAVE_PARQUET:
            atomic_save_parquet(merged, path)

        check = self.integrity_check(merged, "1h", "funding_rate")

        self.add_catalog_record({
            "dataset_kind": "funding_rate",
            "source": source,
            "asset": asset,
            "symbol": symbol,
            "timeframe": "event_based_8h_usually",
            "path": str(path),
            "old_rows": old_rows,
            "new_rows": 0 if df_new is None else len(df_new),
            "rows": check["rows"],
            "start": check["start"],
            "end": check["end"],
            "status": check["status"],
            "timezone": TIMEZONE_LOCAL,
            "notes": "Bybit funding history. Frequency usually 8h depending on instrument.",
        })

        logging.info(
            f"[JOB FINISH] FUNDING | Bybit | {asset} | {symbol} | "
            f"final_rows={len(merged):,} | status={check['status']} | "
            f"saved={path} | elapsed={format_seconds(time.time() - t0)}"
        )

    def process_bybit_open_interest(self, asset: str, symbol: str, timeframe: str = "1h") -> None:
        t0 = time.time()

        source = "bybit_linear"
        metric = f"open_interest_{timeframe}"
        path = self.paths.metric_path(metric, source, asset, symbol)
        existing = self.load_existing(path, "open_interest")
        start_ms, end_ms = self._get_start_end_for_update(existing, timeframe)

        old_rows = 0 if existing is None else len(existing)

        logging.info(
            f"[JOB START] OPEN_INTEREST | Bybit | {asset} | {symbol} | {timeframe} | "
            f"old_rows={old_rows:,} | download_from_utc={ms_to_utc_str(start_ms)}"
        )

        df_new = self.bybit.fetch_open_interest(asset, symbol, timeframe, start_ms, end_ms)
        merged = self.merge_timeseries(existing, df_new, "open_interest")

        if SAVE_PARQUET:
            atomic_save_parquet(merged, path)

        check = self.integrity_check(merged, timeframe, "open_interest")

        self.add_catalog_record({
            "dataset_kind": "open_interest",
            "source": source,
            "asset": asset,
            "symbol": symbol,
            "timeframe": timeframe,
            "path": str(path),
            "old_rows": old_rows,
            "new_rows": 0 if df_new is None else len(df_new),
            "rows": check["rows"],
            "start": check["start"],
            "end": check["end"],
            "status": check["status"],
            "timezone": TIMEZONE_LOCAL,
        })

        logging.info(
            f"[JOB FINISH] OPEN_INTEREST | Bybit | {asset} | {symbol} | {timeframe} | "
            f"final_rows={len(merged):,} | status={check['status']} | "
            f"saved={path} | elapsed={format_seconds(time.time() - t0)}"
        )

    # -------------------------------------------------------------------------
    # ASSET/SOURCE TASKS
    # -------------------------------------------------------------------------

    def process_binance_asset(self, asset_no: int, total_assets: int, asset: str, maps: Dict[str, str]) -> None:
        if not ENABLE_BINANCE_SPOT or not maps.get("binance_spot"):
            return

        source = "binance_spot"
        symbol = maps["binance_spot"]

        logging.info(f"[SOURCE START] {asset_no}/{total_assets} | {asset} | {source} | {symbol}")

        native_to_download = [
            tf for tf in NATIVE_TIMEFRAMES
            if tf in set(BINANCE_NATIVE_TIMEFRAMES_TO_DOWNLOAD)
        ]
        if "1min" not in native_to_download:
            logging.warning(
                f"[BINANCE FAST WARNING] {asset} | 1min não está em "
                f"BINANCE_NATIVE_TIMEFRAMES_TO_DOWNLOAD; resample dependerá de base local existente."
            )

        base_1min_df: Optional[pd.DataFrame] = None

        for tf in native_to_download:
            if tf not in BINANCE_INTERVAL_MAP:
                logging.info(f"[SKIP] Binance não suporta {tf}")
                continue

            try:
                merged = self.process_ohlcv_source(source, asset, symbol, tf)
                if tf == "1min":
                    base_1min_df = merged
            except Exception as e:
                logging.exception(f"[ERROR] Binance OHLCV | {asset} | {tf}: {e}")

        if ENABLE_FAST_CUSTOM_TIMEFRAMES:
            resample_targets = [
                tf for tf in RESAMPLE_FROM_1MIN_TIMEFRAMES
                if tf not in set(native_to_download)
            ]
            self.process_all_custom_ohlcv_fast(
                source,
                asset,
                symbol,
                target_timeframes=resample_targets,
                base_df=base_1min_df,
            )
        else:
            for tf in CUSTOM_TIMEFRAMES:
                try:
                    self.process_custom_ohlcv(source, asset, symbol, tf)
                except Exception as e:
                    logging.exception(f"[ERROR] Binance custom OHLCV | {asset} | {tf}: {e}")

        logging.info(f"[SOURCE FINISH] {asset_no}/{total_assets} | {asset} | {source} | {symbol}")

    def process_bybit_asset(self, asset_no: int, total_assets: int, asset: str, maps: Dict[str, str]) -> None:
        if not ENABLE_BYBIT_LINEAR or not maps.get("bybit_linear"):
            return

        source = "bybit_linear"
        symbol = maps["bybit_linear"]

        logging.info(f"[SOURCE START] {asset_no}/{total_assets} | {asset} | {source} | {symbol}")

        for tf in NATIVE_TIMEFRAMES:
            if tf not in BYBIT_INTERVAL_MAP:
                logging.info(f"[SKIP] Bybit não suporta {tf}")
                continue

            try:
                self.process_ohlcv_source(source, asset, symbol, tf)
            except Exception as e:
                logging.exception(f"[ERROR] Bybit OHLCV | {asset} | {tf}: {e}")

        if ENABLE_FAST_CUSTOM_TIMEFRAMES:
            self.process_all_custom_ohlcv_fast(source, asset, symbol)
        else:
            for tf in CUSTOM_TIMEFRAMES:
                try:
                    self.process_custom_ohlcv(source, asset, symbol, tf)
                except Exception as e:
                    logging.exception(f"[ERROR] Bybit custom OHLCV | {asset} | {tf}: {e}")

        if ENABLE_BYBIT_FUNDING:
            try:
                self.process_bybit_funding(asset, symbol)
            except Exception as e:
                logging.exception(f"[ERROR] Bybit funding | {asset}: {e}")

        if ENABLE_BYBIT_OPEN_INTEREST:
            try:
                self.process_bybit_open_interest(
                    asset,
                    symbol,
                    timeframe=BYBIT_OPEN_INTEREST_TIMEFRAME
                )
            except Exception as e:
                logging.exception(f"[ERROR] Bybit OI | {asset}: {e}")

        logging.info(f"[SOURCE FINISH] {asset_no}/{total_assets} | {asset} | {source} | {symbol}")

    # -------------------------------------------------------------------------
    # CATALOG / DIAGNOSTICS
    # -------------------------------------------------------------------------

    def save_catalog(self) -> None:
        with self.catalog_lock:
            records_snapshot = list(self.catalog_records)

        catalog = {
            "schema_version": "ARCHANGEL_CRYPTO_DATA_ENGINE_BINANCE_ONLY_OPTIMIZED_2.2_TIME_AWARE",
            "generated_at_local": datetime.now().isoformat(),
            "timezone": TIMEZONE_LOCAL,
            "time_policy": get_time_policy_metadata(),
            "base_data_dir": str(Path(BASE_DATA_DIR).resolve()),
            "sources_enabled": {
                "binance_spot": ENABLE_BINANCE_SPOT,
                "bybit_linear": ENABLE_BYBIT_LINEAR,
                "bybit_funding": ENABLE_BYBIT_FUNDING,
                "bybit_open_interest": ENABLE_BYBIT_OPEN_INTEREST,
                "bybit_disabled_by_policy": True,
                "kraken_removed": True,
            },
            "performance_config": {
                "enable_parallel_downloads": ENABLE_PARALLEL_DOWNLOADS,
                "max_workers_download": MAX_WORKERS_DOWNLOAD,
                "binance_native_timeframes_to_download": BINANCE_NATIVE_TIMEFRAMES_TO_DOWNLOAD,
                "resample_from_1min_timeframes": RESAMPLE_FROM_1MIN_TIMEFRAMES,
                "enable_fast_custom_timeframes": ENABLE_FAST_CUSTOM_TIMEFRAMES,
                "http_sleep_between_batches": HTTP_SLEEP_BETWEEN_BATCHES,
                "http_pool_connections": HTTP_POOL_CONNECTIONS,
                "http_pool_maxsize": HTTP_POOL_MAXSIZE,
                "parquet_engine": PARQUET_ENGINE,
                "parquet_compression": PARQUET_COMPRESSION,
                "generate_ai_inventory_in_engine": GENERATE_AI_INVENTORY_IN_ENGINE,
            },
            "integrity_policy": {
                "atomic_parquet_save": True,
                "incremental_overlap_bars": INCREMENTAL_OVERLAP_BARS,
                "deduplicate_by_datetime": False,
                "deduplicate_by_timestamp_utc_ms": True,
                "ohlcv_cleaning": True,
                "sample_integrity_check": True,
                "threadsafe_catalog_lock": True,
                "no_parallel_same_file_writes": True,
                "bar_timestamp_policy": BAR_TIMESTAMP_POLICY,
            },
            "purpose": [
                "system_trading",
                "backtesting",
                "machine_learning",
                "deep_learning",
            ],
            "notes": {
                "custom_timeframes": (
                    "Criados localmente via resample OHLCV a partir de 1min. "
                    "Resample usa label='right' e closed='right' para política close_time."
                ),
                "datetime_policy": (
                    "DateTime salvo como horário local de Dubai, sem timezone attached. "
                    "timestamp_utc_ms salvo como referência universal UTC em milissegundos."
                ),
                "lookahead_control": (
                    "OHLCV é marcado pelo fechamento da barra. "
                    "Features e labels devem usar somente dados disponíveis até timestamp_utc_ms."
                ),
                "ai_inventory": (
                    "Inventário completo separado. Use MAPA_ATIVOS.py para gerar "
                    f"{PROJECT_ROOT / '0_REGRAS_MANDATO' / 'BASE_JSON' / 'MAPA_ATIVOS.json'}"
                ),
            },
            "records": records_snapshot,
            "summary": {
                "total_records": len(records_snapshot),
                "total_ohlcv": sum(1 for r in records_snapshot if r.get("dataset_kind") == "ohlcv"),
                "total_funding": sum(1 for r in records_snapshot if r.get("dataset_kind") == "funding_rate"),
                "total_open_interest": sum(1 for r in records_snapshot if r.get("dataset_kind") == "open_interest"),
                "warnings": sum(1 for r in records_snapshot if r.get("status") != "OK"),
            },
        }

        path = self.paths.catalog_path()
        atomic_save_json(catalog, path)

        logging.info(f"[CATALOG SAVED] {path}")

    def save_diagnostics(self) -> None:
        with self.catalog_lock:
            records_snapshot = list(self.catalog_records)

        if not records_snapshot:
            return

        df = pd.DataFrame(records_snapshot)
        path = self.paths.diagnostics_path()
        atomic_save_parquet(df, path)

        logging.info(f"[DIAGNOSTICS SAVED] {path}")

    def save_ai_data_inventory(self) -> None:
        """
        Mantido apenas como aviso.
        O inventário completo agora deve ser gerado pelo MAPA_ATIVOS.py.
        """
        logging.info(
            "[AI INVENTORY SKIP] Inventário completo desativado neste engine. "
            "Execute MAPA_ATIVOS.py para gerar MAPA_ATIVOS.json."
        )

    # -------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------

    def run(self) -> None:
        logging.info("=" * 100)
        logging.info("ARCHANGEL CRYPTO DATA ENGINE - BINANCE ONLY - OPTIMIZED START")
        logging.info("=" * 100)
        logging.info(f"Base dir: {BASE_DATA_DIR}")
        logging.info(f"Timezone local: {TIMEZONE_LOCAL}")
        logging.info(f"Time policy: {get_time_policy_metadata()}")
        logging.info(f"Download start UTC: {DOWNLOAD_START_UTC}")
        logging.info(f"Native timeframes: {NATIVE_TIMEFRAMES}")
        logging.info(f"Custom timeframes: {CUSTOM_TIMEFRAMES}")
        logging.info(f"Use incremental update: {USE_INCREMENTAL_UPDATE}")
        logging.info(f"Incremental overlap bars: {INCREMENTAL_OVERLAP_BARS}")
        logging.info(f"Max total bars per series: {MAX_TOTAL_BARS_PER_SERIES}")
        logging.info(f"Parallel downloads enabled: {ENABLE_PARALLEL_DOWNLOADS}")
        logging.info(f"Max workers download: {MAX_WORKERS_DOWNLOAD}")
        logging.info(f"Binance native timeframes to download: {BINANCE_NATIVE_TIMEFRAMES_TO_DOWNLOAD}")
        logging.info(f"Resample from 1min timeframes: {RESAMPLE_FROM_1MIN_TIMEFRAMES}")
        logging.info(f"Fast custom timeframes: {ENABLE_FAST_CUSTOM_TIMEFRAMES}")
        logging.info(
            f"HTTP sleep={HTTP_SLEEP_BETWEEN_BATCHES}s | "
            f"pool_connections={HTTP_POOL_CONNECTIONS} | pool_maxsize={HTTP_POOL_MAXSIZE}"
        )
        logging.info(f"Generate AI inventory in engine: {GENERATE_AI_INVENTORY_IN_ENGINE}")
        logging.info(f"Parquet: engine={PARQUET_ENGINE}, compression={PARQUET_COMPRESSION}")

        t0 = time.time()
        total_assets = len(SYMBOLS)

        if ENABLE_PARALLEL_DOWNLOADS:
            logging.info("[RUN MODE] Paralelo moderado por ativo/fonte")

            jobs = []

            with ThreadPoolExecutor(max_workers=MAX_WORKERS_DOWNLOAD) as executor:
                for asset_no, (asset, maps) in enumerate(SYMBOLS.items(), start=1):
                    if ENABLE_BINANCE_SPOT and maps.get("binance_spot"):
                        jobs.append(
                            executor.submit(
                                self.process_binance_asset,
                                asset_no,
                                total_assets,
                                asset,
                                maps,
                            )
                        )

                    if ENABLE_BYBIT_LINEAR and maps.get("bybit_linear"):
                        jobs.append(
                            executor.submit(
                                self.process_bybit_asset,
                                asset_no,
                                total_assets,
                                asset,
                                maps,
                            )
                        )

                for future in as_completed(jobs):
                    try:
                        future.result()
                    except Exception as e:
                        logging.exception(f"[PARALLEL JOB ERROR] {e}")

        else:
            logging.info("[RUN MODE] Sequencial")

            for asset_no, (asset, maps) in enumerate(SYMBOLS.items(), start=1):
                asset_t0 = time.time()

                logging.info("-" * 100)
                logging.info(f"[ASSET START] {asset_no}/{total_assets} | {asset}")
                logging.info("-" * 100)

                self.process_binance_asset(asset_no, total_assets, asset, maps)
                if ENABLE_BYBIT_LINEAR:
                    self.process_bybit_asset(asset_no, total_assets, asset, maps)

                logging.info(
                    f"[ASSET FINISH] {asset_no}/{total_assets} | {asset} | "
                    f"elapsed_asset={format_seconds(time.time() - asset_t0)} | "
                    f"elapsed_total={format_seconds(time.time() - t0)}"
                )

        self.save_catalog()
        self.save_diagnostics()

        if GENERATE_AI_INVENTORY_IN_ENGINE:
            self.save_ai_data_inventory()
        else:
            logging.info(
                "[AI INVENTORY SKIP] Use MAPA_ATIVOS.py para gerar "
                f"{PROJECT_ROOT / '0_REGRAS_MANDATO' / 'BASE_JSON' / 'MAPA_ATIVOS.json'}"
            )

        self._save_run_state()

        elapsed = time.time() - t0

        with self.catalog_lock:
            total_catalog = len(self.catalog_records)

        logging.info("=" * 100)
        logging.info("ARCHANGEL CRYPTO DATA ENGINE - OPTIMIZED END")
        logging.info(f"Elapsed total: {format_seconds(elapsed)}")
        logging.info(f"Catalog records: {total_catalog}")
        logging.info("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    setup_logging()

    try:
        engine = ArchangelCryptoDataEngine()
        engine.run()
        logging.info("Processo concluído com sucesso.")

    except KeyboardInterrupt:
        logging.warning("Processo interrompido pelo usuário.")

    except Exception as e:
        logging.exception(f"Erro fatal: {e}")


if __name__ == "__main__":
    main()
