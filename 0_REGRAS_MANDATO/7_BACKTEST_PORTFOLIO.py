# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# 1. CONFIGURACAO
# =============================================================================

SCRIPT_NAME = "7_BACKTEST_PORTFOLIO.py"
SCHEMA_VERSION = "ARCHANGEL_BACKTEST_PORTFOLIO_1.0"

RULES_DIR = Path(__file__).resolve().parent
ROOT_DIR = RULES_DIR.parent
BASE_JSON_DIR = RULES_DIR / "BASE_JSON"

WALK_FORWARD_JSON_PATH = BASE_JSON_DIR / "6_JSON_WALK_FORWARD.json"
DATASETS_JSON_PATH = BASE_JSON_DIR / "5_JSON_DATASETS_ML.json"
COST_MODEL_PATH = BASE_JSON_DIR / "COST_MODEL.json"
AI_CONTEXT_INDEX_PATH = BASE_JSON_DIR / "ARCHANGEL_AI_CONTEXT_INDEX.json"

BACKTEST_DIR = ROOT_DIR / "7_BACKTEST_PORTFOLIO"
LOGS_DIR = BACKTEST_DIR / "_logs"
TRADES_DIR = BACKTEST_DIR / "TRADES_PARQUET"
EQUITY_DIR = BACKTEST_DIR / "EQUITY_PARQUET"
REGISTRY_DIR = BACKTEST_DIR / "REGISTRY"
VALIDATION_DIR = BACKTEST_DIR / "VALIDATION"
STRESS_DIR = BACKTEST_DIR / "STRESS"

BACKTEST_JSON_PATH = BASE_JSON_DIR / "7_JSON_BACKTEST_PORTFOLIO.json"
RUN_REPORT_LATEST_PATH = BASE_JSON_DIR / "7_BACKTEST_PORTFOLIO_RUN_REPORT_LATEST.json"
PARAM_SEARCH_JSON_PATH = BASE_JSON_DIR / "7_BACKTEST_PARAM_SEARCH_LATEST.json"
PARAM_SEARCH_RUN_REPORT_LATEST_PATH = BASE_JSON_DIR / "7_BACKTEST_PARAM_SEARCH_RUN_REPORT_LATEST.json"
EXPERIMENT_REGISTRY_JSON_PATH = BASE_JSON_DIR / "7_EXPERIMENT_REGISTRY_LATEST.json"
EXPERIMENT_REGISTRY_JSONL_PATH = REGISTRY_DIR / "7_EXPERIMENT_REGISTRY.jsonl"
VALIDATION_JSON_PATH = BASE_JSON_DIR / "7_BACKTEST_VALIDATION_LATEST.json"
STRESS_JSON_PATH = BASE_JSON_DIR / "7_BACKTEST_STRESS_LATEST.json"

DEFAULT_TARGET_ANNUAL_RETURN = 0.20
DEFAULT_REFERENCE_DRAWDOWN_LIMIT = 0.08
DEFAULT_INITIAL_CAPITAL = 100000.0

COST_COLUMNS = [
    "cost_fee_bps_round_trip",
    "cost_slippage_bps_round_trip",
    "cost_funding_bps_per_day",
    "cost_min_net_edge_bps",
    "cost_spread_bps_one_way",
    "cost_market_impact_bps_one_way",
    "cost_liquidation_buffer_pct",
]


# =============================================================================
# 2. UTILITARIOS
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def run_id_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    for path in [BASE_JSON_DIR, BACKTEST_DIR, LOGS_DIR, TRADES_DIR, EQUITY_DIR, REGISTRY_DIR, VALIDATION_DIR, STRESS_DIR]:
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
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        path.write_text(tmp_path.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            tmp_path.unlink()
        except OSError:
            pass


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "sim", "s"}:
        return True
    if text in {"0", "false", "no", "nao", "n"}:
        return False
    return default


def parse_float_grid(text: str | None, default: list[float]) -> list[float]:
    if not text:
        return default
    values: list[float] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = safe_float(raw, default=float("nan"))
        if math.isfinite(value):
            values.append(value)
    return values or default


def parse_bool_grid(text: str | None, default: list[bool]) -> list[bool]:
    if not text:
        return default
    values: list[bool] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(safe_bool(raw, default=False))
    return values or default


def process_memory_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)
    except Exception:
        return None


def parquet_columns(path: str | Path) -> set[str]:
    return set(pq.ParquetFile(str(path)).schema_arrow.names)


def parse_horizon_bars(target_col: str | None) -> int:
    if not target_col:
        return 20
    match = re.search(r"_h(\d+)(?:_|$)", target_col)
    if match:
        return int(match.group(1))
    match = re.search(r"h(\d+)", target_col)
    if match:
        return int(match.group(1))
    return 20


def timeframe_to_seconds(timeframe: str | None) -> int:
    if not timeframe:
        return 60
    text = timeframe.strip().lower()
    match = re.fullmatch(r"(\d+)\s*([a-z]+)", text)
    if not match:
        return 60
    value = int(match.group(1))
    unit = match.group(2)
    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return value * 60
    if unit in {"h", "hour", "hours"}:
        return value * 60 * 60
    if unit in {"d", "day", "days"}:
        return value * 24 * 60 * 60
    return value * 60


def clean_symbol(symbol: str | None, asset: str | None) -> str:
    return str(symbol or asset or "UNKNOWN").upper()


def dataset_path_from_experiment(experiment: dict[str, Any], datasets_by_series: dict[str, dict[str, Any]]) -> str | None:
    direct = experiment.get("dataset_path") or experiment.get("output_path")
    if direct:
        return str(direct)

    key = "|".join(
        str(experiment.get(part) or "").lower()
        for part in ["source", "asset", "symbol", "timeframe"]
    )
    dataset = datasets_by_series.get(key)
    if dataset:
        return dataset.get("output_path") or dataset.get("dataset_path")
    return None


def build_datasets_index(datasets_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in datasets_json.get("datasets", []):
        key = "|".join(
            str(item.get(part) or "").lower()
            for part in ["source", "asset", "symbol", "timeframe"]
        )
        index[key] = item
    return index


def cost_defaults(cost_model: dict[str, Any], source: str | None, symbol: str | None) -> dict[str, float]:
    default = cost_model.get("default", {}) if isinstance(cost_model.get("default"), dict) else {}
    source_cfg = cost_model.get(source or "", {}) if isinstance(cost_model.get(source or ""), dict) else {}
    symbol_cfg = (
        cost_model.get("symbols", {}).get(symbol or "", {})
        if isinstance(cost_model.get("symbols"), dict)
        else {}
    )
    merged = {**default, **source_cfg}
    if "slippage_bps_round_trip_override" in symbol_cfg:
        merged["slippage_bps_round_trip"] = symbol_cfg["slippage_bps_round_trip_override"]
    for key in ["spread_bps_one_way", "market_impact_bps_one_way"]:
        if key in symbol_cfg:
            merged[key] = symbol_cfg[key]
    return {key: safe_float(value) for key, value in merged.items()}


def row_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row.index:
        return default
    return safe_float(row[column], default)


def price_value(row: pd.Series, column: str, fallback: float = float("nan")) -> float:
    value = safe_float(row.get(column), default=fallback)
    if value <= 0 or not math.isfinite(value):
        return fallback
    return value


def exchange_funding_bps_per_day(source: str, row: pd.Series, defaults: dict[str, float], args: dict[str, Any]) -> float:
    dataset_value = row_value(row, "cost_funding_bps_per_day", default=float("nan"))
    if math.isfinite(dataset_value) and dataset_value != 0:
        return dataset_value
    if args.get("funding_bps_per_day") is not None:
        return safe_float(args.get("funding_bps_per_day"), default=0.0)
    source_text = str(source).lower()
    if "spot" in source_text:
        return 0.0
    exchange_defaults = {
        "binance": 1.5,
        "bybit": 2.0,
        "kraken": 1.8,
    }
    for key, value in exchange_defaults.items():
        if key in source_text:
            return value
    return defaults.get("funding_bps_per_day", 1.5)


def estimate_quote_volume(row: pd.Series) -> float:
    close = price_value(row, "Close")
    volume = safe_float(row.get("Volume"), default=0.0)
    if not math.isfinite(close) or close <= 0 or volume <= 0:
        return 0.0
    return close * volume


def estimate_dynamic_slippage_bps(row: pd.Series, defaults: dict[str, float], notional: float, args: dict[str, Any]) -> tuple[float, float, float]:
    base_slippage = row_value(
        row,
        "cost_slippage_bps_round_trip",
        defaults.get("slippage_bps_round_trip", 10.0),
    )
    quote_volume = estimate_quote_volume(row)
    participation = 0.0 if quote_volume <= 0 else max(0.0, notional / quote_volume)
    participation_cap = max(0.000001, safe_float(args.get("max_volume_participation"), 0.02))
    capped_participation = min(participation, participation_cap)
    impact_multiplier = safe_float(args.get("slippage_impact_multiplier"), 250.0)
    liquidity_penalty = capped_participation * impact_multiplier
    return round(base_slippage + liquidity_penalty, 8), round(participation, 10), round(quote_volume, 4)


def estimate_cost_bps(
    row: pd.Series,
    defaults: dict[str, float],
    horizon_bars: int,
    timeframe_seconds: int,
    source: str,
    notional: float,
    args: dict[str, Any],
) -> dict[str, float]:
    fee = row_value(row, "cost_fee_bps_round_trip", defaults.get("fee_bps_round_trip", 20.0))
    slippage, participation, quote_volume = estimate_dynamic_slippage_bps(row, defaults, notional, args)
    spread = row_value(row, "cost_spread_bps_one_way", defaults.get("spread_bps_one_way", 0.0))
    impact = row_value(
        row,
        "cost_market_impact_bps_one_way",
        defaults.get("market_impact_bps_one_way", 0.0),
    )
    funding_per_day = exchange_funding_bps_per_day(source, row, defaults, args)
    horizon_days = (horizon_bars * timeframe_seconds) / 86400.0
    funding = funding_per_day * horizon_days
    spread_impact = (2.0 * spread) + (2.0 * impact)
    total = fee + slippage + spread_impact + funding
    return {
        "fee_bps": round(fee, 8),
        "slippage_bps": round(slippage, 8),
        "spread_impact_bps": round(spread_impact, 8),
        "funding_bps": round(funding, 8),
        "total_cost_bps": round(total, 8),
        "funding_bps_per_day": round(funding_per_day, 8),
        "volume_participation": participation,
        "estimated_quote_volume": quote_volume,
    }


def estimate_fill_ratio(row: pd.Series, requested_notional: float, args: dict[str, Any]) -> float:
    quote_volume = estimate_quote_volume(row)
    if requested_notional <= 0:
        return 0.0
    if quote_volume <= 0:
        return safe_float(args.get("missing_volume_fill_ratio"), 0.50)
    max_participation = max(0.000001, safe_float(args.get("max_volume_participation"), 0.02))
    max_fill_notional = quote_volume * max_participation
    min_fill_ratio = min(max(safe_float(args.get("min_partial_fill_ratio"), 0.10), 0.0), 1.0)
    return min(1.0, max(min_fill_ratio, max_fill_notional / requested_notional))


def compute_position_sizing(row: pd.Series, side: str, args: dict[str, Any], equity_value: float) -> dict[str, float]:
    stop_loss_bps = max(abs(safe_float(args.get("stop_loss_bps"), 70.0)), 1.0)
    risk_budget = equity_value * safe_float(args.get("risk_per_trade_pct"), 0.005)
    leverage = max(safe_float(args.get("leverage"), 1.0), 0.0)
    max_leverage = max(safe_float(args.get("max_leverage"), 1.0), 0.0)
    leverage = min(leverage, max_leverage)
    stop_loss_return = stop_loss_bps / 10000.0
    risk_notional = risk_budget / stop_loss_return
    max_position_notional = equity_value * safe_float(args.get("max_position_fraction"), 0.20) * max(leverage, 1.0)
    requested_notional = min(risk_notional, max_position_notional)
    fill_ratio = estimate_fill_ratio(row, requested_notional, args)
    filled_notional = requested_notional * fill_ratio
    margin_used = 0.0 if leverage <= 0 else filled_notional / leverage
    return {
        "equity_before": round(equity_value, 8),
        "risk_budget": round(risk_budget, 8),
        "requested_notional": round(requested_notional, 8),
        "filled_notional": round(filled_notional, 8),
        "fill_ratio": round(fill_ratio, 8),
        "margin_used": round(margin_used, 8),
        "leverage": round(leverage, 8),
        "capital_fraction": round(0.0 if equity_value <= 0 else margin_used / equity_value, 10),
    }


def execution_prices(
    entry_row: pd.Series,
    side: str,
    order_type: str,
    slippage_bps_round_trip: float,
) -> dict[str, Any]:
    close = price_value(entry_row, "Close")
    open_price = price_value(entry_row, "Open", fallback=close)
    high = price_value(entry_row, "High", fallback=max(open_price, close))
    low = price_value(entry_row, "Low", fallback=min(open_price, close))
    mid = open_price if order_type == "market" else close
    one_way_slippage = max(slippage_bps_round_trip / 2.0, 0.0)
    if order_type == "limit":
        one_way_slippage *= 0.35
    direction = 1.0 if side == "long" else -1.0
    entry_price = mid * (1.0 + direction * one_way_slippage / 10000.0)
    if side == "long":
        entry_price = min(max(entry_price, low), high)
    else:
        entry_price = min(max(entry_price, low), high)
    return {
        "signal_close_price": close,
        "entry_reference_price": mid,
        "entry_fill_price": round(entry_price, 12),
        "entry_bar_open": open_price,
        "entry_bar_high": high,
        "entry_bar_low": low,
        "order_type": order_type,
        "entry_one_way_slippage_bps": round(one_way_slippage, 8),
    }


def choose_columns(columns: set[str], horizon: int) -> dict[str, str | None]:
    candidates = {
        "gross_long_bps": f"label_fwd_ret_gross_long_bps_h{horizon}",
        "net_long_bps": f"label_fwd_ret_net_long_bps_h{horizon}",
        "gross_long": f"label_fwd_ret_gross_long_h{horizon}",
        "net_long": f"label_fwd_ret_net_long_h{horizon}",
        "gross_short_bps": f"label_fwd_ret_gross_short_bps_h{horizon}",
        "net_short_bps": f"label_fwd_ret_net_short_bps_h{horizon}",
        "gross_short": f"label_fwd_ret_gross_short_h{horizon}",
        "net_short": f"label_fwd_ret_net_short_h{horizon}",
        "mfe_long_bps": f"label_future_mfe_long_bps_h{horizon}",
        "mae_long_bps": f"label_future_mae_long_bps_h{horizon}",
        "mfe_short_bps": f"label_future_mfe_short_bps_h{horizon}",
        "mae_short_bps": f"label_future_mae_short_bps_h{horizon}",
        "future_rr_long": f"label_future_rr_long_h{horizon}",
        "future_vol_regime": f"label_future_vol_regime_h{horizon}",
    }
    return {key: value if value in columns else None for key, value in candidates.items()}


def annualize_return(total_return: float, days: float) -> float | None:
    if days <= 0:
        return None
    base = 1.0 + total_return
    if base <= 0:
        return -1.0
    try:
        cagr = base ** (365.0 / days) - 1.0
    except OverflowError:
        return None
    if not math.isfinite(cagr):
        return None
    return cagr


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity / peaks) - 1.0
    return float(np.min(drawdowns))


def longest_losing_streak(returns: list[float]) -> int:
    longest = 0
    current = 0
    for value in returns:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def compact_metrics(trades: pd.DataFrame, equity: pd.DataFrame, target_annual_return: float, reference_drawdown_limit: float | None, min_coverage_days: float, min_trades: int) -> dict[str, Any]:
    if trades.empty or equity.empty:
        return {
            "status": "NO_TRADES",
            "approval_status": "NOT_AN_APPROVAL_ENGINE",
            "total_trades": 0,
            "total_return": 0.0,
            "cagr": None,
            "max_drawdown": 0.0,
            "win_rate": None,
            "profit_factor": None,
            "avg_trade_net_bps": None,
            "coverage_days": 0.0,
            "target_annual_return_min": target_annual_return,
            "reference_drawdown_limit": reference_drawdown_limit,
            "meets_target_annual_return": False,
            "within_reference_drawdown": True,
        }

    returns = trades["portfolio_return"].astype(float).to_numpy()
    equity_values = equity["equity"].astype(float).to_numpy()
    total_return = float(equity_values[-1] - 1.0)
    first_ts = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce").min()
    last_ts = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce").max()
    coverage_days = 0.0
    if pd.notna(first_ts) and pd.notna(last_ts):
        coverage_days = max(float((last_ts - first_ts).total_seconds() / 86400.0), 0.0)
    cagr = annualize_return(total_return, coverage_days)
    mdd = max_drawdown(equity_values)
    gains = float(np.sum(returns[returns > 0]))
    losses = float(abs(np.sum(returns[returns < 0])))
    profit_factor = None if losses == 0 else gains / losses
    win_rate = float(np.mean(returns > 0)) if returns.size else None
    avg_net_bps = float(np.mean(trades["net_return_bps"].astype(float))) if len(trades) else None
    median_net_bps = float(np.median(trades["net_return_bps"].astype(float))) if len(trades) else None
    meets_return = cagr is not None and cagr >= target_annual_return
    within_reference_drawdown = (
        True if reference_drawdown_limit is None else abs(mdd) <= reference_drawdown_limit
    )

    if coverage_days < min_coverage_days or len(trades) < min_trades:
        status = "RESEARCH_ONLY_INSUFFICIENT_SAMPLE"
    else:
        status = "RESEARCH_METRICS_READY"

    return {
        "status": status,
        "approval_status": "NOT_AN_APPROVAL_ENGINE",
        "total_trades": int(len(trades)),
        "total_return": round(total_return, 10),
        "cagr": None if cagr is None else round(float(cagr), 10),
        "max_drawdown": round(float(mdd), 10),
        "calmar": None if cagr is None or mdd == 0 else round(float(cagr / abs(mdd)), 10),
        "win_rate": None if win_rate is None else round(win_rate, 10),
        "profit_factor": None if profit_factor is None else round(float(profit_factor), 10),
        "avg_trade_net_bps": None if avg_net_bps is None else round(avg_net_bps, 8),
        "median_trade_net_bps": None if median_net_bps is None else round(median_net_bps, 8),
        "best_trade_portfolio_return": round(float(np.max(returns)), 10),
        "worst_trade_portfolio_return": round(float(np.min(returns)), 10),
        "longest_losing_streak": longest_losing_streak(list(returns)),
        "coverage_days": round(coverage_days, 6),
        "target_annual_return_min": target_annual_return,
        "reference_drawdown_limit": reference_drawdown_limit,
        "meets_target_annual_return": bool(meets_return),
        "within_reference_drawdown": bool(within_reference_drawdown),
        "annualization_warning": coverage_days < min_coverage_days,
        "approval_criteria_note": "Essas flags sao referencias de pesquisa; nao aprovam execucao nem testnet por si mesmas.",
    }


def make_equity_curve(trades: pd.DataFrame, return_column: str = "portfolio_return") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["event_index", "timestamp_utc_ms", "time", "equity", "drawdown"])
    sorted_trades = trades.sort_values(["exit_timestamp_utc_ms", "entry_timestamp_utc_ms"]).reset_index(drop=True)
    equity_values: list[float] = []
    timestamps: list[int] = []
    times: list[Any] = []
    equity = 1.0
    peak = 1.0
    drawdowns: list[float] = []
    for _, row in sorted_trades.iterrows():
        trade_return = safe_float(row[return_column], 0.0)
        trade_return = max(trade_return, -0.99)
        equity *= 1.0 + trade_return
        peak = max(peak, equity)
        equity_values.append(equity)
        timestamps.append(safe_int(row["exit_timestamp_utc_ms"]))
        times.append(row.get("exit_time"))
        drawdowns.append((equity / peak) - 1.0)
    return pd.DataFrame(
        {
            "event_index": np.arange(1, len(equity_values) + 1, dtype=np.int64),
            "timestamp_utc_ms": timestamps,
            "time": times,
            "equity": equity_values,
            "drawdown": drawdowns,
        }
    )


def apply_path_exit(
    row: pd.Series,
    selected_cols: dict[str, str | None],
    side: str,
    gross_fwd_bps: float,
    stop_loss_bps: float,
    take_profit_bps: float,
    use_path_labels: bool,
) -> tuple[float, str, bool]:
    if not use_path_labels:
        return gross_fwd_bps, "HORIZON", False

    mfe_col = selected_cols.get(f"mfe_{side}_bps")
    mae_col = selected_cols.get(f"mae_{side}_bps")
    if not mfe_col or not mae_col:
        return gross_fwd_bps, "HORIZON_NO_PATH_LABELS", False

    mfe = safe_float(row.get(mfe_col), default=float("nan"))
    mae = safe_float(row.get(mae_col), default=float("nan"))
    if not math.isfinite(mfe) or not math.isfinite(mae):
        return gross_fwd_bps, "HORIZON_NO_PATH_LABELS", False

    favorable_bps = max(mfe, 0.0)
    adverse_bps = mae if mae <= 0 else -abs(mae)

    # Conservador quando stop e take profit aparecem dentro da mesma janela:
    # assume stop primeiro, pois os labels nao registram a ordem intrabar.
    if stop_loss_bps > 0 and adverse_bps <= -abs(stop_loss_bps):
        return -abs(stop_loss_bps), "STOP_LOSS_CONSERVATIVE", True
    if take_profit_bps > 0 and favorable_bps >= abs(take_profit_bps):
        return abs(take_profit_bps), "TAKE_PROFIT", True
    return gross_fwd_bps, "HORIZON", True


def simulate_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    args = payload["args"]
    experiment = payload["experiment"]
    cost_model = payload["cost_model"]
    datasets_by_series = payload["datasets_by_series"]
    run_id = payload["run_id"]

    started = time.perf_counter()
    phase_timings: dict[str, float] = {}
    experiment_id = str(experiment.get("experiment_id") or "UNKNOWN")
    asset = str(experiment.get("asset") or "UNKNOWN")
    symbol = clean_symbol(experiment.get("symbol"), asset)
    source = str(experiment.get("source") or "unknown_source")
    timeframe = str(experiment.get("timeframe") or "unknown_timeframe")
    target_col = str(experiment.get("target_col") or args["target"])
    horizon_bars = parse_horizon_bars(target_col)
    timeframe_seconds = timeframe_to_seconds(timeframe)
    horizon_ms = horizon_bars * timeframe_seconds * 1000

    dataset_path = dataset_path_from_experiment(experiment, datasets_by_series)
    predictions_path = experiment.get("predictions_path")
    result_base = {
        "experiment_id": experiment_id,
        "asset": asset,
        "symbol": symbol,
        "source": source,
        "timeframe": timeframe,
        "target_col": target_col,
        "horizon_bars": horizon_bars,
    }

    try:
        if experiment.get("status") not in {None, "OK"}:
            return {
                **result_base,
                "status": "SKIPPED_NON_OK_EXPERIMENT",
                "reason": experiment.get("status"),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
        if not dataset_path or not Path(dataset_path).is_file():
            return {
                **result_base,
                "status": "ERROR",
                "error": "dataset_path_missing",
                "dataset_path": dataset_path,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
        if not predictions_path or not Path(str(predictions_path)).is_file():
            return {
                **result_base,
                "status": "ERROR",
                "error": "predictions_path_missing",
                "predictions_path": predictions_path,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }

        t0 = time.perf_counter()
        pred = pd.read_parquet(str(predictions_path))
        phase_timings["read_predictions_seconds"] = round(time.perf_counter() - t0, 6)

        if "timestamp_utc_ms" not in pred.columns or "proba_long" not in pred.columns:
            return {
                **result_base,
                "status": "ERROR",
                "error": "prediction_schema_missing_timestamp_or_proba",
                "predictions_path": str(predictions_path),
                "prediction_columns": list(pred.columns),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }

        t0 = time.perf_counter()
        columns = parquet_columns(dataset_path)
        selected_cols = choose_columns(columns, horizon_bars)
        read_cols = ["timestamp_utc_ms"]
        if "DateTime" in columns:
            read_cols.append("DateTime")
        for price_col in ["Open", "High", "Low", "Close", "Volume"]:
            if price_col in columns:
                read_cols.append(price_col)
        for col in [value for value in selected_cols.values() if value]:
            if col not in read_cols:
                read_cols.append(col)
        for col in COST_COLUMNS:
            if col in columns and col not in read_cols:
                read_cols.append(col)
        data = pd.read_parquet(dataset_path, columns=read_cols)
        data = data.sort_values("timestamp_utc_ms").reset_index(drop=True)
        data["dataset_row_index"] = np.arange(len(data), dtype=np.int64)
        phase_timings["read_dataset_columns_seconds"] = round(time.perf_counter() - t0, 6)

        t0 = time.perf_counter()
        merged = pred.merge(data, on="timestamp_utc_ms", how="inner", suffixes=("_pred", ""))
        merged = merged.sort_values("timestamp_utc_ms").reset_index(drop=True)
        phase_timings["join_seconds"] = round(time.perf_counter() - t0, 6)

        if merged.empty:
            return {
                **result_base,
                "status": "NO_JOINED_ROWS",
                "dataset_path": dataset_path,
                "predictions_path": str(predictions_path),
                "predictions_rows": int(len(pred)),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }

        has_long_returns = any(
            selected_cols.get(key)
            for key in ["gross_long_bps", "net_long_bps", "gross_long", "net_long"]
        )
        has_short_returns = any(
            selected_cols.get(key)
            for key in ["gross_short_bps", "net_short_bps", "gross_short", "net_short"]
        )
        if not has_long_returns and not has_short_returns:
            return {
                **result_base,
                "status": "ERROR",
                "error": "return_columns_missing_for_horizon",
                "horizon_bars": horizon_bars,
                "dataset_path": dataset_path,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }

        t0 = time.perf_counter()
        defaults = cost_defaults(cost_model, source, symbol)
        trades: list[dict[str, Any]] = []
        skipped_overlap = 0
        skipped_threshold = 0
        skipped_not_finite = 0
        skipped_execution = 0
        halted_after_ts: int | None = None
        last_exit_ts = -1
        initial_capital = safe_float(args.get("initial_capital"), DEFAULT_INITIAL_CAPITAL)
        equity_capital = initial_capital
        peak_capital = initial_capital
        current_drawdown = 0.0

        for _, row in merged.iterrows():
            signal_ts = safe_int(row.get("timestamp_utc_ms"), default=0)
            proba_long = safe_float(row.get("proba_long"), default=float("nan"))
            if not math.isfinite(proba_long):
                skipped_not_finite += 1
                continue
            if halted_after_ts is not None and signal_ts > halted_after_ts:
                continue
            side = None
            if has_long_returns and proba_long >= args["entry_threshold"]:
                side = "long"
            elif has_short_returns and args["allow_short"] and proba_long <= args["short_entry_threshold"]:
                side = "short"
            if side is None:
                skipped_threshold += 1
                continue
            signal_idx = safe_int(row.get("dataset_row_index"), default=-1)
            entry_idx = signal_idx + safe_int(args.get("execution_latency_bars"), default=1)
            exit_idx = entry_idx + horizon_bars
            if signal_idx < 0 or entry_idx >= len(data) or exit_idx >= len(data):
                skipped_execution += 1
                continue

            entry_row = data.iloc[entry_idx]
            exit_row = data.iloc[exit_idx]
            entry_ts = safe_int(entry_row.get("timestamp_utc_ms"), default=0)
            exit_ts = safe_int(exit_row.get("timestamp_utc_ms"), default=0)
            if entry_ts <= last_exit_ts:
                skipped_overlap += 1
                continue

            sizing = compute_position_sizing(entry_row, side, args, equity_capital)
            if sizing["filled_notional"] <= 0:
                skipped_execution += 1
                continue
            cost = estimate_cost_bps(
                entry_row,
                defaults,
                horizon_bars,
                timeframe_seconds,
                source,
                sizing["filled_notional"],
                args,
            )
            fills = execution_prices(
                entry_row,
                side,
                str(args.get("order_type", "market")).lower(),
                cost["slippage_bps"],
            )
            gross_col = selected_cols.get(f"gross_{side}_bps")
            net_col = selected_cols.get(f"net_{side}_bps")
            gross_decimal_col = selected_cols.get(f"gross_{side}")
            net_decimal_col = selected_cols.get(f"net_{side}")
            if gross_col:
                gross_fwd_bps = safe_float(entry_row.get(gross_col), default=float("nan"))
            elif gross_decimal_col:
                gross_fwd_bps = safe_float(entry_row.get(gross_decimal_col), default=float("nan")) * 10000.0
            elif net_col:
                gross_fwd_bps = safe_float(entry_row.get(net_col), default=float("nan")) + cost["total_cost_bps"]
            else:
                gross_fwd_bps = safe_float(entry_row.get(net_decimal_col), default=float("nan")) * 10000.0 + cost["total_cost_bps"]

            if not math.isfinite(gross_fwd_bps):
                skipped_not_finite += 1
                continue

            gross_exec_bps, exit_reason, used_path = apply_path_exit(
                row=entry_row,
                selected_cols=selected_cols,
                side=side,
                gross_fwd_bps=gross_fwd_bps,
                stop_loss_bps=args["stop_loss_bps"],
                take_profit_bps=args["take_profit_bps"],
                use_path_labels=args["use_path_labels"],
            )
            net_return_bps = gross_exec_bps - cost["total_cost_bps"]
            edge_bps = ((proba_long - 0.5) if side == "long" else (0.5 - proba_long)) * 10000.0
            min_edge_bps = max(
                args["min_edge_bps"],
                row_value(row, "cost_min_net_edge_bps", defaults.get("min_net_edge_bps", 0.0)),
            )
            if args["require_edge_over_cost"] and edge_bps < (cost["total_cost_bps"] + min_edge_bps):
                skipped_threshold += 1
                continue

            net_return_unlevered = net_return_bps / 10000.0
            pnl = sizing["filled_notional"] * net_return_unlevered
            portfolio_return = max(-0.99, min(10.0, pnl / max(equity_capital, 1.0)))
            net_return_levered = portfolio_return
            signal_time = pd.to_datetime(signal_ts, unit="ms", utc=True)
            entry_time = pd.to_datetime(entry_ts, unit="ms", utc=True)
            exit_time = pd.to_datetime(exit_ts, unit="ms", utc=True)
            if side == "long":
                exit_fill_price = fills["entry_fill_price"] * (1.0 + gross_exec_bps / 10000.0)
                signal_to_entry_gap_bps = (
                    (fills["entry_reference_price"] / fills["signal_close_price"]) - 1.0
                ) * 10000.0
            else:
                exit_fill_price = fills["entry_fill_price"] * (1.0 - gross_exec_bps / 10000.0)
                signal_to_entry_gap_bps = (
                    (fills["signal_close_price"] / fills["entry_reference_price"]) - 1.0
                ) * 10000.0

            equity_capital += pnl
            peak_capital = max(peak_capital, equity_capital)
            current_drawdown = (equity_capital / peak_capital) - 1.0
            if (
                halted_after_ts is None
                and args["drawdown_kill_switch_pct"] > 0
                and abs(current_drawdown) >= args["drawdown_kill_switch_pct"]
            ):
                halted_after_ts = exit_ts

            trades.append(
                {
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "asset": asset,
                    "symbol": symbol,
                    "source": source,
                    "timeframe": timeframe,
                    "target_col": target_col,
                    "horizon_bars": horizon_bars,
                    "signal_timestamp_utc_ms": signal_ts,
                    "entry_timestamp_utc_ms": entry_ts,
                    "exit_timestamp_utc_ms": exit_ts,
                    "signal_time": signal_time.isoformat(),
                    "entry_time": entry_time.isoformat(),
                    "exit_time": exit_time.isoformat(),
                    "side": side,
                    "order_type": fills["order_type"],
                    "execution_latency_bars": args["execution_latency_bars"],
                    "proba_long": proba_long,
                    "pred_binary": safe_int(row.get("pred_binary"), default=1),
                    "actual_binary": safe_int(row.get("actual_binary"), default=0),
                    "signal_close_price": round(fills["signal_close_price"], 12),
                    "entry_reference_price": round(fills["entry_reference_price"], 12),
                    "entry_fill_price": round(fills["entry_fill_price"], 12),
                    "exit_fill_price": round(exit_fill_price, 12),
                    "signal_to_entry_gap_bps": round(signal_to_entry_gap_bps, 8),
                    "gross_forward_bps": round(gross_fwd_bps, 8),
                    "gross_execution_bps": round(gross_exec_bps, 8),
                    "net_return_bps": round(net_return_bps, 8),
                    "net_return_unlevered": round(net_return_unlevered, 10),
                    "net_return_levered": round(net_return_levered, 10),
                    "portfolio_return": round(portfolio_return, 10),
                    "pnl": round(pnl, 8),
                    "equity_before": sizing["equity_before"],
                    "equity_after": round(equity_capital, 8),
                    "risk_budget": sizing["risk_budget"],
                    "requested_notional": sizing["requested_notional"],
                    "filled_notional": sizing["filled_notional"],
                    "fill_ratio": sizing["fill_ratio"],
                    "margin_used": sizing["margin_used"],
                    "leverage": sizing["leverage"],
                    "position_fraction": sizing["capital_fraction"],
                    "exit_reason": exit_reason,
                    "used_path_labels": used_path,
                    "fee_bps": cost["fee_bps"],
                    "slippage_bps": cost["slippage_bps"],
                    "spread_impact_bps": cost["spread_impact_bps"],
                    "funding_bps": cost["funding_bps"],
                    "funding_bps_per_day": cost["funding_bps_per_day"],
                    "total_cost_bps": cost["total_cost_bps"],
                    "volume_participation": cost["volume_participation"],
                    "estimated_quote_volume": cost["estimated_quote_volume"],
                    "edge_bps_proxy": round(edge_bps, 8),
                    "min_edge_bps": round(min_edge_bps, 8),
                    "equity_after_trade": round(equity_capital / initial_capital, 10),
                    "drawdown_after_trade": round(current_drawdown, 10),
                }
            )
            last_exit_ts = exit_ts

        trades_df = pd.DataFrame(trades)
        equity_df = make_equity_curve(trades_df) if not trades_df.empty else pd.DataFrame()
        metrics = compact_metrics(
            trades_df,
            equity_df,
            args["target_annual_return"],
            args["reference_drawdown_limit"],
            args["min_coverage_days"],
            args["min_trades"],
        )
        phase_timings["simulation_seconds"] = round(time.perf_counter() - t0, 6)

        safe_name = f"{experiment_id}_{symbol}_{timeframe}".replace("/", "_").replace("\\", "_")
        trades_path = TRADES_DIR / f"{safe_name}_trades.parquet"
        equity_path = EQUITY_DIR / f"{safe_name}_equity.parquet"
        if args.get("persist_artifacts", True) and not trades_df.empty:
            trades_df.to_parquet(trades_path, index=False)
            equity_df.to_parquet(equity_path, index=False)
        else:
            trades_path = None
            equity_path = None

        return {
            **result_base,
            "status": "OK",
            "dataset_path": dataset_path,
            "predictions_path": str(predictions_path),
            "trades_path": str(trades_path) if trades_path else None,
            "equity_path": str(equity_path) if equity_path else None,
            "rows": {
                "predictions_rows": int(len(pred)),
                "dataset_rows_read": int(len(data)),
                "joined_rows": int(len(merged)),
                "trades": int(len(trades_df)),
                "skipped_threshold_or_edge": int(skipped_threshold),
                "skipped_overlap": int(skipped_overlap),
                "skipped_not_finite": int(skipped_not_finite),
                "skipped_execution": int(skipped_execution),
            },
            "risk_controls": {
                "halted_after_timestamp_utc_ms": halted_after_ts,
                "drawdown_kill_switch_pct": args["drawdown_kill_switch_pct"],
                "stop_loss_bps": args["stop_loss_bps"],
                "take_profit_bps": args["take_profit_bps"],
                "use_path_labels": args["use_path_labels"],
                "reference_drawdown_limit": args["reference_drawdown_limit"],
            },
            "side_distribution": trades_df["side"].value_counts(dropna=False).to_dict() if not trades_df.empty else {},
            "metrics": metrics,
            "selected_columns": selected_cols,
            "cost_defaults": defaults,
            "phase_timings": phase_timings,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "_trades_records": (
                trades_df.to_dict(orient="records")
                if args.get("return_trades_inline", False) and not trades_df.empty
                else []
            ),
        }
    except Exception as exc:
        return {
            **result_base,
            "status": "ERROR",
            "error": str(exc),
            "traceback_tail": traceback.format_exc(limit=8),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }


def build_portfolio_result(
    experiment_results: list[dict[str, Any]],
    args: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    trade_frames = []
    for result in experiment_results:
        path = result.get("trades_path")
        if path and Path(path).is_file():
            trade_frames.append(pd.read_parquet(path))
        elif result.get("_trades_records"):
            trade_frames.append(pd.DataFrame(result["_trades_records"]))

    if not trade_frames:
        return {
            "status": "NO_TRADES",
            "trades_path": None,
            "equity_path": None,
            "metrics": compact_metrics(
                pd.DataFrame(),
                pd.DataFrame(),
                args["target_annual_return"],
                args["reference_drawdown_limit"],
                args["min_coverage_days"],
                args["min_trades"],
            ),
        }

    trades = pd.concat(trade_frames, ignore_index=True)
    trades = trades.sort_values(["entry_timestamp_utc_ms", "exit_timestamp_utc_ms"]).reset_index(drop=True)
    initial_capital = safe_float(args.get("initial_capital"), DEFAULT_INITIAL_CAPITAL)
    cash_equity = initial_capital
    peak_equity = initial_capital
    active: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    equity_events: list[dict[str, Any]] = []
    skipped_capacity = 0
    skipped_conflict = 0
    max_active_positions_observed = 0
    max_notional_exposure_observed = 0.0

    def close_positions_until(timestamp_ms: int) -> None:
        nonlocal active, cash_equity, peak_equity, max_notional_exposure_observed
        remaining: list[dict[str, Any]] = []
        for pos in active:
            if safe_int(pos["exit_timestamp_utc_ms"]) <= timestamp_ms:
                equity_before_close = cash_equity
                pnl = safe_float(pos["portfolio_pnl"], 0.0)
                cash_equity += pnl
                peak_equity = max(peak_equity, cash_equity)
                drawdown = (cash_equity / peak_equity) - 1.0 if peak_equity > 0 else 0.0
                pos["portfolio_equity_before_close"] = round(equity_before_close, 8)
                pos["portfolio_equity_after_close"] = round(cash_equity, 8)
                pos["portfolio_return"] = round(pnl / max(equity_before_close, 1.0), 10)
                pos["portfolio_drawdown_after_close"] = round(drawdown, 10)
                accepted_rows.append(pos)
                equity_events.append(
                    {
                        "event_index": len(equity_events) + 1,
                        "timestamp_utc_ms": safe_int(pos["exit_timestamp_utc_ms"]),
                        "time": pos.get("exit_time"),
                        "equity": cash_equity / initial_capital if initial_capital > 0 else 1.0,
                        "drawdown": drawdown,
                    }
                )
            else:
                remaining.append(pos)
        active = remaining
        active_notional = sum(safe_float(item.get("portfolio_allocated_notional"), 0.0) for item in active)
        max_notional_exposure_observed = max(max_notional_exposure_observed, active_notional)

    for _, row in trades.iterrows():
        entry_ts = safe_int(row.get("entry_timestamp_utc_ms"), default=0)
        close_positions_until(entry_ts)
        max_active_positions_observed = max(max_active_positions_observed, len(active))
        if len(active) >= args["max_concurrent_positions"]:
            skipped_capacity += 1
            continue
        symbol = str(row.get("symbol") or "")
        side = str(row.get("side") or "")
        if args["conflict_policy"] == "skip_same_symbol" and any(str(pos.get("symbol")) == symbol for pos in active):
            skipped_conflict += 1
            continue

        active_notional = sum(safe_float(item.get("portfolio_allocated_notional"), 0.0) for item in active)
        active_asset_notional = sum(
            safe_float(item.get("portfolio_allocated_notional"), 0.0)
            for item in active
            if str(item.get("symbol")) == symbol
        )
        leverage = max(safe_float(row.get("leverage"), args["leverage"]), 1.0)
        max_total_notional = cash_equity * args["max_total_exposure_fraction"] * leverage
        max_asset_notional = cash_equity * args["max_asset_exposure_fraction"] * leverage
        requested = safe_float(row.get("filled_notional"), default=0.0)
        alloc = min(
            requested,
            max(0.0, max_total_notional - active_notional),
            max(0.0, max_asset_notional - active_asset_notional),
            cash_equity * args["max_position_fraction"] * leverage,
        )
        if alloc <= 0:
            skipped_capacity += 1
            continue
        net_return_unlevered = safe_float(row.get("net_return_unlevered"), default=0.0)
        portfolio_pnl = alloc * net_return_unlevered
        margin_used = alloc / leverage
        position = row.to_dict()
        position.update(
            {
                "portfolio_allocated_notional": round(alloc, 8),
                "portfolio_margin_used": round(margin_used, 8),
                "portfolio_active_positions_at_entry": len(active),
                "portfolio_notional_exposure_before_entry": round(active_notional, 8),
                "portfolio_asset_notional_before_entry": round(active_asset_notional, 8),
                "portfolio_pnl": round(portfolio_pnl, 8),
                "portfolio_return": 0.0,
                "portfolio_conflict_policy": args["conflict_policy"],
            }
        )
        active.append(position)
        max_active_positions_observed = max(max_active_positions_observed, len(active))
        max_notional_exposure_observed = max(max_notional_exposure_observed, active_notional + alloc)

    close_positions_until(10**30)
    accepted = pd.DataFrame(accepted_rows)
    if accepted.empty:
        return {
            "status": "NO_ACCEPTED_TRADES",
            "trades_path": None,
            "equity_path": None,
            "metrics": compact_metrics(
                pd.DataFrame(),
                pd.DataFrame(),
                args["target_annual_return"],
                args["reference_drawdown_limit"],
                args["min_coverage_days"],
                args["min_trades"],
            ),
            "portfolio_construction": {
                "method": "event_driven_active_positions",
                "skipped_capacity": skipped_capacity,
                "skipped_conflict": skipped_conflict,
            },
        }
    equity = pd.DataFrame(equity_events)
    metrics = compact_metrics(
        accepted,
        equity,
        args["target_annual_return"],
        args["reference_drawdown_limit"],
        args["min_coverage_days"],
        args["min_trades"],
    )

    trades_path = TRADES_DIR / f"PORTFOLIO_{run_id}_trades.parquet"
    equity_path = EQUITY_DIR / f"PORTFOLIO_{run_id}_equity.parquet"
    if args.get("persist_artifacts", True):
        accepted.to_parquet(trades_path, index=False)
        equity.to_parquet(equity_path, index=False)
    else:
        trades_path = None
        equity_path = None

    by_asset = (
        accepted.groupby(["source", "asset", "symbol", "timeframe", "side"], dropna=False)
        .agg(
            trades=("portfolio_return", "size"),
            avg_net_bps=("net_return_bps", "mean"),
            win_rate=("portfolio_return", lambda s: float(np.mean(s > 0))),
            total_slot_return=("portfolio_return", "sum"),
        )
        .reset_index()
        .sort_values("trades", ascending=False)
    )

    return {
        "status": "OK",
        "trades_path": str(trades_path) if trades_path else None,
        "equity_path": str(equity_path) if equity_path else None,
        "metrics": metrics,
        "portfolio_construction": {
            "method": "event_driven_active_positions_with_capital_and_margin",
            "engine": "event_driven_active_positions",
            "max_concurrent_positions": args["max_concurrent_positions"],
            "max_position_fraction": args["max_position_fraction"],
            "max_asset_exposure_fraction": args["max_asset_exposure_fraction"],
            "max_total_exposure_fraction": args["max_total_exposure_fraction"],
            "conflict_policy": args["conflict_policy"],
            "skipped_capacity": skipped_capacity,
            "skipped_conflict": skipped_conflict,
            "max_active_positions_observed": max_active_positions_observed,
            "max_notional_exposure_observed": round(max_notional_exposure_observed, 8),
            "note": "Motor event-driven de pesquisa com posicoes simultaneas; ainda nao substitui simulador completo de exchange com book e filas reais.",
        },
        "side_distribution": accepted["side"].value_counts(dropna=False).to_dict() if "side" in accepted.columns else {},
        "top_groups_by_trade_count": by_asset.head(30).to_dict(orient="records"),
    }


def update_ai_context_index(backtest_summary: dict[str, Any]) -> None:
    if not AI_CONTEXT_INDEX_PATH.is_file():
        return
    try:
        index = load_json(AI_CONTEXT_INDEX_PATH, default={})
        if not isinstance(index, dict):
            return

        files = index.setdefault("files", [])
        existing = {item.get("file"): item for item in files if isinstance(item, dict)}
        for file_name, role, hint in [
            (
                "7_JSON_BACKTEST_PORTFOLIO.json",
                "backtest de portfolio",
                "Use para avaliar PnL liquido, custos, stops, take profit, sizing, drawdown e meta anual.",
            ),
            (
                "7_BACKTEST_PORTFOLIO_RUN_REPORT_LATEST.json",
                "telemetria da etapa 7",
                "Use para performance do codigo, tempos por fase e erros do backtest.",
            ),
            (
                "7_BACKTEST_PARAM_SEARCH_LATEST.json",
                "busca de parametros do backtest",
                "Use para comparar thresholds, stops, take profit, long-only vs long-short e filtros de custo.",
            ),
            (
                "7_BACKTEST_PARAM_SEARCH_RUN_REPORT_LATEST.json",
                "telemetria da busca de parametros",
                "Use para auditar ranking, score e custo computacional da calibracao da etapa 7.",
            ),
            (
                "7_BACKTEST_VALIDATION_LATEST.json",
                "validacao automatica do backtest",
                "Use para verificar schema, trades, equity, custos, fill ratio e consistencia temporal.",
            ),
            (
                "7_BACKTEST_STRESS_LATEST.json",
                "stress test do backtest",
                "Use para avaliar sensibilidade a custos, slippage, funding e piores regimes aproximados.",
            ),
            (
                "7_EXPERIMENT_REGISTRY_LATEST.json",
                "registry de experimentos",
                "Use para versionar dataset, previsoes, custos, config, metricas, validacao e stress.",
            ),
        ]:
            path = BASE_JSON_DIR / file_name
            file_data = load_json(path, default={}) if path.is_file() else {}
            file_summary = file_data.get("summary", backtest_summary) if isinstance(file_data, dict) else backtest_summary
            existing[file_name] = {
                "file": file_name,
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "schema_version": file_data.get("schema_version") if isinstance(file_data, dict) else None,
                "run_id": file_summary.get("run_id") if isinstance(file_summary, dict) else backtest_summary.get("run_id"),
                "generated_at": file_summary.get("generated_at_utc") if isinstance(file_summary, dict) else backtest_summary.get("generated_at_utc"),
                "role": role,
                "ai_reading_hint": hint,
                "summary": file_summary,
            }
        index["files"] = list(existing.values())

        order = index.setdefault("ai_reading_order", [])
        for name in [
            "7_JSON_BACKTEST_PORTFOLIO.json",
            "7_BACKTEST_PORTFOLIO_RUN_REPORT_LATEST.json",
            "7_BACKTEST_PARAM_SEARCH_LATEST.json",
            "7_BACKTEST_PARAM_SEARCH_RUN_REPORT_LATEST.json",
            "7_BACKTEST_VALIDATION_LATEST.json",
            "7_BACKTEST_STRESS_LATEST.json",
            "7_EXPERIMENT_REGISTRY_LATEST.json",
        ]:
            if name not in order:
                order.append(name)

        summary = index.setdefault("summary", {})
        summary["backtest_portfolio_status"] = backtest_summary.get("status")
        summary["backtest_portfolio_total_trades"] = backtest_summary.get("portfolio_total_trades")
        summary["backtest_portfolio_total_return"] = backtest_summary.get("portfolio_total_return")
        summary["backtest_portfolio_cagr"] = backtest_summary.get("portfolio_cagr")
        summary["backtest_portfolio_max_drawdown"] = backtest_summary.get("portfolio_max_drawdown")
        summary["backtest_portfolio_research_status"] = backtest_summary.get("portfolio_research_status")
        summary["backtest_reference_drawdown_limit"] = backtest_summary.get("reference_drawdown_limit")
        summary["backtest_approval_status"] = backtest_summary.get("approval_status")
        param_search = load_json(PARAM_SEARCH_JSON_PATH, default={})
        param_summary = param_search.get("summary", {}) if isinstance(param_search, dict) else {}
        summary["backtest_param_search_status"] = param_summary.get("status")
        summary["backtest_param_search_candidates"] = param_summary.get("candidates_evaluated")
        summary["backtest_param_search_best_score"] = param_summary.get("best_score")
        summary["backtest_param_search_best_cagr"] = param_summary.get("best_cagr")
        summary["backtest_param_search_best_max_drawdown"] = param_summary.get("best_max_drawdown")
        summary["backtest_param_search_passes_research_references"] = param_summary.get("passes_research_references")
        validation = load_json(VALIDATION_JSON_PATH, default={})
        stress = load_json(STRESS_JSON_PATH, default={})
        registry = load_json(EXPERIMENT_REGISTRY_JSON_PATH, default={})
        summary["backtest_validation_status"] = (validation.get("summary") or {}).get("status") if isinstance(validation, dict) else None
        summary["backtest_stress_worst_scenario"] = (stress.get("summary") or {}).get("worst_scenario") if isinstance(stress, dict) else None
        summary["backtest_registry_latest_config_hash"] = (registry.get("summary") or {}).get("latest_config_hash") if isinstance(registry, dict) else None

        recommendations = index.setdefault("recommendations_for_codex", [])
        recommendation = (
            "Antes de ligar execucao em testnet, use 7_JSON_BACKTEST_PORTFOLIO.json "
            "para validar PnL liquido, drawdown, custos, stop/take profit e estabilidade."
        )
        if recommendation not in recommendations:
            recommendations.append(recommendation)

        write_json_atomic(AI_CONTEXT_INDEX_PATH, index)
    except Exception:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARCHANGEL etapa 7 - backtest de portfolio.")
    parser.add_argument("--target", default="label_dir_h20_thr25bps")
    parser.add_argument("--entry-threshold", type=float, default=0.55)
    parser.add_argument("--short-entry-threshold", type=float, default=None)
    parser.add_argument("--allow-short", default="true")
    parser.add_argument("--min-edge-bps", type=float, default=0.0)
    parser.add_argument("--require-edge-over-cost", action="store_true")
    parser.add_argument("--stop-loss-bps", type=float, default=70.0)
    parser.add_argument("--take-profit-bps", type=float, default=100.0)
    parser.add_argument("--use-path-labels", default="true")
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--max-leverage", type=float, default=5.0)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.005)
    parser.add_argument("--max-position-fraction", type=float, default=0.20)
    parser.add_argument("--max-asset-exposure-fraction", type=float, default=0.30)
    parser.add_argument("--max-total-exposure-fraction", type=float, default=1.00)
    parser.add_argument("--max-concurrent-positions", type=int, default=5)
    parser.add_argument("--portfolio-slot-fraction", type=float, default=0.20)
    parser.add_argument("--drawdown-kill-switch-pct", type=float, default=0.08)
    parser.add_argument("--reference-drawdown-limit", type=float, default=DEFAULT_REFERENCE_DRAWDOWN_LIMIT)
    parser.add_argument("--target-max-drawdown", type=float, default=None, help="Alias legado: tratado como reference_drawdown_limit.")
    parser.add_argument("--target-annual-return", type=float, default=DEFAULT_TARGET_ANNUAL_RETURN)
    parser.add_argument("--order-type", choices=["market", "limit"], default="market")
    parser.add_argument("--execution-latency-bars", type=int, default=1)
    parser.add_argument("--max-volume-participation", type=float, default=0.02)
    parser.add_argument("--slippage-impact-multiplier", type=float, default=250.0)
    parser.add_argument("--min-partial-fill-ratio", type=float, default=0.10)
    parser.add_argument("--missing-volume-fill-ratio", type=float, default=0.50)
    parser.add_argument("--funding-bps-per-day", type=float, default=None)
    parser.add_argument("--conflict-policy", choices=["skip_same_symbol", "allow"], default="skip_same_symbol")
    parser.add_argument("--min-coverage-days", type=float, default=30.0)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--search-mode", choices=["none", "grid", "optuna"], default="none")
    parser.add_argument("--run-validations", default="true")
    parser.add_argument("--run-stress-tests", default="true")
    parser.add_argument("--threshold-grid", default="0.52,0.55,0.58,0.60")
    parser.add_argument("--stop-loss-grid", default="50,70,100")
    parser.add_argument("--take-profit-grid", default="70,100,150")
    parser.add_argument("--allow-short-grid", default="true,false")
    parser.add_argument("--require-edge-grid", default="false")
    parser.add_argument("--leverage-grid", default="")
    parser.add_argument("--max-search-runs", type=int, default=36)
    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> dict[str, Any]:
    leverage = min(max(args.leverage, 0.0), max(args.max_leverage, 0.0))
    max_workers = args.max_workers
    if max_workers <= 0:
        max_workers = min(8, max(1, (os.cpu_count() or 2) - 2))
    short_entry_threshold = args.short_entry_threshold
    if short_entry_threshold is None:
        short_entry_threshold = 1.0 - float(args.entry_threshold)
    reference_drawdown_limit = args.reference_drawdown_limit
    if args.target_max_drawdown is not None:
        reference_drawdown_limit = args.target_max_drawdown
    return {
        "target": args.target,
        "entry_threshold": float(args.entry_threshold),
        "short_entry_threshold": float(short_entry_threshold),
        "allow_short": safe_bool(args.allow_short, default=True),
        "min_edge_bps": float(args.min_edge_bps),
        "require_edge_over_cost": bool(args.require_edge_over_cost),
        "stop_loss_bps": float(args.stop_loss_bps),
        "take_profit_bps": float(args.take_profit_bps),
        "use_path_labels": safe_bool(args.use_path_labels, default=True),
        "initial_capital": max(float(args.initial_capital), 1.0),
        "risk_per_trade_pct": min(max(float(args.risk_per_trade_pct), 0.0), 1.0),
        "max_position_fraction": min(max(float(args.max_position_fraction), 0.0), 1.0),
        "max_asset_exposure_fraction": min(max(float(args.max_asset_exposure_fraction), 0.0), 5.0),
        "max_total_exposure_fraction": min(max(float(args.max_total_exposure_fraction), 0.0), 10.0),
        "leverage": float(leverage),
        "max_leverage": float(args.max_leverage),
        "position_fraction": min(max(float(args.position_fraction), 0.0), 1.0),
        "max_concurrent_positions": max(1, int(args.max_concurrent_positions)),
        "portfolio_slot_fraction": min(max(float(args.portfolio_slot_fraction), 0.0), 1.0),
        "drawdown_kill_switch_pct": min(max(float(args.drawdown_kill_switch_pct), 0.0), 1.0),
        "target_annual_return": float(args.target_annual_return),
        "reference_drawdown_limit": min(max(float(reference_drawdown_limit), 0.0), 1.0),
        "order_type": args.order_type,
        "execution_latency_bars": max(0, int(args.execution_latency_bars)),
        "max_volume_participation": min(max(float(args.max_volume_participation), 0.000001), 1.0),
        "slippage_impact_multiplier": max(float(args.slippage_impact_multiplier), 0.0),
        "min_partial_fill_ratio": min(max(float(args.min_partial_fill_ratio), 0.0), 1.0),
        "missing_volume_fill_ratio": min(max(float(args.missing_volume_fill_ratio), 0.0), 1.0),
        "funding_bps_per_day": args.funding_bps_per_day,
        "conflict_policy": args.conflict_policy,
        "min_coverage_days": float(args.min_coverage_days),
        "min_trades": int(args.min_trades),
        "max_workers": int(max_workers),
        "limit": max(0, int(args.limit)),
        "search_mode": args.search_mode,
        "run_validations": safe_bool(args.run_validations, default=True),
        "run_stress_tests": safe_bool(args.run_stress_tests, default=True),
        "threshold_grid": args.threshold_grid,
        "stop_loss_grid": args.stop_loss_grid,
        "take_profit_grid": args.take_profit_grid,
        "allow_short_grid": args.allow_short_grid,
        "require_edge_grid": args.require_edge_grid,
        "leverage_grid": args.leverage_grid,
        "max_search_runs": max(1, int(args.max_search_runs)),
        "persist_artifacts": True,
        "return_trades_inline": False,
    }


# =============================================================================
# 3. BACKTEST E BUSCA DE PARAMETROS
# =============================================================================

def load_backtest_inputs(args: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    walk_forward = load_json(WALK_FORWARD_JSON_PATH, default={})
    datasets_json = load_json(DATASETS_JSON_PATH, default={})
    cost_model = load_json(COST_MODEL_PATH, default={})
    datasets_by_series = build_datasets_index(datasets_json if isinstance(datasets_json, dict) else {})

    experiments = walk_forward.get("experiments", []) if isinstance(walk_forward, dict) else []
    if args["limit"] > 0:
        experiments = experiments[: args["limit"]]
    return experiments, cost_model, datasets_by_series


def no_experiments_payload(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "system": {
            "name": "ARCHANGEL",
            "version": "v1",
            "layer": "7_BACKTEST_PORTFOLIO",
            "script": SCRIPT_NAME,
            "run_id": run_id,
            "generated_at_utc": utc_now_iso(),
        },
        "paths": {
            "walk_forward_json_path": str(WALK_FORWARD_JSON_PATH),
            "datasets_json_path": str(DATASETS_JSON_PATH),
            "cost_model_path": str(COST_MODEL_PATH),
            "backtest_json_path": str(BACKTEST_JSON_PATH),
            "run_report_latest_path": str(RUN_REPORT_LATEST_PATH),
        },
        "summary": {
            "run_id": run_id,
            "generated_at_utc": utc_now_iso(),
            "status": "ERROR_NO_EXPERIMENTS",
            "experiments_found": 0,
        },
    }


def strip_inline_trades(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for item in results:
        clean = dict(item)
        clean.pop("_trades_records", None)
        stripped.append(clean)
    return stripped


def config_hash(config: dict[str, Any]) -> str:
    text = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def validate_backtest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    summary = payload.get("summary", {})
    portfolio = payload.get("portfolio", {})
    trades_path = portfolio.get("trades_path")
    equity_path = portfolio.get("equity_path")
    add("summary_status_ok", summary.get("status") == "OK", f"status={summary.get('status')}")
    add("portfolio_exists", portfolio.get("status") == "OK", f"portfolio_status={portfolio.get('status')}")
    add("trades_path_exists", bool(trades_path and Path(trades_path).is_file()), str(trades_path))
    add("equity_path_exists", bool(equity_path and Path(equity_path).is_file()), str(equity_path))

    if trades_path and Path(trades_path).is_file():
        trades = pd.read_parquet(trades_path)
        required = [
            "entry_timestamp_utc_ms",
            "exit_timestamp_utc_ms",
            "side",
            "net_return_bps",
            "portfolio_return",
            "total_cost_bps",
            "fill_ratio",
            "portfolio_allocated_notional",
            "portfolio_margin_used",
        ]
        missing = [col for col in required if col not in trades.columns]
        add("required_trade_columns", not missing, f"missing={missing}")
        if not trades.empty:
            add("entry_before_exit", bool((trades["entry_timestamp_utc_ms"] < trades["exit_timestamp_utc_ms"]).all()), "entry_timestamp_utc_ms < exit_timestamp_utc_ms")
            add("finite_returns", bool(np.isfinite(trades["portfolio_return"].astype(float)).all()), "portfolio_return finite")
            add("nonnegative_costs", bool((trades["total_cost_bps"].astype(float) >= 0).all()), "total_cost_bps >= 0")
            add("fill_ratio_bounds", bool(((trades["fill_ratio"].astype(float) >= 0) & (trades["fill_ratio"].astype(float) <= 1)).all()), "0 <= fill_ratio <= 1")
            add("nonnegative_notional", bool((trades["portfolio_allocated_notional"].astype(float) >= 0).all()), "portfolio_allocated_notional >= 0")
            duplicated = trades.duplicated(subset=["experiment_id", "entry_timestamp_utc_ms", "side"]).sum()
            add("no_duplicate_trade_keys", duplicated == 0, f"duplicates={int(duplicated)}")

    if equity_path and Path(equity_path).is_file():
        equity = pd.read_parquet(equity_path)
        if not equity.empty:
            add("equity_positive", bool((equity["equity"].astype(float) > 0).all()), "equity > 0")
            add("drawdown_nonpositive", bool((equity["drawdown"].astype(float) <= 0.0000001).all()), "drawdown <= 0")
            add("equity_timestamps_sorted", bool(equity["timestamp_utc_ms"].is_monotonic_increasing), "timestamp_utc_ms monotonic")

    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": "ARCHANGEL_BACKTEST_VALIDATION_1.0",
        "system": {
            "name": "ARCHANGEL",
            "layer": "7_BACKTEST_VALIDATION",
            "script": SCRIPT_NAME,
            "run_id": summary.get("run_id"),
            "generated_at_utc": utc_now_iso(),
        },
        "summary": {
            "status": "PASS" if passed else "FAIL",
            "checks_total": len(checks),
            "checks_passed": sum(1 for item in checks if item["passed"]),
            "checks_failed": sum(1 for item in checks if not item["passed"]),
            "run_id": summary.get("run_id"),
        },
        "checks": checks,
    }


def stressed_metrics_from_trades(
    trades: pd.DataFrame,
    scenario: dict[str, float],
    args: dict[str, Any],
) -> dict[str, Any]:
    if trades.empty:
        return compact_metrics(
            pd.DataFrame(),
            pd.DataFrame(),
            args["target_annual_return"],
            args["reference_drawdown_limit"],
            args["min_coverage_days"],
            args["min_trades"],
        )
    stressed = trades.copy()
    fee = stressed["fee_bps"].astype(float) * scenario.get("fee_mult", 1.0)
    slippage = stressed["slippage_bps"].astype(float) * scenario.get("slippage_mult", 1.0)
    spread_impact = stressed["spread_impact_bps"].astype(float) * scenario.get("spread_impact_mult", 1.0)
    funding = stressed["funding_bps"].astype(float) * scenario.get("funding_mult", 1.0)
    extra = scenario.get("extra_adverse_bps", 0.0)
    gross = stressed["gross_execution_bps"].astype(float) - extra
    stressed["net_return_bps"] = gross - fee - slippage - spread_impact - funding
    stressed["net_return_unlevered"] = stressed["net_return_bps"] / 10000.0

    equity_value = safe_float(args.get("initial_capital"), DEFAULT_INITIAL_CAPITAL)
    peak = equity_value
    returns: list[float] = []
    equity_rows: list[dict[str, Any]] = []
    for _, row in stressed.sort_values(["exit_timestamp_utc_ms", "entry_timestamp_utc_ms"]).iterrows():
        before = equity_value
        pnl = safe_float(row.get("portfolio_allocated_notional"), 0.0) * safe_float(row.get("net_return_unlevered"), 0.0)
        equity_value += pnl
        peak = max(peak, equity_value)
        ret = pnl / max(before, 1.0)
        returns.append(ret)
        equity_rows.append(
            {
                "event_index": len(equity_rows) + 1,
                "timestamp_utc_ms": safe_int(row.get("exit_timestamp_utc_ms")),
                "time": row.get("exit_time"),
                "equity": equity_value / safe_float(args.get("initial_capital"), DEFAULT_INITIAL_CAPITAL),
                "drawdown": (equity_value / peak) - 1.0,
            }
        )
    stressed["portfolio_return"] = returns
    equity = pd.DataFrame(equity_rows)
    return compact_metrics(
        stressed,
        equity,
        args["target_annual_return"],
        args["reference_drawdown_limit"],
        args["min_coverage_days"],
        args["min_trades"],
    )


def run_stress_tests(payload: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    trades_path = payload.get("portfolio", {}).get("trades_path")
    scenarios = [
        {"name": "BASE_REPLAY", "fee_mult": 1.0, "slippage_mult": 1.0, "spread_impact_mult": 1.0, "funding_mult": 1.0, "extra_adverse_bps": 0.0},
        {"name": "COSTS_150PCT", "fee_mult": 1.5, "slippage_mult": 1.5, "spread_impact_mult": 1.5, "funding_mult": 1.5, "extra_adverse_bps": 0.0},
        {"name": "COSTS_200PCT", "fee_mult": 2.0, "slippage_mult": 2.0, "spread_impact_mult": 2.0, "funding_mult": 2.0, "extra_adverse_bps": 0.0},
        {"name": "SLIPPAGE_SHOCK", "fee_mult": 1.0, "slippage_mult": 3.0, "spread_impact_mult": 2.0, "funding_mult": 1.0, "extra_adverse_bps": 5.0},
        {"name": "FUNDING_SHOCK", "fee_mult": 1.0, "slippage_mult": 1.0, "spread_impact_mult": 1.0, "funding_mult": 4.0, "extra_adverse_bps": 0.0},
        {"name": "WORST_REGIME_PROXY", "fee_mult": 1.5, "slippage_mult": 2.0, "spread_impact_mult": 2.0, "funding_mult": 2.0, "extra_adverse_bps": 25.0},
    ]
    if not trades_path or not Path(trades_path).is_file():
        results = [{"scenario": item["name"], "status": "NO_TRADES"} for item in scenarios]
    else:
        trades = pd.read_parquet(trades_path)
        results = []
        for scenario in scenarios:
            metrics = stressed_metrics_from_trades(trades, scenario, args)
            results.append({"scenario": scenario["name"], "assumptions": scenario, "metrics": metrics})

    worst = None
    metric_results = [item for item in results if isinstance(item.get("metrics"), dict)]
    if metric_results:
        worst = min(metric_results, key=lambda item: safe_float(item["metrics"].get("total_return"), 0.0))
    return {
        "schema_version": "ARCHANGEL_BACKTEST_STRESS_1.0",
        "system": {
            "name": "ARCHANGEL",
            "layer": "7_BACKTEST_STRESS",
            "script": SCRIPT_NAME,
            "run_id": payload.get("summary", {}).get("run_id"),
            "generated_at_utc": utc_now_iso(),
        },
        "summary": {
            "status": "OK",
            "scenarios": len(results),
            "worst_scenario": worst.get("scenario") if worst else None,
            "worst_total_return": worst.get("metrics", {}).get("total_return") if worst else None,
            "worst_max_drawdown": worst.get("metrics", {}).get("max_drawdown") if worst else None,
            "run_id": payload.get("summary", {}).get("run_id"),
        },
        "scenarios": results,
    }


def update_experiment_registry(payload: dict[str, Any], validation: dict[str, Any], stress: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    entry = {
        "run_id": summary.get("run_id"),
        "generated_at_utc": utc_now_iso(),
        "schema_version": "ARCHANGEL_EXPERIMENT_REGISTRY_ENTRY_1.0",
        "script": SCRIPT_NAME,
        "config_hash": config_hash(payload.get("config", {})),
        "dataset_manifest": str(DATASETS_JSON_PATH),
        "walk_forward_manifest": str(WALK_FORWARD_JSON_PATH),
        "cost_model": str(COST_MODEL_PATH),
        "backtest_json_path": str(BACKTEST_JSON_PATH),
        "trades_path": payload.get("portfolio", {}).get("trades_path"),
        "equity_path": payload.get("portfolio", {}).get("equity_path"),
        "config": payload.get("config", {}),
        "metrics": payload.get("portfolio", {}).get("metrics", {}),
        "validation_status": validation.get("summary", {}).get("status"),
        "stress_summary": stress.get("summary", {}),
        "status": summary.get("status"),
        "research_note": "Registro de experimento para rastrear dataset, modelo, custos, risco, metricas e status. Nao e aprovacao operacional.",
    }
    EXPERIMENT_REGISTRY_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_REGISTRY_JSONL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    entries: list[dict[str, Any]] = []
    if EXPERIMENT_REGISTRY_JSONL_PATH.is_file():
        lines = EXPERIMENT_REGISTRY_JSONL_PATH.read_text(encoding="utf-8").splitlines()
        for line in lines[-200:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    registry = {
        "schema_version": "ARCHANGEL_EXPERIMENT_REGISTRY_1.0",
        "system": {
            "name": "ARCHANGEL",
            "layer": "7_EXPERIMENT_REGISTRY",
            "script": SCRIPT_NAME,
            "generated_at_utc": utc_now_iso(),
        },
        "paths": {
            "registry_json_path": str(EXPERIMENT_REGISTRY_JSON_PATH),
            "registry_jsonl_path": str(EXPERIMENT_REGISTRY_JSONL_PATH),
        },
        "summary": {
            "status": "OK",
            "entries_retained": len(entries),
            "latest_run_id": summary.get("run_id"),
            "latest_config_hash": entry["config_hash"],
        },
        "latest_entry": entry,
        "entries_tail": entries,
    }
    write_json_atomic(EXPERIMENT_REGISTRY_JSON_PATH, registry)
    return registry


def run_backtest_once(
    args: dict[str, Any],
    run_id: str,
    experiments: list[dict[str, Any]],
    cost_model: dict[str, Any],
    datasets_by_series: dict[str, dict[str, Any]],
    started_dt: datetime | None = None,
    persist_json: bool = True,
    print_progress: bool = True,
    search_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    if started_dt is None:
        started_dt = utc_now()
    started = time.perf_counter()

    if not experiments:
        payload = no_experiments_payload(run_id)
        if persist_json:
            write_json_atomic(BACKTEST_JSON_PATH, payload)
            write_json_atomic(RUN_REPORT_LATEST_PATH, payload)
        return payload, 1

    jobs = [
        {
            "args": args,
            "experiment": experiment,
            "cost_model": cost_model,
            "datasets_by_series": datasets_by_series,
            "run_id": run_id,
        }
        for experiment in experiments
    ]

    results: list[dict[str, Any]] = []
    if print_progress:
        print(f"[INFO] Experimentos selecionados: {len(jobs)}")
    if args["max_workers"] <= 1 or len(jobs) == 1:
        for job in jobs:
            results.append(simulate_experiment(job))
    else:
        with ProcessPoolExecutor(max_workers=args["max_workers"]) as executor:
            future_to_id = {
                executor.submit(simulate_experiment, job): job["experiment"].get("experiment_id")
                for job in jobs
            }
            for future in as_completed(future_to_id):
                results.append(future.result())

    results = sorted(results, key=lambda item: (str(item.get("asset")), str(item.get("timeframe")), str(item.get("experiment_id"))))
    portfolio = build_portfolio_result(results, args, run_id)

    finished_dt = utc_now()
    elapsed = time.perf_counter() - started
    ok_results = [item for item in results if item.get("status") == "OK"]
    error_results = [item for item in results if item.get("status") == "ERROR"]
    no_trade_results = [item for item in results if item.get("status") == "OK" and item.get("metrics", {}).get("total_trades") == 0]
    portfolio_metrics = portfolio.get("metrics", {})
    status = "OK" if not error_results else "OK_WITH_ERRORS"

    summary = {
        "run_id": run_id,
        "generated_at_utc": finished_dt.isoformat(timespec="seconds"),
        "status": status,
        "experiments_selected": int(len(results)),
        "experiments_ok": int(len(ok_results)),
        "experiments_error": int(len(error_results)),
        "experiments_without_trades": int(len(no_trade_results)),
        "portfolio_status": portfolio.get("status"),
        "portfolio_research_status": portfolio_metrics.get("status"),
        "portfolio_total_trades": portfolio_metrics.get("total_trades"),
        "portfolio_total_return": portfolio_metrics.get("total_return"),
        "portfolio_cagr": portfolio_metrics.get("cagr"),
        "portfolio_max_drawdown": portfolio_metrics.get("max_drawdown"),
        "portfolio_win_rate": portfolio_metrics.get("win_rate"),
        "portfolio_profit_factor": portfolio_metrics.get("profit_factor"),
        "target_annual_return_min": args["target_annual_return"],
        "reference_drawdown_limit": args["reference_drawdown_limit"],
        "approval_status": "NOT_AN_APPROVAL_ENGINE",
        "elapsed_seconds": round(elapsed, 6),
    }
    if search_metadata:
        summary["search_metadata"] = search_metadata

    payload = {
        "schema_version": SCHEMA_VERSION,
        "system": {
            "name": "ARCHANGEL",
            "version": "v1",
            "layer": "7_BACKTEST_PORTFOLIO",
            "script": SCRIPT_NAME,
            "run_id": run_id,
            "generated_at_utc": finished_dt.isoformat(timespec="seconds"),
        },
        "paths": {
            "root_dir": str(ROOT_DIR),
            "rules_dir": str(RULES_DIR),
            "base_json_dir": str(BASE_JSON_DIR),
            "walk_forward_json_path": str(WALK_FORWARD_JSON_PATH),
            "datasets_json_path": str(DATASETS_JSON_PATH),
            "cost_model_path": str(COST_MODEL_PATH),
            "backtest_dir": str(BACKTEST_DIR),
            "trades_dir": str(TRADES_DIR),
            "equity_dir": str(EQUITY_DIR),
            "registry_dir": str(REGISTRY_DIR),
            "validation_dir": str(VALIDATION_DIR),
            "stress_dir": str(STRESS_DIR),
            "backtest_json_path": str(BACKTEST_JSON_PATH),
            "run_report_path": str(LOGS_DIR / f"7_BACKTEST_PORTFOLIO_RUN_REPORT_{run_id}.json"),
            "run_report_latest_path": str(RUN_REPORT_LATEST_PATH),
            "validation_json_path": str(VALIDATION_JSON_PATH),
            "stress_json_path": str(STRESS_JSON_PATH),
            "experiment_registry_json_path": str(EXPERIMENT_REGISTRY_JSON_PATH),
            "experiment_registry_jsonl_path": str(EXPERIMENT_REGISTRY_JSONL_PATH),
        },
        "policy": {
            "source_predictions": "Somente previsoes OOS salvas pela etapa 6.",
            "source_returns": "Labels futuros da etapa 4/5; labels nao sao usados como features, apenas para medir PnL realizado no backtest.",
            "costs": "Custos explicitos do dataset e COST_MODEL.json; fee, slippage, spread, impacto e funding por horizonte.",
            "stops": "Stop/take profit sao aproximados por MFE/MAE. Se ambos ocorrem na janela, assume stop primeiro por conservadorismo.",
            "sides": "Long quando proba_long >= entry_threshold; short quando allow_short=True e proba_long <= short_entry_threshold.",
            "live_trading": "Nenhuma execucao live. Este modulo e pesquisa/backtest.",
        },
        "config": args,
        "summary": summary,
        "portfolio": portfolio,
        "experiments": strip_inline_trades(results),
        "telemetry": {
            "started_at_utc": started_dt.isoformat(timespec="seconds"),
            "finished_at_utc": finished_dt.isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 6),
            "process_memory_mb_end": process_memory_mb(),
            "cpu_count": os.cpu_count(),
            "max_workers": args["max_workers"],
            "rows_joined_total": int(sum(item.get("rows", {}).get("joined_rows", 0) for item in results)),
            "trades_total": int(sum(item.get("rows", {}).get("trades", 0) for item in results)),
        },
        "unused_or_future_code_notice": {
            "commented_but_missing_scripts": [
                "4B_GERA_LABELS_PANEL.py",
                "8_EXECUTION_MODULE.py",
            ],
            "note": "Esses nomes aparecem no roteiro do projeto como etapas futuras, mas nao existem como .py neste diretorio neste momento.",
        },
        "next_steps": [
            "Validar threshold, stop, take profit e sizing por grid/Optuna antes de qualquer testnet.",
            "Calibrar short-side e long/short simetrico antes de qualquer testnet.",
            "Criar backtest intrabar/event-driven quando houver trades/order book para reduzir aproximacao por MFE/MAE.",
            "Usar este JSON como gate antes do modulo de execucao em testnet Bybit/Binance/Kraken Pro.",
        ],
    }

    if persist_json:
        validation = validate_backtest_payload(payload) if args.get("run_validations", True) else {}
        stress = run_stress_tests(payload, args) if args.get("run_stress_tests", True) else {}
        registry = update_experiment_registry(payload, validation, stress)
        payload["validation"] = validation.get("summary", {})
        payload["stress_tests"] = stress.get("summary", {})
        payload["experiment_registry"] = registry.get("summary", {})
        summary["validation_status"] = validation.get("summary", {}).get("status")
        summary["stress_status"] = stress.get("summary", {}).get("status")
        summary["registry_status"] = registry.get("summary", {}).get("status")
        if validation:
            write_json_atomic(VALIDATION_JSON_PATH, validation)
        if stress:
            write_json_atomic(STRESS_JSON_PATH, stress)
        write_json_atomic(BACKTEST_JSON_PATH, payload)
        run_report_path = LOGS_DIR / f"7_BACKTEST_PORTFOLIO_RUN_REPORT_{run_id}.json"
        write_json_atomic(run_report_path, payload)
        write_json_atomic(RUN_REPORT_LATEST_PATH, payload)
        update_ai_context_index(summary)

    return payload, 0 if not error_results else 1


def score_backtest(metrics: dict[str, Any], target_annual_return: float, reference_drawdown_limit: float) -> float:
    cagr = metrics.get("cagr")
    if cagr is None:
        cagr = -1.0
    mdd = abs(safe_float(metrics.get("max_drawdown"), default=1.0))
    profit_factor = metrics.get("profit_factor")
    if profit_factor is None:
        profit_factor = 0.0
    win_rate = metrics.get("win_rate")
    if win_rate is None:
        win_rate = 0.0
    trades = safe_float(metrics.get("total_trades"), default=0.0)
    drawdown_penalty = max(0.0, mdd - reference_drawdown_limit) * 3.0
    low_trade_penalty = 0.20 if trades < 30 else 0.0
    target_bonus = 0.50 if cagr >= target_annual_return and mdd <= reference_drawdown_limit else 0.0
    return round(float(cagr) - drawdown_penalty + (0.10 * float(profit_factor)) + (0.05 * float(win_rate)) + target_bonus - low_trade_penalty, 10)


def build_grid_configs(base_args: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = parse_float_grid(base_args.get("threshold_grid"), [0.52, 0.55, 0.58, 0.60])
    stops = parse_float_grid(base_args.get("stop_loss_grid"), [50.0, 70.0, 100.0])
    takes = parse_float_grid(base_args.get("take_profit_grid"), [70.0, 100.0, 150.0])
    allow_shorts = parse_bool_grid(base_args.get("allow_short_grid"), [True, False])
    require_edges = parse_bool_grid(base_args.get("require_edge_grid"), [False])
    leverages = parse_float_grid(base_args.get("leverage_grid"), [base_args["leverage"]])

    configs: list[dict[str, Any]] = []
    for threshold in thresholds:
        for stop in stops:
            for take in takes:
                for allow_short in allow_shorts:
                    for require_edge in require_edges:
                        for leverage in leverages:
                            cfg = dict(base_args)
                            cfg["entry_threshold"] = float(threshold)
                            cfg["short_entry_threshold"] = 1.0 - float(threshold)
                            cfg["stop_loss_bps"] = float(stop)
                            cfg["take_profit_bps"] = float(take)
                            cfg["allow_short"] = bool(allow_short)
                            cfg["require_edge_over_cost"] = bool(require_edge)
                            cfg["leverage"] = min(float(leverage), cfg["max_leverage"])
                            cfg["persist_artifacts"] = False
                            cfg["return_trades_inline"] = True
                            configs.append(cfg)
    max_runs = max(1, int(base_args.get("max_search_runs", 36)))
    return configs[:max_runs]


def run_parameter_search(
    base_args: dict[str, Any],
    run_id: str,
    experiments: list[dict[str, Any]],
    cost_model: dict[str, Any],
    datasets_by_series: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    started_dt = utc_now()
    started = time.perf_counter()
    mode = str(base_args.get("search_mode") or "grid").lower()
    requested_mode = mode
    mode_note = None
    if mode not in {"grid", "optuna"}:
        mode = "grid"

    if mode == "optuna":
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except Exception as exc:
            mode = "grid"
            mode_note = f"Optuna indisponivel; fallback para grid. Erro: {exc}"

    rankings: list[dict[str, Any]] = []

    def evaluate_candidate(index: int, cfg: dict[str, Any], total: int) -> float:
        child_run_id = f"{run_id}_search_{index:03d}"
        payload, returncode = run_backtest_once(
            cfg,
            child_run_id,
            experiments,
            cost_model,
            datasets_by_series,
            persist_json=False,
            print_progress=False,
            search_metadata={"parent_run_id": run_id, "candidate_index": index},
        )
        metrics = payload.get("portfolio", {}).get("metrics", {})
        score = score_backtest(
            metrics,
            cfg["target_annual_return"],
            cfg["reference_drawdown_limit"],
        )
        rankings.append(
            {
                "rank": None,
                "candidate_index": index,
                "returncode": returncode,
                "score": score,
                "config": {
                    "entry_threshold": cfg["entry_threshold"],
                    "short_entry_threshold": cfg["short_entry_threshold"],
                    "allow_short": cfg["allow_short"],
                    "require_edge_over_cost": cfg["require_edge_over_cost"],
                    "stop_loss_bps": cfg["stop_loss_bps"],
                    "take_profit_bps": cfg["take_profit_bps"],
                    "leverage": cfg["leverage"],
                    "position_fraction": cfg["position_fraction"],
                    "portfolio_slot_fraction": cfg["portfolio_slot_fraction"],
                    "max_concurrent_positions": cfg["max_concurrent_positions"],
                },
                "metrics": metrics,
            }
        )
        print(
            "[BUSCA] "
            f"{index:03d}/{total:03d} "
            f"score={score} "
            f"ret={metrics.get('total_return')} "
            f"dd={metrics.get('max_drawdown')} "
            f"trades={metrics.get('total_trades')}"
        )
        return score

    max_search_runs = max(1, int(base_args.get("max_search_runs", 36)))
    print(f"[BUSCA] Modo solicitado: {requested_mode}")
    print(f"[BUSCA] Modo efetivo: {mode}")
    if mode_note:
        print(f"[BUSCA] Nota: {mode_note}")

    if mode == "optuna":
        thresholds = parse_float_grid(base_args.get("threshold_grid"), [0.52, 0.55, 0.58, 0.60])
        stops = parse_float_grid(base_args.get("stop_loss_grid"), [50.0, 70.0, 100.0])
        takes = parse_float_grid(base_args.get("take_profit_grid"), [70.0, 100.0, 150.0])
        allow_shorts = parse_bool_grid(base_args.get("allow_short_grid"), [True, False])
        require_edges = parse_bool_grid(base_args.get("require_edge_grid"), [False])
        leverages = parse_float_grid(base_args.get("leverage_grid"), [base_args["leverage"]])
        print(f"[BUSCA] Trials Optuna: {max_search_runs}")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        def objective(trial: Any) -> float:
            index = int(trial.number) + 1
            threshold = float(trial.suggest_categorical("entry_threshold", thresholds))
            cfg = dict(base_args)
            cfg["entry_threshold"] = threshold
            cfg["short_entry_threshold"] = 1.0 - threshold
            cfg["stop_loss_bps"] = float(trial.suggest_categorical("stop_loss_bps", stops))
            cfg["take_profit_bps"] = float(trial.suggest_categorical("take_profit_bps", takes))
            cfg["allow_short"] = bool(trial.suggest_categorical("allow_short", allow_shorts))
            cfg["require_edge_over_cost"] = bool(trial.suggest_categorical("require_edge_over_cost", require_edges))
            cfg["leverage"] = min(float(trial.suggest_categorical("leverage", leverages)), cfg["max_leverage"])
            cfg["persist_artifacts"] = False
            cfg["return_trades_inline"] = True
            return evaluate_candidate(index, cfg, max_search_runs)

        study.optimize(objective, n_trials=max_search_runs, show_progress_bar=False)
    else:
        configs = build_grid_configs(base_args)
        print(f"[BUSCA] Combinações avaliadas: {len(configs)}")
        for index, cfg in enumerate(configs, start=1):
            evaluate_candidate(index, cfg, len(configs))

    rankings = sorted(rankings, key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(rankings, start=1):
        item["rank"] = rank

    best = rankings[0] if rankings else None
    best_payload: dict[str, Any] | None = None
    best_returncode = 1
    if best:
        best_cfg = dict(base_args)
        best_cfg.update(best["config"])
        best_cfg["persist_artifacts"] = True
        best_cfg["return_trades_inline"] = False
        best_payload, best_returncode = run_backtest_once(
            best_cfg,
            f"{run_id}_best",
            experiments,
            cost_model,
            datasets_by_series,
            started_dt=utc_now(),
            persist_json=True,
            print_progress=True,
            search_metadata={
                "parent_search_run_id": run_id,
                "selected_candidate_index": best["candidate_index"],
                "selection_score": best["score"],
                "search_mode": mode,
            },
        )

    finished_dt = utc_now()
    best_metrics = best.get("metrics", {}) if best else {}
    passes_research_references = bool(
        best_metrics.get("cagr") is not None
        and best_metrics.get("cagr") >= base_args["target_annual_return"]
        and abs(safe_float(best_metrics.get("max_drawdown"), default=1.0)) <= base_args["reference_drawdown_limit"]
        and safe_int(best_metrics.get("total_trades"), default=0) >= base_args["min_trades"]
    )
    search_summary = {
        "run_id": run_id,
        "generated_at_utc": finished_dt.isoformat(timespec="seconds"),
        "status": "OK",
        "mode": mode,
        "requested_mode": requested_mode,
        "mode_note": mode_note,
        "candidates_evaluated": len(rankings),
        "best_candidate_index": best.get("candidate_index") if best else None,
        "best_score": best.get("score") if best else None,
        "best_research_status": best_metrics.get("status"),
        "best_total_trades": best_metrics.get("total_trades"),
        "best_total_return": best_metrics.get("total_return"),
        "best_cagr": best_metrics.get("cagr"),
        "best_max_drawdown": best_metrics.get("max_drawdown"),
        "best_win_rate": best_metrics.get("win_rate"),
        "best_profit_factor": best_metrics.get("profit_factor"),
        "passes_research_references": passes_research_references,
        "approval_status": "NOT_AN_APPROVAL_ENGINE",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    payload = {
        "schema_version": "ARCHANGEL_BACKTEST_PARAM_SEARCH_1.0",
        "system": {
            "name": "ARCHANGEL",
            "version": "v1",
            "layer": "7_BACKTEST_PARAM_SEARCH",
            "script": SCRIPT_NAME,
            "run_id": run_id,
            "generated_at_utc": finished_dt.isoformat(timespec="seconds"),
        },
        "paths": {
            "base_json_dir": str(BASE_JSON_DIR),
            "param_search_json_path": str(PARAM_SEARCH_JSON_PATH),
            "param_search_run_report_latest_path": str(PARAM_SEARCH_RUN_REPORT_LATEST_PATH),
            "best_backtest_json_path": str(BACKTEST_JSON_PATH),
            "best_backtest_run_report_latest_path": str(RUN_REPORT_LATEST_PATH),
        },
        "policy": {
            "purpose": "Comparar parametros da etapa 7. Nao buscar objetivo operacional nem aprovar estrategia.",
            "selection_rule": "Maior score ajustado por CAGR, drawdown, profit factor, win rate e penalizacao por pouca amostra.",
            "live_trading": "Nenhuma execucao live. Resultado e apenas filtro de pesquisa.",
            "warning": "Otimizar em poucos windows OOS pode overfitar. Confirmar em novas janelas e ativos antes de execucao.",
        },
        "config": {
            "base_config": {key: value for key, value in base_args.items() if not key.endswith("_grid")},
            "grids": {
                "threshold_grid": base_args.get("threshold_grid"),
                "stop_loss_grid": base_args.get("stop_loss_grid"),
                "take_profit_grid": base_args.get("take_profit_grid"),
                "allow_short_grid": base_args.get("allow_short_grid"),
                "require_edge_grid": base_args.get("require_edge_grid"),
                "leverage_grid": base_args.get("leverage_grid"),
                "max_search_runs": base_args.get("max_search_runs"),
            },
        },
        "summary": search_summary,
        "best_candidate": best,
        "top_candidates": rankings[:20],
        "all_candidates_compact": rankings,
        "best_backtest_summary": (best_payload or {}).get("summary"),
        "telemetry": {
            "started_at_utc": started_dt.isoformat(timespec="seconds"),
            "finished_at_utc": finished_dt.isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "process_memory_mb_end": process_memory_mb(),
            "cpu_count": os.cpu_count(),
            "inner_max_workers": base_args["max_workers"],
        },
        "next_steps": [
            "Usar a busca apenas como diagnostico de sensibilidade do backtest.",
            "Rodar nova busca apos ampliar features/modelos e walk-forward.",
            "Adicionar validacao por regime e por ativo antes de escolher parametros finais.",
        ],
    }
    write_json_atomic(PARAM_SEARCH_JSON_PATH, payload)
    write_json_atomic(PARAM_SEARCH_RUN_REPORT_LATEST_PATH, payload)
    update_ai_context_index((best_payload or {}).get("summary", search_summary))
    return payload, best_returncode


# =============================================================================
# 4. MAIN
# =============================================================================

def main() -> int:
    ensure_dirs()
    started_dt = utc_now()
    rid = run_id_now()
    parsed = parse_args()
    args = args_to_config(parsed)

    print("\nARCHANGEL v1 - ETAPA 7 BACKTEST PORTFOLIO")
    print("=" * 100)
    print(f"Run ID: {rid}")
    print(f"Rules dir: {RULES_DIR}")
    print(f"Base JSON: {BASE_JSON_DIR}")
    print(f"Workers: {args['max_workers']}")
    print(f"Entry threshold: {args['entry_threshold']}")
    print(f"Short threshold: {args['short_entry_threshold']} | allow_short={args['allow_short']}")
    print(f"Stop/TP bps: {args['stop_loss_bps']} / {args['take_profit_bps']}")
    print(f"Search mode: {args['search_mode']}")
    print("=" * 100)

    experiments, cost_model, datasets_by_series = load_backtest_inputs(args)

    if args["search_mode"] != "none":
        search_payload, returncode = run_parameter_search(
            args,
            rid,
            experiments,
            cost_model,
            datasets_by_series,
        )
        summary = search_payload["summary"]
        print("\nRESUMO BUSCA PARAMETROS")
        print("-" * 100)
        print(f"Candidatos: {summary['candidates_evaluated']}")
        print(f"Melhor score: {summary['best_score']}")
        print(f"Melhor retorno total: {summary['best_total_return']}")
        print(f"Melhor CAGR: {summary['best_cagr']}")
        print(f"Melhor max drawdown: {summary['best_max_drawdown']}")
        print(f"Passa referencias de pesquisa: {summary['passes_research_references']}")
        print(f"JSON busca: {PARAM_SEARCH_JSON_PATH}")
        print("-" * 100)
        return returncode

    args["persist_artifacts"] = True
    args["return_trades_inline"] = False
    payload, returncode = run_backtest_once(
        args,
        rid,
        experiments,
        cost_model,
        datasets_by_series,
        started_dt=started_dt,
        persist_json=True,
        print_progress=True,
    )
    summary = payload["summary"]

    print("\nRESUMO BACKTEST")
    print("-" * 100)
    print(f"Status: {summary['status']}")
    print(f"Experimentos OK/Erro: {summary['experiments_ok']} / {summary['experiments_error']}")
    print(f"Trades portfolio: {summary['portfolio_total_trades']}")
    print(f"Retorno total portfolio: {summary['portfolio_total_return']}")
    print(f"CAGR portfolio: {summary['portfolio_cagr']}")
    print(f"Max drawdown portfolio: {summary['portfolio_max_drawdown']}")
    print(f"Win rate: {summary['portfolio_win_rate']}")
    print(f"Profit factor: {summary['portfolio_profit_factor']}")
    print(f"JSON: {BACKTEST_JSON_PATH}")
    print("-" * 100)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
