"""
Background data pre-computation layer.
Replaces Streamlit's @st.cache_data with a background thread that
rebuilds all sub-fund data every 15 minutes.
"""

import threading
import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np

import portfolio as pf

log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

SUBFUND_FILES = {
    "Systematic": DATA_DIR / "systematic.csv",
    "Opportunistic": DATA_DIR / "opportunistic.csv",
    "Thematic": DATA_DIR / "thematic.csv",
    "Fixed Income": DATA_DIR / "fixed_income.csv",
}

SUBFUND_FILE_MAP = {
    "Systematic": "systematic.csv",
    "Opportunistic": "opportunistic.csv",
    "Thematic": "thematic.csv",
    "Fixed Income": "fixed_income.csv",
}

_cache = {}
_lock = threading.Lock()
_ready = threading.Event()


def refresh():
    """Rebuild all sub-fund data. Called by background thread every 15 min."""
    import concurrent.futures

    log.info("data_cache: refresh starting")
    new = {}

    existing = [(name, path) for name, path in SUBFUND_FILES.items() if path.exists()]
    if not existing:
        log.warning("data_cache: no sub-fund CSV files found")
        with _lock:
            _cache.clear()
        _ready.set()
        return

    # Build sub-funds in parallel (I/O bound — Alpaca API)
    subfund_data = {}
    errors = []

    def _load_one(name_path):
        name, path = name_path
        try:
            return name, pf.build_subfund(str(path)), None
        except Exception as e:
            return name, None, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(existing)) as executor:
        for name, data, err in executor.map(_load_one, existing):
            if err is not None:
                errors.append((name, err))
                log.error(f"data_cache: error loading {name}: {err}")
            elif data is not None:
                subfund_data[name] = data

    new["subfund_data"] = subfund_data

    # Benchmark returns
    bench_rets = {}
    for name, d in subfund_data.items():
        ticker = pf.BENCHMARKS.get(name)
        if not ticker:
            continue
        start = (d["first_date"] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end = (d["end_date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        bench_rets[name] = pf.fetch_benchmark_returns(ticker, start, end)
    new["benchmark_rets"] = bench_rets

    # Blended benchmark
    blended_bench_rets = pd.Series(dtype=float)
    if subfund_data:
        all_starts = [d["first_date"] for d in subfund_data.values()]
        all_ends = [d["end_date"] for d in subfund_data.values()]
        earliest = min(all_starts) - pd.Timedelta(days=5)
        bench_start = earliest.strftime("%Y-%m-%d")
        bench_end = (max(all_ends) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        blended_bench_rets = _load_blended_benchmark(bench_start, bench_end)
    new["blended_bench_rets"] = blended_bench_rets

    # Combined returns
    new["combined_rets"] = _combine_returns(subfund_data)

    with _lock:
        _cache.update(new)

    _ready.set()
    log.info(f"data_cache: refresh complete — {len(subfund_data)} sub-funds loaded")


def get_cache() -> dict:
    """Return a snapshot of the current cache. Blocks until first refresh completes."""
    _ready.wait()
    with _lock:
        return dict(_cache)


def _load_blended_benchmark(start: str, end: str) -> pd.Series:
    """Compute blended benchmark: 50% SPY + 25% IWV + 25% AGG."""
    tickers = ["SPY", "IWV", "AGG"]
    weights = {"SPY": 0.50, "IWV": 0.25, "AGG": 0.25}
    prices = pf.fetch_prices(tickers, start, end)
    if prices.empty:
        return pd.Series(dtype=float)
    rets = prices.pct_change().dropna()
    available = [t for t in tickers if t in rets.columns]
    if not available:
        return pd.Series(dtype=float)
    blended = sum(weights[t] * rets[t] for t in available)
    return blended


def _combine_returns(subfund_data: dict) -> pd.Series:
    """AUM-weighted average of sub-fund daily returns."""
    if len(subfund_data) == 1:
        return list(subfund_data.values())[0]["returns"]

    fund_rets = {}
    fund_aum = {}
    for name, d in subfund_data.items():
        r = d["returns"]
        pv = d["portfolio_values"]
        if r.empty or len(pv) < 2:
            continue
        fund_rets[name] = r
        fund_aum[name] = pv["Total"].clip(lower=0)

    if not fund_rets:
        return pd.Series(dtype=float)

    ret_df = pd.DataFrame(fund_rets).sort_index()
    aum_df = pd.DataFrame(fund_aum).sort_index().ffill().reindex(ret_df.index).ffill().fillna(0)

    weights = aum_df.shift(1)
    for name in fund_rets:
        first_date = fund_rets[name].index[0]
        weights.loc[weights.index < first_date, name] = np.nan

    row_totals = weights.sum(axis=1, min_count=1)
    weights = weights.div(row_totals, axis=0)

    common = weights.index.intersection(ret_df.index)
    combined = (ret_df.loc[common].fillna(0) * weights.loc[common].fillna(0)).sum(axis=1)
    combined = combined.dropna()
    combined = combined[combined.index >= ret_df.index[0]]
    return combined
