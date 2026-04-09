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

FACTOR_NAMES = ['Market', 'Momentum', 'Growth', 'Value']

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

    # Pre-compute expensive per-fund extras (factor betas, sectors, weekly attribution)
    fund_extras = {}
    for name, d in subfund_data.items():
        try:
            fund_extras[name] = _precompute_fund_extras(name, d)
            log.info(f"data_cache: precomputed extras for {name}")
        except Exception as e:
            log.error(f"data_cache: error precomputing extras for {name}: {e}")
            fund_extras[name] = {}
    new["fund_extras"] = fund_extras

    with _lock:
        _cache.update(new)

    _ready.set()
    log.info(f"data_cache: refresh complete — {len(subfund_data)} sub-funds loaded")


def get_cache() -> dict:
    """Return a snapshot of the current cache. Blocks until first refresh completes."""
    _ready.wait()
    with _lock:
        return dict(_cache)


# ── Per-fund pre-computation ─────────────────────────────────────────────

def _precompute_fund_extras(name, d):
    """Pre-compute expensive per-fund data: factor betas, sectors, weekly attribution."""
    extras = {}
    rets = d["returns"]
    holdings = d["holdings"]

    if rets.empty:
        return extras

    start_str = (d["first_date"] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end_str = (d["end_date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    extras["start_str"] = start_str
    extras["end_str"] = end_str

    # Factor betas (Fama-French style, stock-by-stock)
    try:
        extras["ff_betas"] = pf.compute_factor_betas(rets, holdings, start_str, end_str)
    except Exception as e:
        log.warning(f"  ff_betas failed for {name}: {e}")
        extras["ff_betas"] = {}

    # ETF factor betas
    try:
        extras["etf_betas"] = pf.compute_etf_factor_betas(rets, start_str, end_str)
    except Exception as e:
        log.warning(f"  etf_betas failed for {name}: {e}")
        extras["etf_betas"] = {}

    # Systematic-specific: 3 top-down regressions
    if name == "Systematic":
        # 1. Fund regression (actual fund returns vs all factors)
        try:
            extras["fund_regression"] = pf.regress_on_factors(rets, start_str, end_str)
        except Exception as e:
            log.warning(f"  fund_regression failed: {e}")
            extras["fund_regression"] = {}

        # 2. Current portfolio regression (holdings-weighted returns vs all factors)
        try:
            port_rets = pf.construct_portfolio_returns(holdings, start_str, end_str)
            if not port_rets.empty:
                extras["portfolio_regression"] = pf.regress_on_factors(port_rets, start_str, end_str)
            else:
                extras["portfolio_regression"] = {}
        except Exception as e:
            log.warning(f"  portfolio_regression failed: {e}")
            extras["portfolio_regression"] = {}

        # 3. Current portfolio ex-Momentum (same as #2 but drop momentum)
        try:
            if not port_rets.empty:
                extras["portfolio_regression_ex_mom"] = pf.regress_on_factors(
                    port_rets, start_str, end_str, exclude_factors=["momentum"]
                )
            else:
                extras["portfolio_regression_ex_mom"] = {}
        except Exception as e:
            log.warning(f"  portfolio_regression_ex_mom failed: {e}")
            extras["portfolio_regression_ex_mom"] = {}

        # 4. Factor correlation matrix
        try:
            factor_data = pd.read_parquet('data.parquet')
            factor_data.index = pd.to_datetime(factor_data.index).normalize()
            fd = factor_data.loc[pd.to_datetime(start_str):pd.to_datetime(end_str)]
            if len(fd) >= 10:
                extras["factor_corr"] = fd.corr()
            else:
                extras["factor_corr"] = None
        except Exception as e:
            log.warning(f"  factor_corr failed: {e}")
            extras["factor_corr"] = None

    # Sector exposure
    if not holdings.empty:
        try:
            extras["sectors"] = pf.get_sectors(holdings["Ticker"].tolist())
        except Exception as e:
            log.warning(f"  sectors failed for {name}: {e}")
            extras["sectors"] = {}
    else:
        extras["sectors"] = {}

    # Weekly return attribution blocks
    try:
        extras["weekly_attribution_blocks"] = _build_weekly_attribution(
            name, rets, holdings, start_str, end_str
        )
    except Exception as e:
        log.warning(f"  weekly_attribution failed for {name}: {e}")
        extras["weekly_attribution_blocks"] = []

    # Weekly factor attribution
    etf_betas = extras.get("etf_betas", {})
    try:
        weekly_attr = pf.weekly_factor_attribution(rets, etf_betas, start_str, end_str)
        if not weekly_attr.empty:
            wa_display = []
            for _, row in weekly_attr.iterrows():
                r = {}
                for col in weekly_attr.columns:
                    if col == "Week Ending":
                        r[col] = row[col]
                    else:
                        r[col] = f"{row[col]:+.3f}%"
                wa_display.append(r)
            extras["weekly_factor_attr"] = wa_display
            extras["weekly_factor_cols"] = weekly_attr.columns.tolist()
        else:
            extras["weekly_factor_attr"] = []
            extras["weekly_factor_cols"] = []
    except Exception as e:
        log.warning(f"  weekly_factor_attr failed for {name}: {e}")
        extras["weekly_factor_attr"] = []
        extras["weekly_factor_cols"] = []

    # Weekly theme attribution (Thematic only)
    if name == "Thematic":
        try:
            theme_map = pf.load_theme_map()
            weekly_theme = pf.weekly_theme_attribution(rets, holdings, theme_map, start_str, end_str)
            if not weekly_theme.empty:
                wt_display = []
                for _, row in weekly_theme.iterrows():
                    r = {}
                    for col in weekly_theme.columns:
                        if col == "Week Ending":
                            r[col] = row[col]
                        else:
                            r[col] = f"{row[col]:+.3f}%"
                    wt_display.append(r)
                extras["weekly_theme_attr"] = wt_display
                extras["weekly_theme_cols"] = weekly_theme.columns.tolist()
                extras["weekly_theme_raw"] = weekly_theme
            else:
                extras["weekly_theme_attr"] = []
                extras["weekly_theme_cols"] = []
                extras["weekly_theme_raw"] = None
        except Exception as e:
            log.warning(f"  weekly_theme_attr failed for {name}: {e}")
            extras["weekly_theme_attr"] = []
            extras["weekly_theme_cols"] = []
            extras["weekly_theme_raw"] = None

    return extras


def _extract_beta_map(betas_dict):
    """Extract factor name → beta value from a compute_factor_betas result."""
    bm = {}
    for fname in FACTOR_NAMES:
        for key, val in betas_dict.items():
            if key.startswith("_"):
                continue
            if fname.lower() in key.lower():
                bm[fname] = val
                break
    return bm


def _build_weekly_attribution(name, rets, holdings, start_str, end_str):
    """Build weekly return attribution data for a sub-fund."""
    factor_data = pd.read_parquet('data.parquet')
    factor_data.index = pd.to_datetime(factor_data.index).normalize()
    factor_data = factor_data.rename(columns={
        'mkt': 'Market', 'momentum': 'Momentum', 'growth': 'Growth', 'value': 'Value'
    })

    port_clean = rets.copy()
    port_clean.index = pd.to_datetime(port_clean.index).normalize()
    port_weekly = (1 + port_clean).resample("W-FRI").prod() - 1
    port_weekly = port_weekly.dropna()

    factor_weekly = (1 + factor_data).resample("W-FRI").prod() - 1

    spy_prices = pf.fetch_prices(["SPY"], start_str, end_str)
    spy_daily = spy_prices["SPY"].pct_change().dropna() if "SPY" in spy_prices.columns else pd.Series(dtype=float)
    spy_daily.index = pd.to_datetime(spy_daily.index).normalize()
    spy_weekly = (1 + spy_daily).resample("W-FRI").prod() - 1 if not spy_daily.empty else pd.Series(dtype=float)

    blocks = []

    snapshots = pf.list_holdings_snapshots(name)
    if snapshots:
        for _friday, _path in snapshots:
            try:
                snap_df = pf.load_holdings_snapshot(name, _friday)
            except Exception:
                continue
            if snap_df.empty:
                continue
            label = f"Week ending {_friday.strftime('%b %d, %Y')} \u00b7 uploaded snapshot ({len(snap_df)} tickers)"
            block = _render_attribution_block(
                _friday, snap_df, label, rets, port_weekly, factor_weekly,
                spy_weekly, start_str, end_str, holdings
            )
            if block:
                blocks.append(block)
    else:
        common_dates = port_weekly.index.intersection(factor_weekly.index)
        if len(common_dates) > 0:
            last_week = common_dates[-1]
            label = f"Week ending {last_week.strftime('%b %d, %Y')} \u00b7 reconstructed holdings"
            block = _render_attribution_block(
                last_week, holdings, label, rets, port_weekly, factor_weekly,
                spy_weekly, start_str, end_str, holdings
            )
            if block:
                blocks.append(block)

    return blocks


def _render_attribution_block(friday, holdings_for_week, label, rets, port_weekly,
                              factor_weekly, spy_weekly, start_str, end_str, fallback_holdings):
    """Build data for one week's Absolute Return + Excess Return tables."""
    if friday not in port_weekly.index or friday not in factor_weekly.index:
        return None

    week_result = pf.compute_factor_betas(rets, holdings_for_week, start_str, end_str)
    week_beta_map = _extract_beta_map(week_result)

    port_ret_week = port_weekly.loc[friday]
    factor_rets_week = {f: factor_weekly.loc[friday, f] for f in FACTOR_NAMES if f in factor_weekly.columns}
    spy_ret_week = spy_weekly.loc[friday] if friday in spy_weekly.index else 0.0

    imputed_total = sum(week_beta_map.get(f, 0) * factor_rets_week.get(f, 0) for f in FACTOR_NAMES)
    alpha_residual = port_ret_week - imputed_total

    abs_rows = [
        {"": "Sub-Fund", "Total Return": f"{port_ret_week * 100:+.3f}%",
         "Market \u03b2": f"{week_beta_map.get('Market', 0):.3f}",
         "Value \u03b2": f"{week_beta_map.get('Value', 0):.3f}",
         "Momentum \u03b2": f"{week_beta_map.get('Momentum', 0):.3f}",
         "Growth \u03b2": f"{week_beta_map.get('Growth', 0):.3f}"},
        {"": "Factor Returns", "Total Return": "",
         "Market \u03b2": f"{factor_rets_week.get('Market', 0) * 100:+.3f}%",
         "Value \u03b2": f"{factor_rets_week.get('Value', 0) * 100:+.3f}%",
         "Momentum \u03b2": f"{factor_rets_week.get('Momentum', 0) * 100:+.3f}%",
         "Growth \u03b2": f"{factor_rets_week.get('Growth', 0) * 100:+.3f}%"},
        {"": "Imputed Return", "Total Return": f"{imputed_total * 100:+.3f}%",
         "Market \u03b2": f"{week_beta_map.get('Market', 0) * factor_rets_week.get('Market', 0) * 100:+.3f}%",
         "Value \u03b2": f"{week_beta_map.get('Value', 0) * factor_rets_week.get('Value', 0) * 100:+.3f}%",
         "Momentum \u03b2": f"{week_beta_map.get('Momentum', 0) * factor_rets_week.get('Momentum', 0) * 100:+.3f}%",
         "Growth \u03b2": f"{week_beta_map.get('Growth', 0) * factor_rets_week.get('Growth', 0) * 100:+.3f}%"},
        {"": "Alpha", "Total Return": f"{alpha_residual * 100:+.3f}%",
         "Market \u03b2": "", "Value \u03b2": "", "Momentum \u03b2": "", "Growth \u03b2": ""},
    ]

    excess_port = port_ret_week - spy_ret_week
    excess_imputed = imputed_total - spy_ret_week
    excess_alpha = excess_port - imputed_total

    excess_rows = [
        {"": "Sub-Fund (Excess)", "Total Return": f"{excess_port * 100:+.3f}%",
         "Market \u03b2": f"{week_beta_map.get('Market', 0):.3f}",
         "Value \u03b2": f"{week_beta_map.get('Value', 0):.3f}",
         "Momentum \u03b2": f"{week_beta_map.get('Momentum', 0):.3f}",
         "Growth \u03b2": f"{week_beta_map.get('Growth', 0):.3f}"},
        {"": "SPY Return", "Total Return": f"{spy_ret_week * 100:+.3f}%",
         "Market \u03b2": "", "Value \u03b2": "", "Momentum \u03b2": "", "Growth \u03b2": ""},
        {"": "Factor Returns", "Total Return": "",
         "Market \u03b2": f"{factor_rets_week.get('Market', 0) * 100:+.3f}%",
         "Value \u03b2": f"{factor_rets_week.get('Value', 0) * 100:+.3f}%",
         "Momentum \u03b2": f"{factor_rets_week.get('Momentum', 0) * 100:+.3f}%",
         "Growth \u03b2": f"{factor_rets_week.get('Growth', 0) * 100:+.3f}%"},
        {"": "Imputed Excess Return", "Total Return": f"{excess_imputed * 100:+.3f}%",
         "Market \u03b2": f"{week_beta_map.get('Market', 0) * factor_rets_week.get('Market', 0) * 100:+.3f}%",
         "Value \u03b2": f"{week_beta_map.get('Value', 0) * factor_rets_week.get('Value', 0) * 100:+.3f}%",
         "Momentum \u03b2": f"{week_beta_map.get('Momentum', 0) * factor_rets_week.get('Momentum', 0) * 100:+.3f}%",
         "Growth \u03b2": f"{week_beta_map.get('Growth', 0) * factor_rets_week.get('Growth', 0) * 100:+.3f}%"},
        {"": "Alpha", "Total Return": f"{excess_alpha * 100:+.3f}%",
         "Market \u03b2": "", "Value \u03b2": "", "Momentum \u03b2": "", "Growth \u03b2": ""},
    ]

    return {"label": label, "abs_rows": abs_rows, "excess_rows": excess_rows}


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
