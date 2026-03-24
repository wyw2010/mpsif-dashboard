"""
Portfolio reconstruction and analytics engine.
Parses Fidelity transaction CSVs (one per sub-fund), reconstructs daily
positions, fetches prices from yfinance, and computes return/risk metrics.
"""

import pandas as pd
import numpy as np
import requests as _requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import os
import json
import logging
import streamlit as st
from zoneinfo import ZoneInfo

load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

IGNORE_SYMBOLS = {"SPAXX"}
SUBFUNDS = ["Systematic", "Opportunistic", "Thematic", "Fixed Income"]
BENCHMARKS = {
    "Systematic": "SPY",
    "Thematic": "SPY",
    "Opportunistic": "IWV",
    "Fixed Income": "AGG",
}
RISK_FREE_RATE = 0.05  # annualised
EST = ZoneInfo("America/New_York")


def _now_est() -> datetime:
    """Current wall-clock time in US Eastern."""
    return datetime.now(EST)


def _today_est() -> pd.Timestamp:
    """Today's date in US Eastern (tz-naive Timestamp)."""
    return pd.Timestamp(_now_est().date())


# ── Config management ──────────────────────────────────────────────────────
CONFIG_PATH = Path("data/subfund_config.json")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ── 1. Parse Fidelity CSV ─────────────────────────────────────────────────
def parse_fidelity_csv(filepath: str):
    """Parse Fidelity CSV. Returns (transactions_df, initial_cash).

    initial_cash is the account cash balance right before the first BUY trade,
    computed from the raw CSV (including SPAXX rows) before any filtering.
    """
    log.info(f"Parsing CSV: {filepath}")
    raw = pd.read_csv(filepath, skiprows=2)
    raw = raw.dropna(subset=["Run Date"])
    raw = raw[raw["Run Date"].str.match(r"\d{2}/\d{2}/\d{4}", na=False)]

    # Compute initial cash from UNFILTERED data
    raw_dates = pd.to_datetime(raw["Run Date"], format="%m/%d/%Y")
    raw_actions = raw["Action"].str.upper()
    raw_cb = pd.to_numeric(
        raw["Cash Balance ($)"].astype(str).str.replace(",", ""), errors="coerce"
    ).fillna(0)
    raw_amt = pd.to_numeric(
        raw["Amount ($)"].astype(str).str.replace(",", ""), errors="coerce"
    ).fillna(0)

    # Find first BUY date
    buy_mask = raw_actions.str.contains("YOU BOUGHT", na=False)
    first_buy_date = raw_dates[buy_mask].min()

    # Get cash balance right before first buy:
    # Among first buy date transactions, find the one with the HIGHEST cash balance
    # (that's the first executed trade of the day). initial_cash = CB + |Amount|.
    first_day_buys = (raw_dates == first_buy_date) & buy_mask
    if first_day_buys.any():
        idx_max_cb = raw_cb[first_day_buys].idxmax()
        initial_cash = raw_cb.loc[idx_max_cb] + abs(raw_amt.loc[idx_max_cb])
    else:
        initial_cash = 0

    # Now build the filtered DataFrame
    df = pd.DataFrame()
    df["Date"] = raw_dates.values
    df["Symbol"] = raw["Symbol"].str.strip().values

    def _action(a: str) -> str:
        a = str(a).upper()
        if "YOU BOUGHT" in a:
            return "BUY"
        if "YOU SOLD" in a:
            return "SELL"
        if "DIVIDEND RECEIVED" in a:
            return "DIVIDEND"
        if "FEE CHARGED" in a:
            return "FEE"
        if "REINVESTMENT" in a:
            return "REINVESTMENT"
        return "OTHER"

    df["ActionType"] = raw["Action"].apply(_action).values
    for src, dst in [
        ("Price ($)", "Price"),
        ("Quantity", "Quantity"),
        ("Amount ($)", "Amount"),
        ("Cash Balance ($)", "CashBalance"),
    ]:
        df[dst] = pd.to_numeric(
            raw[src].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0).values

    df = df[~df["Symbol"].isin(IGNORE_SYMBOLS)]
    df = df.sort_values("Date").reset_index(drop=True)
    return df, initial_cash


# ── 2. Reconstruct positions ──────────────────────────────────────────────
def reconstruct_positions(txns: pd.DataFrame):
    positions: dict[str, float] = {}
    snapshots: list[tuple] = []
    dividends: list[tuple] = []
    cash_flows: list[tuple] = []

    for date, grp in txns.groupby("Date", sort=True):
        for _, r in grp.iterrows():
            sym, action, qty, amt = r["Symbol"], r["ActionType"], r["Quantity"], r["Amount"]

            if action == "BUY":
                positions[sym] = positions.get(sym, 0) + qty
                cash_flows.append((date, amt))

            elif action == "SELL":
                if positions.get(sym, 0) <= 0:
                    continue
                positions[sym] = positions[sym] - abs(qty)
                if positions[sym] < 0.001:
                    positions[sym] = 0
                cash_flows.append((date, amt))

            elif action == "DIVIDEND":
                dividends.append((date, sym, amt))
                cash_flows.append((date, amt))

            elif action == "FEE":
                dividends.append((date, sym, amt))
                cash_flows.append((date, amt))

        snapshots.append((date, {k: v for k, v in positions.items() if v > 0.001}))

    return snapshots, dividends, cash_flows


def build_daily_positions(snapshots, start, end):
    bdays = pd.bdate_range(start, end)
    snap = dict(snapshots)
    cdates = sorted(snap.keys())
    daily, cur, ci = {}, {}, 0
    for d in bdays:
        while ci < len(cdates) and cdates[ci] <= d:
            cur = snap[cdates[ci]].copy()
            ci += 1
        daily[d] = cur.copy()
    return daily


# ── 3. Prices (Alpaca Market Data REST + file cache) ─────────────────────
PRICE_CACHE_PATH = Path("data/price_cache.csv")
_ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"


def _load_price_cache() -> pd.DataFrame:
    if PRICE_CACHE_PATH.exists():
        try:
            df = pd.read_csv(PRICE_CACHE_PATH, index_col=0, parse_dates=True)
            if df.empty:
                return pd.DataFrame()
            return df
        except Exception:
            pass
    return pd.DataFrame()


def _save_price_cache(df: pd.DataFrame):
    PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRICE_CACHE_PATH)


def _fetch_alpaca_batch(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch close prices via Alpaca REST API (no SDK needed)."""
    headers = {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }
    all_bars = []
    # Alpaca REST accepts comma-separated symbols
    params = {
        "symbols": ",".join(tickers),
        "timeframe": "1Day",
        "start": pd.Timestamp(start).strftime("%Y-%m-%dT00:00:00Z"),
        "end": pd.Timestamp(end).strftime("%Y-%m-%dT00:00:00Z"),
        "feed": "iex",
        "limit": 10000,
    }
    next_token = None
    while True:
        if next_token:
            params["page_token"] = next_token
        resp = _requests.get(_ALPACA_DATA_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for sym, bars in data.get("bars", {}).items():
            for bar in bars:
                all_bars.append({"symbol": sym, "timestamp": bar["t"], "close": bar["c"]})
        next_token = data.get("next_page_token")
        if not next_token:
            break

    if not all_bars:
        return pd.DataFrame()
    df = pd.DataFrame(all_bars)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    pivot = df.pivot_table(index="date", columns="symbol", values="close")
    return pivot


_ALPACA_SNAPSHOT_URL = "https://data.alpaca.markets/v2/stocks/snapshots"


def _fetch_live_prices(tickers: list) -> dict:
    """Fetch latest trade price for each ticker via Alpaca snapshots endpoint.
    Returns {ticker: price} dict. Works during and after market hours."""
    if not tickers:
        return {}
    headers = {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }
    params = {"symbols": ",".join(tickers), "feed": "iex"}
    try:
        resp = _requests.get(_ALPACA_SNAPSHOT_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = {}
        for sym, snap in data.items():
            # Use latest trade price (most recent actual trade)
            trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
            price = trade.get("p", 0)
            if price > 0:
                result[sym] = price
        log.info(f"  Live prices fetched for {len(result)}/{len(tickers)} tickers")
        return result
    except Exception as e:
        log.error(f"  Live price fetch failed: {e}")
        return {}


def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch daily close prices. Uses a local CSV cache to minimise API calls.
    Only fetches from Alpaca when cache is missing or stale."""
    if not tickers:
        return pd.DataFrame()

    log.info(f"fetch_prices called for {len(tickers)} tickers, range {start} → {end}")
    today = _today_est()
    now = _now_est()
    # During market hours (9:30am-4pm ET on weekdays), always re-fetch
    market_open = now.weekday() < 5 and 9 <= now.hour < 17

    # Load file-based cache
    cache = _load_price_cache()
    cached_tickers = set(cache.columns) if not cache.empty else set()
    log.info(f"  Price cache: {len(cached_tickers)} tickers, {len(cache)} rows on disk")

    # Determine which tickers need fetching
    tickers_to_fetch = []
    for t in tickers:
        if t not in cached_tickers:
            tickers_to_fetch.append(t)
        else:
            last_date = cache[t].dropna().index.max() if cache[t].notna().any() else None
            if last_date is None:
                tickers_to_fetch.append(t)
            elif market_open and last_date < today:
                # During market hours, re-fetch if cache doesn't have today
                tickers_to_fetch.append(t)
            elif not market_open and last_date < today - pd.Timedelta(days=1):
                # After hours, re-fetch if cache is more than 1 day old
                tickers_to_fetch.append(t)

    if tickers_to_fetch:
        log.info(f"  Fetching {len(tickers_to_fetch)} tickers from Alpaca in one batch …")
        try:
            new_data = _fetch_alpaca_batch(tickers_to_fetch, start, end)
            if not new_data.empty:
                log.info(f"  Downloaded {len(new_data.columns)} tickers, {len(new_data)} rows")
                if cache.empty:
                    cache = new_data
                else:
                    for col in new_data.columns:
                        cache[col] = new_data[col]
                _save_price_cache(cache)
                log.info(f"  Price cache saved ({len(cache.columns)} tickers, {len(cache)} rows)")
            else:
                log.warning(f"  Alpaca returned empty data")
        except Exception as e:
            log.error(f"  Alpaca fetch failed: {e}")
    else:
        log.info(f"  All {len(tickers)} tickers served from cache")

    # Return only requested tickers that exist in cache
    available = [t for t in tickers if t in cache.columns and cache[t].notna().any()]
    if not available:
        log.warning(f"  No price data available for any requested tickers")
        return pd.DataFrame()
    result = cache[available].loc[start:end]

    # ── Live intraday overlay ──
    # During market hours, fetch real-time prices and add/update today's row
    if market_open and available:
        live = _fetch_live_prices(available)
        if live:
            today_row = pd.DataFrame(
                {t: [live.get(t, np.nan)] for t in available},
                index=[today],
            )
            if today in result.index:
                # Update today's row with live prices
                for t, p in live.items():
                    if t in result.columns:
                        result.at[today, t] = p
            else:
                result = pd.concat([result, today_row])
            log.info(f"  Live prices overlaid for {len(live)} tickers at {today.date()}")

    log.info(f"  Returning prices: {len(available)} tickers, {len(result)} rows")
    return result


# ── 4. Portfolio values ───────────────────────────────────────────────────
def _price_at(prices, ticker, d):
    if ticker not in prices.columns:
        return 0
    col = prices[ticker].dropna()
    if d in col.index:
        return float(col.loc[d])
    prior = col[col.index <= d]
    return float(prior.iloc[-1]) if len(prior) else 0


def compute_portfolio_values(daily_pos, prices, dividends, cash_flows, initial_cash):
    """Compute daily Total Value = Equity + Cash (where cash tracks
    all buy/sell/dividend flows starting from initial_cash)."""
    dates = sorted(daily_pos.keys())
    if not dates:
        return pd.DataFrame()

    cf_daily = pd.DataFrame(cash_flows, columns=["Date", "Amount"]).groupby("Date")["Amount"].sum()
    div_daily = (
        pd.DataFrame(dividends, columns=["Date", "Symbol", "Amount"]).groupby("Date")["Amount"].sum()
        if dividends
        else pd.Series(dtype=float)
    )

    recs = []
    cash = initial_cash
    cum_div = 0.0
    for d in dates:
        if d in cf_daily.index:
            cash += cf_daily.loc[d]
        if len(div_daily) and d in div_daily.index:
            cum_div += div_daily.loc[d]
        equity = sum(s * _price_at(prices, t, d) for t, s in daily_pos[d].items())
        recs.append({
            "Date": d,
            "Equity": equity,
            "Cash": cash,
            "Dividends_Cum": cum_div,
            "Total": equity + cash,
        })
    return pd.DataFrame(recs).set_index("Date")


def compute_ticker_values(daily_pos, prices):
    dates = sorted(daily_pos.keys())
    tickers = sorted({t for p in daily_pos.values() for t in p})
    data = {t: [] for t in tickers}
    idx = []
    for d in dates:
        idx.append(d)
        pos = daily_pos[d]
        for t in tickers:
            s = pos.get(t, 0)
            data[t].append(s * _price_at(prices, t, d) if s > 0 else 0)
    return pd.DataFrame(data, index=idx)


# ── 5. Return / risk metrics ─────────────────────────────────────────────
def daily_returns(series: pd.Series) -> pd.Series:
    s = series[series > 0]
    return s.pct_change().dropna()


def cum_return(rets: pd.Series) -> pd.Series:
    return (1 + rets).cumprod() - 1


def total_ret(rets: pd.Series) -> float:
    return float((1 + rets).prod() - 1) if len(rets) else 0.0


def ann_return(rets: pd.Series) -> float:
    n = len(rets)
    if n < 2:
        return 0.0
    return float((1 + rets).prod() ** (252 / n) - 1)


def ann_vol(rets: pd.Series) -> float:
    return float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0


def sharpe(rets: pd.Series, rf=RISK_FREE_RATE) -> float:
    v = ann_vol(rets)
    return (ann_return(rets) - rf) / v if v else 0.0


def sortino(rets: pd.Series, rf=RISK_FREE_RATE) -> float:
    down = rets[rets < 0]
    dv = float(down.std() * np.sqrt(252)) if len(down) > 1 else 0.0
    return (ann_return(rets) - rf) / dv if dv else 0.0


def max_dd(rets: pd.Series) -> float:
    c = (1 + rets).cumprod()
    return float(((c - c.cummax()) / c.cummax()).min()) if len(c) else 0.0


def calmar(rets: pd.Series) -> float:
    m = max_dd(rets)
    return ann_return(rets) / abs(m) if m != 0 else 0.0


def period_returns(rets: pd.Series):
    if rets.empty:
        return {}
    d = rets.index[-1]
    today = _today_est()
    out = {}
    if len(rets) >= 1:
        out["1D"] = float(rets.iloc[-1])
    out["1W"] = total_ret(rets[rets.index > d - pd.Timedelta(days=7)])
    out["1M"] = total_ret(rets[rets.index > d - pd.DateOffset(months=1)])
    # Use EST year for YTD
    ytd_year = today.year
    out["YTD"] = total_ret(rets[rets.index >= pd.Timestamp(ytd_year, 1, 1)])
    out["1Y"] = total_ret(rets[rets.index > d - pd.DateOffset(years=1)])
    return out


# ── Benchmark-relative metrics ───────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def fetch_benchmark_returns(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch daily returns for a benchmark ETF."""
    prices = fetch_prices([ticker], start, end)
    if prices.empty or ticker not in prices.columns:
        return pd.Series(dtype=float)
    return daily_returns(prices[ticker].dropna())


def excess_returns(port_rets: pd.Series, bench_rets: pd.Series) -> pd.Series:
    aligned = pd.concat([port_rets, bench_rets], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0] - aligned.iloc[:, 1]


def information_ratio(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    er = excess_returns(port_rets, bench_rets)
    if len(er) < 2:
        return 0.0
    te = float(er.std() * np.sqrt(252))
    return (float(er.mean() * 252)) / te if te > 0 else 0.0


def beta(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    aligned = pd.concat([port_rets, bench_rets], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 0.0


def alpha_jensen(port_rets: pd.Series, bench_rets: pd.Series, rf=RISK_FREE_RATE) -> float:
    b = beta(port_rets, bench_rets)
    return ann_return(port_rets) - (rf + b * (ann_return(bench_rets) - rf))


# ── Factor exposure ──────────────────────────────────────────────────────
FACTOR_ETFS = {
    "Momentum": "MTUM",
    "Value": "VLUE",
    "Growth": "VUG",
    "Profitability": "QUAL",
    "Defensive": "USMV",
}


def compute_factor_betas(port_rets: pd.Series, start: str, end: str) -> dict:
    """Regress portfolio returns on factor ETF returns (multivariate OLS)."""
    tickers = list(FACTOR_ETFS.values())
    prices = fetch_prices(tickers, start, end)
    if prices.empty:
        log.warning("Factor betas: no price data returned")
        return {name: 0.0 for name in FACTOR_ETFS}

    factor_rets = prices.pct_change().dropna()

    # Normalize both indices to date-only for alignment
    port_clean = port_rets.copy()
    port_clean.index = port_clean.index.normalize()
    factor_rets.index = factor_rets.index.normalize()

    aligned = pd.concat([port_clean.rename("port")] + [
        factor_rets[t].rename(name) for name, t in FACTOR_ETFS.items()
        if t in factor_rets.columns
    ], axis=1).dropna()

    log.info(f"Factor betas: {len(port_clean)} port days, {len(factor_rets)} factor days, {len(aligned)} aligned days")

    if len(aligned) < 10:
        log.warning(f"Factor betas: only {len(aligned)} aligned days, need at least 10")
        return {name: 0.0 for name in FACTOR_ETFS}

    y = aligned["port"].values
    factor_names = [c for c in aligned.columns if c != "port"]
    X = aligned[factor_names].values
    X = np.column_stack([np.ones(len(X)), X])

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    result = {name: round(float(coeffs[i + 1]), 3) for i, name in enumerate(factor_names)}
    log.info(f"Factor betas: {result}")
    return result


# ── 6. Average cost basis ─────────────────────────────────────────────────
def compute_avg_costs(txns: pd.DataFrame) -> dict:
    """Compute VWAP cost basis per ticker from buy transactions."""
    costs = {}
    for ticker, grp in txns[txns["ActionType"] == "BUY"].groupby("Symbol"):
        total_cost = (grp["Price"] * grp["Quantity"]).sum()
        total_qty = grp["Quantity"].sum()
        costs[ticker] = round(total_cost / total_qty, 2) if total_qty > 0 else 0
    return costs


# ── 7. Attribution (daily chained) ────────────────────────────────────────
def attribution_table(tv: pd.DataFrame, port_val_total: pd.Series,
                      prices: pd.DataFrame, start, end,
                      dividends=None, avg_costs=None):
    """Daily-chained return attribution. Correctly handles mid-period buys/sells."""
    tv_slc = tv.loc[start:end]
    pv_slc = port_val_total.loc[start:end]

    if len(tv_slc) < 2 or len(pv_slc) < 2:
        return pd.DataFrame()

    recs = []
    for ticker in tv_slc.columns:
        vals = tv_slc[ticker]

        if vals.max() <= 0:
            continue

        if ticker not in prices.columns:
            continue
        p = prices[ticker].reindex(tv_slc.index).ffill()

        # Daily chained contribution
        total_contribution = 0.0
        total_pnl = 0.0
        for i in range(1, len(tv_slc)):
            prev_val = vals.iloc[i - 1]
            prev_total = pv_slc.iloc[i - 1]

            if prev_val <= 0 or prev_total <= 0:
                continue

            prev_price = p.iloc[i - 1]
            cur_price = p.iloc[i]

            if pd.isna(prev_price) or pd.isna(cur_price) or prev_price <= 0:
                continue

            weight = prev_val / prev_total
            price_ret = cur_price / prev_price - 1
            total_contribution += weight * price_ret
            total_pnl += prev_val * price_ret

        held_mask = vals > 0
        avg_weight = (vals[held_mask] / pv_slc[held_mask]).mean() if held_mask.any() else 0
        ticker_return = total_contribution / avg_weight if avg_weight > 0 else 0

        div = 0.0
        if dividends:
            div = sum(a for dd, s, a in dividends if s == ticker and start <= dd <= end)

        avg_price = avg_costs.get(ticker, 0) if avg_costs else 0

        recs.append({
            "Ticker": ticker,
            "Weight (%)": round(avg_weight * 100, 2),
            "Return (%)": round(ticker_return * 100, 2),
            "Avg Price ($)": avg_price,
            "Total P&L ($)": round(total_pnl + div, 2),
            "Div Income ($)": round(div, 2),
        })
    return pd.DataFrame(recs).sort_values("Total P&L ($)", ascending=False).reset_index(drop=True)


# ── 7. Current holdings snapshot ──────────────────────────────────────────
def current_holdings(daily_pos, prices):
    dates = sorted(daily_pos.keys())
    if not dates:
        return pd.DataFrame()
    d = dates[-1]
    pos = daily_pos[d]
    recs = []
    total_val = sum(s * _price_at(prices, t, d) for t, s in pos.items())
    for t, s in pos.items():
        p = _price_at(prices, t, d)
        v = s * p
        recs.append({
            "Ticker": t,
            "Shares": round(s, 3),
            "Price ($)": round(p, 2),
            "Value ($)": round(v, 2),
            "Weight (%)": round(v / total_val * 100, 2) if total_val else 0,
        })
    return pd.DataFrame(recs).sort_values("Value ($)", ascending=False).reset_index(drop=True)


# ── 8. Master builder (per sub-fund) ─────────────────────────────────────
@st.cache_data(ttl=900, show_spinner="Loading sub-fund data and fetching prices …")
def build_subfund(csv_path: str):
    log.info(f"═══ build_subfund START: {csv_path} ═══")

    txns, initial_cash = parse_fidelity_csv(csv_path)
    log.info(f"  Parsed {len(txns)} transactions, initial cash=${initial_cash:,.2f}")

    snapshots, dividends, cash_flows = reconstruct_positions(txns)
    log.info(f"  Reconstructed {len(snapshots)} position snapshots, {len(dividends)} dividends")

    first_buy = txns[txns["ActionType"] == "BUY"]["Date"].min()
    end_date = _today_est()
    log.info(f"  Date range: {first_buy.date()} → {end_date.date()}")

    daily_pos = build_daily_positions(snapshots, first_buy, end_date)
    tickers = sorted({t for p in daily_pos.values() for t in p})
    log.info(f"  {len(daily_pos)} business days, {len(tickers)} unique tickers: {tickers}")

    prices = fetch_prices(
        tickers,
        (first_buy - timedelta(days=5)).strftime("%Y-%m-%d"),
        (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    log.info(f"  Computing portfolio values …")
    port_val = compute_portfolio_values(daily_pos, prices, dividends, cash_flows, initial_cash)
    tv = compute_ticker_values(daily_pos, prices)
    rets = daily_returns(port_val["Total"])
    holdings = current_holdings(daily_pos, prices)
    avg_costs = compute_avg_costs(txns)
    log.info(f"  Portfolio: {len(rets)} return days, {len(holdings)} current holdings")
    log.info(f"═══ build_subfund DONE: {csv_path} ═══")

    return {
        "txns": txns,
        "dividends": dividends,
        "daily_positions": daily_pos,
        "prices": prices,
        "portfolio_values": port_val,
        "ticker_values": tv,
        "returns": rets,
        "holdings": holdings,
        "avg_costs": avg_costs,
        "tickers": tickers,
        "first_date": first_buy,
        "end_date": end_date,
        "initial_cash": initial_cash,
    }
