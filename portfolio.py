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


def _is_cusip(sym: str) -> bool:
    """Detect CUSIP identifiers (alphanumeric, contain digits) vs equity tickers."""
    s = str(sym).strip()
    return bool(s) and any(c.isdigit() for c in s) and s not in IGNORE_SYMBOLS
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
_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CONFIG_PATH = _DATA_DIR / "subfund_config.json"


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
    # Try to detect header row — Fidelity CSVs have 2 lines of account info before
    # the header, but re-exported files (e.g., from Google Sheets) may not.
    peek = pd.read_csv(filepath, nrows=0)
    if "Run Date" in peek.columns:
        raw = pd.read_csv(filepath)
    else:
        raw = pd.read_csv(filepath, skiprows=2)
    raw = raw.dropna(subset=["Run Date"])
    raw = raw[raw["Run Date"].str.match(r"\d{1,2}/\d{1,2}/\d{4}", na=False)]

    # Compute initial cash from UNFILTERED data
    raw_dates = pd.to_datetime(raw["Run Date"], format="mixed")
    raw_actions = raw["Action"].str.upper()
    raw_cb = pd.to_numeric(
        raw["Cash Balance ($)"].astype(str).str.replace(",", ""), errors="coerce"
    ).fillna(0)
    raw_amt = pd.to_numeric(
        raw["Amount ($)"].astype(str).str.replace(",", ""), errors="coerce"
    ).fillna(0)

    # Find first BUY date
    buy_mask = raw_actions.str.contains("YOU BOUGHT", na=False)
    sell_mask = raw_actions.str.contains("YOU SOLD", na=False)
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

    # Orphan sells: SELL transactions for positions we never saw a BUY for.
    # These are liquidations from the previous PM and are skipped in
    # reconstruct_positions(). However, the cash proceeds from orphan sells
    # that happen AFTER the first buy represent capital that funds the current
    # PM's trades and must be added to initial_cash.
    bought_syms = set(raw.loc[buy_mask, "Symbol"].str.strip())
    orphan_sell_mask = sell_mask & ~raw["Symbol"].str.strip().isin(bought_syms)
    n_orphan = orphan_sell_mask.sum()
    if n_orphan > 0:
        after_first_buy = orphan_sell_mask & (raw_dates >= first_buy_date)
        orphan_proceeds = pd.to_numeric(
            raw.loc[after_first_buy, "Amount ($)"].astype(str).str.replace(",", ""),
            errors="coerce",
        ).fillna(0).sum()
        if orphan_proceeds > 0:
            initial_cash += orphan_proceeds
            log.info(f"  Added ${orphan_proceeds:,.2f} orphan sell proceeds to initial_cash ({n_orphan} orphan txns total)")
        else:
            log.info(f"  Ignoring {n_orphan} orphan sell transactions (all before first buy)")

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
        if "TRANSFERRED" in a:
            return "TRANSFER"
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
    # Corporate bonds (CUSIPs): rewrite Qty/Price so position value = dollar cost.
    # Set Quantity = abs(Amount) and Price = 1.0, so shares × price = cost in dollars.
    # On sell, the actual cash proceeds are in Amount; the position is reduced by
    # the sell amount. Manual price overrides can adjust the $1 valuation later.
    cusip_mask = df["Symbol"].apply(_is_cusip)
    n_cusip = cusip_mask.sum()
    if n_cusip > 0:
        cusip_syms = df.loc[cusip_mask, "Symbol"].unique()
        log.info(f"  Normalizing {n_cusip} CUSIP transactions ({list(cusip_syms)}) to dollar-unit positions")
        df.loc[cusip_mask, "Quantity"] = df.loc[cusip_mask, "Amount"].abs()
        df.loc[cusip_mask, "Price"] = 1.0
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
                    # Orphan sell (pre-existing position we never saw a BUY for).
                    # Cash is already accounted for via initial_cash adjustment
                    # in parse_fidelity_csv, so skip to avoid double-counting.
                    continue
                positions[sym] = positions[sym] - abs(qty)
                # For CUSIPs, qty is dollar amount which may differ between buy
                # and sell (due to price changes). Close position if remainder
                # is small relative to original position (i.e., the bond is fully sold
                # but buy/sell dollar amounts differ due to price movement).
                if _is_cusip(sym) and 0 < positions[sym] < abs(qty) * 0.5:
                    positions[sym] = 0
                if positions[sym] < 0.001:
                    positions[sym] = 0
                cash_flows.append((date, amt))

            elif action == "DIVIDEND":
                dividends.append((date, sym, amt))
                cash_flows.append((date, amt))

            elif action == "FEE":
                dividends.append((date, sym, amt))
                cash_flows.append((date, amt))

            elif action == "TRANSFER":
                # Cash transfer in/out of the account
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


# ── Bond price overrides (manual entry, stored as JSON) ──────────────────
BOND_PRICE_PATH = _DATA_DIR / "bond_prices.json"


def load_bond_prices() -> dict:
    """Load manually entered bond prices. Returns {cusip: price_per_dollar_unit}.
    A value of 1.0 means hold at cost. A value of 0.95 means the bond is
    trading at 95 cents on the dollar (5% loss from par)."""
    if BOND_PRICE_PATH.exists():
        try:
            with open(BOND_PRICE_PATH) as f:
                data = json.load(f)
            return {k: float(v.get("price_ratio", 1.0)) for k, v in data.items()}
        except Exception:
            pass
    return {}


def compute_bond_accrual_series(bond_info: dict, dates: pd.DatetimeIndex) -> pd.Series:
    """Compute daily accrued interest for a bond as a fraction of cost.
    Returns a Series of accrual amounts (in dollars) indexed by date."""
    face_value = bond_info.get("face_value", 0)
    coupon_rate = bond_info.get("coupon_rate", 0)  # annual rate, e.g. 0.065
    purchase_date_str = bond_info.get("purchase_date")
    if not face_value or not coupon_rate or not purchase_date_str:
        return pd.Series(0.0, index=dates)

    purchase_date = pd.Timestamp(purchase_date_str)
    daily_accrual = face_value * coupon_rate / 365  # simple day count

    # Accrual grows from purchase date, resets on coupon payment dates
    # For simplicity, assume continuous accrual (no reset until coupon pays in CSV)
    days_since = np.maximum((dates - purchase_date).days, 0)
    return pd.Series(daily_accrual * days_since, index=dates)


def load_bond_prices_full() -> dict:
    """Load full bond price data including descriptions."""
    if BOND_PRICE_PATH.exists():
        try:
            with open(BOND_PRICE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_bond_prices(data: dict):
    """Save bond price data. Format: {cusip: {description, face_value, price_per_100, price_ratio}}"""
    BOND_PRICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BOND_PRICE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"  Saved bond prices for {len(data)} CUSIPs to {BOND_PRICE_PATH}")


# ── 3. Prices (Alpaca Market Data REST + file cache) ─────────────────────
PRICE_CACHE_PATH = _DATA_DIR / "price_cache.csv"
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
                    # Merge new data into cache, expanding the date range if needed
                    cache = new_data.combine_first(cache)
                    cache = cache.sort_index()
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
def daily_returns(series: pd.Series, transfers: list = None) -> pd.Series:
    """Compute time-weighted daily returns, adjusting for cash transfers.
    Transfers (in/out) are excluded from return calculations so they don't
    appear as gains/losses."""
    s = series[series > 0]
    if transfers:
        # Build a Series of transfer amounts by date
        tf = pd.Series(dtype=float)
        for date, amt in transfers:
            if date in tf.index:
                tf[date] += amt
            else:
                tf[date] = amt

        # Time-weighted return: on transfer days, adjust the denominator
        # return = end_value / (start_value + transfer) - 1
        rets = pd.Series(dtype=float, index=s.index[1:])
        for i in range(1, len(s)):
            prev_val = s.iloc[i - 1]
            cur_val = s.iloc[i]
            cur_date = s.index[i]
            transfer_amt = tf.get(cur_date, 0)
            adjusted_prev = prev_val + transfer_amt
            if adjusted_prev > 0:
                rets.iloc[i - 1] = cur_val / adjusted_prev - 1
            else:
                rets.iloc[i - 1] = 0.0
    else:
        rets = s.pct_change().dropna()

    # Cap returns that are clearly cash flows, not market moves
    rets[rets > 2.0] = 0.0
    rets[rets < -0.8] = 0.0
    return rets


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


def regression_stats(port_rets: pd.Series, bench_rets: pd.Series, rf=RISK_FREE_RATE) -> dict:
    """Run CAPM regression: R_p - R_f = alpha + beta * (R_m - R_f) + epsilon.
    Returns dict with alpha (ann.), beta, excess return, and idiosyncratic vol."""
    
    end = _today_est().strftime("%Y-%m-%d")
    
    aligned_raw = pd.concat([port_rets, bench_rets], axis=1).dropna()
    if len(aligned_raw) < 30:
        return {"alpha": 0.0, "beta": 0.0, "excess_return": 0.0, "idio_vol": 0.0}

    # Restrict to trailing 6 months window
    end_dt = pd.to_datetime(end)
    start_dt = end_dt - pd.DateOffset(months=6)
    aligned = aligned_raw.loc[start_dt:end_dt].copy()
    
    rf_daily = rf / 252
    y = (aligned.iloc[:, 0] - rf_daily).values.astype(np.float64)  # R_p - R_f
    x = (aligned.iloc[:, 1] - rf_daily).values.astype(np.float64)  # R_m - R_f
    X = np.column_stack([np.ones(len(x)), x])

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily = coeffs[0]
    beta_val = coeffs[1]

    # Residuals = epsilon (idiosyncratic component)
    residuals = y - X @ coeffs
    idio_vol = float(np.std(residuals, ddof=1) * np.sqrt(252))  # annualized

    # Excess return = cumulative (Rp - Rb)
    excess_cum = float(((1 + aligned.iloc[:, 0]).prod() / (1 + aligned.iloc[:, 1]).prod()) - 1)

    return {
        "alpha": float(alpha_daily * 252),       # annualized alpha
        "beta": float(beta_val),
        "excess_return": excess_cum,
        "idio_vol": idio_vol,
    }


# ── Factor exposure ──────────────────────────────────────────────────────
FACTOR_ETFS = {
    "Momentum": "SPMO",
    "Value": "SPYV",
    "Growth": "SPYG",
    "Defensive": "SPLV",
}

# Uses custom factor construction from data.pk (contributed by al7816-cmd)
def compute_factor_betas(port_rets: pd.Series, start: str, end: str) -> dict:
    """Regress portfolio returns on factor returns from data.pk (multivariate OLS, trailing 6 months)."""
    
    # Load factor return data
    factor_data = pd.read_pickle('data.pk')

    if factor_data.empty:
        log.warning("Factor betas: no factor data found")
        return {}

    # Ensure datetime index
    factor_data.index = pd.to_datetime(factor_data.index).normalize()

    # Restrict to trailing 6 months window
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    factor_rets = factor_data.loc[start_dt:end_dt].copy()

    # Clean portfolio returns
    port_clean = port_rets.copy()
    port_clean.index = pd.to_datetime(port_clean.index).normalize()

    # Align data
    aligned = pd.concat(
        [port_clean.rename("port"), factor_rets],
        axis=1
    ).dropna()

    log.info(f"Factor betas: {len(port_clean)} port days, {len(factor_rets)} factor days, {len(aligned)} aligned days")

    if len(aligned) < 10:
        log.warning(f"Factor betas: only {len(aligned)} aligned days, need at least 10")
        return {col: 0.0 for col in factor_rets.columns}

    # Regression — force float64 to avoid object dtype from mixed sources
    y = aligned["port"].values.astype(np.float64)
    factor_names = [c for c in aligned.columns if c != "port"]
    X = aligned[factor_names].values.astype(np.float64)
    X = np.column_stack([np.ones(len(X)), X])  # add intercept

    coeffs, residuals_ss, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    residuals = y - y_hat
    n, k = X.shape

    # R-squared
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors, t-stats, p-values
    from scipy import stats as sp_stats
    mse = ss_res / max(n - k, 1)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = coeffs / se
    p_values = [2 * (1 - sp_stats.t.cdf(abs(t), df=max(n - k, 1))) for t in t_stats]

    betas = {}
    beta_stats = {}
    for i, fname in enumerate(factor_names):
        label = fname.title()
        betas[label] = round(float(coeffs[i + 1]), 3)
        beta_stats[label] = {
            "t_stat": round(float(t_stats[i + 1]), 3),
            "p_value": round(float(p_values[i + 1]), 3),
        }

    alpha_daily = float(coeffs[0])
    idio_vol = float(np.std(residuals, ddof=1) * np.sqrt(252))

    result = {
        "_alpha": round(alpha_daily * 252, 3),
        "_idio_vol": round(idio_vol, 3),
        "_r_squared": round(r_squared, 3),
        "_alpha_t": round(float(t_stats[0]), 3),
        "_alpha_p": round(float(p_values[0]), 3),
        "_stats": beta_stats,
    }
    result.update(betas)

    log.info(f"Factor betas: {result}")
    return result


def compute_etf_factor_betas(port_rets: pd.Series, start: str, end: str) -> dict:
    """Regress portfolio returns on factor ETF returns (multivariate OLS)."""
    tickers = list(FACTOR_ETFS.values())
    prices = fetch_prices(tickers, start, end)
    if prices.empty:
        log.warning("ETF factor betas: no price data returned")
        return {f"{name} ({t})": 0.0 for name, t in FACTOR_ETFS.items()}

    factor_rets = prices.pct_change().dropna()

    port_clean = port_rets.copy()
    port_clean.index = port_clean.index.normalize()
    factor_rets.index = factor_rets.index.normalize()

    label_map = {t: f"{name} ({t})" for name, t in FACTOR_ETFS.items()}
    aligned = pd.concat([port_clean.rename("port")] + [
        factor_rets[t].rename(label_map[t]) for name, t in FACTOR_ETFS.items()
        if t in factor_rets.columns
    ], axis=1).dropna()

    log.info(f"ETF factor betas: {len(port_clean)} port days, {len(factor_rets)} factor days, {len(aligned)} aligned days")

    if len(aligned) < 10:
        log.warning(f"ETF factor betas: only {len(aligned)} aligned days, need at least 10")
        return {f"{name} ({t})": 0.0 for name, t in FACTOR_ETFS.items()}

    y = aligned["port"].values.astype(np.float64)
    factor_names = [c for c in aligned.columns if c != "port"]
    X = aligned[factor_names].values.astype(np.float64)
    X = np.column_stack([np.ones(len(X)), X])

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    residuals = y - y_hat
    n, k = X.shape

    # R-squared
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors, t-stats, p-values
    from scipy import stats as sp_stats
    mse = ss_res / max(n - k, 1)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = coeffs / se
    p_values = [2 * (1 - sp_stats.t.cdf(abs(t), df=max(n - k, 1))) for t in t_stats]

    betas = {}
    beta_stats = {}
    for i, fname in enumerate(factor_names):
        betas[fname] = round(float(coeffs[i + 1]), 3)
        beta_stats[fname] = {
            "t_stat": round(float(t_stats[i + 1]), 3),
            "p_value": round(float(p_values[i + 1]), 3),
        }

    alpha_daily = float(coeffs[0])
    idio_vol = float(np.std(residuals, ddof=1) * np.sqrt(252))

    result = {
        "_alpha": round(alpha_daily * 252, 3),
        "_idio_vol": round(idio_vol, 3),
        "_r_squared": round(r_squared, 3),
        "_alpha_t": round(float(t_stats[0]), 3),
        "_alpha_p": round(float(p_values[0]), 3),
        "_stats": beta_stats,
    }
    result.update(betas)

    log.info(f"ETF factor betas: {result}")
    return result


def load_theme_map(filepath="data/Returns Attribution v2.xlsx") -> dict:
    """Load ticker → theme mapping from the Portfolio Positions tab.
    Returns dict like {'APLD': 'Digital Infra', 'ABNB': 'Experientials', ...}
    """
    df = pd.read_excel(filepath, sheet_name="Portfolio Positions")
    df = df.dropna(subset=["Symbol", "Subtheme"])
    # Filter out non-string subthemes (junk rows like 649.71)
    df = df[df["Subtheme"].apply(lambda x: isinstance(x, str))]
    # Exclude benchmark
    df = df[df["Subtheme"] != "SPY"]
    return dict(zip(df["Symbol"], df["Subtheme"]))


def weekly_theme_attribution(
    port_rets: pd.Series,
    holdings_df: pd.DataFrame,
    theme_map: dict,
    start: str,
    end: str,
    asset_returns_path: str = "asset_returns.pk",
) -> pd.DataFrame:
    """Decompose weekly portfolio returns into theme-level contributions.
    Theme contribution = sum of (portfolio_weight_i × weekly_return_i) for all tickers in theme.
    Sum of all theme contributions = portfolio weekly return.
    """
    asset_daily = pd.read_pickle(asset_returns_path)
    asset_daily.index = pd.to_datetime(asset_daily.index).normalize()

    port_clean = port_rets.copy()
    port_clean.index = pd.to_datetime(port_clean.index).normalize()

    port_weekly = (1 + port_clean).resample("W-FRI").prod() - 1
    port_weekly = port_weekly.dropna()
    port_weekly = port_weekly[port_weekly.index >= pd.Timestamp(start)]

    asset_weekly = (1 + asset_daily).resample("W-FRI").prod() - 1

    weight_map = {}
    if holdings_df is not None and not holdings_df.empty:
        for _, row in holdings_df.iterrows():
            t = row.get("Ticker", row.get("Symbol", ""))
            w = row.get("Weight (%)", row.get("Weight", 0))
            if isinstance(w, (int, float)) and pd.notna(w):
                weight_map[t] = w / 100 if w > 1 else w

    all_themes = sorted(set(theme_map.values()))

    rows = []
    for date in port_weekly.index:
        if date not in asset_weekly.index:
            continue
        row = {"Week Ending": date.strftime("%b %d, %Y")}
        row["Portfolio"] = round(port_weekly.loc[date] * 100, 3)

        for theme in all_themes:
            theme_tickers = [t for t, th in theme_map.items() if th == theme]
            contrib = 0.0
            for t in theme_tickers:
                if t in asset_weekly.columns and t in weight_map:
                    contrib += weight_map[t] * asset_weekly.loc[date, t] * 100
            row[theme] = round(contrib, 3)

        rows.append(row)

    return pd.DataFrame(rows)

"""
def weekly_factor_attribution(port_rets: pd.Series, betas: dict, start: str, end: str) -> pd.DataFrame:
    # Decompose weekly portfolio returns into factor contributions. Uses ETF factor betas × weekly ETF factor returns.
    if port_rets.empty or not betas:
        return pd.DataFrame()

    # Get factor ETF prices
    tickers = list(FACTOR_ETFS.values())
    prices = fetch_prices(tickers, start, end)
    if prices.empty:
        return pd.DataFrame()

    # Daily factor ETF returns
    factor_daily = prices.pct_change().dropna()
    factor_daily.index = factor_daily.index.normalize()

    # Clean portfolio returns
    port_clean = port_rets.copy()
    port_clean.index = port_clean.index.normalize()

    # Resample to weekly (Friday-ending)
    port_weekly = (1 + port_clean).resample("W-FRI").prod() - 1
    port_weekly = port_weekly.dropna()
    port_weekly = port_weekly[port_weekly.index >= pd.Timestamp(start)]

    factor_weekly = {}
    for name, ticker in FACTOR_ETFS.items():
        if ticker in factor_daily.columns:
            fw = (1 + factor_daily[ticker]).resample("W-FRI").prod() - 1
            factor_weekly[name] = fw

    factor_weekly_df = pd.DataFrame(factor_weekly)

    # Build attribution rows
    rows = []
    # Extract betas — handle both "Name (TICKER)" and plain "Name" keys
    beta_map = {}
    for name in FACTOR_ETFS.keys():
        for key, val in betas.items():
            if key.startswith("_"):
                continue
            if name in key:
                beta_map[name] = val
                break

    alpha_weekly = betas.get("_alpha", 0) / 52  # annualized → weekly

    for date in port_weekly.index:
        if date not in factor_weekly_df.index:
            continue
        row = {"Week Ending": date.strftime("%b %d, %Y")}
        row["Portfolio"] = round(port_weekly.loc[date] * 100, 3)

        explained = alpha_weekly * 100
        row["Alpha"] = round(alpha_weekly * 100, 3)

        for name in FACTOR_ETFS.keys():
            if name in factor_weekly_df.columns and name in beta_map:
                contrib = beta_map[name] * factor_weekly_df.loc[date, name] * 100
                row[name] = round(contrib, 3)
                explained += contrib
            else:
                row[name] = 0.0

        row["Residual"] = round(row["Portfolio"] - explained, 3)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Most recent week first
    df = df.iloc[::-1].reset_index(drop=True)
    return df
"""

# VERSION THAT USES CONSTRUCTED FACTORS 
def weekly_factor_attribution(port_rets: pd.Series, betas: dict, start: str, end: str) -> pd.DataFrame:
    # Daily factor returns from pickle
    factor_daily = pd.read_pickle('data.pk')
    factor_daily = factor_daily.rename(columns={'mkt':'Market', 'momentum':'Momentum', 'growth':'Growth', 'value':'Value'})
    factor_daily.index = pd.to_datetime(factor_daily.index).normalize()

    # Clean portfolio returns
    port_clean = port_rets.copy()
    port_clean.index = port_clean.index.normalize()

    # Resample to weekly (Friday-ending)
    port_weekly = (1 + port_clean).resample("W-FRI").prod() - 1
    port_weekly = port_weekly.dropna()
    port_weekly = port_weekly[port_weekly.index >= pd.Timestamp(start)]

    # Resample each factor column to weekly
    factor_weekly_df = (1 + factor_daily).resample("W-FRI").prod() - 1

    # Extract betas — match factor names from data.pk columns
    factor_names = factor_daily.columns.tolist()
    beta_map = {}
    for name in factor_names:
        for key, val in betas.items():
            if key.startswith("_"):
                continue
            if name in key or key in name:
                beta_map[name] = val
                break

    alpha_weekly = betas.get("_alpha", 0) / 52

    # Build attribution rows
    rows = []
    for date in port_weekly.index:
        if date not in factor_weekly_df.index:
            continue
        row = {"Week Ending": date.strftime("%b %d, %Y")}
        row["Portfolio"] = round(port_weekly.loc[date] * 100, 3)
        explained = alpha_weekly * 100
        row["Alpha"] = round(alpha_weekly * 100, 3)
        for name in factor_names:
            if name in factor_weekly_df.columns and name in beta_map:
                contrib = beta_map[name] * factor_weekly_df.loc[date, name] * 100
                row[name] = round(contrib, 3)
                explained += contrib
            else:
                row[name] = 0.0
        row["Residual"] = round(row["Portfolio"] - explained, 3)
        rows.append(row)

    return pd.DataFrame(rows)

# ── 6. Sector mapping ─────────────────────────────────────────────────────
_SECTOR_CACHE_PATH = Path(os.getenv("DATA_DIR", "data")) / "sector_cache.json"


def _load_sector_cache() -> dict:
    try:
        if _SECTOR_CACHE_PATH.exists():
            import json
            return json.loads(_SECTOR_CACHE_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_sector_cache(cache: dict):
    import json
    _SECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SECTOR_CACHE_PATH.write_text(json.dumps(cache))


def get_sectors(tickers: list) -> dict:
    """Return {ticker: sector} mapping. Uses Alpaca assets API + local cache."""
    cache = _load_sector_cache()
    result = {}
    missing = []
    for t in tickers:
        if t in cache and cache[t] != "Other":
            result[t] = cache[t]
        else:
            missing.append(t)

    if missing:
        headers = {
            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
        }
        # Alpaca paper trading API has asset info
        for t in missing:
            try:
                resp = _requests.get(
                    f"https://paper-api.alpaca.markets/v2/assets/{t}",
                    headers=headers, timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Alpaca provides 'attributes' but not sector directly
                    # Use a heuristic based on common ETF/stock mappings
                    result[t] = "Other"
                    cache[t] = "Other"
            except Exception:
                result[t] = "Other"
                cache[t] = "Other"

        # Fallback: well-known sector mapping for common US stocks
        # This covers most S&P 500 and popular tickers
        _KNOWN_SECTORS = {
            # Technology
            "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology", "GOOGL": "Technology",
            "META": "Technology", "NVDA": "Technology", "AMD": "Technology", "INTC": "Technology",
            "AVGO": "Technology", "ORCL": "Technology", "CRM": "Technology", "ADBE": "Technology",
            "CSCO": "Technology", "TXN": "Technology", "QCOM": "Technology", "AMAT": "Technology",
            "MU": "Technology", "LRCX": "Technology", "KLAC": "Technology", "SNPS": "Technology",
            "NOW": "Technology", "PLTR": "Technology", "COIN": "Technology", "MELI": "Technology",
            "SNOW": "Technology", "MDB": "Technology", "TEAM": "Technology", "NET": "Technology",
            "CRWD": "Technology", "FTNT": "Technology", "PANW": "Technology", "ZS": "Technology",
            "FSLR": "Technology", "STX": "Technology", "WDC": "Technology", "GEV": "Technology",
            "GLW": "Technology", "TEL": "Technology", "APH": "Technology", "TER": "Technology",
            "COHR": "Technology", "AAOI": "Technology", "ACMR": "Technology", "LITE": "Technology",
            "ANET": "Technology", "FISV": "Technology", "FI": "Technology",
            # Healthcare
            "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare",
            "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
            "AMGN": "Healthcare", "GILD": "Healthcare", "REGN": "Healthcare", "VRTX": "Healthcare",
            "ISRG": "Healthcare", "HCA": "Healthcare", "MCK": "Healthcare", "INCY": "Healthcare",
            "PODD": "Healthcare", "MEDP": "Healthcare", "DCTH": "Healthcare", "GRCE": "Healthcare",
            # Financials
            "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
            "MS": "Financials", "C": "Financials", "BLK": "Financials", "SCHW": "Financials",
            "AXP": "Financials", "COF": "Financials", "USB": "Financials", "TFC": "Financials",
            "PNC": "Financials", "CFG": "Financials", "KEY": "Financials", "RF": "Financials",
            "STT": "Financials", "IVZ": "Financials", "SYF": "Financials", "PFG": "Financials",
            "ALL": "Financials", "AIG": "Financials", "PRU": "Financials", "AFL": "Financials",
            "CINF": "Financials", "BX": "Financials", "KKR": "Financials", "MCO": "Financials",
            # Consumer Discretionary
            "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
            "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary", "TJX": "Consumer Discretionary",
            "LULU": "Consumer Discretionary", "ONON": "Consumer Discretionary", "GM": "Consumer Discretionary",
            "F": "Consumer Discretionary", "UBER": "Consumer Discretionary", "EXPE": "Consumer Discretionary",
            "CPNG": "Consumer Discretionary", "DKNG": "Consumer Discretionary", "DG": "Consumer Discretionary",
            "DLTR": "Consumer Discretionary", "CHTR": "Consumer Discretionary", "SN": "Consumer Discretionary",
            "BROS": "Consumer Discretionary", "TPR": "Consumer Discretionary", "HAS": "Consumer Discretionary",
            "TTWO": "Consumer Discretionary", "SPOT": "Consumer Discretionary", "RDDT": "Consumer Discretionary",
            "DUOL": "Consumer Discretionary", "LYV": "Consumer Discretionary", "SRAD": "Consumer Discretionary",
            "FLUT": "Consumer Discretionary",
            # Industrials
            "CAT": "Industrials", "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
            "DE": "Industrials", "BA": "Industrials", "LMT": "Industrials", "GE": "Industrials",
            "OMC": "Industrials", "HII": "Industrials", "SOLV": "Industrials", "VTRS": "Industrials",
            "ECL": "Industrials", "PNR": "Industrials", "BMI": "Industrials", "MWA": "Industrials",
            "WTS": "Industrials", "VVX": "Industrials", "FANG": "Industrials",
            # Energy
            "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
            "EOG": "Energy", "PXD": "Energy", "VLO": "Energy", "MPC": "Energy",
            "PSX": "Energy", "EQT": "Energy", "APA": "Energy", "BG": "Energy",
            "MOS": "Energy", "ALB": "Energy", "NEM": "Energy", "AMCR": "Energy",
            "CEG": "Energy", "APLD": "Energy",
            # Communication Services
            "DIS": "Communication", "NFLX": "Communication", "CMCSA": "Communication",
            "T": "Communication", "VZ": "Communication", "TMUS": "Communication",
            "WBD": "Communication",
            # Consumer Staples
            "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
            "WMT": "Consumer Staples", "COST": "Consumer Staples",
            # Utilities
            "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
            "EIX": "Utilities", "ES": "Utilities", "SPG": "Real Estate",
            # Real Estate
            "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
            # ETFs
            "SPY": "ETF", "IWV": "ETF", "AGG": "ETF", "MTUM": "ETF",
            "VLUE": "ETF", "VUG": "ETF", "USMV": "ETF", "SPYG": "ETF",
            "SPHQ": "ETF", "VTHR": "ETF", "JPXN": "ETF",
            # Others from portfolios
            "LOGI": "Technology", "GRMN": "Technology", "AS": "Consumer Discretionary",
            "PBR": "Energy", "SDRL": "Energy", "ACN": "Technology",
            "MMYT": "Consumer Discretionary", "PCOR": "Technology", "NCNO": "Technology",
            "HYNE": "Financials", "GTX": "Industrials", "SERV": "Technology",
            "SNAP": "Communication", "STUB": "Consumer Discretionary", "TPC": "Industrials",
            "ACIC": "Financials", "RUSHA": "Industrials", "RKLB": "Industrials",
            "AIR": "Industrials", "PSYTF": "Energy", "GRBK": "Consumer Discretionary",
            "AXON": "Technology", "CCC": "Technology", "CYH": "Healthcare",
            "LQDA": "Healthcare", "KRMN": "Industrials", "AVTX": "Healthcare",
            "ATRO": "Industrials", "ATI": "Industrials", "POET": "Technology",
            "VLTO": "Industrials", "DRVN": "Consumer Discretionary",
        }
        for t in missing:
            if t in _KNOWN_SECTORS:
                result[t] = _KNOWN_SECTORS[t]
                cache[t] = _KNOWN_SECTORS[t]

        _save_sector_cache(cache)
        log.info(f"Sector lookup: {len(result)} resolved, {sum(1 for v in result.values() if v == 'Other')} unknown")

    return result


# ── 7. Average cost basis ─────────────────────────────────────────────────
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

    # Separate CUSIP tickers from equity tickers
    cusip_tickers = [t for t in tickers if _is_cusip(t)]
    equity_tickers = [t for t in tickers if not _is_cusip(t)]

    # Ensure price history goes back to at least 2025-09-01 for full cache coverage
    price_start = min(first_buy - timedelta(days=5), pd.Timestamp("2025-09-01"))
    prices = fetch_prices(
        equity_tickers,
        price_start.strftime("%Y-%m-%d"),
        (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    # Add CUSIP columns: base price from manual entry (or $1 at cost),
    # plus daily coupon accrual that grows over time.
    if cusip_tickers and not prices.empty:
        bond_prices_raw = load_bond_prices()  # {cusip: price_ratio}
        bond_full = load_bond_prices_full()   # {cusip: {all fields}}
        for ct in cusip_tickers:
            base_price = bond_prices_raw.get(ct, 1.0)
            info = bond_full.get(ct, {})
            accrual = compute_bond_accrual_series(info, prices.index)
            # Find the cost basis (position qty in dollar units) to convert accrual to price ratio
            ct_buys = txns[(txns["Symbol"] == ct) & (txns["ActionType"] == "BUY")]
            cost_basis = ct_buys["Amount"].abs().sum() if not ct_buys.empty else 1.0
            # Price = base mark-to-market ratio + accrual as fraction of cost
            if cost_basis > 0:
                prices[ct] = base_price + accrual / cost_basis
            else:
                prices[ct] = base_price
            log.info(f"  CUSIP {ct}: base={base_price:.4f}, accrual=${accrual.iloc[-1]:,.2f}, effective price={prices[ct].iloc[-1]:.4f}")
    elif cusip_tickers and prices.empty:
        bdays = pd.bdate_range(price_start, end_date)
        prices = pd.DataFrame(index=bdays)
        bond_prices_raw = load_bond_prices()
        bond_full = load_bond_prices_full()
        for ct in cusip_tickers:
            base_price = bond_prices_raw.get(ct, 1.0)
            info = bond_full.get(ct, {})
            accrual = compute_bond_accrual_series(info, prices.index)
            ct_buys = txns[(txns["Symbol"] == ct) & (txns["ActionType"] == "BUY")]
            cost_basis = ct_buys["Amount"].abs().sum() if not ct_buys.empty else 1.0
            if cost_basis > 0:
                prices[ct] = base_price + accrual / cost_basis
            else:
                prices[ct] = base_price

    log.info(f"  Computing portfolio values …")
    port_val = compute_portfolio_values(daily_pos, prices, dividends, cash_flows, initial_cash)
    tv = compute_ticker_values(daily_pos, prices)
    # Extract transfer-only cash flows for time-weighted return adjustment
    transfer_dates = set(txns[txns["ActionType"] == "TRANSFER"]["Date"])
    transfer_flows = [(d, a) for d, a in cash_flows if d in transfer_dates]
    rets = daily_returns(port_val["Total"], transfers=transfer_flows if transfer_flows else None)
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
