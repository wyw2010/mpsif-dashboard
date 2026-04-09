"""
NYU MPSIF Return Attribution Dashboard — FastAPI + Jinja2 + Plotly.js
"""

import os
import io
import json
import base64
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import portfolio as pf
import data_cache

log = logging.getLogger(__name__)

app = FastAPI(title="NYU MPSIF Return Attribution")
templates = Jinja2Templates(directory="templates")

# Serve static assets (favicon, logo)
_assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
if os.path.isdir(_assets_dir):
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

# ── Design constants ──────────────────────────────────────────────────────
NYU_PURPLE = "#57068C"
NYU_PURPLE_LIGHT = "#8900e1"
NYU_PURPLE_BG = "#F5F0FA"
BLACK = "#1A1A1A"
WHITE = "#FFFFFF"
GRAY = "#6B7280"
GREEN = "#10B981"
RED = "#EF4444"

PLOTLY_CFG = {
    "displayModeBar": "hover",
    "toImageButtonOptions": {"format": "png", "scale": 4},
}
_PLOTLY_FONT = dict(family="Helvetica Neue, Helvetica, Inter, Arial, sans-serif", size=13, color=GRAY)
_PLOTLY_AXIS_FONT = dict(family="Helvetica Neue, Helvetica, Inter, Arial, sans-serif", size=12, color=GRAY)
_PLOTLY_TITLE_FONT = dict(family="Helvetica Neue, Helvetica, Inter, Arial, sans-serif", size=12, color=GRAY)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

SUBFUND_FILE_MAP = {
    "Systematic": "systematic.csv",
    "Opportunistic": "opportunistic.csv",
    "Thematic": "thematic.csv",
    "Fixed Income": "fixed_income.csv",
}

FUND_COLORS = {
    "Total Fund": BLACK,
    "Systematic": "#2563EB",
    "Opportunistic": "#DC2626",
    "Thematic": "#059669",
    "Fixed Income": "#7C3AED",
    "Benchmark": "#F59E0B",
}

SECTOR_COLORS = {
    "Technology": "#2563EB", "Healthcare": "#059669",
    "Financial Services": "#D97706", "Consumer Cyclical": "#DC2626",
    "Industrials": "#7C3AED", "Energy": "#0EA5E9",
    "Communication Services": "#EC4899", "Consumer Defensive": "#10B981",
    "Basic Materials": "#F59E0B", "Real Estate": "#6366F1",
    "Utilities": "#14B8A6", "Other": "#94A3B8",
}

# ── Jinja2 filters ────────────────────────────────────────────────────────
def fmt_pct(v):
    try:
        return f"{float(v):+.3f}%"
    except (ValueError, TypeError):
        return "—"

def fmt_dollar(v):
    try:
        v = float(v)
    except (ValueError, TypeError):
        return "—"
    if abs(v) >= 1e6:
        return f"${v/1e6:,.3f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:,.3f}K"
    return f"${v:,.3f}"

def color_class(v):
    try:
        return "metric-positive" if float(v) >= 0 else "metric-negative"
    except (ValueError, TypeError):
        return ""

templates.env.filters["fmt_pct"] = fmt_pct
templates.env.filters["fmt_dollar"] = fmt_dollar
templates.env.filters["color_class"] = color_class
templates.env.filters["tojson_safe"] = lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v

# ── Logo base64 ───────────────────────────────────────────────────────────
_logo_path = Path("assets/nyu_stern_logo.png")
LOGO_B64 = ""
if _logo_path.exists():
    LOGO_B64 = base64.b64encode(_logo_path.read_bytes()).decode()


# ── Chart builders ────────────────────────────────────────────────────────
def make_return_chart(rets, height=380, bench_rets=None, bench_label=""):
    if rets.empty:
        return go.Figure()
    cum = pf.cum_return(rets) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[t.isoformat() for t in cum.index], y=cum.values.tolist(),
        mode="lines", name="Portfolio",
        line=dict(color=NYU_PURPLE, width=2.5),
        fill="tozeroy", fillcolor="rgba(87, 6, 140, 0.07)",
        yhoverformat="+.3f",
        hovertemplate="Portfolio: %{y:.3f}%<extra></extra>",
    ))
    if bench_rets is not None and not bench_rets.empty:
        bench_aligned = bench_rets.reindex(rets.index).dropna()
        if not bench_aligned.empty:
            bench_cum = pf.cum_return(bench_aligned) * 100
            fig.add_trace(go.Scatter(
                x=[t.isoformat() for t in bench_cum.index], y=bench_cum.values.tolist(),
                mode="lines", name=bench_label,
                line=dict(color="#F59E0B", width=2, dash="dash"),
                yhoverformat="+.3f",
                hovertemplate=f"{bench_label}: %{{y:.3f}}%<extra></extra>",
            ))
    show_legend = bench_rets is not None and not bench_rets.empty
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False, showline=False, tickfont=_PLOTLY_AXIS_FONT),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Cumulative Return (%)",
                   tickformat="+.3f", zeroline=True, zerolinecolor="#E5E7EB",
                   tickfont=_PLOTLY_AXIS_FONT, title_font=_PLOTLY_TITLE_FONT),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=_PLOTLY_FONT,
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)),
    )
    return fig


def make_drawdown_chart(rets, height=200):
    if rets.empty:
        return go.Figure()
    cum = (1 + rets).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[t.isoformat() for t in dd.index], y=dd.values.tolist(),
        mode="lines",
        line=dict(color=RED, width=1.5),
        fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.1)",
        hovertemplate="%{x|%b %d}<br>Drawdown: %{y:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, showline=False, tickfont=_PLOTLY_AXIS_FONT),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Drawdown (%)", tickformat=".3f",
                   tickfont=_PLOTLY_AXIS_FONT, title_font=_PLOTLY_TITLE_FONT),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=_PLOTLY_FONT,
    )
    return fig


def make_multi_fund_chart(fund_data, visible, height=400):
    fig = go.Figure()
    for name, rets in fund_data.items():
        if rets is None or rets.empty:
            continue
        cum = pf.cum_return(rets) * 100
        line_width = 3.5 if name == "Total Fund" else 1.8
        line_dash = "dash" if name == "Benchmark" else None
        fig.add_trace(go.Scatter(
            x=[t.isoformat() for t in cum.index], y=cum.values.tolist(),
            mode="lines", name=name,
            line=dict(color=FUND_COLORS.get(name, GRAY), width=line_width, dash=line_dash),
            visible=True if name in visible else "legendonly",
            yhoverformat="+.3f",
            hovertemplate=f"{name}: %{{y:.3f}}%<extra></extra>",
        ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, showline=False, tickfont=_PLOTLY_AXIS_FONT),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Cumulative Return (%)",
                   tickformat="+.3f", zeroline=True, zerolinecolor="#E5E7EB",
                   tickfont=_PLOTLY_AXIS_FONT, title_font=_PLOTLY_TITLE_FONT),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=_PLOTLY_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12)),
        hovermode="x unified",
    )
    return fig


def make_holdings_pie(holdings_df, height=400, max_slices=10):
    if holdings_df.empty:
        return go.Figure()
    df = holdings_df.copy().sort_values("Weight (%)", ascending=False)
    # Keep top N slices, group rest into "Other"
    if len(df) > max_slices:
        top = df.iloc[:max_slices]
        rest = df.iloc[max_slices:]
        other = pd.DataFrame([{
            "Ticker": f"Other ({len(rest)})",
            "Weight (%)": rest["Weight (%)"].sum(),
            "Value ($)": rest["Value ($)"].sum() if "Value ($)" in rest.columns else 0,
        }])
        df = pd.concat([top, other], ignore_index=True)
    fig = go.Figure(go.Pie(
        labels=df["Ticker"].tolist(), values=df["Weight (%)"].tolist(), hole=0.45,
        textinfo="label+percent", textfont_size=11,
        textposition="inside",
        insidetextorientation="horizontal",
        marker=dict(colors=[
            NYU_PURPLE, "#8900e1", "#2563EB", "#3B82F6", "#60A5FA",
            "#D97706", "#F59E0B", "#059669", "#10B981", "#34D399",
            "#EF4444",
        ]),
        hovertemplate="%{label}<br>Weight: %{value:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor=WHITE, font=_PLOTLY_FONT,
        showlegend=False,
    )
    return fig


def make_sector_pie(sector_agg, height=350):
    sec_colors = [SECTOR_COLORS.get(s, "#94A3B8") for s in sector_agg["Sector"]]
    fig = go.Figure(go.Pie(
        labels=sector_agg["Sector"].tolist(), values=sector_agg["Weight (%)"].tolist(), hole=0.45,
        textinfo="label+percent", textfont_size=11,
        marker=dict(colors=sec_colors),
        hovertemplate="%{label}<br>Weight: %{value:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=WHITE,
        font=dict(family="'Helvetica Neue', Helvetica, Arial, sans-serif", weight=300),
        showlegend=False,
    )
    return fig


def make_theme_attribution_bar(attribution_df, height=400):
    if attribution_df.empty:
        return go.Figure()
    last_row = attribution_df.iloc[-1]
    skip_cols = {"Week Ending", "Portfolio"}
    data = []
    for col in attribution_df.columns:
        if col in skip_cols:
            continue
        val = last_row[col]
        if isinstance(val, str):
            val = float(val.replace("%", "").replace("+", ""))
        data.append({"Theme": col, "Contribution": val})
    if not data:
        return go.Figure()
    df = pd.DataFrame(data).sort_values("Contribution")
    colors = [NYU_PURPLE if v >= 0 else "#EF4444" for v in df["Contribution"]]
    fig = go.Figure(go.Bar(
        x=df["Contribution"].tolist(), y=df["Theme"].tolist(),
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="%{y}<br>Contribution: %{x:+.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=20, t=10, b=0),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE, font=_PLOTLY_FONT,
        xaxis=dict(title="Contribution (%)", zeroline=True, zerolinecolor="#CBD5E1"),
        yaxis=dict(title=""),
    )
    return fig


def make_correlation_heatmap(subfund_data, height=300):
    corr_rets = {}
    for sname, sd in subfund_data.items():
        corr_rets[sname] = sd["returns"]
    corr_df = pd.DataFrame(corr_rets).dropna()
    if len(corr_df) < 5:
        return None
    corr_matrix = corr_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values.tolist(),
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=[[0, "#DC2626"], [0.5, WHITE], [1, NYU_PURPLE]],
        zmin=-1, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in corr_matrix.values],
        texttemplate="%{text}",
        textfont=dict(size=14),
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        font=dict(family="'Helvetica Neue', Helvetica, Arial, sans-serif", weight=300),
        xaxis=dict(side="bottom"),
    )
    return fig


def fig_to_json(fig):
    """Convert a Plotly figure to a JSON string for client-side rendering."""
    return fig.to_json()



# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/")
async def overview(request: Request):
    cache = data_cache.get_cache()
    subfund_data = cache.get("subfund_data", {})
    benchmark_rets = cache.get("benchmark_rets", {})
    blended_bench_rets = cache.get("blended_bench_rets", pd.Series(dtype=float))
    combined_rets = cache.get("combined_rets", pd.Series(dtype=float))

    ctx = {
        "active_page": "overview",
        "subfunds": pf.SUBFUNDS,
        "logo_b64": LOGO_B64,
        "has_data": bool(subfund_data),
        "plotly_cfg": json.dumps(PLOTLY_CFG),
    }

    if not subfund_data or combined_rets.empty:
        return templates.TemplateResponse(request, "overview.html", ctx)

    # Period returns
    period_rets = pf.period_returns(combined_rets)
    ctx["period_rets"] = {k: v * 100 for k, v in period_rets.items()}

    # Key metrics
    total_aum = sum(
        d["portfolio_values"]["Total"].iloc[-1]
        for d in subfund_data.values()
        if len(d["portfolio_values"])
    )
    ctx["total_aum"] = total_aum
    ctx["sharpe"] = pf.sharpe(combined_rets)
    ctx["max_dd"] = pf.max_dd(combined_rets) * 100
    ctx["ann_vol"] = pf.ann_vol(combined_rets) * 100
    ctx["risk_free_rate"] = pf.RISK_FREE_RATE

    # Blended benchmark metrics
    if not blended_bench_rets.empty:
        reg = pf.regression_stats(combined_rets, blended_bench_rets)
        ctx["bench_reg"] = {k: v * 100 if k in ("excess_return", "alpha", "idio_vol") else v for k, v in reg.items()}
    else:
        ctx["bench_reg"] = None

    # Multi-fund chart
    fund_rets = {"Total Fund": combined_rets}
    for name, d in subfund_data.items():
        fund_rets[name] = d["returns"]
    if not blended_bench_rets.empty:
        fund_rets["Benchmark"] = blended_bench_rets
    fund_names = list(fund_rets.keys())
    ctx["fund_names"] = fund_names
    ctx["multi_fund_chart"] = fig_to_json(make_multi_fund_chart(fund_rets, fund_names))

    # Drawdown chart
    ctx["drawdown_chart"] = fig_to_json(make_drawdown_chart(combined_rets))

    # Summary table
    summary_rows = []
    for name, d in subfund_data.items():
        r = d["returns"]
        pr = pf.period_returns(r)
        aum = d["portfolio_values"]["Total"].iloc[-1] if len(d["portfolio_values"]) else 0
        bench_r = benchmark_rets.get(name, pd.Series(dtype=float))
        excess_ytd = 0
        if not bench_r.empty and not r.empty:
            ytd_r = r[r.index >= pd.Timestamp(r.index[-1].year, 1, 1)]
            ytd_b = bench_r[bench_r.index >= pd.Timestamp(r.index[-1].year, 1, 1)]
            excess_ytd = pf.total_ret(pf.excess_returns(ytd_r, ytd_b))
        summary_rows.append({
            "Sub-Fund": name,
            "Benchmark": pf.BENCHMARKS.get(name, ""),
            "AUM": fmt_dollar(aum),
            "1D": fmt_pct(pr.get("1D", 0) * 100),
            "1W": fmt_pct(pr.get("1W", 0) * 100),
            "1M": fmt_pct(pr.get("1M", 0) * 100),
            "YTD": fmt_pct(pr.get("YTD", 0) * 100),
            "Excess (YTD)": fmt_pct(excess_ytd * 100),
            "Sharpe": f"{pf.sharpe(r):.3f}",
            "Max DD": fmt_pct(pf.max_dd(r) * 100),
        })
    ctx["summary_rows"] = summary_rows

    # Correlation heatmap
    if len(subfund_data) >= 2:
        corr_fig = make_correlation_heatmap(subfund_data)
        ctx["correlation_chart"] = fig_to_json(corr_fig) if corr_fig else None
    else:
        ctx["correlation_chart"] = None

    return templates.TemplateResponse(request, "overview.html", ctx)


@app.get("/fund/{fund_slug}")
async def subfund_page(request: Request, fund_slug: str):
    # Convert slug to display name
    slug_map = {n.lower().replace(" ", "-"): n for n in pf.SUBFUNDS}
    name = slug_map.get(fund_slug)
    if not name:
        return RedirectResponse("/")

    cache = data_cache.get_cache()
    subfund_data = cache.get("subfund_data", {})
    benchmark_rets = cache.get("benchmark_rets", {})

    ctx = {
        "active_page": fund_slug,
        "subfunds": pf.SUBFUNDS,
        "logo_b64": LOGO_B64,
        "fund_name": name,
        "fund_slug": fund_slug,
        "has_data": name in subfund_data,
        "plotly_cfg": json.dumps(PLOTLY_CFG),
    }

    if name not in subfund_data:
        return templates.TemplateResponse(request, "subfund.html", ctx)

    d = subfund_data[name]
    rets = d["returns"]
    holdings = d["holdings"]
    dividends = d["dividends"]
    port_val = d["portfolio_values"]
    bench_r = benchmark_rets.get(name, pd.Series(dtype=float))
    bench_ticker = pf.BENCHMARKS.get(name, "")

    if rets.empty:
        ctx["has_data"] = False
        return templates.TemplateResponse(request, "subfund.html", ctx)

    # Period returns
    pr = pf.period_returns(rets)
    ctx["period_rets"] = {k: v * 100 for k, v in pr.items()}

    # Key metrics
    aum = port_val["Total"].iloc[-1]
    ctx["aum"] = aum
    ctx["sharpe"] = pf.sharpe(rets)
    ctx["max_dd"] = pf.max_dd(rets) * 100
    ctx["ann_vol"] = pf.ann_vol(rets) * 100
    ctx["risk_free_rate"] = pf.RISK_FREE_RATE
    ctx["bench_ticker"] = bench_ticker

    # Benchmark metrics
    if not bench_r.empty:
        reg = pf.regression_stats(rets, bench_r)
        ctx["bench_reg"] = {k: v * 100 if k in ("excess_return", "alpha", "idio_vol") else v for k, v in reg.items()}
    else:
        ctx["bench_reg"] = None

    # Cumulative return chart — build all period variants
    # Full chart with benchmark
    ctx["return_chart_all"] = fig_to_json(make_return_chart(rets, bench_rets=bench_r, bench_label=bench_ticker))

    # Period-specific charts
    for period_key, period_rets_slice in _get_period_slices(rets).items():
        chart_bench = bench_r
        if not bench_r.empty and not period_rets_slice.empty:
            chart_bench = bench_r[bench_r.index.isin(period_rets_slice.index)]
        ctx[f"return_chart_{period_key}"] = fig_to_json(
            make_return_chart(period_rets_slice, bench_rets=chart_bench, bench_label=bench_ticker)
        )

    # Chart date range for custom period
    ctx["chart_start_date"] = rets.index[0].strftime("%Y-%m-%d")
    ctx["chart_end_date"] = rets.index[-1].strftime("%Y-%m-%d")

    # Drawdown chart
    ctx["drawdown_chart"] = fig_to_json(make_drawdown_chart(rets, height=180))

    # Holdings
    if not holdings.empty:
        avg_costs = d["avg_costs"]
        display_h = holdings.copy()
        display_h["Avg Price ($)"] = display_h["Ticker"].map(avg_costs).fillna(0)
        display_h["P&L ($)"] = (display_h["Price ($)"] - display_h["Avg Price ($)"]) * display_h["Shares"]

        raw_cols = ["Ticker", "Shares", "Weight (%)", "Avg Price ($)", "Price ($)", "P&L ($)"]
        raw_tbl = display_h[raw_cols].copy().sort_values("Weight (%)", ascending=False)

        # Build table data for template
        holdings_table = []
        for _, row in raw_tbl.iterrows():
            holdings_table.append({
                "Ticker": row["Ticker"],
                "Shares": f"{row['Shares']:.3f}",
                "Weight (%)": f"{row['Weight (%)']:.3f}%",
                "Avg Price ($)": f"${row['Avg Price ($)']:,.3f}",
                "Price ($)": f"${row['Price ($)']:,.3f}",
                "P&L ($)": f"${row['P&L ($)']:,.3f}",
                "_weight_raw": row["Weight (%)"],
                "_pnl_raw": row["P&L ($)"],
                "_shares_raw": row["Shares"],
                "_avg_raw": row["Avg Price ($)"],
                "_price_raw": row["Price ($)"],
            })
        ctx["holdings_table"] = holdings_table
        ctx["holdings_pie"] = fig_to_json(make_holdings_pie(holdings))
    else:
        ctx["holdings_table"] = []
        ctx["holdings_pie"] = None

    # Read pre-computed extras from background cache
    extras = cache.get("fund_extras", {}).get(name, {})

    # Sector exposure (pre-computed sectors dict)
    if not holdings.empty:
        sectors = extras.get("sectors", {})
        sector_df = holdings.copy()
        sector_df["Sector"] = sector_df["Ticker"].map(sectors).fillna("Other")
        sector_agg = sector_df.groupby("Sector")["Weight (%)"].sum().reset_index()
        sector_agg = sector_agg.sort_values("Weight (%)", ascending=False)
        sec_counts = sector_df.groupby("Sector")["Ticker"].count().reset_index()
        sec_counts.columns = ["Sector", "Holdings"]
        sector_display = sector_agg.merge(sec_counts, on="Sector")
        ctx["sector_table"] = sector_display.to_dict("records")
        ctx["sector_pie"] = fig_to_json(make_sector_pie(sector_agg))
    else:
        ctx["sector_table"] = []
        ctx["sector_pie"] = None

    # Factor exposure (pre-computed)
    ff_result = dict(extras.get("ff_betas", {}))
    ctx["ff_factors"] = _format_factor_result(ff_result)
    etf_result = dict(extras.get("etf_betas", {}))
    ctx["etf_factors"] = _format_factor_result(etf_result)

    if name == "Systematic":
        ctx["fund_regression"] = _format_factor_result(dict(extras.get("fund_regression", {})))
        ctx["portfolio_regression"] = _format_factor_result(dict(extras.get("portfolio_regression", {})))
        ctx["portfolio_regression_ex_mom"] = _format_factor_result(dict(extras.get("portfolio_regression_ex_mom", {})))
    else:
        ctx["fund_regression"] = None
        ctx["portfolio_regression"] = None
        ctx["portfolio_regression_ex_mom"] = None

    # Weekly Return Attribution (pre-computed)
    ctx["weekly_attribution_blocks"] = extras.get("weekly_attribution_blocks", [])

    # Weekly Factor Attribution (pre-computed)
    ctx["weekly_factor_attr"] = extras.get("weekly_factor_attr", [])
    ctx["weekly_factor_cols"] = extras.get("weekly_factor_cols", [])

    # Weekly Theme Attribution (pre-computed, Thematic only)
    if name == "Thematic":
        weekly_theme_raw = extras.get("weekly_theme_raw")
        if weekly_theme_raw is not None and not weekly_theme_raw.empty:
            ctx["theme_bar_chart"] = fig_to_json(make_theme_attribution_bar(weekly_theme_raw))
        else:
            ctx["theme_bar_chart"] = None
        ctx["weekly_theme_attr"] = extras.get("weekly_theme_attr", [])
        ctx["weekly_theme_cols"] = extras.get("weekly_theme_cols", [])
    else:
        ctx["theme_bar_chart"] = None
        ctx["weekly_theme_attr"] = []
        ctx["weekly_theme_cols"] = []

    # Dividends
    if dividends:
        div_df = pd.DataFrame(dividends, columns=["Date", "Ticker", "Amount ($)"])
        div_df = div_df.sort_values("Date", ascending=False)
        total_div = div_df["Amount ($)"].sum()
        div_table = []
        for _, row in div_df.iterrows():
            div_table.append({
                "Date": row["Date"].strftime("%Y-%m-%d"),
                "Ticker": row["Ticker"],
                "Amount ($)": f"${row['Amount ($)']:,.3f}",
                "_amount_raw": row["Amount ($)"],
            })
        ctx["dividends"] = div_table
        ctx["total_dividends"] = total_div
    else:
        ctx["dividends"] = []
        ctx["total_dividends"] = 0

    # Bond prices (Fixed Income only)
    if name == "Fixed Income":
        cusip_holdings = [t for t in (holdings["Ticker"].tolist() if not holdings.empty else []) if pf._is_cusip(t)]
        if cusip_holdings:
            existing = pf.load_bond_prices_full()
            bond_entries = []
            for cusip in cusip_holdings:
                info = existing.get(cusip, {})
                bond_entries.append({
                    "cusip": cusip,
                    "description": info.get("description", cusip),
                    "current_price": info.get("current_price_per_100", 100.0),
                    "purchase_price": info.get("purchase_price_per_100", 100.0),
                    "face_value": info.get("face_value", 0.0),
                    "coupon_rate": info.get("coupon_rate", 0.0) * 100,
                    "purchase_date": info.get("purchase_date", ""),
                })
            ctx["bond_entries"] = bond_entries
        else:
            ctx["bond_entries"] = []
    else:
        ctx["bond_entries"] = []

    return templates.TemplateResponse(request, "subfund.html", ctx)


def _get_period_slices(rets):
    """Return dict of period-key → return slice."""
    slices = {}
    slices["1w"] = rets[rets.index > rets.index[-1] - pd.Timedelta(days=7)]
    slices["1m"] = rets[rets.index > rets.index[-1] - pd.DateOffset(months=1)]
    slices["ytd"] = rets[rets.index >= pd.Timestamp(rets.index[-1].year, 1, 1)]
    slices["all"] = rets
    return slices


def _format_factor_result(result):
    """Extract factor betas, alpha, R², idio vol from a compute_factor_betas result."""
    if not result:
        return None
    out = {}
    out["alpha"] = result.pop("_alpha", 0.0)
    out["idio_vol"] = result.pop("_idio_vol", 0.0)
    out["r_squared"] = result.pop("_r_squared", 0.0)
    result.pop("_alpha_t", None)
    result.pop("_alpha_p", None)
    result.pop("_stats", None)
    out["factors"] = {k: v for k, v in result.items() if not k.startswith("_")}
    return out


# ── Upload routes ─────────────────────────────────────────────────────────

@app.get("/upload")
async def upload_page(request: Request, msg: str = "", error: str = ""):
    cache = data_cache.get_cache()

    # List existing snapshots
    all_snapshots = {}
    for fund_name in SUBFUND_FILE_MAP.keys():
        snaps = pf.list_holdings_snapshots(fund_name)
        if snaps:
            snap_list = []
            for friday, path in snaps:
                try:
                    n_tickers = len(pf.load_holdings_snapshot(fund_name, friday))
                except Exception:
                    n_tickers = -1
                snap_list.append({
                    "friday": friday.strftime("%b %d, %Y"),
                    "friday_raw": friday.strftime("%Y-%m-%d"),
                    "n_tickers": n_tickers,
                })
            all_snapshots[fund_name] = snap_list

    # Default friday
    today = pd.Timestamp.now().normalize()
    days_since_friday = (today.weekday() - 4) % 7
    default_friday = (today - pd.Timedelta(days=days_since_friday)).strftime("%Y-%m-%d")

    ctx = {
        "active_page": "upload",
        "subfunds": pf.SUBFUNDS,
        "logo_b64": LOGO_B64,
        "subfund_names": list(SUBFUND_FILE_MAP.keys()),
        "all_snapshots": all_snapshots,
        "default_friday": default_friday,
        "msg": msg,
        "error": error,
        "plotly_cfg": json.dumps(PLOTLY_CFG),
    }
    return templates.TemplateResponse(request, "upload.html", ctx)


@app.post("/upload/transactions")
async def upload_transactions(subfund: str = Form(...), csv_file: UploadFile = File(...)):
    if subfund not in SUBFUND_FILE_MAP:
        return RedirectResponse("/upload?error=Invalid+sub-fund", status_code=303)

    target_path = DATA_DIR / SUBFUND_FILE_MAP[subfund]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    content = await csv_file.read()
    target_path.write_bytes(content)

    # Trigger background refresh
    import threading
    threading.Thread(target=data_cache.refresh, daemon=True).start()

    return RedirectResponse(f"/upload?msg={subfund}+updated+successfully", status_code=303)


@app.post("/upload/snapshot")
async def upload_snapshot(subfund: str = Form(...), friday_date: str = Form(...), csv_file: UploadFile = File(...)):
    try:
        friday_ts = pd.Timestamp(friday_date)
    except Exception:
        return RedirectResponse("/upload?error=Invalid+date+format", status_code=303)

    if friday_ts.weekday() != 4:
        return RedirectResponse(f"/upload?error={friday_date}+is+not+a+Friday", status_code=303)

    content = await csv_file.read()

    # Validate parse
    try:
        parsed = pf.parse_fidelity_positions_csv(io.BytesIO(content))
        if parsed.empty:
            return RedirectResponse("/upload?error=Could+not+parse+any+positions", status_code=303)
    except Exception as e:
        return RedirectResponse(f"/upload?error=Parse+error:+{str(e)}", status_code=303)

    pf.save_holdings_snapshot(subfund, friday_ts, content)
    return RedirectResponse(f"/upload?msg=Saved+{subfund}+snapshot+for+{friday_date}+({len(parsed)}+tickers)", status_code=303)


@app.post("/upload/snapshot/delete")
async def delete_snapshot(subfund: str = Form(...), friday_date: str = Form(...)):
    try:
        friday_ts = pd.Timestamp(friday_date)
        pf.delete_holdings_snapshot(subfund, friday_ts)
    except Exception:
        pass
    return RedirectResponse("/upload?msg=Snapshot+deleted", status_code=303)


@app.post("/bond-prices")
async def update_bond_prices(request: Request):
    form = await request.form()
    existing = pf.load_bond_prices_full()
    bond_data = dict(existing)

    for key in form.keys():
        if key.startswith("price_"):
            cusip = key.replace("price_", "")
            info = existing.get(cusip, {})
            bond_data[cusip] = {
                "description": info.get("description", cusip),
                "purchase_price_per_100": float(form.get(f"purchase_{cusip}", 100.0)),
                "current_price_per_100": float(form.get(f"price_{cusip}", 100.0)),
                "price_ratio": float(form.get(f"price_{cusip}", 100.0)) / max(float(form.get(f"purchase_{cusip}", 100.0)), 0.01),
                "face_value": float(form.get(f"face_{cusip}", 0.0)),
                "coupon_rate": float(form.get(f"coupon_{cusip}", 0.0)) / 100,
                "purchase_date": form.get(f"date_{cusip}", ""),
            }

    pf.save_bond_prices(bond_data)

    # Trigger background refresh
    import threading
    threading.Thread(target=data_cache.refresh, daemon=True).start()

    return RedirectResponse("/fund/fixed-income?msg=Bond+prices+updated", status_code=303)


@app.get("/api/live-prices/{fund_slug}")
async def live_prices(fund_slug: str):
    """API endpoint for live intraday price overlay."""
    slug_map = {n.lower().replace(" ", "-"): n for n in pf.SUBFUNDS}
    name = slug_map.get(fund_slug)
    if not name:
        return JSONResponse({"error": "unknown fund"}, status_code=404)

    cache = data_cache.get_cache()
    subfund_data = cache.get("subfund_data", {})
    if name not in subfund_data:
        return JSONResponse({"error": "no data"}, status_code=404)

    d = subfund_data[name]
    rets = d["returns"]
    if rets.empty:
        return JSONResponse({"error": "no returns"}, status_code=404)

    # Return the latest return data point
    last_date = rets.index[-1].isoformat()
    cum = float(pf.cum_return(rets).iloc[-1] * 100)
    return JSONResponse({"date": last_date, "cum_return": cum})
