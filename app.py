"""
NYU MPSIF Return Attribution Dashboard
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import os
from pathlib import Path
from PIL import Image
import portfolio as pf

# ── Page config ────────────────────────────────────────────────────────────
_favicon = Image.open(Path("assets/favicon.png"))
st.set_page_config(
    page_title="NYU MPSIF",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-refresh every 15 minutes via HTML meta tag
st.markdown('<meta http-equiv="refresh" content="900">', unsafe_allow_html=True)

# ── Colors ────────────────────────────────────────────────────────────────
NYU_PURPLE = "#57068C"
NYU_PURPLE_LIGHT = "#8900e1"
NYU_PURPLE_BG = "#F5F0FA"
BLACK = "#1A1A1A"
WHITE = "#FFFFFF"
GRAY = "#6B7280"
GREEN = "#10B981"
RED = "#EF4444"

st.markdown(f"""
<style>
    /* ── Web font fallback for Linux/Render ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Typography ── */
    * {{
        font-family: 'Helvetica Neue', Helvetica, Inter, Arial, sans-serif !important;
        font-weight: 300;
    }}

    /* ── Force Plotly SVG text to use our font ── */
    .js-plotly-plot text,
    .js-plotly-plot .gtitle,
    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text,
    .js-plotly-plot .g-xtitle text,
    .js-plotly-plot .g-ytitle text,
    .js-plotly-plot .legendtext {{
        font-family: 'Helvetica Neue', Helvetica, Inter, Arial, sans-serif !important;
        font-weight: 300 !important;
    }}
    h1, h2, h3, h4, h5, h6,
    .metric-label, .metric-value, .section-header, .logo,
    .stTabs [data-baseweb="tab"],
    th, thead {{
        font-weight: 500 !important;
    }}

    .main .block-container {{ padding-top: 0 !important; max-width: 1200px; }}
    .stAppHeader, [data-testid="stHeader"] {{ display: none !important; height: 0 !important; }}
    .stApp > header {{ display: none !important; }}
    .stMainBlockContainer {{ padding-top: 0.5rem !important; }}

    /* ── Logo ── */
    .logo {{
        font-size: 2rem; font-weight: 700 !important;
        letter-spacing: -0.5px; margin-bottom: 0.25rem;
    }}
    .logo-nyu {{ color: {NYU_PURPLE}; }}
    .logo-mpsif {{ color: {BLACK}; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 2px solid #E5E7EB; }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.95rem; color: {GRAY};
        padding: 0.75rem 1.5rem; border-bottom: 2px solid transparent; margin-bottom: -2px;
    }}
    .stTabs [aria-selected="true"] {{
        color: {NYU_PURPLE} !important;
        border-bottom-color: {NYU_PURPLE} !important;
        font-weight: 600 !important;
    }}

    /* ── Metric cards ── */
    .metric-card {{
        background: {WHITE}; border: 1px solid #E5E7EB;
        border-radius: 12px; padding: 1.25rem; text-align: center;
    }}
    .metric-label {{
        font-size: 0.75rem; color: {GRAY};
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;
    }}
    .metric-value {{ font-size: 1.5rem; font-weight: 700 !important; color: {BLACK}; }}
    .metric-positive {{ color: {GREEN}; }}
    .metric-negative {{ color: {RED}; }}

    /* ── Section headers ── */
    .section-header {{
        font-size: 1.1rem; font-weight: 600 !important; color: {BLACK};
        margin-top: 1.5rem; margin-bottom: 0.75rem;
        padding-bottom: 0.5rem; border-bottom: 1px solid #E5E7EB;
    }}

    /* ── HTML tables ── */
    .clean-table {{
        width: 100%; border-collapse: collapse;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 300; font-size: 0.85rem;
    }}
    .clean-table thead th {{
        background: {NYU_PURPLE_BG}; color: {BLACK};
        font-weight: 500; font-size: 0.8rem;
        padding: 0.6rem 0.75rem; text-align: left;
        border-bottom: 2px solid #E5E7EB;
    }}
    .clean-table tbody td {{
        padding: 0.5rem 0.75rem; border-bottom: 1px solid #F3F4F6;
        font-weight: 300;
    }}
    .clean-table tbody tr:hover {{ background: #FAFAFA; }}

    hr {{ border: none; border-top: 1px solid #E5E7EB; margin: 1.5rem 0; }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ── File uploader ── */
    [data-testid="stFileUploader"] button {{
        border-radius: 12px !important;
        padding: 0.15rem 0.5rem !important;
        font-size: 0.55rem !important;
        font-family: 'Helvetica Neue Light', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-weight: 300 !important;
        border: 1px solid #E5E7EB !important;
        background: {WHITE} !important;
        color: {GRAY} !important;
        transition: all 0.2s ease !important;
        min-height: unset !important;
        line-height: 1.2 !important;
    }}
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] label {{
        font-size: 0.7rem !important;
        font-family: 'Helvetica Neue Light', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }}
    [data-testid="stFileUploader"] small {{
        font-size: 0.6rem !important;
        font-family: 'Helvetica Neue Light', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }}
    [data-testid="stFileUploader"] button:hover {{
        background: {NYU_PURPLE_BG} !important;
        border-color: {NYU_PURPLE} !important;
        color: {NYU_PURPLE} !important;
    }}

    /* Refresh button */
    button[data-testid="stBaseButton-secondary"] {{
        border-radius: 50% !important;
        width: 2.5rem !important; height: 2.5rem !important;
        padding: 0 !important;
        border: 1px solid #E5E7EB !important;
        background: {WHITE} !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }}
    button[data-testid="stBaseButton-secondary"]:hover {{
        background: {NYU_PURPLE_BG} !important;
        border-color: {NYU_PURPLE} !important;
        box-shadow: 0 4px 12px rgba(87, 6, 140, 0.15) !important;
        transform: translateY(-1px) !important;
    }}

    /* ── Mobile responsive ── */
    @media (max-width: 768px) {{
        .modebar {{ display: none !important; }}
        .main .block-container {{
            padding-top: 0.75rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }}

        /* Logo */
        .logo {{ font-size: 1.4rem !important; }}

        /* Tabs: scrollable, smaller text */
        .stTabs [data-baseweb="tab-list"] {{
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            -webkit-overflow-scrolling: touch;
        }}
        .stTabs [data-baseweb="tab"] {{
            font-size: 0.8rem !important;
            padding: 0.5rem 0.75rem !important;
            white-space: nowrap !important;
        }}

        /* Metric cards: smaller + vertical spacing */
        .metric-card {{
            padding: 0.6rem 0.4rem !important;
            border-radius: 8px !important;
            margin-bottom: 1rem !important;
        }}
        [data-testid="stHorizontalBlock"] {{
            row-gap: 1.25rem !important;
        }}
        .metric-label {{
            font-size: 0.6rem !important;
            letter-spacing: 0.25px !important;
        }}
        .metric-value {{
            font-size: 0.95rem !important;
        }}

        /* Section headers */
        .section-header {{
            font-size: 0.95rem !important;
            margin-top: 1rem !important;
        }}

        /* Columns: allow wrapping on small screens */
        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
            gap: 0.35rem !important;
        }}
        /* 2-col grid on mobile */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            min-width: 46% !important;
            flex: 1 1 46% !important;
        }}

        /* Segmented controls: wrap and smaller */
        [data-testid="stSegmentedControl"] {{
            flex-wrap: wrap !important;
            gap: 0.3rem !important;
        }}
        button[data-testid="stBaseButton-segmented_control"] {{
            font-size: 0.75rem !important;
            padding: 0.3rem 0.7rem !important;
        }}

        /* Tables: smaller text */
        .clean-table {{ font-size: 0.75rem !important; }}
        .clean-table thead th {{ font-size: 0.7rem !important; padding: 0.4rem 0.5rem !important; }}
        .clean-table tbody td {{ padding: 0.35rem 0.5rem !important; }}

        /* Footer */
        .footer-logo {{ height: 30px !important; }}
    }}

    /* Extra small phones */
    @media (max-width: 480px) {{
        .logo {{ font-size: 1.2rem !important; }}
        .metric-value {{ font-size: 0.85rem !important; }}
        .metric-label {{ font-size: 0.55rem !important; }}
        .metric-card {{ padding: 0.5rem 0.3rem !important; }}
        .stTabs [data-baseweb="tab"] {{
            font-size: 0.7rem !important;
            padding: 0.4rem 0.5rem !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def fmt_pct(v: float) -> str:
    return f"{v:+.3f}%"

def fmt_dollar(v: float) -> str:
    if abs(v) >= 1e6:
        return f"${v/1e6:,.3f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:,.3f}K"
    return f"${v:,.3f}"

def color_class(v: float) -> str:
    return "metric-positive" if v >= 0 else "metric-negative"

def metric_card(label: str, value: str, css_class: str = ""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
    </div>
    """


_table_counter = {"n": 0}

def html_table(df: pd.DataFrame, raw_df: pd.DataFrame = None,
               max_height: str = "400px", default_sort: str = None,
               default_asc: bool = False) -> str:
    """Render a DataFrame as a sortable HTML table with Helvetica Neue Light.

    df: formatted DataFrame (display strings)
    raw_df: optional DataFrame with raw numeric values for sorting.
            If None, sorts lexicographically on display values.
    default_sort: column name to sort by initially
    default_asc: True for ascending, False for descending
    """
    _table_counter["n"] += 1
    tid = f"tbl{_table_counter['n']}"

    header = ""
    for i, c in enumerate(df.columns):
        arrow = ""
        if c == default_sort:
            arrow = " ▲" if default_asc else " ▼"
        header += f'<th style="cursor:pointer;user-select:none;" onclick="{tid}_sort({i})">{c}<span id="{tid}_arr{i}">{arrow}</span></th>'

    rows = ""
    for idx in range(len(df)):
        cells = ""
        for i, c in enumerate(df.columns):
            sv = raw_df.iloc[idx, i] if raw_df is not None and c in raw_df.columns else df.iloc[idx, i]
            try:
                sv = float(sv)
            except (ValueError, TypeError):
                sv = str(df.iloc[idx, i])
            cells += f'<td data-sv="{sv}">{df.iloc[idx, i]}</td>'
        rows += f"<tr>{cells}</tr>"

    # Default sort column index
    sort_col = 0
    if default_sort and default_sort in df.columns:
        sort_col = list(df.columns).index(default_sort)
    asc_js = "true" if default_asc else "false"

    script = f"""<script>
    (function(){{
      var tbl=document.getElementById("{tid}");
      var dirs={{}};
      var ncols={len(df.columns)};
      for(var i=0;i<ncols;i++) dirs[i]=null;
      dirs[{sort_col}]={asc_js};
      window.{tid}_sort=function(col){{
        var tbody=tbl.querySelector("tbody");
        var rows=Array.from(tbody.querySelectorAll("tr"));
        var asc=dirs[col]===null?true:!dirs[col];
        dirs[col]=asc;
        rows.sort(function(a,b){{
          var av=a.children[col].getAttribute("data-sv");
          var bv=b.children[col].getAttribute("data-sv");
          var an=parseFloat(av),bn=parseFloat(bv);
          if(!isNaN(an)&&!isNaN(bn)) return asc?an-bn:bn-an;
          return asc?av.localeCompare(bv):bv.localeCompare(av);
        }});
        rows.forEach(function(r){{tbody.appendChild(r);}});
        for(var i=0;i<ncols;i++){{
          var el=document.getElementById("{tid}_arr"+i);
          if(el) el.textContent=i===col?(asc?" ▲":" ▼"):"";
        }}
      }};
    }})();
    </script>"""

    return f"""<html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>
    * {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 300; margin: 0; padding: 0; }}
    .clean-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
    .clean-table thead th {{ background:{NYU_PURPLE_BG}; color:{BLACK}; font-weight:500; font-size:0.8rem;
        padding:0.6rem 0.75rem; text-align:left; border-bottom:2px solid #E5E7EB; cursor:pointer; user-select:none; white-space:nowrap; }}
    .clean-table tbody td {{ padding:0.5rem 0.75rem; border-bottom:1px solid #F3F4F6; font-weight:300; white-space:nowrap; }}
    .clean-table tbody tr:hover {{ background:#FAFAFA; }}
    @media (max-width: 768px) {{
        .clean-table {{ font-size:0.75rem; }}
        .clean-table thead th {{ font-size:0.7rem; padding:0.4rem 0.5rem; }}
        .clean-table tbody td {{ padding:0.35rem 0.5rem; }}
    }}
    </style></head><body>
    <div style="max-height:{max_height}; overflow-y:auto;">
    <table class="clean-table" id="{tid}"><thead><tr>{header}</tr></thead>
    <tbody>{rows}</tbody></table></div>{script}
    </body></html>"""


# Plotly config: hide toolbar on mobile, minimal on desktop
PLOTLY_CFG = {
    "displayModeBar": "hover",
    "toImageButtonOptions": {"format": "png", "scale": 4},
}
_PLOTLY_FONT = dict(family="Helvetica Neue, Helvetica, Inter, Arial, sans-serif", size=13, color=GRAY)
_PLOTLY_AXIS_FONT = dict(family="Helvetica Neue, Helvetica, Inter, Arial, sans-serif", size=12, color=GRAY)
_PLOTLY_TITLE_FONT = dict(family="Helvetica Neue, Helvetica, Inter, Arial, sans-serif", size=12, color=GRAY)

# ── Charts ─────────────────────────────────────────────────────────────────
def make_return_chart(rets: pd.Series, height: int = 380,
                      bench_rets: pd.Series = None, bench_label: str = ""):
    if rets.empty:
        return go.Figure()
    cum = pf.cum_return(rets) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values, mode="lines", name="Portfolio",
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
                x=bench_cum.index, y=bench_cum.values, mode="lines",
                name=bench_label,
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


def make_drawdown_chart(rets: pd.Series, height: int = 200):
    if rets.empty:
        return go.Figure()
    cum = (1 + rets).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines",
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


def make_multi_fund_chart(fund_data: dict, visible: list, height: int = 400):
    colors = {
        "Total Fund": BLACK,
        "Systematic": "#2563EB",
        "Opportunistic": "#DC2626",
        "Thematic": "#059669",
        "Fixed Income": "#7C3AED",
        "Benchmark": "#F59E0B",
    }
    fig = go.Figure()
    for name, rets in fund_data.items():
        if rets is None or rets.empty:
            continue
        cum = pf.cum_return(rets) * 100
        line_width = 3.5 if name == "Total Fund" else 1.8
        line_dash = "dash" if name == "Benchmark" else None
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines", name=name,
            line=dict(color=colors.get(name, GRAY), width=line_width, dash=line_dash),
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


def make_attribution_bar(attr_df: pd.DataFrame, top_n: int = 15, height: int = 400):
    if attr_df.empty:
        return go.Figure()
    df = attr_df.head(top_n).copy().sort_values("Contribution (bps)")
    colors = [GREEN if v >= 0 else RED for v in df["Contribution (bps)"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["Ticker"], x=df["Contribution (bps)"], orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:+.3f} bps<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Contribution (bps)", showgrid=True, gridcolor="#F3F4F6",
                   tickfont=_PLOTLY_AXIS_FONT, title_font=_PLOTLY_TITLE_FONT),
        yaxis=dict(showgrid=False, tickfont=_PLOTLY_AXIS_FONT),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=_PLOTLY_FONT,
    )
    return fig


def make_holdings_pie(holdings_df: pd.DataFrame, height: int = 350):
    if holdings_df.empty:
        return go.Figure()
    df = holdings_df.copy()
    threshold = 2.0
    small = df[df["Weight (%)"] < threshold]
    large = df[df["Weight (%)"] >= threshold]
    if len(small) > 0:
        other = pd.DataFrame([{
            "Ticker": f"Other ({len(small)})",
            "Weight (%)": small["Weight (%)"].sum(),
            "Value ($)": small["Value ($)"].sum(),
        }])
        df = pd.concat([large, other], ignore_index=True)
    else:
        df = large
    fig = go.Figure(go.Pie(
        labels=df["Ticker"], values=df["Weight (%)"], hole=0.45,
        textinfo="label+percent", textfont_size=11,
        marker=dict(colors=[
            NYU_PURPLE, "#8900e1", "#2563EB", "#3B82F6", "#60A5FA",
            "#D97706", "#F59E0B", "#059669", "#10B981", "#34D399",
            "#EF4444", "#6366F1", "#8B5CF6", "#EC4899", "#F97316",
            "#14B8A6", "#64748B", "#A78BFA", "#FB923C", "#4ADE80",
        ]),
        hovertemplate="%{label}<br>Weight: %{value:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=WHITE, font=_PLOTLY_FONT,
        showlegend=False,
    )
    return fig


# ── Data loading ───────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
SUBFUND_FILES = {
    "Systematic": DATA_DIR / "systematic.csv",
    "Opportunistic": DATA_DIR / "opportunistic.csv",
    "Thematic": DATA_DIR / "thematic.csv",
    "Fixed Income": DATA_DIR / "fixed_income.csv",
}


@st.cache_data(ttl=900, show_spinner=False)
def load_all_subfunds():
    results = {}
    for name, path in SUBFUND_FILES.items():
        if path.exists():
            try:
                results[name] = pf.build_subfund(str(path))
            except Exception as e:
                st.warning(f"Error loading {name}: {e}")
    return results


def load_benchmark_returns(subfund_data):
    bench_rets = {}
    for name, d in subfund_data.items():
        ticker = pf.BENCHMARKS.get(name)
        if not ticker:
            continue
        start = (d["first_date"] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end = (d["end_date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        bench_rets[name] = pf.fetch_benchmark_returns(ticker, start, end)
    return bench_rets


def combine_returns(subfund_data: dict) -> pd.Series:
    """AUM-weighted average of sub-fund daily returns.
    Each fund is weighted by its prior-day Total AUM, but only included
    from the date it first has return data (avoids cash inflow spikes)."""
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

    # Build aligned DataFrames
    ret_df = pd.DataFrame(fund_rets).sort_index()
    aum_df = pd.DataFrame(fund_aum).sort_index().ffill().reindex(ret_df.index).ffill().fillna(0)

    # Prior-day AUM weights — use NaN (not 0) for days before a fund has data,
    # so those funds are excluded from weighting rather than given 0 weight
    weights = aum_df.shift(1)
    for name in fund_rets:
        first_date = fund_rets[name].index[0]
        weights.loc[weights.index < first_date, name] = np.nan

    row_totals = weights.sum(axis=1, min_count=1)  # NaN if all NaN
    weights = weights.div(row_totals, axis=0)

    # Weighted return: only include funds that have data (NaN weights → excluded)
    common = weights.index.intersection(ret_df.index)
    combined = (ret_df.loc[common].fillna(0) * weights.loc[common].fillna(0)).sum(axis=1)
    combined = combined.dropna()
    combined = combined[combined.index >= ret_df.index[0]]
    return combined

# used to make Thematic fund's weekly returns attributed by themes
def make_theme_attribution_bar(attribution_df: pd.DataFrame, height: int = 400):
    """Bar chart of last week's returns attributed to themes."""
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
        x=df["Contribution"],
        y=df["Theme"],
        orientation="h",
        marker=dict(color=colors),
        hovertemplate="%{y}<br>Contribution: %{x:+.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=20, t=10, b=0),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=_PLOTLY_FONT,
        xaxis=dict(title="Contribution (%)", zeroline=True, zerolinecolor="#CBD5E1"),
        yaxis=dict(title=""),
    )
    return fig


@st.cache_data(ttl=900, show_spinner=False)
def load_blended_benchmark(start: str, end: str) -> pd.Series:
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


# ── Load data ──────────────────────────────────────────────────────────────
subfund_data = load_all_subfunds()
benchmark_rets = load_benchmark_returns(subfund_data) if subfund_data else {}

# Load blended benchmark aligned to portfolio date range
blended_bench_rets = pd.Series(dtype=float)
if subfund_data:
    _all_starts = [d["first_date"] for d in subfund_data.values()]
    _all_ends = [d["end_date"] for d in subfund_data.values()]
    # Start benchmark from when the first fund actually started trading
    _earliest = min(_all_starts) - pd.Timedelta(days=5)
    _bench_start = _earliest.strftime("%Y-%m-%d")
    _bench_end = (max(_all_ends) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    blended_bench_rets = load_blended_benchmark(_bench_start, _bench_end)

# ── Header ─────────────────────────────────────────────────────────────────
hdr1, hdr2 = st.columns([10, 1])
with hdr1:
    st.markdown('<div class="logo"><span class="logo-nyu">NYU</span> <span class="logo-mpsif">MPSIF</span></div>', unsafe_allow_html=True)
    st.caption("Return Attribution Dashboard")
with hdr2:
    if st.button("↻", help="Refresh prices & data"):
        st.cache_data.clear()
        st.rerun()

# ── Tabs (always show all) ────────────────────────────────────────────────
tab_names = ["Overview"] + pf.SUBFUNDS + ["Upload"]
tabs = st.tabs(tab_names)


# ═══════════════════════════════════════════════════════════════════════════
#  OVERVIEW TAB
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if not subfund_data:
        st.info("No transaction data loaded for any sub-fund yet.")
    else:
        combined_rets = combine_returns(subfund_data)

        if combined_rets.empty:
            st.info("No return data available yet.")
        else:
            period_rets = pf.period_returns(combined_rets)
            cols = st.columns(5)
            for i, (period, val) in enumerate(period_rets.items()):
                with cols[i]:
                    st.markdown(metric_card(period, fmt_pct(val * 100), color_class(val)), unsafe_allow_html=True)
            st.markdown("")

            total_aum = sum(
                d["portfolio_values"]["Total"].iloc[-1]
                for d in subfund_data.values()
                if len(d["portfolio_values"])
            )
            mcols = st.columns(4)
            with mcols[0]:
                st.markdown(metric_card("Total AUM", fmt_dollar(total_aum)), unsafe_allow_html=True)
            with mcols[1]:
                st.markdown(metric_card(f"Sharpe Ratio (Rf={pf.RISK_FREE_RATE:.1%})", f"{pf.sharpe(combined_rets):.3f}"), unsafe_allow_html=True)
            with mcols[2]:
                st.markdown(metric_card("Max Drawdown", fmt_pct(pf.max_dd(combined_rets) * 100), color_class(pf.max_dd(combined_rets))), unsafe_allow_html=True)
            with mcols[3]:
                st.markdown(metric_card("Ann. Volatility", fmt_pct(pf.ann_vol(combined_rets) * 100)), unsafe_allow_html=True)
            st.markdown("")

            # ── Blended benchmark metrics ──
            if not blended_bench_rets.empty:
                st.markdown('<div class="section-header">vs. Blended Benchmark (50% SPY + 25% IWV + 25% AGG)</div>', unsafe_allow_html=True)
                reg = pf.regression_stats(combined_rets, blended_bench_rets)
                bcols = st.columns(4)
                with bcols[0]:
                    st.markdown(metric_card("Excess Return", fmt_pct(reg["excess_return"] * 100), color_class(reg["excess_return"])), unsafe_allow_html=True)
                with bcols[1]:
                    st.markdown(metric_card("Alpha (Ann.)", fmt_pct(reg["alpha"] * 100), color_class(reg["alpha"])), unsafe_allow_html=True)
                with bcols[2]:
                    st.markdown(metric_card("Market Beta", f"{reg['beta']:.3f}"), unsafe_allow_html=True)
                with bcols[3]:
                    st.markdown(metric_card("Idio. Vol (Ann.)", fmt_pct(reg["idio_vol"] * 100)), unsafe_allow_html=True)
            st.markdown("")

            # ── Multi-fund chart ──
            st.markdown('<div class="section-header">Cumulative Returns</div>', unsafe_allow_html=True)
            bench_label = "Benchmark"
            fund_names = ["Total Fund"] + list(subfund_data.keys()) + ([bench_label] if not blended_bench_rets.empty else [])
            visible_funds = st.segmented_control(
                "Funds", fund_names, default=fund_names,
                selection_mode="multi", label_visibility="collapsed",
                key="home_fund_toggle",
            )
            if visible_funds is None:
                visible_funds = fund_names

            fund_rets = {"Total Fund": combined_rets}
            for name, d in subfund_data.items():
                fund_rets[name] = d["returns"]
            if not blended_bench_rets.empty:
                fund_rets[bench_label] = blended_bench_rets
            st.plotly_chart(make_multi_fund_chart(fund_rets, visible_funds), use_container_width=True, config=PLOTLY_CFG)

            # ── Drawdown ──
            st.markdown('<div class="section-header">Drawdown</div>', unsafe_allow_html=True)
            st.plotly_chart(make_drawdown_chart(combined_rets), use_container_width=True, config=PLOTLY_CFG)

            # ── Summary table ──
            st.markdown('<div class="section-header">Sub-Fund Summary</div>', unsafe_allow_html=True)
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
            components.html(html_table(pd.DataFrame(summary_rows), default_sort="Sub-Fund"), height=min(400, 45 * len(summary_rows) + 50), scrolling=True)

            # ── Correlation Matrix ──
            if len(subfund_data) >= 2:
                st.markdown('<div class="section-header">Sub-Fund Correlation Matrix</div>', unsafe_allow_html=True)
                corr_rets = {}
                for sname, sd in subfund_data.items():
                    corr_rets[sname] = sd["returns"]
                corr_df = pd.DataFrame(corr_rets).dropna()
                if len(corr_df) >= 5:
                    corr_matrix = corr_df.corr()
                    # Heatmap
                    fig_corr = go.Figure(go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns.tolist(),
                        y=corr_matrix.index.tolist(),
                        colorscale=[[0, "#DC2626"], [0.5, WHITE], [1, NYU_PURPLE]],
                        zmin=-1, zmax=1,
                        text=[[f"{v:.3f}" for v in row] for row in corr_matrix.values],
                        texttemplate="%{text}",
                        textfont=dict(size=14),
                        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
                    ))
                    fig_corr.update_layout(
                        height=300, margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                        font=dict(family="'Helvetica Neue', Helvetica, Arial, sans-serif", weight=300),
                        xaxis=dict(side="bottom"),
                    )
                    st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CFG)


# ═══════════════════════════════════════════════════════════════════════════
#  SUB-FUND TABS
# ═══════════════════════════════════════════════════════════════════════════
for idx, name in enumerate(pf.SUBFUNDS):
    with tabs[idx + 1]:
        if name not in subfund_data:
            st.info(f"No transaction data available for **{name}**. Provide the Fidelity CSV to load this sub-fund.")
            continue

        d = subfund_data[name]
        rets = d["returns"]
        tv = d["ticker_values"]
        holdings = d["holdings"]
        dividends = d["dividends"]
        port_val = d["portfolio_values"]
        bench_r = benchmark_rets.get(name, pd.Series(dtype=float))
        bench_ticker = pf.BENCHMARKS.get(name, "")

        if rets.empty:
            st.info(f"No return data for {name}.")
            continue

        # ── Period returns ──
        pr = pf.period_returns(rets)
        pcols = st.columns(5)
        for i, (period, val) in enumerate(pr.items()):
            with pcols[i]:
                st.markdown(metric_card(period, fmt_pct(val * 100), color_class(val)), unsafe_allow_html=True)
        st.markdown("")

        # ── Key metrics ──
        aum = port_val["Total"].iloc[-1]
        mcols = st.columns(4)
        with mcols[0]:
            st.markdown(metric_card("AUM", fmt_dollar(aum)), unsafe_allow_html=True)
        with mcols[1]:
            st.markdown(metric_card(f"Sharpe (Rf={pf.RISK_FREE_RATE:.1%})", f"{pf.sharpe(rets):.3f}"), unsafe_allow_html=True)
        with mcols[2]:
            st.markdown(metric_card("Max DD", fmt_pct(pf.max_dd(rets) * 100), color_class(pf.max_dd(rets))), unsafe_allow_html=True)
        with mcols[3]:
            st.markdown(metric_card("Ann. Vol", fmt_pct(pf.ann_vol(rets) * 100)), unsafe_allow_html=True)

        # ── Benchmark metrics ──
        if not bench_r.empty:
            st.markdown(f'<div class="section-header">vs. {bench_ticker} Benchmark</div>', unsafe_allow_html=True)
            reg = pf.regression_stats(rets, bench_r)
            bcols = st.columns(4)
            with bcols[0]:
                st.markdown(metric_card("Excess Return", fmt_pct(reg["excess_return"] * 100), color_class(reg["excess_return"])), unsafe_allow_html=True)
            with bcols[1]:
                st.markdown(metric_card("Alpha (Ann.)", fmt_pct(reg["alpha"] * 100), color_class(reg["alpha"])), unsafe_allow_html=True)
            with bcols[2]:
                st.markdown(metric_card("Market Beta", f"{reg['beta']:.3f}"), unsafe_allow_html=True)
            with bcols[3]:
                st.markdown(metric_card("Idio. Vol (Ann.)", fmt_pct(reg["idio_vol"] * 100)), unsafe_allow_html=True)

        st.markdown("")

        # ── Cumulative return chart ──
        st.markdown(f'<div class="section-header">{name} — Cumulative Return</div>', unsafe_allow_html=True)
        chart_period = st.segmented_control(
            "Chart period", ["1W", "1M", "YTD", "All", "Custom"],
            default="YTD", key=f"chart_period_{name}",
            label_visibility="collapsed",
        )
        if chart_period is None:
            chart_period = "YTD"

        chart_rets = rets.copy()
        if chart_period == "1W":
            chart_rets = rets[rets.index > rets.index[-1] - pd.Timedelta(days=7)]
        elif chart_period == "1M":
            chart_rets = rets[rets.index > rets.index[-1] - pd.DateOffset(months=1)]
        elif chart_period == "YTD":
            chart_rets = rets[rets.index >= pd.Timestamp(rets.index[-1].year, 1, 1)]
        elif chart_period == "Custom":
            dc1, dc2 = st.columns(2)
            with dc1:
                custom_start = st.date_input("Start date", value=rets.index[0].date(),
                                             min_value=rets.index[0].date(), max_value=rets.index[-1].date(),
                                             key=f"custom_start_{name}")
            with dc2:
                custom_end = st.date_input("End date", value=rets.index[-1].date(),
                                           min_value=rets.index[0].date(), max_value=rets.index[-1].date(),
                                           key=f"custom_end_{name}")
            chart_rets = rets[(rets.index >= pd.Timestamp(custom_start)) & (rets.index <= pd.Timestamp(custom_end))]

        # Align benchmark to chart period
        chart_bench = bench_r
        if not bench_r.empty and not chart_rets.empty:
            chart_bench = bench_r[bench_r.index.isin(chart_rets.index)]

        st.plotly_chart(
            make_return_chart(chart_rets, bench_rets=chart_bench, bench_label=bench_ticker),
            use_container_width=True, config=PLOTLY_CFG,
        )

        # ── Drawdown ──
        st.markdown('<div class="section-header">Drawdown</div>', unsafe_allow_html=True)
        st.plotly_chart(make_drawdown_chart(rets, height=180), use_container_width=True, config=PLOTLY_CFG)

        # ── Current Holdings ──
        st.markdown('<div class="section-header">Current Holdings</div>', unsafe_allow_html=True)
        if not holdings.empty:
            avg_costs = d["avg_costs"]
            display_h = holdings.copy()
            display_h["Avg Price ($)"] = display_h["Ticker"].map(avg_costs).fillna(0)
            display_h["P&L ($)"] = (display_h["Price ($)"] - display_h["Avg Price ($)"]) * display_h["Shares"]

            # Raw values for sorting
            raw_cols = ["Ticker", "Shares", "Weight (%)", "Avg Price ($)", "Price ($)", "P&L ($)"]
            raw_tbl = display_h[raw_cols].copy().sort_values("Weight (%)", ascending=False)

            # Formatted for display
            tbl = raw_tbl.copy()
            tbl["Shares"] = tbl["Shares"].apply(lambda x: f"{x:.3f}")
            tbl["Weight (%)"] = tbl["Weight (%)"].apply(lambda x: f"{x:.3f}%")
            tbl["Avg Price ($)"] = tbl["Avg Price ($)"].apply(lambda x: f"${x:,.3f}")
            tbl["Price ($)"] = tbl["Price ($)"].apply(lambda x: f"${x:,.3f}")
            tbl["P&L ($)"] = tbl["P&L ($)"].apply(lambda x: f"${x:,.3f}")

            hcol1, hcol2 = st.columns([1, 2])
            with hcol1:
                st.plotly_chart(make_holdings_pie(holdings), use_container_width=True, config=PLOTLY_CFG)
            with hcol2:
                components.html(html_table(tbl, raw_df=raw_tbl, max_height="450px",
                                       default_sort="Weight (%)", default_asc=False), height=min(500, 40 * len(tbl) + 55), scrolling=True)
        else:
            st.info("No current holdings.")

        # ── Sector Exposure ──
        if not holdings.empty:
            st.markdown('<div class="section-header">Sector Exposure</div>', unsafe_allow_html=True)
            sectors = pf.get_sectors(holdings["Ticker"].tolist())
            sector_df = holdings.copy()
            sector_df["Sector"] = sector_df["Ticker"].map(sectors).fillna("Other")
            sector_agg = sector_df.groupby("Sector")["Weight (%)"].sum().reset_index()
            sector_agg = sector_agg.sort_values("Weight (%)", ascending=False)

            SECTOR_COLORS = {
                "Technology": "#2563EB", "Healthcare": "#059669",
                "Financial Services": "#D97706", "Consumer Cyclical": "#DC2626",
                "Industrials": "#7C3AED", "Energy": "#0EA5E9",
                "Communication Services": "#EC4899", "Consumer Defensive": "#10B981",
                "Basic Materials": "#F59E0B", "Real Estate": "#6366F1",
                "Utilities": "#14B8A6", "Other": "#94A3B8",
            }
            sec_colors = [SECTOR_COLORS.get(s, "#94A3B8") for s in sector_agg["Sector"]]

            scol1, scol2 = st.columns([1, 1])
            with scol1:
                fig_sec = go.Figure(go.Pie(
                    labels=sector_agg["Sector"], values=sector_agg["Weight (%)"], hole=0.45,
                    textinfo="label+percent", textfont_size=11,
                    marker=dict(colors=sec_colors),
                    hovertemplate="%{label}<br>Weight: %{value:.3f}%<extra></extra>",
                ))
                fig_sec.update_layout(
                    height=350, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor=WHITE, font=dict(family="'Helvetica Neue', Helvetica, Arial, sans-serif", weight=300),
                    showlegend=False,
                )
                st.plotly_chart(fig_sec, use_container_width=True, config=PLOTLY_CFG)
            with scol2:
                sec_display = sector_agg.copy()
                sec_display["Weight (%)"] = sec_display["Weight (%)"].apply(lambda x: f"{x:.3f}%")
                # Count holdings per sector
                sec_counts = sector_df.groupby("Sector")["Ticker"].count().reset_index()
                sec_counts.columns = ["Sector", "# Holdings"]
                sec_display = sec_display.merge(sec_counts, on="Sector")
                components.html(html_table(sec_display, max_height="350px"), height=min(400, 40 * len(sec_display) + 55), scrolling=True)

        # ── Factor Exposure ──
        if not rets.empty:
            start_str = (d["first_date"] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            end_str = (d["end_date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            def render_factor_table(result: dict, title: str, key_suffix: str):
                """Render factor exposure as metric cards + stats table."""
                if not result:
                    return
                st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
                alpha = result.pop("_alpha", 0.0)
                idio = result.pop("_idio_vol", 0.0)
                r_sq = result.pop("_r_squared", 0.0)
                alpha_t = result.pop("_alpha_t", 0.0)
                alpha_p = result.pop("_alpha_p", 0.0)
                stats = result.pop("_stats", {})

                # Factor beta cards + R²
                factor_items = {k: v for k, v in result.items() if not k.startswith("_")}
                fcols = st.columns(len(factor_items) + 3)
                with fcols[0]:
                    st.markdown(metric_card("R²", f"{r_sq:.3f}"), unsafe_allow_html=True)
                with fcols[1]:
                    st.markdown(metric_card("Alpha (Ann.)", fmt_pct(alpha * 100), color_class(alpha)), unsafe_allow_html=True)
                for i, (fname, bval) in enumerate(factor_items.items()):
                    with fcols[i + 2]:
                        st.markdown(metric_card(fname, f"{bval:.3f}", color_class(bval)), unsafe_allow_html=True)
                with fcols[-1]:
                    st.markdown(metric_card("Idio. Vol (Ann.)", fmt_pct(idio * 100)), unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                
                
                # Stats table
                # stat_rows = [{"Factor": "Alpha (Ann.)", "Beta": f"{alpha:.3f}", "t-stat": f"{alpha_t:.3f}", "p-value": f"{alpha_p:.3f}"}]
                # for fname, bval in factor_items.items():
                #     s = stats.get(fname, {})
                #     t = s.get("t_stat", 0.0)
                #     p = s.get("p_value", 1.0)
                #     stat_rows.append({"Factor": fname, "Beta": f"{bval:.3f}", "t-stat": f"{t:.3f}", "p-value": f"{p:.3f}"})
                # stat_df = pd.DataFrame(stat_rows)
                # components.html(html_table(stat_df, max_height="250px"), height=min(250, 40 * len(stat_df) + 55), scrolling=True)

            render_factor_table(pf.compute_factor_betas(rets, holdings, start_str, end_str), "Factor Exposure (Fama-French)", f"ff_{name}")
            render_factor_table(pf.compute_etf_factor_betas(rets, start_str, end_str), "Factor Exposure (ETF Proxies)", f"etf_{name}")

        # Compute factor betas (save copy before render_factor_table pops keys)
        ff_result = pf.compute_factor_betas(rets, holdings, start_str, end_str)
        ff_betas_copy = dict(ff_result)  # preserve before .pop() calls
        render_factor_table(ff_result, "Factor Exposure (Fama-French)", f"ff_{name}")
        render_factor_table(pf.compute_etf_factor_betas(rets, start_str, end_str), "Factor Exposure (ETF Proxies)", f"etf_{name}")

        # ── Weekly Return Attribution Tables ──
        # ── Factor Exposure ──
        if not rets.empty:
            start_str = (d["first_date"] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            end_str = (d["end_date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            def render_factor_table(result: dict, title: str, key_suffix: str):
                """Render factor exposure as metric cards + stats table."""
                if not result:
                    return
                st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
                alpha = result.pop("_alpha", 0.0)
                idio = result.pop("_idio_vol", 0.0)
                r_sq = result.pop("_r_squared", 0.0)
                alpha_t = result.pop("_alpha_t", 0.0)
                alpha_p = result.pop("_alpha_p", 0.0)
                stats = result.pop("_stats", {})
                # Factor beta cards + R²
                factor_items = {k: v for k, v in result.items() if not k.startswith("_")}
                fcols = st.columns(len(factor_items) + 3)
                with fcols[0]:
                    st.markdown(metric_card("R²", f"{r_sq:.3f}"), unsafe_allow_html=True)
                with fcols[1]:
                    st.markdown(metric_card("Alpha (Ann.)", fmt_pct(alpha * 100), color_class(alpha)), unsafe_allow_html=True)
                for i, (fname, bval) in enumerate(factor_items.items()):
                    with fcols[i + 2]:
                        st.markdown(metric_card(fname, f"{bval:.3f}", color_class(bval)), unsafe_allow_html=True)
                with fcols[-1]:
                    st.markdown(metric_card("Idio. Vol (Ann.)", fmt_pct(idio * 100)), unsafe_allow_html=True)
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

            # Compute factor betas (save a copy before render_factor_table pops keys)
            ff_result = pf.compute_factor_betas(rets, holdings, start_str, end_str)
            ff_betas_copy = dict(ff_result)  # preserve before .pop() calls
            render_factor_table(ff_result, "Factor Exposure", f"ff_{name}")
            render_factor_table(pf.compute_etf_factor_betas(rets, start_str, end_str), "Factor Exposure (ETF Proxies)", f"etf_{name}")

            # ── Weekly Return Attribution Tables ──
            st.markdown('<div class="section-header">Weekly Return Attribution</div>', unsafe_allow_html=True)

            # Load factor data
            factor_data = pd.read_pickle('data.pk')
            factor_data.index = pd.to_datetime(factor_data.index).normalize()
            factor_data = factor_data.rename(columns={'mkt': 'Market', 'momentum': 'Momentum', 'growth': 'Growth', 'value': 'Value'})

            # Portfolio weekly returns
            port_clean = rets.copy()
            port_clean.index = pd.to_datetime(port_clean.index).normalize()
            port_weekly = (1 + port_clean).resample("W-FRI").prod() - 1
            port_weekly = port_weekly.dropna()

            # Factor weekly returns
            factor_weekly = (1 + factor_data).resample("W-FRI").prod() - 1

            # SPY weekly returns for excess return table
            spy_prices = pf.fetch_prices(["SPY"], start_str, end_str)
            spy_daily = spy_prices["SPY"].pct_change().dropna() if "SPY" in spy_prices.columns else pd.Series(dtype=float)
            spy_daily.index = pd.to_datetime(spy_daily.index).normalize()
            spy_weekly = (1 + spy_daily).resample("W-FRI").prod() - 1 if not spy_daily.empty else pd.Series(dtype=float)

            # Extract betas from the saved copy
            beta_map = {}
            factor_names = ['Market', 'Momentum', 'Growth', 'Value']
            for fname in factor_names:
                for key, val in ff_betas_copy.items():
                    if key.startswith("_"):
                        continue
                    if fname.lower() in key.lower():
                        beta_map[fname] = val
                        break

            # Get last week
            common_dates = port_weekly.index.intersection(factor_weekly.index)
            if len(common_dates) > 0:
                last_week = common_dates[-1]
                port_ret_week = port_weekly.loc[last_week]
                factor_rets_week = {f: factor_weekly.loc[last_week, f] for f in factor_names if f in factor_weekly.columns}
                spy_ret_week = spy_weekly.loc[last_week] if last_week in spy_weekly.index else 0.0

                # ── Absolute Return Table ──
                imputed_total = sum(beta_map.get(f, 0) * factor_rets_week.get(f, 0) for f in factor_names)
                alpha_residual = port_ret_week - imputed_total

                abs_rows = [
                    {
                        "": "Sub-Fund",
                        "Total Return": f"{port_ret_week * 100:+.3f}%",
                        "Market β": f"{beta_map.get('Market', 0):.3f}",
                        "Value β": f"{beta_map.get('Value', 0):.3f}",
                        "Momentum β": f"{beta_map.get('Momentum', 0):.3f}",
                        "Growth β": f"{beta_map.get('Growth', 0):.3f}",
                    },
                    {
                        "": "Factor Returns",
                        "Total Return": "",
                        "Market β": f"{factor_rets_week.get('Market', 0) * 100:+.3f}%",
                        "Value β": f"{factor_rets_week.get('Value', 0) * 100:+.3f}%",
                        "Momentum β": f"{factor_rets_week.get('Momentum', 0) * 100:+.3f}%",
                        "Growth β": f"{factor_rets_week.get('Growth', 0) * 100:+.3f}%",
                    },
                    {
                        "": "Imputed Return",
                        "Total Return": f"{imputed_total * 100:+.3f}%",
                        "Market β": f"{beta_map.get('Market', 0) * factor_rets_week.get('Market', 0) * 100:+.3f}%",
                        "Value β": f"{beta_map.get('Value', 0) * factor_rets_week.get('Value', 0) * 100:+.3f}%",
                        "Momentum β": f"{beta_map.get('Momentum', 0) * factor_rets_week.get('Momentum', 0) * 100:+.3f}%",
                        "Growth β": f"{beta_map.get('Growth', 0) * factor_rets_week.get('Growth', 0) * 100:+.3f}%",
                    },
                    {
                        "": "Alpha",
                        "Total Return": f"{alpha_residual * 100:+.3f}%",
                        "Market β": "",
                        "Value β": "",
                        "Momentum β": "",
                        "Growth β": "",
                    },
                ]
                abs_df = pd.DataFrame(abs_rows)
                st.markdown("**Absolute Return**")
                st.caption(f"Week ending {last_week.strftime('%b %d, %Y')}")
                components.html(html_table(abs_df, max_height="250px"), height=230, scrolling=True)

                st.markdown("")

                # ── Excess Return Table (vs SPY) ──
                excess_port = port_ret_week - spy_ret_week
                excess_imputed = imputed_total - spy_ret_week
                excess_alpha = excess_port - imputed_total  # same residual, different framing

                excess_rows = [
                    {
                        "": "Sub-Fund (Excess)",
                        "Total Return": f"{excess_port * 100:+.3f}%",
                        "Market β": f"{beta_map.get('Market', 0):.3f}",
                        "Value β": f"{beta_map.get('Value', 0):.3f}",
                        "Momentum β": f"{beta_map.get('Momentum', 0):.3f}",
                        "Growth β": f"{beta_map.get('Growth', 0):.3f}",
                    },
                    {
                        "": "SPY Return",
                        "Total Return": f"{spy_ret_week * 100:+.3f}%",
                        "Market β": "",
                        "Value β": "",
                        "Momentum β": "",
                        "Growth β": "",
                    },
                    {
                        "": "Factor Returns",
                        "Total Return": "",
                        "Market β": f"{factor_rets_week.get('Market', 0) * 100:+.3f}%",
                        "Value β": f"{factor_rets_week.get('Value', 0) * 100:+.3f}%",
                        "Momentum β": f"{factor_rets_week.get('Momentum', 0) * 100:+.3f}%",
                        "Growth β": f"{factor_rets_week.get('Growth', 0) * 100:+.3f}%",
                    },
                    {
                        "": "Imputed Excess Return",
                        "Total Return": f"{excess_imputed * 100:+.3f}%",
                        "Market β": f"{beta_map.get('Market', 0) * factor_rets_week.get('Market', 0) * 100:+.3f}%",
                        "Value β": f"{beta_map.get('Value', 0) * factor_rets_week.get('Value', 0) * 100:+.3f}%",
                        "Momentum β": f"{beta_map.get('Momentum', 0) * factor_rets_week.get('Momentum', 0) * 100:+.3f}%",
                        "Growth β": f"{beta_map.get('Growth', 0) * factor_rets_week.get('Growth', 0) * 100:+.3f}%",
                    },
                    {
                        "": "Alpha",
                        "Total Return": f"{excess_alpha * 100:+.3f}%",
                        "Market β": "",
                        "Value β": "",
                        "Momentum β": "",
                        "Growth β": "",
                    },
                ]
                excess_df = pd.DataFrame(excess_rows)
                st.markdown("**Excess Return (vs SPY)**")
                st.caption(f"Week ending {last_week.strftime('%b %d, %Y')}")
                components.html(html_table(excess_df, max_height="280px"), height=260, scrolling=True)
            else:
                st.info("Insufficient overlapping data for weekly attribution tables.")

        # ── Weekly Factor Attribution ──
        if not rets.empty:
            st.markdown('<div class="section-header">Weekly Factor Return Attribution</div>', unsafe_allow_html=True)
            etf_betas = pf.compute_etf_factor_betas(rets, start_str, end_str)
            weekly_attr = pf.weekly_factor_attribution(rets, etf_betas, start_str, end_str)
            if not weekly_attr.empty:
                st.caption("Factor contributions (%) = β × weekly factor return. Residual = unexplained by factors.")
                # Format all numeric columns to 3dp with % suffix
                display_wa = weekly_attr.copy()
                for col in display_wa.columns:
                    if col != "Week Ending":
                        display_wa[col] = display_wa[col].apply(lambda x: f"{x:+.3f}%")
                components.html(html_table(display_wa, max_height="400px"), height=min(450, 40 * len(display_wa) + 55), scrolling=True)
            else:
                st.info("Insufficient data for weekly attribution.")
            st.markdown("")

        # ── Weekly Theme Attribution ──
        if name == "Thematic":
            st.markdown('<div class="section-header">Weekly Theme Attribution</div>', unsafe_allow_html=True)
            theme_map = pf.load_theme_map()
            weekly_theme = pf.weekly_theme_attribution(rets, holdings, theme_map, start_str, end_str)
            if not weekly_theme.empty:
                st.caption("Theme contributions (%) = Σ(weight × asset weekly return) / total return, computed for each theme.")

                # Pie chart of last week's theme attribution
                col_pie, col_table = st.columns([1, 2])
                with col_pie:
                    theme_pie = make_theme_attribution_bar(weekly_theme)
                    st.plotly_chart(theme_pie, use_container_width=True)

                with col_table:
                    display_wt = weekly_theme.copy()
                    for col in display_wt.columns:
                        if col != "Week Ending":
                            display_wt[col] = display_wt[col].apply(lambda x: f"{x:+.3f}%")
                    components.html(
                        html_table(display_wt, max_height="400px"),
                        height=min(450, 40 * len(display_wt) + 55),
                        scrolling=True,
                    )
            else:
                st.info("Insufficient data for weekly theme attribution.")
            st.markdown("")

        # ── Individual Stock Factor Exposure ──
        if not holdings.empty and not rets.empty:
            st.markdown('<div class="section-header">Stock Factor Explorer</div>', unsafe_allow_html=True)
            available_tickers = sorted(holdings["Ticker"].tolist())
            selected_stocks = st.multiselect(
                "Select stocks to analyze",
                available_tickers,
                default=available_tickers[:3] if len(available_tickers) >= 3 else available_tickers,
                key=f"stock_factor_{name}",
            )
            if selected_stocks:
                stock_prices = pf.fetch_prices(selected_stocks, start_str, end_str)
                if not stock_prices.empty:
                    ff_rows = []
                    etf_rows = []
                    for ticker in selected_stocks:
                        if ticker not in stock_prices.columns:
                            continue
                        stock_rets = stock_prices[ticker].pct_change().dropna()
                        if len(stock_rets) < 10:
                            continue
                        ff = pf.compute_factor_betas(stock_rets, pd.DataFrame([{"Ticker": ticker, "Weight (%)": 100.0}]), start_str, end_str)
                        etf = pf.compute_etf_factor_betas(stock_rets, start_str, end_str)

                        def _build_row(result, ticker):
                            if not result:
                                return None
                            alpha = result.pop("_alpha", 0.0)
                            idio = result.pop("_idio_vol", 0.0)
                            result.pop("_r_squared", None)
                            result.pop("_alpha_t", None)
                            result.pop("_alpha_p", None)
                            result.pop("_stats", None)
                            factors = {k: v for k, v in result.items() if not k.startswith("_")}
                            row = {"Ticker": ticker}
                            for fname, bval in factors.items():
                                row[fname] = bval
                            row["Alpha (Ann.)"] = round(alpha * 100, 3)
                            row["Idio. Vol (Ann.)"] = round(idio * 100, 3)
                            return row

                        ff_row = _build_row(ff, ticker)
                        etf_row = _build_row(etf, ticker)
                        if ff_row:
                            ff_rows.append(ff_row)
                        if etf_row:
                            etf_rows.append(etf_row)

                    if ff_rows:
                        st.markdown("**Fama-French Factors**")
                        ff_df = pd.DataFrame(ff_rows)
                        components.html(html_table(ff_df, max_height="300px"), height=min(350, 40 * len(ff_df) + 55), scrolling=True)
                    if etf_rows:
                        st.markdown("")
                        st.markdown("**ETF Proxy Factors**")
                        etf_df = pd.DataFrame(etf_rows)
                        components.html(html_table(etf_df, max_height="300px"), height=min(350, 40 * len(etf_df) + 55), scrolling=True)
                    if not ff_rows and not etf_rows:
                        st.info("Not enough data to compute factor betas for selected stocks.")
                else:
                    st.info("Could not fetch price data for selected stocks.")

        # ── Dividends ──
        if dividends:
            st.markdown('<div class="section-header">Dividend & Fee History</div>', unsafe_allow_html=True)
            div_df = pd.DataFrame(dividends, columns=["Date", "Ticker", "Amount ($)"])
            div_df = div_df.sort_values("Date", ascending=False)
            div_df["Date"] = div_df["Date"].dt.strftime("%Y-%m-%d")
            total_div = div_df["Amount ($)"].sum()
            st.caption(f"Total dividend income: **${total_div:,.3f}**")
            div_df["Amount ($)"] = div_df["Amount ($)"].apply(lambda x: f"${x:,.3f}")
            components.html(html_table(div_df, max_height="250px"), height=min(300, 40 * len(div_df) + 55), scrolling=True)

        # ── Bond Price Entry (Fixed Income only) ──
        if name == "Fixed Income":
            # Find open CUSIP positions
            cusip_holdings = [t for t in (holdings["Ticker"].tolist() if not holdings.empty else []) if pf._is_cusip(t)]
            if cusip_holdings:
                st.markdown('<div class="section-header">Corporate Bond Prices</div>', unsafe_allow_html=True)
                st.caption("Enter current bond prices and coupon details. Accrued interest is calculated automatically.")

                existing = pf.load_bond_prices_full()
                bond_data = dict(existing)

                for cusip in cusip_holdings:
                    info = existing.get(cusip, {})
                    desc = info.get("description", cusip)
                    current_price = info.get("current_price_per_100", 100.0)
                    purchase_price = info.get("purchase_price_per_100", 100.0)
                    face_value = info.get("face_value", 0.0)
                    coupon_rate = info.get("coupon_rate", 0.0)
                    purchase_date = info.get("purchase_date", "")

                    st.markdown(f"**{cusip}** — {desc}")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        new_price = st.number_input(
                            "Current Price (per $100)",
                            value=float(current_price),
                            min_value=0.0, max_value=200.0, step=0.01,
                            key=f"bond_price_{cusip}",
                        )
                    with c2:
                        new_purchase = st.number_input(
                            "Purchase Price (per $100)",
                            value=float(purchase_price),
                            min_value=0.0, max_value=200.0, step=0.01,
                            key=f"bond_purchase_{cusip}",
                        )
                    with c3:
                        new_face = st.number_input(
                            "Face Value ($)",
                            value=float(face_value),
                            min_value=0.0, step=1000.0,
                            key=f"bond_face_{cusip}",
                        )
                    with c4:
                        new_coupon = st.number_input(
                            "Coupon Rate (%)",
                            value=float(coupon_rate * 100),
                            min_value=0.0, max_value=20.0, step=0.125,
                            key=f"bond_coupon_{cusip}",
                        )
                    with c5:
                        new_date = st.text_input(
                            "Purchase Date",
                            value=purchase_date,
                            placeholder="YYYY-MM-DD",
                            key=f"bond_date_{cusip}",
                        )

                    bond_data[cusip] = {
                        "description": desc,
                        "purchase_price_per_100": new_purchase,
                        "current_price_per_100": new_price,
                        "price_ratio": new_price / new_purchase if new_purchase > 0 else 1.0,
                        "face_value": new_face,
                        "coupon_rate": new_coupon / 100,
                        "purchase_date": new_date,
                    }

                if st.button("Update Bond Prices", key="update_bond_prices"):
                    # Save all entries
                    for cusip in cusip_holdings:
                        if cusip not in bond_data:
                            info = existing.get(cusip, {})
                            bond_data[cusip] = info if info else {
                                "description": cusip,
                                "purchase_price_per_100": 100.0,
                                "current_price_per_100": 100.0,
                                "price_ratio": 1.0,
                            }
                    pf.save_bond_prices(bond_data)
                    st.success("Bond prices updated. Refresh the page to see updated valuations.")
                    st.cache_data.clear()

# ═══════════════════════════════════════════════════════════════════════════
#  UPLOAD TAB
# ═══════════════════════════════════════════════════════════════════════════
SUBFUND_FILE_MAP = {
    "Systematic": "systematic.csv",
    "Opportunistic": "opportunistic.csv",
    "Thematic": "thematic.csv",
    "Fixed Income": "fixed_income.csv",
}

with tabs[-1]:
    st.markdown('<div class="section-header">Upload Transaction Report</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload your **Fidelity CSV export** to update a sub-fund's transaction history. "
        "The file will replace the existing data for that sub-fund."
    )

    upload_fund = st.selectbox("Select Sub-Fund", list(SUBFUND_FILE_MAP.keys()), key="upload_fund")
    uploaded_file = st.file_uploader(
        "Upload Fidelity CSV", type=["csv"], key="upload_csv",
        help="Export your transaction history from Fidelity and upload the CSV here.",
    )

    if uploaded_file is not None:
        try:
            # Preview the upload
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # Reset for saving
            st.markdown(f"**Preview** — {len(preview_df)} rows, {len(preview_df.columns)} columns")
            st.dataframe(preview_df.head(10), use_container_width=True, hide_index=True, height=300)

            if st.button("✅ Save & Update", key="upload_save", type="primary"):
                target_path = DATA_DIR / SUBFUND_FILE_MAP[upload_fund]
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(uploaded_file.read())
                # Clear caches so the new data is picked up
                st.cache_data.clear()
                st.success(f"✅ **{upload_fund}** updated successfully! Refreshing…")
                st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")


# ── Footer ────────────────────────────────────────────────────────────────
_logo_path = Path("assets/nyu_stern_logo.png")
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; padding: 1.5rem 0 2rem 0;">
        <img src="data:image/png;base64,{_logo_b64}" alt="NYU Stern" style="height:45px; margin-bottom:0.5rem;" />
        <div style="font-family:'Helvetica Neue',Helvetica,Arial,sans-serif; font-weight:300; font-size:0.75rem; color:#6B7280;">
            &copy; 2026 NYU Michael Price Student Investment Fund | Contact willwu@stern.nyu.edu for technical help
        </div>
    </div>
    """, unsafe_allow_html=True)
