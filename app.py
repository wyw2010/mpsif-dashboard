"""
NYU MPSIF Return Attribution Dashboard
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
from pathlib import Path
from PIL import Image
import portfolio as pf

# ── Page config ────────────────────────────────────────────────────────────
_logo = Image.open(Path("assets/nyu_stern_logo.png"))
st.set_page_config(
    page_title="NYU MPSIF",
    page_icon=_logo,
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
    /* ── Typography ── */
    * {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-weight: 300;
    }}
    h1, h2, h3, h4, h5, h6,
    .metric-label, .metric-value, .section-header, .logo,
    .stTabs [data-baseweb="tab"],
    th, thead {{
        font-weight: 500 !important;
    }}

    .main .block-container {{ padding-top: 1.5rem; max-width: 1200px; }}

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

    return f"""<html><head><style>
    * {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 300; margin: 0; padding: 0; }}
    .clean-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
    .clean-table thead th {{ background:{NYU_PURPLE_BG}; color:{BLACK}; font-weight:500; font-size:0.8rem;
        padding:0.6rem 0.75rem; text-align:left; border-bottom:2px solid #E5E7EB; cursor:pointer; user-select:none; }}
    .clean-table tbody td {{ padding:0.5rem 0.75rem; border-bottom:1px solid #F3F4F6; font-weight:300; }}
    .clean-table tbody tr:hover {{ background:#FAFAFA; }}
    </style></head><body>
    <div style="max-height:{max_height}; overflow-y:auto;">
    <table class="clean-table" id="{tid}"><thead><tr>{header}</tr></thead>
    <tbody>{rows}</tbody></table></div>{script}
    </body></html>"""


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
        hovertemplate="Portfolio<br>%{x|%b %d, %Y}: %{y:+.3f}%<extra></extra>",
    ))
    if bench_rets is not None and not bench_rets.empty:
        bench_aligned = bench_rets.reindex(rets.index).dropna()
        if not bench_aligned.empty:
            bench_cum = pf.cum_return(bench_aligned) * 100
            fig.add_trace(go.Scatter(
                x=bench_cum.index, y=bench_cum.values, mode="lines",
                name=bench_label,
                line=dict(color="#F59E0B", width=2, dash="dash"),
                hovertemplate=f"{bench_label}<br>%{{x|%b %d, %Y}}: %{{y:+.3f}}%<extra></extra>",
            ))
    show_legend = bench_rets is not None and not bench_rets.empty
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Cumulative Return (%)",
                   tickformat="+.3f", zeroline=True, zerolinecolor="#E5E7EB"),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif"),
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
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Drawdown (%)", tickformat=".3f"),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif"),
    )
    return fig


def make_multi_fund_chart(fund_data: dict, visible: list, height: int = 400):
    colors = {
        "Total Fund": BLACK,
        "Systematic": NYU_PURPLE,
        "Opportunistic": "#2563EB",
        "Thematic": "#D97706",
        "Fixed Income": "#059669",
        "Benchmark": "#F59E0B",
    }
    fig = go.Figure()
    for name, rets in fund_data.items():
        if rets is None or rets.empty:
            continue
        cum = pf.cum_return(rets) * 100
        line_width = 2.5 if name == "Total Fund" else 2
        line_dash = "dash" if name == "Benchmark" else None
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines", name=name,
            line=dict(color=colors.get(name, GRAY), width=line_width, dash=line_dash),
            visible=True if name in visible else "legendonly",
            hovertemplate=f"{name}<br>%{{x|%b %d}}: %{{y:+.3f}}%<extra></extra>",
        ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Cumulative Return (%)",
                   tickformat="+.3f", zeroline=True, zerolinecolor="#E5E7EB"),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif"),
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
        xaxis=dict(title="Contribution (bps)", showgrid=True, gridcolor="#F3F4F6"),
        yaxis=dict(showgrid=False),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif"),
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
        paper_bgcolor=WHITE, font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif"),
        showlegend=False,
    )
    return fig


# ── Data loading ───────────────────────────────────────────────────────────
DATA_DIR = Path("data")
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
    all_vals = {}
    for name, d in subfund_data.items():
        pv = d["portfolio_values"]["Total"]
        all_vals[name] = pv
    if not all_vals:
        return pd.Series(dtype=float)
    combined = pd.DataFrame(all_vals).sort_index().ffill().fillna(0)
    total = combined.sum(axis=1)
    total = total[total > 0]
    return pf.daily_returns(total)


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
    _bench_start = (min(_all_starts) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
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
tab_names = ["Overview"] + pf.SUBFUNDS
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
                st.markdown(metric_card("Sharpe Ratio", f"{pf.sharpe(combined_rets):.3f}"), unsafe_allow_html=True)
            with mcols[2]:
                st.markdown(metric_card("Max Drawdown", fmt_pct(pf.max_dd(combined_rets) * 100), color_class(pf.max_dd(combined_rets))), unsafe_allow_html=True)
            with mcols[3]:
                st.markdown(metric_card("Ann. Volatility", fmt_pct(pf.ann_vol(combined_rets) * 100)), unsafe_allow_html=True)
            st.markdown("")

            # ── Blended benchmark metrics ──
            if not blended_bench_rets.empty:
                st.markdown('<div class="section-header">vs. Blended Benchmark (50% SPY + 25% IWV + 25% AGG)</div>', unsafe_allow_html=True)
                er = pf.excess_returns(combined_rets, blended_bench_rets)
                bcols = st.columns(4)
                with bcols[0]:
                    excess_total = pf.total_ret(er)
                    st.markdown(metric_card("Excess Return", fmt_pct(excess_total * 100), color_class(excess_total)), unsafe_allow_html=True)
                with bcols[1]:
                    a = pf.alpha_jensen(combined_rets, blended_bench_rets)
                    st.markdown(metric_card("Alpha (Ann.)", fmt_pct(a * 100), color_class(a)), unsafe_allow_html=True)
                with bcols[2]:
                    b = pf.beta(combined_rets, blended_bench_rets)
                    st.markdown(metric_card("Beta", f"{b:.3f}"), unsafe_allow_html=True)
                with bcols[3]:
                    ir = pf.information_ratio(combined_rets, blended_bench_rets)
                    st.markdown(metric_card("Info Ratio", f"{ir:.3f}"), unsafe_allow_html=True)
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
            st.plotly_chart(make_multi_fund_chart(fund_rets, visible_funds), use_container_width=True)

            # ── Drawdown ──
            st.markdown('<div class="section-header">Drawdown</div>', unsafe_allow_html=True)
            st.plotly_chart(make_drawdown_chart(combined_rets), use_container_width=True)

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
            st.markdown(metric_card("Sharpe", f"{pf.sharpe(rets):.3f}"), unsafe_allow_html=True)
        with mcols[2]:
            st.markdown(metric_card("Max DD", fmt_pct(pf.max_dd(rets) * 100), color_class(pf.max_dd(rets))), unsafe_allow_html=True)
        with mcols[3]:
            st.markdown(metric_card("Ann. Vol", fmt_pct(pf.ann_vol(rets) * 100)), unsafe_allow_html=True)

        # ── Benchmark metrics ──
        if not bench_r.empty:
            st.markdown(f'<div class="section-header">vs. {bench_ticker} Benchmark</div>', unsafe_allow_html=True)
            er = pf.excess_returns(rets, bench_r)
            bcols = st.columns(4)
            with bcols[0]:
                excess_total = pf.total_ret(er)
                st.markdown(metric_card("Excess Return", fmt_pct(excess_total * 100), color_class(excess_total)), unsafe_allow_html=True)
            with bcols[1]:
                a = pf.alpha_jensen(rets, bench_r)
                st.markdown(metric_card("Alpha (Ann.)", fmt_pct(a * 100), color_class(a)), unsafe_allow_html=True)
            with bcols[2]:
                b = pf.beta(rets, bench_r)
                st.markdown(metric_card("Beta", f"{b:.3f}"), unsafe_allow_html=True)
            with bcols[3]:
                ir = pf.information_ratio(rets, bench_r)
                st.markdown(metric_card("Info Ratio", f"{ir:.3f}"), unsafe_allow_html=True)

        st.markdown("")

        # ── Cumulative return chart ──
        st.markdown(f'<div class="section-header">{name} — Cumulative Return</div>', unsafe_allow_html=True)
        chart_period = st.segmented_control(
            "Chart period", ["1W", "1M", "YTD", "All"],
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

        # Align benchmark to chart period
        chart_bench = bench_r
        if not bench_r.empty and not chart_rets.empty:
            chart_bench = bench_r[bench_r.index.isin(chart_rets.index)]

        st.plotly_chart(
            make_return_chart(chart_rets, bench_rets=chart_bench, bench_label=bench_ticker),
            use_container_width=True,
        )

        # ── Drawdown ──
        st.markdown('<div class="section-header">Drawdown</div>', unsafe_allow_html=True)
        st.plotly_chart(make_drawdown_chart(rets, height=180), use_container_width=True)

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
                st.plotly_chart(make_holdings_pie(holdings), use_container_width=True)
            with hcol2:
                components.html(html_table(tbl, raw_df=raw_tbl, max_height="450px",
                                       default_sort="Weight (%)", default_asc=False), height=min(500, 40 * len(tbl) + 55), scrolling=True)
        else:
            st.info("No current holdings.")

        # ── Factor Exposure ──
        if not rets.empty:
            st.markdown('<div class="section-header">Factor Exposure</div>', unsafe_allow_html=True)
            start_str = (d["first_date"] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            end_str = (d["end_date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            factor_betas = pf.compute_factor_betas(rets, start_str, end_str)
            fcols = st.columns(len(factor_betas))
            for i, (factor_name, beta_val) in enumerate(factor_betas.items()):
                with fcols[i]:
                    st.markdown(metric_card(factor_name, f"{beta_val:.3f}", color_class(beta_val)), unsafe_allow_html=True)

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

# ── Footer ────────────────────────────────────────────────────────────────
_logo_path = Path("assets/nyu_stern_logo.png")
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; padding: 1.5rem 0 2rem 0;">
        <img src="data:image/png;base64,{_logo_b64}" alt="NYU Stern" style="height:45px; margin-bottom:0.5rem;" />
        <div style="font-family:'Helvetica Neue',Helvetica,Arial,sans-serif; font-weight:300; font-size:0.75rem; color:#6B7280;">
            &copy; 2026 NYU Michael Price Student Investment Fund | Will Wu (willwu@stern.nyu.edu)
        </div>
    </div>
    """, unsafe_allow_html=True)
