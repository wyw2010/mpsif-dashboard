import pandas as pd
import numpy as np
import wrds
import datetime
import os

####################################################################
# CONSTANTS & DATA LOADING FUNCTIONS

# for cron log purposes:
print(f"%%% {datetime.datetime.now()} : FACTOR_BUILDER SCRIPT STARTED %%%")

now = datetime.datetime.now()
DEFAULT_STARTING_DATE = (now - datetime.timedelta(days=3*366)).strftime('%Y-%m-%d')

# FILL IN WITH YOUR OWN USERNAME 
CONN = wrds.Connection()

# for now, based on current sp500 constituents; need to coordinate with CRSP data to make fuller list. comprehensive list of inclusion/exclusion in S&P500 is not fully available.
"""
@param sp500: boolean indicating if data should be pulled for entire set of sp500 tickers
@param starting_date: string type for the starting data for data being pulled
@param tickers: list object of strings of tickers
@param db: wrds Connection object so as not to overload the wrds system
@return: full pandas DataFrame with daily returns in the period indicated for all stocks indicated; not easily comprehended without filters/pivoting
"""
def load_returns(sp500 = True, starting_date = DEFAULT_STARTING_DATE, tickers=None, db=None):
    if db is None:
        db = wrds.Connection()
    
    if sp500:
        gvkeys = load_sp500_gvkey(starting_date=starting_date, db=db).columns.tolist()
    elif tickers is not None:
        names = get_name_map(db)
        gvkeys = names[names['tic'].isin(tickers)]['gvkey']
        gvkeys = gvkeys.tolist()
    else:
        print("NEED INPUT GROUP")
        return None
    
    query = f"select gvkey, iid, tic, prccd, cshtrd, datadate from comp.secd where datadate > '{starting_date}' and gvkey in {tuple(gvkeys)} and iid = '01'"
    df = db.raw_sql(query)
    
    # df = df.drop_duplicates(subset=['gvkey', 'datadate', 'iid'], keep='first')
    
    prices = df.pivot_table(values='prccd',columns='gvkey',index='datadate')
    returns = (prices/prices.shift()).iloc[1:] # --> how to get rid of first row?
    volumes = df.pivot_table(values='cshtrd', columns='gvkey', index='datadate')

    prices.index = pd.to_datetime(prices.index)
    returns.index = pd.to_datetime(returns.index)
    volumes.index = pd.to_datetime(volumes.index)
    
    return prices, returns, volumes

"""
@param sp500: boolean indicating if data should be pulled for entire set of sp500 tickers
@param starting_date: string type for the starting data for data being pulled
@param tickers: list object of strings of tickers
@param db: wrds Connection object so as not to overload the wrds system
@return: full table with all financial data in the period indicated for all stocks indicated; not easily comprehended without filters/pivoting
"""
def load_financial_metrics(sp500 = True, starting_date = DEFAULT_STARTING_DATE, tickers = None, db = None):
    if db is None:
        db = wrds.Connection()
    
    if sp500:
        # full data table including gvkey, from, and thru dates of inclusion in S&P 500
        gvkeys = load_sp500_gvkey(starting_date=starting_date, db=db).columns.tolist()
    elif tickers is not None:
        # uses name map to find the corresponding gvkey, from which financial metrics can be pulled
        names = get_name_map(db)
        gvkeys = names[names['tic'].isin(tickers)]['gvkey']
        gvkeys = gvkeys.tolist()
    else:
        print("NEED INPUT GROUP")
        return None
    
    # pulling data through WRDS --> every data endpoint that we need to compute factors included in this table
    query = "select gvkey, tic, iid, datadate, prccq, cshoq, ceqq, teqq, niq, revtq, oancfy, capxy, ltq, atq, cogsq, xoprq, epspxq, fyearq, fqtr from comp.fundq where gvkey in ("
    query += str(gvkeys)[1:-1]
    query += f") and (datadate > '{starting_date}') order by datadate, gvkey"

    financials = db.raw_sql(query)
    
    # processing data table --> reformatting capex and operating cash flow to reflect quarterly data; renaming columns clearly; adding computed metrics
    financials['capxy_q'] = financials.groupby(['gvkey', 'fyearq'])['capxy'].diff().fillna(financials['capxy'])
    financials['oancfy_q'] = financials.groupby(['gvkey', 'fyearq'])['oancfy'].diff().fillna(financials['oancfy'])
    financials['Price-Earnings Ratio'] = financials['prccq'] * financials['cshoq'] / financials['niq']
    financials['Price-Book Ratio'] = financials['prccq'] * financials['cshoq'] / financials['ceqq']
    financials['Price-Sales Ratio'] = financials['prccq'] * financials['cshoq'] / financials['revtq']
    # financials['Book-Market Ratio'] = (financials['atq']-financials['ltq']) / (financials['cshoq'] * financials['prccq'])
    financials['Book-Market Ratio'] = (financials['ceqq']) / (financials['cshoq'] * financials['prccq'])
    financials = financials.rename(columns={'prccq':'EOQ Closing Price', 'cshoq':'Shares Outstanding','ceqq':'Shareholders Equity Value','niq':'Net Income','revtq':'Revenue','oancfy_q':'Cash Flow From Operations','capxy_q':'CapEx','ltq':'Total Liabilities','atq':'Total Assets','cogsq':'Cost of Goods Sold','xoprq':'Operating Expense'})
    financials = financials.drop(columns={'oancfy','capxy'})
                                 
    return financials

"""
@param db: wrds Connection object so as not to overload the wrds system
@return: pandas DataFrame object that has columns of different naming systems (corporation names, gvkey, tic, iid, cusip, cik, sic) and rows for each corporation
""" 
def get_name_map(db=None):
    if db is None:
        db = wrds.Connection()
    
    query = f"""select * from comp.namesd"""
    
    return db.raw_sql(query)

NAME_MAP = get_name_map(CONN).drop(columns={'cusip','cik','sic','naics','gsubind','gind'})
NAME_MAP = NAME_MAP[NAME_MAP['iid'] == '01']

query = f"""select * from comp.idxcst_his where (thru is null or thru >= '{DEFAULT_STARTING_DATE}') and gvkeyx='000003';"""
sp500_gvkey = CONN.raw_sql(query)
SP500 = sp500_gvkey.merge(NAME_MAP, how='left', left_on='gvkey', right_on='gvkey').drop(columns={'thru', 'iid_y'})

PRICES, RETURNS, VOLUMES = load_returns(sp500=False, starting_date=DEFAULT_STARTING_DATE, tickers = SP500['tic'].tolist(), db=CONN)
RAW_DATA = load_financial_metrics(sp500=False,starting_date=DEFAULT_STARTING_DATE, tickers=SP500['tic'].tolist(), db=CONN)

####################################################################

def build_momentum_factor(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1,
    long_quantile: float = 0.7,
    short_quantile: float = 0.3,
    eval_window_months: int = 6,
):
    """
    Build a cross-sectional 12-1m momentum factor with monthly rebalancing
    and return the daily factor return series over the trailing eval_window_months.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices, index is DatetimeIndex, columns are assets.
    lookback_months : int, default 12
        Number of months in the momentum lookback window (e.g., 12).
    skip_months : int, default 1
        Number of most recent months to skip (e.g., 1 for 12-1).
    long_quantile : float, default 0.9
        Upper quantile threshold for long leg (e.g., 0.9 -> top decile).
    short_quantile : float, default 0.1
        Lower quantile threshold for short leg (e.g., 0.1 -> bottom decile).
    eval_window_months : int, default 6
        Length of trailing evaluation window for factor returns (in months).

    Returns
    -------
    factor_ret_daily : pd.Series
        Daily returns of the momentum factor over the trailing eval_window_months.
    weights_daily : pd.DataFrame
        Daily portfolio weights over the same evaluation window.
    ret_daily : pd.DataFrame
        Daily simple returns of the underlying assets over the same window.
    """

    # 1) Daily simple returns
    ret_daily = prices.pct_change(fill_method=None).dropna(how='all')

    # 2) Monthly prices (month-end)
    prices_m = prices.resample('ME').last()
    ret_m = prices_m.pct_change(fill_method=None)

    # 3) Rebalancing dates: month-ends where we have data
    rebal_dates = prices_m.index

    # 4) Compute cross-sectional 12-1m momentum signal at each rebalance date
    mom_signal_list = []

    for d in rebal_dates:
        # end_month for the signal = d - skip_months
        end_month = d - pd.offsets.MonthEnd(skip_months)
        start_month = end_month - pd.DateOffset(months=lookback_months - 1)

        # make sure we have the entire window
        window = ret_m.loc[start_month:end_month]
        if window.shape[0] < lookback_months:
            # not enough history, skip this rebalance date
            mom_signal_list.append(pd.Series(dtype=float, name=d))
            continue

        # cumulative return over the lookback window
        sig = (1.0 + window).prod() - 1.0
        sig.name = d
        mom_signal_list.append(sig)

    mom_signal = pd.DataFrame(mom_signal_list)
    mom_signal = mom_signal.reindex(index=rebal_dates)

    # 5) Convert signals to monthly weights
    def make_weights_from_signal(sig: pd.Series,
                                 long_q: float,
                                 short_q: float) -> pd.Series:
        sig = sig.dropna()
        if sig.empty:
            return sig

        q_hi = sig.quantile(long_q)
        q_lo = sig.quantile(short_q)

        long_names = sig[sig >= q_hi].index
        short_names = sig[sig <= q_lo].index

        w = pd.Series(0.0, index=sig.index)
        if len(long_names) > 0:
            w.loc[long_names] =  1.0 / len(long_names)
        if len(short_names) > 0:
            w.loc[short_names] = -1.0 / len(short_names)

        # Optional normalization (e.g., target gross=1)
        gross = np.abs(w).sum()
        if gross > 0:
            w = w / gross

        return w

    weights_m_list = []
    for d in mom_signal.index:
        sig = mom_signal.loc[d]
        w = make_weights_from_signal(sig, long_quantile, short_quantile)
        # align to full column set of prices
        w = w.reindex(prices.columns).fillna(0.0)
        w.name = d
        weights_m_list.append(w)

    weights_m = pd.DataFrame(weights_m_list)
    weights_m.index = mom_signal.index

    # 6) Expand monthly weights to daily weights via forward-fill
    #    We assume weights decided at month-end d are applied from
    #    the first trading day AFTER d until the next rebalance.
    #    A simple way: reindex to daily and ffill.
    weights_daily = weights_m.reindex(ret_daily.index, method='ffill').fillna(0.0)

    # 7) Restrict to trailing evaluation window
    last_day = ret_daily.index.max()
    start_eval = last_day - pd.DateOffset(months=eval_window_months)

    mask_eval = (ret_daily.index >= start_eval)
    ret_eval = ret_daily.loc[mask_eval]
    weights_eval = weights_daily.loc[mask_eval]

    # 8) Factor daily returns = row-wise dot product
    factor_ret_daily = (weights_eval * ret_eval).sum(axis=1)

    return factor_ret_daily, weights_eval, ret_eval

def build_eps_growth_factor(
    prices: pd.DataFrame,
    fund_raw: pd.DataFrame,
    id_col: str = "gvkey",
    date_col: str = "datadate",
    eps_col: str = "epspxq",
    long_quantile: float = 0.7,
    short_quantile: float = 0.3,
    eval_window_months: int = 6
):
    """
    Build EPS growth factor with quarterly rebalancing using lagged fundamentals.
    """
    # Daily returns
    ret_daily = prices.pct_change(fill_method=None).dropna(how='all')
    
    # Prepare fundamentals
    fund = fund_raw[[id_col, date_col, eps_col]].copy()
    fund[date_col] = pd.to_datetime(fund[date_col])
    fund[eps_col] = pd.to_numeric(fund[eps_col], errors="coerce")
    fund = fund.dropna()

    # Pivot to get EPS by ticker and quarter
    eps_q = fund.pivot_table(index=date_col, columns=id_col, values=eps_col, aggfunc="last").sort_index().ffill()
    eps_q = eps_q[eps_q.index.month.isin([3, 6, 9, 12])].dropna(axis=1)

    # Calculate TTM EPS (trailing 4-quarter sum), then YoY growth
    eps_ttm = eps_q.rolling(4, min_periods=4).sum()
    eps_growth_q = eps_ttm.pct_change(4).dropna()

    # Quarterly rebalance dates from prices
    prices_q = prices.resample('QE').last()

    rebal_dates = prices_q.index
    
    # transform to weights
    ranking_q = eps_growth_q.rank(axis=1, ascending=True) # LONG 500, 499, 498..., SHORT 1, 2, 3, ...
    
    n = ranking_q.shape[1]
    q = int(n * short_quantile)

    weights_q = (
        (ranking_q >= n - q + 1).astype(float) / q
        - (ranking_q <= q).astype(float) / q
    )

    # shift to account for data availability lag
    weights_q = weights_q.shift(2)
    
    # Expand to daily weights
    weights_daily = weights_q.reindex(ret_daily.index, method='ffill').fillna(0.0)

    last_day = ret_daily.index.max()
    start_eval = last_day - pd.DateOffset(months=eval_window_months)

    mask_eval = (ret_daily.index >= start_eval)
    ret_eval = ret_daily.loc[mask_eval]
    weights_eval = weights_daily.loc[mask_eval]
    
    # Calculate daily factor returns
    factor_ret_daily = (weights_eval * ret_eval).fillna(0).sum(axis=1)
    
    return factor_ret_daily, weights_daily, ret_daily

def build_value_factor(
    prices: pd.DataFrame,
    fund_raw: pd.DataFrame,
    id_col: str = "gvkey",
    date_col: str = "datadate",
    val_col: str = "Book-Market Ratio",
    long_quantile: float = 0.7,
    short_quantile: float = 0.3,
    eval_window_months: int = 6
):
    """
    Build value factor with quarterly rebalancing using lagged fundamentals.
    """
    # Daily returns
    ret_daily = prices.pct_change(fill_method=None).dropna(how='all')
    
    # Prepare fundamentals
    fund = fund_raw[[id_col, date_col, val_col]].copy()
    fund[date_col] = pd.to_datetime(fund[date_col])
    fund[val_col] = pd.to_numeric(fund[val_col], errors="coerce")
    fund = fund.dropna()
    
    # Pivot to get EPS by ticker and quarter
    val_q = fund.pivot_table(index=date_col, columns=id_col, values=val_col, aggfunc="last").sort_index().ffill()
    val_q = val_q[val_q.index.month.isin([3, 6, 9, 12])]

    # Quarterly rebalance dates from prices
    prices_q = prices.resample('QE').last()

    rebal_dates = prices_q.index
    
    # transform to weights
    ranking_q = val_q.rank(axis=1, ascending=True) # LONG 500, 499, 498..., SHORT 1, 2, 3, ...
    
    n = ranking_q.shape[1]
    q = int(n * short_quantile)

    weights_q = (
        (ranking_q >= n - q + 1).astype(float) / q
        - (ranking_q <= q).astype(float) / q
    )
    
    # shift to account for data availability lag
    weights_q = weights_q.shift(2)
    
    # Expand to daily weights
    weights_daily = weights_q.reindex(ret_daily.index, method='ffill').fillna(0.0)

    last_day = ret_daily.index.max()
    start_eval = last_day - pd.DateOffset(months=eval_window_months)

    mask_eval = (ret_daily.index >= start_eval)
    ret_eval = ret_daily.loc[mask_eval]
    weights_eval = weights_daily.loc[mask_eval]
    
    # Calculate daily factor returns
    factor_ret_daily = (weights_eval * ret_eval).fillna(0).sum(axis=1)
    
    return factor_ret_daily, weights_daily, ret_daily

def save_factors():
    momentum = build_momentum_factor(prices=PRICES, eval_window_months=12)[0].to_frame('momentum')
    growth = build_eps_growth_factor(prices=PRICES, fund_raw=RAW_DATA, eval_window_months=12)[0].to_frame('growth')
    value = build_value_factor(prices=PRICES, fund_raw=RAW_DATA, eval_window_months=12)[0].to_frame('value')
    prices_mkt, returns_mkt, volumes_mkt = load_returns(sp500=False, starting_date='2023-06-01', tickers=['SPY','IVE'], db=CONN)

    returns_mkt = returns_mkt.rename(columns={'108132':'mkt'})
    returns_mkt = returns_mkt.drop(columns={'136544'})
    returns_mkt['mkt'] = returns_mkt['mkt']-1

    factors = momentum.merge(growth, how='inner', left_on='datadate', right_on='datadate')
    factors = factors.merge(value, how='inner', left_on='datadate', right_on='datadate')
    factors = factors.merge(returns_mkt, how='inner', left_on='datadate', right_index=True)

    factors.to_pickle('data.pk')
    factors.to_csv('data.csv')

if __name__ == '__main__':
    save_factors()

    # os.system("git add data/data.pk")
    # os.system("git add data/data.csv")
    # os.system("git add cron.log")
    # os.system('git commit -m "auto update data"')
    # os.system("git push")
    
    CONN.close()
