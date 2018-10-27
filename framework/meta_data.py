import os.path
import pandas as pd
from framework.utils import *
from framework.meta_data_dfs import *
from framework.symbol import *

def is_etf(s):
    return get_ticker_name(s) in etf_metadata_df.index

def is_cef(s):
    return get_ticker_name(s) in cef_metadata_df.index

def get_meta(s, fld, defval=0, silent=False, cef_only=False):
    if is_etf(s) and not cef_only:
        return _get_meta(s, fld, etf_metadata_df, "ETF", defval=defval, silent=silent)
    if is_cef(s):
        return _get_meta(s, fld, cef_metadata_df, "CEF", defval=defval, silent=silent)
    warn(f"{get_name(s)} is not an ETF and not a CEF, can't get {fld}")
    return defval

def _get_meta(s, fld, df, type, defval=0, silent=False):
    if df is None:
        return defval
    ticker = get_ticker_name(s)
    try:
        data = df.loc[ticker]
    except:
        if not silent:
            warn(f"no {type} metadata for {ticker}")
        return defval
    val = data[fld]
    if pd.isnull(val):
        if not silent:
            warn(f"no {fld} data found for {ticker} in {type} metadata")
        return defval
    return val

def get_etf_cef_meta(s, etf_fld, cef_fld, etf_alt_fld=None, defval=0):
    ticker = get_ticker_name(s)
    if is_etf(s):
        res = get_meta(s, etf_fld, defval=None, silent=True)
        if res is None and etf_alt_fld:
            res = get_meta(s, etf_alt_fld, silent=True)
            if res is None:
                warn(f"no {etf_fld} data found for {ticker} in ETF metadata")
                return defval
            warn(f"only '{etf_alt_fld}' but no '{etf_fld}' for {ticker} in ETF metadata")
        return defval if res is None else res
    if is_cef(s):
        res = get_meta(s, cef_fld)
        return defval if res is None else res
    warn(f"{ticker} is not an ETF and not a CEF, can't get {etf_fld}")
    return 0

def get_cef_meta(s, fld):
    if is_cef(s):
        return get_meta(s, fld, cef_only=True)
    #warn(f"{get_name(s)} is not an CEF, can't get {fld}")
    return None # was zero

def get_meta_fee(s):
    return get_etf_cef_meta(s, 'fees', 'expense_ratio', etf_alt_fld='mw_fees')

def get_meta_aum(s):
    return get_etf_cef_meta(s, "aum", "net_aum", etf_alt_fld="mw_aum")

def get_meta_yield(s, net=True):
    yld = get_etf_cef_meta(s, "yc_yield_ttm", "market_yield", etf_alt_fld="mw_yield")
    if net:
        yld *= 0.75
    return yld
