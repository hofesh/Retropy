import os.path
import pandas as pd
from framework.utils import *
from framework.meta_data_dfs import *
from framework.symbol import *

def is_etf(s):
    return get_ticker_name(s) in etf_metadata_df.index

def is_cef(s):
    return get_ticker_name(s) in cef_metadata_df.index

def get_meta(s, fld, defval=0, silent=False):
    if is_etf(s):
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

def get_meta_fee(s):
    if is_cef(s):
        return get_meta(s, "expense_ratio")
    if is_etf(s):
        fee = get_meta(s, "fees")
        if pd.isnull(fee):
            fee = get_meta(s, "mw_fees")
            if pd.isnull(fee):
                warn(f"no fee data found for {ticker} in ETF metadata")
                return 0
            else:
                warn(f"only 'mw_fees' but no 'fees' for {ticker} in ETF metadata")
        return fee
    warn(f"{get_name(s)} is not an ETF and not a CEF, can't get fee")
    return 0

def get_etf_cef_meta(s, etf_fld, cef_fld):
    if is_cef(s):
        return get_meta(s, cef_fld)
    if is_etf(s):
        return get_meta(s, etf_fld)
    warn(f"{get_name(s)} is not an ETF and not a CEF, can't get {fld}")
    return 0

def get_cef_meta(s, fld):
    if is_cef(s):
        return get_meta(s, fld)
    warn(f"{get_name(s)} is not an CEF, can't get {fld}")
    return 0

def get_meta_aum(s):
    return get_etf_cef_meta(s, "aum", "net_aum")
