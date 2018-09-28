from scipy.stats import zscore, zmap
import numpy as np
import math

from framework.utils import *
from framework.symbol import *
from framework.base import *
from framework.yields import *
from framework.draw_downs import *
from framework.lr import *


def ma(s, n):
    n = int(n)
    return wrap(s.rolling(n).mean(), "ma({}, {})".format(s.name, n))

def mm(s, n):
    n = int(n)
    return wrap(s.rolling(n).median(), "mm({}, {})".format(s.name, n))

def mmax(s, n):
    n = int(n)
    return wrap(s.rolling(n).max(), "mmax({}, {})".format(s.name, n))

def mmin(s, n):
    n = int(n)
    return wrap(s.rolling(n).min(), "mmin({}, {})".format(s.name, n))

############### risk / return metrics ###############
def max_dd(s):
    return max(-dd(s))

def max_dd_pr(s):
    return max(-dd(pr(s)))

    
# std of monthly returns
def stdmret(s):
    return ret(s).std()*math.sqrt(12)*100

def pr_beta(s):
    return lr_beta(price(s))

def pr_cagr(s):
    return cagr(price(s))

def pr_lr_cagr(s):
    x = lr(price(s))
    x = x[x>0] # we won't be able to calc cagr for negative values
    return cagr(x)

def pr_cagr_full(s):
    return cagr(get(s, untrim=True, mode="PR"))

def lrretm_beta_SPY(s):
    return lrret_beta(s, 'SPY', freq="M")




def reject_outliers(data, m = 2.):
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

# zscore with mean and std calculated from the source data cleaned from outliers
def zscore_modified(vals):
    if len(vals) > 0 and isinstance(vals[0], str):
        return np.zeros(len(vals))
    vals_no_outliers = reject_outliers(np.array(vals)).flatten()
    return zmap(vals, vals_no_outliers)
    

# NOTE: this method rewards USD-bonds and they express lower risk-volatility
def zscores(all, *funcs, weights=None):
    if weights is None:
        weights = [1] * len(funcs)
    d_vals = {get_func_name(f): lmap(f, all) for f in funcs}
    d = dict(d_vals)
    d["name"] = names(all)
    df = pd.DataFrame(d).set_index("name")
    
    df = zscores_update(df, weights)
    return df

    # d_zs = {k: zscore_modified(v)*sign for (k, v), sign in zip(d_vals.items(), weights)}
    # d_zs["name"] = names(all)
    # df_z = pd.DataFrame(d_zs).set_index("name")

    # zmean = df_z.mean(axis=1)
    # df["zmean"] = zmean
    # df = df.sort_values("zmean", ascending=False)
    # return df

def zscores_update(df, weights):
    def apply_w(v, w):
        if isinstance(w, tuple):
            w, f = w
            if f == "log":
                return np.log(v) * w
            raise Exception(f"unsupported weight: {f}")
        if is_number(w):
            return v * w
        if w == "log":
            return np.log(v)
        raise Exception(f"unsupported weight: {w}")

    if "zmean" in df.columns:
        df = df.drop("zmean", axis=1)
    d_zs = {k: apply_w(zscore_modified(df[k]), w) for k, w in zip(df.columns, weights)}
    d_zs["name"] = df.index
    df_z = pd.DataFrame(d_zs).set_index("name")

    zmean = df_z.sum(axis=1) / df_z.shape[1]
    df["zmean"] = zmean
    df = df.sort_values("zmean", ascending=False)

    return df

def zscore_df_style(df, names):
    df.columns = names
    df.index = df.index.str.replace(" NTR", "")

    import seaborn as sns
    cm = sns.light_palette("orange", as_cmap=True)
    #df.style.background_gradient(cmap=cm)
    
    #df.style.highlight_max(axis=0, color='green').highlight_min(axis=0, color='red')
    
    #df.style.bar(subset=['cagr', 'get_curr_yield_min2', 'mutual_dd_rolling_pr_SPY', 'ulcer_pr', 'get_meta_aum_log', 'get_meta_fee'], align='left', color=['#5fba7d'])
    return df.style\
        .bar(subset=['cagr', 'start_yield', 'curr_yield'], align='left', color=['#5fba7d'])\
        .bar(subset=['ulcer_pr'], align='left', color=['#d65f5f'])\
        .bar(subset=['mutual_dd'], align='mid', color=['#5fba7d', '#d65f5f'])\
        .bar(subset=['aum'], align='left', color=['#9fdfbe'])\
        .bar(subset=['fee'], align='left', color=['#ffb3b3'])\
        .format({'aum': "{:,.0f}"})

def display_zscores(all, n=50, funcs=None, names=None, weights=None, _cache=[None]):
    if funcs is None:
        funcs = [cagr, get_start_yield, get_curr_yield_min2, ulcer_pr, mutual_dd_rolling_pr_SPY, get_meta_aum, get_meta_fee]
    if names is None:
        names = ['cagr', 'start_yield', 'curr_yield', 'ulcer_pr', 'mutual_dd', 'aum', 'fee', 'zmean']
    if weights is None:
        weights = [1, 0, 1, -1, -1, (0.5, "log"), -0.5]
    print(f"weights: {weights}")
    dfz = _cache[0]
    if dfz is None:
        dfz = zscores(all, *funcs, weights=[1, 0, 1, -1, -1, (0.5, "log"), -0.5])
    _cache[0] = dfz
    with pd.option_context('display.max_rows', n):
        display(zscore_df_style(zscores_update(dfz, weights)[:n], names))
