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

def ulcer(x):
    cmax = np.maximum.accumulate(x)
    r = (x/cmax-1)*100
    return math.sqrt(np.sum(r*r)/x.shape[0])

def ulcer_pr(x):
    return ulcer(pr(x))
    
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
    vals_no_outliers = reject_outliers(np.array(vals))
    return zmap(vals, vals_no_outliers)
    
def zscores(all, *funcs, signs=None):
    if signs is None:
        signs = [1] * len(funcs)
    d_vals = {get_func_name(f): lmap(f, all) for f in funcs}
    d = dict(d_vals)
    d["name"] = names(all)
    df = pd.DataFrame(d).set_index("name")
    
    d_zs = {k: zscore_modified(v)*sign for (k, v), sign in zip(d_vals.items(), signs)}
    d_zs["name"] = names(all)
    df_z = pd.DataFrame(d_zs).set_index("name")

    zmean = df_z.mean(axis=1)
    df["zmean"] = zmean
    df = df.sort_values("zmean", ascending=False)
    return df

def display_zscores(*all, n=50):
    dfz = zscores(all, cagr, get_curr_yield_min2, ulcer_pr, mutual_dd_rolling_pr_SPY, signs=[1, 1, -1, -1])
    with pd.option_context('display.max_rows', n):
        display(dfz[:n])    