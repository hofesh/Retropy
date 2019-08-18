from scipy.stats import zscore, zmap
import numpy as np
import math

from framework.utils import *
from framework.symbol import *
from framework.base import *
from framework.yields import *
from framework.draw_downs import *
from framework.lr import *
import framework.asset_classes as ac


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

    
def pr_lr_cagr(s):
    x = lr(price(s))
    x = x[x>0] # we won't be able to calc cagr for negative values
    return cagr(x)

def lrretm_beta_SPY(s):
    return lrret_beta(s, 'SPY', freq="M")


def get_usd_corr(s):
    return -corr(logret(eow(s)), logret(eow(get(ac.usdBroad))))
    
def get_usd_pvalue(s):
    return corr(logret(eow(s)), logret(eow(get(ac.usdBroad))), p_value=True)[1]

def get_future_return(s, days):
    x = mcagr(s, days).dropna()
    x = x.reindex(pd.date_range(x.index[0]-days, x.index[-1]))
    fret = x.shift(-days).dropna()
    return fret

def get_future_return_monthly(s, years):
    s = s.asfreq("MS") # TODO: this depends on the series month freq M or MS, mjust be set if not defined
    months = 12*years
    x = mcagr_monthly(s, years).dropna()
#     x = x.reindex(pd.date_range(x.index[0]-months, x.index[-1]))
    fret = x.shift(-months).dropna()
    return fret

def extrapolate_to_today(s, n_last):
    s_ret = ret(s)
    rate = s_ret[:-n_last].mean()
    s_ret = s_ret.reindex(pd.date_range(s_ret.index[0], pd.datetime.today()))
    s_ret = s_ret.fillna(rate)
    s_ret.iloc[0] = np.nan
    s = i_ret(s_ret) * s.iloc[0]
    return s
