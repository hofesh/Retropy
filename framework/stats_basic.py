from scipy.stats import zscore, zmap
import numpy as np
import math

from framework.utils import *
from framework.RpySeries import *
from framework.base import *

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

def cagr(s):
    days = (s.index[-1] - s.index[0]).days
    if days <= 0:
        return np.nan
    years = days/365
    val = s[-1] / s[0]
    if val < 0:
        raise Exception("Can't calc cagr for a negative value") # this indicates that the series is not an equity curve
    return (math.pow(val, 1/years)-1)*100

def cagr_pr(s):
    return cagr(pr(s))

def ret(s):
    return s.pct_change()

def i_ret(s):
    s = s.fillna(0)
    return np.cumprod(s + 1)

def logret(s, dropna=True, fillna=False):
    res = np.log(s) - np.log(s.shift(1))
    if "name" in dir(res) and s.name:
        res.name = "logret(" + s.name + ")"
    if fillna:
        res[0] = 0
    elif dropna:
        res = res.dropna()
    return res

# we sometimes get overflow encountered in exp RuntimeWarning from i_logret, so we disable them
np.seterr(over='ignore') 
def i_logret(s):
    res = np.exp(np.cumsum(s))
    if np.isnan(s[0]):
        res[0] = 1
    return res

    
def percentile(s, p):
    return s.quantile(p/100)   

def past(s, n):
    return shift(s, n)

def future(s, n):
    return shift(s, -n)

def mcagr_future(s, years=5):
    return future(mcagr(s, n=365*years), 365*years)

def mcagr(s, n=365, dropna=True):
    return name(roll_ts(s, cagr, n, dropna=dropna), s.name + " cagr")

def mstdret(s, n=365, dropna=True):
    res = name(ret(s).rolling(n).std()*math.sqrt(n)*100, s.name + " std")
    if dropna:
        res = res.dropna()
    return res

def mstd(s, n=365, dropna=True):
    res = name(s.rolling(n).std(), s.name + " std")
    if dropna:
        res = res.dropna()
    return res

def msharpe(s, n=365, dropna=True):
    return name(mcagr(s, n, dropna) / mstd(s, n, dropna), s.name + " sharpe")
