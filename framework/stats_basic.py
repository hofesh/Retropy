from scipy.stats import zscore, zmap
import numpy as np
import math

from framework.utils import *
from framework.RpySeries import *
from framework.base import *



# def bom(s):
#     idx = s.index.values.astype('datetime64[M]') # convert to monthly representation
#     idx = np.unique(idx) # remove duplicates
#     return s[idx].dropna()

# def boy(s):
#     idx = s.index.values.astype('datetime64[Y]') # convert to monthly representation
#     idx = np.unique(idx) # remove duplicates
#     return s[idx].dropna()

# see: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
def bow(s): return s.asfreq("W") # TODO: WS freq doesn't exist
def eow(s): return s.asfreq("W")
def bom(s): return s.asfreq("MS")
def eom(s): return s.asfreq("M")
def boq(s): return s.asfreq("QS")
def eoq(s): return s.asfreq("Q")
def boy(s): return s.asfreq("YS")
def eoy(s): return s.asfreq("Y")


def last(s): return s[-1]
def first(s): return s[0]

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
    if np.any(s <= 0):
        warn(f"{get_name(s)} has non-positive values, can't calc cagr()")
        return np.nan
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

def shift(s, n):
    if len(s) == 0:
        return s
    start = s.index[0]+n if n < 0 else s.index[0]
    end = s.index[-1]+n if n > 0 else s.index[-1]
    index = pd.date_range(start, end)
    s = s.reindex(index)
    return s.shift(n)
    

def past(s, n):
    return shift(s, n)

def future(s, n):
    return shift(s, -n)

# https://stackoverflow.com/questions/38878917/how-to-invoke-pandas-rolling-apply-with-parameters-from-multiple-column
# https://stackoverflow.com/questions/18316211/access-index-in-pandas-series-apply
def roll_ts(s, func, n, dropna=True):
    if s.index[0].year < 1970:
        warn("roll_ts has a bug for dates before 1970-01-01, don't use it") # it will makes all periods after 1970 start with 1970
    # note that rolling auto-converts int to float: https://github.com/pandas-dev/pandas/issues/15599
    # i_ser = pd.Series(range(s.shape[0]))
    # res = i_ser.rolling(n).apply(lambda x: func(pd.Series(s.values[x.astype(int)], s.index[x.astype(int)])))
    res = s.rolling(n).apply(func, raw=False) # with raw=False, we get a rolling Series :)

    res = pd.Series(res.values, s.index)
    if dropna:
        res = res.dropna()
    return res

def mcagr_monthly(s, years):
    return (np.power(s / s.shift(12*years), 1/years)-1)*100
    
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

def ulcer(x):
    if x is None:
        return np.nan
    cmax = np.maximum.accumulate(x)
    r = (x/cmax-1)*100
    return math.sqrt(np.sum(r*r)/x.shape[0])

def ulcer_pr(x):
    return ulcer(pr(x))


def mret(s):
    return ret(eom(s)).dropna()
def wret(s):
    return ret(eow(s)).dropna()

def _capture(x, idx):
    return np.power(np.prod(1+x[idx]), 1/len(idx))-1

def _get_capture(s, b, is_pos):
    s, b = sync(tr(s), tr(b))
    s_ret = mret(s)
    b_ret = mret(b)
    idx = b_ret > 0 if is_pos else b_ret < 0
    return _capture(s_ret, idx) / _capture(b_ret, idx) * 100

def get_upside_capture(s, b):
    return _get_capture(s, b, True)

def get_downside_capture(s, b):
    return _get_capture(s, b, False)

def get_capture_ratio(s, b):
    return get_upside_capture(s, b) / get_downside_capture(s, b)

def get_downside_capture_SPY(s):
    return get_downside_capture(s, 'SPY')

def get_upside_capture_SPY(s):
    return get_upside_capture(s, 'SPY')

def get_capture_ratio_SPY(s, b):
    return get_capture_ratio(s, 'SPY')

def reject_outliers(data, m = 2.):
    data = data[~pd.isnull(data)]
    data = data.astype(float)
    if len(data) == 0:
        return data
    med = np.nanmedian(data)
    if pd.isnull(med):
        return data
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d/mdev if mdev else 0.
    return data[s<m]
drop_outliers = reject_outliers

# zscore with mean and std calculated from the source data cleaned from outliers
def zscore_modified(vals):
    if any(isinstance(v, str) for v in vals):
        return np.zeros(len(vals))
    vals = [np.nan if v is None else v for v in vals] # replace None with np.nan
    # print(type(vals[0]))
    # print(vals[:10])
    vals_no_outliers = reject_outliers(np.array(vals)).flatten()
    return zmap(vals, vals_no_outliers)
    

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


