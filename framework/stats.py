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


