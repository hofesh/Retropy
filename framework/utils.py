import numbers
import functools as ft
import types
import sys

import numpy as np
import pandas as pd

trimmed_messages = set()
def print_norep(msg, **args):
    if not msg in trimmed_messages:
        trimmed_messages.add(msg)
        print(msg, **args)

def warn(*arg, **args):
    set_if_none(args, "file", sys.stderr)
    print_norep(*arg, **args)

def get_func_name(f):
    if hasattr(f, "__name__"):
        return f.__name__
    return "<unnamed function>" 

def lmap(f, l):
    return list(map(f, l))

def lfilter(f, l):
    return list(filter(f, l))

def flatten(lists):
    return [x for lst in lists for x in lst]

def flattenLists(items):
    res = []
    for x in items:
        if isinstance(x, list):
            res += x
        elif isinstance(x, range) or isinstance(x, map) or isinstance(x, types.GeneratorType):
            res += list(x)
        else:
            res.append(x)
    return res

def fixtext(s):
    if isinstance(s, str):
        return bidialg.get_display(s)
    return [fixtext(x) for x in s]
    

def set_if_none(d, key, value):
    v = d.get(key, None)
    if v is None:
        d[key] = value
    return d
        
def list_rm(l, *items):
    l = l.copy()
    for x in items:
        if x in l:
            l.remove(x)
    return l    


def compose(*functions):
    return ft.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
reduce = compose

def partial(func, *args, **kwargs):
    partial_func = ft.partial(func, *args, **kwargs)
    ft.update_wrapper(partial_func, func)
    return partial_func

def is_number(v):
    return isinstance(v, numbers.Real)

def as_int(s):
    try:
        return int(s)
    except:
        return None

def drop_duplicates_index(s):
    return s[~s.index.duplicated()]


# we need this functions in base.py, but don't want a cirular dependency with base and stats_basic, so it's here for now
def ret(s, n=1):
    return s.pct_change(n)

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

# safely convert a float/string/mixed series to floats
# to remove commas we need the data type to be "str"
# but if we assume it's "str" wihtout converting first, and some are numbers
# those numbers will become NaN's.
def series_as_float(ser):
    return pd.to_numeric(ser.astype(str).str.replace(",", "").str.replace("%", "").str.replace("$", ""), errors="coerce")

