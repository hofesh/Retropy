import pandas as pd
import numbers

from framework.utils import *
import framework.base
#import framework.symbol

def _get_pretty_name(s):
    if type(s.name).__name__ == "Symbol":
        return s.name.pretty_name
    return s.name

class RpySeries(pd.Series):
    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return id(self) == id(other)

    def sname(self, n):
        framework.base.name(self, n)
        return self

    def __mul__(self, other):
        if is_number(other):
            return super().__mul__(other)
        # other = framework.base.get(other)
        res = super().__mul__(other)
#        res.name = symb.get_pretty_name(self) + " * " + symb.get_pretty_name(other)
        if other.name is None:
            res.name = _get_pretty_name(self)
        else:
            res.name = _get_pretty_name(self) + " * " + _get_pretty_name(other)
        return rpy(res)

    def __truediv__(self, other):
        if is_number(other):
            return super().__truediv__(other)
        # other = framework.base.get(other)
        res = super().__truediv__(other)
        #res.name = symb.get_pretty_name(self) + " / " + symb.get_pretty_name(other)
        res.name = _get_pretty_name(self) + " / " + _get_pretty_name(other)
        return rpy(res)

# def sdiv(a, b):
#     if isinstance(a,list):
#         return lmap(lambda x: sdiv(x, b), a)
#     a, b = get([a, b])
#     x = a / b
#     x.name = get_pretty_name(a) + " / " + get_pretty_name(b)
#     return x
def is_series(x):
    return isinstance(x, pd.Series)

def is_not_series(x):
    return not is_series(x)

def is_series_or_str(x):
    return isinstance(x, str) or is_series(x)

def is_not_series_or_str(x):
    return not is_series_or_str(x)

def is_number(s):
    return isinstance(s, numbers.Real)

def is_named_number(val):
    return isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], numbers.Real) and isinstance(val[1], str)    

def rpy(s):
    if isinstance(s, RpySeries):
        return s
    if is_series(s):
        res = RpySeries(s, s.index)
        #res.name = s.name # name is picked up automatically
        return res
    return s

def is_rpy(s):
    return isinstance(s, RpySeries)

def wrap(s, name=""):
    return s

    # name = name or s.name
    # #if not name:
    # #    raise Exception("no name")
    # if isinstance(s, pd.Series):
    #     s = Wrapper(s)
    #     s.name = name
    # elif isinstance(s, Wrapper):
    #     s.name = name
    # return s

def unwrap(s):
    return s
    # if isinstance(s, Wrapper):
    #     return s.s
    # return s
    
def rename(s, n):
    return name(s.copy(), n)


########## mutating series ###########

def sync(a, b):
    a = a.dropna()
    b = b.dropna()
    idx = a.index.intersection(b.index)
    a = a.reindex(idx)
    b = b.reindex(idx)
    return a, b

def expand(a, b):
    a = a.dropna()
    b = b.dropna()
    idx = a.index.union(b.index)
    a = a.reindex(idx)
    b = b.reindex(idx)
    return a, b

def trimBy(trimmed, by):
    not_list = False
    if not isinstance(trimmed, list):
        not_list = True
        trimmed = [trimmed]
    if not isinstance(by, list):
        by = [by]
    if len(by) == 0:
        return []
    start = max(s.index[0] for s in by)
    res = [s[start:] for s in get(trimmed)]
    if not_list:
        return res[0]
    return res

def align_with(s, w, center=False):
    if center:
        idx = len(s) // 2
        if s.index[idx] in w:
            return s * w[s.index[idx]] / s[idx]
        idx = len(w) // 2
        if w.index[idx] in s:
            return s * w[idx] / s[w.index[idx]]
    else:        
        if s.index[0] in w:
            return s * w[s.index[0]] / s[0]
        if w.index[0] in s:
            return s * w[0] / s[w.index[0]]
    raise Exception(f"Cannot align {_get_pretty_name(s)} with {_get_pretty_name(w)}, no common start date found")
    #raise Exception(f"Cannot align {symb.get_pretty_name(s)} with {symb.get_pretty_name(w)}, no common start date found")

def align(s):
    return s / s[0]

def align_rel(all, base=None):
    if len(all) == 0:
        return all
    non_series = lfilter(is_not_series_or_str, all)
    all = lfilter(is_series_or_str, all)
    all = sorted(all, key=lambda s: s.index[0])
    if base is None:
        base = all[0]
        base = base / base[0]
        res = [base]
        all = all[1:]
    else:
        base = base / base[0]
        res = []
    for s in all:
        s = align_with(s, base)
        res.append(s)
    #    base = s
    return res + non_series

###################################
