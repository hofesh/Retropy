import pandas as pd
import numbers

import framework.base


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
        other = framework.base.get(other)
        res = super().__mul__(other)
        res.name = framework.base.get_pretty_name(self) + " * " + framework.base.get_pretty_name(other)
        return rpy(res)

    def __truediv__(self, other):
        if is_number(other):
            return super().__truediv__(other)
        other = framework.base.get(other)
        res = super().__truediv__(other)
        res.name = framework.base.get_pretty_name(self) + " / " + framework.base.get_pretty_name(other)
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
