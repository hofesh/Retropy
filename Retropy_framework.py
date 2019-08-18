# coding: utf-8

ipy = False
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython:
        ipy = True
except:
    pass
if not ipy:
    print(f"iPy: {ipy}")

import warnings
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import os
import sys
import datetime
import numbers
import subprocess
import uuid
import string
import json 
import requests
from io import StringIO
import re
import math
import types

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import linear_model






import plotly.offline as py
import plotly.graph_objs as go
import plotly.graph_objs.layout as gol
if ipy:
    py.init_notebook_mode()

from pathlib import Path
from bs4 import BeautifulSoup

from pyfinance import ols

import statsmodels.api as sm


from bidi import algorithm as bidialg

import matplotlib
import matplotlib.pyplot as plt
if ipy:
    get_ipython().run_line_magic('matplotlib', 'inline')
#matplotlib.rcParams['figure.figsize'] = (20.0, 10.0) # Make plots bigger
plt.rcParams['figure.figsize'] = [12.0, 8.0]


# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.float_format', lambda x: '{:,.4f}'.format(x))
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 30)

from IPython.display import clear_output, display


from framework.utils import *
from framework.base import *
from framework.pca import *
from framework.meta_data import *
from framework.stats_basic import *
from framework.stats import *
from framework.draw_downs import *
from framework.RpySeries import *
from framework.asset_classes import *
from framework.zscores_table import *
from framework.yields import *
from framework.data_sources_special import *
import framework.meta_data_dfs as meta_dfs
import framework.conf as conf
import framework.cefs as cefs
import framework.etfs as etfs
import framework.etfs_high_yield as etfs_high_yield

# In[ ]:


def pd_from_dict(d):
    return pd.DataFrame.from_dict(d, orient='index').T.sort_index()


# In[ ]:





# In[ ]:


    


# In[ ]:



# In[ ]:


import scipy.optimize
from datetime import datetime as dt
def xnpv(rate, values, dates):
    '''Equivalent of Excel's XNPV function.

    >>> from datetime import date
    >>> dates = [date(2010, 12, 29), date(2012, 1, 25), date(2012, 3, 8)]
    >>> values = [-10000, 20, 10100]
    >>> xnpv(0.1, values, dates)
    -966.4345...
    '''
    if rate <= -1.0:
        return float('inf')
    d0 = dates[0]    # or min(dates)
    return sum([ vi / (1.0 + rate)**((di - d0).days / 365.0) for vi, di in zip(values, dates)])


def xirr(values, dates):
    '''Equivalent of Excel's XIRR function.

    >>> from datetime import date
    >>> dates = [date(2010, 12, 29), date(2012, 1, 25), date(2012, 3, 8)]
    >>> values = [-10000, 20, 10100]
    >>> xirr(values, dates)
    0.0100612...
    '''
    # we prefer to try brentq first as newton keeps outputting tolerance warnings
    try:
        return scipy.optimize.brentq(lambda r: xnpv(r, values, dates), -1.0, 1e10)
        #return scipy.optimize.newton(lambda r: xnpv(r, values, dates), 0.0, tol=0.0002)
    except RuntimeError:    # Failed to converge?
        return scipy.optimize.newton(lambda r: xnpv(r, values, dates), 0.0, tol=0.0002)
        #return scipy.optimize.brentq(lambda r: xnpv(r, values, dates), -1.0, 1e10)

#xirr([-100, 100, 200], [dt(2000, 1, 1), dt(2001, 1, 1), dt(2002, 1, 1)])


# In[ ]:


def curr_price(symbol):
    if symbol in conf.ignoredAssets: return 0
    return get(symbol)[-1]

#def getForex(fromCur, toCur):
#    if fromCur == toCur: return 1
#    if toCur == "USD":
#        return get(fromCur + "=X", "Y")
#    if fromCur == "USD":
#        return get(toCur + "=X", "Y").map(lambda x: 1.0/x)


# In[ ]:


# In[ ]:


# conf_cache_fails = False    
# conf_cache_memory = False
# conf_cache_disk = False
# conf = GetConf(splitAdj=True, divAdj=True, cache=True, mode="TR", source="TASE", secondary=None)
# ds = TASEDataSource("TASE")
# df = ds.get(Symbol("01116441"), conf)
# df = ds.get(Symbol("05117478"), conf)
# df = ds.get(Symbol("137"), conf)
# df


# In[ ]:


# fetching data

if not "Wrapper" in locals():
    class Wrapper(object):

        def __init__(self, s):
            #self.s = s
            object.__setattr__(self, "s", s)

        def __getattr__(self, name):
            attr = self.s.__getattribute__(name)

            if hasattr(attr, '__call__'):
                def newfunc(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    if type(result) is pd.Series:
                        result = Wrapper(result)
                    return result
                return newfunc

            if type(attr) is pd.Series:
                attr = Wrapper(attr)
            return attr

        def __setattr__(self, name, value):
            self.s.__setattr__(name, value)

        def __getitem__(self, item):
             return wrap(self.s.__getitem__(item), self.s.name)

#         def __truediv__(self, other):
#             divisor = other
#             if type(other) is Wrapper:
#                 divisor = other.s
#             series = self.s / divisor
#             name = self.name
#             if type(other) is Wrapper:
#                 name = self.s.name + " / " + other.s.name
#             return wrap(series, name)

        def __truediv__(self, other):
            return Wrapper.doop(self, other, "/", lambda x, y: x / y)
        def __rtruediv__(self, other):
            return Wrapper.doop(self, other, "/", lambda x, y: x / y, right=True)
        
        def doop(self, other, opname, opLambda, right=False):
            divisor = other
            if type(other) is Wrapper:
                divisor = other.s
            if right:
                series = opLambda(divisor, self.s)
            else:
                series = opLambda(self.s, divisor)
            name = self.name
            if type(other) is Wrapper:
                if right:
                    name = other.s.name + " " + opname + " " + self.s.name
                else:
                    name = self.s.name + " " + opname + " " + other.s.name
            return wrap(series, name)

        def __sub__(self, other):
            return Wrapper.doop(self, other, "-", lambda x, y: x - y)
        #def __rsub__(self, other):
        #    return Wrapper.doop(self, other, "-", lambda x, y: x - y, right=True)

        def __mul__(self, other):
            return Wrapper.doop(self, other, "*", lambda x, y: x * y)
        def __rmul__(self, other):
            return Wrapper.doop(self, other, "*", lambda x, y: x * y, right=True)


# error in ('raise', 'ignore'), ignore will return None

# plotting

from plotly.graph_objs import *

def createVerticalLine(xval):
    shape = {
            'type': 'line',
            #'xref': 'x',
            'x0': xval,
            'x1': xval,
            'yref': 'paper',
            'y0': 0,
            'y1': 1,
            #'fillcolor': 'blue',
            'opacity': 1,
            'line': {
                'width': 1,
                'color': 'red'
            }
        }
    return shape
    
def createHorizontalLine(yval):
    shape = {
            'type': 'line',
            'xref': 'paper',
            'x0': 0,
            'x1': 1,
            #'yref': 'x',
            'y0': yval,
            'y1': yval,
            #'fillcolor': 'blue',
            'opacity': 1,
            'line': {
                'width': 1,
                'color': 'red'
            }
        }
    return shape


def plot(*arr, log=True, title=None, legend=True, lines=True, markers=False, annotations=False, xlabel=None, ylabel=None, show_zero_point=False, same_ratio=False):
    data = []
    shapes = []
    mode = ''
    if lines and markers:
        mode = 'lines+markers'
    elif lines:
        mode = 'lines'
    elif markers:
        mode = 'markers'
    if annotations:
        mode += '+text'
    hlines = []
    min_date = None
    for val in arr:
        # series
        if is_series(val):
            val = unwrap(val)
            name = get_pretty_name(val.name)
            text = name
            try:
                text = lmap(get_pretty_name ,val.names)
            except:
                pass
            # we add .to_pydatetime() since in Windows (and Chrome mobile iOS?) we get a numeric axis instead of date axis without this
            x = val.index
            if isinstance(x, pd.DatetimeIndex):
                x = x.to_pydatetime()
            data.append(go.Scatter(x=x, y=val, name=name, text=text, mode=mode, textposition='middle right', connectgaps=False))
            start_date = s_start(val)
            if start_date:
                if min_date is None:
                    min_date = start_date
                elif start_date:
                    min_date = min(min_date, start_date)
        # vertical date line
        elif isinstance(val, datetime.datetime):
            shapes.append(createVerticalLine(val))
        # vertical date line
        elif isinstance(val, np.datetime64):
            shapes.append(createVerticalLine(val.astype(datetime.datetime)))
        # horizontal value line
        elif isinstance(val, numbers.Real):
            shapes.append(createHorizontalLine(val))
            if val == 0:
                log = False
        elif is_named_number(val):
            hlines.append(val)
        else:
            raise Exception("unsupported value type: " + str(type(val)))
    
    for val, txt in hlines:
        shapes.append(createHorizontalLine(val))
        data.append(go.Scatter(x=[min_date], y=[val], text=txt, mode='text', textposition='top right', showlegend=False))
        if val == 0:
            log = False

    for d in data:
        d = d.y
        if isinstance(d, tuple): # for named numbers
            continue
        if d is None:
            continue
        if isinstance(d, np.ndarray):
            d = d[~pd.isnull(d)]
        if np.any(d <= 0):
            log = False
            
    mar = 30
    margin=gol.Margin(
        l=mar,
        r=mar,
        b=mar,
        t=mar,
        pad=0
    )
    
    #bgcolor='#FFFFFFBB',bordercolor='#888888',borderwidth=1,
    if legend:
        legendArgs=dict(x=0,y=-0.06,traceorder='normal', orientation='h', yanchor='top',
            bgcolor='rgb(255,255,255,50)',bordercolor='#888888',borderwidth=1,
            font=dict(family='sans-serif',size=12,color='#000'),
        )    
    else:
        legendArgs = {}
    yaxisScale = "log" if log else None
    rangemode = "tozero" if show_zero_point else "normal"
    yaxis = dict(rangemode=rangemode, type=yaxisScale, autorange=True, title=ylabel)
    if same_ratio:
        yaxis['scaleanchor'] = 'x'
        yaxis['scaleratio'] = 1
    layout = go.Layout(legend=legendArgs, 
                       showlegend=legend, 
                       margin=margin, 
                       yaxis=yaxis,  # titlefont=dict(size=18)
                       xaxis=dict(rangemode=rangemode, title=xlabel), # titlefont=dict(size=18) 
                       shapes=shapes, 
                       title=title,
                       hovermode = 'closest')
    fig = go.Figure(data=data, layout=layout)
    if not ipy:
        warn("not plotting, no iPython env")
        return
    py.iplot(fig)

# simple X, Y scatter
def plot_scatter_xy(x, y, names=None, title=None, xlabel=None, ylabel=None, show_zero_point=False, same_ratio=False):
    ser = pd.Series(y, x)
    if names:
        ser.names = names
    plot(ser, lines=False, markers=True, annotations=True, legend=False, log=False, title=title, xlabel=xlabel, ylabel=ylabel, show_zero_point=show_zero_point, same_ratio=same_ratio)

# this also supports line-series and single points
# each point must be a series with length=1
def plot_scatter(*lst, title=None, xlabel=None, ylabel=None, show_zero_point=False, same_ratio=False):
    plot(*lst, lines=True, markers=True, annotations=True, legend=False, log=False, title=title, xlabel=xlabel, ylabel=ylabel, show_zero_point=show_zero_point, same_ratio=same_ratio)

# show a stacked area chart normalized to 100% of multiple time series
def plotly_area(df, title=None):
    tt = df.div(df.sum(axis=1), axis=0)*100 # normalize to summ 100
    tt = tt.reindex(tt.mean().sort_values(ascending=False).index, axis=1) # sort columns by mean value
    tt = tt.sort_index()
    tt2 = tt.cumsum(axis=1) # calc cum-sum
    data = []
    for col in tt2:
        s = tt2[col]
        x = s.index
        if isinstance(x, pd.DatetimeIndex):
            x = x.to_pydatetime()
        trace = go.Scatter(
            name=col,
            x=x,
            y=s.values,
            text=["{:.1f}%".format(v) for v in tt[col].values], # use text as non-cumsum values
            hoverinfo='name+x+text',
            mode='lines',
            fill='tonexty'
        )
        data.append(trace)

    mar = 30
    margin=gol.Margin(l=mar,r=mar,b=mar,t=mar,pad=0)
    legend=dict(x=0,y=1,traceorder='reversed',
        #bgcolor='#FFFFFFBB',bordercolor='#888888',borderwidth=1,
        bgcolor='rgb(255,255,255,50)',bordercolor='#888888',borderwidth=1,
        font=dict(family='sans-serif',size=12,color='#000'),
    )    
    layout = go.Layout(margin=margin, legend=legend, title=title,
        showlegend=True,
        xaxis=dict(
            type='date',
        ),
        yaxis=dict(
           type='linear',
           range=[1, 100],
           dtick=20,
           ticksuffix='%'
        )
     )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='stacked-area-plot')
       


# In[ ]:


# data processing





def doClean(data):
    return [s.dropna() if is_series(s) else s for s in data]

def try_parse_date(s, format):
    try:
        return datetime.datetime.strptime(s, format)
    except ValueError:
        return None    

def easy_try_parse_date(s):
    return try_parse_date(s, "%d/%m/%Y") or try_parse_date(s, "%d.%m.%Y") or try_parse_date(s, "%d-%m-%Y")
    
def do_sort(data):
    sers = lfilter(is_series, data)
    non_sers = lfilter(is_not_series, data)
    sers = sorted(sers, key=lambda x: x.index[0])
    return sers + non_sers

def show(*data, trim=True, trim_end=False, align=True, align_base=None, ta=True, cache=None, mode=None, source=None, remode=None, untrim=None, silent=False, sort=True, drop_corrupt=False, **plotArgs):
    getArgs = {}
    if not mode is None:
        getArgs["mode"] = mode
    if not cache is None:
        getArgs["cache"] = cache
    if not source is None:
        getArgs["source"] = source
    if not remode is None:
        getArgs["remode"] = remode
    if not untrim is None:
        getArgs["untrim"] = untrim
    if not drop_corrupt is None:
        getArgs["drop_corrupt"] = drop_corrupt
    
    data = flattenLists(data)
    items = []
    for x in data:
        if x is None:
            continue
        if isinstance(x, pd.DataFrame):
            items += [x[c] for c in x]
        elif isinstance(x, datetime.datetime) or isinstance(x, np.datetime64):
            items.append(x)
        elif isinstance(x, str) and easy_try_parse_date(x):
            items.append(easy_try_parse_date(x))
        elif isinstance(x, numbers.Real):
            items.append(x)
        elif is_named_number(x):
            items.append(x)
        else:
            x = get(x, **getArgs)
            items.append(x)
    data = items
    data = doClean(data)
    data = [s for s in data if not is_series(s) or len(s) > 0]
    
    dataSeries = [s for s in data if is_series(s)]
    if ta == False:
        trim = False
        align = False
    elif ta == 'rel':
        trim = False
        align = 'rel'
    if any([s[unwrap(s)<0].any() for s in dataSeries]):
        align = False
    
    if trim: 
        data = doTrim(data, trim=trim, trim_end=trim_end)
    
    if align:
        if align == "rel":
            data = align_rel(data, base=align_base)
        else:
            data = doAlign(data)
    
    if sort:
        data = do_sort(data)
    
    if not silent:
        plot(*data, **plotArgs)
    else:
        return dataSeries

def show_series(s, **args):
    show_scatter(range(len(s)), s.values, lines=True, annotations=s.index, show_zero_point=False, **args)
    
def show_scatter(xs, ys, setlim=True, lines=False, color=None, annotations=None, xlabel=None, ylabel=None, label=None, same_ratio=False, show_zero_point=False, fixtext=False, figure=False):
    def margin(s, m=0.05, show_zero_point=False):
        mn = min(s)
        mx = max(s)
        rng = mx-min(mn, 0)
        mn = mn - rng*m
        mx = mx + rng*m
        if show_zero_point:
            mn = min(0, mn)
        return mn, mx
        
    if len(xs) == 0 or len(ys) == 0:
        return
    if annotations is None:
        if "name" in dir(xs[0]) or "s" in dir(xs[0]):
            annotations = [s.name for s in xs]
    if figure:
        if same_ratio:
            plt.figure(figsize=(12, 12))
        else:
            plt.figure(figsize=(16, 12))
    if lines:
        plt.plot(xs, ys, marker="o", color=color, label=label)
    else:
        plt.scatter(xs, ys, color=color, label=label)
    if setlim:
        if same_ratio:
            xmin, xmax = margin(xs, show_zero_point=show_zero_point)
            ymin, ymax = margin(ys, show_zero_point=show_zero_point)
            mn, mx = min(xmin, ymin), max(xmax, ymax)
            plt.xlim(mn, mx)
            plt.ylim(mn, mx)
        else:
            plt.xlim(*margin(xs, show_zero_point=show_zero_point))
            plt.ylim(*margin(ys, show_zero_point=show_zero_point))
    plt.axhline(0, color='gray', linewidth=1)
    plt.axvline(0, color='gray', linewidth=1)
    if xlabel: plt.xlabel(xlabel, fontsize=16)
    if ylabel: plt.ylabel(ylabel, fontsize=16)
    if not annotations is None:
        for i, txt in enumerate(annotations):
            if fixtext:
                txt = globals()["fixtext"](txt)
            plt.annotate(txt, (xs[i], ys[i]), fontsize=14)

def show_modes(*lst, **args):
    show(*lmap(modes, lst), **args, title="PR/NTR/TR modes")

def show_modes_comp(a, b, show_zero=True):
    show([sdiv(x, y) for x,y in zip(modes(a), modes(b))], 1, 0 if show_zero else None, title="relative modes")

def show_scatter_returns(y_sym, x_sym, freq=None):
    x_sym, y_sym = get(x_sym), get(y_sym)
    x, y = doTrim([x_sym, y_sym])
    x, y = sync(x, y)
    
    if freq:
        x = x.asfreq(freq)
        y = y.asfreq(freq)
        
    x, y = logret(x), logret(y)
    
    show_scatter(x, y, same_ratio=True, xlabel=x_sym.name, ylabel=y_sym.name)



def reduce_series(lst, g_func=None, y_func=None, x_func=None, trim=True):
    # first we must set the trim limits for all assets
    if not trim is None:
        lst = get(lst, trim=trim)
    # only them we apply the get function (it will preserve the trims by default)
    if not isinstance(g_func, list):
        lst = lmap(g_func, lst)
    if isinstance(g_func, list):
        ys = [[y_func(gf(s)) for gf in g_func] for s in lst]
        xs = [[x_func(gf(s)) for gf in g_func] for s in lst]
    elif isinstance(y_func, list):
        ys = [[yf(s) for yf in y_func] for s in lst]
        xs = [[x_func(s)] * len(y_func) for s in lst]
    elif isinstance(x_func, list):
        xs = [[xf(s) for xf in x_func] for s in lst]
        ys = [[y_func(s)] * len(x_func) for s in lst]
    
    res = [pd.Series(y, x) for y, x in zip(ys, xs)]
    res = [name(r, get_name(s, nick_or_name=True)) for r, s in zip(res, lst)]
    for r, s in zip(res, lst):
        r.name = start_year_full_with_name(s)
        r.names = [''] * r.shape[0]
        r.names[0] = str(s.name)
    return res

# def reduce_series(lst, g_func=None, y_func=None, x_func=None, trim=trim):
#     if not isinstance(g_func, list):
#         lst = lmap(g_func, lst)
#     if isinstance(g_func, list):
#         ys = [[y_func(gf(s)) for gf in g_func] for s in lst]
#         xs = [[x_func(gf(s)) for gf in g_func] for s in lst]
#     elif isinstance(y_func, list):
#         ys = [[yf(s) for yf in y_func] for s in lst]
#         xs = [[x_func(s)] * len(y_func) for s in lst]
#     elif isinstance(x_func, list):
#         xs = [[xf(s) for xf in x_func] for s in lst]
#         ys = [[y_func(s)] * len(x_func) for s in lst]
    
#     res = [pd.Series(y, x) for y, x in zip(ys, xs)]
#     res = [name(r, get_name(s, nick_or_name=True)) for r, s in zip(res, lst)]
#     for r, s in zip(res, lst):
#         r.name = start_year_full_with_name(s)
#         r.names = [''] * r.shape[0]
#         r.names[0] = str(s.name)
#     return res

# experimental
def show_rr2(*lst, g_func=None, y_func=None, x_func=None, risk_func=None, ret_func=None, ret_func_names=None, trim=True, **args):
    y_func = y_func or ret_func
    x_func = x_func or risk_func
    # if g_func is None:
    #     starts = set(map(_start, filter(is_series, lst)))
    #     if len(starts) > 1:
    #         warn("show_rr2 called with untrimmed data, not trimming, results may be inconsistent")
    y_func = y_func or cagr
    x_func = x_func or ulcer
    g_func = g_func or get
    sers = lfilter(is_series_or_str, lst)
    non_ser = lfilter(is_not_series_or_str, lst)
    r = reduce_series(sers, g_func=g_func, y_func=y_func, x_func=x_func, trim=trim)
    def f_names(f):
        if isinstance(f, list):
            return " ➜ ".join(lmap(lambda x: get_func_name(x), f))
        return get_func_name(f)
    ylabel = " ➜ ".join(ret_func_names) if not ret_func_names is None else f_names(y_func)
    set_if_none(args, 'xlabel', f_names(x_func))
    set_if_none(args, 'ylabel', ylabel)
    set_if_none(args, 'title', f"{sers[0].name.mode} {args['ylabel']} <==> {args['xlabel']} [{f_names(g_func)}]")
    plot_scatter(*r, *non_ser, show_zero_point=True, **args)
# e.g.:
# show_risk_return2(*all, g_func=[ft.partial(get, despike=False), get])

def show_rr(*lst, ret_func=None, risk_func=None, trim=True, mode_names=False, lr_fit=False, same_ratio=False, **args):
    if ret_func is None: ret_func = cagr
    if risk_func is None: risk_func = ulcer
    non_ser = lfilter(lambda x: not(is_series_or_str(x) or isinstance(x, list)), lst)
    lst = lfilter(lambda x: is_series_or_str(x) or isinstance(x, list), lst)
    lst = get(lst, trim=trim)
    lst = [x if isinstance(x, list) else [x] for x in lst]
    res = [get_risk_return_series(x, ret_func=ret_func, risk_func=risk_func, mode_names=mode_names) for x in lst]
    if lr_fit:
        xs = [s.index[0] for s in res]
        ys = [s.iloc[0] for s in res]
        fit = lr(ys, xs, print_r2=True)
        if len(fit) > 1:
            fit = fit.iloc[::len(fit)-1] # keep first and last only
            fit.name = ''
            res.insert(0, fit)
    if same_ratio:
        xs = [s.index[0] for s in res]
        ys = [s.iloc[0] for s in res]
        rng = [min(min(ys), 0), max(ys)]
        f = pd.Series(rng, rng)
        f.name = ''
        res.insert(0, f)

    args['show_zero_point'] = True
    set_if_none(args, 'title',  f"{get_mode(lst[0][0])} {get_func_name(ret_func)} <==> {get_func_name(risk_func)}")
    set_if_none(args, 'xlabel', get_func_name(risk_func))
    set_if_none(args, 'ylabel', get_func_name(ret_func))
    set_if_none(args, 'same_ratio', same_ratio)
    plot_scatter(*non_ser, *res, **args)

showRiskReturn = show_rr # legacy

def get_risk_return_series(lst, ret_func, risk_func, mode_names, **args):
    if len(lst) == 0:
        return
    lst = [get(s) for s in lst]
    ys = [ret_func(unwrap(s)) for s in lst]
    xs = [risk_func(unwrap(s)) for s in lst]
    names = [get_name(s.name, use_sym_name=False, nomode=not mode_names, nick_or_name=True) for s in lst]

    res = pd.Series(ys, xs)
    res.name = names[0]
    res.names = names
    return res
    #plot(pd.Series(xs, ys), pd.Series(xs, ys+1), lines=False, markers=True)

# def showRiskReturnUtil(lst, ret_func=None, risk_func=None, **args):
#     if len(lst) == 0:
#         return
#     if ret_func is None: ret_func = cagr
#     if risk_func is None: risk_func = ulcer
#     lst = [get(s) for s in lst]
#     ys = [ret_func(s) for s in lst]
#     xs = [risk_func(s) for s in lst]
    
#     names = None
#     if args.get("annotations", None) is None:
#         if "name" in dir(lst[0]) or "s" in dir(lst[0]):
#             names = [s.name for s in lst]
#     elif args.get("annotations", None) != False:
#         names = args.get("annotations", None)
#     if names is None:
#         names = ["unnamed"] * len(lst)
#     names = ["nan" if n is None else n for n in names]
    
#     df = pd.DataFrame({"x": xs, "y": ys, "name": names})
#     nans = df[df.isnull().any(axis=1)]["name"]
#     if nans is None:
#         nans = []
#     if len(nans) > 0:
#         print(f'dropping series with nan risk/return: {" | ".join(nans)}')
#     df = df.dropna()
#     xs = df["x"].values
#     ys = df["y"].values
#     names = df["name"].values
            
#     if args.get("annotations", None) == False:
#         names = None
#     args['annotations'] = names
    
#     xlabel=risk_func.__name__
#     ylabel=ret_func.__name__
#     args = set_if_none(args, "show_zero_point", True)
#     show_scatter(xs, ys, xlabel=xlabel, ylabel=ylabel, **args)

def show_rr_capture_ratios(*all):
    show_rr(*all, ret_func=get_upside_capture_SPY, risk_func=get_downside_capture_SPY, same_ratio=True, lr_fit=True)

def show_rr_modes(*lst, ret_func=None, risk_func=None, modes=['TR', 'NTR', 'PR'], title=None):
    def get_data(lst, mode):
        return get(lst, mode=mode, trim=True)
    data_lst = [get_data(lst, mode) for mode in modes]
    all = [list(tup) for tup in zip(*data_lst)]
    
    ret_func = ret_func or cagr
    risk_func = risk_func or ulcer
    title = title or f"modes {get_func_name(ret_func)} vs {get_func_name(risk_func)}"
    show_rr(*all, ret_func=ret_func, risk_func=risk_func, title=title, mode_names=True)
    #showRiskReturn(*ntr, ret_func=ret_func)
    #for a, b in zip(tr, ntr):
    #    showRiskReturn([a, b], setlim=False, lines=True, ret_func=ret_func, annotations=False)    

def show_rr_modes_mutual_dd_risk_rolling_SPY(*all):
    title = f"modes CAGR vs PR mutual_dd_risk_rolling_SPY"
    show_rr_modes(*all, risk_func=mutual_dd_rolling_SPY)

def show_rr__cagr__mutual_dd_risk_rolling_pr_SPY(*all, lr_fit=False):
    title = f"{all[0].name.mode} CAGR vs PR mutual_dd_risk_rolling_SPY"
    show_rr(*all, risk_func=mutual_dd_rolling_pr_SPY, title=title, lr_fit=lr_fit)

def show_rr__yield__mutual_dd_risk_rolling_pr_SPY(*all, yield_func=None):
    yield_func = yield_func or get_curr_yield_min2
    title = f"{all[0].name.mode} Yield vs PR mutual_dd_risk_rolling_SPY"
    show_rr(2, 3, 4, 5, *all, ret_func=yield_func, risk_func=mutual_dd_rolling_pr_SPY, title=title, ylabel=f"{all[0].name.mode} {get_func_name(yield_func)}")

def show_rr_yield(*all, yield_func=None, risk_func=None):
    yield_func = yield_func or get_curr_yield_min2
    show_rr(2, 3, 4, 5, *all, ret_func=yield_func, risk_func=risk_func)

def show_rr__yield_range__mutual_dd_rolling_pr_SPY(*all):
    show_rr2(*all, 2, 3, 4, 5, ret_func=[get_curr_yield_max, get_curr_yield_min], ret_func_names=['max', 'min'], risk_func=mutual_dd_rolling_pr_SPY)

def show_rr__yield_types__ulcer(*lst, ret_func=None, types=['true', 'normal', 'rolling'], mode="TR", title=None):
    def get_data(lst, type):
        yld = [get_curr_yield(s, type=type) for s in lst]
        rsk = lmap(ulcer, lst)
        return pd.Series(yld, rsk)

    lst = get(lst, mode=mode, trim=True)
    res = []
    for s in lst:
        yld = []
        rsk = []
        for type in types:
            yld.append(get_curr_yield(s, type=type))
            rsk.append(ulcer(s))
        ser = pd.Series(yld, rsk)
        ser.name = s.name
        ser.names = [f"{s.name} {t}" for t in types]
        res.append(ser)
    
    title = title or f"Risk - {mode} Yield Types"
    plot_scatter(*res, title=title, xlabel="ulcer", ylabel="current yield", show_zero_point=True)

def show_risk_itr_pr(*lst, title=None):
    def get_data(lst, type):
        yld = [get_curr_yield(s, type=type) for s in lst]
        rsk = lmap(ulcer, lst)
        return pd.Series(yld, rsk)

    lst = get(lst, mode=mode, trim=True)
    res = []
    for s in lst:
        pr = get(s, mode="PR")
        itr = get(s, mode="ITR")
        pr_ulcer = ulcer(pr)
        x = [pr_ulcer, pr_ulcer]
        y = [cagr(pr), cagr(itr)]
        ser = pd.Series(y, index=x)
        ser.name = s.name
        ser.names = [s.name, '']
        res.append(ser)
    
    title = title or f"PR Risk - ITR Return"
    plot_scatter(*res, title=title, xlabel="ulcer", ylabel="cagr", show_zero_point=True)


def show_rr_yield_tr_ntr(*lst, title="Risk - 12m Yield TR-NTR"):
    show_rr_modes(*lst, ret_func=get_curr_yield_rolling, modes=['TR', 'NTR'], title=title)


def show_min_max_bands(symbol, n=365, show_symbol=True, ma_=False, elr_fit=True, rlr_fit=True, lr_fit=False, log=True):
    n = int(n)
    x = get(symbol)
    if log:
        x = np.log(x)
    a = name(mmax(x, n), 'max')
    b = name(mmin(x, n), 'min')
    c = name(mm(x, n), 'median')
    if not show_symbol:
        x = None
    _ma = ma(x, n) if ma_ else None
    _lr = lr(x) if lr_fit else None
    _elr = lr_expanding(x, freq="W") if elr_fit else None
    _rlr = lr_rolling(x, n // 7, freq="W") if rlr_fit else None
    show(c, a, b, x, _ma, _lr, _elr, _rlr, ta=False, sort=False, log=not log)

        

def show_rolling_beta(target, sources, window=None, rsq=True, betaSum=False, pvalue=False, freq=None, extra=None):
    if not isinstance(sources, list):
        sources = [sources]
        
    target = get(target)
    sources = get(sources)
    names = [s.name for s in sources]
    
    target = logret(target)
    sources = lmap(logret, sources)
    
    target = unwrap(target)
    sources = lmap(unwrap, sources)
    
    sources = pd.DataFrame(sources).T.dropna()

    target, sources = sync(target, sources)
    
    if freq:
        target = target.asfreq(freq)
        sources = sources.asfreq(freq)
        if window is None:
            window = int(get_anlz_factor(freq))
    else:
        if window is None:
            window = 365
    
    rolling = ols.PandasRollingOLS(y=target, x=sources, window=window)
    #rolling.beta.head()
    #rolling.alpha.head()
    #rolling.pvalue_alpha.head()
    #type(rolling.beta["feature1"])
    
    
    res = []
    
    if pvalue:
        _pvalue = rolling.pvalue_beta
        _pvalue.columns = [s + " pvalue" for s in names]
        res += [_pvalue, 0.05]
    
    if rsq:
        rsq = rolling.rsq
        rsq.name = "R^2"
        res += [rsq]

    _beta = rolling.beta
    _beta.columns = [s + " beta" for s in names]
    res += [_beta]
        
    if betaSum and len(names) > 1:
        _betaSum = rolling.beta.sum(axis=1)
        _betaSum.name = "beta sum"
        res += [_betaSum]
        
    res += [-1, 0, 1]
    
    if not extra is None:
        if isinstance(extra, list):
            res += extra
        else:
            res += [extra]
    
    show(res, ta=False)

def mix(s1, s2, n=10, do_get=False, **getArgs):
    part = 100/n
    res = []
    for i in range(n+1):
        x = {s1: i*part, s2: (100-i*part)}
        port = dict_to_port_name(x, drop_zero=True, drop_100=True, use_sym_name=True)
        name = dict_to_port_name(x, drop_zero=True, drop_100=True, use_sym_name=False)
        if i > 0 and i < n:
            name = ''
        x = f"{port}={name}"
        if do_get:
            x = get(x, **getArgs)
            #x.name = name
        # else:
        #     x = f"{port}={name}"
        res.append(x)
    return lmap(unwrap, res)
        


# https://stackoverflow.com/questions/38878917/how-to-invoke-pandas-rolling-apply-with-parameters-from-multiple-column
# https://stackoverflow.com/questions/18316211/access-index-in-pandas-series-apply
def roll_ts(s, func, n, dropna=True):
    # note that rolling auto-converts int to float: https://github.com/pandas-dev/pandas/issues/15599
    # i_ser = pd.Series(range(s.shape[0]))
    # res = i_ser.rolling(n).apply(lambda x: func(pd.Series(s.values[x.astype(int)], s.index[x.astype(int)])))
    res = s.rolling(n).apply(func, raw=False) # with raw=False, we get a rolling Series :)

    res = pd.Series(res.values, s.index)
    if dropna:
        res = res.dropna()
    return res



# In[ ]:


from scipy.optimize import minimize

def prep_as_df(target, sources, mode, as_logret=False, as_geom_value=False, freq=None):
    if not isinstance(sources, list):
        sources = [sources]

    target = get(target, mode=mode)
    sources = get(sources, mode=mode)
    names = [s.name for s in sources]

    if freq:
        target = target.asfreq(freq).dropna()
        sources = [s.asfreq(freq).dropna() for s in sources]
    
    if as_logret:
        target = logret(target)
        sources = lmap(logret, sources)

    target = unwrap(target)
    sources = lmap(unwrap, sources)

    sources = pd.DataFrame(sources).T.dropna()

    target, sources = sync(target, sources)

    if as_geom_value:
        target = target/target[0]
        sources = sources.apply(lambda x: x/x[0], axis=0)

    return target, sources


import sklearn.metrics
def lrret(target, sources, pos_weights=True, sum_max1=True, sum1=True, fit_values=True, 
          return_res=False, return_ser=True, return_pred=False, return_pred_fit=False, res_pred=False, show_res=True, freq=None, obj="sum_sq_log", mode=None, do_align=True):
    def apply(x, bias):
        res = x[0]*int(bias) + i_logret((sources_logret * x[1:]).sum(axis=1))
        return res

    def value_objective(x):
        pred = apply(x, bias=True)
        
        if obj == "log_sum_sq":
            # using log seem to work really well, it solves "Positive directional derivative for linesearch" issue 
            # for lrret(sc, lc).
            # https://stackoverflow.com/questions/11155721/positive-directional-derivative-for-linesearch
            # here it's mentioned that large valued objective functions can cause this
            # thus we use the log
            # add +1 to avoid log(0)
            return np.log(1+np.sum((target - pred) ** 2))
        
        if obj == "sum_sq":
            return np.sum((target - pred) ** 2)
        
        # this provides a better overall fit, avoiding excess weights to later (larger) values
        if obj == "sum_sq_log":
            return np.sum((np.log(target) - np.log(pred)) ** 2)
        
        raise Exception("invalid obj type: " + obj)

    def returns_objective(x):
        pred = apply(x, bias=False)
        return np.sum((logret(target) - logret(pred)) ** 2)

    # prep data
    if not isinstance(sources, list):
        sources = [sources]
    sources = [s for s in sources if (not s is target) and getName(s) != getName(target)]
    orig_sources = sources
    orig_target = get(target, mode=mode)

    target, sources = prep_as_df(target, sources, mode, as_geom_value=fit_values and do_align, freq=freq)
    sources_logret = sources.apply(lambda x: logret(x, dropna=False), axis=0)
    n_sources = sources_logret.shape[1]
    
    # miniization args
    cons = []
    bounds = None
    if pos_weights:
        # using bounds, instead of cons, works much better
        #cons.append({'type': 'python ', 'fun' : lambda x: np.min(x[1:])})
        if sum1:
            x_bound = (0, 1)
        else:
            x_bound = (0, None)
        bounds = [(None, None)] + ([x_bound] * n_sources)
    if sum1:
        if sum_max1:
            cons.append({'type': 'ineq', 'fun' : lambda x: 1-np.sum(x[1:])}) # sum<=1  same as   1-sum>=0
        else:
            cons.append({'type': 'eq', 'fun' : lambda x: np.sum(x[1:])-1})
        
    objective = value_objective if fit_values else returns_objective
    
    def run_optimize(rand_x0):
        n = sources_logret.shape[1]
        if rand_x0:
            x0 = np.random.rand(n+1)
            if sum1:
                x0 /= np.sum(x0)
        else:
            x0 = np.full(n+1, 1/n)
            #x0 += np.random.randn(n+1)*(1/n)
            #x0 = np.maximum(x0, 0)
            x0[0] = 0
    
        # minimize, to use constrains, we can choose from COBYLA / SLSQP / trust-constr

        # COBYLA: results are not stable, and vary greatly from run to run
        # also doesn't support equality constraint (sum1)
        #options={'rhobeg': 0.1, 'maxiter': 10000, 'disp': True, 'catol': 0.0002}
        #res = minimize(objective, x0, constraints=cons, method="COBYLA", options=options)

        # SLSQP: provides stable results from run to run, and support eq constraints (sum1)
        # using a much smaller eps than default works better (more stable and better results)
        options={'maxiter': 1000, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08/1000}
        res = minimize(objective, x0, constraints=cons, method="SLSQP", options=options, bounds=bounds)
        
        return res

    def getR2_vanilla(y, f):
        return sklearn.metrics.r2_score(y, f)
        
    def getR2(y, f):
        ssTot = np.sum((y-np.mean(y))**2)
        ssRes = np.sum((y-f)**2)
        return 1-ssRes/ssTot

    def getR2(y, f):
        y_mean_logret = mean_logret_series(y)
        ssTot = np.sum((y-y_mean_logret)**2)
        ssRes = np.sum((y-f)**2)
        return 1-ssRes/ssTot
        
    def getR2_lr(y, f):
        y_lr = lr(y)
        ssTot = np.sum((y-y_lr)**2)
        ssRes = np.sum((y-f)**2)
        return 1-ssRes/ssTot

    def getR2_lr_log(y, f):
        y_lr = lr(y)
        y_log = np.log(y)
        ssTot = np.sum((y_log - np.log(y_lr))**2)
        ssRes = np.sum((y_log - np.log(f))**2)
        return 1-ssRes/ssTot

    def finalize(res):
        # results
        pred = apply(res.x, bias=fit_values)
        pred.name = "~" + target.name + " - fit"

        if np.isnan(res.x).any():
            r2 = np.nan
            print("nan R^2")
        else:
            if fit_values:
                r2 = getR2_lr_log(target, pred)
            else:
                r2 = getR2_vanilla(logret(target), logret(pred))
            r2 = np.exp(r2) / math.e
            
        # calc adjusted R2
        n = sources.shape[0]
        k = sources.shape[1]
        r2 = 1-((1-r2)*(n-1)/(n-k-1))

        res["R^2"] = r2
        return pred

    
    # uniform x0 works best usually, but when it doesn't random seems to work well
    res = run_optimize(rand_x0=False)
    if not res.success:
        silent = False
        if not sum1 and sum_max1 and res["message"] == "Positive directional derivative for linesearch":
            silent = True
        #if not silent:
        print("lrret: 1st attempt failed with: " + res["message"])
        res2 = run_optimize(rand_x0=True)
        if not res2.success:
            silent = False
            if not sum1 and sum_max1 and res2["message"] == "Positive directional derivative for linesearch":
                silent = True
            #if not silent:
            print("lrret: 2nd attempt failed with: " + res2["message"])
        if res["R^2"] > res2["R^2"] and not np.isnan(res["R^2"]) and not (pos_weights and res["message"] == "Inequality constraints incompatible"):
            #if not silent:
            print(f"lrret: 1st attempt (uniform) was better, 1st:{res['R^2']}, 2nd: {res2['R^2']}")
        else:
            #if not silent:
            print(f"lrret: 2nd attempt (random) was better, 1st:{res['R^2']}, 2nd: {res2['R^2']}")
            res = res2

    names = sources.columns
    ser = pd.Series(dict(zip(names, [round(x, 6) for x in res.x[1:]])))
    ser = ser.sort_values(ascending=False)
    
    _pred = finalize(res)
    pred = _pred
    if True:
        #sources_dict = {s.name: s for s in sources}
        #d = Portfolio([(s, ser[getName(s)]*100) for s in orig_sources])
        d = (ser*100).to_dict()
        if True and not orig_target.name.startswith("~"):
            try:
                pred = name(get(d, mode=get_mode(orig_target.name)), get_pretty_name(orig_target.name) + " - fit")
                pred = pred / pred[_pred.index[0]] * _pred[0]
                port = dict_to_port_name(d, drop_zero=True, drop_100=True, use_sym_name=True)
                pred_str = f"{port} = {target.name} - fit"
            except:
                pred_str = '<NA>'
                warn("failed to get portfolio based on regerssion")

        #    pred, _pred = doAlign([pred, _pred])

    if show_res:
        show(pred, _pred, target, align=not fit_values, trim=False)
        print(f"R^2: {res['R^2']}")
    
    if pos_weights and np.any(ser < -0.001):
        print("lrret WARNING: pos_weights requirement violated!")
    
    if return_pred_fit:
        return _pred

    if return_pred:
        print(ser)
        return pred_str

    #if res_pred:
    res["pred"] = pred_str
        
    if return_res:
        res["ser"] = ser
        return res    
    
    if return_ser:
        return ser
    
def lrret_old(target, regressors, sum1=False):
    regressors = [get(x) for x in regressors]
    target = get(target)
    all = [unwrap(logret(x)) for x in (regressors + [target])]
    
    # based on: https://stats.stackexchange.com/questions/21565/how-do-i-fit-a-constrained-regression-in-r-so-that-coefficients-total-1?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # NOTE: note finished, not working
    if sum1:
        allOrig = all
        last = all[-2]
        all = [r - last for r in (all[:-2] + [all[-1]])]
        
    data = pd.DataFrame(all).T
    data = data.dropna()
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(X, y)
    
    if sum1:
        weights = np.append(regr.coef_, 1-np.sum(regr.coef_))
        
        all = allOrig
        data = pd.DataFrame(all).T
        data = data.dropna()
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(X, y)
        
        regr.coef_ = weights
    
    y_pred = regr.predict(X)

    
    print('Regressors:', [s.name for s in regressors])
    print('Coefficients:', regr.coef_)
    #print('Coefficients*:', list(regr.coef_) + [1-np.sum(regr.coef_)])
    #print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    print('Variance score r^2: %.3f' % sk.metrics.r2_score(y, y_pred))

    y_pred = i_logret(pd.Series(y_pred, X.index))
    y_pred.name = target.name + " fit"
    #y_pred = "fit"
    y_pred = Wrapper(y_pred)
    show(target , y_pred)
    return y_pred
        
def lrret_incremental(target, sources, show=True, show_steps=False, max_n=None, **lrret_args):
    if not isinstance(sources, list):
        sources = [sources]
    target, *sources = get([target] + sources, trim=True)
    sources = sources.copy()
    top = []
    cum_sources = []
    while len(sources) > 0:
        if max_n and len(top) == max_n:
            break
        allres = [lrret(target, cum_sources + [source], return_res=True, return_ser=False, res_pred=True, show_res=False, **lrret_args) for source in sources]
        max_i = np.argmax([res["R^2"] for res in allres])
        max_res = allres[max_i]
        max_source = sources[max_i]

        top.append((max_res["R^2"], max_source.name))
        cum_sources.append(max_source)
        del sources[max_i]
        
        port = dict_to_port_name((max_res["ser"]*100).to_dict())
        print(f"{port}    R^2: {max_res['R^2']:.3f}   start:{max_res['pred'].index[0]}")
    
        if show_steps:
            res = pd.Series(*list(zip(*top))).fillna(method='bfill')
            #clear_output()
            #plt.figure()
            show_series(res)
            plt.axhline(1, c="blue");
            plt.axhline(0.995, c="green");        
            plt.axhline(0.99, c="orange");        
            plt.ylim(ymax=1.003)        
            plt.title("Cumulative R^2")
            plt.show()

    res = pd.Series(*list(zip(*top))).fillna(method='bfill')
    if show:
        #clear_output()
        plt.figure()
        show_series(res)
        plt.axhline(1/math.e, c="orange");        
        plt.axhline(math.exp(0.9)/math.e, c="green");        
        plt.ylim(ymax=1.003)        
        plt.title("Cumulative R^2")
        plt.show()
            
    return res
        
def lrret_mutual_cross(*sources, show=True, **lrret_args):
    if len(sources) <= 1:
        return pd.Series()
    sources = get(sources, trim=True)
    res = []
    for target in sources:
        rest = [s for s in sources if s.name != target.name]
        #rest = sources.copy()
        #rest.remove(target)
        rs = lrret(target, rest, return_res=True, return_ser=False, show_res=False, **lrret_args)
        res.append((rs['R^2'], target.name))
        port = dict_to_port_name((rs["ser"]*100).to_dict(), drop_zero=True)
        print(f"{target.name}: {port}   R^2: {rs['R^2']:.3f}")
        

    res = pd.Series(*list(zip(*res))).fillna(method='bfill')
    res = res.sort_values()
    if show:
        show_series(res, figure=False)
        #plt.axhline(, c="blue");
        plt.axhline(1/math.e, c="orange");        
        plt.axhline(math.exp(0.9)/math.e, c="green");        
        plt.ylim(ymax=1.003)
        plt.title("mutual R^2")
        #plt.show()
            
    return res

def lrret_mutual_incremental(*sources, base=None, show=True, max_n=None, **lrret_args):
    if base is None:
        base = lc
    base, *sources = get([base] + list(sources), trim=True)
    cum_sources = [base]
    top = []
    while len(sources) > 0:
        if max_n and len(top) == max_n:
            break
        allres = [lrret(target, cum_sources, return_res=True, return_ser=False, show_res=False, **lrret_args) for target in sources]
        max_i = np.argmin([res["R^2"] for res in allres])
        max_res = allres[max_i]
        max_source = sources[max_i]

        top.append((max_res["R^2"], max_source.name))
        cum_sources.append(max_source)
        del sources[max_i]
        
        port = dict_to_port_name((max_res["ser"]*100).to_dict(), drop_zero=True)
        print(f"{max_source.name}: {port}   R^2: {max_res['R^2']:.3f}")
        
        if len(top) == 1:
            cum_sources.remove(base) # we only need the base for the first one
        

    res = pd.Series(*list(zip(*top))).fillna(method='bfill')
    if show:
        #clear_output()
        #plt.figure()
        show_series(res, figure=False)
        #plt.axhline(, c="blue");
        plt.axhline(1/math.e, c="orange");        
        plt.axhline(math.exp(0.9)/math.e, c="green");        
        plt.ylim(ymax=1.003)        
        plt.title("incremental R^2 (S&P500 seed)")
        plt.show()
            
    return res
           
def lrret_mutual(*sources, base=None, show=True, max_n=None, **lrret_args):
    print()
    print("Cross:")
    res_cross = lrret_mutual_cross(*sources, show=False)
    print()
    print("Incremental:")
    res_inc = lrret_mutual_incremental(*sources, show=False)    
    if show:
        plt.figure()
        show_series(res_cross, figure=True, label="Cross")
        #plt.gca().lines[-1].set_label("cross")
        show_series(res_inc, figure=False, label="Incremental")
        #plt.gca().lines[-1].set_label("inc")
        plt.axhline(1/math.e, c="orange");        
        plt.axhline(math.exp(0.9)/math.e, c="green");        
        plt.ylim(ymax=1.003)        
        plt.title("R^2")
        plt.legend()
        plt.show()
    

def mean_logret_series(y):
    res =  name(pd.Series(i_logret(np.full_like(y, logret(y).mean())), y.index), y.name + " mean logret")
    res *= y[0]/res[0]
    return res

def liquidation(s):
    return (s/s[0]-1)*0.75+1

def mean_series_perf(all):
    df = pd.DataFrame(lmap(ret, all)).T.dropna(how='all')
    ss = rpy(i_ret(df.mean(axis=1)))
    return name(ss, "~mean perf")

def median_series_perf(all):
    df = pd.DataFrame(lmap(ret, all)).T.dropna(how='all')
    ss = rpy(i_ret(df.median(axis=1)))
    return name(ss, "~median perf")

def median_series(lst, align):
    if align:
        lst = doAlign(lst)
    df = pd.DataFrame([unwrap(s) for s in lst]).T
    return name(rpy(df.median(axis=1)), "~median")

def mean_series(lst, align):
    if align:
        lst = doAlign(lst)
    df = pd.DataFrame([unwrap(s) for s in lst]).T
    return name(rpy(df.mean(axis=1)), "~mean")

def sdiv(a, b):
    if isinstance(a,list):
        return lmap(lambda x: sdiv(x, b), a)
    a, b = get([a, b])
    x = a / b
    x.name = get_pretty_name(a) + " / " + get_pretty_name(b)
    return x

# In[ ]:

def html_title(text):
    display(HTML(f"<h1>{text}</h1>"))

from IPython.core.display import Javascript
import time, os, stat

def save_notebook(verbose=True, sleep=True):
    Javascript('console.log(document.querySelector("div#save-notbook button").click())')
    if verbose:
        print("save requested, sleeping to ensure execution ..")
    if sleep:
        time.sleep(15)
    if verbose:
        print("done")

# save live notebook at first run to make sure it's the latest modified file in the folder (for later publishing)
save_notebook(False, False)

def publish(name=None):
    def file_age_in_seconds(pathname):
        return time.time() - os.stat(pathname)[stat.ST_MTIME]

    filename = get_ipython().getoutput('ls -t *.ipynb | grep -v /$ | head -1')
    filename = filename[0]

    age = int(file_age_in_seconds(filename))
    min_age = 5
    if age > min_age:
        print(filename + " file age is " + str(age) + " seconds old, auto saving current notebook ..")
        save_notebook()
        filename = get_ipython().getoutput('ls -t *.ipynb | grep -v /$ | head -1')
        filename = filename[0]
    
    if not name:
        name = str(uuid.uuid4().hex.upper())
    save()
    print("Publishing " + filename + " ..")
    res = subprocess.call(['bash', './publish.sh', name])
    if res == 0:
        print("published successfuly!")
        print("https://nbviewer.jupyter.org/github/ertpload/test/blob/master/__name__.ipynb".replace("__name__", name))
    else:
        res = subprocess.call(['bash', './../publish.sh', name])
        if res == 0:
            print("published successfuly!")
            print("https://nbviewer.jupyter.org/github/ertpload/test/blob/master/__name__.ipynb".replace("__name__", name))
        else:
            print("Failed!")


# In[ ]:


from IPython.display import display,Javascript 
def save():
    display(Javascript('IPython.notebook.save_checkpoint();'))


# In[ ]:


# make the plotly graphs look wider on mobile
from IPython.core.display import display, HTML
s = """
<style>
div.rendered_html {
    max-width: 10000px;
}
#nbextension-scratchpad {
    width: 80%;
}
.container {
    width: 95%;
}
</style>
"""
display(HTML(s))


# In[ ]:


# interception to auto-fetch hardcoded symbols e.g:
# show(SPY)
# this should run last in the framework code, or it attempts to download unrelated symbols :)

from IPython.core.inputtransformer import *
intercept = ipy == True
intercept = False # rewrite the code: https://ipython.readthedocs.io/en/stable/config/inputtransforms.html
if intercept and not "my_transformer_tokens_instance" in locals():
    #print("transformation hook init")
    attempted_implied_fetches = set()
    
    ip = get_ipython()

    @StatelessInputTransformer.wrap
    def my_transformer(line):
        if line.startswith("x"):
            return "specialcommand(" + repr(line) + ")"
        return line

    @TokenInputTransformer.wrap
    def my_transformer_tokens(tokens):
        global trimmed_messages
        trimmed_messages.clear()
        
        for i, x in enumerate(tokens):
            if x.type == 1 and x.string.isupper() and x.string.isalpha() and len(x.string) >= 2: ## type=1 is NAME token
                if i < len(tokens)-1 and tokens[i+1].type == 53 and tokens[i+1].string == "=":
                    attempted_implied_fetches.add(x.string)
                    continue
                if x.string in attempted_implied_fetches or x.string in ip.user_ns:
                    continue
                try:
                    ip.user_ns[x.string] = get(x.string)
                except:
                    print("Failed to fetch implied symbol: " + x.string)
                    attempted_implied_fetches.add(x.string)
        return tokens

    my_transformer_tokens_instance = my_transformer_tokens()
    
    ip.input_splitter.logical_line_transforms.append(my_transformer_tokens_instance)
    ip.input_transformer_manager.logical_line_transforms.append(my_transformer_tokens_instance)


# In[ ]:


def date(s):
    return pd.to_datetime(s, format="%Y-%m-%d")


# In[5]:

# another options for interception:
# ```python
# class VarWatcher(object):
#     def __init__(self, ip):
#         self.shell = ip
#         self.last_x = None
# 
#     def pre_execute(self):
#         if False:
#             for k in dir(self.shell):
#                 print(k, ":", getattr(self.shell, k))
#                 print()
#         #print("\n".join(dir(self.shell)))
#         if "content" in self.shell.parent_header:
#             code = self.shell.parent_header['content']['code']
#             self.shell.user_ns[code] = 42
#         #print(self.shell.user_ns.get('ASDF', None))
# 
#     def post_execute(self):
#         pass
#         #if self.shell.user_ns.get('x', None) != self.last_x:
#         #    print("x changed!")
# 
# def load_ipython_extension(ip):
#     vw = VarWatcher(ip)
#     ip.events.register('pre_execute', vw.pre_execute)
#     ip.events.register('post_execute', vw.post_execute)
#     
# ip = get_ipython()
# 
# load_ipython_extension(ip)   
# 
# ```

# In[ ]:


# def divs(symbolName, period=None, fill=False):
#     if isinstance(symbolName, tuple) and period is None:
#         symbolName, period = symbolName
#     if isinstance(symbolName, Wrapper) or isinstance(symbolName, pd.Series):
#         sym = symbolName
#         symbolName = symbolName.name
#     if symbolName.startswith("~"):
#         divs = sym[-1:0] # we just want an empty series with DatetimeIndex
#         #divs = pd.Series(index=pd.DatetimeIndex(freq="D"))
#         divs.name = symbolName
#     else:
#         divs = get(symbolName, mode="divs")
#         divs = divs[divs>0]
#     if period:
#         divs = wrap(divs.rolling(period).sum())
#     if fill:
#         price = get(symbolName)
#         divs = divs.reindex(price.index.union(divs.index), fill_value=0)
#     divs.name = divs.name + " divs"
#     return divs

def show_yield_types(*lst, drop_special_divs=False, **args):
    yields = lmap(partial(get_yield_types, drop_special_divs=drop_special_divs, **args), lst)
    rets = [get_named(x, cagr) for x in get(lst, trim=True)]
    show(*yields, rets, 0, ta=False, log=False, title=f"Yields {'without special divs' if drop_special_divs else ''}")


def show_income(*all, smooth, inf_adj=False):
    income = lmap(partial(get_income, smooth=smooth), all)
    if inf_adj:
        income = lmap(adj_inf, income)
    show(income, 0, ta=False, log=False, title=f"net income (smooth={smooth}) {'inf-adjusted' if inf_adj else ''}")

def show_cum_income(*all):
    income = lmap(get_cum_income, all)
    show(income, ta=False, log=False, legend=False, title="cumulative net income")

def show_cum_income_relative(*all, base):
    income = lmap(get_cum_income, all)
    base_income = get_cum_income(base)
    income = [sdiv(x, base_income) for x in income]
    show(income, ta=False, log=False, legend=False, title="relative cumulative net income")

def show_rr__yield_fees__mutual_dd_rolling_pr_SPY(*all):
    show_rr2(*all, ret_func=[get_curr_yield_rolling_no_fees, get_curr_yield_rolling], risk_func=mutual_dd_rolling_pr_SPY, title="Impact of fees on yield")

def show_comp(target, base, extra=None, mode="NTR", despike=False, cache=True):
    if extra is None:
        extra = []
    elif isinstance(extra, list):
        pass
    else:
        extra = [extra]
    analyze_assets(*extra, target=target, base=base, mode=mode, despike=despike, cache=cache)

def analyze_assets(*all, target=None, base=None, mode="NTR", start=None, end=None, despike=False, few=None, detailed=False, cache=True):
    if any(map(lambda x:isinstance(x, list), all)):
        raise Exception("analyze_assets individual argument cannot be lists")

    if len(all) == 1 and target is None and base is None:
        target = all[0]
        all = []
    has_target_and_base = not target is None and not base is None
    has_target = not target is None
    has_base = not base is None

    all = list(all)
    all_modes = set(map(lambda s: s.name.mode, filter(is_series, all + [target, base])))
    print(f"Analyzing assets with internal modes {list(all_modes)} and requested mode [{mode}]")

    all = get([target, base] + all, start=start, end=end, despike=despike, mode=mode, cache=cache) # note we don't trim
    target, base, *extra = all
    all = [s for s in all if not s is None]
    all_trim = get(all, trim=True)
    if few is None:
        few = len(all) <= 5

    # equity curve
    html_title("Equity Curves")
    if few:
        show(*all, (1, 'start'), (0.5, '50% draw-down'), trim=False, align='rel', title=mode + " equity") # use 0.5 instead of 0 to keep the log scale
        show_modes(*all)
        if detailed:
            show(lmap(adj_inf, lmap(price, all)), 1, title="real price")
    if has_target:
        show_min_max_bands(target)
    if has_target_and_base:
        r = (target / base).dropna()
        show_min_max_bands(r)
        show_modes_comp(target, base)
        show_port_flow_comp(target, base)

    # draw-down
    html_title("Draw-Down")
    if has_target_and_base:
        show_dd_price_actions(target, base, dd_func=dd)
        show_dd_price_actions(target, base, dd_func=dd_rolling)
    if has_target:
        show_dd_chunks(target)
    if few:
        show_dd(*all)
        show_dd(*all_trim, title_prefix='trimmed')

#############
#############

    # risk-return: cagr
    html_title("RR: cagr")
    bases = [mix(lc, gb, do_get=False), mix(i_ac, gb, do_get=False)]
    all_with_bases = get(all + bases, mode=mode)
    print("-------")
    #show_rr__cagr__mutual_dd_risk_rolling_pr_SPY(*all_with_bases)
    show_rr(*all_with_bases, risk_func=mutual_dd_rolling_pr_SPY)

    if detailed:
        show_rr(*get(all + bases, mode="TR"))
        show_rr(*get(all + bases, mode="NTR"))
        show_rr(*get(all + bases, mode="PR"))
        show_rr_modes(*all)
        show_rr_modes_mutual_dd_risk_rolling_SPY(*all)
    else:
        show_rr(*all_with_bases, risk_func=ulcer_pr)
        show_rr(*all_with_bases, risk_func=max_dd_pr)

    show_rr_capture_ratios(*all)

    if has_target_and_base:
        show_rr(mix(target, base))

    if few:
        html_title("AUM")
        show_aum(*all)

    # Yields
    if few:
        html_title("Yields")
        if has_target:
            show_yield(target, detailed=True)
        elif len(all) == 1:
            show_yield(all[0], detailed=True)
        if len(all) > 1:
            show_yield(*all, detailed=False)        
        show_yield_types(*all)
        show_yield_types(*all, drop_special_divs=True)

    # risk-return: Yields
    html_title("RR: yield")
    show_rr_yield(*all, risk_func=mutual_dd_rolling_pr_SPY)
    show_rr_yield(*all, risk_func=ulcer_pr)
    show_rr_yield(*all, risk_func=max_dd)
    show_rr(*all, ret_func=cagr, risk_func=get_start_yield, lr_fit=True, same_ratio=True)    
    # show_rr__yield__mutual_dd_risk_rolling_pr_SPY(*all)
    show_rr__yield_range__mutual_dd_rolling_pr_SPY(*all)
    show_rr__yield_fees__mutual_dd_rolling_pr_SPY(*all)
    show_rr__yield_min_cagrpr__mutual_dd_rolling_pr_SPY(*all)
    show_rr__yield_cagrpr__ulcerpr_trim(*all)
    
    if detailed:
        show_rr__yield_cagrpr__ulcerpr_notrim(*all)
        show_rr__yield_types__ulcer(*all)
    
    if detailed:
        show_rr_yield_tr_ntr(*all)
        # show_rr_modes(*all, ret_func=get_curr_yield_rolling, modes=['TR'], title='Risk - 12m Yield TR')
        show_rr(*get(all, mode='TR'), ret_func=get_curr_yield_rolling, title='Risk - 12m Yield TR')
    #show_rr_modes(*all, ret_func=get_curr_yield_rolling, modes=['NTR'], title='Risk - 12m Yield NTR')
    show_rr(*get(all, mode='NTR'), ret_func=get_curr_yield_rolling, title='Risk - 12m Yield NTR')

    show_rr__yield_ntr_pr_diff__pr(*all)
    if detailed:
        show_rr_yield_ntr_pr_diff_pr_full_alt(*all)
        show_rr_yield_ntr_pr_diff_pr_full_alt(*all, trim=False)

    # zscores
    html_title("z-scores")
    display_zscores(all, _cache=[None])

    # withdraw flows
    html_title("RR: flows")
    show_rr_flows(*all)

    # risk-return: cross risks
    html_title("RR: cross-risks")
    show_rr(*all, ret_func=ulcer_pr, risk_func=mutual_dd_rolling_pr_SPY)
    show_rr(*all, ret_func=max_dd_pr, risk_func=mutual_dd_rolling_pr_SPY)
    show_rr(*all, ret_func=mutual_dd_rolling_pr_TLT, risk_func=mutual_dd_rolling_pr_SPY, same_ratio=True)    
    show_rr2(*all, x_func=[mutual_dd_rolling_pr_SPY, mutual_dd_pr_SPY],  y_func=ulcer)
    show_rr2(*all, x_func=[mutual_dd_rolling_pr_SPY, mutual_dd_pr_SPY],  y_func=lrretm_beta_SPY, same_ratio=True)
    show_rr(*all, ret_func=get_downside_capture_SPY, risk_func=mutual_dd_rolling_pr_SPY, lr_fit=True)
    if detailed:
        show_rr(*all, ret_func=mutual_dd_rolling_pr_SPY_weighted, risk_func=mutual_dd_rolling_pr_SPY_unweighted, same_ratio=True)
    # show_rr(*all,                               risk_func=mutual_dd_pr_SPY, ret_func=ulcer,)
    # show_rr(*all,                               risk_func=mutual_dd_pr_SPY, ret_func=lrretm_beta_SPY, same_ratio=True)

#######################

    # Income
    html_title("Income")
    if few:
        show_income_ulcer(*all)
        show_income(*all, smooth=12)
        show_income(*all, smooth=3)
        show_income(*all, smooth=0)
        if detailed:
            show_income(*all, smooth=12, inf_adj=True)
            show_income(*all, smooth=3, inf_adj=True)
            show_income(*all, smooth=0, inf_adj=True)

        show_cum_income(*all)
        show_cum_income(*all_trim)
    if has_base:
        #show_cum_income_relative(*all_trim)
        show_cum_income_relative(*all_trim, base=base)

    # show(lmap(roi, all), ta=False, log=False, title="net ROI")

    # lrret
    html_title("Mutual lrret")
    if len(all) < 30:
        lrret_mutual(*all)

    # PCA / MDS
    html_title("MDS")
    if len(all) < 30:
        show_mds(*all)

def show_dd_chunks(s, min_days=10, min_depth=1, dd_func=dd, mode="PR"):
    s = get(s, mode=mode)
    ranges = get_dds(s, min_days=min_days, min_depth=min_depth, dd_func=dd_func)
    chunks = [s[i:j] for i,j,_ in ranges]
    show_dd(*chunks, mode=mode, dd_func=dd_func, legend=False, title_prefix=f"chunked {get_pretty_name_no_mode(s)}")

def show_dd_price_actions(target, base, min_days=0, min_depth=3, dd_func=dd_rolling):
    # target = pr(target)
    # base = pr(base)
    # base_dd = dd_func(base)
    # base_dd = trimBy(base_dd, target)
    # # dd_target = dd_func(target)
    # # base_dd, dd_target = doTrim([base_dd, dd_target], trim=True)
    target, base = get([target, base], mode="PR", trim=True)
    base_dd = dd_func(base)

    ranges = get_dds(base, min_days=min_days, min_depth=min_depth, dd_func=dd_func)

    if dd_func == dd:
#        target_price_actions = get_price_actions(target, ranges)
        target_price_actions = get_price_actions_with_rolling_base(target, ranges, base, n=0) # this is less efficient, but it takes care of alining the first draw-down
    elif dd_func == dd_rolling:
        target_price_actions = get_price_actions_with_rolling_base(target, ranges, base)
    else:
        raise Exception(f"unsupported dd_func: {get_func_name(dd_func)}")

    #base_dd_actions = [base_dd[i:j] for i,j,_ in ranges]
    show(base_dd, target_price_actions, -10, -20, -30, -40, -50, legend=False, ta=False, title=f"{get_func_name(dd_func)} price action {target.name} vs {base.name}")

    print("mutual_dd_risk: ", mutual_dd(target, base, dd_func=dd_func, min_depth=min_depth))

def show_dd(*all, mode="PR", dd_func=dd, legend=True, title_prefix='', do_get=True, **args):
    all = [s for s in all if not s is None]
    if do_get:
        all = get(all, mode=mode)
    for s in all:
        print(f"ulcer {get_name(s)}: {ulcer(s):.2f}")
    show(lmap(dd_func, all), -10, -20, -30, -40, -50, ta=False, title=f"{title_prefix} {mode} {get_func_name(dd_func)} draw-down", legend=legend, **args)

##########################################

def adj_inf(s):
    cpi = get(cpiUS)
    s = get(s)
    return name(s / cpi, s.name)

from functools import lru_cache

@lru_cache(maxsize=12)
def get_inflation(smooth=None, interpolate=True):
    cpi = get(cpiUS, interpolate=False)
    inf = (cpi / cpi.shift(12) - 1) * 100
    if interpolate:
        inf = inf.asfreq("D").interpolate()
    if smooth:
        inf = ma(inf, smooth)
    return name(inf, "inflation")

def _inf(s):
    inf = 100 * ret(s, 12).dropna().resample("M").sum()
    return name(inf, f"inf_{s.name}")

def _get_cpi(type, core, alt):
    if type == "cpi":
        if core:
             return get('FRED/CPILFESL@Q = cpiu_core', interpolate=False)
        if alt == 0:
            return get('RATEINF/CPI_USA@Q', interpolate=False)
        return get('FRED/CPIAUCSL@Q = cpiu', interpolate=False)
    if type == "pce":
        if core:
            return get('FRED/PCEPILFE@Q = pce_core', interpolate=False)
        else:
            return get('FRED/PCEPI@Q = pce', interpolate=False)
        #get('FRED/PCE@Q', interpolate=False)
    raise Exception(f"unknown cpi type: {type}")
    
def get_cpi(type='cpi', core=True, alt=0):
    return name(_get_cpi(type, core, alt), f"CPI-{type} {'core' if core else 'all'} {alt if alt else ''}")
    
def _get_inf(type, core, alt):
    if type == "cpi":
        if core:
            return _inf(get_cpi('cpi', core=True))
        else:
            if alt == 0:
                return get('RATEINF/INFLATION_USA@Q = inf_us', interpolate=False) # 1914+
            if alt == 1:
                return _inf(get_cpi('cpi', core, alt=0)) # same, calculated from CPI
            if alt == 2:
                return get('FRBC/USINFL^2@Q = inf_all_cpiu', interpolate=False).resample("M").sum() # same, less history
    elif type == "pce":
        if not core and alt == 1:
            return get('FRBC/USINFL^18@Q = inf_pce', interpolate=False).resample("M").sum() # the col name is wrong, this is the full, not core, index
        return _inf(get_cpi('pce', core=core))
    elif type == "ppi":
        if core:
            return get('FRBC/USINFL^14@Q = inf_ppi_core', interpolate=False).resample("M").sum()
        else:
            return get('FRBC/USINFL^10@Q = inf_ppi', interpolate=False).resample("M").sum()
    elif type == "mean":
        return mean_series([get_inf('cpi', core), get_inf('pce', core), get_inf('ppi', core)], align=False)
    raise Exception(f"unknown inf type: {type}")

def get_inf(type='mean', core=True, alt=0):
    return name(_get_inf(type, core, alt), f"Inflation-{type} {'core' if core else 'all'} {alt if alt else ''}")

##########################################

def get_real_yield(s, type=None):
    yld = get_yield(s, type=type)
    inf = get_inflation(365*7)
    return name(yld - inf, f"{yld.name} real").dropna()

def roi(s,value=100000):
    income = get_income(s, value=value, nis=False, per_month=True)
    return income/value * 100 * 12

def cum_cagr(s):
    s = get(s)
    days = (s.index - s.index[0]).days
    years = days/365
    val = s / s[0]
    return (np.power(val, 1/years)-1)*100
    
def modes(s, **get_args):
    res = [get(s, mode="TR", **get_args), get(s, mode="NTR", **get_args), get(s, mode="PR", **get_args)]
    # we can't rename exisitng series, it messes up future gets
    # res[0].name += "-TR"
    # res[1].name += "-NTR"
    # res[2].name += "-ITR"
    # res[3].name += "-PR"
    return res

# ## Generic Utils

# In[ ]:



# In[ ]:


# https://raw.githubusercontent.com/bsolomon1124/pyfinance/master/pyfinance/utils.py
# fixed to use 365 days for "D"

from pandas.tseries.frequencies import FreqGroup, get_freq_code
PERIODS_PER_YEAR = {
    FreqGroup.FR_ANN: 1.,
    FreqGroup.FR_QTR: 4.,
    FreqGroup.FR_MTH: 12.,
    FreqGroup.FR_WK: 52.,
    FreqGroup.FR_BUS: 252.,
    FreqGroup.FR_DAY: 365.,  # All days are business days
    FreqGroup.FR_HR: 365. * 6.5,
    FreqGroup.FR_MIN: 365. * 6.5 * 60,
    FreqGroup.FR_SEC: 365. * 6.5 * 60 * 60,
    FreqGroup.FR_MS: 365. * 6.5 * 60 * 60,
    FreqGroup.FR_US: 365. * 6.5 * 60 * 60 * 1000,
    FreqGroup.FR_NS: 365. * 6.5 * 60 * 60 * 1000 * 1000  # someday...
    }
def get_anlz_factor(freq):
    """Find the number of periods per year given a frequency.

    Parameters
    ----------
    freq : str
        Any frequency str or anchored offset str recognized by Pandas.

    Returns
    -------
    float

    Example
    -------
    >>> get_periods_per_year('D')
    252.0
    >>> get_periods_per_year('5D')  # 5-business-day periods per year
    50.4

    >>> get_periods_per_year('Q')
    4.0
    >>> get_periods_per_year('Q-DEC')
    4.0
    >>> get_periods_per_year('BQS-APR')
    4.0
    """

    # 'Q-NOV' would give us (2001, 1); we just want (2000, 1).
    try:
        base, mult = get_freq_code(freq)
    except ValueError:
        # The above will fail for a bunch of irregular frequencies, such
        # as 'Q-NOV' or 'BQS-APR'
        freq = freq.upper()
        if freq.startswith(('A-', 'BA-', 'AS-', 'BAS-')):
            freq = 'A'
        elif freq.startswith(('Q-', 'BQ-', 'QS-', 'BQS-')):
            freq = 'Q'
        elif freq in {'MS', 'BMS'}:
            freq = 'M'
        else:
            raise ValueError('Invalid frequency: %s' % freq)
        base, mult = get_freq_code(freq)
    return PERIODS_PER_YEAR[(base // 1000) * 1000] / mult

####################   PCA
from sklearn.decomposition import PCA



def get_ols_beta_dist(*all):
    df = get_ret_df(*all)
    n = df.shape[1]
    res = np.empty((n, n))
    for c1 in range(n):
        for c2 in range(n):
            y = df.iloc[:, c1]
            X = df.iloc[:, c2]
            beta1 = sm.OLS(y, X).fit().params[0]
            beta2 = sm.OLS(X, y).fit().params[0]
            x1 = np.array([beta1, beta2])
            x2 = np.abs(x1 - 1)
            val = x1[np.argmin(x2)]
            res[c1, c2] = val
    return pd.DataFrame(res, columns=df.columns, index=df.columns)


def get_beta_dist(*all, type):
    all = get(all)
    names = lmap(get_name, all)
    n = len(all)
    data = np.empty((n, n))
    for c1 in range(n):
        for c2 in range(n):
            if c1 == c2:
                val = 1
            else:
                y = all[c1]
                X = all[c2]
#                print(y.name, X.name)
                res = lrret(y, [X], return_res=True, show_res=False, sum1=(type=="R2"), pos_weights=(type=="R2"))
                if type == 'R2':
                    val = res['R^2']
                elif type == 'weight':
                    val = res['ser'][0]
            data[c1, c2] = val
    for c1 in range(n):
        for c2 in range(n):
            if type == "R2":
                val = max(data[c1, c2], data[c2, c1])
            elif type == "weight":
                x1 = np.array([data[c1, c2], data[c2, c1]])
                x2 = np.abs(x1 - 1)
                val = x1[np.argmin(x2)]
            data[c1, c2] = val
            data[c2, c1] = val
    df = pd.DataFrame(data, columns=names, index=names)
    return df


def get_ret_df(*lst):
    lst = get(lst, trim=True)
    df = pd.DataFrame({x.name: logret(x) for x in lst}).dropna()
    return df

def get_df(*lst):
    lst = get(lst, trim=True)
    df = pd.DataFrame({x.name: x for x in lst}).dropna()
    return df

def _show_mds(*all, type='cor'):
    if type == 'cor':
        df = get_ret_df(*all)
        sim = np.corrcoef(df.T)
        dist = 1-sim
    elif type == 'cov':
#         df = get_df(*all)
        df = get_ret_df(*all)
        sim = np.cov(df.T)
        np.fill_diagonal(sim, 1)
        dist = np.abs(1-sim)
    elif type == 'weight':
        dist = get_beta_dist(*all, type='weight')
        dist = np.abs(1 - dist)
    elif type == 'R2':
        dist = get_beta_dist(*all, type='R2')
        dist = 1 - dist
    elif type == 'beta':
        dist = get_ols_beta_dist(*all)
        dist = np.abs(1 - dist)

    names = lmap(get_name, all)
    #dist = dist - dist.mean(axis=1)
    if not isinstance(dist, pd.DataFrame):
        dist = pd.DataFrame(dist, columns=names, index=names)
    display(dist)
    
    pca = PCA(n_components=2)
    tr = pca.fit_transform(dist)
    plot_scatter_xy(tr[:, 0], tr[:, 1], names=names, title=f"{type} MDS")

def show_mds(*all, type=['cor', 'cov', 'beta', 'weight', 'R2']):
    if isinstance(type, str):
        type = [type]
    for t in type:
        _show_mds(*all, type=t)

####################   PCA


#################### Func Tools #####################
# e.g.:
# compose(cagr, despike, get)(SPY)
# partial(get, mode="TR")(SPY)


# def func(*functions, **args):
#     if len(functions) > 1:
#         f = compose(*functions)
#     else:
#         f = functions[0]
#     if len(args) > 0:
#         f = wrapped_partial(f, **args)
#     return f

#################### portfolio value and flow ############
def port_value(s, flow=None, cash=100000):
    start = None
    if is_series(s):
        start = s.index[0]

    pr = price(s)
    dv = divs(s)
    dv = dv * 0.75
    if not start is None:
        pr = pr[start:]
    if not start is None:
        dv = dv[start:]
 
    purchace_price = pr[0]
    units = cash / purchace_price
    dv = dv.reindex(pr.index).fillna(0)
    flow = flow.reindex(pr.index).fillna(0)
    res = pd.Series(0.0, pr.index)
    accum_cash = 0
    for dt in pr.index:
        cpr = pr[dt]
        cdv = dv[dt]
        cfl = flow[dt]
        if cdv > 0:
            accum_cash += units * cdv
        if cfl < 0: # we assume only negatives for now
            take_from_cash = min(accum_cash, abs(cfl))
            accum_cash -= take_from_cash
            cfl += take_from_cash
            if cfl != 0:
                diff_units = -cfl / cpr
                units -= diff_units
                if cpr > purchace_price:
                    gain = diff_units * (cpr - purchace_price)
                    tax = gain * 0.25
                    accum_cash -= tax
        if accum_cash > 0:
            new_units = accum_cash / cpr
            units += new_units
            accum_cash = 0
        c_val = units * cpr
        if c_val < 0:
            c_val = 0
        res[dt] = c_val
        
    res.name = get_name(s) + " -flow"
#    print(f"left with accum' cash {accum_cash}")
    return res

def get_flow(s, amount=None, rate=None, freq="M", inf=0.03):
    if amount is None and rate is None:
        rate = 0.04
#        raise Exception(f"amount or rate must be defined")
    pr = price(s)
    if amount is None and not rate is None:
        amount = rate * 100000 / 12
    flow = pd.Series(0.0, index=pr.index)
    flow = flow.resample("M").sum()
    flow -= amount
    mult = np.full(len(flow), math.pow(1+inf, 1/12)).cumprod()
    flow *= mult
    flow.name = f"{pr.name} flow"
    return flow

def get_port_with_flow(s, amount=None, rate=None, freq="M", inf=0.03):
    flow = get_flow(s, amount=amount, rate=rate, freq=freq, inf=inf)
    res = port_value(s, flow)
    if not rate is None:
        res.name = f"{s.name} {rate*100:.0f}%"
    return res

def show_port_with_flow(s, amount=None, rate=None, freq="M", inf=0.03, income_smooth=0):
    s_ntr = get(s, mode="NTR")
    flow = get_flow(s, amount=amount, rate=rate, freq=freq, inf=inf)
    s_flow = port_value(s, flow)
    show(s_ntr, s_flow, price(s))
    show(0, get_income(s, smooth=income_smooth), -flow, ta=False, log=False)
    wrate = -flow / s_flow * 12 * 100
    show(get_yield_true(s), wrate, 0, ta=False, log=False)
    
def show_port_flow_comp(target, base):
    base, target = get([base, target], trim=True, mode="NTR")
    flow = -get_income(target, smooth=0)
    base_flow = port_value(base, flow)
    target_pr = get(target, mode="PR")
    #show(base, base_flow, target, target_pr, 0, 1)
    show(base_flow, target_pr, 0, 1, title="base with flow vs target PR")

    relative_value = target_pr / base_flow
    relative_value.name = "target_pr / base_flow"
    relative_ntr = ntr(target) / ntr(base)
    relative_ntr.name = "relative NTR"
    show(relative_value, relative_ntr, 0, 1, title="relative base with flow / target PR")

def get_flows(s, n=None, rng=None):
    sers = []
    if rng is None and n is None:
        rng = range(5)
    if rng is None:
        rng = range(n)
    for i in rng:
        ser = get_port_with_flow(s, rate=i/100, inf=0)
        ser.name = f"{s.name} {i}%"
        sers.append(ser)
    return sers
    
########################################## 

def show_rr_flows(*all, n=None, rng=None):
    if rng is None:
        rng = [0, 5]
    all = get(all, trim=True)
    all = lmap(lambda x: get_flows(x, n=n, rng=rng), all)
    show_rr(*all, title="net flows")

def show_rr__yield_min_cagrpr__mutual_dd_rolling_pr_SPY(*all):
    show_rr2(2, 3, 4, 5, *all, y_func=[cagr_pr, lambda x: get_curr_yield_min2(ntr(x))], x_func=mutual_dd_rolling_pr_SPY, xlabel="mutual_dd_rolling_pr_SPY", ylabel="PR CAGR ➜ min net yield")

def show_rr__yield_prcagr__ulcerpr(*all, trim=True, title="PR CAGR ➜ 12m net yield vs PR ulcer"):
    show_rr2(2, 3, 4, 5, *all, trim=trim, g_func=pr, y_func=[cagr, lambda x: get_curr_yield_min2(ntr(x))], title=title, xlabel="PR ulcer", ylabel="PR CAGR ➜ 12m net yield")

def show_rr__yield_cagrpr__ulcerpr_trim(*all):
    all = get(all, trim=True)
    show_rr__yield_prcagr__ulcerpr(*all, title="PR CAGR ➜ 12m net yield vs PR ulcer (trim)")

def show_rr__yield_cagrpr__ulcerpr_notrim(*all):
    all = get(all, untrim=True)
    show_rr__yield_prcagr__ulcerpr(*all, trim=False, title="PR CAGR ➜ 12m net yield vs PR ulcer (no trim)")





############ special risk-return ####################

# def show_risk_itr_pr(*lst, title=None):
#     prs = get(lst, mode="PR", trim=True)
#     itrs = get(lst, mode="ITR", trim=True)
#     res = []
#     for pr, itr in zip(prs, itrs):
#         pr_ulcer = ulcer(pr)
#         x = [pr_ulcer, pr_ulcer]
#         y = [cagr(pr), cagr(itr)]
#         ser = pd.Series(y, index=x)
#         ser.name = pr.name
#         ser.names = [pr.name, '']
#         res.append(ser)
    
#     title = title or f"PR Risk - ITR Return"
#     plot_scatter(*res, title=title, xlabel="ulcer", ylabel="cagr", show_zero_point=True)

# def show_risk_itr_pr_diff(*lst, title=None):
#     prs = get(lst, mode="PR", trim=True)
#     itrs = get(lst, mode="ITR", trim=True)
#     res = []
#     for pr, itr in zip(prs, itrs):
#         pr_ulcer = ulcer(pr)
#         x = [pr_ulcer]
#         y = [cagr(itr)-cagr(pr)]
#         ser = pd.Series(y, index=x)
#         ser.name = pr.name
#         ser.names = [pr.name]
#         res.append(ser)
    
#     title = title or f"PR Risk - ITR Return"
#     plot_scatter(*res, title=title, xlabel="ulcer", ylabel="cagr", show_zero_point=True)
def pr_cagr_full(s):
    return cagr(get(s, untrim=True, mode="PR"))


def start_year_full(s):
    s = get(s, untrim=True)
    return str(s.index[0].year)
def start_year_full_with_name(s):
    return f"{s.name} {start_year_full(s)}"

def show_rr_yield_ntr_pr_diff_pr_full_alt(*lst, trim=True):
    alt_text = start_year_full if trim else start_year_full_with_name
    show_rr__yield_ntr_pr_diff__pr(*lst, alt_risk_func=pr_cagr_full, alt_risk_text=alt_text, trim=trim)

def show_rr__yield_ntr_pr_diff__pr(*lst, risk_func=cagr, alt_risk_func=pr_lr_cagr, alt_risk_text=None, title=None, trim=True):
    # date = getCommonDate(lst, 'start')
    # prs = get(lst, mode="PR", trim=date)
    # ntrs = get(lst, mode="NTR", trim=date)
    prs = get(lst, mode="PR", trim=trim)
    ntrs = get(lst, mode="NTR", trim=trim)
    res = []
    for pr, ntr in zip(prs, ntrs):
        if pr.shape[0] == 0:
            continue
        pr_ulcer = ulcer(pr)
        yld = get_curr_yield(get(pr, mode="NTR"), type='rolling')
        risk1 = risk_func(pr)
        x = [risk1, risk1]
        y = [cagr(ntr)-cagr(pr), yld]
        if not alt_risk_func is None:
            x.insert(0, alt_risk_func(pr))
            y.insert(0, y[0])
        ser = pd.Series(y, index=x)
        ser.name = pr.name.pretty_name_no_mode
        ser.names = [ser.name, '']
        if not alt_risk_func is None:
            txt = '' if alt_risk_text is None else alt_risk_text(pr)
            ser.names.insert(0, txt)
        res.append(ser)
    
    ally = flatten(res)
    mx = np.max(ally)+2
    mn = min(np.min(ally)-2, 0)
    def add_base(offset):
        base = pd.Series([mn, mx], [offset-mn, offset-mx])
        base.name = f"{offset}% net return"
        base.names = ['', '']
        res.append(base)
    add_base(0)
    add_base(5)
    add_base(10)
    add_base(15)
    res.append(5)
    title = title or f"PR Risk - NTR above PR Return"
    plot_scatter(*res, title=title, xlabel=f"{get_func_name(risk_func)} ➜ {get_func_name(alt_risk_func)}", ylabel="cagr(NTR) - cagr(PR) ➜ curr 12m net yield", show_zero_point=True, same_ratio=True)

############################# python utils ###########


############################

def get_real(s):
    return rename(get(s) / get(cpiUS), f"{get_pretty_name(s)} real")


###########################
def dd_match(s, base):
    s_orig = s
    s, base = get([s, base], mode="PR", untrim=False) # NOTE: untrim makes a HUGE difference in results
    s, base = dd_rolling(s), dd_rolling(base)
    s, base = doTrim([s, base], trim=s_orig, silent=True)
    total_base_dd = -base.sum()
    match = np.minimum(-s, -base).sum()
    return match / total_base_dd

#dd_match_SPY = partial(dd_match, base=SPY)
def dd_match_SPY(x):
    return dd_match(x, 'SPY')

# def show_rr_yield_dd_match_spy(*all):
#     show_rr(5, *all, ret_func=get_curr_yield_rolling, risk_func=dd_match_SPY)    

# def show_rr_cagr_dd_match_spy(*all):
#     show_rr(5, *all, ret_func=cagr, risk_func=dd_match_SPY)    

#################################


#################################

from pathlib import Path, PosixPath
def iter_cached_symbols(skip_fails=True):
    path = get_symbols_path()
    for file in Path(path).glob('**/*'):
        if not file.is_file():
            continue
        if str(file).endswith("_FAIL_") and skip_fails:
            continue
        symname = file.name.replace("._FAIL_", "").replace(".gz", "")
        source = file.parts[-2]
        yield (symname, source)

def get_all_cached_symbols():
    for sym, source in iter_cached_symbols():
        get(sym, source=source, error='ignore')

#list(iter_cached_symbols())
#get_all_cached_symbols()

#############################
def mean_series_incremental(all, mode, mean_func=mean_series):
    all = get(all, mode=mode) # prefetch once
    start_year = min([s.index[0].year for s in all])
    end_year = max([s.index[0].year for s in all])
    rng = range(start_year, end_year+1)
    last_count = 0
    res = []
    for year in rng:
        sers = get(all, start=year, trim=True, silent=True)
        if len(sers) == last_count:
            continue
        last_count = len(sers)
        mean_ser = mean_func(sers, align=True)
        mean_ser = rpy(mean_ser).sname(str(year))
        res.append(mean_ser)
    return res

def join_rel_align_series(all):
    if len(all) == 0:
        return all
    all = sorted(all, key=lambda s: s.index[0])
    base = all[0]
    res = base.copy()
    all = all[1:]
    for s in all:
        s = align_with(s, res)
        res, _ = expand(res, s)
        res[s.index[0]:] = np.nan
        res[s.index[0]:s.index[-1]] = s
    return rpy(res).sname("~joined").dropna()

def join_rel_align_mean_series_incremental(all, mode):
    sers = mean_series_incremental(all, mode=mode)
    return name(join_rel_align_series(sers), "mean")

def join_rel_align_median_series_incremental(all, mode):
    sers = mean_series_incremental(all, mode=mode, mean_func=median_series)
    return name(join_rel_align_series(sers), "median")

def show_mean_series_incremental(all, mode="NTR"):
    all_sers = mean_series_incremental(all, mode=mode)
    all_joined = join_rel_align_series(all_sers)
    show(align_rel(all_sers, base=all_joined), all_joined, ta=False, sort=False)
#############################
    
def show_aum(*all, extra=None, log=False, cache=True):
    show(lmap(partial(aum_flow, cache=cache), all), extra, ta=False, log=log, title="AUM flow")

# def show_aum_vs_return(s, cache=True):
#     flow = aum_flow(s, cache=cache)
#     s = pr(s)
#     s = s/s[0]-1
#     show(s*max(flow)/max(s), flow, ta=False, sort=False, title='AUM vs return')

def show_aum_vs_return(s, cache=True):
    flow = aum_flow(s, cache=cache)
    s = pr(s)
    s = s/s[0]
    rng = max(s) - min(s)
    flow = (flow - min(flow))/(max(flow)-min(flow))*rng+min(s)
    show(1, name(flow/s, "flow/price"), s, flow, ta=False, sort=False, title='AUM vs return')

#############################
from framework.cefs import *
#############################
