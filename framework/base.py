from datetime import datetime as dt
import numbers
import types
import string
import datetime

import pandas as pd
import numpy as np

import os.path

from framework.utils import *
from framework.symbol import *
from framework.RpySeries import *
from framework.data_sources import *
from framework.stats_basic import *
#from framework.stats_basic import *



def s_start(s):
    if s.shape[0] > 0:
        return s.index[0]
    return None

def s_end(s):
    return s.index[-1]

def getCommonDate(data, pos, agg=max, get_fault=False):
    data = flattenLists(data)
    data = [s for s in data if is_series(s)]
    if not data:
        warn('getCommonDate: no series in data')
        if get_fault:
            return None, None
        else:
            return None
    if pos == 'start':
        dates = [s_start(s) for s in data if s.shape[0] > 0]
    elif pos == 'end':
        dates = [s_end(s) for s in data if s.shape[0] > 0]
    else:
        raise Exception(f"Invalid pos: {pos}")
    if len(dates) == 0:
        warn('getCommonDate: no dates in data')
        if get_fault:
            return None, None
        else:
            return None
    val = agg(dates)
    if get_fault:
        names = [(s.name or "<noname>") for date, s in zip(dates, data) if date == val]
        fault = ", ".join(names[:2])
        if len(names) > 2:
            fault += f", +{len(names)} more"
        return val, fault
    return val

def doAlign(data):
    date = getCommonDate(data, 'start')
    if date is None:
        return data
    newArr = []
    for s in data:
        if is_series(s):
            #s = s / s[date] # this can sometime fail for messing data were not all series have the same index
            base = s[date:]
            if base.shape[0] == 0:
                continue
            if base[0] != 0:
                s = s / base.iloc[0]
        newArr.append(s)
    return newArr


def doTrim(data, silent=False, trim=True, trim_end=True):
    data = _doTrim(data, 'start', silent=silent, trim=trim)
    if trim_end:
        data = _doTrim(data, 'end', silent=silent, trim=not trim is False)
    return data

def _doTrim(data, pos, silent=False, trim=True):
    if trim is False or trim is None:
        return data

    # we should first dropna, as there is no point in trimming to a common date
    # where some of the series starts with nan's
    data = [s.dropna() if is_series(s) else s for s in data]

    if pos == 'start':
        agg = max
        r_agg = min
    else:
        agg = min
        r_agg = max
    
    # find common date
    if trim is True:
        if silent:
            date = getCommonDate(data, pos, agg=agg)
        else:
            date, max_fault = getCommonDate(data, pos, agg=agg, get_fault=True)
    elif is_series(trim):
        date = trim.index[0]
        max_fault = trim.name
    elif isinstance(trim, pd.Timestamp) or isinstance(trim, datetime.datetime) or isinstance(trim, datetime.date):
        date = trim
        max_fault = "date"
    elif isinstance(trim, int):
        date = datetime.datetime(trim, 1, 1)
        max_fault = "year"
    else:
        raise Exception(f"unsupported trim type {type(trim)}")
        
    # nothing to trim
    if date is None:
        if not silent:
            print("Unable to trim data")
        return data

    # trim
    newArr = []
    for s in data:
        if is_series(s):
            if pos == 'start':
                s = s[date:]
            else:
                s = s[:date]
            if s.shape[0] == 0:
                continue
        elif isinstance(s, list):
            if pos == 'start':
                s = [x[date:] for x in s]
            else:
                s = [x[:date] for x in s]
            s = [x for x in s if x.shape[0] > 0]
        elif isinstance(s, numbers.Real) or isinstance(s, pd.Timestamp):
            pass
        else:
            warn(f"not trimming type {type(s)}, {s}")
        newArr.append(s)

    # report results
    if not silent:
        min_date, min_fault = getCommonDate(data, pos, agg=r_agg, get_fault=True)
        if min_date != date:
            msg = f"trimmed |{pos}| data from {min_date:%Y-%m-%d} [{min_fault}] to {date:%Y-%m-%d} [{max_fault}]"
            print_norep(msg)

    return newArr



class GetConf:
    def __init__(self, splitAdj, divAdj, cache, cache_fails, mode, source, secondary):
        self.splitAdj = splitAdj
        self.divAdj = divAdj
        self.cache = cache
        self.cache_fails = cache_fails
        self.mode = mode
        self.source = source
        self.secondary = secondary


def getFrom(symbol, conf, error):
    # special handling for forex
    # if a match, if will recurse and return here with XXXUSD@CUR
    if len(symbol.name) == 6 and not symbol.source and not conf.source:
        parts = symbol.name[:3], symbol.name[3:]
        if parts[0] == "USD" or parts[1] == "USD":
            return getForex(parts[0], parts[1])
    
    source = symbol.source or conf.source or "AV"
    if not source in data_sources:
        raise Exception("Unsupported source: " + source)
    if not conf.secondary:
        return data_sources[source].get(symbol, conf)
    try:
        return data_sources[source].get(symbol, conf)
    except Exception as e:
        # if the source wasn't explicitly stated, try from secondary
        if not symbol.source and not conf.source:
            print(f"Failed to fetch {symbol} from {source}, trying from {conf.secondary} .. ", end="")
            try:
                res = data_sources[conf.secondary].get(symbol, conf)
            except Exception as e:
                if error == 'raise':
                    raise Exception from e
                return None
            print("DONE")
            return res
        else:
            if error == 'raise':
                raise Exception from e
            return None





class Portfolio():
    def __init__(self, items):
        self.items = items




# i_logret(weighted logret) - this is in effect daily rebalancing
# weighted get - this is in effect no rebalancing
def get_port(d, name, getArgs):
    if isinstance(d, str):
        res = parse_portfolio_def(d)
        if not res:
            raise Exception("Invalid portfolio definition: " + d)
        d = res
    if not isinstance(d, dict):
        raise Exception("Portfolio definition must be str or dict, was: " + type(d))        

    if not is_symbol(name):
        raise Exception("portfolio must have a valid Symbol as name")
    # if is_symbol(name):
    #     pass
    # elif isinstance(name, dict):
    #     name = Symbol(dict_to_port_name(d))
    # elif isinstance(name, str):
    #     name = Symbol(name)
    # else:
    #     raise Exception("a proper portfolio name must be specified")
        # parts = name.split("=")
        # if len(parts) == 2:
        #     name = parts[1].strip()

    rebal = name.rebal
    mode = name.mode
    if mode == 'divs':
        #raise Exception("port divs not supported")
        args = getArgs.copy() 
        args["trim"] = True
        syms = get(list(d.keys()), **args)
        syms = dict(zip(d.keys(), syms))
        df = pd.DataFrame(syms[k]*v/100 for k,v in d.items()).T
        df = df.dropna(how='all').fillna(0)
        res = df.sum(axis=1)
    else:

        # if getArgs['rebal'] == 'none':
        #     syms = get(list(d.keys()), **getArgs)
        #     syms = doTrim(syms)
        #     if getArgs['mode'] != 'PR':
        #         syms = doAlign(syms)
        #     syms = [s * w/100 for s, w in zip(syms, d.values())]
        #     res = pd.DataFrame(syms).sum()

        if rebal == 'none':
            syms = get(list(d.keys()), **getArgs)
            syms = doTrim(syms)
            if mode == 'PR':
                base = [s[0] * w for s, w in zip(syms, d.values())]
                base = np.sum(base) / np.sum(list(d.values()))
            else:
                base = 1
            syms = doAlign(syms)
            syms = [(s-1) * w/100 for s, w in zip(syms, d.values())]
            res = pd.DataFrame(syms).sum() + 1
            res = res * base


            # if getArgs['mode'] == 'PR':
            #     syms = [s * w/100 for s, w in zip(syms, d.values())]
            #     res = pd.DataFrame(syms).sum() / np.sum(list(d.values()))
            # else:
            #     syms = doAlign(syms)
            #     syms = [(s-1) * w/100 for s, w in zip(syms, d.values())]
            #     res = pd.DataFrame(syms).sum() + 1

        if rebal == 'day':
            args = getArgs.copy() 
            args["trim"] = True
            syms = get(list(d.keys()), **args)
            syms = dict(zip(d.keys(), syms))
            df = pd.DataFrame(logret(syms[k], fillna=True)*v/100 for k,v in d.items()).T
            df = df.dropna() # should we use any or all ?
            res = i_logret(df.sum(axis=1))

    res.name = name
    return res

def parse_portfolio_def(s):
    if isinstance(s, dict):
        return s
    if isinstance(s, Portfolio):
        return s
    if not isinstance(s, str):
        return None
    d = {}
    parts = s.split('=')
    if len(parts) == 2:
        s = parts[0].strip()
    parts = s.split("|")
    for p in parts:
        parts2 = p.split(":")
        if len(parts2) > 2:
            return None
        elif len(parts2) == 2:
            d[parts2[0]] = float(parts2[1])
        else:
            d[parts2[0]] = None
    
    # equal weights
    if np.all([x is None for x in d.values()]):
        if len(d) == 1: # a single equal weight, is just a symbol, not a portfolio
            return None
        d = {k: 100/len(d) for k in d.keys()}
    return d

def getNtr(s, getArgs, tax=0.25, alt_price_symbol=None):
    mode = getArgs["mode"]
    getArgs["mode"] = "PR"
    pr = get(alt_price_symbol if not alt_price_symbol is None else s, **getArgs)
    getArgs["mode"] = "divs"
    divs = get(s, **getArgs)
    getArgs["mode"] = mode

    if pr is None:
        return None
    
    if is_series(s):
        start = s.index[0]
        divs = divs[start:]
        pr = pr[start:]

    divs = divs * (1-tax)   # strip divs from their taxes
    divs = divs / pr        # get the div to price ratio
    divs = divs.fillna(0)   # fill gaps with zero
    r = 1 + divs          # add 1 for product later
    r = r.cumprod()         # build the cum prod ratio
    ntr = (pr * r).dropna() # mul the price with the ratio - this is the dividend reinvestment
    #ntr = wrap(ntr, s.name + " NTR")
    ntr.name = s.name
    return ntr

def get_intr(s, getArgs, alt_price_symbol=None):
    mode = getArgs.get("mode", None)
    
    getArgs["mode"] = "PR"
    pr = get(alt_price_symbol if not alt_price_symbol is None else s, **getArgs)
    
    getArgs["mode"] = "divs"
    dv = get(s, **getArgs)
    
    getArgs["mode"] = mode

    if is_series(s):
        start = s.index[0]
        pr = pr[start:]
        dv = dv[start:]
    
#     dv = divs(s)
#     pr = price(s)
    
    tax = 0.25
    dv = dv * (1-tax)   # strip divs from their taxes
    dv = dv.reindex(pr.index).fillna(0)
    res = dv.cumsum() + pr
    res.name = get_name(s)
    return res


def is_not_corrupt(s):
    if not is_series(s):
        return True
    if " flow" in s.name:
        return True
    if len(s) > 0 and (s.index[-1] - s.index[0]).days/365 > 50:
        return True
    if np.max(s) / np.min(s) > 1000:
        warn(f"Dropping corrupt series: {get_name(s)}")
        return False
    return True


def do_interpolate(s):
    s = s.reindex(pd.date_range(start=s.index[0], end=s.index[-1]))
    return s.interpolate()

def _despike(s, std, window, shift):
    if isinstance(s, list):
        return [despike(x) for x in s]
    s = unwrap(s)
    new_s = s.copy()
    ret = logret(s, dropna=False).fillna(0)
    new_s[(ret - ret.mean()).abs() > ret.shift(shift).rolling(window).std().fillna(ret.max()) * std] = np.nan
    return name(new_s.interpolate(), s.name)

# we despike from both directions, since the method has to warm up for the forst window
def despike(s, std=8, window=30, shift=10):
    os = s
    s = _despike(s, std=std, window=window, shift=shift)
    s = _despike(s[::-1], std=std, window=window, shift=shift)[::-1]
    # if np.any(os != s):
    #     print(f"{s.name} was despiked")
    return s



def get(symbol, source=None, cache=True, cache_fails=False, splitAdj=True, divAdj=True, adj=None, mode=None, secondary="Y", interpolate=True, despike=False, trim=False, untrim=False, remode=True, start=None, end=None, freq=None, rebal=None, silent=False, error='raise', drop_corrupt=True, drop_zero=True):
    # tmp
    # if isinstance(symbol, list) and len(symbol) == 2 and symbol[1] in data_sources.keys():
    #     raise Exception("Invalid get() API usage")

    if not error in ['raise', 'ignore']:
        raise Exception(f"error should be in [raise, ignore], not: {error}")

    if is_number(symbol):
        return symbol

    #print(f"get: {symbol} [{type(symbol)}] [source: {source}]")
    getArgs = {}
    getArgs["source"] = source
    getArgs["cache"] = cache
    getArgs["cache_fails"] = cache_fails
    getArgs["splitAdj"] = splitAdj
    getArgs["divAdj"] = divAdj
    getArgs["adj"] = adj
    getArgs["mode"] = mode
    getArgs["secondary"] = secondary
    getArgs["interpolate"] = interpolate
    getArgs["despike"] = despike
    getArgs["trim"] = trim
    #getArgs["reget"] = reget
    getArgs["untrim"] = untrim
    getArgs["remode"] = remode
    getArgs["start"] = start
    getArgs["end"] = end
    getArgs["freq"] = freq
    getArgs["rebal"] = rebal
    getArgs["silent"] = silent
    getArgs["error"] = error
    getArgs["drop_zero"] = drop_zero
    getArgs["drop_corrupt"] = drop_corrupt
    

    if symbol is None:
        return None
    
    if isinstance(symbol, tuple) or isinstance(symbol, map) or isinstance(symbol, types.GeneratorType):
        symbol = list(symbol)
    if isinstance(symbol, set):
        symbol = list(symbol)
    if isinstance(symbol, list):
        lst = symbol
        #if reget is None and trim == True:
        #    getArgs["reget"] = True # this is a sensible default behaviour
        lst = [get(s, **getArgs) for s in lst]
        if not start is None:
            lst = filterByStart(lst, start, silent=silent)
        if not end is None:
            lst = filterByEnd(lst, end)
        if not trim is False:
            lst = doTrim(lst, trim=trim, silent=silent)
        if drop_corrupt:
            lst = lfilter(is_not_corrupt, lst)
        return lst
    
    if isinstance(source, list):
        res = []
        for s in source:
            getArgs["source"] = s
            res.append(get(symbol, **getArgs))
        return res
    # support for yield period tuples, e.g.: (SPY, 4)
    #if isinstance(symbol, tuple) and len(symbol) == 2:
    #    symbol, _ = symbol
    
    reget = None
    if is_series(symbol):
        
        # these are regression series, we can't get them from sources (yet)
        if symbol.name and symbol.name.startswith("~"):
            reget = False

        if cache == False:
            reget = True

        if reget != False:
            
            # if a mode has changed, reget (and if not, keep source mode)
            if is_symbol(symbol.name) and symbol.name.mode != mode:
                if mode is None:
                    mode = symbol.name.mode # just in case we do reget (say if trim=True), keep the symbol mode
                elif remode:
                    reget = True

            # if a rebal has changed, reget (and if not, keep source mode)
            if is_symbol(symbol.name) and symbol.name.rebal != rebal:
                if rebal is None:
                    rebal = symbol.name.rebal # just in case we do reget (say if trim=True), keep the symbol rebal
                else:
                    reget = True

            # if a source has changed, reget (and if not, keep source source)
            if is_symbol(symbol.name) and symbol.name.source != source:
                if source is None:
                    source = symbol.name.source
                else:
                    reget = True

            # if any trim is requested, reget
            # why? trim alone should not imply a reget
            # if not trim is False:
            #     reget = True

            # this is to un-trim a trimmed series
            if untrim:
                reget = True
                if trim is False:
                    trim = True
            else:
                # this is to keep an existing trim, if we reget (say due to remode)
                if trim is False or trim is True:
                    trim = symbol

            # if not untrim and trim is False:
            #     trim = symbol

        if not reget:
            # if freq:
            #     symbol = symbol.asfreq(freq)
            # return symbol
            s = symbol
        else:
            symbol = symbol.name
    else:
        reget = True

    if reget:
        if symbol == "":
            raise Exception("attemping to get an empty string as symbol name")
        
        if "ignoredAssets" in globals() and ignoredAssets and symbol in ignoredAssets:
            return wrap(pd.Series(), "<empty>")

        # special handing for composite portfolios
        port = parse_portfolio_def(symbol)
        
        mode = mode or "TR"
        rebal = rebal or 'none'
        symbol = toSymbol(symbol, source, mode, rebal)

        if port:
            s = get_port(port, symbol, getArgs)
        else:
            if mode == "NTR":
                s = getNtr(symbol, getArgs)
            elif mode == "GTR":
                s = getNtr(symbol, getArgs, tax=0)
            elif mode == "ITR":
                s = get_intr(symbol, getArgs)
            else:
                if adj == False:
                    splitAdj = False
                    divAdj = False
                s = getFrom(symbol, GetConf(splitAdj, divAdj, cache, cache_fails, mode, source, secondary), error)
        if s is None: # can happen in error=='ignore'
            return None

        s.name = symbol
        if drop_zero:
            if np.any(s != 0):
                s = s[s != 0] # clean up broken yahoo data, etc ..

        if mode != "divs" and mode != "raw":        
            if despike:
                s = globals()["despike"](s)

            if interpolate and s.shape[0] > 0:
                s = s.reindex(pd.date_range(start=s.index[0], end=s.index[-1]))
                s = s.interpolate()


    # given we have "s" ready, some operations should be performed if reuested regardless if the symbols was reget or not
    if not trim is False:
        if s.shape[0] > 0: # this is an odd edge-case, trim() fails for empty series (with trim=s), so we just skip it
            trimmed = doTrim([s], trim=trim, silent=True)
            if len(trimmed) == 0:
                s = s[-1:0] # empty series
            else:
                s = trimmed[0]

    if freq:
        s = s.asfreq(freq)
    
    if s.shape[0] == 0:
        warn(f"get() is returning an empty series for {s.name}")

    #return s
    return rpy(s)

def filterByStart(lst, start=None, silent=False):
    if start is None:
        return lst
    start = get_date(start)
    res = [s for s in lst if not s is None and len(s) > 0 and s.index[0] <= start]
    dropped = [s for s in lst if s is None or len(s) == 0 or s.index[0] > start]
    
    #dropped = set(lst) - set(res)
    if not silent and len(dropped) > 0:
        dropped = ['None' if s is None else s.name  for s in dropped]
        print(f"start dropped: {', '.join(dropped)}")
    return res        

def filterByEnd(lst, end=None):
    if end is None:
        return lst
    end = get_date(end)
    res = [s for s in lst if not s is None and s.index[-1] >= end]
    dropped = [s for s in lst if s is None or s.index[-1] < end]
    
    #dropped = set(lst) - set(res)
    if len(dropped) > 0:
        dropped = ['None' if s is None else s.name  for s in dropped]
        print(f"end dropped: {', '.join(dropped)}")
    return res        


def get_date(x):
    if is_series(x):
        return x.index[0]
    elif isinstance(x, int):
        return datetime.datetime(x, 1, 1)
    elif isinstance(x, pd.Timestamp) or isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
        return x
    raise Exception(f"not supported date type {type(x)}")


def tr(s):
    return get(s, mode="TR")

def ntr(s):
    return get(s, mode="NTR")

def pr(sym):
    return get(sym, mode="PR")
price = pr

def reget_old_tickers(all, source=None, days_old=None):
    # if not source is None:
    #     all = get(etfs.all, source=source, error='ignore', cache_fails=True)
    # elif not all is None:
    #     pass
    # else:
    #     raise Exception("all or source must be defined")

    if isinstance(all[0], str):
        all = get(all, source=source, error='ignore')
    all = [s for s in all if not s is None and s.index[-1].date() <= dt.now().date() - datetime.timedelta(days=days_old)]
    if len(all) > 0:
        print(f"re-fetching {len(all)} symbols ..")
    else:
        print("All symbols are up-to date")
    _ = get(all, cache=False, source=source, error='ignore')
