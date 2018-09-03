# coding: utf-8

ipy = False
if "get_ipython" in globals():
    ipy = True
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

import quandl
quandl.ApiConfig.api_key = "9nrUn7Sm1SdoeLdQGQB-"

pd.core.common.is_list_like = pd.api.types.is_list_like # patch until pandas_datareader is fixed
import pandas_datareader
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies



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


from framework.pca import *


# In[ ]:


def pd_from_dict(d):
    return pd.DataFrame.from_dict(d, orient='index').T.sort_index()


# In[ ]:


# (hack) Global configs
conf_cache_disk = True
conf_cache_memory = True
conf_cache_fails = False

class GetConf:
    def __init__(self, splitAdj, divAdj, cache, mode, source, secondary):
        self.splitAdj = splitAdj
        self.divAdj = divAdj
        self.cache = cache
        self.mode = mode
        self.source = source
        self.secondary = secondary


# In[ ]:


if not "fetchCache" in globals():
    fetchCache = {}
    
if not "symbols_mem_cache" in globals():
    symbols_mem_cache = {}
    


# In[ ]:


class Portfolio():
    def __init__(self, items):
        self.items = items


class Symbol(str):
    def __init__(self, fullname):
        if is_symbol(fullname):
            raise Exception("should not set symbol name as symbol instance")

        self.fullname = fullname

        parts = fullname.split("=")
        if len(parts) == 2:
            fullname = parts[0].strip()
            self.nick = parts[1].strip()
        else:
            self.nick = None
        self.fullname_nonick = fullname
        
        parts = fullname.split("!")
        if len(parts) == 2:
            fullname = parts[0]
            self.currency = parts[1]
        else:
            self.currency = ""
            
        parts = fullname.split("@")
        self.name = parts[0] #legacy
        self.ticker = parts[0]
        if len(parts) == 2:
            self.source = parts[1]
        else:
            self.source = ""

        self.diskname = self.ticker
        if self.source:
            self.diskname += "@" + self.source

    @property
    def pretty_name(self):
        if not self.nick is None:
            res = self.nick
            #return f"{self.fullname_nonick} = {self.nick}"
        else:
            res = self.fullname
        if hasattr(self, "mode") and self.mode:
            res = f"{res} {self.mode}"
        return res

    @property
    def pretty_name_no_mode(self):
        if not self.nick is None:
            res = self.nick
            #return f"{self.fullname_nonick} = {self.nick}"
        else:
            res = self.fullname
        return res
        
    def __str__(self):
        return self.fullname # temp, to resolve the get, reget issue with named symbols

        if not self.nick is None:
            return self.nick
            #return f"{self.fullname_nonick} = {self.nick}"
        return self.fullname


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
    if symbol in ignoredAssets: return 0
    return get(symbol)[-1]

#def getForex(fromCur, toCur):
#    if fromCur == toCur: return 1
#    if toCur == "USD":
#        return get(fromCur + "=X", "Y")
#    if fromCur == "USD":
#        return get(toCur + "=X", "Y").map(lambda x: 1.0/x)

def getForex(fromCur, toCur, inv=False):
    if fromCur == toCur: return 1
    #tmp = get(fromCur + toCur + "@CUR")
    if inv:
        tmp = 1/getForex(toCur, fromCur, inv=False)
        tmp.name = fromCur + "/" + toCur + "@IC"
        return tmp
    tmp = get(fromCur + "/" + toCur + "@IC")
    tmp = tmp.reindex(pd.date_range(start=tmp.index[0], end=tmp.index[-1]))
    tmp = tmp.fillna(method="ffill")
    return tmp
    #return wrap(tmp, fromCur+toCur)

def convertSeries(s, fromCur, toCur, inv=False):
    if fromCur == toCur: return s
    rate = getForex(fromCur, toCur, inv=inv)
    s = get(s)
    s = (s*rate).dropna()
    return s
    
def convertToday(value, fromCur, toCur):
    if fromCur == toCur: return value
    return value * getForex(fromCur, toCur)[-1]


# In[ ]:

def get_pretty_name(s):
    return get_name(s, use_sym_name=False)

def get_pretty_name_no_mode(s):
    return get_name(s, use_sym_name=False, nomode=True)

def get_name(s, use_sym_name=False, nomode=False):
    if s is None:
        return ""
    if is_series(s):
        s = s.name
    if not is_symbol(s):
        s = Symbol(s)
    if use_sym_name:
        return s.fullname_nonick
    else:
        if nomode:
            return s.pretty_name_no_mode
        else:
            return s.pretty_name
getName = get_name

def toSymbol(sym, source, mode):
    if isinstance(sym, dict):
        sym = dict_to_port_name(sym, use_sym_name=True)
    if is_symbol(sym):
        res = Symbol(sym.fullname)
        res.mode = mode or sym.mode
        return res
    if isinstance(sym, str):
        if source is None:
            res = Symbol(sym)
        else:
            res = Symbol(sym + "@" + source)
        res.mode = mode
        return res
    assert False, "invalid type for Symbol: " + str(type(sym)) + ", " + str(sym)

class DataSource:
    
    def __init__(self, source):
        self.source = source
    
    def fetch(self, symbol, conf):
        pass
    
    def process(self, symbol, df, conf):
        pass
    
    def get(self, symbol, conf):
        global conf_cache_disk, conf_cache_memory, conf_cache_fails

        df = None

        mem_key = self.source + "#" + symbol.fullname
        
        # get from mem cache
        if conf.cache and conf_cache_memory:
            if mem_key in symbols_mem_cache:
                df = symbols_mem_cache[mem_key]
        
        # get from disk cache
        if df is None and conf.cache and conf_cache_disk:
            df = cache_get(symbol, self.source)
        
        # attempt to fetch the symbol
        if df is None:
            failpath = cache_file(symbol, self.source) + "._FAIL_"
            if os.path.isfile(failpath):
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(failpath))
                diff = datetime.datetime.now() - mtime
                if conf_cache_fails and diff.total_seconds() <= 24 * 3600:
                    raise Exception("Fetching has previously failed for {0}, will try again later".format(symbol))

            try:
                # Attempt to actually fetch the symbol
                if df is None:
                    print("Fetching %s from %s .. " % (symbol.fullname, self.source), end="")
                    df = self.fetch(symbol, conf)
                    print("DONE")
                if df is None:
                    print("FAILED")
                    raise Exception("Failed to fetch symbol: " + str(symbol) + " from " + self.source)
                if len(df) == 0:
                    print("FAILED")
                    raise Exception("Symbol fetched but is empty: " + str(symbol) + " from " + self.source)
            except:
                # save a note that we failed
                Path(failpath).touch()
                raise
        
        # write to disk cache
        cache_set(symbol, self.source, df)
        # write to mem cache
        symbols_mem_cache[mem_key] = df
        
        if conf.mode == "raw":
            return df
        else:
            res = self.process(symbol, df, conf)
        return res.sort_index()

fred_forex_codes = """
AUD	DEXUSAL
BRL	DEXBZUS
GBP	DEXUSUK
CAD	DEXCAUS
CNY	DEXCHUS
DKK	DEXDNUS
EUR	DEXUSEU
HKD	DEXHKUS
INR	DEXINUS
JPY	DEXJPUS
MYR	DEXMAUS
MXN	DEXMXUS
TWD	DEXTAUS
NOK	DEXNOUS
SGD	DEXSIUS
ZAR	DEXSFUS
KRW	DEXKOUS
LKR	DEXSLUS
SEK	DEXSDUS
CHF	DEXSZUS
VEF	DEXVZUS
"""

boe_forex_codes = """
AUD	XUDLADD
CAD	XUDLCDD
CNY	XUDLBK73
CZK	XUDLBK27
DKK	XUDLDKD
HKD	XUDLHDD
HUF	XUDLBK35
INR	XUDLBK64
NIS	XUDLBK65
JPY	XUDLJYD
LTL	XUDLBK38
MYR	XUDLBK66
NZD	XUDLNDD
NOK	XUDLNKD
PLN	XUDLBK49
GBP	XUDLGBD
RUB	XUDLBK69
SAR	XUDLSRD
SGD	XUDLSGD
ZAR	XUDLZRD
KRW	XUDLBK74
SEK	XUDLSKD
CHF	XUDLSFD
TWD	XUDLTWD
THB	XUDLBK72
TRY	XUDLBK75
"""

# https://blog.quandl.com/api-for-currency-data
class ForexDataSource(DataSource):
    def __init__(self, source):
        self.fred_code_map = dict([s.split("\t") for s in fred_forex_codes.split("\n")[1:-1]])
        self.boe_code_map = dict([s.split("\t") for s in boe_forex_codes.split("\n")[1:-1]])
        self.boe_code_map["ILS"] = self.boe_code_map["NIS"]
        super().__init__(source)
    
    def fetch(self, symbol, conf):
        assert len(symbol.name) == 6
        _from = symbol.name[:3]
        _to = symbol.name[3:]
        if _to != "USD" and _from != "USD":
            raise Exception("Can only convert to/from USD")
        invert = _from == "USD"
        curr = _to if invert else _from
        
        div100 = 1
        if curr == "GBC":
            div100 = 100
            curr = "GBP"
        
        if curr in self.fred_code_map:
            code = self.fred_code_map[curr]
            df = quandl.get("FRED/" + code)
            if code.endswith("US") != invert: # some of the FRED currencies are inverted vs the US dollar, argh..
                df = df.apply(lambda x: 1.0/x)
            return df / div100

        if curr in self.boe_code_map:
            code = self.boe_code_map[curr]
            df = quandl.get("BOE/" + code)
            if not invert: # not sure if some of BEO currencies are NOT inverted vs USD, checked a few and they weren't
                df = df.apply(lambda x: 1.0/x)
            return df / div100

        raise Exception("Currency pair is not supported: " + symbol.name)
        
    def process(self, symbol, df, conf):
        return df.iloc[:, 0]
      
# https://github.com/ranaroussi/fix-yahoo-finance
class YahooDataSource(DataSource):
    def fetch(self, symbol, conf):
        return pdr.get_data_yahoo(symbol.name, progress=False, actions=True)

    def adjustSplits(self, price, splits):
        r = splits[::-1].cumprod().shift().fillna(method="bfill")
        return price / r

    def process(self, symbol, df, conf):
        if conf.mode == "TR":
            assert conf.splitAdj and conf.divAdj
            return df["Adj Close"]
        elif conf.mode == "PR":
            # Yahoo "Close" data is split adjusted. 
            if conf.splitAdj:
                return df["Close"]
            else:
                return self.adjustSplits(df["Close"], df["Stock Splits"])
            # We find the unadjusted data using the splits data
            # splitMul = df["Stock Splits"][::-1].cumprod().shift().fillna(method="bfill")
            # return df["Close"] / splitMul        
        elif conf.mode == "divs":
            # Yahoo divs are NOT split adjusted, or are they?
            # if conf.splitAdj:
            #     return self.adjustSplits(df["Dividends"], df["Stock Splits"])
            # else:
            #     return df["Dividends"]
            return df["Dividends"]
        else:
            raise Exception("Unsupported mode [" + conf.mode + "] for YahooDataSource")

class QuandlDataSource(DataSource):
    def fetch(self, symbol, conf):
        return quandl.get(symbol.name)

    def process(self, symbol, df, conf):
        if conf.mode == "TR" or conf.mode == "PR":
            if "Close" in df.columns:
                return df["Close"]
            return df.iloc[:, 0]
        elif conf.mode == "divs":
            return df.iloc[:, 0][0:0] # empty
        else:
            raise Exception("Unsupported mode [" + conf.mode + "] for QuandlDataSource")

    
class GoogleDataSource(DataSource):
    def fetch(self, symbol, conf):
        return pandas_datareader.data.DataReader(symbol.name, 'google')

    def process(self, symbol, df, conf):
        return df["Close"]
    
from ratelimiter import RateLimiter
def limited(until):
    duration = int(round(until - time.time()))
    print('Rate limited, sleeping for {:d} seconds'.format(duration))
    
    
AV_API_KEY = 'BB18'
#AV_API_KEYS = ['BB18']
AV_API_KEYS = ['HBTU90Z2A5LZG7T1', 'SC2LFF11DAEBY5XY', 'PMZO6PMXLZ0RTDRA', 'LYSN4091GT6HW4WP']
class AlphaVantageDataSource(DataSource):

    def __init__(self, source):
        super(AlphaVantageDataSource, self).__init__(source)
        self.key_i = 0
        self.ratelimiter = RateLimiter(max_calls=4, period=60, callback=limited)
    
    def adjustSplits(self, price, splits):
        r = splits[::-1].cumprod().shift().fillna(method="bfill")
#        r = splits.cumprod()
        return price / r
    
    # AV sometimes have duplicate split multiplers, we only use the last one 
    def fixAVSplits(self, df):
        df = df.sort_index()
        split = df["8. split coefficient"]
        count = 0
        for t, s in list(split.items())[::-1]:
            if s == 1.0:
                count = 0
                continue
            count += 1
            if count == 1:
                continue
            if count > 1:
                split[t] = 1.0
        df["8. split coefficient"] = split
        return df

    def fetch(self, symbol, conf):
        with self.ratelimiter:
            key = AV_API_KEYS[self.key_i]
            self.key_i = (self.key_i + 1) % len(AV_API_KEYS)
            ts = TimeSeries(key=key, output_format='pandas')
            df, meta_data = ts.get_daily_adjusted(symbol.name, outputsize="full")
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
            df = self.fixAVSplits(df)
            return df

    def process(self, symbol, df, conf):
        if conf.mode == "TR":
            return df["5. adjusted close"]
        elif conf.mode == "PR":
            if conf.splitAdj:
                return self.adjustSplits(df["4. close"], df['8. split coefficient'])
            else:
                return df["4. close"]
        elif conf.mode == "divs":
            if conf.splitAdj:
                return self.adjustSplits(df["7. dividend amount"], df['8. split coefficient'])
            else:
                return df["7. dividend amount"]
            #return df["7. dividend amount"]
        else:
            raise Exception("Unsupported mode [" + conf.mode + "] for AlphaVantageDataSource")
        
class AlphaVantageCryptoDataSource(DataSource):

    def fetch(self, symbol, conf):
        cc = CryptoCurrencies(key=AV_API_KEY, output_format='pandas')
        df, meta_data = cc.get_digital_currency_daily(symbol=symbol.name, market='USD')
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return df

    def process(self, symbol, df, conf):
        return df['4a. close (USD)']

class CryptoCompareDataSource(DataSource):
    def fetch(self, symbol, conf):
        url = "https://min-api.cryptocompare.com/data/histoday?fsym=__sym__&tsym=USD&limit=600000&aggregate=1&e=CCCAGG"
        d = json.loads(requests.get(url.replace("__sym__", symbol.name)).text)
        df = pd.DataFrame(d["Data"])
        if len(df) == 0:
            return None
        df["time"] = pd.to_datetime(df.time, unit="s")
        df.set_index("time", inplace=True)
        return df

    def process(self, symbol, df, conf):
        return df.close

# NOTE: data is SPLIT adjusted, but has no dividends and is NOT DIVIDEND adjusted 
# NOTE: it has data all the way to the start, but returned result is capped in length for ~20 years
#       and results are trimmed from the END, not from the start. TBD to handle this properly.
#       for now we start at 1.1.2000
class InvestingComDataSource(DataSource):

    def getUrl(self, symbol):
        symbol = symbol.name
        data = {
            'search_text': symbol,
            'term': symbol, 
            'country_id': '0',
            'tab_id': 'All'
        }
        headers = {
                    'Origin': 'https://www.investing.com',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36',
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Referer': 'https://www.investing.com/search?q=' + symbol,
                    'X-Requested-With': 'XMLHttpRequest',
                    'Connection': 'keep-alive'    
                }
        r = requests.post("https://www.investing.com/search/service/search", data=data, headers=headers)
        res = r.text
        res = json.loads(res)
        return res["All"][0]["link"]
    
    def getCodes(self, url):
        url = "https://www.investing.com" + url + "-historical-data"
        
        headers = {
                    'Origin': 'https://www.investing.com',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36',
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Referer': 'https://www.investing.com/',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Connection': 'keep-alive'    
                }
        r = requests.get(url,headers=headers)
        text = r.text
        
        m = re.search("smlId:\s+(\d+)", text)
        smlId = m.group(1)
        
        m = re.search("pairId:\s+(\d+)", text)
        pairId = m.group(1)
        
        return pairId, smlId
    
    def getHtml(self, pairId, smlId):
        data = [
            'curr_id=' + pairId,
            'smlID=' + smlId,
            'header=',
            'st_date=01%2F01%2F2000',
            'end_date=01%2F01%2F2100',
            'interval_sec=Daily',
            'sort_col=date',
            'sort_ord=DESC', 
            'action=historical_data'
        ]
        data = "&".join(data)
        headers = {
            'Origin': 'https://www.investing.com',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'text/plain, */*; q=0.01',
            'Referer': 'https://www.investing.com/',
            'X-Requested-With': 'XMLHttpRequest',
            'Connection': 'keep-alive'    
        }
        r = requests.post("https://www.investing.com/instruments/HistoricalDataAjax", data=data, headers=headers)
        return r.text
    
    def fetch(self, symbol, conf):
        symbolUrl = self.getUrl(symbol)
        
        pairId, smlId = self.getCodes(symbolUrl)
        
        html = self.getHtml(pairId, smlId)
        #print(html)
        parsed_html = BeautifulSoup(html, "lxml")
        df = pd.DataFrame(columns=["date", "price"])
        
        for i, tr in enumerate(parsed_html.find_all("tr")[1:]): # skip header
            data = [x.get("data-real-value") for x in tr.find_all("td")]
            if len(data) == 0 or data[0] is None:
                continue
            date = datetime.datetime.utcfromtimestamp(int(data[0]))
            close = float(data[1].replace(",", ""))
            #open = data[2]
            #high = data[3]
            #low = data[4]
            #volume = data[5]
            df.loc[i] = [date, close]
            
        df = df.set_index("date")
        return df

    def process(self, symbol, df, conf):
        return df['price']

import time    
class JustEtfDataSource(DataSource):


    def parseDate(self, s):
        s = s.strip(" {x:Date.UTC()")
        p = [int(x) for x in s.split(",")]
        dt = datetime.datetime(p[0], p[1] + 1, p[2])
        return dt

    def parseDividends(self, x):
        x = re.split("data: \[", x)[1]
        x = re.split("\]\^,", x)[0]
        data = []
        for x in re.split("\},\{", x):
            p = re.split(", events: \{click: function\(\) \{  \}\}, title: 'D', text: 'Dividend ", x)
            dt = self.parseDate(p[0])
            p = p[1].strip("',id: }").split(" ")
            currency = p[0]
            value = float(p[1])
            data.append((dt, value))
        return pd.DataFrame(data, columns=['dt', 'divs']).set_index("dt")

    def parsePrice(self, s):
        data = []
        line = s
        t = "data: ["
        line = line[line.find(t) + len(t):]
        t = "^]^"
        line = line[:line.find(t)]
        #print(line)
        parts = line.split("^")
        for p in parts:
            p = p.strip("[],")
            p = p.split(")")
            value = float(p[1].replace(",", ""))
            dateStr = p[0].split("(")[1]
            p = [int(x) for x in dateStr.split(",")]
            dt = datetime.datetime(p[0], p[1] + 1, p[2])
            data.append((dt, value))
            #print(dt, value)
        df = pd.DataFrame(data, columns=['dt', 'price']).set_index("dt")
        return df

    def parseRawText(self, s):
        x = re.split("addSeries", s)
        df = self.parsePrice(x[1])
        divs = self.parseDividends(x[2])
        df["divs"] = divs["divs"]
        return df

    def getIsin(self, symbol):
        symbolName = symbol.name
        data = {
            'draw': '1',
            'start': '0', 
            'length': '25', 
            'search[regex]': 'false', 
            'lang': 'en', 
            'country': 'GB', 
            'universeType': 'private', 
            'etfsParams': 'query=' + symbolName, 
            'groupField': 'index', 
        }
        headers = {
                    'Accept-Encoding': 'gzip, deflate, br',
                }
        session = requests.Session()
        
        
        r = session.get("https://www.justetf.com/en/etf-profile.html?tab=chart&isin=IE00B5L65R35", headers=headers)
        
        r = session.post("https://www.justetf.com/servlet/etfs-table", data=data, headers=headers)
        res = r.text
        
        res = json.loads(res)
        for d in res["data"]:
            if d["ticker"] == symbolName:
                return (d["isin"], session)
        raise Exception("Symbol not found in source: " + str(symbol))
    
    def getData(self, isin, session, conf, raw=False):
        if not session:
            session = requests.Session()
            
        headers = {
                    'Accept-Encoding': 'gzip, deflate, br',
                }
        
        url3 = "https://www.justetf.com/uk/etf-profile.html?groupField=index&from=search&isin=" + isin + "&tab=chart"
        r = session.get(url3, headers=headers)

        r = session.get("https://www.justetf.com/sw.js", headers=headers)
        text = r.text

        headers = {
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9,he;q=0.8',
            'wicket-focusedelementid': 'id1b',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.62 Safari/537.36',
            'accept': 'text/xml',
            'referer': 'https://www.justetf.com/uk/etf-profile.html?groupField=index&from=search&isin=IE00B5L65R35&tab=chart',
            'authority': 'www.justetf.com',
            'wicket-ajax': 'true'
        }

        if conf.mode == "PR":
            headers["wicket-focusedelementid"] = "includePayment"
            url = "https://www.justetf.com/uk/?wicket:interface=:2:tabs:panel:chart:optionsPanel:selectContainer:includePaymentContainer:includePayment::IBehaviorListener:0:1&random=0.8852086768595453"
            r = session.post(url, headers=headers)
            text = r.text
            headers["wicket-focusedelementid"] = "id1b"
        
        url = "https://www.justetf.com/en/?wicket:interface=:0:tabs:panel:chart:dates:ptl_3y::IBehaviorListener:0:1&random=0.2525050377785838"
        r = session.get(url, headers=headers)
        text = r.text

        
        # PRICE (instead of percent change)
        data = { 'tabs:panel:chart:optionsPanel:selectContainer:valueType': 'market_value' }
        url = "https://www.justetf.com/en/?wicket:interface=:0:tabs:panel:chart:optionsPanel:selectContainer:valueType::IBehaviorListener:0:1&random=0.7560635418741075"
        r = session.post(url, headers=headers, data=data)
        text = r.text
        
        # CURRENCY
        #data = { 'tabs:panel:chart:optionsPanel:selectContainer:currencies': '3' }
        #url = "https://www.justetf.com/en/?wicket:interface=:0:tabs:panel:chart:optionsPanel:selectContainer:currencies::IBehaviorListener:0:1&random=0.8898086171718949"
        #r = session.post(url, headers=headers, data=data)
        #text = r.text
        
        
        
        url = "https://www.justetf.com/en/?wicket:interface=:0:tabs:panel:chart:dates:ptl_max::IBehaviorListener:0:1&random=0.2525050377785838"
        #url = "https://www.justetf.com/uk/?wicket:interface=:3:tabs:panel:chart:dates:ptl_max::IBehaviorListener:0:1"
        
        #plain_cookie = 'locale_=en_GB; universeCountry_=GB; universeDisclaimerAccepted_=false; JSESSIONID=5C4770C8CE62E823C17E292486D04112.production01; AWSALB=Wy2YQ+nfXWR+lTtsGly/hBDFD5pCCtYo/VxE0lIXBPlA/SdQDbRxhg+0q2E8UybYawqQiy3/1m2Bs4xvN8yFW3cs/2zy8385MuhGGCN/FUwnstSvbL7T8rfcV03k'
        #cj = requests.utils.cookiejar_from_dict(dict(p.split('=') for p in plain_cookie.split('; ')))
        #session.cookies = cj
        
        r = session.get(url, headers=headers)
        text = r.text
        #print(text)
        if raw:
            return text
        
        return self.parseRawText(text)
        
        
    
    def fetch(self, symbol, conf):
        return self.getData(symbol.name, None, conf)

    def process(self, symbol, df, conf):
        if conf.mode == "TR":
            return df["price"]
        elif conf.mode == "PR":
            raise Exception("Unsupported mode [" + conf.mode + "] for JustEtfDataSource")
        elif conf.mode == "divs":
            return df["divs"]
        else:
            raise Exception("Unsupported mode [" + conf.mode + "] for JustEtfDataSource")
        
        return df['price']
    
#x = JustEtfDataSource("XXX")
#isin, session = x.getIsin(Symbol("ERNS"))
#t = x.getData(isin, session)

#conf = lambda x: x
#conf.mode = "TR"
#t = x.getData("IE00B5L65R35", None, conf, True)

class BloombergDataSource(DataSource):
    def fetch(self, symbol, conf):
        headers = {
                    'Origin': 'https://www.bloomberg.com',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36',
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Referer': 'https://www.bloomberg.com',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Connection': 'keep-alive'    
                }
        url = "https://www.bloomberg.com/markets/api/bulk-time-series/price/__sym__?timeFrame=5_YEAR"
        sym = symbol.name.replace(";", ":")
        if not ':' in sym:
            sym += ":US" # default US equities
        text = requests.get(url.replace("__sym__", sym), headers=headers).text
        if len(text) < 100:
            raise Exception("Failed to fetch from B")
        d = json.loads(text)
        #print(d)
        df = pd.DataFrame(d[0]["price"])
        if len(df) == 0:
            return None
        df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d")
        df.set_index("date", inplace=True)
        return df

    def process(self, symbol, df, conf):
        if conf.mode == "TR":
            return df.value # we don't support TR, but let it slide
        elif conf.mode == "PR":
            return df.value
        elif conf.mode == "divs":
            return pd.Series()
        else:
            raise Exception("Unsupported mode [" + conf.mode + "] for AlphaVantageDataSource")

    
# source: https://info.tase.co.il/Heb/General/ETF/Pages/ETFGraph.aspx?companyID=001523&subDataType=0&shareID=01116441
# graphs -> "export data" menu
# source: https://www.tase.co.il/he/market_data/index/142/graph
# graphs -> "export data" menu
class TASEDataSource(DataSource):
    def fetch(self, symbol, conf):
        sym = symbol.name
        typ = None
        if 7 <= len(sym) <= 8:
            if sym.startswith("1"):
                sym = "0" + sym
            if sym.startswith("5"):
                sym = "0" + sym
            if sym.startswith("01"):
                url = "https://info.tase.co.il/_layouts/15/Tase/ManagementPages/Export.aspx?sn=none&GridId=128&ct=1&oid=__sym__&ot=1&lang=he-IL&cp=8&CmpId=001523&ExportType=3&cf=0&cv=0&cl=0&cgt=1&dFrom=__from__&dTo=__to__"
                typ = "sal"
            elif sym.startswith("05"):
                url = "https://info.tase.co.il/_layouts/15/Tase/ManagementPages/Export.aspx?sn=none&GridId=128&ct=3&oid=__sym__&ot=4&lang=he-IL&cp=8&fundC=0&ExportType=3&cf=0&cv=0&cl=0&cgt=1&dFrom=__from__&dTo=__to__"
                typ = "krn"
            else:
                raise Exception("unsupported prefix: " + sym)
                
            headers = {
                'Origin': 'https://info.tase.co.il',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'text/plain, */*; q=0.01',
                'Referer': 'https://info.tase.co.il/',
                'X-Requested-With': 'XMLHttpRequest',
                'Connection': 'keep-alive'    
            }                
        else:
            url = "https://api.tase.co.il/api/ChartData/ChartData/?ct=1&ot=2&lang=1&cf=0&cp=8&cv=0&cl=0&cgt=1&dFrom=__from__&dTo=__to__&oid=__sym__"
            typ = "idx"
            
            headers = {
                    'Origin': 'https://api.tase.co.il',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36',
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'text/plain, */*; q=0.01',
                    'Referer': 'https://api.tase.co.il/',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Connection': 'keep-alive'    
                }            
            
        url = url.replace("__from__", "01/01/2000")
        today = datetime.date.today().strftime("%d/%m/%Y")
        url = url.replace("__to__", today)
        url = url.replace("__sym__", sym)

        r = requests.get(url, headers=headers)        
        text = r.text
        
        if typ == "idx":
            js = json.loads(text)
            df = pd.read_json(json.dumps(js["PointsForHistoryChart"]))
            df["date"] = pd.to_datetime(df["TradeDate"], format="%d/%m/%Y")
            df["price"] = df["ClosingRate"]
        else:
            # skip junk data
            lines = text.split("\n")
            text = "\n".join(lines[4:])

            from io import StringIO
            text = StringIO(text)

            if typ == "sal":
                names = ["date", "adj_close", "close", "turnover"]
                price_col = "adj_close"
            if typ == "krn":
                names = ["date", "sell", "buy"]
                price_col = "sell"

            df = pd.read_csv(text, header=None, names=names)
            df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
            df["price"] = df[price_col]

        df = df.set_index("date")
        return df

    def process(self, symbol, df, conf):
        return df["price"]


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

def wrap(s, name=""):
    return s

    name = name or s.name
    #if not name:
    #    raise Exception("no name")
    if isinstance(s, pd.Series):
        s = Wrapper(s)
        s.name = name
    elif isinstance(s, Wrapper):
        s.name = name
    return s

def unwrap(s):
    if isinstance(s, Wrapper):
        return s.s
    return s

def name(s, n):
    if is_series(s):
        if is_symbol(n):
            s.name = n
        elif is_symbol(s.name):
            sym = Symbol(s.name.fullname_nonick + "=" + n)
            sym.mode = s.name.mode
            s.name = sym
        else:
            s.name = n
    return s
    
def rename(s, n):
    return name(s.copy(), n)
    
data_sources = {
    
    "TASE": TASEDataSource("TASE"),
    "B": BloombergDataSource("B"),
    "JT": JustEtfDataSource("JT"),
    "Y": YahooDataSource("Y"),
    "IC": InvestingComDataSource("IC"),
    "Q": QuandlDataSource("Q"),
    "AV": AlphaVantageDataSource("AV"),
    "CC": CryptoCompareDataSource("CC"),
    "CCAV": AlphaVantageCryptoDataSource("CCAV"),
    "CUR": ForexDataSource("CUR"),
    "G": GoogleDataSource("G")
               }

def getFrom(symbol, conf):
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
            res = data_sources[conf.secondary].get(symbol, conf)
            print("DONE")
            return res
        else:
            raise e

def format_filename(s):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c if c in valid_chars else '_' for c in s )
    filename = filename.replace(' ','_')
    return filename
    
import os.path
 
def cache_file(symbol, source):
    if os.path.isfile('Retropy_framework.ipynb'):
        base = ''
    elif os.path.isfile('../Retropy_framework.ipynb'):
        base = '../'
    elif os.path.isfile('../../Retropy_framework.ipynb'):
        base = '../../'
    else:
        raise Exception('base path not found')
    
    filepath = os.path.join(base, "symbols", source, format_filename(symbol.ticker))
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return filepath

def cache_clear():
    global symbols_mem_cache
    dirpath = "symbols"
    if not os.path.exists(dirpath):
        print(f"path not found: {dirpath}")
    import shutil
    shutil.rmtree(dirpath)        
    symbols_mem_cache = {}
    print("cache cleared")
    
def cache_get(symbol, source):
    filepath = cache_file(symbol, source)
    if os.path.exists(filepath):
        #res = pd.read_csv(filepath, squeeze=True, names=["date", "value"], index_col="date")
        res = pd.read_csv(filepath, squeeze=False, index_col="date")
        res.index = pd.to_datetime(res.index, format="%Y-%m-%d")
        return res
    return None

def cache_set(symbol, source, s):
    filepath = cache_file(symbol, source)
    s.to_csv(filepath, date_format="%Y-%m-%d", index_label="date")

def dict_to_port_name(d, rnd=1, drop_zero=False, drop_100=False, use_sym_name=False):
    res = []
    for k, v in d.items():
        if drop_zero and v == 0:
            continue
        if drop_100 and v == 100:
            res.append(f"{getName(k, use_sym_name=use_sym_name)}")
        else:
            res.append(f"{getName(k, use_sym_name=use_sym_name)}:{round(v, rnd)}")
    return "|".join(res)
    
# def get_port(d, name, getArgs):
#     if isinstance(d, str):
#         res = parse_portfolio_def(d)
#         if not res:
#             raise Exception("Invalid portfolio definition: " + d)
#         d = res
#     if not isinstance(d, dict):
#         raise Exception("Portfolio definition must be str or dict, was: " + type(d))        
#     if isinstance(name, dict):
#         name = dict_to_port_name(d)
#     if isinstance(name, str):
#         parts = name.split("=")
#         if len(parts) == 2:
#             name = parts[1].strip()
#     args = getArgs.copy()
#     args["trim"] = True
#     syms = get(list(d.keys()), **args)
#     syms = dict(zip(d.keys(), syms))
#     if args['mode'] == 'divs':
#         #raise Exception("port divs not supported")
#         df = pd.DataFrame(syms[k]*v/100 for k,v in d.items()).T
#         df = df.dropna(how='all').fillna(0)
#         res = df.sum(axis=1)
#     else:
#         df = pd.DataFrame(logret(syms[k], fillna=True)*v/100 for k,v in d.items()).T
#         df = df.dropna() # should we use any or all ?
#         res = i_logret(df.sum(axis=1))
#     res.name = name
#     return res

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

    if is_symbol(name):
        pass
    elif isinstance(name, dict):
        name = Symbol(dict_to_port_name(d))
    elif isinstance(name, str):
        name = Symbol(name)
    else:
        raise Exception("a proper portfolio name must be specified")
        # parts = name.split("=")
        # if len(parts) == 2:
        #     name = parts[1].strip()

    if getArgs['mode'] == 'divs':
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

        if getArgs['rebal'] == 'none':
            syms = get(list(d.keys()), **getArgs)
            syms = doTrim(syms)
            if getArgs['mode'] == 'PR':
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

        if getArgs['rebal'] == 'day':
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

def getNtr(s, getArgs):
    mode = getArgs["mode"]
    getArgs["mode"] = "PR"
    pr = get(s, **getArgs)
    getArgs["mode"] = "divs"
    divs = get(s, **getArgs)
    getArgs["mode"] = mode
    
    tax = 0.25
    divs = divs * (1-tax)   # strip divs from their taxes
    divs = divs / pr        # get the div to price ratio
    divs = divs.fillna(0)   # fill gaps with zero
    r = 1 + divs          # add 1 for product later
    r = r.cumprod()         # build the cum prod ratio
    ntr = (pr * r).dropna() # mul the price with the ratio - this is the dividend reinvestment
    #ntr = wrap(ntr, s.name + " NTR")
    ntr.name = s.name
    return ntr

def get_intr(s, getArgs):
    mode = getArgs.get("mode", None)
    
    getArgs["mode"] = "PR"
    pr = get(s, **getArgs)
    
    getArgs["mode"] = "divs"
    dv = get(s, **getArgs)
    
    getArgs["mode"] = mode

    if is_series(s):
        start = s.index[0]
        pr = pr[start:]
        divs = divs[start:]
    
#     dv = divs(s)
#     pr = price(s)
    
    tax = 0.25
    dv = dv * (1-tax)   # strip divs from their taxes
    dv = dv.reindex(pr.index).fillna(0)
    res = dv.cumsum() + pr
    res.name = get_name(s)
    return res

def is_symbol(s):
    return type(s).__name__ == "Symbol"
    # isinstance(s, Symbol) - this doesn't work, as the Symbol class seems to have seperate instances in the python and Jupyter scopes

def get(symbol, source=None, cache=True, splitAdj=True, divAdj=True, adj=None, mode=None, secondary="Y", interpolate=True, despike=True, trim=False, untrim=False, remode=True, start=None, freq=None, rebal='none', silent=False):

    # tmp
    # if isinstance(symbol, list) and len(symbol) == 2 and symbol[1] in data_sources.keys():
    #     raise Exception("Invalid get() API usage")

    #print(f"get: {symbol} [{type(symbol)}] [source: {source}]")
    getArgs = {}
    getArgs["source"] = source
    getArgs["cache"] = cache
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
    getArgs["freq"] = freq
    getArgs["rebal"] = rebal
    getArgs["silent"] = silent

    if symbol is None:
        return None
    
    if isinstance(symbol, tuple) or isinstance(symbol, map) or isinstance(symbol, types.GeneratorType):
        symbol = list(symbol)
    if isinstance(symbol, list):
        lst = symbol
        #if reget is None and trim == True:
        #    getArgs["reget"] = True # this is a sensible default behaviour
        lst = [get(s, **getArgs) for s in lst]
        if not start is None:
            lst = filterByStart(lst, start)
        if not trim is False:
            lst = doTrim(lst, trim=trim)
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
    
    if is_series(symbol):
        reget = None
        
        # these are regression series, we can't get them from sources (yet)
        if symbol.name and symbol.name.startswith("~"):
            reget = False

        if reget != False:
            
            # if a mode has changed, reget (and if not, keep source mode)
            if is_symbol(symbol.name) and symbol.name.mode != mode:
                if mode is None:
                    mode = symbol.name.mode # just in case we do reget (say if trim=True), keep the symbol mode
                elif remode:
                    reget = True

            # if any trim is requested, reget
            # why? trim alone should not imply a reget
            # if not trim is False:
            #     reget = True

            # this is to un-trim a trimmed series
            if untrim:
                reget = True

            # this is to keep an existing trim, if we reget (say due to remode)
            if not untrim and trim is False:
                trim = symbol

        if not reget:
            if freq:
                symbol = symbol.asfreq(freq)
            return symbol
        symbol = symbol.name

    if symbol == "":
        raise Exception("attemping to get an empty string as symbol name")
    
    if "ignoredAssets" in globals() and ignoredAssets and symbol in ignoredAssets:
        return wrap(pd.Series(), "<empty>")

    # special handing for composite portfolios
    port = parse_portfolio_def(symbol)
    
    mode = mode or "TR"
    symbol = toSymbol(symbol, source, mode)

    if port:
        s = get_port(port, symbol, getArgs)
    else:
        if mode == "NTR":
            s = getNtr(symbol, getArgs)
        elif mode == "ITR":
            s = get_intr(symbol, getArgs)
        else:
            if adj == False:
                splitAdj = False
                divAdj = False
            s = getFrom(symbol, GetConf(splitAdj, divAdj, cache, mode, source, secondary))

    s.name = symbol
    if np.any(s != 0):
        s = s[s != 0] # clean up broken yahoo data, etc ..
    
    if despike:
        s = globals()["despike"](s)

    if not trim is False:
        trimmed = doTrim([s], trim=trim, silent=True)
        if len(trimmed) == 0:
            s = s[-1:0] # empty series
        else:
            s = trimmed[0]

    if interpolate and s.shape[0] > 0 and mode != "divs" and mode != "raw":
        s = s.reindex(pd.date_range(start=s.index[0], end=s.index[-1]))
        s = s.interpolate()

    if freq:
        s = s.asfreq(freq)
    
    return s


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

def is_named_number(val):
    return isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], numbers.Real) and isinstance(val[1], str)    

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
            data.append(go.Scatter(x=val.index, y=val, name=name, text=text, mode=mode, textposition='middle right'))
            start_date = _start(val)
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
        trace = go.Scatter(
            name=col,
            x=s.index.to_datetime(),
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
        bgcolor='#FFFFFFBB',bordercolor='#888888',borderwidth=1,
        font=dict(family='sans-serif',size=12,color='#000'),
    )    
    layout = go.Layout(margin=margin, legend=legend, title=title,
        #showlegend=True,
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

def _start(s):
    if s.shape[0] > 0:
        return s.index[0]
    return None

def _end(s):
    return s.index[-1]

def getCommonDate(data, pos, agg=max, get_fault=False):
    data = flattenLists(data)
    data = [s for s in data if is_series(s)]
    if not data:
        if get_fault:
            return None, None
        else:
            return None
    if pos == 'start':
        dates = [_start(s) for s in data if s.shape[0] > 0]
    elif pos == 'end':
        dates = [_end(s) for s in data if s.shape[0] > 0]
    else:
        raise Exception(f"Invalid pos: {pos}")
    if len(dates) == 0:
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

def is_series(x):
    return isinstance(x, pd.Series) or isinstance(x, Wrapper)

trimmed_messages = set()
def doTrim(data, silent=False, trim=True):
    data = _doTrim(data, 'start', silent=silent, trim=trim)
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
        elif isinstance(s, numbers.Real):
            pass
        else:
            warn(f"not trimming type {type(s)}, {s}")
        newArr.append(s)

    # report results
    if not silent:
        min_date, min_fault = getCommonDate(data, pos, agg=r_agg, get_fault=True)
        if min_date != date:
            msg = f"trimmed |{pos}| data from {min_date:%Y-%m-%d} [{min_fault}] to {date:%Y-%m-%d} [{max_fault}]"
            if not msg in trimmed_messages:
                trimmed_messages.add(msg)
                print(msg)

    return newArr

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

def sync(a, b):
    idx = a.index.intersection(b.index)
    a = a.reindex(idx)
    b = b.reindex(idx)
    return a, b

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
                s = s / base[0]
        newArr.append(s)
    return newArr

def align_with(s, w):
    if s.index[0] in w:
        return s * w[s.index[0]] / s[0]
    if w.index[0] in s:
        return s * w[0] / s[w.index[0]]
    raise Exception(f"Cannot align {get_pretty_name(s)} with {get_pretty_name(w)}, no common start date found")

def align_rel(all, base=None):
    if len(all) == 0:
        return all
    all = sorted(all, key=lambda s: s.index[0])
    if base is None:
        base = all[0]
        res = [base]
        all = all[1:]
    else:
        res = []
    for s in all:
        s = align_with(s, base)
        res.append(s)
    #    base = s
    return res

def doClean(data):
    return [s.dropna() if is_series(s) else s for s in data]

def try_parse_date(s, format):
    try:
        return datetime.datetime.strptime(s, format)
    except ValueError:
        return None    

def easy_try_parse_date(s):
    return try_parse_date(s, "%d/%m/%Y") or try_parse_date(s, "%d.%m.%Y") or try_parse_date(s, "%d-%m-%Y")
    
def show(*data, trim=True, align=True, align_base=None, ta=True, cache=None, mode=None, source=None, remode=None, untrim=None, silent=False, **plotArgs):
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
    dataSeries = [s for s in data if is_series(s)]
    if not ta:
        trim = False
        align = False
    if any([s[unwrap(s)<0].any() for s in dataSeries]):
        align = False
    if trim: data = doTrim(data, trim=trim)
    if align:
        if align == "rel":
            data = align_rel(data, base=align_base)
        else:
            data = doAlign(data)
        
    if not silent:
        plot(*data, **plotArgs)
    else:
        return dataSeries

def show_series(s, **args):
    show_scatter(range(len(s)), s.values, lines=True, annotations=s.index, show_zero_point=False, **args)
    
def show_scatter(xs, ys, setlim=True, lines=False, color=None, annotations=None, xlabel=None, ylabel=None, label=None, fixed_aspect_ratio=False, show_zero_point=False, fixtext=False, figure=False):
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
        if fixed_aspect_ratio:
            plt.figure(figsize=(12, 12))
        else:
            plt.figure(figsize=(16, 12))
    if lines:
        plt.plot(xs, ys, marker="o", color=color, label=label)
    else:
        plt.scatter(xs, ys, color=color, label=label)
    if setlim:
        if fixed_aspect_ratio:
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
    show(*lmap(modes, lst), **args)

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
    
    show_scatter(x, y, fixed_aspect_ratio=True, xlabel=x_sym.name, ylabel=y_sym.name)

def warn(*arg, **args):
    print(*arg, file=sys.stderr, **args)


def reduce_series(lst, g_func=None, y_func=None, x_func=None):
    if g_func:
        ys = [[y_func(gf(s)) for gf in g_func] for s in lst]
        xs = [[x_func(gf(s)) for gf in g_func] for s in lst]
    else:
        ys = [[yf(s) for yf in y_func] for s in lst]
        xs = [[x_func(s)] * len(y_func) for s in lst]
    
    res = [pd.Series(y, x) for y, x in zip(ys, xs)]
    res = [name(r, get_name(s)) for r, s in zip(res, lst)]
    return res

# experimental
def show_risk_return2(*lst, g_func=None, y_func=None, x_func=None):
    y_func = y_func or cagr
    x_func = x_func or ulcer
    r = reduce_series(lst, g_func=g_func, y_func=y_func, x_func=x_func)
    plot_scatter(*r, show_zero_point=True)
# e.g.:
# show_risk_return2(*all, g_func=[ft.partial(get, despike=False), get])

def show_risk_return(*lst, ret_func=None, risk_func=None, trim=True, **args):
    if ret_func is None: ret_func = cagr
    if risk_func is None: risk_func = ulcer
    lst = get(lst, trim=trim)
    lst = [x if isinstance(x, list) else [x] for x in lst]
    res = [get_risk_return_series(x, ret_func=ret_func, risk_func=risk_func) for x in lst]
    args['show_zero_point'] = True
    set_if_none(args, 'title',  "Risk-Return")
    plot_scatter(*res, xlabel=risk_func.__name__, ylabel=ret_func.__name__, **args)

showRiskReturn = show_risk_return # legacy

def get_risk_return_series(lst, ret_func, risk_func, **args):
    if len(lst) == 0:
        return
    lst = [get(s) for s in lst]
    ys = [ret_func(unwrap(s)) for s in lst]
    xs = [risk_func(unwrap(s)) for s in lst]
    names = [get_pretty_name_no_mode(s.name) for s in lst]

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

def show_risk_return_modes(*lst, ret_func=None, modes=['TR', 'NTR', 'PR'], title=None):
    def get_data(lst, mode):
        return get(lst, mode=mode, despike=True, trim=True)

    data_lst = [get_data(lst, mode) for mode in modes]
    all = [list(tup) for tup in zip(*data_lst)]
    show_risk_return(*all, ret_func=ret_func, title=title)
    #showRiskReturn(*ntr, ret_func=ret_func)
    #for a, b in zip(tr, ntr):
    #    showRiskReturn([a, b], setlim=False, lines=True, ret_func=ret_func, annotations=False)    

def show_risk_yield_types(*lst, ret_func=None, types=['true', 'normal', 'rolling'], mode="TR", title=None):
    def get_data(lst, type):
        yld = [get_curr_yield(s, type=type) for s in lst]
        rsk = lmap(ulcer, lst)
        return pd.Series(yld, rsk)

    lst = get(lst, mode=mode, trim=True, despike=True)
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

    lst = get(lst, mode=mode, trim=True, despike=True)
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


def show_risk_yield(*lst, title="Risk - NORMAL Yield TR-NTR"):
    show_risk_return_modes(*lst, ret_func=get_curr_yield_normal, modes=['TR', 'NTR'], title=title)


def show_min_max_bands(symbol, n=365, showSymbol=False):
    x = get(symbol)
    a = mmax(x, n)
    b = mmin(x, n)
    c = mm(x, n)
    if showSymbol:
        show(c, a, b, x, ta=False)
    else:
        show(c, a, b, ta=False)
        

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
    
def mcagr(s, n=365, dropna=True):
    return name(roll_ts(s, cagr, n, dropna=dropna), s.name + " cagr")

def mstd(s, n=365, dropna=True):
    res = name(ret(s).rolling(n).std()*math.sqrt(n)*100, s.name + " std")
    if dropna:
        res = res.dropna()
    return res

def msharpe(s, n=365, dropna=True):
    return name(mcagr(s, n, dropna) / mstd(s, n, dropna), s.name + " sharpe")

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
    
def ret(s):
    return s.pct_change()

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

def dd(x):
    x = get(x)
    x = unwrap(x)
    x = x.dropna()
    res = (x / np.maximum.accumulate(x) - 1) * 100
    return res
    
def percentile(s, p):
    return s.quantile(p/100)   


# In[ ]:


from scipy.optimize import minimize

def prep_as_df(target, sources, as_logret=False, as_geom_value=False, freq=None):
    if not isinstance(sources, list):
        sources = [sources]

    target = get(target)
    sources = get(sources)
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
          return_res=False, return_ser=True, return_pred=False, res_pred=False, show_res=True, freq=None, obj="log_sum_sq"):
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
    orig_target = get(target)

    target, sources = prep_as_df(target, sources, as_geom_value=fit_values, freq=freq)
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
        pred.name = target.name + " - fit"

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
        if True:
            pred = name(get(d, mode=orig_target.name.mode), orig_target.name.pretty_name + " - fit")
            pred = pred / pred[_pred.index[0]] * _pred[0]
            port = dict_to_port_name(d, drop_zero=True, drop_100=True, use_sym_name=True)
            pred_str = f"{port} = {target.name} - fit"

        #    pred, _pred = doAlign([pred, _pred])

    if show_res:
        show(pred, _pred, target, align=not fit_values, trim=False)
        print(f"R^2: {res['R^2']}")
    
    if pos_weights and np.any(ser < -0.001):
        print("lrret WARNING: pos_weights requirement violated!")
    
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
    
def lr(y):
    X = np.arange(y.shape[0])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    pred = model.predict(X)
    pred = name(pd.Series(pred, y.index), y.name + " fit")
    return pred
    
def lr_beta(y, X=None, pvalue=False):
    if X is None:
        X = np.arange(y.shape[0])
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        if pvalue:
            return model.params[1], model.pvalues[1]
        return model.params[1]
    else:
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        if pvalue:
            return model.params[1], model.pvalues[1]
        return model.params[1]
    
def lrret_beta(y, X, freq=None, pvalue=False):
    y = get(y, freq=freq)
    X = get(X, freq=freq)
    y = logret(y)
    X = logret(X)
    y, X = sync(y, X)
    return lr_beta(y, X, pvalue=pvalue)

def mean_logret_series(y):
    res =  name(pd.Series(i_logret(np.full_like(y, logret(y).mean())), y.index), y.name + " mean logret")
    res *= y[0]/res[0]
    return res

def liquidation(s):
    return (s/s[0]-1)*0.75+1

def getMedianSer(lst):
    df = pd.DataFrame([unwrap(s) for s in lst]).T
    return wrap(df.median(axis=1), "median")

def getMeanSer(lst):
    df = pd.DataFrame([unwrap(s) for s in lst]).T
    return wrap(df.mean(axis=1), "mean")

def sdiv(a, b):
    a, b = get([a, b])
    x = a / b
    x.name = a.name.pretty_name + " / " + b.name.pretty_name
    return x

# In[ ]:


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
</style>
"""
display(HTML(s))


# In[ ]:


# interception to auto-fetch hardcoded symbols e.g:
# show(SPY)
# this should run last in the framework code, or it attempts to download unrelated symbols :)

from IPython.core.inputtransformer import *
intercept = ipy == True
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


if not "fixed_globals_once" in globals():

    # ************* SYMBOLS ***************
    # these are shorthand variables representing asset classes

    # ==== SPECIAL ====
    # https://www.federalreserve.gov/pubs/bulletin/2005/winter05_index.pdf
    # Nominal Daily
    usdMajor = 'FRED/DTWEXM@Q' # Trade Weighted U.S. Dollar Index: Major Currencies
    usdBroad = 'FRED/DTWEXB@Q' # Trade Weighted U.S. Dollar Index: Broad
    usdOther = 'FRED/DTWEXO@Q' # Trade Weighted U.S. Dollar Index: Other Important Trading Partners
    # Nominal Monthly
    usdMajorM = 'FRED/TWEXMMTH@Q'
    usdBroadM = 'FRED/TWEXBMTH@Q'
    usdOtherM = 'FRED/TWEXOMTH@Q'
    # Real Monthly
    usdMajorReal = 'FRED/TWEXMPA@Q' # Real Trade Weighted U.S. Dollar Index: Major Currencies
    usdBroadReal = 'FRED/TWEXBPA@Q' # Real Trade Weighted U.S. Dollar Index: Broad
    usdOtherReal = 'FRED/TWEXOPA@Q' # Real Trade Weighted U.S. Dollar Index: Other Important Trading Partners
    usd = usdBroad

    cpiUS ='RATEINF/CPI_USA@Q'


    #bitcoinAvg = price("BAVERAGE/USD@Q") # data 2010-2016
    #bitcoinBitstamp = price("BCHARTS/BITSTAMPUSD@Q") # data 2011-now

    # ==== STOCKS ====
    # Global
    g_ac = 'VTSMX:45|VGTSX:55' # VTWSX, VT # global all-cap
    d_ac = 'URTH' # developed world
    # US
    ac = 'VTSMX' # VTI # all-cap
    lc = 'VFINX' # VOO, SPY # large-cap
    mc = 'VIMSX' # VO # mid-cap
    sc = 'NAESX' # VB # small-cap
    mcc = 'BRSIX' # micro-cap
    lcv = 'VIVAX' # IUSV # large-cap-value
    mcv = 'VMVIX' # mid-cap-value
    scv = 'VISVX' # VBR # small-cap-value
    lcg = 'VIGRX' # large-cap-growth 
    mcg = 'VMGIX' # mid-cap-growth
    scg = 'VISGX' # VBK # small-cap-growth
    # ex-US
    i_ac = 'VGTSX' # VXUS # intl' all-cap
    i_sc = 'VINEX' # VSS, SCZ # intl' small-cap
    i_dev = 'VTMGX' # EFA, VEA # intl' developed
    i_acv = 'DFIVX' # EFV # intl' all-cap-value
    i_scv = 'DISVX' # DLS # intl' small-cap-value
    em_ac = 'VEIEX' # VWO # emerging markets
#    em = em_ac # legacy
    em_sc = 'EEMS' # emerging markets small cap
    fr_ac = 'FRN' # FM # frontier markets

    # ==== BONDS ====
    # US GOVT
    sgb = 'VFISX' # SHY, VGSH # short term govt bonds
    tips = 'VIPSX' # TIP # inflation protected treasuries
    lgb = 'VUSTX' # TLT, VGLT # long govt bonds
    elgb = 'PEDIX@Y' # EDV # extra-long (extended duration) govt bonds, note PEDIX is missing it's divs in AV
    gb = 'VFITX' # IEI # intermediate govt bonds
    fgb = 'TFLO' # floating govt bonds
    # US CORP 
    cb = 'MFBFX' # LQD # corp bonds
    scb = 'VCSH' # short-term-corp-bonds
    lcb = 'VCLT' # long-term-corp-bonds
    fcb = 'FLOT' # floating corp bonds
    # US CORP+GOVT
    gcb = 'VBMFX' # AGG, BND # govt/corp bonds
    sgcb = 'VFSTX' # BSV # short-term-govt-corp-bonds
    # International
    i_tips = 'WIP' # # intl' local currency inflation protected bonds
    i_gcbUsd = 'PFORX' # BNDX # ex-US govt/copr bonds (USD hedged)
    i_gbLcl = 'BEGBX' # (getBwx()) BWX, IGOV # ex-US govt bonds (non hedged)
#    i_gb = i_gbLcl # legacy
    i_cb = 'PIGLX' # PICB, ex-US corp bonds
    i_cjb = 'IHY' # intl-corp-junk-bonds
    g_gcbLcl = 'PIGLX' # Global bonds (non hedged)
    g_gcbUsd = 'PGBIX' # Global bonds (USD hedged)
    g_sgcb = 'LDUR' # Global short-term govt-corp bonds
    g_usgcb = 'MINT' # Global ultra-short-term govt-corp bonds
    em_gbUsd = 'FNMIX' # VWOB, EMB # emerging market govt bonds (USD hedged)
#    emb = em_gbUsd # legacy
    em_gbLcl = 'PELBX' # LEMB, EBND, EMLC emerging-markets-govt-bonds (local currency) [LEMB Yahoo data is broken]
    em_cjb = 'EMHY' # emerging-markets-corp-junk-bonds
    cjb = 'VWEHX' # JNK, HYG # junk bonds
#    junk = cjb # legacy
    scjb = 'HYS' # short-term-corp-junk-bonds
    aggg_idx = "LEGATRUU;IND@B" # AGGG.L Global bonds unhedged (TR - Total Return)

    # ==== CASH ====
    rfr = 'SHV' # BIL # risk free return (1-3 month t-bills)
    cash = rfr # SHV # risk free return
    cashLike = 'VFISX:30' # a poor approximation for rfr returns 

    # ==== OTHER ====
    fedRate = 'FRED/DFF@Q'
    reit = 'DFREX' # VNQ # REIT
    i_reit = 'RWX' # VNQI # ex-US REIT
    g_reit = 'DFREX:50|RWX:50' # RWO # global REIT
    gold = 'LBMA/GOLD@Q' # GLD # gold
    silver = 'LBMA/SILVER@Q' # SLV # silver
    palladium = 'LPPM/PALL@Q'
    platinum = 'LPPM/PLAT@Q'
    #metals = gold|silver|palladium|platinum # GLTR # precious metals (VGPMX is a stocks fund)
    comm = 'DBC' # # commodities
    oilWtiQ = 'FRED/DCOILWTICO@Q'
    oilBrentQ = 'FRED/DCOILBRENTEU@Q'
    oilBrentK = 'oil-prices@OKFN' # only loads first series which is brent
    eden = 'EdenAlpha@MAN'

    # ==== INDICES ====
    spxPR = '^GSPC'
    spxTR = '^SP500TR'
    spx = spxPR
    

    # ==== TASE ====
    # exactly the same data as from TASE, but less indices supported
    ta125_IC = 'TA125@IC'
    ta35_IC = 'TA35@IC'

    # https://www.tase.co.il/he/market_data/indices
    ta35 = "142@TASE = TA-35"
    ta125 = "137@TASE = TA-125"
    ta90 = "143@TASE = TA-90"
    taSME60 = "147@TASE = TA-SME60"
    telDiv = "166@TASE = TA-Div"
    telAllShare = "168@TASE = TA-AllShare"
    ta_stocks = [ta35, ta125, ta90, taSME60, telDiv, telAllShare]

    taBonds = "601@TASE = IL-Bonds"
    taGovtBonds = "602@TASE = TA-GovtBonds"
    taCorpBonds = "603@TASE = TA-CorpBonds"
    taTips = "604@TASE = TA-Tips"
    taGovtTips = "605@TASE = TA-GovtTips"
    taCorpTips = "606@TASE = TA-CorpTips"
    ta_bonds = [taBonds, taGovtBonds, taCorpBonds, taTips, taGovtTips, taCorpTips]

    telCorpBond60ILS = "720@TASE = TA-CorpBond60ILS"
    telCorpBondUsd = "739@TASE = TA-CorpBondUsd"
    telCorpBond20 = "707@TASE = TA-CorpBond20"
    telCorpBond40 = "708@TASE = TA-CorpBond40"
    telCorpBond60 = "709@TASE = TA-CorpBond60"
    ta_corpBonds = [telCorpBond20, telCorpBond40, telCorpBond60, telCorpBond60ILS, telCorpBondUsd]

    taMakam = "800@TASE = TA-Makam"
    ta_makam = [taMakam]

    ta_all = ta_stocks + ta_bonds + ta_corpBonds + ta_makam    
    # ==== TASE END ====
    
    
    
    glb = globals().copy()
    for k in glb.keys():
        if k.startswith("_"):
            continue
        val = glb[k]
        if not isinstance(val, str):
            continue
        if "\n" in val:
            continue
        if k.isupper():
            continue
        if "=" in val:
            continue
        globals()[k] = f"{val} = {k}"
    
    fixed_globals_once = True

all_assets = [
# ==== STOCKS ====
# Global
d_ac,
# US
ac,
lc,
mc,
sc,
mcc,
lcv,
mcv,
scv,
lcg,
mcg,
scg,
# ex-US
i_ac,
i_sc,
i_dev,
i_acv,
i_scv,
em_ac,
em_sc,
fr_ac,

# ==== BONDS ====
# US GOVT
sgb,
tips,
lgb,
elgb,
gb,
fgb,
# US CORP 
cb,
scb,
lcb,
fcb,
# US CORP+GOVT
gcb,
sgcb,
# International
i_tips,
i_gcbUsd,
i_gbLcl,
i_cb,
i_cjb,
g_gcbLcl,
g_gcbUsd,
g_sgcb,
g_usgcb,
em_gbUsd,
em_gbLcl,
em_cjb,
cjb,
scjb,

# ==== CASH ====
rfr,

# ==== OTHER ====
#fedRate,
reit,
i_reit,
gold,
silver,
palladium,
platinum,
#metals,
comm,
oilWtiQ,
oilBrentQ,
]

assets_core = [
    # equities
    lc,
    i_ac,
    i_dev,
    em_ac,
    # reit
    reit,
    i_reit,
    # bonds
    gb,
    lgb,
    cb,
    i_cb,
    em_gbUsd,
    tips,
    # commodities
    gold,
    comm,
    # cash
    cash
]

# https://www.federalreserve.gov/pubs/bulletin/2005/winter05_index.pdf
usdMajorCurrencies = ["USDEUR", "USDCAD", "USDJPY", "USDGBP", "USDCHF", "USDAUD", "USDSEK"]
usdOtherCurrencies = ["USDMXN", "USDCNY", "USDTWD", "USDKRW", "USDSGD", "USDHKD", "USDMYR", "USDBRL", "USDTHB", "USDINR"] # "USDPHP"
usdBroadCurrencies = usdMajorCurrencies + usdOtherCurrencies

interestingCurrencies = ["USDEUR", "USDCAD", "USDJPY", "USDAUD", "USDJPY", "USDCNY"]


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

def get_named(s, func):
    return (func(s), f"{get_pretty_name(s)} {func.__name__}")

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

def divs(symbolName, period=None, fill=False):
    name = get_name(symbolName)
    if name.startswith("~"):
        divs = sym[-1:0] # we just want an empty series with DatetimeIndex
        #divs = pd.Series(index=pd.DatetimeIndex(freq="D"))
        divs.name = name
    else:
        divs = get(symbolName, mode="divs")
        divs = divs[divs>0]
    if period:
        divs = wrap(divs.rolling(period).sum())
    if fill:
        price = get(symbolName)
        divs = divs.reindex(price.index.union(divs.index), fill_value=0)
    divs.name = divs.name + " divs"
    return divs

def get_divs_interval(divs):
    divs = divs.resample("M").sum()
    divs = divs[divs>0]
    monthds_diff = (divs.index.to_series().diff().dt.days/30).dropna().apply(lambda x: int(round(x)))
    monthds_diff = monthds_diff[-5:].median()
    return monthds_diff

def get_yields(sym):
    n = get_name(sym)
    true_yield = name(get_yield(sym, type='true'), n + " true")
    normal_yield = name(get_yield(sym, type='normal'), n + " normal")
    rolling_yield = name(get_yield(sym, type='rolling'), n + " rolling")
    return[true_yield, normal_yield, rolling_yield]

def show_yields(*lst):
    yields = lmap(get_yields, lst)
    rets = [get_named(x, cagr) for x in get(lst, trim=True)]
    show(*yields, rets, 0, ta=False, log=False)

def get_yield_true(sym):
    return get_yield(sym, type='true')

def get_yield_normal(sym):
    return get_yield(sym, type='normal')

def get_yield_rolling(sym):
    return get_yield(sym, type='rolling')

def get_yield(sym, type=None):
    type = type or 'true'
    if type == 'true':
        return _get_yield(sym, window_months=1)
    if type == 'normal':
        yld_true = _get_yield(sym, window_months=1)
        return mm(yld_true, 5)
    if type == 'rolling':
        return _get_yield(sym, window_months=12)
    raise Exception(f"Invalid yield type {type}")

def _get_yield(symbolName, dists_per_year=None, altPriceName=None, window_months=12):
    # if isinstance(symbolName, tuple) and dists_per_year is None:
    #     symbolName, dists_per_year = symbolName
    if is_series(symbolName):
        symbolName = symbolName.name
    if isinstance(symbolName, Symbol):
        sym_mode = symbolName.mode
    else:
        sym_mode = None
    if symbolName.startswith("~"):
        return pd.Series()
    price = get(altPriceName or symbolName, mode="PR")
    divs = get(symbolName, mode="divs")
    divs = divs[divs>0]
    if len(divs) == 0:
        return divs
    
    if sym_mode == "NTR":
        divs *= 0.75
    elif sym_mode == "PR":
        divs *= 0

    # sometimes 2 or more divs can happen in the same month (capital gains)
    # we must resample and sum to 1-month resolution to correctly calculate the months_between_dists
    # and later to correctly do a rolling sum
    divs = divs.resample("M").sum()
    divs = divs[divs>0]

    if dists_per_year is None:
        months_between_dists = get_divs_interval(divs)
        dists_per_year = int(12 // months_between_dists)
    else:
        months_between_dists = int(12 // dists_per_year)

    if window_months < months_between_dists:
        #warn(f"auto correcting window_months to {months_between_dists}")
        window_months = months_between_dists

    n = int(window_months / months_between_dists)
    mult = 12 / window_months
    if n > 0:
        divs = divs.rolling(n).sum() * mult
    yld = divs / price * 100
    return name(yld, divs.name).dropna()

def get_curr_yield_normal(s):
    return get_curr_yield(s, type='normal')

def get_curr_yield_rolling(s):
    return get_curr_yield(s, type='rolling')

def get_curr_yield(s, type=None):
    type = type or 'normal'
    yld = get_yield(s, type=type).dropna()
    if yld.shape[0] == 0:
        return 0
    return yld[-1]

def get_curr_net_yield(s, type=None):
    return get_curr_yield(s, type=type)*0.75
    
def get_TR_from_PR_and_divs(pr, divs):
    m = d / pr + 1
    mCP = m.cumprod().fillna(method="ffill")
    tr = pr * mCP
    return wrap(tr, pr.name + " TR")

def show_income(*all, **args):
    income = lmap(partial(get_income, **args), all)
    show(income, 0, ta=False, log=False, title="net income")

def show_cum_income(*all):
    income = lmap(get_cum_income, all)
    show(income, ta=False, log=False, legend=False, title="cumulative net income")

def show_cum_income_relative(*all, base_i=0):
    income = lmap(get_cum_income, all)
    income = [sdiv(x, income[base_i]) for x in income]
    show(income, ta=False, log=False, legend=False, title="relative cumulative net income")

def analyze_assets(*all, start=None, despike=True, few=None):
    all = get(all, start=start, despike=despike) # note we don't trim
    all_trim = get(all, trim=True)
    if few is None:
        few = len(all) <= 5

    if few:
        show(*all, trim=False, align='rel')
        show_modes(*all)

    # risk-return
    bases = [ mix(lc, gb, do_get=False), mix(i_ac, gb, do_get=False)]
    lst = get(all + bases, mode="TR")
    show_risk_return(*lst, title="TR Risk-Return")
    lst = get(all + bases, mode="NTR")
    show_risk_return(*lst, title="NTR Risk-Return")
    lst = get(all + bases, mode="ITR")
    show_risk_return(*lst, title="ITR Risk-Return")
    lst = get(all + bases, mode="PR")
    show_risk_return(*lst, title="PR Risk-Return")
    show_risk_return_modes(*all)
    
    # draw-down
    if few:
        show_dd(*all)

    # withdraw flows
    show_flows(*all)

    # Yields
    show_risk_ntr_pr_diff_pr(*all)
    show_risk_ntr_pr_diff_pr_full_alt(*all)
    show_risk_ntr_pr_diff_pr_full_alt(*lst, trim=False)


    show_risk_yield_types(*all)
    show_risk_yield(*all)
    show_risk_return_modes(*all, ret_func=get_curr_yield_normal, modes=['TR'], title='Risk - NORMAL Yield TR')
    show_risk_return_modes(*all, ret_func=get_curr_yield_normal, modes=['NTR'], title='Risk - NORMAL Yield NTR')

    yields = lmap(get_yield_true, all)
    inf = get_inflation(365*7)[getCommonDate(yields, 'start', agg=min):]
    show(yields, inf, 0, ta=False, log=False, title="nominal TRUE gross yield")    
    yields = lmap(get_yield_normal, all)
    show(yields, inf, 0, ta=False, log=False, title="nominal NORMAL gross yield")    
    yields = lmap(get_yield_rolling, all)
    show(yields, inf, 0, ta=False, log=False, title="nominal ROLLING gross yield")    
    
    yields = lmap(lambda x: get_real_yield(x, 'normal'), all)
    show(yields, 0, ta=False, log=False, title="REAL gross NORMAL yield")


    # Income
    show_cum_income(*all)
    show_cum_income(*all_trim)
    show_cum_income_relative(*all_trim)
    show_income(*all)

    show(lmap(adj_inf, lmap(price, all)), 1, title="real price")

    # show(lmap(roi, all), ta=False, log=False, title="net ROI")

    # lrret
    lrret_mutual(*all)

    # PCA / MDS
    show_mds(*all)

    # modes

def show_dd(*all, mode="PR"):
    all = get(all, mode=mode)
    show(lmap(dd, all), -10, -20, -30, -40, -50, ta=False, title=f"{mode} draw-down")

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

def get_date(x):
    if is_series(x):
        return x.index[0]
    elif isinstance(x, int):
        return datetime.datetime(x, 1, 1)
    elif isinstance(x, pd.Timestamp) or isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
        return x
    raise Exception(f"not supported date type {type(x)}")


def filterByStart(lst, start=None):
    if start is None:
        return lst
    start = get_date(start)
    res = [s for s in lst if s.index[0] <= start]
    dropped = [s for s in lst if s.index[0] > start]
    
    #dropped = set(lst) - set(res)
    if len(dropped) > 0:
        dropped = [s.name for s in dropped]
        print(f"dropped: {', '.join(dropped)}")
    return res        

def tr(s):
    return get(s, mode="TR")

def ntr(s):
    return get(s, mode="NTR")

def pr(sym):
    return get(sym, mode="PR")
price = pr

def get_income(sym, value=100000, nis=False, per_month=True, smooth=12, net=True):
    start = None
    if is_series(sym):
        start = sym.index[0]
    prc = price(sym)
    if not start is None:
        prc = prc[start:]
    units = value / prc[0]
    div = divs(sym)
    if not start is None:
        div = div[start:]
    income = div * units
    if net:
        income *= 0.75
    if per_month:
        income = income.resample("M").sum()
        #income = income[income > 0]

        #interval = get_divs_interval(div)
        #income = ma(income, interval)
    else:
        income = income.resample("Y").sum()/12
    #income = income.replace(0, np.nan).interpolate()
#     if per_month:
#         income /= 12
    if nis:
        income = convertSeries(income, "USD", "ILS")
    if smooth:
        income = ma(income, smooth)
    return name(income, prc.name)

def get_cum_income(sym):
    income = get_income(sym, smooth=0)
    return income.cumsum()

def adj_inf(s):
    cpi = get(cpiUS)
    s = get(s)
    return name(s / cpi, s.name)

from functools import lru_cache

@lru_cache(maxsize=12)
def get_inflation(smooth=None):
    cpi = get(cpiUS, interpolate=False)
    inf = (cpi / cpi.shift(12) - 1) * 100
    inf = inf.asfreq("D").interpolate()
    if smooth:
        inf = ma(inf, smooth)
    return name(inf, "inflation")

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
    res = [get(s, mode="TR", **get_args), get(s, mode="NTR", **get_args), get(s, mode="ITR", **get_args), get(s, mode="PR", **get_args)]
    # we can't rename exisitng series, it messes up future gets
    # res[0].name += "-TR"
    # res[1].name += "-NTR"
    # res[2].name += "-ITR"
    # res[3].name += "-PR"
    return res

# ## Generic Utils

# In[ ]:


# safely convert a float/string/mixed series to floats
# to remove commas we need the data type to be "str"
# but if we assume it's "str" wihtout converting first, and some are numbers
# those numbers will become NaN's.
def series_as_float(ser):
    return pd.to_numeric(ser.astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")

def lmap(f, l):
    return list(map(f, l))

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
import functools as ft
from functools import partial
def compose(*functions):
    return ft.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


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

def show_flows(*all, n=None, rng=None):
    if rng is None:
        rng = [0, 5]
    all = get(all, trim=True)
    all = lmap(lambda x: get_flows(x, n=n, rng=rng), all)
    show_risk_return(*all, title="net flows")


def show_comp(target, base, extra=None, mode="NTR"):
    target, base, extra = get([target, base, extra], mode=mode)
    all = [target, base]
    if not extra is None:
        if isinstance(extra, list):
            all += extra
        else:
            all.append(extra)
    all_trim = get(all, trim=True)

    show(*all, 0.5, 1, trim=False, title=mode + " equity") # use 0.5 instead of 0 to keep the log scale
    show_dd(*all)
    show_dd(*all_trim)
    show_yields(*all)
    show_income(*all)
    show_income(*all, smooth=0)
    show_cum_income(*all_trim)
    show_cum_income_relative(*all_trim, base_i=1)
    show_modes_comp(target, base)
    show_port_flow_comp(target, base)
    show_flows(*all)


############### risk / return metrics ###############

def cagr(s):
    days = (s.index[-1] - s.index[0]).days
    if days <= 0:
        return np.nan
    years = days/365
    val = s[-1] / s[0]
    if val < 0:
        raise Exception("Can't calc cagr for a negative value") # this indicates that the series is not an equity curve
    return (math.pow(val, 1/years)-1)*100

def ulcer(x):
    cmax = np.maximum.accumulate(x)
    r = (x/cmax-1)*100
    return math.sqrt(np.sum(r*r)/x.shape[0])

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

def start_year_full(s):
    s = get(s, untrim=True)
    return str(s.index[0].year)
def start_year_full_with_name(s):
    return f"{s.name} {start_year_full(s)}"

def show_risk_ntr_pr_diff_pr_full_alt(*lst, trim=True):
    alt_text = start_year_full if trim else start_year_full_with_name
    show_risk_ntr_pr_diff_pr(*lst, alt_risk_func=cagr_full, alt_risk_text=alt_text, trim=trim)

def show_risk_ntr_pr_diff_pr(*lst, risk_func=cagr, alt_risk_func=pr_lr_cagr, alt_risk_text=None, title=None, trim=True):
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
    plot_scatter(*res, title=title, xlabel=f"{risk_func.__name__} ➜ {alt_risk_func.__name__}", ylabel="cagr(NTR) - cagr(PR) ➜ curr 12m net yield", show_zero_point=True, same_ratio=True)

############################# python utils ###########

def list_rm(l, *items):
    l = l.copy()
    for x in items:
        l.remove(x)
    return l    

############################

def get_real(s):
    return rename(get(s) / get(cpiUS), f"{get_pretty_name(s)} real")
