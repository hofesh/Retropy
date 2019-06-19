import datetime
from pathlib import Path
import requests
import json
import re
from bs4 import BeautifulSoup
from io import StringIO

import pandas as pd

pd.core.common.is_list_like = pd.api.types.is_list_like # patch until pandas_datareader is fixed
import pandas_datareader
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies

import quandl
quandl.ApiConfig.api_key = "9nrUn7Sm1SdoeLdQGQB-"

from framework.utils import *
from framework.cache import *
import framework.conf as gconf
import framework.mem as gmem

# other data sources:
# Investopedia:
# https://www.investopedia.com/markets/api/partial/historical/?Symbol=VFINX&Type=Historical+Prices&Timeframe=Daily&StartDate=Apr+21%2C+1970&EndDate=May+21%2C+2018

class DataSource:

    def __init__(self, source):
        self.source = source

    def fetch(self, symbol, conf):
        pass

    def process(self, symbol, df, conf):
        pass

    def get(self, symbol, conf):
        df = None

        mem_key = self.source + "#" + symbol.fullname

        # get from mem cache
        if conf.cache and gconf.conf_cache_memory:
            if mem_key in gmem.symbols_mem_cache:
                df = gmem.symbols_mem_cache[mem_key]

        # get from disk cache
        if df is None and conf.cache and gconf.conf_cache_disk:
            df = cache_get(symbol, self.source)

        # attempt to fetch the symbol
        if df is None:
            failpath = cache_file(symbol, self.source) + "._FAIL_"
            if os.path.isfile(failpath):
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(failpath))
                diff = datetime.datetime.now() - mtime
                if (conf.cache_fails or gconf.conf_cache_fails) and diff.total_seconds() <= 24 * 3600 * 365:
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
                print("ERROR")
                # save a note that we failed
                Path(failpath).touch()
                raise

            # write to disk cache
            cache_set(symbol, self.source, df)

        # write to mem cache
        gmem.symbols_mem_cache[mem_key] = df

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
        return quandl.get(symbol.ticker)

    def process(self, symbol, df, conf):
        if not symbol.field is None:
            int_field = as_int(symbol.field)
            if not int_field is None:
                return df[df.columns[int_field]].dropna()
            if not symbol.field in df.columns:
                warn(f"field {symbol.field} not found in {symbol} raw DataFrame")
                return None
            return df[symbol.field].dropna()
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

class FundFlowDataSource(DataSource):
    def fetch(self, symbol, conf):
        url = "https://www.etf.com/etf-chart/ajax/fundflows/__sym__/1993-01-01/2020-10-11/__sym__%20Daily%20Fund%20Flows/Net%20Flows/%24USD%2Cmm/yes"
        url = url.replace("__sym__", symbol.ticker)

        headers = {
            'Origin': 'https://www.etf.com',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Referer': 'https://www.etf.com/etfanalytics/etf-fund-flows-tool',
            'Authority': 'www.etf.com',
            'X-Requested-With': 'XMLHttpRequest',
            'Connection': 'keep-alive'
        }
        r = requests.get(url, headers=headers)
        text = r.text
        js = json.loads(text)
        data = js[1]['output']
        data = data.split('data:')[1].strip()
        data = data.split('\n')[0]
        data = json.loads(data)
        df = pd.DataFrame(data)
        df.columns = ["date", "flow"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    def process(self, symbol, df, conf):
        return df["flow"]


class EdenDataSource(DataSource):
    def fetch(self, symbol, conf):
        EDEN = """
Date	Value
01-Jan-2016	$1,000.0000
01-Feb-2016	$910.5607
01-Mar-2016	$918.3250
01-Apr-2016	$979.3130
01-May-2016	$964.3864
01-Jun-2016	$993.0645
01-Jul-2016	$947.6794
01-Aug-2016	$982.4717
01-Sep-2016	$1,051.3543
01-Oct-2016	$1,047.5805
01-Nov-2016	$995.6747
01-Dec-2016	$1,077.7583
31-Dec-2016	$1,084.7515
31-Jan-2017	$1,120.1306
28-Feb-2017	$1,139.9899
01-Apr-2017	$1,154.4490
01-May-2017	$1,194.7856
01-Jun-2017	$1,204.7632
30-Jun-2017	$1,236.1665
30-Jul-2017	$1,280.3044
30-Aug-2017	$1,297.5405
30-Sep-2017	$1,288.2847
30-Oct-2017	$1,293.6846
30-Nov-2017	$1,268.1199
31-Dec-2017	$1,254.4560
31-Jan-2018	$1,362.5164
28-Feb-2018	$1,377.6356
31-Mar-2018	$1,315.6067
30-Apr-2018	$1,407.3427
31-May-2018	$1,459.2245
30-Jun-2018	$1,456.8649
31-Jul-2018	$1,452.3415
31-Aug-2018	$1,418.2482
30-Sep-2018	$1,415.6050
"""
        df = pd.read_csv(StringIO(EDEN), sep="\t", thousands=",", index_col="Date", parse_dates=True, dayfirst=True)
        return df

    def process(self, symbol, df, conf):
        return series_as_float(df["Value"])#.str.replace("$", "").str.replace(",", "").astype("float")

data_sources = {
    "EDEN": EdenDataSource("EDEN"),
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
    "G": GoogleDataSource("G"),
    "FF":FundFlowDataSource("FF")
               }
