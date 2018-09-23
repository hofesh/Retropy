from framework.utils import *
from framework.RpySeries import *
from framework.symbol import *
from framework.base import *
from framework.meta_data import *
from framework.stats_basic import *

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

def get_yield_types(sym, **args):
    n = get_name(sym)
    true_yield = name(get_yield(sym, type='true', **args), n + " true")
    normal_yield = name(get_yield(sym, type='normal', **args), n + " normal")
    rolling_yield = name(get_yield(sym, type='rolling', **args), n + " rolling")
    return[true_yield, normal_yield, rolling_yield]

def get_named(s, func):
    return (func(s), f"{get_pretty_name(s)} {get_func_name(func)}")

def get_yield_true(sym):
    return get_yield(sym, type='true')

def get_yield_true_no_fees(sym):
    return get_yield(sym, type='true', reduce_fees=False)

def get_yield_normal(sym):
    return get_yield(sym, type='normal')

def get_yield_normal_no_fees(sym):
    return get_yield(sym, type='normal', reduce_fees=False)

def get_yield_rolling(sym):
    return get_yield(sym, type='rolling')

def get_yield_rolling_no_fees(sym):
    return get_yield(sym, type='rolling', reduce_fees=False)

def get_yield(sym, type=None, drop_special_divs=False, keep_trim=False, reduce_fees=True, altPriceName=None):
    type = type or 'true'
    if type == 'true':
        return get_yield_helper(sym, window_months=1, drop_special_divs=drop_special_divs, keep_trim=keep_trim, reduce_fees=reduce_fees, altPriceName=altPriceName)
    if type == 'normal':
        yld_true = get_yield_helper(sym, window_months=1, drop_special_divs=drop_special_divs, keep_trim=keep_trim, reduce_fees=reduce_fees, altPriceName=altPriceName)
        return mm(yld_true, 5)
    if type == 'rolling':
        return get_yield_helper(sym, window_months=12, drop_special_divs=drop_special_divs, keep_trim=keep_trim, reduce_fees=reduce_fees, altPriceName=altPriceName)
    raise Exception(f"Invalid yield type {type}")

# for each div, how any months passed from the previous div (the first one gets the same value as the second)
def get_divs_intervals(divs):
    divs = divs.resample("M").sum()
    divs = divs[divs>0]
    monthds_diff = (divs.index.to_series().diff().dt.days/30).fillna(method='backfill').apply(lambda x: int(round(x)))
    return monthds_diff

def rolling_timeframe(d, f, offset):
    vals = []
    index = d[d.index[0] + offset:].index
    for dt in index:
    #    print(f"start: {dt - pd.DateOffset(years=1)}, end: {dt}")
        vals.append(f(d[(d.index <= dt) & (d.index > dt - offset)]))
    res = pd.Series(vals, index)
    return res

def get_yield_helper(symbolName, dists_per_year=None, altPriceName=None, window_months=12, drop_special_divs=False, keep_trim=False, reduce_fees=True):
    # if isinstance(symbolName, tuple) and dists_per_year is None:
    #     symbolName, dists_per_year = symbolName
    start = None
    if is_series(symbolName):
        start = symbolName.index[0]
        symbolName = symbolName.name
    if is_symbol(symbolName):
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
    if keep_trim and start:
        price = price[start:]
        divs = divs[start:]

    if drop_special_divs:
        vc = divs.index.day.value_counts()
        vc = vc.reindex(range(31))
        df = pd.DataFrame([vc, vc.shift(1), vc.shift(-1)]).T.fillna(0)
        special_divs_days = set(df[(df.iloc[:, 0] > 0) & (df.iloc[:, 0] <= 2) & (df.iloc[:, 1] == 0) & (df.iloc[:, 2] == 0)].index)
        dropped_divs = divs[divs.index.day.isin(special_divs_days)]
        if dropped_divs.shape[0] > 0:
            dates = ', '.join([str(x)[:10] for x in dropped_divs.index.values])
            print_norep(f"Dropped {dropped_divs.shape[0]} special dividends in {get_pretty_name(symbolName)} on {dates}")
        divs = divs[~divs.index.day.isin(special_divs_days)]
    
    if sym_mode == "NTR":
        divs *= 0.75
    elif sym_mode == "PR":
        divs *= 0


    # sometimes 2 or more divs can happen in the same month (capital gains)
    # we must resample and sum to 1-month resolution to correctly calculate the months_between_dists
    # and later to correctly do a rolling sum
    # NOTE: this causes a falw in the yield calculation
    # the actual dividends are usually mid-month, while the resample changes the dates to end of month
    # and thus later we divide by price at end of month, and not at the dividend date
    divs = divs.resample("M").sum()
    divs = divs[divs>0]

    divs, price = sync(divs, price)

    if len(divs) == 1 and window_months == 1: # we can't get_divs_intervals
        return divs[0:0]
    if len(divs) == 0:
        return divs

    if window_months > 1:
        mult = 12 / window_months # annualizer
        offset = pd.DateOffset(months=window_months-1, days=15)
        divs = rolling_timeframe(divs, lambda x: np.sum(x), offset)
        yld = divs * mult / price * 100
    else:
        divs_intervals = get_divs_intervals(divs)
        mult = 12 / divs_intervals # annualizer
        yld = divs * mult / price * 100

    res = name(yld, symbolName).dropna()
    if reduce_fees and sym_mode != "PR":
        fees = get_meta_fee(symbolName)
        res -= fees
    return res

def get_curr_yield_max(s):
    return max([get_curr_yield_true(s), get_curr_yield_normal(s), get_curr_yield_rolling(s)])

def get_curr_yield_min(s):
    return min([get_curr_yield_true(s), get_curr_yield_normal(s), get_curr_yield_rolling(s)])

def get_curr_yield_min2(s):
    return min([get_curr_yield_normal(s), get_curr_yield_rolling(s)])

def get_curr_yield_true(s):
    return get_curr_yield(s, type='true')

def get_curr_yield_true_no_fees(s):
    return get_curr_yield(s, type='true', reduce_fees=False)

def get_curr_yield_normal(s):
    return get_curr_yield(s, type='normal')

def get_curr_yield_normal_no_fees(s):
    return get_curr_yield(s, type='normal', reduce_fees=False)

def get_start_yield_normal(s):
    return get_start_yield(s, type='normal')

def get_start_yield_normal_no_fees(s):
    return get_start_yield(s, type='normal', reduce_fees=False)

def get_curr_yield_rolling(s, reduce_fees=True):
    return get_curr_yield(s, type='rolling')

def get_curr_yield_rolling_no_fees(s):
    return get_curr_yield(s, type='rolling', reduce_fees=False)

def get_curr_yield(s, type=None, reduce_fees=True):
    type = type or 'rolling'
    yld = get_yield(s, type=type, reduce_fees=reduce_fees).dropna()
    if yld.shape[0] == 0:
        return 0
    return yld[-1]

def get_start_yield(s, type=None, reduce_fees=True):
    type = type or 'rolling'
    yld = get_yield(s, type=type, keep_trim=True, reduce_fees=reduce_fees).dropna()
    if yld.shape[0] == 0:
        return 0
    return yld[0]

def get_curr_net_yield(s, type=None):
    return get_curr_yield(s, type=type)*0.75
    
def get_TR_from_PR_and_divs(pr, divs):
    m = d / pr + 1
    mCP = m.cumprod().fillna(method="ffill")
    tr = pr * mCP
    return wrap(tr, pr.name + " TR")
