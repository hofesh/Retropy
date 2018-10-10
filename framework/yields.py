from framework.utils import *
from framework.RpySeries import *
from framework.symbol import *
from framework.base import *
from framework.meta_data import *
from framework.stats_basic import *
import Retropy_framework as frm

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

def get_yield(sym, type=None, drop_special_divs=False, keep_trim=False, reduce_fees=True, altPriceName=None, divs_only=False, do_resample=True):
    type = type or 'true'
    if type == 'true':
        return get_yield_helper(sym, window_months=1, drop_special_divs=drop_special_divs, keep_trim=keep_trim, reduce_fees=reduce_fees, altPriceName=altPriceName, divs_only=divs_only, do_resample=do_resample)
    if type == 'normal':
        yld_true = get_yield_helper(sym, window_months=1, drop_special_divs=drop_special_divs, keep_trim=keep_trim, reduce_fees=reduce_fees, altPriceName=altPriceName, divs_only=divs_only, do_resample=do_resample)
        return mm(yld_true, 5)
    if type == 'rolling':
        return get_yield_helper(sym, window_months=12, drop_special_divs=drop_special_divs, keep_trim=keep_trim, reduce_fees=reduce_fees, altPriceName=altPriceName, divs_only=divs_only, do_resample=do_resample)
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

def get_yield_helper(symbolName, dists_per_year=None, altPriceName=None, window_months=12, drop_special_divs=False, keep_trim=False, reduce_fees=True, divs_only=False, do_resample=True):
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
    if do_resample:
        divs = divs.resample("M").sum()
        price = price.resample("M").mean()
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
        if divs_only:
            yld = divs * mult
        else:
            yld = divs * mult / price * 100
    else:
        divs_intervals = get_divs_intervals(divs)
        mult = 12 / divs_intervals # annualizer
        _, mult = expand(divs, mult)
        mult = mult.fillna(method='bfill')
        if divs_only:
            yld = divs * mult
        else:
            yld = divs * mult / price * 100

    res = name(yld, symbolName).dropna()
    if reduce_fees and sym_mode != "PR" and not divs_only:
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

def get_curr_yield(s, type=None, reduce_fees=True, live=True):
    type = type or 'rolling'
    yld = get_yield(s, type=type, reduce_fees=reduce_fees, divs_only=live).dropna()
    if yld.shape[0] == 0:
        return 0
    if live:
        yld = yld[-1] / pr(s)[-1] * 100
        if reduce_fees and get(s).name.mode != "PR":
            fees = get_meta_fee(s)
            yld -= fees
        return yld
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
    m = divs / pr + 1
    mCP = m.cumprod().fillna(method="ffill")
    tr = pr * mCP
    return wrap(tr, pr.name + " TR")

def get_income_median(s):
    income = get_income(s, smooth=1)
    income = income[income > 0]
    income = mm(income, 12)
    return income

def get_income_ulcer(s):
    return ulcer_pr(get_income_median(s))

def show_income_ulcer(*all):
    frm.show(lmap(get_income_median, all), ta=False, log=False, title="Income 12-month median")

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

def get_curr_yield_zscore(s):
    yld = get_yield_true(s)
    if yld is None:
        return None
    zs = zscore_modified(yld)
    if len(zs) == 0:
        return None
    return zs[-1]

def get_yield_live(s, type=None, reduce_fees=True, live=True, type_name=False):
    type = type or 'rolling'
    yld = get_yield(s, type=type, reduce_fees=reduce_fees, divs_only=live, do_resample=False).dropna()
    if yld.shape[0] == 0:
        return 0
    prc = pr(s)
    yld = yld.reindex(prc.index)
    yld = yld.fillna(method='ffill')
    yld = yld / prc * 100
    if reduce_fees and get(s).name.mode != "PR":
        fees = get_meta_fee(s)
        yld -= fees
    if type_name:
        yld = name(yld, f"{get_name(s)} {type}")
    return yld.dropna()

def get_yield_live_all(s, reduce_fees=True):
    res = [get_yield_live(s, type='rolling', type_name=True, reduce_fees=reduce_fees), 
           get_yield_live(s, type='normal', type_name=True, reduce_fees=reduce_fees), 
           get_yield_live(s, type='true', type_name=True, reduce_fees=reduce_fees)]
    return res

def show_yield(*all, reduce_fees=False, detailed=True):
    ylds = lmap(partial(get_yield_live_all, reduce_fees=reduce_fees), all)
    ylds = flattenLists(ylds)
    title = f"{get_mode(all[0])} Yields {'with fees' if reduce_fees else 'no fees'}"
    if detailed and len(all) == 1:
        s = all[0]
        dist = divs(s)
        dist = align_with(dist, ylds[0])
        prc = price(s)
        prc = align_with(prc, ylds[0])
        extra = [dist, prc]
        title=get_pretty_name_no_mode(s) + " " + title
    else:
        extra = []
    frm.show(0, *extra, ylds, lmap(last, ylds[2::3]), ta=False, log=False, title=title, sort=False)
