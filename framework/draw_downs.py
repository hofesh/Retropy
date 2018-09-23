from collections import namedtuple

from framework.RpySeries import *
from framework.base import *

dd_item = namedtuple("dd_item", ["start", "end", "depth"])

def dd(x):
    x = get(x).dropna()
    res = (x / np.maximum.accumulate(x) - 1) * 100
    return res

def dd_rolling(s, n=365):
    m = s.rolling(n, min_periods=0).max()
    d = (s/m - 1)*100
    d.name = s.name
    return d

# def dd_match(s, base):
#     s_orig = s
#     s, base = get([s, base], mode="PR", untrim=False) # NOTE: untrim makes a HUGE difference in results
#     s, base = dd_rolling(s), dd_rolling(base)
#     s, base = doTrim([s, base], trim=s_orig, silent=True)
#     total_base_dd = -base.sum()
#     match = np.minimum(-s, -base).sum()
#     return match / total_base_dd

# #dd_match_SPY = partial(dd_match, base=SPY)
# def dd_match_SPY(x):
#     return dd_match(x, 'SPY')

def get_dds(s, min_days=0, min_depth=0, dd_func=dd):
    min_depth = -abs(min_depth)
    s_dd = dd_func(get(s))
    ranges = []
    in_dd = s_dd[0] < 0
    dd_start = s_dd.index[0] if in_dd else None
    prev_d = None
    dd_depth = 0
    for d, v in s_dd.iteritems():
        if in_dd:
            dd_depth = min(dd_depth, v)
        if in_dd and v == 0:
            in_dd = False
            dd_end = d
            ranges.append(dd_item(dd_start, dd_end, dd_depth))
            dd_depth = 0
        elif not in_dd and v < 0:
            in_dd = True
            dd_start = prev_d
            dd_depth = v
        prev_d = d
    if in_dd:
        ranges.append(dd_item(dd_start, s_dd.index[-1], dd_depth))

    if min_days:
        ranges = [r for r in ranges if r[1] - r[0] > pd.Timedelta(days=min_days)]
    if min_depth:
        ranges = [r for r in ranges if r[2] <= min_depth]
    return ranges

def get_price_actions(s, ranges):
    return [(doAlign([s[i:j]])[0]-1)*100 for i, j, _ in ranges]

def get_price_actions_with_rolling_base(target, ranges, base, n=365):
    if n == 0:
        base_max = np.maximum.accumulate(base) 
    else:
        base_max = base.rolling(n, min_periods=0).max()
    res = []
    for i,j,_ in ranges:
        tt = target[i:j]
        if len(tt) == 0:
            continue
        tt = align_with(tt, base[i:j])
        tt = (tt/base_max[i:j] - 1)*100
        tt.name = target.name
        res.append(tt)
    return res

def mutual_dd(target, base, dd_func, weighted=True, min_depth=3):
    # target, base = get([target, base], mode="PR", trim=True)
    # base_dd = dd_func(base)
    # ranges = get_dds(base, min_days=min_days, min_depth=min_depth, dd_func=dd_func)

    tup = get([target, base], trim=True, silent=True) # was trim=False
    if len(tup) < 2:
        return 0
    target, base = tup
    ranges = get_dds(base, min_days=0, min_depth=min_depth, dd_func=dd_func)

    if dd_func.__name__ == "dd":
        target_price_actions = get_price_actions(target, ranges)
    elif dd_func.__name__ == "dd_rolling":
        target_price_actions = get_price_actions_with_rolling_base(target, ranges, base)
    else:
        raise Exception(f"unsupported dd_func: {get_func_name(dd_func)}")
    
    dd_base = dd_func(base)
    #dd_base = trimBy(dd_base, target) # was uncommented
    base_dd_actions = [dd_base[i:j] for i,j,_ in ranges]

    pa_target = [-s.sum() for s in target_price_actions]
    pa_base = [-s.sum() for s in base_dd_actions]
    
    if weighted:
        #return np.median([t/b for t,b in zip(pa_target, pa_base)])
        #return np.mean([t/b for t,b in zip(pa_target, pa_base)])
        weights = [np.sqrt(x) for x in pa_base]
        return np.sum([w*t/b for t,b,w in zip(pa_target, pa_base, weights)]) / np.sum(weights)
    else:
        return np.sum(pa_target) / np.sum(pa_base)

def mutual_dd_pr(target, base, dd_func):
    return mutual_dd(pr(target), pr(base), dd_func=dd_func)

def mutual_dd_rolling_pr(target, base):
    return mutual_dd(pr(target), pr(base), dd_func=dd_rolling)

def mutual_dd_rolling_pr_SPY(target):
    return mutual_dd(pr(target), pr('SPY'), dd_func=dd_rolling)

def mutual_dd_rolling_pr_SPY_weighted(target):
    return mutual_dd(pr(target), pr('SPY'), dd_func=dd_rolling, weighted=True)

def mutual_dd_rolling_pr_SPY_unweighted(target):
    return mutual_dd(pr(target), pr('SPY'), dd_func=dd_rolling, weighted=False)

def mutual_dd_rolling_pr_TLT(target):
    return mutual_dd(pr(target), pr('TLT'), dd_func=dd_rolling)

def mutual_dd_rolling_SPY(target):
    return mutual_dd(target, get('SPY', mode=target.name.mode), dd_func=dd_rolling)

def mutual_dd_pr_SPY(target):
    return mutual_dd(pr(target), pr('SPY'), dd_func=dd)

def mutual_dd_SPY(target):
    return mutual_dd(target, get('SPY', mode=target.name.mode), dd_func=dd)

