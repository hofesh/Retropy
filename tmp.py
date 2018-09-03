from Retropy_framework import *

s = 'SPY'
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
all = assets_core
#all = select
#all = mix(lc, lgb)
all = get(all, trim=True, source="Y", start=2006)
all = lmap(lambda x: get_flows(x, rng=[0, 5]), all)
show_risk_return(*all)