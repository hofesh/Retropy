from framework.base import *
from framework.yields import *
import Retropy_framework as frm

def shiller_snp500(taxes=False, inf_adj=False, tax_rate=0.25):
    dv = get('MULTPL/SP500_DIV_MONTH@Q', interpolate=False)
    pr = get('MULTPL/SP500_INFLADJ_MONTH@Q', interpolate=False)

    tax_rate = tax_rate if taxes else 0
    dv = dv.resample("MS").sum() / 12 * (1-tax_rate) # sync dates to start of month, and normalize divs to monthly not yearly, and reduce div taxes
    dv = dv.reindex(pr.index).fillna(method="ffill") # fill few last missing months
    tr = get_TR_from_PR_and_divs(pr, dv).dropna() # join divs and PR to TR
    tr = tr.asfreq("MS") # we need the index to know its frequence
    if not inf_adj: # data is inflation adjusted by default
        cpi = get(frm.cpiUS)
        tr = (tr * cpi).dropna()
    nm = f"~S&P500 {'NTR' if taxes else 'TR'} {'inf-adj' if inf_adj else 'nominal'}"
    return name(tr, nm)

def diff_dates(s):
    return s.index[s!=s.shift(1)]

def us_recession_dates():
    r = get('FRED/JHDUSRGDPBR@Q', interpolate=False, drop_zero=False)
    return diff_dates(r)

