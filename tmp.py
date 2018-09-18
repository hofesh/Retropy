from Retropy_framework import *
from framework.cefs import *

cefs = cef_non_us_other
cefs_all = get(cefs, source="Y")
show_rr(*cefs_all, ret_func=get_curr_yield_normal_no_fees, risk_func=get_cef_curr_premium)