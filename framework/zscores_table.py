from scipy.stats import zscore, zmap
import numpy as np
import math

from framework.utils import *
from framework.symbol import *
from framework.base import *
from framework.cefs import *



# NOTE: this method rewards USD-bonds and they express lower risk-volatility
def zscores(all, *funcs, weights=None):
    if weights is None:
        weights = [1] * len(funcs)
    d_vals = {get_func_name(f): lmap(f, all) for f in funcs}
    d = dict(d_vals)
    d["name"] = names(all)
    df = pd.DataFrame(d).set_index("name")
    
    df = zscores_update(df, weights)
    return df

    # d_zs = {k: zscore_modified(v)*sign for (k, v), sign in zip(d_vals.items(), weights)}
    # d_zs["name"] = names(all)
    # df_z = pd.DataFrame(d_zs).set_index("name")

    # zmean = df_z.mean(axis=1)
    # df["zmean"] = zmean
    # df = df.sort_values("zmean", ascending=False)
    # return df

def zscores_update(df, weights):
    def apply_w(v, w):
        if isinstance(w, tuple):
            w, f = w
            if f == "log":
                return np.log(v) * w
            raise Exception(f"unsupported weight: {f}")
        if is_number(w):
            return v * w
        if w == "log":
            return np.log(v)
        raise Exception(f"unsupported weight: {w}")

    if "zmean" in df.columns:
        df = df.drop("zmean", axis=1)
    d_zs = {k: apply_w(zscore_modified(df[k]), w) for k, w in zip(df.columns, weights)}
    d_zs["name"] = df.index
    df_z = pd.DataFrame(d_zs).set_index("name")

    zmean = df_z.sum(axis=1) / df_z.shape[1]
    df["zmean"] = zmean
    df = df.sort_values("zmean", ascending=False)

    return df

def zscore_df_style(df, names):
    df.columns = names
    df.index = df.index.str.replace(" NTR", "")

    import seaborn as sns
    cm = sns.light_palette("orange", as_cmap=True)
    #df.style.background_gradient(cmap=cm)
    
    #df.style.highlight_max(axis=0, color='green').highlight_min(axis=0, color='red')
    
    #df.style.bar(subset=['cagr', 'get_curr_yield_min2', 'mutual_dd_rolling_pr_SPY', 'ulcer_pr', 'get_meta_aum_log', 'get_meta_fee'], align='left', color=['#5fba7d'])
    return df.style\
        .bar(subset=['cagr', 'start_yield', 'curr_yield'], align='left', color=['#5fba7d'])\
        .bar(subset=['ulcer_pr'], align='left', color=['#d65f5f'])\
        .bar(subset=['mutual_dd'], align='mid', color=['#5fba7d', '#d65f5f'])\
        .bar(subset=['aum'], align='left', color=['#9fdfbe'])\
        .bar(subset=['fee'], align='left', color=['#ffb3b3'])\
        .format({'aum': "{:,.0f}"})

def highlight_name(s, marks=None):
    def color(x):
        x = x.replace("*", "")
        if marks:
            for mrk in marks:
                if x in mrk[0]:
                    return mrk[1]
        if is_etf(x):
            return 'background-color: cyan'
        return ''
    return s.apply(color)
#    return ['background-color: yellow' if v else '' for v in s.apply(lambda x: x.replace("*", "")).isin(top)]
    
def highlight_sec(s):
    def color(x):
        if x == "Prefered":
            return 'background-color: #CD6155'
        if x == "Municipal":
            return 'background-color: #AF7AC5'
        if x == "Prefered":
            return 'background-color: #5499C7'
        if x == "High Yield":
            return 'background-color: #48C9B0'
        if x == "Covered Call":
            return 'background-color: #F8C9B0'
        if x == "Multisector":
            return 'background-color: #F4D03F'
        if x == "Investment Grade":
            return 'background-color: #E67E22'
        if x == "Mortgage":
            return 'background-color: #BDC3C7'
        if x == "Limited Duration":
            return 'background-color: #9A7D0A'
        if x == "Loan Participation":
            return 'background-color: #D2B4DE'
        return ''
    return s.apply(color)
#    return ['background-color: yellow' if v else '' for v in s.apply(lambda x: x.replace("*", "")).isin(top)]
    
#     is_max = s == s.max()
#     return ['background-color: yellow' if v else '' for v in is_max]    

def zscore_df_style(df, names, marks, fillna):
    df.columns = names
    df.index = df.index.str.replace(" NTR", "").str.replace(" TR", "").str.replace("@AV", "").str.replace("@Y", "")

    import seaborn as sns
    cm = sns.light_palette("orange", as_cmap=True)
    #df.style.background_gradient(cmap=cm)
    
    #df.style.highlight_max(axis=0, color='green').highlight_min(axis=0, color='red')
    
    #df.style.bar(subset=['cagr', 'get_curr_yield_min2', 'mutual_dd_rolling_pr_SPY', 'ulcer_pr', 'get_meta_aum_log', 'get_meta_fee'], align='left', color=['#5fba7d'])
    # fillna(0).
    if fillna:
        df = df.fillna(0)
    return df.reset_index().style\
        .bar(subset=['nav_loss_2010', 'nav_loss_2013', 'premium', 'mutual_dd', 'DC', 'zscr'], align='mid', color=['#5fba7d', '#d65f5f'])\
        .bar(subset=['last_week', 'cagr', 'nn_yield', 'yld_zs', 'coverage'], align='mid', color=['#d65f5f', '#5fba7d'])\
        .bar(subset=['UC', 'usd_corr'], align='left', color=['#5fba7d'])\
        .bar(subset=['ulcer_pr_rol', 'ulcer_pr', 'ulcer_nav', 'u_nav_ntr', 'income_ulcer', 'roc_3y', 'ntr_maxdd', 'ntr_mxd_08'], align='left', color=['#d65f5f'])\
        .bar(subset=['start_yield', 'n_yield', 'm_yield'], align='left', color=['gray'])\
        .bar(subset=['aum'], align='left', color=['#9fdfbe'])\
        .bar(subset=['fee', 'usd_pval', 'lev'], align='left', color=['#ffb3b3'])\
        .format({'aum': "{:,.0f}"})\
        .format({'n_yield': "{:.2f}%"})\
        .format({'m_yield': "{:.2f}%"})\
        .format({'nn_yield': "{:.2f}%"})\
        .format({'start_yield': "{:.2f}%"})\
        .format({'cagr': "{:.2f}%"})\
        .format({'nav_loss_2010': "{:.2f}%"})\
        .format({'nav_loss_2013': "{:.2f}%"})\
        .format({'last_week': "{:.2f}%"})\
        .format({'premium': "{:.1f}%"})\
        .format({'lev': "{:.0f}%"})\
        .format({'income_ulcer': "{:.2f}"})\
        .format({'zscr': "{:.2f}"})\
        .format({'ulcer_pr_rol': "{:.2f}"})\
        .format({'ulcer_pr': "{:.2f}"})\
        .format({'ulcer_nav': "{:.2f}"})\
        .format({'u_nav_ntr': "{:.2f}"})\
        .format({'usd_corr': "{:.2f}"})\
        .format({'usd_pval': "{:.2f}"})\
        .format({'ntr_maxdd': "{:.2f}"})\
        .format({'ntr_mxd_08': "{:.2f}"})\
        .format({'coverage': "{:.1f}"})\
        .format({'mutual_dd': "{:.2f}"})\
        .format({'yld_zs': "{:.2f}"})\
        .format({'UC': "{:.0f}"})\
        .format({'DC': "{:.0f}"})\
        .format({'zmean': "{:.2f}"})\
        .apply(partial(highlight_name, marks=marks), subset=['name'])\
        .apply(highlight_sec, subset=['sec'])\
        .hide_index()


def display_zscores(all, n=None, idx=None, funcs=None, names=None, weights=None, _cache=[None], marks=None, fillna=False):
    if funcs is None:
        funcs=[get_cef_section, get_sponsor, get_usd_corr, get_usd_pvalue, get_cef_roc_3y, get_cef_coverage, get_income_ulcer,   get_cef_leverage, get_cef_curr_premium, get_cef_curr_zscore, get_cef_nav_loss_2010, get_cef_nav_loss_2013, get_pr_loss_last_week, get_upside_capture_SPY, cagr, get_start_yield, get_meta_yield, get_curr_yield_normal_no_fees, get_curr_yield_min2, get_curr_yield_zscore, ulcer_pr_rolling, ulcer_pr, ulcer_nav, ulcer_nav_ntr, mutual_dd_rolling_pr_SPY, get_downside_capture_SPY, get_cef_maxdd_nav_ntr, get_cef_maxdd_nav_ntr_2008, get_meta_aum, get_meta_fee]
    if names is None:
        names = ['sec',         'sponsor',  'usd_corr',    'usd_pval',     'roc_3y',       'coverage',         'income_ulcer',   'lev',            'premium',            'zscr',                'nav_loss_2010',      'nav_loss_2013',     'last_week',            'UC',                  'cagr', 'start_yield', 'm_yield',      'n_yield',                      'nn_yield',         'yld_zs',              'ulcer_pr_rol',   'ulcer_pr', 'ulcer_nav', 'u_nav_ntr', 'mutual_dd',              'DC',                    'ntr_maxdd',            'ntr_mxd_08',                 'aum',     'fee', 'zmean']
    if weights is None:
#        weights=[0,               0,        0,            0,               0,              1,                 -1,                -5,               -1,                  -1,                   -5,                    -5,                   0,                    1,                     1,      0,             5,              5,                              1,                 1,                      -5,                 -5,         -5,           -1,              -5,                   -5,                     -1,                      -1,                          0,        -1         ]
#        weights=[0,               0,        0,            0,               0,              1,                 -1,                -5,               -1,                  -1,                   -5,                    -5,                   0,                    1,                     1,      0,             0,              0,                              1,                 1,                      -5,                 -5,         -5,           -1,              -5,                   -5,                     -1,                      -1,                          0,        -1         ]
#        weights=[0,               0,        0,            0,               0,              1,                 -1,                -5,               -1,                  -1,                   -5,                    -5,                   0,                    1,                     1,      0,             1,              1,                              1,                 1,                      -5,                 -5,         -5,           -1,              -5,                   -5,                     -1,                      -1,                         1,        -1         ]
#        weights=[0,               0,        0,            0,               0,              0,                 -0,                -5,               -0,                  -0,                   -5,                    -5,                   0,                    0,                     0,      0,             1,              0,                              0,                 0,                      -5,                 -5,         -5,           -0,              -5,                   -5,                     -0,                      -0,                         0,        -0         ]
# CEFS:-20,               -0,                   -5,                 
        weights=[0,               0,        0,            0,               0,              2,                 -2,                -2,               -2,                  -2,                   -2,                    -2,                   2,                    2,                     2,      0,              2,              2,                              0,                2,                    -2,                   -2,         -2,           -2,              -2,                   -2,                     -2,                      -2,                         0,        -2         ]
#         weights=[0,               0,        0,            0,               0,              2,                 -2,                -5,               -2,                  -2,                   -50,                    -50,                   10,                    2,                     2,      0,              10,              10,                              0,                5,                    -50,                   -50,         -4,           -4,              -10,                   -10,                     -5,                      -5,                         2,        -2         ]
#         weights=[0,               0,        0,            0,               0,              2,                 -2,                -5,               -2,                  -2,                   -100,                    -100,                   10,                    2,                     2,      0,              10,              10,                              0,                5,                 -500,                -500,      -4,         -4,              -100,                   -10,                     -5,                      -5,                         2,        -2         ]
#        weights=[0,               0,        20,          -20,              0,              2,                 -2,                -5,               -2,                  -2,                   -2,                    -2,                   0,                    2,                     2,      0,              10,              10,                              0,                5,                    -5,                   -4,         -4,           -4,              -10,                   -10,                     -5,                      -5,                         2,        -2         ]
#        weights=[0,               0,         0,           0,               0,              2,                 -2,                -5,               -2,                  -2,                   -20,                    -20,                   50,                    2,                     2,      0,              10,              10,                              0,                5,                 -5,                      -4,         -4,           -4,              -10,                   -10,                     -5,                      -5,                         2,        -2         ]
#        weights=[0,               0,        0,            0,               0,              2,                 -2,                -5,               -2,                  -2,                    -2,                    -2,                   200,                    2,                     2,      0,              20,              0,                              0,                5,                  -5,                     -4,         -4,           -4,              -10,                   -10,                     -5,                      -5,                         2,        -2         ]
#        weights=[0,               0,        0,            0,               0,              2,                 -2,                -5,               -2,                  -2,                   -2,                    -2,                   5,                    2,                     2,      0,              3,              3,                              0,                 5,                     -5,                  -4,         -4,           -4,              -10,                   -10,                     -5,                      -5,                         2,        -2         ]
#        weights=[0,               0,        0,            0,               0,              2,                 -2,                -5,               -2,                  -2,                   -2,                    -2,                   5,                    2,                     2,      0,              3,              30000,                              0,                 5,                 -5,                      -4,         -4,           -4,              -10,                   -10,                     -5,                      -5,                         2,        -2         ]
# ETFS:-20,               -0,                0,            0,                 -5,                 
#        weights=[0,               0,        0,            0,               0,              1,                 -1,                -5,               -1,                  -1,                   -5,                    -5,                   0,                    1,                     1,      0,             0,              0,                              1,                 1,                      -5,                 -5,         -5,           -1,              -5,                   -5,                     -1,                      -1,                          0,        -1         ]

    print(f"weights: {weights}")
    dfz = _cache[0]
    if dfz is None:
        dfz = zscores(all, *funcs, weights=weights)
    #_cache[0] = dfz
    if not n:
        n = dfz.shape[0]
    with pd.option_context('display.max_rows', n):
        df = zscores_update(dfz, weights)
        _cache[0] = df
        if not idx is None:
            df = df[idx]
        display(zscore_df_style(df[:n], names, marks, fillna=fillna))

