from framework.asset_classes import *

# this is list of high yield ETFs
all = 'ALTY|AMJ|AMJL|AMLP|AMU|AMUB|AMZA|AOK|ATMP|BDCL|BDCS|BDCZ|BIZD|BLHY|BMLP|BNDX|CDC|CDL|CEFL|CEFS|CEY|CJNK|COMT|CSB|CWB|DBUK|DES|DGRS|DIV|DRW|DTN|DVHL|DVYL|DWFI|EBND|ENY|EPRF|ERUS|EUFL|EWH|EWM|EWY|FAUS|FCVT|FDIV|FEMB|FFR|FFTI|FLN|FPA|FPE|FSZ|FXEP|FXEU|FXU|GCE|GHII|GHYG|GRI|GYLD|HDLV|HDRW|HEWL|HSPX|HYDB|HYEM|HYHG|HYIH|HYLD|HYLS|HYXE|IDHD|IDLV|IFGL|IMLP|IPE|IQDE|ISHG|JPGB|KBWD|KBWY|LBDC|LMLP|LRET|MDIV|MLPA|MLPB|MLPC|MLPE|MLPG|MLPI|MLPO|MLPQ|MLPY|MLPZ|MORT|NFLT|OASI|OEUR|ONTL|OUSM|PAF|PCEF|PELBX|PEX|PFFD|PFFR|PFXF|PGF|PRME|PSCF|PSCU|PSK|PSP|PXJ|PXR|QXMI|QYLD|REM|RORE|SDIV|SDYL|SEA|SMHD|SOVB|SPFF|SPMV|SPVU|SRET|STPZ|TAO|TIPX|TIPZ|URA|VSMV|VSS|VTIP|WFHY|WPS|YDIV|YESR|YMLI|YMLP|YYY|ZMLP'
all = all.split('|')

###################
lev = ['HDLV', 'DVYL', 'SMHD', 'DVHL', 'LBDC', 'BDCL', 'CEFL', 'LMLP', 'SDYL', 'MORL', 'REML', 'LRET']
hy_lowvol = ['SRET', 'SPHD', 'LVHD'] # high yield low volatility
mortgage = ['REM', 'MORT']
reits = ['KBWY', 'DRW', 'HDRW', reit, i_reit, "VNQ", "VNQI"] # WPS, IFGL
financials = ['KBWD', 'PSCF']
preferred = ['PSK', 'FPE', 'PGF', 'PGX', 'PFF'] # PGF is financial sector
# https://investorplace.com/2017/04/5-high-yield-cef-etfs/
# http://investwithanedge.com/etfs-of-cefs
closed_end = ['PCEF', 'YYY', 'XMPT', 'FCEF', 'CEFS', 'MCEF', 'GCE'] # lev: CEFL
priv_equity = ['PSP', 'PEX']
biz_dev = ['BIZD', 'BDCS', 'BDCZ'] # 'BUY', bdcs_fit
alts = ['ALTY']
corp_bonds = [cjb, 'ANGL']
irate_hedge = ['HYIH']
div_stocks = ['SDIV', 'DIV', 'DVY']
mlp = ['AMZA', 'AMJ', 'AMJL', 'YMLP', 'MLPQ', 'ZMLP', 'YMLI', 'AMUB', 'ATMP']
call_put = ['QYLD'] # limited upside, unlimited downside, and PR drawdown is higher than QQQ.
em_bonds = ['EBND', 'EMB', em_gbLcl, em_gbUsd]
infrastructure = ['GHII', 'PSCU'] # + utilities
short_term_high_yield = ['PGHY', 'LMBS']
other = ['MDIV']

base_stocks = ['QQQ', 'SPY', 'VXUS', i_ac]
base_bonds = ['MINT', gb, 'AGG', 'EDV', 'TLT', lgb, elgb, 'BLV', 'IEI', 'IEF']
###################

low = ['SPHD', 'LVHD', 'DVY', 'EBND']
unwanted = ['PSCU', 'PSCF']
bad = ['LMLP', 'FLN', 'HYLD']
drop = 'URA|FIV|CCD|JPT|EHT|CBH'
drop = drop.split('|')

select = hy_lowvol + bad + mortgage + reits + financials + preferred + closed_end + priv_equity + biz_dev + alts + corp_bonds + irate_hedge + div_stocks + mlp + call_put + em_bonds + infrastructure + other + base_stocks + base_bonds
select = set(select) - set(low + unwanted + bad + drop)
select = list(select)

all = list(set(all + select))