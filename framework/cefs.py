# https://www.cefconnect.com/closed-end-funds-screener
# https://www.cefa.com/

# http://cefdata.com/funds/clm/
# https://www.cefconnect.com/fund/CLM
# https://www.cefchannel.com/clm/


from framework.utils import *
from framework.base import *
from framework.meta_data import *
from framework.stats_basic import *
from framework.stats import *
import Retropy_framework as frm
import framework.meta_data_dfs as meta_dfs

# these lists and categories are based on cefconnect
# cef_highyield_taxable = 'ACP|ACV|AFT|AIF|ARDC|AVK|AWF|BBN|BCV|BGB|BGH|BGT|BGX|BHK|BIT|BKT|BLW|BSL|BTZ|CBH|CCD|CHI|CHY|CIF|CIK|DBL|DCF|DFP|DHF|DHY|DMO|DSL|DSU|DUC|EAD|ECC|ECF|EFF|EFL|EFR|EFT|EGF|EHT|ERC|EVF|EVG|EVV|EXD|FCT|FFC|MGF|MPV|NBB|NBD|NCV|NCZ|NHS|NSL|OPP|OXLC|PAI|PCF|PCI|PCM|PCN|PDI|PDT|PFD|PFL|PFN|PFO|PGP|PHD|PHK|PHT|PIM|PKO|PPR|PSF|PTY|RA|TLI|TSI|TSLF|VBF|VGI|VLT|VTA|VVR|WEA|WIA|WIW|XFLT|FIV|FLC|FMY|FPF|FRA|FSD|FT|FTF|GBAB|GDO|GFY|GGM|GHY|HFRO|HIO|HIX|HNW|HPF|HPI|HPS|HYB|HYI|HYT|IGI|IHIT|IHTA|INSI|ISD|IVH|JCO|JFR|JHA|JHB|JHD|JHI|JHS|JHY|JLS|JMM|JMT|JPC|JPI|JPS|JPT|JQC|JRO|JSD|KIO|LDP|MCI'
# cef_highyield_taxable = cef_highyield_taxable.split('|')

# cef_municipal_tax_exempt = 'AKP|AFB|BJZ|BFZ|BFO|BKN|BTA|BZM|BPK|BAF|BYM|MUI|MNE|BTT|MUA|MEN|MUC|MHD|MUH|MFL|MUJ|MHN|MUE|MUS|MVF|MVT|MYD|MZA|MYC|MCA|MYF|MIY|MYJ|MYN|MPA|MYI|MQY|MQT|BKK|BBK|BFK|BLE|BBF|MFT|BLH|BSE|BQH|BFY|BNY|BSD|BHV|DTF|VFL|VMM|KTF|KSM|DMB|DMF|DSM|LEO|EIA|EVM|CEV|MAB|MMV|MIW|EMI|ETX|EIM|EIV|EVN|EMJ|EVJ|NYH|ENX|EVY|EOT|EIO|EVO|EIP|EVP|FMN|VKI|VCV|OIA|VGM|VMO|VKQ|VPV|IQI|VTN|IIM|CCA|CXE|CMU|CXH|MFM|MMD|MHE|NBW|NBH|NBO|NVG|NUW|NEA|NAZ|NKX|NCA|NCB|NAC|NXC|NTC|NEV|NKG|NIQ|NID|NMT|NMY|NUM|NMS|NOM|NHA|NZF|NMZ|NMI|NUV|NNC|NJV|NXJ|NRK|NNY|NYV|NAN|NXN|NUO|NPN|NQP|NAD|NIM|NXP|NXQ|NXR|NTX|NPV|PCQ|PCK|PZC|PMF|PML|PMX|PNF|PNI|PYN|MAV|MHI|PMM|PMO|SBI|MMU|MTT|MHF|MNP'
# cef_municipal_tax_exempt = cef_municipal_tax_exempt.split('|')

# cef_us_equity = 'NIE|ASA|AWP|ADX|PEO|NFJ|BGR|CII|BDJ|BOE|BGY|BME|BCX|BUI|BIF|IGR|CSQ|CEN|CET|CBA|CEM|EMO|CTR|FOF|INB|UTF|MIE|RQI|RNP|RFI|STK|CLM|CRF|SRF|SRV|SZC|DNP|DDF|DNI|DPG|DSE|EOI|EOS|ETJ|ETO|ETG|EVT|ETB|ETV|ETY|ETW|EXG|GRF|FMO|FEN|FIF|FFA|FEI|FPL|FGB|FXBY|GGN|GNT|GCV|GDV|GAB|GLU|GGO|GRX|GUT|GAM|GMZ|GER|GPM|GOF|BTO|HTY|HTD|KYN|KMF|USA|ASG|MFV|MFD|MGU|MCN|MSP|HIE|NML|NRO|NHF|JMLP|JCE|DIAX|JMF|QQQX|JRI|JRS|BXMX|SPXX|JTD|JTA|PGZ|RCG|RIF|UTG|RIV|RMT|RVT|SMM|SOR|SPE|FUND|CEF|PHYS|SPPP|PSLV|HQH|THQ|HQL|THW|NDP|TYG|NTG|TTP|TPZ|TY|ZTR|IGD|IGA|IRR|ERH'
# cef_us_equity = cef_us_equity.split('|')

# cef_non_us_other = 'FAX|IAF|AEF|AGD|FCO|JEQ|AOD|APB|BGIO|BST|BWG|INF|CHW|CGO|CEE|CHN|GLV|GLQ|GLO|DEX|VCF|KMM|KST|EGIF|EEA|FDEU|FEO|FAM|GDL|GGZ|GGT|CUBA|IFN|HEQ|JOF|KF|SCD|LDF|LGI|LOR|MCR|MIN|MMT|APF|CAF|MSF|MSD|EDD|IIF|MXE|MXF|GF|IRL|JDD|JEMD|JGH|RCS|PPT|RGT|EDF|EDI|SWZ|TWN|TDF|EMF|TEI|GIM|ZF|IAE|IHD|IDE|IID|EOD|EMD|EHI'
# cef_non_us_other = cef_non_us_other.split('|')

# all = cef_highyield_taxable + cef_municipal_tax_exempt + cef_us_equity + cef_non_us_other

# http://cefdata.com/screener/
df = meta_dfs.cef_metadata_df

taxable_bond_funds = df.query('sec_main == "Taxable Bond Funds"').index.tolist()

specialty_equity_funds = df.query('sec_main == "Specialty Equity Funds"').index.tolist()
specialty_prefered = df.query('sec_sub == "Prefered Equity Leveraged" and sec_main == "Specialty Equity Funds"').index.tolist()
specialty_covered_call = df.query('sec_sub == "Covered Call Funds" and sec_main == "Specialty Equity Funds"').index.tolist()
specialty_mlp = df.query('sec_sub == "MLP Funds" and sec_main == "Specialty Equity Funds"').index.tolist()
specialty_real_estate = df.query('sec_sub == "Global Real Estate, REIT &amp; Real Assets" and sec_main == "Specialty Equity Funds"').index.tolist()
specialty_utilities = df.query('sec_sub == "Utilities Funds" and sec_main == "Specialty Equity Funds"').index.tolist()

national_muni_bond_funds = df.query('sec_main == "National Muni Bond Funds"').index.tolist()
taxable_muni_bond_funds = df.query('sec_sub == "Taxable Municipal Bond Funds"').index.tolist()

fixed_income = taxable_bond_funds + specialty_prefered + specialty_real_estate + taxable_muni_bond_funds

all =  df.index.tolist()

pimco = df.query('sponsor == "PacificInvestmentManagementCompany"').index.tolist()


# only_new = set(cef_taxable_bond_funds) - set(cef_highyield_taxable)
# only_old = set(cef_highyield_taxable) - set(cef_taxable_bond_funds)
# cef_old_and_new = set(cef_taxable_bond_funds) | set(cef_highyield_taxable)

# these have broken/corrupt data in Yahoo
cefs_bad_yahoo = ['EHT', 'DCF', 'JHY', 'CBH', 'CCD', 'FIV', 'JPT', 'JHD', 'EFL', 'HFRO', 'CBH', 'GGO', 'BST', 'JCO', 'FTSM'] # these have broken/corrupt data in Yahoo

cef_nav_map = {
    'ARDC': 'XADCX',
    'OXLC': 'OXLCX',
    'RA': 'XRAIX',
    'TSI': 'XXCVTXX',
    'TSLF': 'XTSLX',
    'XFLT': 'XFLTX',
    'FT': 'XFUTX',
    'GBAB': 'XGBAX',
    'HFRO': 'XHFOX',
    'HIX': 'XHGIX',
    'HYB': 'XHYBX',
    'IHIT': 'XHITX',
    'IHTA': 'XHTAX',
    'INSI': 'XBDFX',
    'BGX': 'XXBGX'
}
def get_cef_nav_ticker(s):
    if is_series(s):
        s = s.name.ticker
    nav = get_cef_meta(s, "nav_symbol")
    if nav:
        return nav
    nav = cef_nav_map.get(s, '')
    if not nav:
        nav = f'X{s}X'
    return nav

def get_cef_nav(s, source=None):
    if s is None:
        return None
    if not is_cef(s):
        return None
    if is_series(s) and source is None:
        source = s.name.source
    source = source or "AV"
    return get(get_cef_nav_ticker(s), source=source, mode="PR", error='ignore', cache_fails=True)

def get_cef_premium(s, source="AV"):
    nav = get_cef_nav(s, source=source)
    if nav is None:
        warn(f'Unable to get NAV for {get_pretty_name(s)}')
        return None
    pr = get(s, source=source, mode="PR")
    if pr.index[-1] > nav.index[-1]:
        warn(f"{get_ticker_name(s)} filling NAV history gap from {nav.index[-1]} to {pr.index[-1]}")
        nav = nav.reindex(pr.index).fillna(method='ffill')
    prem = (pr / nav - 1) * 100
    return name(prem, f"{get_name(s)} premium")

def show_cef_premium(*all):
    frm.show(1, lmap(get_cef_premium, all), ta=False, log=False, title="Premium")

def show_cef_zscore(*all):
    frm.show(-2, -1, 1, 2, lmap(show_cef_zscore, all), ta=False, log=False, title="3y z-score")

def show_cef_nav_and_pr(*all):
    frm.show(0, lmap(get_cef_nav, all), lmap(pr, all), ta=False, title="NAV and Price")

def show_cef_nav_and_ntr(*all):
    frm.show(lmap(get_cef_nav_ntr, all), lmap(ntr, all), ta=False, title="NAV and market NTR")

def get_cef_curr_premium(s):
    if not is_cef(s):
        return 0
    p = get_cef_premium(s)
    if p is None:
        # warn(f"can't get calculated premium, using meta_data premium instead for {get_ticker_name(s)}")
        p = get_cef_meta(s, "premium")
        if not p is None:
            warn(f"can't get calculated premium, using meta_data premium instead for {get_ticker_name(s)}")
            return p
        return None
    return p.dropna()[-1]

def get_cef_start_premium(s):
    p = get_cef_premium(s)
    if p is None:
        return None
    return p.dropna()[0]


def get_cef_nav_yield_no_fees(s, type='normal'):
    return get_cef_nav_yield(s, type=type, reduce_fees=False)

def get_cef_nav_yield(s, type='normal', reduce_fees=True, source="AV"):
    nav = get_cef_nav(s)
    if nav is None:
        return None
    res = frm.get_yield(s, type=type, altPriceName=get_cef_nav_ticker(s)+"@"+source, reduce_fees=reduce_fees)
    name(res, f"{res.name} NAV")
    return res

def get_cef_curr_nav_yield_no_fees(s):
    res = get_cef_nav_yield_no_fees(s)
    return 0 if res is None or len(res) == 0 else res[-1]

def get_cef_cur_nav_yield(s):
    res = get_cef_nav_yield(s)
    return 0 if res is None or len(res) == 0 else res[-1]

def get_cef_zscore(s, period=365*3):
    p = get_cef_premium(s)
    if p is None:
        return None
    p_avg = ma(p, period)
    p_std = mstd(p, period)
    p_zscore = (p-p_avg)/p_std
    return name(p_zscore, f"{get_name(s)} zscore")

def get_cef_curr_zscore(s, period=365*3):
    if not is_cef(s):
        return None
    res = get_cef_zscore(s, period=period)
    return None if res is None or len(res) == 0 else res[-1]

def show_cef_zscore(*all):
    frm.show(-2, -1, 1, 2, lmap(get_cef_zscore, all), ta=False, log=False, title="3y z-score")

def get_cef_nav_ntr(s):
    if not is_cef(s):
        return None
    if get_cef_nav(s) is None:
        return None
    return name(getNtr(s, {"mode": "NTR"}, alt_price_symbol=get_cef_nav_ticker(s)), f"{get_name(s, nomode=True)} NAV NTR")

def get_cef_nav_intr(s):
    if get_cef_nav(s) is None:
        return None
    return name(get_intr(s, {"mode": "NTR"}, alt_price_symbol=get_cef_nav_ticker(s)), f"{get_name(s, nomode=True)} NAV INTR")


def show_cef_premium_and_returns(s):
    nav = get_cef_nav_ntr(s)
    if nav is None:
        return
    frm.show(get_cef_premium(s), (ntr(s) / nav - 1)*100, ta=False, title="Effect of premium/discount of NTR returns")

def show_cef_relative_premium(a, b):
    a = get_cef_premium(a)
    b = get_cef_premium(b)
    if a is None or b is None:
        return
    frm.show(1, a - b, ta=False, log=False, title="Relative Premium")

def analyze_cef(s, base='SPY'):
    s = get(s)
    if is_cef(base):
        base_cef = base
        base_ntr = ntr(base)
    else:
        base_cef = None
    ntr_s = ntr(s)
    frm.show(get_cef_premium(s), frm.get_income(s, smooth=1)/10, ta=False)
    if not base_cef is None:
        frm.show(get_cef_premium(base_cef), frm.get_income(base_cef, smooth=1)/10, ta=False)
    frm.show(0, 5, get_cef_nav_yield(ntr_s, type='true', reduce_fees=False), get_yield_true_no_fees(ntr_s), ta=False, title="NAV and Market net-yield (no fees)")
    if not base_cef is None:
        frm.show(0, 5, get_cef_nav_yield(base_ntr, type='true', reduce_fees=False), get_yield_true_no_fees(base_ntr), ta=False, title="NAV and Market net-yield (no fees)")
    show_cef_premium(s, base_cef)
    if not base_cef is None:
        show_cef_relative_premium(s, base)
    show_cef_premium_and_returns(s)
    show_cef_zscore(s, base_cef)
    show_cef_nav_and_pr(s, base_cef)
    show_cef_nav_and_ntr(s, base_cef)
    frm.show_dd(get_cef_nav(s), get_cef_nav_ntr(s), get_cef_nav(base_cef), get_cef_nav_ntr(base_cef), do_get=False, mode='', title_prefix="NAV / NAV-NTR")
    frm.show_comp(s, base)

def get_pr_loss_last_week(s):
    # if is_cef(s):
    #     nav = get_cef_nav(s)
    #     if nav is None:
    #         return None
    # else:
    nav = get(s, mode="PR", untrim=True)
    nav = nav["2018-10-01":]
    if len(nav) < 10:
        return None
    return (nav[-1] / nav[0] - 1) * 100
    
def get_cef_nav_loss_2010(s):
    if is_cef(s):
        nav = get_cef_nav(s)
        if nav is None:
            return None
        return -cagr(lr(nav["2010":]))
    else:
        return -cagr(lr(get(s, mode="PR", untrim=True)["2010":]))
    
def get_cef_nav_loss_2013(s):
    if is_cef(s):
        nav = get_cef_nav(s)
        if nav is None:
            return None
        return -cagr(lr(nav["2013":]))
    else:
        return -cagr(lr(get(s, mode="PR", untrim=True)["2013":]))
    
def get_cef_roc_3y(s):
    return get_cef_meta(s, "roc_3y")

def get_cef_coverage(s):
    r = get_cef_meta(s, "coverage")
    if r is None:
        return r
    return r - 100

def get_cef_leverage(s):
    return get_cef_meta(s, "total_leverage")

def get_cef_nav_or_pr(s, untrim):
    if is_cef(s):
        nav = get_cef_nav(s)
        if not nav is None:
            if not untrim:
                nav = nav[s.index[0]:]
            return nav
    return get(s, mode="PR", untrim=untrim)
    
def ulcer_nav(s):
    return ulcer(get_cef_nav_or_pr(s, untrim=False))

def ulcer_nav_ntr(s):
    if not is_cef(s):
        return None
    ntr = get_cef_nav_ntr(s)
    if ntr is None:
        return None
    return ulcer(ntr)

def get_cef_section(s):
    sec = get_etf_cef_meta(s, 'yc_sub_category', 'sec_sub')
    if not sec:
        return '<NA>'
    return sec.replace(" Bond Funds", "").replace(" Equity Leveraged", "").replace(" Bond", "").replace(" Funds", "").replace("Taxable Municipal", "Municipal").replace("US Government", "US Govt").replace("Emerging Market Income", "EM Income").replace("Global Real Estate, REIT &amp; Real Assets", 'Real Estate')

def get_cef_maxdd_nav_ntr(s):
    ntr = get_cef_nav_ntr(s)
    if ntr is None:
        return None
    #ntr = drop_outliers(ntr) # this completely messes NRO for example
    if len(ntr) == 0:
        return None
    return max_dd(ntr)

def get_cef_maxdd_nav_ntr_2008(s):
    nav = get_cef_nav(s)
    if nav is None or len(nav[:"2007-02"]) == 0:
        return None
    return get_cef_maxdd_nav_ntr(get(s, untrim=True)["2007-02":])

def get_sponsor(s):
    spn = get_cef_meta(s, "sponsor")
    if not spn:
        return None
    return spn.replace("Fund", '').replace("Advisors", '').replace("Management", '').replace("Partners", '').replace("Investment", '').replace("Company", '').replace("Advisers", '').replace("Services", '').replace("Financial", '').replace("Capital", '').replace("Management", '').replace('Incorporated', '')
