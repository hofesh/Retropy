# https://www.cefconnect.com/closed-end-funds-screener
# https://www.cefa.com/

# http://cefdata.com/funds/clm/
# https://www.cefconnect.com/fund/CLM
# https://www.cefchannel.com/clm/


from framework.utils import *
from framework.base import *
from framework.meta_data import *
from framework.stats_basic import *
import Retropy_framework as frm

# these lists and categories are based on cefconnect
cef_highyield_taxable = 'ACP|ACV|AFT|AIF|ARDC|AVK|AWF|BBN|BCV|BGB|BGH|BGT|BGX|BHK|BIT|BKT|BLW|BSL|BTZ|CBH|CCD|CHI|CHY|CIF|CIK|DBL|DCF|DFP|DHF|DHY|DMO|DSL|DSU|DUC|EAD|ECC|ECF|EFF|EFL|EFR|EFT|EGF|EHT|ERC|EVF|EVG|EVV|EXD|FCT|FFC|MGF|MPV|NBB|NBD|NCV|NCZ|NHS|NSL|OPP|OXLC|PAI|PCF|PCI|PCM|PCN|PDI|PDT|PFD|PFL|PFN|PFO|PGP|PHD|PHK|PHT|PIM|PKO|PPR|PSF|PTY|RA|TLI|TSI|TSLF|VBF|VGI|VLT|VTA|VVR|WEA|WIA|WIW|XFLT|FIV|FLC|FMY|FPF|FRA|FSD|FT|FTF|GBAB|GDO|GFY|GGM|GHY|HFRO|HIO|HIX|HNW|HPF|HPI|HPS|HYB|HYI|HYT|IGI|IHIT|IHTA|INSI|ISD|IVH|JCO|JFR|JHA|JHB|JHD|JHI|JHS|JHY|JLS|JMM|JMT|JPC|JPI|JPS|JPT|JQC|JRO|JSD|KIO|LDP|MCI'
cef_highyield_taxable = cef_highyield_taxable.split('|')

cef_municipal_tax_exempt = 'AKP|AFB|BJZ|BFZ|BFO|BKN|BTA|BZM|BPK|BAF|BYM|MUI|MNE|BTT|MUA|MEN|MUC|MHD|MUH|MFL|MUJ|MHN|MUE|MUS|MVF|MVT|MYD|MZA|MYC|MCA|MYF|MIY|MYJ|MYN|MPA|MYI|MQY|MQT|BKK|BBK|BFK|BLE|BBF|MFT|BLH|BSE|BQH|BFY|BNY|BSD|BHV|DTF|VFL|VMM|KTF|KSM|DMB|DMF|DSM|LEO|EIA|EVM|CEV|MAB|MMV|MIW|EMI|ETX|EIM|EIV|EVN|EMJ|EVJ|NYH|ENX|EVY|EOT|EIO|EVO|EIP|EVP|FMN|VKI|VCV|OIA|VGM|VMO|VKQ|VPV|IQI|VTN|IIM|CCA|CXE|CMU|CXH|MFM|MMD|MHE|NBW|NBH|NBO|NVG|NUW|NEA|NAZ|NKX|NCA|NCB|NAC|NXC|NTC|NEV|NKG|NIQ|NID|NMT|NMY|NUM|NMS|NOM|NHA|NZF|NMZ|NMI|NUV|NNC|NJV|NXJ|NRK|NNY|NYV|NAN|NXN|NUO|NPN|NQP|NAD|NIM|NXP|NXQ|NXR|NTX|NPV|PCQ|PCK|PZC|PMF|PML|PMX|PNF|PNI|PYN|MAV|MHI|PMM|PMO|SBI|MMU|MTT|MHF|MNP'
cef_municipal_tax_exempt = cef_municipal_tax_exempt.split('|')

cef_us_equity = 'NIE|ASA|AWP|ADX|PEO|NFJ|BGR|CII|BDJ|BOE|BGY|BME|BCX|BUI|BIF|IGR|CSQ|CEN|CET|CBA|CEM|EMO|CTR|FOF|INB|UTF|MIE|RQI|RNP|RFI|STK|CLM|CRF|SRF|SRV|SZC|DNP|DDF|DNI|DPG|DSE|EOI|EOS|ETJ|ETO|ETG|EVT|ETB|ETV|ETY|ETW|EXG|GRF|FMO|FEN|FIF|FFA|FEI|FPL|FGB|FXBY|GGN|GNT|GCV|GDV|GAB|GLU|GGO|GRX|GUT|GAM|GMZ|GER|GPM|GOF|BTO|HTY|HTD|KYN|KMF|USA|ASG|MFV|MFD|MGU|MCN|MSP|HIE|NML|NRO|NHF|JMLP|JCE|DIAX|JMF|QQQX|JRI|JRS|BXMX|SPXX|JTD|JTA|PGZ|RCG|RIF|UTG|RIV|RMT|RVT|SMM|SOR|SPE|FUND|CEF|PHYS|SPPP|PSLV|HQH|THQ|HQL|THW|NDP|TYG|NTG|TTP|TPZ|TY|ZTR|IGD|IGA|IRR|ERH'
cef_us_equity = cef_us_equity.split('|')

cef_non_us_other = 'FAX|IAF|AEF|AGD|FCO|JEQ|AOD|APB|BGIO|BST|BWG|INF|CHW|CGO|CEE|CHN|GLV|GLQ|GLO|DEX|VCF|KMM|KST|EGIF|EEA|FDEU|FEO|FAM|GDL|GGZ|GGT|CUBA|IFN|HEQ|JOF|KF|SCD|LDF|LGI|LOR|MCR|MIN|MMT|APF|CAF|MSF|MSD|EDD|IIF|MXE|MXF|GF|IRL|JDD|JEMD|JGH|RCS|PPT|RGT|EDF|EDI|SWZ|TWN|TDF|EMF|TEI|GIM|ZF|IAE|IHD|IDE|IID|EOD|EMD|EHI'
cef_non_us_other = cef_non_us_other.split('|')

cef_all = cef_highyield_taxable + cef_municipal_tax_exempt + cef_us_equity + cef_non_us_other

# these have broken/corrupt data in Yahoo
cefs_bad_yahoo = ['EHT', 'DCF', 'JHY', 'CBH', 'CCD', 'FIV', 'JPT', 'JHD', 'EFL', 'HFRO', 'CBH', 'GGO', 'BST'] # these have broken/corrupt data in Yahoo

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

def get_cef_nav(s, source="Y"):
    return get(get_cef_nav_ticker(s), source=source, mode="PR", error='ignore', cache_fails=True)

def get_cef_premium(s, source="Y"):
    nav = get_cef_nav(s, source=source)
    if nav is None:
        warn(f'Unable to get NAV for {get_pretty_name(s)}')
        return None
    pr = get(s, source=source, mode="PR")
    return name((pr / nav - 1) * 100, f"{get_name(s)} premium")

def show_cef_premium(*all):
    frm.show(1, lmap(get_cef_premium, all), ta=False, log=False, title="Premium")

def show_cef_zscore(*all):
    frm.show(-2, -1, 1, 2, lmap(show_cef_zscore, all), ta=False, log=False, title="3y z-score")

def show_cef_nav_and_pr(*all):
    frm.show(lmap(get_cef_nav, all), lmap(pr, all), ta=False, title="NAV and Price")

def get_cef_curr_premium(s):
    p = get_cef_premium(s)
    if p is None:
        return None
    return p.dropna()[-1]

def get_cef_start_premium(s):
    p = get_cef_premium(s)
    if p is None:
        return None
    return p.dropna()[0]


def get_cef_nav_yield_rolling_no_fees(s):
    nav = get_cef_nav(s)
    if nav is None:
        return None
    res = frm.get_yield(s, type='rolling', altPriceName=get_cef_nav_ticker(s)+"@Y", reduce_fees=False)
    name(res, f"{res.name} NAV")
    return res

def get_cef_nav_yield_rolling(s):
    nav = get_cef_nav(s)
    if nav is None:
        return None
    res = frm.get_yield(s, type='rolling', altPriceName=get_cef_nav_ticker(s)+"@Y", reduce_fees=True)
    name(res, f"{res.name} NAV")
    return res

def get_cef_curr_nav_yield_rolling_no_fees(s):
    res = get_cef_nav_yield_rolling_no_fees(s)
    return 0 if res is None or len(res) == 0 else res[-1]

def get_cef_cur_nav_yield_rolling(s):
    res = get_cef_nav_yield_rolling(s)
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
    res = get_cef_zscore(s, period=period)
    return None if res is None or len(res) == 0 else res[-1]

def show_cef_zscore(*all):
    frm.show(-2, -1, 1, 2, lmap(get_cef_zscore, all), ta=False, log=False, title="3y z-score")

