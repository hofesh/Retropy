if not "fixed_globals_once" in globals():

    # ************* SYMBOLS ***************
    # these are shorthand variables representing asset classes

    # ==== SPECIAL ====
    # https://www.federalreserve.gov/pubs/bulletin/2005/winter05_index.pdf
    # Nominal Daily
    usdMajor = 'FRED/DTWEXM@Q' # Trade Weighted U.S. Dollar Index: Major Currencies
    usdBroad = 'FRED/DTWEXB@Q' # Trade Weighted U.S. Dollar Index: Broad
    usdOther = 'FRED/DTWEXO@Q' # Trade Weighted U.S. Dollar Index: Other Important Trading Partners
    # Nominal Monthly
    usdMajorM = 'FRED/TWEXMMTH@Q'
    usdBroadM = 'FRED/TWEXBMTH@Q'
    usdOtherM = 'FRED/TWEXOMTH@Q'
    # Real Monthly
    usdMajorReal = 'FRED/TWEXMPA@Q' # Real Trade Weighted U.S. Dollar Index: Major Currencies
    usdBroadReal = 'FRED/TWEXBPA@Q' # Real Trade Weighted U.S. Dollar Index: Broad
    usdOtherReal = 'FRED/TWEXOPA@Q' # Real Trade Weighted U.S. Dollar Index: Other Important Trading Partners
    usd = usdBroad

    cpiUS ='RATEINF/CPI_USA@Q'


    #bitcoinAvg = price("BAVERAGE/USD@Q") # data 2010-2016
    #bitcoinBitstamp = price("BCHARTS/BITSTAMPUSD@Q") # data 2011-now

    # ==== STOCKS ====
    # Global
    g_ac = 'VTSMX:45|VGTSX:55' # VTWSX, VT # global all-cap
    d_ac = 'URTH' # developed world
    # US
    ac = 'VTSMX' # VTI # all-cap
    lc = 'VFINX' # VOO, SPY # large-cap
    mc = 'VIMSX' # VO # mid-cap
    sc = 'NAESX' # VB # small-cap
    mcc = 'BRSIX' # micro-cap
    lcv = 'VIVAX' # IUSV # large-cap-value
    mcv = 'VMVIX' # mid-cap-value
    scv = 'VISVX' # VBR # small-cap-value
    lcg = 'VIGRX' # large-cap-growth 
    mcg = 'VMGIX' # mid-cap-growth
    scg = 'VISGX' # VBK # small-cap-growth
    # ex-US
    i_ac = 'VGTSX' # VXUS # intl' all-cap
    i_sc = 'VINEX' # VSS, SCZ # intl' small-cap
    i_dev = 'VTMGX' # EFA, VEA # intl' developed
    i_acv = 'DFIVX' # EFV # intl' all-cap-value
    i_scv = 'DISVX' # DLS # intl' small-cap-value
    em_ac = 'VEIEX' # VWO # emerging markets
#    em = em_ac # legacy
    em_sc = 'EEMS' # emerging markets small cap
    fr_ac = 'FRN' # FM # frontier markets

    # ==== BONDS ====
    # US GOVT
    sgb = 'VFISX' # SHY, VGSH # short term govt bonds
    tips = 'VIPSX' # TIP # inflation protected treasuries
    lgb = 'VUSTX' # TLT, VGLT # long govt bonds
    elgb = 'PEDIX@Y' # EDV # extra-long (extended duration) govt bonds, note PEDIX is missing it's divs in AV
    gb = 'VFITX' # IEI # intermediate govt bonds
    fgb = 'TFLO' # floating govt bonds
    # US CORP 
    cb = 'MFBFX' # LQD # corp bonds
    scb = 'VCSH' # short-term-corp-bonds
    lcb = 'VCLT' # long-term-corp-bonds
    fcb = 'FLOT' # floating corp bonds
    # US CORP+GOVT
    gcb = 'VBMFX' # AGG, BND # govt/corp bonds
    sgcb = 'VFSTX' # BSV # short-term-govt-corp-bonds
    # International
    i_tips = 'WIP' # # intl' local currency inflation protected bonds
    i_gcbUsd = 'PFORX' # BNDX # ex-US govt/copr bonds (USD hedged)
    i_gbLcl = 'BEGBX' # (getBwx()) BWX, IGOV # ex-US govt bonds (non hedged)
#    i_gb = i_gbLcl # legacy
    i_cb = 'PIGLX' # PICB, ex-US corp bonds
    i_cjb = 'IHY' # intl-corp-junk-bonds
    g_gcbLcl = 'PIGLX' # Global bonds (non hedged)
    g_gcbUsd = 'PGBIX' # Global bonds (USD hedged)
    g_sgcb = 'LDUR' # Global short-term govt-corp bonds
    g_usgcb = 'MINT' # Global ultra-short-term govt-corp bonds
    em_gbUsd = 'FNMIX' # VWOB, EMB # emerging market govt bonds (USD hedged)
#    emb = em_gbUsd # legacy
    em_gbLcl = 'PELBX' # LEMB, EBND, EMLC emerging-markets-govt-bonds (local currency) [LEMB Yahoo data is broken]
    em_cjb = 'EMHY' # emerging-markets-corp-junk-bonds
    cjb = 'VWEHX' # JNK, HYG # junk bonds
#    junk = cjb # legacy
    scjb = 'HYS' # short-term-corp-junk-bonds
    aggg_idx = "LEGATRUU;IND@B" # AGGG.L Global bonds unhedged (TR - Total Return)

    # ==== CASH ====
    rfr = 'SHV' # BIL # risk free return (1-3 month t-bills)
    cash = rfr # SHV # risk free return
    cashLike = 'VFISX:30' # a poor approximation for rfr returns 

    # ==== OTHER ====
    fedRate = 'FRED/DFF@Q'
    reit = 'DFREX' # VNQ # REIT
    i_reit = 'RWX' # VNQI # ex-US REIT
    g_reit = 'DFREX:50|RWX:50' # RWO # global REIT
    gold = 'LBMA/GOLD@Q' # GLD # gold
    silver = 'LBMA/SILVER@Q' # SLV # silver
    palladium = 'LPPM/PALL@Q'
    platinum = 'LPPM/PLAT@Q'
    #metals = gold|silver|palladium|platinum # GLTR # precious metals (VGPMX is a stocks fund)
    comm = 'DBC' # # commodities
    oilWtiQ = 'FRED/DCOILWTICO@Q'
    oilBrentQ = 'FRED/DCOILBRENTEU@Q'
    oilBrentK = 'oil-prices@OKFN' # only loads first series which is brent
    eden = 'EdenAlpha@MAN'

    # ==== INDICES ====
    spxPR = '^GSPC'
    spxTR = '^SP500TR'
    spx = spxPR
    

    # ==== TASE ====
    # exactly the same data as from TASE, but less indices supported
    ta125_IC = 'TA125@IC'
    ta35_IC = 'TA35@IC'

    # https://www.tase.co.il/he/market_data/indices
    ta35 = "142@TASE = TA-35"
    ta125 = "137@TASE = TA-125"
    ta90 = "143@TASE = TA-90"
    taSME60 = "147@TASE = TA-SME60"
    telDiv = "166@TASE = TA-Div"
    telAllShare = "168@TASE = TA-AllShare"
    ta_stocks = [ta35, ta125, ta90, taSME60, telDiv, telAllShare]

    taBonds = "601@TASE = IL-Bonds"
    taGovtBonds = "602@TASE = TA-GovtBonds"
    taCorpBonds = "603@TASE = TA-CorpBonds"
    taTips = "604@TASE = TA-Tips"
    taGovtTips = "605@TASE = TA-GovtTips"
    taCorpTips = "606@TASE = TA-CorpTips"
    ta_bonds = [taBonds, taGovtBonds, taCorpBonds, taTips, taGovtTips, taCorpTips]

    telCorpBond60ILS = "720@TASE = TA-CorpBond60ILS"
    telCorpBondUsd = "739@TASE = TA-CorpBondUsd"
    telCorpBond20 = "707@TASE = TA-CorpBond20"
    telCorpBond40 = "708@TASE = TA-CorpBond40"
    telCorpBond60 = "709@TASE = TA-CorpBond60"
    ta_corpBonds = [telCorpBond20, telCorpBond40, telCorpBond60, telCorpBond60ILS, telCorpBondUsd]

    taMakam = "800@TASE = TA-Makam"
    ta_makam = [taMakam]

    ta_all = ta_stocks + ta_bonds + ta_corpBonds + ta_makam    
    # ==== TASE END ====

    # Corp bonds Yield indeces
    # https://www.quandl.com/data/ML-Merrill-Lynch

    ml_cb_aaa = 'ML/AAAEY@Q=cb_aaa'
    ml_cb_aa = 'ML/AAY@Q=cb_aa'
    ml_cb_a = 'ML/AEY@Q=cb_a'
    ml_cb_ALL = 'ML/USEY@Q=cb_ALL'
    ml_cb_bbb = 'ML/BBBEY@Q=cb_bbb'
    ml_cb_bb = 'ML/BBY@Q=cb_bb'
    ml_cb_hy = 'ML/USTRI@Q=cb_hy'
    ml_cb_ccc = 'ML/CCCY@Q=cb_ccc'
    ml_em_hg = 'ML/EMHGY@Q=em_hg'
    ml_em_hy = 'ML/EMHYY@Q=em_hy'
    
    
    
    glb = globals().copy()
    for k in glb.keys():
        if k.startswith("_"):
            continue
        val = glb[k]
        if not isinstance(val, str):
            continue
        if "\n" in val:
            continue
        if k.isupper():
            continue
        if "=" in val:
            continue
        globals()[k] = f"{val} = {k}"
    
    fixed_globals_once = True

all_assets = [
# ==== STOCKS ====
# Global
d_ac,
# US
ac,
lc,
mc,
sc,
mcc,
lcv,
mcv,
scv,
lcg,
mcg,
scg,
# ex-US
i_ac,
i_sc,
i_dev,
i_acv,
i_scv,
em_ac,
em_sc,
fr_ac,

# ==== BONDS ====
# US GOVT
sgb,
tips,
lgb,
elgb,
gb,
fgb,
# US CORP 
cb,
scb,
lcb,
fcb,
# US CORP+GOVT
gcb,
sgcb,
# International
i_tips,
i_gcbUsd,
i_gbLcl,
i_cb,
i_cjb,
g_gcbLcl,
g_gcbUsd,
g_sgcb,
g_usgcb,
em_gbUsd,
em_gbLcl,
em_cjb,
cjb,
scjb,

# ==== CASH ====
rfr,

# ==== OTHER ====
#fedRate,
reit,
i_reit,
gold,
silver,
palladium,
platinum,
#metals,
comm,
oilWtiQ,
oilBrentQ,
]

assets_core = [
    # equities
    lc,
    i_ac,
    i_dev,
    em_ac,
    # reit
    reit,
    i_reit,
    # bonds
    gb,
    lgb,
    cb,
    i_cb,
    em_gbUsd,
    tips,
    # commodities
    gold,
    comm,
    # cash
    cash
]

# https://www.federalreserve.gov/pubs/bulletin/2005/winter05_index.pdf
usdMajorCurrencies = ["USDEUR", "USDCAD", "USDJPY", "USDGBP", "USDCHF", "USDAUD", "USDSEK"]
usdOtherCurrencies = ["USDMXN", "USDCNY", "USDTWD", "USDKRW", "USDSGD", "USDHKD", "USDMYR", "USDBRL", "USDTHB", "USDINR"] # "USDPHP"
usdBroadCurrencies = usdMajorCurrencies + usdOtherCurrencies

interestingCurrencies = ["USDEUR", "USDCAD", "USDJPY", "USDAUD", "USDJPY", "USDCNY"]

