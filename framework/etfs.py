import framework.meta_data_dfs as dfs
import pandas as pd

# These are mutually exclusive from the ETFs meta_data set
bonds_1B_AUM = 'AGG|LQD|BND|TIP|BSV|VCSH|VCIT|PFF|HYG|BIV|JNK|EMB|MBB|CSJ|SHY|BNDX|MINT|BKLN|SHV|IEF|CIU|TLT|IEI|FLOT|PGX|GOVT|PCY|SCHZ|VMBS|VTIP|SJNK|EMLC|TOTL|SHYG|SPSB|FPE|NEAR|SCHP|SPIB|VCLT|BLV|SRLN|SCHO|BOND|TDTT|GVI|IUSB|VRP|VGSH|BIL|FLRN|PGF|VGIT|HYS|CRED|BWX|FTSM|STIP|SPAB|FTSL|HYLS|ISTB|ANGL|SCHR|PHB|BSCK|GSY|BSJI|BSCJ|IPE|STPZ'
bonds_1B_AUM = bonds_1B_AUM.split('|')

bonds_100M_AUM = 'VWOB|BSCI|BAB|LMBS|BSJJ|GBIL|TDTF|IGOV|CORP|BSCL|CLY|SPTL|ITE|BSCM|SLQD|BSJK|EMHY|EDV|IAGG|VGLT|PSK|SNLN|VBND|PFXF|BSCH|IBDM|WIP|TLH|AGZ|IBDL|CLTL|IBDK|HYLB|IGHG|EBND|IBDN|HYEM|AGGP|IBDH|FBND|RIGS|BSJH|LEMB|BSJL|BSCN|AGGY|IBDO|ILTB|SPLB|IBDQ|CMBS|LTPZ|FLTR|MBG|BWZ|PGHY|SPFF|BSCO|SPTS|GHYG|FIXD|PLW|VNLA|MINC|ELD|IBND|ZROZ|USHY|IBDP|BSJM|IHY|TIPX|NFLT|PICB|GIGB|VRIG|HYLD|FFTI|LDUR|FTSD|HYHG|ICSH|FIBR|RAVI|GNMA|IBDJ|JPGB|HYGH|QLTA|GBF|DWFI|FLTB|JPST|BLHY|IBDC|HYLV'
bonds_100M_AUM = bonds_100M_AUM.split('|')

bonds_tax_pref_100M_AUM = 'MUB|SHM|TFI|HYD|VTEB|ITM|PZA|SUB|CMF|HYMB|NYF|MUNI|FMB|PWZ|SMB|IBMI|IBMH|IBMG|MLN|IBMK|IBMJ|SHYD|XMPT'
bonds_tax_pref_100M_AUM = bonds_tax_pref_100M_AUM.split('|')

alts_100M_AUM = 'TQQQ|SSO|TBT|FAS|QLD|SH|NUGT|UPRO|QAI|SVXY|SDS|VXX|UYG|JNUG|SPXL|TNA|SOXL|UUP|TBF|TECL|SPXU|UCO|TZA|SQQQ|UDOW|DDM|ERX|BIB|LABU|MORL|UVXY|TMV|MNA|PBP|PSQ|SPXS|DUST|QID|FXE|PUTW|ROM|USLV|EDC|RWM|CEFL|BDCL|DOG|YINN|SCO|AGQ|TVIX|FXB|ZIV|UWM|EUO|BRZU|DWT|QYLD|UWT|WTMF|DXD|FXC|URE|JPHF|FAZ|HDGE|MVV|CURE|URTY|FXF|SDOW|EUM|VIXY|RUSL|FTLS|DGAZ|FXA|YCS|DIG|TWM|CCOR|PST|GUSH|CHAD|JDST|SJB|RXL|DGP|USDU|FXY|UGLD|HTUS|DYLS|CWEB'
alts_100M_AUM = alts_100M_AUM.split('|')

stocks_10B_AUM = 'SPY|IVV|VTI|VOO|EFA|VEA|VWO|QQQ|IWM|IJH|IEFA|IEMG|IWD|IWF|EEM|VTV|IJR|VNQ|XLF|VUG|VIG|VEU|DIA|VO|VB|VYM|IWB|MDY|IVW|XLK|EWJ|VGK|DVY|VGT|XLV|XLE|IWR|SDY|EZU|IVE|USMV|RSP|SCHF|XLY|VBR|XLI|ITOT|VV|SCHB|SCHX|IWS|VT|VXUS|SCZ|IBB'
stocks_10B_AUM = stocks_10B_AUM.split('|')

stocks_1B_AUM = 'AMLP|XLP|IWN|DXJ|IWO|ACWI|IWP|IWV|VOE|IXUS|EFAV|HEDJ|GDX|IJK|SPLV|XLU|EWZ|VFH|VHT|VBK|SCHD|DBEF|HDV|SCHA|EFV|IJJ|VNQI|VXF|FDN|MTUM|INDA|PRF|VPL|IJS|VOT|SCHG|OEF|IJT|GUNR|XLB|ITA|IDV|KRE|VSS|EWG|AAXJ|HEFA|EEMV|IYR|SCHE|QUAL|GDXJ|FEZ|FVD|IYW|SCHM|EWY|SCHV|XBI|SCHH|VDE|VDC|KBE|RWX|FNDX|ACWV|FXI|EWT|IUSG|AMJ|VIS|NOBL|EFG|MGK|FNDF|EPP|VLUE|IUSV|DON|IEV|ACWX|EWC|SPHD|ICF|RWO|RWR|GSLC|IEUR|VPU|FNDA|MCHI|DGRO|XOP|EWU|FV|ITB|VCR|XLRE|SDOG|RPG|QTEC|VAW|DBEU|EMLP|IGF|EUFN|DLN|DES|DEM|IYH|EWH|IYF|DBJP|QDF|DGRW|HEZU|MGV|MLPI|ROBO|VOOG|PRFZ|IOO|FNDE|SPDW|DLS|RSX|EPI|FLGE|FDL|EWA|SCHC|FXR|GEM|FXL|FNDC|IXJ|IYG|SOXX|CWI|RYT|PDP|XT|VONG|DGS|ONEQ|PWV|IHI|FEX|BOTZ|OIH|ILF|IXN|IXN|FBGX|SKYY|FTEC|MGC|SPHQ|PKW|MOAT|GNR|VONV|DWX|IGM|SMH|JPIN|XMLV|EWL|SLYG|VTWO|DHS|FTXO|KWEB|PXF|TILT|FBT|INDY|IYJ|SPYG|XSLV|FXO|DFE|XHB|REM|HACK|MDYG|FNCL|IYY|HEWJ|IGV|FTA|FIHD|LIT|EWW|VOX|SLYV|IYE|GXC|NANR|XAR|TLTD|KBWB|FXH|IQDF|IQDF|IXC|EWP|EZM'
stocks_1B_AUM = stocks_1B_AUM.split('|')

all_manual_bonds = bonds_1B_AUM + bonds_100M_AUM + bonds_tax_pref_100M_AUM
all_manual_bonds_alts = all_manual_bonds + alts_100M_AUM
all_manual_stocks = stocks_10B_AUM + stocks_1B_AUM
all_manual = all_manual_stocks + all_manual_bonds_alts

if not dfs.etf_metadata_df is None:
    _df = dfs.etf_metadata_df.query('aum > 100 or mw_aum > 100').sort_values("aum", ascending=False)
    all = list(_df.index)
    
    all_bonds = list(_df.query("yc_category == 'FixedIncome'").index)
    all_stocks = list(_df.query("yc_category == 'Equity'").index)
    all_alts = list(_df.query("yc_category == 'Alternative'").index)
    all_tax_preferred = list(_df.query("yc_category == 'TaxPreferred'").index)

    all_bonds_and_tax_preferred = all_bonds + all_tax_preferred
    all_bonds_alts = all_bonds + all_alts

    all_no_category = list(_df[pd.isnull(_df["yc_category"])].index)
    all_no_aum = list(dfs.etf_metadata_df[(pd.isnull(dfs.etf_metadata_df["aum"])) & (pd.isnull(dfs.etf_metadata_df["mw_aum"]))].index)
#    all_no_aum = list(dfs.etf_metadata_df[(pd.isnull(dfs.etf_metadata_df["aum"]))].index)
    all_no_aum = [s for s in all_no_aum if not ':' in s]

    all_govt_long = list(_df.query("yc_sub_category == 'LongGovernment'").index)
    all_ultra_short = list(_df.query("yc_sub_category == 'UltrashortBond'").index)

pimco_etf = 'MINT|BOND|HYS|STPZ|CORP|MFEM|MUNI|LTPZ|LDUR|ZROZ|TUZ|MFUS|SMMU|TIPZ|MFDX'
pimco_etf = pimco_etf.split('|')

