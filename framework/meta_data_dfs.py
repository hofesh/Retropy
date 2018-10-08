import os.path
import pandas as pd
from framework.utils import *

def load_etf_metadata(fname="../data/etfs.msgpack"):
    global etf_metadata_df
    etf_metadata_df = None
    if not os.path.isfile(fname):
        fname = "data/etfs.msgpack"
        if not os.path.isfile(fname):
            warn("etfs.msgpack not found")
            return
    etf_metadata_df = pd.read_msgpack(fname)
    etf_metadata_df.index = etf_metadata_df.index.str.strip()
    etf_metadata_df = etf_metadata_df[~etf_metadata_df.index.duplicated(keep='first')]


def load_cef_metadata(fname="../data/cefs.msgpack"):
    global cef_metadata_df
    if not os.path.isfile(fname):
        fname = "data/cefs.msgpack"
        if not os.path.isfile(fname):
            warn("cefs.msgpack not found")
            return
    cef_metadata_df = pd.read_msgpack(fname)
    cef_metadata_df.index = cef_metadata_df.index.str.strip()
    cef_metadata_df = cef_metadata_df[~cef_metadata_df.index.duplicated(keep='first')]

def is_etf_ticker(s):
    global etf_metadata_df
    return s in etf_metadata_df.index

def is_cef_ticker(s):
    global cef_metadata_df
    return s in cef_metadata_df.index


#############################
load_etf_metadata()
load_cef_metadata()
#############################
