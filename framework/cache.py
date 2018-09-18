import os
import string
import pandas as pd

def format_filename(s):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c if c in valid_chars else '_' for c in s )
    filename = filename.replace(' ','_')
    return filename
    
 
def get_symbols_path():
    if os.path.isfile('Retropy_framework.py'):
        base = ''
    elif os.path.isfile('../Retropy_framework.py'):
        base = '../'
    elif os.path.isfile('../../Retropy_framework.py'):
        base = '../../'
    else:
        raise Exception('base path not found')
    
    return os.path.join(base, "symbols")

def cache_file(symbol, source):
    filepath = os.path.join(get_symbols_path(), source, format_filename(symbol.ticker))
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return filepath

def cache_clear():
    global symbols_mem_cache
    dirpath = "symbols"
    if not os.path.exists(dirpath):
        print(f"path not found: {dirpath}")
    import shutil
    shutil.rmtree(dirpath)        
    symbols_mem_cache = {}
    print("cache cleared")
    
def cache_get(symbol, source):
    filepath = cache_file(symbol, source)
    # if os.path.exists(filepath + '.gz'):
    #     filepath += '.gz'
    if os.path.exists(filepath):
        res = pd.read_csv(filepath, squeeze=False, index_col="date")
        res.index = pd.to_datetime(res.index, format="%Y-%m-%d")
        return res
    return None

def cache_set(symbol, source, s):
    filepath = cache_file(symbol, source)# + '.gz'
    s.to_csv(filepath, date_format="%Y-%m-%d", index_label="date")
