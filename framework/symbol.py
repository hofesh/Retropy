from framework.utils import *
from framework.RpySeries import *
import framework.meta_data_dfs as meta

def get_ticker_name(s):
    return get_name(s, ticker=True)

class Symbol(str):
    def __init__(self, fullname):
        if is_symbol(fullname):
            raise Exception("should not set symbol name as symbol instance")

        self.fullname = fullname

        parts = fullname.split("=")
        if len(parts) == 2:
            fullname = parts[0].strip()
            self.nick = parts[1].strip()
        else:
            self.nick = None
        self.fullname_nonick = fullname
        
        parts = fullname.split("!")
        if len(parts) == 2:
            fullname = parts[0]
            self.currency = parts[1]
        else:
            self.currency = ""
            
        parts = fullname.split("@")
        self.name = parts[0] #legacy
        self.ticker = parts[0]
        if len(parts) == 2:
            self.source = parts[1]
        else:
            self.source = ""

        self.diskname = self.ticker
        if self.source:
            self.diskname += "@" + self.source

    @property
    def pretty_name(self):
        if not self.nick is None:
            res = self.nick
            #return f"{self.fullname_nonick} = {self.nick}"
        else:
            res = self.fullname
        if hasattr(self, "mode") and self.mode:
            res = f"{res} {self.mode}"
        if meta.is_cef_ticker(self):
            res = f"{res}*"
        return res

    @property
    def nick_or_name(self):
        if not self.nick is None:
            return self.nick
        return self.name

    @property
    def nick_or_name_with_mode(self):
        if not self.nick is None:
            res = self.nick
        else:
            res = self.name
        if hasattr(self, "mode") and self.mode:
            res = f"{res} {self.mode}"
        return res

    @property
    def pretty_name_no_mode(self):
        if not self.nick is None:
            res = self.nick
            #return f"{self.fullname_nonick} = {self.nick}"
        else:
            res = self.fullname
        return res
        
    def __str__(self):
        return self.fullname # temp, to resolve the get, reget issue with named symbols

        if not self.nick is None:
            return self.nick
            #return f"{self.fullname_nonick} = {self.nick}"
        return self.fullname

def toSymbol(sym, source, mode, rebal):
    if isinstance(sym, dict):
        sym = dict_to_port_name(sym, use_sym_name=True)
    if is_symbol(sym):
        res = Symbol(sym.fullname)
        res.mode = mode or sym.mode
        res.rebal = rebal or sym.rebal
        return res
    if isinstance(sym, str):
        if source is None:
            res = Symbol(sym)
        else:
            res = Symbol(sym + "@" + source)
        res.mode = mode
        res.rebal = rebal
        return res
    assert False, "invalid type for Symbol: " + str(type(sym)) + ", " + str(sym)

def is_symbol(s):
    return type(s).__name__ == "Symbol"
    # isinstance(s, Symbol) - this doesn't work, as the Symbol class seems to have seperate instances in the python and Jupyter scopes

def get_pretty_name(s):
    return get_name(s, use_sym_name=False)

def get_pretty_name_no_mode(s):
    return get_name(s, use_sym_name=False, nomode=True)

def get_name(s, use_sym_name=False, nomode=False, nick_or_name=False, ticker=False):
    if s is None:
        return ""
    if is_series(s):
        s = s.name
    if not is_symbol(s):
        s = Symbol(s)
    if ticker:
        return s.ticker
    if use_sym_name:
        return s.fullname_nonick
    else:
        if nick_or_name:
            if nomode:
                return s.nick_or_name
            else:
                return s.nick_or_name_with_mode
        else:
            if nomode:
                return s.pretty_name_no_mode
            else:
                return s.pretty_name

getName = get_name

def get_mode(s):
    if is_series(s):
        s = s.name
    if is_symbol(s):
        return s.mode
    return ''

def dict_to_port_name(d, rnd=1, drop_zero=False, drop_100=False, use_sym_name=False):
    res = []
    for k, v in d.items():
        if drop_zero and v == 0:
            continue
        if drop_100 and v == 100:
            res.append(f"{getName(k, use_sym_name=use_sym_name)}")
        else:
            res.append(f"{getName(k, use_sym_name=use_sym_name)}:{round(v, rnd)}")
    return "|".join(res)

def names(all):
    return lmap(get_name, all)

def name(s, n):
    if is_series(s):
        if is_symbol(n):
            s.name = n
        elif is_symbol(s.name):
            sym = Symbol(s.name.fullname_nonick + "=" + n)
            sym.mode = s.name.mode
            sym.rebal = s.name.rebal
            s.name = sym
        else:
            s.name = n
    return s
