import numbers
import functools as ft
import types
import sys

trimmed_messages = set()
def print_norep(msg, **args):
    if not msg in trimmed_messages:
        trimmed_messages.add(msg)
        print(msg, **args)

def warn(*arg, **args):
    set_if_none(args, "file", sys.stderr)
    print_norep(*arg, **args)

def get_func_name(f):
    if hasattr(f, "__name__"):
        return f.__name__
    return "<unnamed function>" 

def lmap(f, l):
    return list(map(f, l))

def lfilter(f, l):
    return list(filter(f, l))

def flatten(lists):
    return [x for lst in lists for x in lst]

def flattenLists(items):
    res = []
    for x in items:
        if isinstance(x, list):
            res += x
        elif isinstance(x, range) or isinstance(x, map) or isinstance(x, types.GeneratorType):
            res += list(x)
        else:
            res.append(x)
    return res

def fixtext(s):
    if isinstance(s, str):
        return bidialg.get_display(s)
    return [fixtext(x) for x in s]
    

def set_if_none(d, key, value):
    v = d.get(key, None)
    if v is None:
        d[key] = value
    return d
        
def list_rm(l, *items):
    l = l.copy()
    for x in items:
        if x in l:
            l.remove(x)
    return l    


def compose(*functions):
    return ft.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
reduce = compose

def partial(func, *args, **kwargs):
    partial_func = ft.partial(func, *args, **kwargs)
    ft.update_wrapper(partial_func, func)
    return partial_func

def is_number(v):
    return isinstance(v, numbers.Real)

def as_int(s):
    try:
        return int(s)
    except:
        return None

def drop_duplicates_index(s):
    return s[~s.index.duplicated()]