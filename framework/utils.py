import types
import sys

trimmed_messages = set()
def print_norep(msg):
    if not msg in trimmed_messages:
        trimmed_messages.add(msg)
        print(msg)

def warn(*arg, **args):
    print(*arg, file=sys.stderr, **args)

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
