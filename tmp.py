from Retropy_framework import *


from pathlib import Path, PosixPath
def iter_cached_symbols(skip_fails=True):
    path = get_symbols_path()
    for file in Path(path).glob('**/*'):
        if not file.is_file():
            continue
        if str(file).endswith("_FAIL_") and skip_fails:
            continue
        symname = file.name.replace("._FAIL_", "").replace(".gz", "")
        source = file.parts[-2]
        yield (symname, source)

def get_all_cached_symbols():
    for sym, source in iter_cached_symbols():
        get(sym, source=source, error='ignore')

#list(iter_cached_symbols())
get_all_cached_symbols()