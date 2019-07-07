
import os
import string

import framework.cache.s3_cache as s3_cache
import framework.cache.file_cache as file_cache

def _format_filename(s):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c if c in valid_chars else '_' for c in s )
    filename = filename.replace(' ','_')
    return filename

def cache_file(symbol, source):
  return os.path.join(source, _format_filename(symbol.ticker))

def cache_get(symbol, source):
  local_file_symbol_body = file_cache.cache_get(symbol, source)
  s3_symbol_body = s3_cache.cache_get(symbol, source)

  if s3_symbol_body is None and local_file_symbol_body is not None:
    print("Writing cache from local file to S3")
    s3_cache.cache_set(symbol, source, local_file_symbol_body)
    return local_file_symbol_body
  elif local_file_symbol_body is None and s3_symbol_body is not None:
    print("Writing cache from S3 to local file")
    file_cache.cache_set(symbol, source, s3_symbol_body)
    return s3_symbol_body
  elif s3_symbol_body is not None:
    print("Got cache from S3")
    return s3_symbol_body
  elif local_file_symbol_body is not None:
    print("Got cache from local file")
    return local_file_symbol_body

  return None

def cache_set(symbol, source, s):
  print("Storing cache")
  file_cache.cache_set(symbol, source, s)
  s3_cache.cache_set(symbol, source, s)
