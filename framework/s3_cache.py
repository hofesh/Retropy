
import boto3
import os
import pandas as pd
import string

from io import StringIO
from botocore.exceptions import ClientError

BUCKET_NAME = "retropy-symbols"

s3_client = boto3.client("s3")

def format_filename(s):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c if c in valid_chars else '_' for c in s )
    filename = filename.replace(' ','_')
    return filename

def cache_file(symbol, source):
  return os.path.join(source, format_filename(symbol.ticker))

def cache_get(symbol, source):
    file_path = cache_file(symbol, source)

    try:
      symbol_body = StringIO(s3_client.get_object(
        Bucket=BUCKET_NAME,
        Key=file_path,
      ).get("Body").read().decode('utf-8'))
    except ClientError as error:
      if error.response["Error"]["Code"] == "NoSuchKey":
          print("File % not found", file_path)
          return None

    if symbol_body:
        res = pd.read_csv(symbol_body, squeeze=False, index_col="date")
        res.index = pd.to_datetime(res.index, format="%Y-%m-%d")
        return res

    return None

def cache_set(symbol, source, s):
    file_path = cache_file(symbol, source)
    s3_client.put_object(
      Bucket=BUCKET_NAME,
      Key=file_path,
      Body=s.to_csv(date_format="%Y-%m-%d", index_label="date")
    )
