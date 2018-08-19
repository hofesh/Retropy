class BloombergDataSource(DataSource):
    def fetch(self, symbol, conf):
        headers = {
                    'Origin': 'https://www.bloomberg.com',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36',
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Referer': 'https://www.bloomberg.com',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Connection': 'keep-alive'    
                }
        url = "https://www.bloomberg.com/markets/api/bulk-time-series/price/__sym__?timeFrame=5_YEAR"
        sym = symbol.name.replace(";", ":")
        text = requests.get(url.replace("__sym__", sym), headers=headers).text
        d = json.loads(text)
        #print(d)
        df = pd.DataFrame(d[0]["price"])
        if len(df) == 0:
            return None
        df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d")
        df.set_index("date", inplace=True)
        return df

    def process(self, symbol, df, conf):
        return df.value
