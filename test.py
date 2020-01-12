import robin_stocks as r
print(r.crypto.get_crypto_info('BTC'))
print(r.crypto.get_crypto_currency_pairs())
print(r.stocks.get_historicals("APPL", span='week', bounds='regular'))