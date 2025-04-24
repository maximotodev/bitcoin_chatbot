## FILE: backend/price.py
import requests

def get_btc_price():
    try:
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
        data = response.json()
        return f"Bitcoin price is ${data['bitcoin']['usd']:,}"
    except Exception:
        return "Couldn't fetch BTC price right now."