import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import requests
from concurrent.futures import ThreadPoolExecutor

@st.cache_data(ttl=3600)
def fetch_tipranks_data(ticker):
    try:
        url = f"https://mobile.tipranks.com/api/stocks/stockAnalysisOverview?tickers={ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()[0]
            price = data.get("price", 0)
            target = data.get("priceTarget", 0)
            diff = round(((target - price) / price) * 100, 2) if price else None
            return data.get("smartScore"), data.get("analystConsensus"), diff
        return None, None, None
    except Exception as e:
        return None, None, None

st.set_page_config(page_title="📈 S&P 500 Scanner", layout="wide")
st.title("🤖 Mozes Super-AI Scanner [DEBUG MODE]")
@st.cache_data(ttl=86400)
def load_sp500():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return df[['Symbol', 'Security', 'GICS Sector']]
@st.cache_data(ttl=3600)
