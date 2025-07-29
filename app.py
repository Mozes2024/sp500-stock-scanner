
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import datetime
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide")
st.title("📈 Mozes Super-AI Scanner")

@st.cache_data(ttl=3600)
def load_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    return df[["Symbol", "Security", "GICS Sector"]].rename(columns={"Symbol": "Ticker", "Security": "Name", "GICS Sector": "Sector"})

@st.cache_data(ttl=3600)
def get_tipranks_data(ticker):
    try:
        url = f"https://mobile.tipranks.com/api/stocks/stockAnalysisOverview?tickers={ticker}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if ticker in data:
            return {
                "SmartScore": data[ticker].get("smartScore", None),
                "PriceTarget %": data[ticker].get("priceTarget", {}).get("percentageChange", None),
                "AnalystConsensus": data[ticker].get("analystConsensus", None),
            }
    except Exception:
        return {}
    return {}

def analyze_stock(symbol):
    try:
        df = yf.download(symbol, period='6mo', interval='1d', progress=False)
        if df.empty or len(df) < 50:
            return None

        # Ensure 1D arrays for indicators
        open_ = df["Open"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        df = ta.add_all_ta_features(df, open=open_, high=high, low=low, close=close, volume=volume, fillna=True)

        score = 0
        signals = []

        if df["trend_macd_diff"].iloc[-1] > 0:
            score += 1
            signals.append("MACD Bullish")
        if df["trend_ema_fast"].iloc[-1] > df["trend_ema_slow"].iloc[-1]:
            score += 1
            signals.append("EMA Crossover")
        if df["momentum_rsi"].iloc[-1] < 70 and df["momentum_rsi"].iloc[-1] > 50:
            score += 1
            signals.append("RSI Moderate")
        if df["momentum_stoch"].iloc[-1] < 80 and df["momentum_stoch"].iloc[-1] > 20:
            score += 1
            signals.append("Stoch OK")
        if df["volatility_bbm"].iloc[-1] < df["Close"].iloc[-1] < df["volatility_bbu"].iloc[-1]:
            score += 1
            signals.append("Inside Bollinger")

        return {
            "Symbol": symbol,
            "Score": score,
            "Signals": ", ".join(signals)
        }
    except Exception as e:
        return {"Symbol": symbol, "Error": f"{type(e).__name__} - {e}"}

@st.cache_data(ttl=1800)
def scan_stocks(min_score=3, sector_filter=None):
    sp500 = load_sp500()
    if sector_filter != "All":
        sp500 = sp500[sp500["Sector"] == sector_filter]

    results = []
    debug_info = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock, row["Ticker"]): row for _, row in sp500.iterrows()}
        for future in futures:
            result = future.result()
            if result:
                if "Error" in result:
                    debug_info.append(f"{result['Symbol']} skipped - exception: {result['Error']}")
                elif result["Score"] >= min_score:
                    meta = futures[future]
                    tip_data = get_tipranks_data(result["Symbol"]) or {}
                    results.append({
                        "Symbol": result["Symbol"],
                        "Name": meta["Name"],
                        "Sector": meta["Sector"],
                        "Score": result["Score"],
                        "Signals": result["Signals"],
                        **tip_data
                    })

    return pd.DataFrame(results).sort_values(by="Score", ascending=False), debug_info

# Sidebar controls
min_score = st.sidebar.slider("Minimum Score", 0, 5, 3)
sectors = ["All"] + sorted(load_sp500()["Sector"].unique())
sector_choice = st.sidebar.selectbox("Filter by Sector", sectors)

df_result, debug_output = scan_stocks(min_score=min_score, sector_filter=sector_choice)

if not df_result.empty:
    def make_link(symbol):
        tip = f"https://www.tipranks.com/stocks/{symbol}"
        tv = f"https://il.tradingview.com/chart/?symbol={symbol}"
        inv = f"https://il.investing.com/search?q={symbol}"
        return f"[{symbol}]({tip}) | [TV]({tv}) | [INV]({inv})"

    df_result["Links"] = df_result["Symbol"].apply(make_link)
    df_result_display = df_result[["Links", "Name", "Sector", "Score", "Signals", "SmartScore", "PriceTarget %", "AnalystConsensus"]]
    st.dataframe(df_result_display, use_container_width=True)
else:
    st.warning("No stocks matched your filters.")

with st.expander("🐞 דיאגנוסטיקה (Debug)"):
    st.text("
".join(debug_output))
