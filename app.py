import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

st.set_page_config(page_title="Mozes Super-AI Scanner", layout="wide")
st.title("🚀 Mozes Super-AI Scanner")

# Load S&P 500 tickers
@st.cache_data
def load_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return table[0]["Symbol"].tolist(), table[0][["Symbol", "GICS Sector"]]

tickers, ticker_info = load_sp500_tickers()

# Sidebar filters
min_score = st.sidebar.slider("Minimum Technical Score", 0, 10, 4)
sectors = ["All"] + sorted(ticker_info["GICS Sector"].unique().tolist())
selected_sector = st.sidebar.selectbox("Filter by Sector", sectors)

# Filter tickers by sector
if selected_sector != "All":
    filtered = ticker_info[ticker_info["GICS Sector"] == selected_sector]
    tickers = filtered["Symbol"].tolist()
else:
    filtered = ticker_info

results = []

progress = st.progress(0)
for i, symbol in enumerate(tickers):
    try:
        data = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if data.empty:
            continue
        close = data["Close"]
        df = pd.DataFrame()
        df["RSI"] = ta.rsi(close)
        df["MACD"] = ta.macd(close)["MACD_12_26_9"]
        df["SMA50"] = ta.sma(close, length=50)
        df["SMA200"] = ta.sma(close, length=200)
        score = 0
        if df["RSI"].iloc[-1] > 50:
            score += 1
        if df["MACD"].iloc[-1] > 0:
            score += 1
        if close.iloc[-1] > df["SMA50"].iloc[-1]:
            score += 1
        if close.iloc[-1] > df["SMA200"].iloc[-1]:
            score += 1
        results.append({
            "Symbol": symbol,
            "Sector": filtered[filtered["Symbol"] == symbol]["GICS Sector"].values[0],
            "Score": score,
            "Close": round(close.iloc[-1], 2),
            "Links": f"[TipRanks](https://www.tipranks.com/stocks/{symbol.lower()}) | "
                     f"[TradingView](https://il.tradingview.com/chart/SWXo0urZ/?symbol={symbol}) | "
                     f"[Investing](https://il.investing.com/search?q={symbol})"
        })
    except Exception:
        continue
    progress.progress((i + 1) / len(tickers))

df_result = pd.DataFrame(results)
st.dataframe(df_result.sort_values("Score", ascending=False), use_container_width=True)