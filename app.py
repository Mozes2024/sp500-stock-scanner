import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import requests
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="📈 S&P 500 Scanner", layout="wide")
st.title("🤖 Mozes Super-AI Scanner [DEBUG MODE]")

@st.cache_data(ttl=86400)
def load_sp500():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return df[['Symbol', 'Security', 'GICS Sector']]

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
    except:
        return None, None, None
    return None, None, None

def scan_symbol(row, min_score):
    symbol = row['Symbol']
    name = row['Security']
    sector = row['GICS Sector']
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=False)
        if df.empty or len(df) < 50:
            return None

        df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
        close = df['Close'].iloc[-1]
        score = 0
        signals = []

        if close > df['trend_ema200'].iloc[-1]:
            score += 1
            signals.append("Price > EMA200")
        if df['trend_ema_fast'].iloc[-1] > df['trend_ema_slow'].iloc[-1]:
            score += 1
            signals.append("EMA20 > EMA50")
        rsi = df['momentum_rsi'].iloc[-1]
        if 50 < rsi < 70:
            score += 1
            signals.append("RSI 50-70")
        if df['trend_macd'].iloc[-1] > 0:
            score += 1
            signals.append("MACD > 0")
        if df['trend_psar_up'].iloc[-1] < close:
            score += 1
            signals.append("PSAR Bullish")
        if df['momentum_stoch'].iloc[-1] > 50:
            score += 1
            signals.append("Stochastic > 50")
        if df['momentum_wr'].iloc[-1] > -80:
            score += 1
            signals.append("Williams %R > -80")
        if df['momentum_cci'].iloc[-1] > 0:
            score += 1
            signals.append("CCI > 0")
        if df['trend_adx'].iloc[-1] > 20:
            score += 1
            signals.append("ADX > 20")

        if score < min_score:
            return None

        marketcap = round(yf.Ticker(symbol).info.get("marketCap", 0) / 1e9, 2)
        smart, consensus, target_diff = fetch_tipranks_data(symbol)

        return {
            "Symbol": symbol,
            "Company": name,
            "Sector": sector,
            "Market Cap ($B)": marketcap,
            "Score": score,
            "Matching Signals": ", ".join(signals),
            "SmartScore": smart,
            "AnalystConsensus": consensus,
            "PriceTargetPercent": target_diff
        }
    except:
        return None

@st.cache_data
def scan_stocks_parallel(min_score, sector_filter):
    sp500 = load_sp500()
    if sector_filter != "All":
        sp500 = sp500[sp500["GICS Sector"] == sector_filter]

    results = []
debug_logs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(scan_symbol, row, min_score) for _, row in sp500.iterrows()]
        for future in futures:
            result = future.result()
            if result:
                if isinstance(result, dict):
                    results.append(result)
                elif isinstance(result, tuple):
                    results.append(result[0])
                    debug_logs.extend(result[1])
                results.append(result)

    df = pd.DataFrame(results)
    if df.empty:
        return df
    st.session_state['debug_logs'] = debug_logs
    return df.sort_values(by="Score", ascending=False)

# UI Sidebar
st.sidebar.header("פילטרים")
min_score = st.sidebar.slider("מינימום אינדיקטורים חיוביים", 0, 10, 4)
sectors = ["All"] + sorted(load_sp500()["GICS Sector"].unique().tolist())
sector_choice = st.sidebar.selectbox("בחר סקטור", sectors)

# Scan and display
with st.spinner("🔍 סורק את מניות ה-S&P500..."):
    df_result = scan_stocks_parallel(min_score, sector_choice)

st.markdown("### 📊 תוצאות סריקה:")
if not df_result.empty:
    for i, row in df_result.iterrows():
        with st.expander(f"🔹 {row['Symbol']} – {row['Company']}"):
            st.markdown(f"**סקטור:** {row['Sector']}")
            st.markdown(f"**שווי שוק:** {row['Market Cap ($B)']} מיליארד דולר")
            st.markdown(f"**ציון אינדיקטורים:** {row['Score']}")
            st.markdown(f"**התאמות:** {row['Matching Signals']}")
            st.markdown(f"**TipRanks SmartScore:** {row['SmartScore']}")
            st.markdown(f"**המלצת אנליסטים:** {row['AnalystConsensus']}")
            st.markdown(f"**פער יעד ↑:** {row['PriceTargetPercent']}%")
            st.markdown("**🔗 קישורים חיצוניים:**")
            st.markdown(f"[📈 TipRanks](https://www.tipranks.com/stocks/{row['Symbol']})")
            st.markdown(f"[📊 TradingView](https://il.tradingview.com/chart/SWXo0urZ/?symbol={row['Symbol']})")
            st.markdown(f"[💹 Investing](https://il.investing.com/search?q={row['Symbol']})")
else:
    st.warning("😕 לא נמצאו מניות שעוברות את הסינון.")

# Display skipped stocks (debug)
st.markdown("### 🐞 דיאגנוסטיקה (Debug)")
if 'debug_logs' in st.session_state:
    for log in st.session_state['debug_logs']:
        st.text(log)
