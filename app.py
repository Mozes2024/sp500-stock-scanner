import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import requests
import pandas_ta as ta # For technical indicators

# הגדרות עמוד
st.set_page_config(
    page_title="סורק מניות S&P 500 - מוזס AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# אתחול מצב סשן לנתונים קבועים
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = []
if 'scan_status_message' not in st.session_state:
    st.session_state.scan_status_message = "מוכן לסריקה"
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'scan_settings_hash' not in st.session_state:
    st.session_state.scan_settings_hash = None
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False

# CSS מותאם אישית לעיצוב טוב יותר
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .bullish {
        color: green;
        font-weight: bold;
    }
    .bearish {
        color: red;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .stDataFrame { /* Adjusting DataFrame display */
        font-size: 0.85em; /* Smaller font for table */
    }
    /* Hide the default Streamlit progress text inside st.status to prevent duplication */
    .stStatus > div > div > .stProgress > div > div > div:first-child {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# טעינת רשימת מניות S&P 500
@st.cache_data(ttl=86400) # שמירה במטמון ליום אחד
def load_sp500_symbols():
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        symbols = df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols]
        with open('sp500_symbols.json', 'w') as f:
            json.dump(symbols, f)
        return symbols
    except Exception as e:
        st.warning(f"אזהרה: לא ניתן היה לטעון רשימת S&P 500 מהאינטרנט. מנסה לטעון מגיבוי מקומי. שגיאה: {e}")
        try:
            with open('sp500_symbols.json', 'r') as f:
                symbols = json.load(f)
            return symbols
        except FileNotFoundError:
            st.error("שגיאה: קובץ גיבוי של סמלי S&P 500 לא נמצא. אנא וודא שיש חיבור אינטרנט או קובץ 'sp500_symbols.json'.")
            return []
        except Exception as e:
            st.error(f"שגיאה בטעינת סמלי S&P 500 מגיבוי: {e}")
            return []

SP500_SYMBOLS = load_sp500_symbols()

# פונקציה לקבלת נתוני TipRanks

def get_tipranks_data(symbol):
    url = f"https://mobile.tipranks.com/api/stocks/stockAnalysisOverview?tickers={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        if data and symbol in data:
            stock_data = data[symbol]
            analysts_score = stock_data.get("analystConsensus", {}).get("score", None)
            target_price = stock_data.get("analystConsensus", {}).get("priceTarget", None)
            current_price = stock_data.get("analystConsensus", {}).get("stockPrice", None)

            price_target_diff = (
                ((target_price - current_price) / current_price) * 100
                if target_price and current_price
                else np.nan
            )

            return {
                "Score": analysts_score,
                "PriceTarget": target_price,
                "PriceTarget %↑": price_target_diff,
            }

    except Exception as e:
        print(f"{symbol} TipRanks error: {e}")

    return {"Score": np.nan, "PriceTarget": np.nan, "PriceTarget %↑": np.nan}

