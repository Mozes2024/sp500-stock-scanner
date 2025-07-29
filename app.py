import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import requests

st.set_page_config(page_title="📈 S&P 500 Scanner", layout="wide")
st.title("📈 סורק מניות S&P 500 עם אינדיקטורים ו-TipRanks")

@st.cache_data
def load_sp500():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return df[['Symbol', 'Security', 'GICS Sector']]

sp500_df = load_sp500()

st.sidebar.header("פילטרים")
min_score = st.sidebar.slider("מינימום אינדיקטורים חיוביים", 0, 10, 4)

st.write("### מניות מתוך S&P500:")
st.dataframe(sp500_df.head())
st.warning("הסריקה המלאה תתווסף בהמשך – זהו דמו ראשוני")

