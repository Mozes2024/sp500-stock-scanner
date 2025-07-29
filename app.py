import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np

st.title("📈 בדיקת טעינת ספריות")

# נבדוק אם אפשר לחשב אינדיקטור פשוט
try:
    df = yf.download("AAPL", period="6mo")
    rsi = ta.rsi(df['Close'])
    st.success("pandas_ta נטען בהצלחה! הנה דוגמה ל-RSI:")
    st.line_chart(rsi)
except Exception as e:
    st.error(f"שגיאה בטעינת pandas_ta: {e}")
