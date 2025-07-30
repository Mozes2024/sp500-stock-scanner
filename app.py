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
import pandas_ta as ta  # For technical indicators

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
@st.cache_data(ttl=86400)  # שמירה במטמון ליום אחד
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
    tipranks_data = {
        "SmartScore": np.nan,
        "AnalystConsensus": "לא זמין",
        "PriceTarget %↑": np.nan,
        "TipRanks_URL": f"https://www.tipranks.com/stocks/{symbol}"
    }
    try:
        url = f"https://mobile.tipranks.com/api/stocks/stockAnalysisOverview?tickers={symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data and isinstance(data, list) and len(data) > 0:
            stock_info = data[0]
            
            # Smart Score
            smart_score = stock_info.get('smartScore', {}).get('score', np.nan)
            if smart_score:
                tipranks_data["SmartScore"] = smart_score
            
            # Analyst Consensus and Price Target
            analyst_data = stock_info.get('analystConsensus', {})
            tipranks_data["AnalystConsensus"] = analyst_data.get('consensus', 'לא זמין')

            price_target_obj = analyst_data.get('priceTarget', {})
            current_price_from_tipranks = price_target_obj.get('price', np.nan)
            target_price = price_target_obj.get('target', np.nan)
            
            if pd.notna(current_price_from_tipranks) and pd.notna(target_price) and current_price_from_tipranks != 0:
                price_target_percentage = ((target_price - current_price_from_tipranks) / current_price_from_tipranks) * 100
                tipranks_data["PriceTarget %↑"] = price_target_percentage

    except requests.exceptions.RequestException:
        pass
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
    return tipranks_data

# פונקציה לקבלת נתונים למניה בודדת (Yfinance + Indicators + TipRanks)
def get_stock_data(symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if history.empty:
            return None
        
        # קבלת מידע בסיסי מ-yfinance
        info = ticker.info
        sector = info.get('sector', 'לא ידוע')
        industry = info.get('industry', 'לא ידוע')
        long_name = info.get('longName', symbol)
        market_cap = info.get('marketCap', np.nan)
        current_price = history['Close'].iloc[-1]
        average_volume = history['Volume'].mean()

        # חישוב אינדיקטורים טכניים באמצעות pandas_ta
        df_ta = history.copy()
        
        # RSI
        df_ta.ta.rsi(length=14, append=True)
        rsi = df_ta['RSI_14'].iloc[-1] if 'RSI_14' in df_ta.columns else np.nan

        # MACD
        df_ta.ta.macd(fast=12, slow=26, signal=9, append=True)
        macd = df_ta['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in df_ta.columns else np.nan
        signal_line = df_ta['MACDS_12_26_9'].iloc[-1] if 'MACDS_12_26_9' in df_ta.columns else np.nan
        macd_prev = df_ta['MACD_12_26_9'].iloc[-2] if len(df_ta) > 1 and 'MACD_12_26_9' in df_ta.columns else np.nan
        signal_prev = df_ta['MACDS_12_26_9'].iloc[-2] if len(df_ta) > 1 and 'MACDS_12_26_9' in df_ta.columns else np.nan

        # Moving Averages
        df_ta.ta.sma(length=50, append=True)
        df_ta.ta.sma(length=200, append=True)
        sma50 = df_ta['SMA_50'].iloc[-1] if 'SMA_50' in df_ta.columns else np.nan
        sma200 = df_ta['SMA_200'].iloc[-1] if 'SMA_200' in df_ta.columns else np.nan
        sma50_prev = df_ta['SMA_50'].iloc[-2] if len(df_ta) > 1 and 'SMA_50' in df_ta.columns else np.nan
        sma200_prev = df_ta['SMA_200'].iloc[-2] if len(df_ta) > 1 and 'SMA_200' in df_ta.columns else np.nan

        # Bollinger Bands
        df_ta.ta.bbands(length=20, std=2, append=True)
        bb_lower = df_ta['BBL_20_2.0'].iloc[-1] if 'BBL_20_2.0' in df_ta.columns else np.nan

        # חישוב שינוי באחוזים ל-20 יום
        if len(history) >= 20:
            change_20d = ((current_price - history['Close'].iloc[-20]) / history['Close'].iloc[-20]) * 100
        else:
            change_20d = np.nan
        
        # חישוב "AI Score" ו-"Matching Signals" משופרים
        ai_score = 0
        matching_signals = 0

        # 1. RSI: מכירה יתר (מתחת ל-30)
        if pd.notna(rsi) and rsi < 30:
            ai_score += 25
            matching_signals += 1

        # 2. MACD: חצייה שורית (MACD חוצה מעל Signal)
        if (pd.notna(macd) and pd.notna(signal_line) and pd.notna(macd_prev) and pd.notna(signal_prev)):
            if macd > signal_line and macd_prev <= signal_prev:
                ai_score += 25
                matching_signals += 1
            elif macd > signal_line:
                ai_score += 15
                matching_signals += 1

        # 3. Bollinger Bands: מחיר מתחת לרצועה התחתונה
        if pd.notna(current_price) and pd.notna(bb_lower) and current_price < bb_lower:
            ai_score += 25
            matching_signals += 1

        # 4. Moving Averages: חצייה שורית של MA50 מעל MA200
        if (pd.notna(sma50) and pd.notna(sma200) and pd.notna(sma50_prev) and pd.notna(sma200_prev)):
            if sma50 > sma200 and sma50_prev <= sma200_prev:
                ai_score += 25
                matching_signals += 1
            elif sma50 > sma200:
                ai_score += 15
                matching_signals += 1

        # תוספת משקל לפי נתונים נוספים
        if pd.notna(change_20d) and change_20d > 0:
            ai_score += min(change_20d, 20)

        if average_volume > 1_000_000:
            ai_score += 10

        ai_score = max(0, min(100, ai_score))

        # קבלת נתוני TipRanks
        tipranks_data = get_tipranks_data(symbol)

        return {
            "Symbol": symbol,
            "Company": long_name,
            "Sector": sector,
            "Industry": industry,
            "Market Cap": market_cap,
            "Current Price": current_price,
            "20D Change %": change_20d,
            "Average Volume": average_volume,
            "AI Score": ai_score,
            "Matching Signals": matching_signals,
            "SmartScore": tipranks_data["SmartScore"],
            "AnalystConsensus": tipranks_data["AnalystConsensus"],
            "PriceTarget %↑": tipranks_data["PriceTarget %↑"],
            "TipRanks_URL": tipranks_data["TipRanks_URL"],
            "Historical Data": history
        }
    except Exception:
        return None

# פונקציית סריקה ראשית המשתמשת ב-st.status
def run_scanner_with_status(symbols, start_date, end_date):
    st.session_state.is_scanning = True
    st.session_state.scanner_results = []
    st.session_state.failed_symbols = []
    
    total_symbols = len(symbols)
    results = []
    
    max_workers = min(15, total_symbols)
    
    with st.status("מתחיל סריקה...", expanded=True) as status_container:
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(get_stock_data, symbol, start_date, end_date): symbol for symbol in symbols}
            
            for i, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                except Exception:
                    st.session_state.failed_symbols.append(symbol)
                
                progress_percent = (i + 1) / total_symbols
                progress_bar.progress(progress_percent)
                status_text.text(f"סורק מניות... {i+1}/{total_symbols} ({int(progress_percent * 100)}%)")
        
        status_container.update(label="סריקה הושלמה!", state="complete", expanded=False)

    st.session_state.scanner_results = results
    st.session_state.last_scan_time = datetime.now()
    st.session_state.scan_status_message = "סריקה הושלמה!"
    st.session_state.is_scanning = False
    
    if st.session_state.failed_symbols:
        st.warning(f"אזהרה: לא ניתן היה לסרוק את המניות הבאות: {', '.join(st.session_state.failed_symbols)}")

# כותרת האפליקציה
st.title("🚀 סורק מניות S&P 500 מבוסס AI")
st.markdown("כלי זה סורק מניות מתוך רשימת S&P 500 ומדרג אותם לפי אותות שוריים מבוססי RSI, MACD, בולינגר בנדס וממוצעים נעים.")

# סרגל צד לסינון
st.sidebar.header("⚙️ הגדרות סריקה")

today = datetime.now().date()
default_start_date = today - timedelta(days=30)
date_range_option = st.sidebar.selectbox(
    "בחר טווח תאריכים:",
    ["חודש אחרון", "3 חודשים אחרונים", "6 חודשים אחרונים", "שנה אחרונה", "3 שנים אחרונות", "הכל (מקסימום זמין)"],
    index=0
)

if date_range_option == "חודש אחרון":
    start_date = today - timedelta(days=30)
elif date_range_option == "3 חודשים אחרונים":
    start_date = today - timedelta(days=90)
elif date_range_option == "6 חודשים אחרונים":
    start_date = today - timedelta(days=180)
elif date_range_option == "שנה אחרונה":
    start_date = today - timedelta(days=365)
elif date_range_option == "3 שנים אחרונות":
    start_date = today - timedelta(days=365 * 3)
elif date_range_option == "הכל (מקסימום זמין)":
    start_date = datetime(1990, 1, 1).date()

end_date = today

st.sidebar.write(f"**טווח נבחר:** {start_date.strftime('%Y-%m-%d')} עד {end_date.strftime('%Y-%m-%d')}")

specific_symbols_input = st.sidebar.text_area("סמלים ספציפיים (הפרד בפסיק, אופציונלי):", value="")
if specific_symbols_input:
    specific_symbols = [s.strip().upper() for s in specific_symbols_input.split(',') if s.strip()]
    symbols_to_scan = [s for s in specific_symbols if s in SP500_SYMBOLS]
    if not symbols_to_scan:
        st.sidebar.warning("אף אחד מהסמלים שהוזנו אינו ברשימת S&P 500. וודא איות נכון.")
        symbols_to_scan = []
else:
    symbols_to_scan = SP500_SYMBOLS

current_settings_hash = hashlib.md5(json.dumps({
    "start_date": start_date.strftime('%Y-%m-%d'),
    "end_date": end_date.strftime('%Y-%m-%d'),
    "symbols_to_scan": sorted(symbols_to_scan)
}).encode()).hexdigest()

should_scan_button_be_enabled = not st.session_state.is_scanning and len(symbols_to_scan) > 0
button_label = "התחל סריקה חדשה"
if st.session_state.is_scanning:
    button_label = "הסריקה פועלת..."
elif st.session_state.last_scan_time and st.session_state.scan_settings_hash == current_settings_hash:
    button_label = "רענן סריקה (נתונים קיימים)"
    
if st.button(button_label, disabled=not should_scan_button_be_enabled):
    if not st.session_state.is_scanning:
        st.session_state.scan_status_message = "מתחיל סריקה..."
        st.session_state.is_scanning = True
        st.session_state.scan_settings_hash = current_settings_hash
        run_scanner_with_status(symbols_to_scan, start_date, end_date)
        
if st.session_state.is_scanning:
    st.info(f"**סטטוס:** {st.session_state.scan_status_message}")
else:
    if st.session_state.last_scan_time:
        st.info(f"**סריקה אחרונה בוצעה ב:** {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}. **סטטוס:** {st.session_state.scan_status_message}")
        if st.session_state.last_scan_time.date() == datetime.now().date():
            st.success("✔ הנתונים מעודכנים להיום!")
        else:
            st.warning("⚠️ הנתונים אינם מעודכנים להיום. אנא בצע סריקה חדשה.")
    else:
        st.info("כדי להתחיל, בחר את ההגדרות שלך בסרגל הצד ולחץ על 'התחל סריקה חדשה'.")

# הצגת תוצאות הסריקה
if st.session_state.scanner_results:
    st.subheader("📊 תוצאות הסריקה")
    st.markdown("**הערה:** הציון AI מבוסס על אותות שוריים (עד 25 נקודות לכל אינדיקטור: RSI, MACD, Bollinger Bands, MA) עם תוספות עבור נפח ושינוי חיובי.")

    df_results = pd.DataFrame(st.session_state.scanner_results)
    
    df_results['20D Change %'] = df_results['20D Change %'].fillna(0)
    df_results['AI Score'] = df_results['AI Score'].fillna(0)
    df_results['Market Cap'] = df_results['Market Cap'].fillna(0)
    df_results['SmartScore'] = df_results['SmartScore'].fillna(np.nan)
    df_results['PriceTarget %↑'] = df_results['PriceTarget %↑'].fillna(np.nan)

    # הגדרת סדר העמודות מחוץ לבלוק התנאי
    desired_columns_order = [
        "Symbol",
        "Company",
        "Sector",
        "Market Cap",
        "AI Score",
        "Matching Signals",
        "SmartScore",
        "AnalystConsensus",
        "PriceTarget %↑",
        "Current Price",
        "20D Change %",
        "Average Volume"
    ]

    # סינון
    all_sectors_from_data = sorted(list(set([s for s in df_results['Sector'].unique() if s != 'לא ידוע'])))
    selected_sectors = st.sidebar.multiselect("סנן לפי סקטור:", all_sectors_from_data, default=all_sectors_from_data)

    min_ai_score = st.sidebar.slider("ציון AI מינימלי:", 0, 100, 50)
    min_matching_signals = st.sidebar.slider("מספר אותות תואמים מינימלי:", 0, 4, 0)
    min_change_percent = st.sidebar.slider("שינוי מינימלי ב-20 יום (%):", -50.0, 50.0, 0.0, step=0.1)
    min_volume = st.sidebar.number_input("נפח מסחר ממוצע מינימלי:", min_value=0, value=1000000)
    
    df_filtered = df_results[
        (df_results['AI Score'] >= min_ai_score) &
        (df_results['Matching Signals'] >= min_matching_signals) &
        (df_results['20D Change %'] >= min_change_percent) &
        (df_results['Average Volume'] >= min_volume)
    ]
    
    if selected_sectors:
        df_filtered = df_filtered[df_filtered['Sector'].isin(selected_sectors)]

    if not df_filtered.empty:
        df_filtered = df_filtered.sort_values(by="AI Score", ascending=False).reset_index(drop=True)

        final_columns = [col for col in desired_columns_order if col in df_filtered.columns]
        for col in df_filtered.columns:
            if col not in final_columns and col not in ["Historical Data", "TipRanks_URL", "Industry"]:
                final_columns.append(col)
        
        df_filtered_display = df_filtered[final_columns].copy()

        st.dataframe(df_filtered_display.style.format({
            "Current Price": "${:.2f}",
            "20D Change %": "{:.2f}%",
            "Average Volume": "{:,.0f}",
            "AI Score": "{:.2f}",
            "Market Cap": "${:,.0f}",
            "SmartScore": "{:.0f}",
            "PriceTarget %↑": "{:.2f}%"
        }), use_container_width=True)

        # הצגת גרף מניה נבחרת וקישורים
        st.subheader("📈 ניתוח מניה בודדת")
        
        selected_symbol_for_chart = st.selectbox(
            "בחר מניה להצגת גרף וקישורים:",
            ['בחר מניה'] + sorted(df_filtered['Symbol'].tolist())
        )

        if selected_symbol_for_chart != 'בחר מניה':
            stock_data_for_chart = df_filtered[df_filtered['Symbol'] == selected_symbol_for_chart].iloc[0]
            history_data = stock_data_for_chart["Historical Data"]
            
            st.markdown(f"#### גרף מחיר היסטורי עבור {selected_symbol_for_chart} - {stock_data_for_chart['Company']}")
            
            chart_period = st.radio(
                "טווח תצוגה לגרף:",
                ('1 חודש', '3 חודשים', '6 חודשים', 'שנה', '3 שנים', 'הכל'),
                horizontal=True
            )

            if chart_period == '1 חודש':
                chart_start_date = today - timedelta(days=30)
            elif chart_period == '3 חודשים':
                chart_start_date = today - timedelta(days=90)
            elif chart_period == '6 חודשים':
                chart_start_date = today - timedelta(days=180)
            elif chart_period == 'שנה':
                chart_start_date = today - timedelta(days=365)
            elif chart_period == '3 שנים':
                chart_start_date = today - timedelta(days=365 * 3)
            else:
                chart_start_date = datetime(1990, 1, 1).date()

            chart_history = history_data[history_data.index.date >= chart_start_date]

            if not chart_history.empty:
                fig = go.Figure(data=[go.Candlestick(
                    x=chart_history.index,
                    open=chart_history['Open'],
                    high=chart_history['High'],
                    low=chart_history['Low'],
                    close=chart_history['Close']
                )])
                fig.update_layout(title=f"גרף נרות עבור {selected_symbol_for_chart}",
                                  xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("אין מספיק נתונים היסטוריים עבור טווח הזמנים הנבחר.")

            st.markdown("#### קישורים חיצוניים")
            col_links_1, col_links_2, col_links_3 = st.columns(3)
            
            with col_links_1:
                tipranks_url = stock_data_for_chart.get("TipRanks_URL", f"https://www.tipranks.com/stocks/{selected_symbol_for_chart}")
                st.markdown(f"[🔗 TipRanks]({tipranks_url})")
            with col_links_2:
                tradingview_url = f"https://il.tradingview.com/chart/?symbol={selected_symbol_for_chart}"
                st.markdown(f"[🔗 TradingView]({tradingview_url})")
            with col_links_3:
                investing_url = f"https://il.investing.com/search?q={selected_symbol_for_chart}"
                st.markdown(f"[🔗 Investing.com]({investing_url})")

        # גרפים נוספים
        st.subheader("📈 הדמיית נתונים כללית")

        fig_scatter = px.scatter(df_filtered, x="AI Score", y="20D Change %",
                                 color="Sector", size="Average Volume",
                                 title="ציון AI מול שינוי ב-20 יום (נפח לפי גודל)",
                                 hover_data=["Symbol", "Company", "Current Price", "Average Volume", "Industry", "Matching Signals", "SmartScore", "AnalystConsensus"])
        st.plotly_chart(fig_scatter, use_container_width=True)

        sector_avg_score = df_filtered.groupby('Sector')['AI Score'].mean().sort_values(ascending=False).reset_index()
        fig_bar_sector = px.bar(sector_avg_score, x='Sector', y='AI Score',
                                title='ציון AI ממוצע לפי סקטור',
                                color='AI Score', color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_bar_sector, use_container_width=True)

        top_10_ai = df_filtered.head(10).sort_values(by='AI Score', ascending=True)
        fig_bar_top_10 = px.bar(top_10_ai, x='AI Score', y='Company', orientation='h',
                                title='10 המניות המובילות לפי ציון AI',
                                color='AI Score', color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_bar_top_10, use_container_width=True)

    else:
        st.warning(f"לא נמצאו מניות העומדות בקריטריונים הנבחרים. נסה לשנות את הסינון.")
        
        st.subheader("🔍 10 המניות המובילות (ללא סינון)")
        if not df_results.empty:
            df_top_10_all = df_results.sort_values(by="AI Score", ascending=False).head(10).reset_index(drop=True)
            final_columns_top_10 = [col for col in desired_columns_order if col in df_top_10_all.columns]
            for col in df_top_10_all.columns:
                if col not in final_columns_top_10 and col not in ["Historical Data", "TipRanks_URL", "Industry"]:
                    final_columns_top_10.append(col)

            st.dataframe(df_top_10_all[final_columns_top_10].style.format({
                "Current Price": "${:.2f}",
                "20D Change %": "{:.2f}%",
                "Average Volume": "{:,.0f}",
                "AI Score": "{:.2f}",
                "Market Cap": "${:,.0f}",
                "SmartScore": "{:.0f}",
                "PriceTarget %↑": "{:.2f}%"
            }), use_container_width=True)
        else:
            st.info("אין נתונים זמינים כדי להציג את המניות המובילות. בבקשה בצע סריקה.")

    st.subheader("💾 ייצוא וניתוח")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df_filtered_display.to_csv(index=False)
        st.download_button(
            label="📥 הורד CSV",
            data=csv,
            file_name=f"mozes_scanner_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    with col2:
        top_5 = df_filtered_display.head(5)
        top_symbols = ", ".join(top_5["Symbol"].tolist())
        st.text_area("🏆 5 הסמלים המובילים (בסינון הנוכחי)", top_symbols, height=100)

    with col3:
        if not df_filtered.empty:
            st.metric("🔥 הציון הגבוה ביותר", f"{df_filtered['AI Score'].max():.2f}")
            st.metric("📈 ממוצע שינוי 20 יום", f"{df_filtered['20D Change %'].mean():.1f}%")
        else:
            st.metric("🔥 הציון הגבוה ביותר", "אין נתונים")
            st.metric("📈 ממוצע שינוי 20 יום", "אין נתונים")
