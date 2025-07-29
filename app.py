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

# Page settings
st.set_page_config(
    page_title="Mozes Super-AI S&P 500 Scanner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for persistent data
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = []
if 'scan_status_message' not in st.session_state:
    st.session_state.scan_status_message = "Ready to scan"
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'scan_settings_hash' not in st.session_state:
    st.session_state.scan_settings_hash = None
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False

# Custom CSS for better styling
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
        color: #28a745; /* Green */
        font-weight: bold;
    }
    .bearish {
        color: #dc3545; /* Red */
        font-weight: bold;
    }
    .neutral {
        color: #ffc107; /* Yellow/Orange */
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50; /* Green progress bar */
    }
    .stSpinner > div > div {
        border-top-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- S&P 500 Tickers (simplified list for example, replace with full list) ---
# In a real application, you might load this from a file or API
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "LLY", "XOM", "CVX", "KO", "PEP", "MRNA", "AES",
    "NCLH", "ON", "KHC", "MCD", "MMM", "GS", "NFLX", "INTC", "CSCO", "VZ", "T", "ORCL",
    "CRM", "ADBE", "PYPL", "CMCSA", "DIS", "BAC", "WFC", "C", "MS", "GE", "BA", "CAT",
    "WMT", "COST", "NKE", "SBUX", "PFE", "MRK", "ABBV", "AMGN", "GILD", "BMY", "SPG",
    "PLD", "EQIX", "AMT", "CCI", "DUK", "SO", "NEE", "D", "PCG", "EXC", "AEP", "SRE",
    "XLU", "IVZ", "SMCI", "TTD", "XYZ", # XYZ is a placeholder for 'SQ' or 'Block, Inc.'
    "MDLZ", "BF-B", "CBRE", # Additional tickers from user's CSV
    # Add more S&P 500 tickers here as needed
]
# For testing purposes, uncomment a smaller subset:
# SP500_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "MRNA", "AES", "NCLH"]

@st.cache_data(ttl=timedelta(hours=12)) # Cache results for 12 hours
def get_sp500_tickers_list():
    """Fetches a more comprehensive list of S&P 500 tickers.
    In a real-world scenario, you might parse from Wikipedia or a financial API."""
    try:
        # Using a reliable Wikipedia page for S&P 500 components
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        tickers = df['Symbol'].tolist()
        # Handle cases like "BRK.B" which yfinance handles as "BRK-B"
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        st.error(f"Failed to fetch S&P 500 tickers from Wikipedia: {e}. Using fallback list.")
        return SP500_TICKERS # Fallback to predefined list

# --- Data Fetching and Indicator Calculation ---
def get_tipranks_data(symbol):
    """Fetches Smart Score, Analyst Consensus, and Price Target from TipRanks."""
    tipranks_data = {
        "SmartScore": np.nan,
        "AnalystConsensus": "לא זמין",
        "PriceTarget %↑": np.nan,
        "TipRanks_URL": f"https://www.tipranks.com/stocks/{symbol}"
    }
    try:
        # Using a more standard User-Agent to improve request success
        url = f"https://mobile.tipranks.com/api/stocks/stockAnalysisOverview?tickers={symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,he;q=0.8',
            'Referer': 'https://www.tipranks.com/'
        }
        response = requests.get(url, headers=headers, timeout=10) # Increased timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data and isinstance(data, list) and len(data) > 0:
            stock_info = data[0] # Assuming it's a list with one item for the ticker

            # Smart Score
            # TipRanks API structure for smartScore can be { 'score': 6 } or just 6.
            smart_score_raw = stock_info.get('smartScore')
            if isinstance(smart_score_raw, dict):
                smart_score = smart_score_raw.get('score', np.nan)
            else:
                smart_score = smart_score_raw # Directly use if it's not a dict (e.g., an int)

            if pd.notna(smart_score) and smart_score: # Ensure it's not None/empty
                tipranks_data["SmartScore"] = smart_score

            # Analyst Consensus and Price Target
            analyst_data = stock_info.get('analystConsensus', {})
            tipranks_data["AnalystConsensus"] = analyst_data.get('consensus', 'לא זמין')

            price_target_obj = analyst_data.get('priceTarget', {})
            # Ensure these keys exist and are not None
            current_price_from_tipranks = price_target_obj.get('price')
            target_price = price_target_obj.get('target')

            if pd.notna(current_price_from_tipranks) and pd.notna(target_price) and current_price_from_tipranks != 0:
                price_target_percentage = ((target_price - current_price_from_tipranks) / current_price_from_tipranks) * 100
                tipranks_data["PriceTarget %↑"] = price_target_percentage

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.warning(f"TipRanks data for {symbol} not found (404).")
        else:
            st.error(f"TipRanks HTTP error for {symbol}: {e}")
    except requests.exceptions.ConnectionError as e:
        st.error(f"TipRanks connection error for {symbol}: {e}")
    except requests.exceptions.Timeout:
        st.error(f"TipRanks request timed out for {symbol}.")
    except json.JSONDecodeError as e:
        st.error(f"TipRanks JSON decode failed for {symbol}: {e}")
    except Exception as e:
        st.error(f"General error fetching TipRanks for {symbol}: {e}")

    return tipranks_data

def get_stock_data(symbol, period, interval):
    """Fetches stock data, calculates technical indicators, and determines AI Score."""
    ticker = yf.Ticker(symbol)
    try:
        # Adjust period to ensure enough data for indicators like SMA200
        if period == "3mo":
            history_period = "6mo" # Need more history for SMA200
        elif period == "1mo":
            history_period = "3mo"
        else:
            history_period = period # For "1y" or "5y", period is usually sufficient

        df = ticker.history(period=history_period, interval=interval)

        if df.empty:
            return None

        # Filter to the actual period requested after history is fetched
        if period == "3mo":
            df = df[df.index >= datetime.now() - timedelta(days=90)]
        elif period == "1mo":
            df = df[df.index >= datetime.now() - timedelta(days=30)]

        if df.empty:
            return None # No data after filtering

        # Calculate technical indicators using pandas_ta
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD']
        df['MACD_Signal'] = macd['MACDS']
        df['MACD_Hist'] = macd['MACDH']

        df['SMA20'] = ta.sma(df['Close'], length=20)
        df['SMA50'] = ta.sma(df['Close'], length=50)
        df['SMA200'] = ta.sma(df['Close'], length=200)

        bbands = ta.bbands(df['Close'])
        df['BBL'] = bbands['BBL_5_2.0'] # Bollinger Lower Band
        df['BBM'] = bbands['BBM_5_2.0'] # Bollinger Middle Band (SMA)
        df['BBU'] = bbands['BBU_5_2.0'] # Bollinger Upper Band

        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']

        # New Indicators
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['STOCHk'] = stoch['STOCHk_14_3_3']
        df['STOCHd'] = stoch['STOCHd_14_3_3']

        df['OBV'] = ta.obv(df['Close'], df['Volume'])

        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)


        # Get latest values
        latest_data = df.iloc[-1]
        current_price = latest_data['Close']
        volume = latest_data['Volume']
        prev_close = df.iloc[-2]['Close'] if len(df) > 1 else current_price
        
        # Calculate 20-day change percentage based on available data
        price_20_days_ago = df['Close'].iloc[-(min(len(df), 20))] # Get 20th day if available
        if price_20_days_ago == 0: # Avoid division by zero
            change_20D_percent = np.nan
        else:
            change_20D_percent = ((current_price - price_20_days_ago) / price_20_days_ago) * 100

        # --- AI Score and Matching Signals Logic (Weighted) ---
        ai_score = 0
        matching_signals = 0
        
        # Define weights for different signals
        weights = {
            'RSI_Bullish': 10, 'RSI_Oversold': 5,
            'MACD_Bullish': 15, 'MACD_Cross': 10,
            'SMA_Bullish_20_50': 10, 'SMA_Bullish_50_200': 15, 'SMA_Price_Above_20': 5,
            'Bollinger_Bullish': 10,
            'ADX_StrongTrend_Bullish': 15, 'ADX_WeakTrend_Bullish': 5,
            'STOCH_Bullish': 10, 'STOCH_Oversold': 5,
            'OBV_Rising': 10,
            'Price_Above_Open': 5,
            'Volume_Higher_Than_Average': 5,
            'Price_Target_Positive': 20, # Higher weight for analyst price target
            'SmartScore_High': 25, # Highest weight for high SmartScore
            'SmartScore_Medium': 10,
            'Analyst_StrongBuy': 20,
            'Analyst_Buy': 15
        }

        # RSI (14)
        if latest_data['RSI'] > 50: # Bullish above 50, strong bullish above 70
            ai_score += weights['RSI_Bullish']
            matching_signals += 1
        if latest_data['RSI'] < 30 and latest_data['RSI'] > 0: # Oversold
            ai_score += weights['RSI_Oversold'] # Potential for rebound
            # Not adding to matching_signals as it's a contrarian signal for 'bullish' list

        # MACD (12, 26, 9)
        if latest_data['MACD'] > latest_data['MACD_Signal']:
            ai_score += weights['MACD_Bullish']
            matching_signals += 1
        if latest_data['MACD_Hist'] > 0 and df['MACD_Hist'].iloc[-2] <= 0: # MACD crossover bullish
            ai_score += weights['MACD_Cross']
            matching_signals += 1

        # SMAs (20, 50, 200)
        if current_price > latest_data['SMA20']:
            ai_score += weights['SMA_Price_Above_20']
            matching_signals += 1
        if latest_data['SMA20'] > latest_data['SMA50']:
            ai_score += weights['SMA_Bullish_20_50']
            matching_signals += 1
        if latest_data['SMA50'] > latest_data['SMA200']:
            ai_score += weights['SMA_Bullish_50_200']
            matching_signals += 1

        # Bollinger Bands (5, 2.0)
        if current_price > latest_data['BBM']: # Price above Middle Band
            ai_score += weights['Bollinger_Bullish']
            matching_signals += 1

        # ADX (14) - Trend Strength (ADX > 25 indicates strong trend)
        # Assuming we combine ADX with +DI vs -DI for direction
        if latest_data['ADX'] > 25:
            plus_di = ta.adx(df['High'], df['Low'], df['Close'], length=14)['DMP_14']
            minus_di = ta.adx(df['High'], df['Low'], df['Close'], length=14)['DMN_14']
            if plus_di.iloc[-1] > minus_di.iloc[-1]: # +DI above -DI indicates bullish trend
                ai_score += weights['ADX_StrongTrend_Bullish']
                matching_signals += 1
            else: # Strong bearish trend, or not bullish
                 ai_score -= 10 # Deduct points if strong trend is bearish
        elif latest_data['ADX'] > 20: # Moderate trend
             plus_di = ta.adx(df['High'], df['Low'], df['Close'], length=14)['DMP_14']
             minus_di = ta.adx(df['High'], df['Low'], df['Close'], length=14)['DMN_14']
             if plus_di.iloc[-1] > minus_di.iloc[-1]:
                 ai_score += weights['ADX_WeakTrend_Bullish']
                 matching_signals += 1

        # Stochastic Oscillator (14,3,3)
        if latest_data['STOCHk'] > latest_data['STOCHd'] and latest_data['STOCHk'] < 80: # Bullish crossover, not overbought
            ai_score += weights['STOCH_Bullish']
            matching_signals += 1
        if latest_data['STOCHk'] < 20 and latest_data['STOCHd'] < 20: # Oversold, potential for rebound
            ai_score += weights['STOCH_Oversold']
            # Not adding to matching_signals as it's a contrarian signal for 'bullish' list

        # OBV (On-Balance Volume)
        if latest_data['OBV'] > df['OBV'].iloc[-2] if len(df) > 1 else False: # OBV is rising
            ai_score += weights['OBV_Rising']
            matching_signals += 1

        # Basic Price & Volume Signals
        if current_price > prev_close:
            ai_score += weights['Price_Above_Open'] # Rename this to 'Price_Above_Previous_Close'
            matching_signals += 1
        
        # Check if latest_data has 'Average Volume' and use it. If not, calculate.
        # This assumes Average Volume is available, if not, it should be calculated over a period
        # For simplicity, let's assume `Average Volume` is available or defined from a broader context
        # If the user's data includes an 'Average Volume' already, we can use that for comparison.
        # Otherwise, we might need to calculate a historical average volume from `df`.
        # For now, let's use the average of the fetched period.
        avg_volume_for_period = df['Volume'].mean()
        if volume > avg_volume_for_period:
            ai_score += weights['Volume_Higher_Than_Average']
            matching_signals += 1

        # --- TipRanks Data Integration ---
        tipranks_info = get_tipranks_data(symbol)
        
        if pd.notna(tipranks_info["SmartScore"]):
            if tipranks_info["SmartScore"] >= 8: # High SmartScore
                ai_score += weights['SmartScore_High']
                matching_signals += 1
            elif tipranks_info["SmartScore"] >= 6: # Medium SmartScore
                ai_score += weights['SmartScore_Medium']
                matching_signals += 1

        if tipranks_info["AnalystConsensus"] == "StrongBuy":
            ai_score += weights['Analyst_StrongBuy']
            matching_signals += 1
        elif tipranks_info["AnalystConsensus"] == "Buy":
            ai_score += weights['Analyst_Buy']
            matching_signals += 1

        if pd.notna(tipranks_info["PriceTarget %↑"]) and tipranks_info["PriceTarget %↑"] > 0:
            ai_score += weights['Price_Target_Positive']
            matching_signals += 1

        # Compile results
        result = {
            "Symbol": symbol,
            "Current Price": current_price,
            "20D Change %": change_20D_percent,
            "Average Volume": volume, # This is current volume, not historical average
            "RSI": latest_data['RSI'],
            "MACD_Hist": latest_data['MACD_Hist'],
            "SMA20": latest_data['SMA20'],
            "SMA50": latest_data['SMA50'],
            "SMA200": latest_data['SMA200'],
            "BBL": latest_data['BBL'],
            "BBM": latest_data['BBM'],
            "BBU": latest_data['BBU'],
            "ADX": latest_data['ADX'],
            "STOCHk": latest_data['STOCHk'],
            "STOCHd": latest_data['STOCHd'],
            "OBV": latest_data['OBV'],
            "ATR": latest_data['ATR'],
            "AI Score": ai_score,
            "Matching Signals": matching_signals,
            "TipRanks_URL": tipranks_info["TipRanks_URL"],
            "SmartScore": tipranks_info["SmartScore"],
            "AnalystConsensus": tipranks_info["AnalystConsensus"],
            "PriceTarget %↑": tipranks_info["PriceTarget %↑"]
        }

        # Add company name and sector using yfinance info
        info = ticker.info
        result["Company"] = info.get('longName', 'N/A')
        result["Sector"] = info.get('sector', 'N/A')
        result["Industry"] = info.get('industry', 'N/A')
        result["Market Cap"] = info.get('marketCap', np.nan)


        return result

    except Exception as e:
        st.error(f"Error processing {symbol}: {e}")
        return None

# --- Streamlit UI ---
st.title("🚀 Mozes Super-AI S&P 500 Scanner")

st.markdown("""
ברוך הבא לסורק מניות ה-S&P 500 החכם!
הכלי הזה סורק את מניות מדד ה-S&P 500, מחשב אינדיקטורים טכניים, אוסף נתוני TipRanks ומעניק ציון "AI Score" לכל מניה.
הציון משקלל סימנים שווריים שונים כדי לעזור לך לזהות מניות עם פוטנציאל.
""")

# Sidebar for controls
st.sidebar.header("⚙️ הגדרות סריקה")

# Dynamic period selection
scan_period_options = {
    "חודש אחרון (1mo)": "1mo",
    "3 חודשים אחרונים (3mo)": "3mo",
    "שנה אחרונה (1y)": "1y",
    "5 שנים אחרונות (5y)": "5y"
}
selected_period_display = st.sidebar.selectbox("בחר תקופת נתונים:", list(scan_period_options.keys()))
selected_period = scan_period_options[selected_period_display]

interval_options = {
    "יומי (1d)": "1d",
    "שבועי (1wk)": "1wk",
}
selected_interval_display = st.sidebar.selectbox("בחר אינטרוול נתונים:", list(interval_options.keys()))
selected_interval = interval_options[selected_interval_display]

min_score = st.sidebar.slider("ציון AI מינימלי להצגה", 0, 150, 70, help="מניות בעלות ציון AI נמוך יותר לא יוצגו בתוצאות המסוננות.")
num_threads = st.sidebar.slider("מספר תהליכי סריקה במקביל", 1, 10, 5, help="מספר המניות שייסרקו במקביל. ככל שמספר המניות גדול יותר, כך ייתכן שהסריקה תהיה מהירה יותר, אך עלולה להיתקל במגבלות API.")

# Generate a hash of current settings to detect changes
current_settings_hash = hashlib.md5(f"{selected_period}-{selected_interval}-{min_score}-{num_threads}".encode()).hexdigest()

st.sidebar.subheader("📊 סינון תוצאות")
min_20d_change = st.sidebar.number_input("שינוי 20 יום מינימלי (%)", value=-10.0, step=1.0)
min_volume = st.sidebar.number_input("נפח מסחר ממוצע מינימלי", value=1000000, step=100000)
selected_sector = st.sidebar.selectbox("בחר סקטור (אופציונלי)", ["כל הסקטורים"] + sorted(list(set([res.get("Sector", "N/A") for res in st.session_state.scanner_results if "Sector" in res]))))
selected_industry = st.sidebar.selectbox("בחר ענף (אופציונלי)", ["כל הענפים"] + sorted(list(set([res.get("Industry", "N/A") for res in st.session_state.scanner_results if "Industry" in res]))))
selected_analyst_consensus = st.sidebar.selectbox("קונצנזוס אנליסטים (אופציונלי)", ["הכל", "StrongBuy", "Buy", "Hold", "Sell", "StrongSell", "לא זמין"])
show_only_price_target_positive = st.sidebar.checkbox("הצג רק מניות עם יעד מחיר חיובי", value=False)


scan_button_col, reset_button_col = st.sidebar.columns(2)

# Scan button logic
if scan_button_col.button("🚀 התחל סריקה", key="start_scan_button") or \
   (st.session_state.last_scan_time and current_settings_hash != st.session_state.scan_settings_hash and not st.session_state.is_scanning):
    
    st.session_state.is_scanning = True
    st.session_state.scan_status_message = "מתחיל סריקה..."
    st.session_state.scanner_results = [] # Clear previous results
    st.session_state.scan_progress = 0
    st.session_state.last_scan_time = datetime.now()
    st.session_state.scan_settings_hash = current_settings_hash
    
    st.experimental_rerun() # Rerun to show initial status

if st.session_state.is_scanning:
    st.info(st.session_state.scan_status_message)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_sp500_tickers = get_sp500_tickers_list() # Get the full list
    total_tickers = len(all_sp500_tickers)
    
    results_list = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_symbol = {executor.submit(get_stock_data, symbol, selected_period, selected_interval): symbol for symbol in all_sp500_tickers}
        
        for i, future in enumerate(as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if data:
                    results_list.append(data)
            except Exception as exc:
                st.error(f"'{symbol}' generated an exception: {exc}")
            
            # Update progress
            st.session_state.scan_progress = (i + 1) / total_tickers
            progress_bar.progress(st.session_state.scan_progress)
            status_text.text(f"סורק מניות... {i+1}/{total_tickers} ({st.session_state.scan_progress:.1%})")
            
    st.session_state.scanner_results = results_list
    st.session_state.is_scanning = False
    st.session_state.scan_status_message = f"סריקה הושלמה! נמצאו {len(results_list)} מניות."
    progress_bar.progress(1.0)
    status_text.text(st.session_state.scan_status_message)
    st.success("סריקה הסתיימה בהצלחה!")
    st.balloons()
    st.experimental_rerun() # Rerun to show final results and hide progress bar

# Reset button logic
if reset_button_col.button("🔄 איפוס סריקה", key="reset_scan_button"):
    st.session_state.scanner_results = []
    st.session_state.scan_status_message = "מוכן לסריקה"
    st.session_state.last_scan_time = None
    st.session_state.scan_settings_hash = None
    st.session_state.is_scanning = False
    st.experimental_rerun()


# Display results
st.header("📈 תוצאות הסריקה")

if st.session_state.last_scan_time:
    st.info(f"סריקה אחרונה בוצעה ב: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")

df_results = pd.DataFrame(st.session_state.scanner_results)

if not df_results.empty:
    # Filter based on sidebar controls
    df_filtered = df_results[df_results['AI Score'] >= min_score]
    df_filtered = df_filtered[df_filtered['20D Change %'] >= min_20d_change]
    df_filtered = df_filtered[df_filtered['Average Volume'] >= min_volume]

    if selected_sector != "כל הסקטורים":
        df_filtered = df_filtered[df_filtered['Sector'] == selected_sector]
    if selected_industry != "כל הענפים":
        df_filtered = df_filtered[df_filtered['Industry'] == selected_industry]
    if selected_analyst_consensus != "הכל":
        df_filtered = df_filtered[df_filtered['AnalystConsensus'] == selected_analyst_consensus]
    if show_only_price_target_positive:
        df_filtered = df_filtered[df_filtered['PriceTarget %↑'] > 0]


    # Sort by AI Score descending
    df_filtered_display = df_filtered.sort_values(by="AI Score", ascending=False).reset_index(drop=True)
    
    if not df_filtered_display.empty:
        st.subheader(f"✅ מניות עם ציון AI של {min_score}+ ({len(df_filtered_display)} נמצאו)")
        
        # Select columns to display in the main table
        display_columns = [
            "Symbol", "Company", "Sector", "Industry", "Market Cap", "Current Price", "20D Change %",
            "Average Volume", "AI Score", "Matching Signals", "SmartScore", "AnalystConsensus", "PriceTarget %↑",
            "RSI", "MACD_Hist", "SMA20", "SMA50", "SMA200", "ADX", "STOCHk", "OBV", "ATR", "TipRanks_URL"
        ]
        
        # Filter out columns that might not exist if data fetching failed for all stocks
        existing_display_columns = [col for col in display_columns if col in df_filtered_display.columns]

        st.dataframe(df_filtered_display[existing_display_columns].style.format({
            "Current Price": "${:.2f}",
            "20D Change %": "{:.2f}%",
            "Average Volume": "{:,.0f}",
            "Market Cap": "${:,.0f}",
            "AI Score": "{:.2f}",
            "RSI": "{:.2f}",
            "MACD_Hist": "{:.2f}",
            "SMA20": "{:.2f}",
            "SMA50": "{:.2f}",
            "SMA200": "{:.2f}",
            "ADX": "{:.2f}",
            "STOCHk": "{:.2f}",
            "STOCHd": "{:.2f}",
            "OBV": "{:,.0f}",
            "ATR": "{:.2f}",
            "SmartScore": "{:.0f}",
            "PriceTarget %↑": "{:.2f}%"
        }), use_container_width=True, hide_index=True)
        
        st.markdown("---")

        # Visualizations
        st.subheader("📊 הדמיות נתונים")

        # Scatter plot: AI Score vs 20-Day Change %
        if '20D Change %' in df_filtered_display.columns and 'AI Score' in df_filtered_display.columns:
            fig_scatter = px.scatter(df_filtered_display,
                                     x="20D Change %",
                                     y="AI Score",
                                     size="Market Cap",
                                     color="Sector",
                                     hover_name="Company",
                                     title="AI Score vs 20-Day Performance by Market Cap & Sector",
                                     labels={"20D Change %": "שינוי 20 יום (%)", "AI Score": "ציון AI"},
                                     size_max=60,
                                     template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("אין מספיק נתונים עבור תרשים פיזור AI Score vs 20-Day Change %.")

        # Bar chart: Top 10 by AI Score
        if not df_filtered_display.empty:
            top_10_ai = df_filtered_display.head(10).sort_values(by="AI Score", ascending=True)
            fig_bar = px.bar(top_10_ai,
                             x="AI Score",
                             y="Symbol",
                             orientation='h',
                             color="AI Score",
                             color_continuous_scale=px.colors.sequential.Viridis,
                             title="10 המניות המובילות לפי ציון AI",
                             labels={"AI Score": "ציון AI", "Symbol": "סימול"},
                             hover_data=["Company", "Sector", "Current Price", "20D Change %"])
            st.plotly_chart(fig_bar, use_container_width=True)

        # Show top 10 stocks regardless of score (if results available)
        if not df_results.empty:
            st.subheader("🔍 10 המניות המובילות (ללא סינון ציון AI)")
            df_top_10_all = df_results.sort_values(by="AI Score", ascending=False).head(10).reset_index(drop=True)
            
            # Select relevant columns for this table
            final_columns_top_10 = ["Symbol", "Company", "Sector", "Current Price", "20D Change %", "AI Score", "Matching Signals"]
            # Add Industry if it exists in the dataframe
            if "Industry" in df_top_10_all.columns:
                final_columns_top_10.insert(3, "Industry") # Insert at index 3
            if "Market Cap" in df_top_10_all.columns:
                final_columns_top_10.insert(4, "Market Cap")

            st.dataframe(df_top_10_all[final_columns_top_10].style.format({
                "Current Price": "${:.2f}",
                "20D Change %": "{:.2f}%",
                "Average Volume": "{:,.0f}",
                "AI Score": "{:.2f}",
                "Market Cap": "${:,.0f}"
            }), use_container_width=True, hide_index=True)
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
        st.metric("🔥 הציון הטוב ביותר (בסינון)", f"{df_filtered_display['AI Score'].max():.2f}")
        st.metric("📈 ממוצע שינוי 20 יום (בסינון)", f"{df_filtered_display['20D Change %'].mean():.1f}%")
        
else:
    st.warning(st.session_state.scan_status_message)
    if not st.session_state.is_scanning:
        st.info("כדי להתחיל, לחץ על 'התחל סריקה' בסרגל הצד.")
