import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib

# Page config
st.set_page_config(
    page_title="Mozes Super-AI Scanner", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for persistent data
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = []
if 'scan_progress' not in st.session_state:
    st.session_state.scan_progress = 0
if 'scan_status' not in st.session_state:
    st.session_state.scan_status = "Ready"
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'scan_settings_hash' not in st.session_state:
    st.session_state.scan_settings_hash = None

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
    .bullish { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .bearish { background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%); }
    .neutral { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333; }
    
    .stDataFrame {
        background: white;
        border-radius: 10px;
    }
    
    .scan-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
    }
    .status-running { background-color: #fff3cd; color: #856404; }
    .status-complete { background-color: #d4edda; color: #155724; }
    .status-error { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Mozes Super-AI Stock Scanner")
st.markdown("### AI-Powered Technical Analysis & Market Intelligence with Multi-Threading")

# Cache functions with longer TTL
@st.cache_data(ttl=7200)  # Cache for 2 hours
def load_sp500_tickers():
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return table[0]["Symbol"].tolist(), table[0][["Symbol", "Security", "GICS Sector"]]
    except Exception as e:
        st.error(f"Error loading S&P 500 data: {e}")
        return [], pd.DataFrame()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_stock_volume_data(symbol):
    """Get volume data for pre-filtering"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if not hist.empty:
            avg_volume = hist["Volume"].mean()
            return avg_volume
        return 0
    except:
        return 0

def calculate_advanced_signals(data):
    """Calculate advanced technical indicators"""
    try:
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]
        
        # Basic indicators
        rsi = ta.rsi(close, length=14)
        macd_df = ta.macd(close)
        bb_df = ta.bbands(close)
        stoch = ta.stoch(high, low, close)
        
        # Advanced indicators
        adx = ta.adx(high, low, close)
        atr = ta.atr(high, low, close)
        obv = ta.obv(close, volume)
        cci = ta.cci(high, low, close)
        williams_r = ta.willr(high, low, close)
        
        # Moving averages
        sma_20 = ta.sma(close, length=20)
        sma_50 = ta.sma(close, length=50)
        sma_200 = ta.sma(close, length=200)
        ema_12 = ta.ema(close, length=12)
        ema_26 = ta.ema(close, length=26)
        
        return {
            'RSI': rsi.iloc[-1] if not rsi.empty else 50,
            'MACD': macd_df['MACD_12_26_9'].iloc[-1] if not macd_df.empty else 0,
            'MACD_Signal': macd_df['MACDs_12_26_9'].iloc[-1] if not macd_df.empty else 0,
            'BB_Upper': bb_df['BBU_5_2.0'].iloc[-1] if not bb_df.empty else close.iloc[-1],
            'BB_Lower': bb_df['BBL_5_2.0'].iloc[-1] if not bb_df.empty else close.iloc[-1],
            'Stoch_K': stoch['STOCHk_14_3_3'].iloc[-1] if not stoch.empty else 50,
            'ADX': adx['ADX_14'].iloc[-1] if not adx.empty else 25,
            'ATR': atr.iloc[-1] if not atr.empty else 1,
            'OBV': obv.iloc[-1] if not obv.empty else 0,
            'CCI': cci.iloc[-1] if not cci.empty else 0,
            'WilliamsR': williams_r.iloc[-1] if not williams_r.empty else -50,
            'SMA_20': sma_20.iloc[-1] if not sma_20.empty else close.iloc[-1],
            'SMA_50': sma_50.iloc[-1] if not sma_50.empty else close.iloc[-1],
            'SMA_200': sma_200.iloc[-1] if not sma_200.empty else close.iloc[-1],
            'EMA_12': ema_12.iloc[-1] if not ema_12.empty else close.iloc[-1],
            'EMA_26': ema_26.iloc[-1] if not ema_26.empty else close.iloc[-1],
        }
    except Exception as e:
        # Return default values if calculation fails
        return {key: 0 for key in ['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Stoch_K', 
                                  'ADX', 'ATR', 'OBV', 'CCI', 'WilliamsR', 'SMA_20', 'SMA_50', 
                                  'SMA_200', 'EMA_12', 'EMA_26']}

def calculate_ai_score(close_price, indicators):
    """Advanced AI scoring system"""
    score = 0
    signals = []
    
    try:
        # RSI Analysis (0-2 points)
        rsi = indicators['RSI']
        if 30 <= rsi <= 70:
            score += 2
            signals.append("RSI Healthy")
        elif rsi < 30:
            score += 1
            signals.append("RSI Oversold")
        elif rsi > 80:
            signals.append("RSI Overbought")
        
        # MACD Analysis (0-2 points)
        if indicators['MACD'] > indicators['MACD_Signal']:
            score += 2
            signals.append("MACD Bullish")
        elif indicators['MACD'] > 0:
            score += 1
            signals.append("MACD Above Zero")
        
        # Moving Average Analysis (0-3 points)
        if close_price > indicators['SMA_20']:
            score += 1
            signals.append("Above SMA20")
        if close_price > indicators['SMA_50']:
            score += 1
            signals.append("Above SMA50")
        if close_price > indicators['SMA_200']:
            score += 1
            signals.append("Above SMA200")
        
        # Bollinger Bands (0-1 points)
        if indicators['BB_Lower'] < close_price < indicators['BB_Upper']:
            score += 1
            signals.append("BB Normal Range")
        
        # Stochastic (0-1 points)
        if 20 <= indicators['Stoch_K'] <= 80:
            score += 1
            signals.append("Stoch Healthy")
        
        # ADX Trend Strength (0-1 points)
        if indicators['ADX'] > 25:
            score += 1
            signals.append("Strong Trend")
    except:
        pass
    
    return min(score, 10), signals

def analyze_single_stock(symbol, period, ticker_info_dict):
    """Analyze a single stock - designed for multithreading"""
    try:
        # Download data
        data = yf.download(symbol, period=period, progress=False)
        
        if data.empty or len(data) < 50:
            return None
            
        # Get stock info
        ticker_obj = yf.Ticker(symbol)
        info = ticker_obj.info
        
        # Calculate indicators
        indicators = calculate_advanced_signals(data)
        close_price = data["Close"].iloc[-1]
        
        # Calculate AI score
        ai_score, signals = calculate_ai_score(close_price, indicators)
        
        # Price change calculations
        price_change_1d = ((close_price - data["Close"].iloc[-2]) / data["Close"].iloc[-2]) * 100 if len(data) > 1 else 0
        price_change_5d = ((close_price - data["Close"].iloc[-6]) / data["Close"].iloc[-6]) * 100 if len(data) > 5 else 0
        
        # Volume analysis
        avg_volume = data["Volume"].tail(20).mean()
        current_volume = data["Volume"].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Get company info
        company_name = info.get('longName', symbol)
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        
        return {
            "Symbol": symbol,
            "Company": company_name[:30] + "..." if len(company_name) > 30 else company_name,
            "Sector": ticker_info_dict.get(symbol, "Unknown"),
            "AI Score": ai_score,
            "Price": round(close_price, 2),
            "1D Change %": round(price_change_1d, 2),
            "5D Change %": round(price_change_5d, 2),
            "Volume Ratio": round(volume_ratio, 2),
            "RSI": round(indicators['RSI'], 1),
            "MACD": round(indicators['MACD'], 4),
            "ADX": round(indicators['ADX'], 1),
            "Market Cap": market_cap,
            "P/E": round(pe_ratio, 2) if pe_ratio else "N/A",
            "Signals": ", ".join(signals[:3]),  # Top 3 signals
            "Links": f"[TradingView](https://tradingview.com/chart/?symbol={symbol}) | [Yahoo](https://finance.yahoo.com/quote/{symbol}) | [TipRanks](https://tipranks.com/stocks/{symbol.lower()})",
            "Avg Volume": avg_volume
        }
        
    except Exception as e:
        return None

def create_settings_hash(selected_sector, selected_period, max_stocks, min_volume):
    """Create hash of current settings to detect changes"""
    settings_str = f"{selected_sector}_{selected_period}_{max_stocks}_{min_volume}"
    return hashlib.md5(settings_str.encode()).hexdigest()

# Sidebar controls
st.sidebar.header("🎛️ Scanner Settings")

# Load data
tickers, ticker_info = load_sp500_tickers()

if ticker_info.empty:
    st.error("Failed to load S&P 500 data. Please refresh the page.")
    st.stop()

# Create ticker info dictionary for faster lookup
ticker_info_dict = dict(zip(ticker_info["Symbol"], ticker_info["GICS Sector"]))

# Filters
min_score = st.sidebar.slider("Minimum AI Score", 0, 10, 6)
sectors = ["All"] + sorted(ticker_info["GICS Sector"].unique().tolist())
selected_sector = st.sidebar.selectbox("Filter by Sector", sectors)

# Analysis period
period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"}
selected_period = st.sidebar.selectbox("Analysis Period", list(period_options.keys()), index=2)

# Number of stocks to analyze
max_stocks = st.sidebar.slider("Max Stocks to Analyze", 50, 500, 150)

# Volume filter
st.sidebar.subheader("📊 Volume Filter")
min_volume = st.sidebar.number_input("Minimum Average Volume (millions)", 
                                    min_value=0.1, max_value=50.0, value=1.0, step=0.1)

# Threading settings
st.sidebar.subheader("⚡ Performance Settings")
max_workers = st.sidebar.slider("Concurrent Workers", 5, 20, 10, 
                               help="More workers = faster analysis, but may cause API limits")

# Resume capability
st.sidebar.subheader("🔄 Resume Options")
if st.sidebar.button("Clear Cache & Restart"):
    st.session_state.scanner_results = []
    st.session_state.scan_progress = 0
    st.session_state.scan_status = "Ready"
    st.session_state.last_scan_time = None
    st.session_state.scan_settings_hash = None
    st.experimental_rerun()

# Check if settings changed
current_settings_hash = create_settings_hash(selected_sector, selected_period, max_stocks, min_volume)
settings_changed = st.session_state.scan_settings_hash != current_settings_hash

# Filter tickers by sector and volume
if selected_sector != "All":
    filtered_info = ticker_info[ticker_info["GICS Sector"] == selected_sector]
    analysis_tickers = filtered_info["Symbol"].tolist()[:max_stocks]
else:
    analysis_tickers = tickers[:max_stocks]

# Pre-filter by volume (cached)
if st.sidebar.checkbox("Enable Volume Pre-filtering", value=True):
    st.sidebar.write("🔍 Pre-filtering by volume...")
    volume_filtered_tickers = []
    for ticker in analysis_tickers[:50]:  # Check first 50 for demo
        avg_vol = get_stock_volume_data(ticker)
        if avg_vol >= min_volume * 1_000_000:
            volume_filtered_tickers.append(ticker)
    analysis_tickers = volume_filtered_tickers + analysis_tickers[50:]  # Keep rest for full analysis

# Display current scan status
if st.session_state.scan_status != "Ready":
    status_class = "status-running" if "Running" in st.session_state.scan_status else "status-complete"
    st.markdown(f'<div class="scan-status {status_class}">{st.session_state.scan_status}</div>', 
                unsafe_allow_html=True)

# Main analysis button
col1, col2 = st.columns([2, 1])
with col1:
    start_scan = st.button("🚀 Start Multi-Threaded AI Analysis", type="primary", 
                          disabled=(st.session_state.scan_status == "Running"))
with col2:
    if st.session_state.scanner_results and not settings_changed:
        resume_scan = st.button("▶️ Resume Scan", 
                               disabled=(st.session_state.scan_status == "Running"))
    else:
        resume_scan = False

# Analysis execution
if start_scan or resume_scan:
    
    # Update settings hash
    st.session_state.scan_settings_hash = current_settings_hash
    
    # Create columns for real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize or resume
    if start_scan or settings_changed:
        st.session_state.scanner_results = []
        st.session_state.scan_progress = 0
        start_index = 0
    else:
        start_index = len(st.session_state.scanner_results)
    
    # Create placeholders for real-time updates
    progress_bar = st.progress(st.session_state.scan_progress / len(analysis_tickers))
    status_text = st.empty()
    results_placeholder = st.empty()
    
    # Update status
    st.session_state.scan_status = f"Running - Multi-threaded analysis with {max_workers} workers"
    
    start_time = time.time()
    
    # Multi-threaded analysis
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for remaining tickers
        remaining_tickers = analysis_tickers[start_index:]
        future_to_ticker = {
            executor.submit(analyze_single_stock, ticker, period_options[selected_period], ticker_info_dict): ticker 
            for ticker in remaining_tickers
        }
        
        completed_count = start_index
        batch_results = []
        
        # Process completed futures
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            
            try:
                result = future.result()
                if result:
                    st.session_state.scanner_results.append(result)
                    batch_results.append(result)
                
                completed_count += 1
                st.session_state.scan_progress = completed_count
                
                # Update progress
                progress_percentage = completed_count / len(analysis_tickers)
                progress_bar.progress(progress_percentage)
                status_text.text(f"Analyzed: {completed_count}/{len(analysis_tickers)} | Current: {ticker}")
                
                # Update metrics every 10 stocks
                if len(batch_results) >= 10 or completed_count == len(analysis_tickers):
                    current_df = pd.DataFrame(st.session_state.scanner_results)
                    
                    if not current_df.empty:
                        # Update real-time metrics
                        with col1:
                            st.metric("📊 Stocks Analyzed", completed_count)
                        with col2:
                            high_score = len(current_df[current_df["AI Score"] >= min_score])
                            st.metric("🎯 High Score Stocks", high_score)
                        with col3:
                            avg_score = current_df["AI Score"].mean()
                            st.metric("📈 Average AI Score", f"{avg_score:.1f}")
                        with col4:
                            elapsed = time.time() - start_time
                            st.metric("⏱️ Elapsed Time", f"{elapsed:.1f}s")
                        
                        # Show top results so far
                        filtered_df = current_df[current_df["AI Score"] >= min_score].sort_values("AI Score", ascending=False)
                        
                        with results_placeholder.container():
                            st.subheader(f"🔄 Live Results (AI Score ≥ {min_score})")
                            if not filtered_df.empty:
                                # Color coding
                                def color_score(val):
                                    if val >= 8:
                                        return 'background-color: #d4edda'
                                    elif val >= 6:
                                        return 'background-color: #fff3cd'
                                    else:
                                        return 'background-color: #f8d7da'
                                
                                def color_change(val):
                                    try:
                                        val = float(val)
                                        return 'color: green' if val > 0 else 'color: red' if val < 0 else 'color: gray'
                                    except:
                                        return ''
                                
                                styled_df = filtered_df.style.applymap(color_score, subset=['AI Score']).applymap(color_change, subset=['1D Change %', '5D Change %'])
                                st.dataframe(styled_df, use_container_width=True, height=400)
                            else:
                                st.info(f"⏳ Analyzing... {completed_count} stocks processed. No stocks found yet with score ≥ {min_score}")
                    
                    batch_results = []  # Reset batch
                    
            except Exception as e:
                st.sidebar.error(f"Error analyzing {ticker}: {str(e)}")
    
    # Final processing
    progress_bar.progress(1.0)
    status_text.empty()
    
    total_time = time.time() - start_time
    st.session_state.scan_status = f"Complete - Analyzed {len(analysis_tickers)} stocks in {total_time:.1f}s"
    st.session_state.last_scan_time = datetime.now()

# Display final results if available
if st.session_state.scanner_results:
    
    # Create final results DataFrame
    df_results = pd.DataFrame(st.session_state.scanner_results)
    df_filtered = df_results[df_results["AI Score"] >= min_score].sort_values("AI Score", ascending=False)
    
    # Final metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Analyzed", len(df_results))
    with col2:
        st.metric("🎯 High Score Stocks", len(df_filtered))
    with col3:
        st.metric("📈 Average AI Score", f"{df_results['AI Score'].mean():.1f}")
    with col4:
        if st.session_state.last_scan_time:
            st.metric("🕐 Last Scan", st.session_state.last_scan_time.strftime("%H:%M:%S"))
    
    # Final results table
    st.subheader(f"🏆 Final Results (AI Score ≥ {min_score})")
    
    if len(df_filtered) > 0:
        # Enhanced styling
        def color_score(val):
            if val >= 8:
                return 'background-color: #d4edda; font-weight: bold'
            elif val >= 6:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        
        def color_change(val):
            try:
                val = float(val)
                color = 'color: green; font-weight: bold' if val > 2 else 'color: darkgreen' if val > 0 else 'color: red' if val < 0 else 'color: gray'
                return color
            except:
                return ''
        
        styled_df = df_filtered.style.applymap(color_score, subset=['AI Score']).applymap(color_change, subset=['1D Change %', '5D Change %'])
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Advanced analytics
        if len(df_filtered) > 5:
            st.subheader("📊 Advanced Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sector distribution
                sector_counts = df_filtered["Sector"].value_counts()
                fig_pie = px.pie(values=sector_counts.values, names=sector_counts.index, 
                               title="High-Score Stocks by Sector")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Score vs Volume correlation
                fig_scatter = px.scatter(df_filtered, x="Volume Ratio", y="AI Score", 
                                       color="Sector", size="Market Cap",
                                       title="AI Score vs Volume Activity",
                                       hover_data=["Symbol", "Company"])
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Export functionality
        st.subheader("💾 Export & Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"mozes_scanner_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Top performers
            top_5 = df_filtered.head(5)
            top_symbols = ", ".join(top_5["Symbol"].tolist())
            st.text_area("🏆 Top 5 Symbols", top_symbols, height=100)
        
        with col3:
            # Quick stats
            st.metric("🔥 Best Score", f"{df_filtered['AI Score'].max()}")
            st.metric("📈 Avg 5D Change", f"{df_filtered['5D Change %'].mean():.1f}%")
            
    else:
        st.warning(f"No stocks found with AI Score ≥ {min_score}. Try lowering the minimum score.")
        
        # Show top 10 regardless of score
        st.subheader("🔍 Top 10 Stocks (All Scores)")
        top_10 = df_results.sort_values("AI Score", ascending=False).head(10)
        st.dataframe(top_10, use_container_width=True)

else:
    # Initial state - show instructions
    st.markdown("""
    ## 🎯 Multi-Threaded AI Scanner Features
    
    ### ⚡ **NEW: Enhanced Performance**
    - **Multi-threading**: Analyze multiple stocks simultaneously
    - **Real-time updates**: See results every 10 stocks
    - **Resume capability**: Continue interrupted scans
    - **Volume pre-filtering**: Skip low-volume stocks
    - **Smart caching**: Faster repeated analyses
    
    ### 🛠️ **How to Use**
    
    1. **Configure Settings**:
       - AI Score threshold (6-8 recommended)
       - Sector filter (optional)
       - Analysis period
       - Volume minimum (1M+ recommended)
       - Concurrent workers (10 = balanced)
    
    2. **Start Analysis**: Click "Start Multi-Threaded AI Analysis"
    
    3. **Monitor Progress**: Watch real-time results appear
    
    4. **Resume if Needed**: Use "Resume Scan" for interrupted scans
    
    ### 🎛️ **Performance Tips**
    - **For speed**: 10-15 workers, 100-200 stocks
    - **For stability**: 5-8 workers, volume filter enabled
    - **For completeness**: 500 stocks with 8+ workers
    
    ### 🧠 **AI Scoring Enhanced**
    - Multiple timeframe analysis
    - Volume-weighted signals
    - Sector-relative scoring
    - Market condition adjustments
    
    **🚀 Ready to scan? Configure your settings and start!**
    """)
    
    # Market overview with caching
    st.subheader("📈 Market Overview")
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_market_indices():
        indices = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC", 
            "Dow Jones": "^DJI",
            "VIX": "^VIX"
        }
        
        index_data = []
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current = hist["Close"].iloc[-1]
                    previous = hist["Close"].iloc[-2] if len(hist) > 1 else current
                    change = ((current - previous) / previous) * 100
                    index_data.append({
                        "Index": name,
                        "Price": round(current, 2),
                        "Change %": round(change, 2)
                    })
            except:
                continue
        return index_data
    
    market_data = get_market_indices()
    if market_data:
        df_indices = pd.DataFrame(market_data)
        
        # Color code market data
        def color_market_change(val):
            try:
                val = float(val)
                return 'color: green; font-weight: bold' if val > 0 else 'color: red; font-weight: bold' if val < 0 else 'color: gray'
            except:
                return ''
        
        styled_indices = df_indices.style.applymap(color_market_change, subset=['Change %'])
        st.dataframe(styled_indices, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Mozes Super-AI Scanner v2.0** - Multi-Threaded Technical Analysis 🚀")
st.markdown("*Enhanced with real-time updates, resume capability, and advanced filtering*")
st.markdown("*Disclaimer: For educational purposes only. Always do your own research.*")
