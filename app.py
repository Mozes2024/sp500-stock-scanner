import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import time
# התיקון כאן: הוספת as_completed בחזרה לייבוא
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib

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
if 'scan_status_message' not in st.session_state: # שינוי שם כדי לא להתנגש עם st.status
    st.session_state.scan_status_message = "מוכן לסריקה"
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'scan_settings_hash' not in st.session_state:
    st.session_state.scan_settings_hash = None
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False # דגל חדש לניהול מצב סריקה

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
</style>
""", unsafe_allow_html=True)

# טעינת רשימת מניות S&P 500
@st.cache_data(ttl=86400) # שמירה במטמון ליום אחד
def load_sp500_symbols():
    try:
        # ננסה לטעון מהאינטרנט
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        symbols = df['Symbol'].tolist()
        
        # ניקוי סמלים בעייתיים אם יש (לדוגמה, 'BRK.B' הופך ל'BRK-B')
        symbols = [s.replace('.', '-') for s in symbols]
        
        # שמירת הרשימה לקובץ גיבוי מקומי
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

# פונקציה לקבלת נתונים למניה בודדת
def get_stock_data(symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if history.empty:
            return None
        
        # קבלת סקטור ותעשייה
        info = ticker.info
        sector = info.get('sector', 'לא ידוע')
        industry = info.get('industry', 'לא ידוע')
        long_name = info.get('longName', symbol)

        # חישוב שינוי באחוזים ל-20 יום
        if len(history) >= 20:
            change_20d = ((history['Close'].iloc[-1] - history['Close'].iloc[-20]) / history['Close'].iloc[-20]) * 100
        else:
            change_20d = np.nan
            
        # נפח ממוצע
        average_volume = history['Volume'].mean()

        # "AI Score" - דוגמה לחישוב ציון.
        ai_score = 0
        current_price = history['Close'].iloc[-1]
        
        # משקל לשינוי 20 יום
        if not np.isnan(change_20d):
            ai_score += change_20d * 0.7 # תורם יותר

        # משקל לשינוי 5 ימים (לרגישות קצרה)
        if len(history) >= 5:
            change_5d = ((history['Close'].iloc[-1] - history['Close'].iloc[-5]) / history['Close'].iloc[-5]) * 100
            if not np.isnan(change_5d):
                ai_score += change_5d * 0.3 # תורם פחות מ-20 יום
        
        # משקל לנפח מסחר ממוצע (עוצמה)
        if average_volume > 0:
            # נרמול נפח: כמה פעמים הנפח הממוצע גדול מ-X (למשל, מיליון מניות)
            volume_factor = min(10, average_volume / 1_000_000) # הגבל ל-10 כדי לא לתת משקל יתר
            ai_score += volume_factor * 2 # תורם נקודות על סמך נפח
            
        # להוסיף ציון בסיס כדי לא להתחיל מאפס לגמרי
        ai_score += 50 # ציון בסיס

        # לוודא שהציון נשאר בטווח הגיוני (לדוגמה, 0-100)
        ai_score = max(0, min(100, ai_score)) # מגביל את הציון בין 0 ל-100

        return {
            "Symbol": symbol,
            "Company": long_name,
            "Sector": sector,
            "Industry": industry,
            "Current Price": current_price,
            "20D Change %": change_20d,
            "Average Volume": average_volume,
            "AI Score": ai_score
        }
    except Exception as e:
        # עדיף לרשום שגיאות ללוג פנימי ולא למשתמש
        # print(f"Error fetching data for {symbol}: {e}")
        return None

# פונקציית סריקה ראשית המשתמשת ב-st.status
def run_scanner_with_status(symbols, start_date, end_date):
    st.session_state.is_scanning = True
    st.session_state.scanner_results = []
    
    total_symbols = len(symbols)
    results = []
    
    # הגדל את מספר ה-workers אם יש לך הרבה מניות
    max_workers = min(15, total_symbols) # הגבל ל-15 או למספר המניות
    
    with st.status("מתחיל סריקה...", expanded=True) as status_container:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(get_stock_data, symbol, start_date, end_date): symbol for symbol in symbols}
            
            for i, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                except Exception as exc:
                    status_container.write(f"אזהרה: {symbol} יצר חריגה ולא נסרק. ייתכן שהנתונים אינם זמינים או שיש בעיית תקשורת.")
                
                # עדכון התקדמות בתוך הסטטוס
                progress_percent = (i + 1) / total_symbols
                status_container.progress(progress_percent, text=f"סורק מניות... {i+1}/{total_symbols} ({int(progress_percent * 100)}%)")
        
        status_container.update(label="סריקה הושלמה!", state="complete", expanded=False)

    st.session_state.scanner_results = results
    st.session_state.last_scan_time = datetime.now()
    st.session_state.scan_status_message = "סריקה הושלמה!"
    st.session_state.is_scanning = False


# כותרת האפליקציה
st.title("🚀 סורק מניות S&P 500 מבוסס AI")
st.markdown("כלי זה סורק מניות מתוך רשימת S&P 500 ומספק תובנות ראשוניות על סמך ביצועים אחרונים וציון 'AI'.")

# סרגל צד לסינון
st.sidebar.header("⚙️ הגדרות סריקה")

# קלט תאריכים
today = datetime.now().date()
default_start_date = today - timedelta(days=30) # חודש אחורה כברירת מחדל
date_range_option = st.sidebar.selectbox(
    "בחר טווח תאריכים:",
    ["חודש אחרון", "3 חודשים אחרונים", "6 חודשים אחרונים", "שנה אחרונה", "הכל (מקסימום זמין)"],
    index=0 # ברירת מחדל: חודש אחרון
)

if date_range_option == "חודש אחרון":
    start_date = today - timedelta(days=30)
elif date_range_option == "3 חודשים אחרונים":
    start_date = today - timedelta(days=90)
elif date_range_option == "6 חודשים אחרונים":
    start_date = today - timedelta(days=180)
elif date_range_option == "שנה אחרונה":
    start_date = today - timedelta(days=365)
elif date_range_option == "הכל (מקסימום זמין)":
    start_date = datetime(1990, 1, 1).date() # כמעט הכל

end_date = today

st.sidebar.write(f"**טווח נבחר:** {start_date.strftime('%Y-%m-%d')} עד {end_date.strftime('%Y-%m-%d')}")

# שטח טקסט לסמלים ספציפיים
specific_symbols_input = st.sidebar.text_area("סמלים ספציפיים (הפרד בפסיק, אופציונלי):", value="")
if specific_symbols_input:
    specific_symbols = [s.strip().upper() for s in specific_symbols_input.split(',') if s.strip()]
    symbols_to_scan = [s for s in specific_symbols if s in SP500_SYMBOLS]
    if not symbols_to_scan:
        st.sidebar.warning("אף אחד מהסמלים שהוזנו אינו ברשימת S&P 500. וודא איות נכון.")
else:
    symbols_to_scan = SP500_SYMBOLS

# יצירת Hash להגדרות הסריקה
current_settings_hash = hashlib.md5(json.dumps({
    "start_date": start_date.strftime('%Y-%m-%d'),
    "end_date": end_date.strftime('%Y-%m-%d'),
    "symbols_to_scan": sorted(symbols_to_scan) # מיון כדי להבטיח עקביות ב-hash
}).encode()).hexdigest()

# בדיקה האם נדרש סריקה מחדש וחישוב הלחצן
should_scan_button_be_enabled = not st.session_state.is_scanning
button_label = "התחל סריקה חדשה"
if st.session_state.is_scanning:
    button_label = "הסריקה פועלת..."
elif st.session_state.last_scan_time and st.session_state.scan_settings_hash == current_settings_hash:
    button_label = "רענן סריקה (נתונים קיימים)" # אם ההגדרות זהות, רענן את הקיימים
    
if st.button(button_label, disabled=not should_scan_button_be_enabled):
    if not st.session_state.is_scanning:
        # סטטוס התחלתי לפני הפעלת ה-thread
        st.session_state.scan_status_message = "מתחיל סריקה..."
        st.session_state.is_scanning = True
        st.session_state.scan_settings_hash = current_settings_hash # שמור את ה-hash של ההגדרות
        
        # הפעלת הסריקה ב-main thread עם st.status
        run_scanner_with_status(symbols_to_scan, start_date, end_date)
        
        # אין צורך ב-st.rerun() כאן, st.status מטפל בזה אוטומטית.


# הצגת סטטוס כללי (לא הפרוגרס בר המפורט)
if st.session_state.is_scanning:
    st.info(f"**סטטוס:** {st.session_state.scan_status_message}")
    # הפרוגרס בר המפורט מוצג בתוך ה-st.status, לכן לא מציגים אותו כאן שוב.
else: # אם הסריקה לא פעילה
    if st.session_state.last_scan_time:
        st.info(f"**סריקה אחרונה בוצעה ב:** {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}. **סטטוס:** {st.session_state.scan_status_message}")
    else:
        st.info("כדי להתחיל, בחר את ההגדרות שלך בסרגל הצד ולחץ על 'התחל סריקה חדשה'.")


# הצגת תוצאות הסריקה
if st.session_state.scanner_results:
    st.subheader("📊 תוצאות הסריקה")

    df_results = pd.DataFrame(st.session_state.scanner_results)
    
    # טיפול בערכי NaN ב-20D Change % וב-AI Score לפני סינון/הצגה
    df_results['20D Change %'] = df_results['20D Change %'].fillna(0)
    df_results['AI Score'] = df_results['AI Score'].fillna(0)

    # **פתרון לסינון סקטורים:**
    # רשימת הסקטורים זמינה רק אחרי שיש נתונים ב-df_results
    all_sectors_from_data = sorted(list(set([s for s in df_results['Sector'].unique() if s != 'לא ידוע'])))
    selected_sectors = st.sidebar.multiselect("סנן לפי סקטור:", all_sectors_from_data, default=all_sectors_from_data)

    # סינון לפי ציון AI מינימלי
    min_ai_score = st.sidebar.slider("ציון AI מינימלי:", 0, 100, 50)

    # סינון לפי שינוי באחוזים
    min_change_percent = st.sidebar.slider("שינוי מינימלי ב-20 יום (%):", -50.0, 50.0, 0.0, step=0.1) # מאפשר ערכים עשרוניים

    # סינון לפי נפח מסחר ממוצע
    min_volume = st.sidebar.number_input("נפח מסחר ממוצע מינימלי:", min_value=0, value=1000000)
    
    # יישום סינונים
    df_filtered = df_results[
        (df_results['AI Score'] >= min_ai_score) &
        (df_results['20D Change %'] >= min_change_percent) &
        (df_results['Average Volume'] >= min_volume)
    ]
    
    if selected_sectors:
        df_filtered = df_filtered[df_filtered['Sector'].isin(selected_sectors)]

    if not df_filtered.empty:
        # מיון לפי ציון AI Score
        df_filtered = df_filtered.sort_values(by="AI Score", ascending=False).reset_index(drop=True)

        st.dataframe(df_filtered.style.format({
            "Current Price": "${:.2f}",
            "20D Change %": "{:.2f}%",
            "Average Volume": "{:,.0f}",
            "AI Score": "{:.2f}"
        }), use_container_width=True)

        # גרפים
        st.subheader("📈 הדמיית נתונים")

        # גרף פיזור: ציון AI מול שינוי 20 יום
        fig_scatter = px.scatter(df_filtered, x="AI Score", y="20D Change %",
                                       color="Sector", size="Average Volume", # גודל הנקודה לפי נפח
                                       title="ציון AI מול שינוי ב-20 יום (נפח לפי גודל)",
                                       hover_data=["Symbol", "Company", "Current Price", "Average Volume", "Industry"])
        st.plotly_chart(fig_scatter, use_container_width=True)

        # גרף עמודות: ממוצע ציון AI לסקטור
        sector_avg_score = df_filtered.groupby('Sector')['AI Score'].mean().sort_values(ascending=False).reset_index()
        fig_bar_sector = px.bar(sector_avg_score, x='Sector', y='AI Score',
                                title='ציון AI ממוצע לפי סקטור',
                                color='AI Score', color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_bar_sector, use_container_width=True)

        # גרף עמודות: 10 המובילות לפי ציון AI
        top_10_ai = df_filtered.head(10).sort_values(by='AI Score', ascending=True) # למיון לצורך גרף עמודות יפה
        fig_bar_top_10 = px.bar(top_10_ai, x='AI Score', y='Company', orientation='h',
                               title='10 המניות המובילות לפי ציון AI',
                               color='AI Score', color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_bar_top_10, use_container_width=True)


    else:
        st.warning(f"לא נמצאו מניות העומדות בקריטריונים הנבחרים. נסה לשנות את הסינון.")
        
        # הצג את ה-10 המובילות ללא קשר לסינון
        st.subheader("🔍 10 המניות המובילות (ללא סינון)")
        if not df_results.empty:
            df_top_10_all = df_results.sort_values(by="AI Score", ascending=False).head(10).reset_index(drop=True)
            st.dataframe(df_top_10_all.style.format({
                "Current Price": "${:.2f}",
                "20D Change %": "{:.2f}%",
                "Average Volume": "{:,.0f}",
                "AI Score": "{:.2f}"
            }), use_container_width=True)
        else:
            st.info("אין נתונים זמינים כדי להציג את המניות המובילות. בבקשה בצע סריקה.")

    # פונקציונליות ייצוא
    st.subheader("💾 ייצוא וניתוח")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 הורד CSV",
            data=csv,
            file_name=f"mozes_scanner_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    with col2:
        # 5 המניות המובילות
        top_5 = df_filtered.head(5)
        top_symbols = ", ".join(top_5["Symbol"].tolist())
        st.text_area("🏆 5 הסמלים המובילים (בסינון הנוכחי)", top_symbols, height=100)

    with col3:
        # סטטיסטיקות מהירות
        if not df_filtered.empty:
            st.metric("🔥 הציון הגבוה ביותר", f"{df_filtered['AI Score'].max():.2f}")
            st.metric("📈 ממוצע שינוי 20 יום", f"{df_filtered['20D Change %'].mean():.1f}%")
        else:
            st.metric("🔥 הציון הגבוה ביותר", "אין נתונים")
            st.metric("📈 ממוצע שינוי 20 יום", "אין נתונים")


else:
    st.info("כדי להתחיל, בחר את ההגדרות שלך בסרגל הצד ולחץ על 'התחל סריקה חדשה'.")
