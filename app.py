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
        matching_signals = 0  # מונה לאותות שוריים

        # 1. RSI: מכירה יתר (מתחת ל-30)
        if pd.notna(rsi) and rsi < 30:
            ai_score += 25
            matching_signals += 1

        # 2. MACD: חצייה שורית (MACD חוצה מעל Signal)
        if (pd.notna(macd) and pd.notna(signal_line) and pd.notna(macd_prev) and pd.notna(signal_prev)):
            if macd > signal_line and macd_prev <= signal_prev:  # חצייה חדשה
                ai_score += 25
                matching_signals += 1
            elif macd > signal_line:  # כבר מעל, אבל לא חצייה חדשה
                ai_score += 15
                matching_signals += 1

        # 3. Bollinger Bands: מחיר מתחת לרצועה התחתונה
        if pd.notna(current_price) and pd.notna(bb_lower) and current_price < bb_lower:
            ai_score += 25
            matching_signals += 1

        # 4. Moving Averages: חצייה שורית של MA50 מעל MA200
        if (pd.notna(sma50) and pd.notna(sma200) and pd.notna(sma50_prev) and pd.notna(sma200_prev)):
            if sma50 > sma200 and sma50_prev <= sma200_prev:  # חצייה חדשה
                ai_score += 25
                matching_signals += 1
            elif sma50 > sma200:  # כבר מעל, אבל לא חצייה חדשה
                ai_score += 15
                matching_signals += 1

        # תוספת משקל לפי נתונים נוספים
        # 5. שינוי חיובי ב-20 יום
        if pd.notna(change_20d) and change_20d > 0:
            ai_score += min(change_20d, 20)  # עד 20 נקודות נוספות

        # 6. נפח מסחר גבוה
        if average_volume > 1_000_000:  # נפח משמעותי
            ai_score += 10

        # נרמול הציון לטווח 0-100
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
    except Exception as e:
        return None
