
# AI Stock Hunter — Quant V3

גרסה חדשה של הסורק הכמותי בממשק דומה ל-AI Stock Hunter V2.

## הרצה

1. חלץ את ה-ZIP.
2. פתח Terminal בתוך התיקייה.
3. הרץ:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. ייפתח בדפדפן, בדרך כלל:
`http://localhost:8501`

## מה חדש
- ממשק Dark דומה לכלי הקודם
- שתי לשוניות: Analyze one stock / Scan watchlist
- חלון אחורה קצר: 3 חודשים / 6 חודשים / שנה
- Quant Score
- BUY / WATCH / NO SIGNAL
- Volume acceleration
- RSI / MACD / ADX / ATR
- Breakout proximity
- אישור שעתי
- Walk-forward historical hit rate
- דירוג Top opportunities

## הערה
הציון 0–100 הוא ציון איכות setup ולא "אחוז סיכוי לרווח".
Hit Rate מוצג בנפרד ורק לפי אירועים היסטוריים דומים.
