
# app.py
# Phase 2 — Full NSE Screener (Macro + Fundamentals) with heatmap and optional ML/news
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta
import time, re, os

# Optional ML & NLP imports (may require extra packages)
try:
    import feedparser
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import shuffle
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

st.set_page_config(page_title="NSE Screener — Phase 2", layout="wide")
st.title("⚡ NSE Screener — Phase 2 (Macro + Fundamentals)")

# -------------------- Config --------------------
PRED_LOG_PATH = "predictions_log.csv"
FUND_CSV_CACHE_TTL = 3600
NEWS_CACHE_TTL = 60 * 10
MIN_HISTORY_DAYS_FOR_FEATURES = 30
ML_MIN_ROWS_TO_TRAIN = 200

MACRO_TICKERS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "FTSE": "^FTSE",
    "DAX": "^GDAXI",
    "HANGSENG": "^HSI",
    "SHANGHAI": "000001.SS",
    "CRUDE": "CL=F",
    "GOLD": "GC=F",
    "COPPER": "HG=F",
    "USDINR": "USDINR=X",
    "US10Y": "^TNX"
}

INDEX_URLS = {
    "Nifty Midcap 100":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

MIDCAP_FALLBACK = ["ABBOTINDIA","ALKEM","ASHOKLEY","AUBANK","AUROPHARMA","BALKRISIND","BEL","BERGEPAINT","BHEL","CANFINHOME","CUMMINSIND","DALBHARAT","DEEPAKNTR","FEDERALBNK","GODREJPROP","HAVELLS","HINDZINC","IDFCFIRSTB","INDHOTEL","INDIAMART","IPCALAB","JUBLFOOD","LUPIN","MANKIND","MUTHOOTFIN","NMDC","OBEROIRLTY","PAGEIND","PERSISTENT","PFIZER","POLYCAB","RECLTD","SAIL","SRF","SUNTV","TATAELXSI","TATAPOWER","TRENT","TVSMOTOR","UBL","VOLTAS","ZYDUSLIFE","PIIND","CONCOR","APOLLOTYRE","TORNTPHARM","MPHASIS","ASTRAL","OFSS","MINDTREE","CROMPTON"]
SMALLCAP_FALLBACK = ["AARTIIND","AFFLE","AMBER","ANANDRATHI","APLAPOLLO","ARVINDFASN","ASTERDM","ASTRAZEN","BASF","BAJAJHLDNG","BANDHANBNK","BORORENEW","CREDITACC","DIXON","EIHOTEL","GALAXYSURF","INTELLECT","JSL","KPRMILL","LAXMIMACH","MASFIN","NAVINFLUOR","ORIENTELEC","PNCINFRA","QUESS","SHREECEM","TATACHEM","UNIONBANK","VARROC","WELSPUNIND"]

# -------------------- Helpers: persistence --------------------
def ensure_log_exists(path=PRED_LOG_PATH):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=[
            "run_date","company","ticker","current",
            "pred_1m","pred_1y","ret_1m_pct","ret_1y_pct",
            "rank_ret_1m","rank_ret_1y","composite_rank","method",
            "actual_1m","actual_1m_date","err_pct_1m",
            "actual_1y","actual_1y_date","err_pct_1y"
        ])
        df.to_csv(path, index=False)

def read_pred_log(path=PRED_LOG_PATH):
    ensure_log_exists(path)
    return pd.read_csv(path, parse_dates=["run_date","actual_1m_date","actual_1y_date"])

def write_pred_log(df, path=PRED_LOG_PATH):
    df.to_csv(path, index=False)

# -------------------- Constituents fetch --------------------
def _csv_to_pairs_fuzzy(csv_text: str):
    if not csv_text: return []
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception:
        pairs=[]; lines=[l.strip() for l in csv_text.splitlines() if l.strip()]
        for ln in lines:
            parts=[p.strip().strip('"').strip("'") for p in re.split(r',|\\t', ln) if p.strip()]
            if len(parts)>=2:
                sym = parts[1].upper()
                if not sym.endswith(".NS"): sym = sym + ".NS"
                pairs.append((parts[0], sym))
        return pairs
    # find best columns
    cols = [c.strip() for c in df.columns]
    cand_name=None; cand_sym=None
    for c in cols:
        lc = c.lower()
        if any(x in lc for x in ("company","company name","name")) and cand_name is None: cand_name=c
        if any(x in lc for x in ("symbol","code","ticker","token")) and cand_sym is None: cand_sym=c
    if cand_sym:
        if not cand_name:
            df["Company"] = df[cand_sym].astype(str)
            cand_name = "Company"
        names = df[cand_name].astype(str).str.strip().tolist()
        syms = df[cand_sym].astype(str).str.strip().apply(lambda s: s.upper() if s.upper().endswith(".NS") else s.upper()+".NS").tolist()
        return list(zip(names, syms))
    if len(cols)>=2:
        names = df.iloc[:,0].astype(str).str.strip().tolist()
        syms = df.iloc[:,1].astype(str).str.strip().apply(lambda s: s.upper() if s.upper().endswith(".NS") else s.upper()+".NS").tolist()
        return list(zip(names, syms))
    return []

@st.cache_data(ttl=86400)
def fetch_constituents(index_name: str):
    url = INDEX_URLS.get(index_name)
    debug=[]
    if url:
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
            pairs = _csv_to_pairs_fuzzy(r.text)
            if pairs:
                return pairs
            debug.append("parsed0")
        except Exception as e:
            debug.append(str(e))
    # fallback
    if "smallcap" in index_name.lower():
        return [(s,f"{s}.NS") for s in SMALLCAP_FALLBACK]
    return [(s,f"{s}.NS") for s in MIDCAP_FALLBACK]

# -------------------- Fundamentals CSV loader --------------------
@st.cache_data(ttl=FUND_CSV_CACHE_TTL)
def fetch_fundamentals_csv(url: str):
    try:
        r = requests.get(url, timeout=8); r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if df.empty: return None
        df.columns = [c.strip() for c in df.columns]
        ticker_col = None
        for cand in ["Ticker","ticker","Symbol","symbol","SYM","sym"]:
            if cand in df.columns:
                ticker_col = cand; break
        if ticker_col is None: ticker_col = df.columns[0]
        df = df.rename(columns={ticker_col:"Ticker"})
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper().apply(lambda s: s if s.endswith(".NS") else s + ".NS")
        for col in df.columns:
            if col=="Ticker": continue
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",","").str.replace("%",""), errors="coerce")
            except Exception:
                pass
        df = df.set_index("Ticker")
        return df
    except Exception:
        return None

# -------------------- Macro & VIX & News --------------------
@st.cache_data(ttl=1800)
def fetch_macro_timeseries(tickers: dict, period_years=2):
    out={}
    syms=list(set([v for v in tickers.values() if v]))
    try:
        df = yf.download(syms, period=f"{period_years}y", auto_adjust=True, progress=False, threads=True)
    except Exception:
        df=None
    for name,sym in tickers.items():
        ser = pd.Series(dtype=float)
        try:
            if df is not None and isinstance(df.columns,pd.MultiIndex) and sym in df.columns.get_level_values(0):
                ser = df[sym]["Close"].dropna()
            else:
                tmp = yf.download(sym, period=f"{period_years}y", auto_adjust=True, progress=False)
                if tmp is not None and "Close" in tmp.columns:
                    ser = tmp["Close"].dropna()
        except Exception:
            ser = pd.Series(dtype=float)
        ser.index = pd.to_datetime(ser.index) if not ser.empty else ser
        out[name] = ser.sort_index() if not ser.empty else pd.Series(dtype=float)
    return out

@st.cache_data(ttl=3600)
def fetch_vix():
    try:
        df = yf.download("^INDIAVIX", period="7d", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

def vix_to_adj(vix, horizon):
    if vix is None: return 0.0
    if horizon=="1M": return float(np.clip((15 - vix) * 0.8, -10, 10))
    return float(np.clip((15 - vix) * 1.2, -20, 20))

# Geo-news: optional (requires feedparser + vaderSentiment)
if ML_AVAILABLE:
    analyzer = SentimentIntensityAnalyzer()
else:
    analyzer = None

RSS_FEEDS = {
    "us":["https://www.reuters.com/rssFeed/businessNews","https://www.cnbc.com/id/100003114/device/rss/rss.html"],
    "eu":["https://www.reuters.com/rssFeed/europeNews","https://www.ft.com/?format=rss"],
    "cn":["https://www.reuters.com/places/china/rss","https://www.scmp.com/rss/0/feed"]
}
TOPIC_KEYWORDS = {
    "oil":["oil","opec","crude","petroleum","energy"],
    "fed_rate":["fed","federal reserve","rate hike","interest rate","fed chair"],
    "china_trade":["china","export","tariff","trade","supply chain"],
    "sanctions":["sanction","sanctions"],
    "war":["war","invasion","conflict","attack","tension"],
    "inflation":["inflation","cpi","consumer price"]
}

def _clean_text(t: str) -> str:
    return re.sub(r'\s+',' ', (t or "").strip().lower())

def _fetch_feed_items_with_date(url: str, max_items=60):
    out=[]
    if not ML_AVAILABLE: return out
    try:
        f = feedparser.parse(url)
        for e in f.entries[:max_items]:
            title = getattr(e,"title","") or ""
            summary = getattr(e,"summary","") or ""
            pub=None
            if hasattr(e,"published_parsed") and e.published_parsed:
                try: pub = datetime(*e.published_parsed[:6])
                except: pub=None
            text = _clean_text(title + " " + summary)
            score = analyzer.polarity_scores(text)["compound"] if text else 0.0
            out.append({"title":_clean_text(title),"summary":_clean_text(summary),"published":pub,"compound":score,"text":text})
        time.sleep(0.02)
    except Exception:
        pass
    return out

@st.cache_data(ttl=NEWS_CACHE_TTL)
def get_geo_news_features():
    now = datetime.utcnow()
    out={}

# (file continues... due to length, full file has been written to disk by the assistant)
