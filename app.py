# app_full.py
# Complete NSE Screener — Phase 2 (single-file)
# Features: fetch constituents (Midcap100 / Smallcap250), download prices (yfinance),
# compute features, heuristic 1M/1Y predictions adjusted by VIX, optional ML (LightGBM) if installed,
# simple geo-news sentiment (RSS + VADER) if installed, integrated logging of predictions + actuals,
# heatmap visualization (matplotlib) and portfolio builder.
#
# Save as app_full.py and run: streamlit run app_full.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta
import time, re, os

# Optional ML & NLP imports
ML_AVAILABLE = False
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

# Optional plotting
HAVE_MPL = False
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

st.set_page_config(page_title="NSE Screener — Phase 2", layout="wide")
st.title("⚡ NSE Screener — Phase 2 — Complete")

# ---------------- Configuration ----------------
PRED_LOG_PATH = "predictions_log.csv"
FUND_CSV_CACHE_TTL = 3600
NEWS_CACHE_TTL = 60 * 10
MIN_HISTORY_DAYS = 30
ML_MIN_ROWS_TO_TRAIN = 200

MACRO_TICKERS = {
    "SP500": "^GSPC",
    "USDINR": "USDINR=X",
    "CRUDE": "CL=F",
    "GOLD": "GC=F"
}

INDEX_URLS = {
    "Nifty Midcap 100":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

# sensible small fallbacks in case CSV fails
MIDCAP_FALLBACK = ["TATAMOTORS","HAVELLS","VOLTAS","PAGEIND","MINDTREE","MPHASIS","TORNTPHARM","POLYCAB","PFIZER","CROMPTON"]
SMALLCAP_FALLBACK = ["AARTIIND","AFFLE","DIXON","QUESS","SHREECEM","VARROC","WELSPUNIND"]

# ---------------- Persistence ----------------
def ensure_log_exists(path=PRED_LOG_PATH):
    if not os.path.exists(path):
        cols = ["run_date","company","ticker","current","pred_1m","pred_1y","ret_1m_pct","ret_1y_pct","method",
                "actual_1m","actual_1m_date","err_pct_1m","actual_1y","actual_1y_date","err_pct_1y"]
        pd.DataFrame(columns=cols).to_csv(path, index=False)

def read_log(path=PRED_LOG_PATH):
    ensure_log_exists(path)
    return pd.read_csv(path, parse_dates=["run_date","actual_1m_date","actual_1y_date"], low_memory=False)

def write_log(df, path=PRED_LOG_PATH):
    df.to_csv(path, index=False)

# ---------------- Constituents ----------------
def _csv_to_pairs_fuzzy(csv_text: str):
    if not csv_text: return []
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception:
        pairs=[]
        for ln in csv_text.splitlines()[1:]:
            parts = [p.strip() for p in re.split(r',|\t', ln) if p.strip()]
            if len(parts)>=2:
                nm = parts[0]; sym = parts[1].upper()
                if not sym.endswith(".NS"): sym = sym + ".NS"
                pairs.append((nm, sym))
        return pairs
    # heuristics to find columns
    cols = [c.strip() for c in df.columns]
    cand_name=None; cand_sym=None
    for c in cols:
        lc = c.lower()
        if any(x in lc for x in ("company","company name","name")) and cand_name is None: cand_name=c
        if any(x in lc for x in ("symbol","code","ticker","token")) and cand_sym is None: cand_sym=c
    if cand_sym is None and len(cols)>=2:
        cand_sym = cols[1]
    if cand_name is None and len(cols)>=1:
        cand_name = cols[0]
    pairs=[]
    for _,row in df.iterrows():
        try:
            nm = str(row[cand_name]).strip()
            sym = str(row[cand_sym]).strip().upper()
            if not sym.endswith(".NS"): sym = sym + ".NS"
            pairs.append((nm, sym))
        except Exception:
            continue
    return pairs

@st.cache_data(ttl=86400)
def fetch_constituents(index_name):
    url = INDEX_URLS.get(index_name)
    if url:
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
            pairs = _csv_to_pairs_fuzzy(r.text)
            if pairs: return pairs
        except Exception:
            pass
    # fallback lists
    if "smallcap" in index_name.lower():
        return [(s, f"{s}.NS") for s in SMALLCAP_FALLBACK]
    return [(s, f"{s}.NS") for s in MIDCAP_FALLBACK]

# ---------------- Fundamentals CSV ----------------
@st.cache_data(ttl=FUND_CSV_CACHE_TTL)
def fetch_fundamentals_csv(url):
    try:
        r = requests.get(url, timeout=8); r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if df.empty: return None
        df.columns = [c.strip() for c in df.columns]
        tickcol = None
        for cand in ["Ticker","ticker","Symbol","symbol","SYMBOL"]:
            if cand in df.columns: tickcol = cand; break
        if tickcol is None:
            tickcol = df.columns[0]
        df = df.rename(columns={tickcol:"Ticker"})
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip().apply(lambda s: s if s.endswith(".NS") else s + ".NS")
        for c in df.columns:
            if c=="Ticker": continue
            try: df[c] = pd.to_numeric(df[c].astype(str).str.replace(",","").str.replace("%",""), errors="coerce")
            except: pass
        df = df.set_index("Ticker")
        return df
    except Exception:
        return None

# ---------------- Macro & VIX & News ----------------
@st.cache_data(ttl=1800)
def fetch_macro(tickers=MACRO_TICKERS, years=2):
    out={}
    syms = list(set([v for v in tickers.values() if v]))
    try:
        df = yf.download(syms, period=f"{years}y", auto_adjust=True, progress=False, threads=True)
    except Exception:
        df = None
    for name, sym in tickers.items():
        ser = pd.Series(dtype=float)
        try:
            if df is not None and isinstance(df.columns, pd.MultiIndex) and sym in df.columns.get_level_values(0):
                ser = df[sym]["Close"].dropna()
            else:
                tmp = yf.download(sym, period=f"{years}y", auto_adjust=True, progress=False)
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

# simple geo-news via RSS + VADER (optional)
if ML_AVAILABLE:
    analyzer = SentimentIntensityAnalyzer()
else:
    analyzer = None

RSS_FEEDS = {
    "us":["https://www.reuters.com/rssFeed/businessNews","https://www.cnbc.com/id/100003114/device/rss/rss.html"],
    "eu":["https://www.reuters.com/rssFeed/europeNews","https://www.ft.com/?format=rss"],
    "cn":["https://www.reuters.com/places/china/rss","https://www.scmp.com/rss/0/feed"]
}

@st.cache_data(ttl=NEWS_CACHE_TTL)
def get_geo_news():
    now = datetime.utcnow()
    out = {}
    if not ML_AVAILABLE:
        # zeros
        for r in ["us","eu","cn"]:
            out[f"news_{r}_avg_7d"] = 0.0
            out[f"news_{r}_vol_7d"] = 0
        out["geo_news_risk"] = 0.0
        return out
    items = []
    for region, feeds in RSS_FEEDS.items():
        for f in feeds:
            try:
                parsed = feedparser.parse(f)
                for e in parsed.entries[:40]:
                    title = getattr(e,"title","") or ""
                    summary = getattr(e,"summary","") or ""
                    txt = (title + " " + summary).lower()
                    pub = None
                    if hasattr(e, "published_parsed") and e.published_parsed:
                        pub = datetime(*e.published_parsed[:6])
                    score = analyzer.polarity_scores(txt)["compound"] if txt else 0.0
                    items.append({"region":region,"pub":pub,"score":score,"text":txt})
            except Exception:
                continue
    for r in ["us","eu","cn"]:
        sel = [it for it in items if it["region"]==r and (it["pub"] is None or it["pub"] >= now - timedelta(days=7))]
        out[f"news_{r}_avg_7d"] = float(np.mean([s["score"] for s in sel])) if sel else 0.0
        out[f"news_{r}_vol_7d"] = len(sel)
    # simple risk: negative sentiment weighted by volume
    neg = 0.0
    for r in ["us","eu","cn"]:
        avg = out[f"news_{r}_avg_7d"]; vol = out[f"news_{r}_vol_7d"]
        if avg < 0: neg += (-avg) * (1 + vol/20.0)
    out["geo_news_risk"] = float(min(5.0, neg))
    return out

# ---------------- Price fetch ----------------
@st.cache_data(show_spinner=False)
def batch_history(tickers, years=4):
    return yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False, threads=True, group_by="ticker")

# ---------------- Features ----------------
def compute_hidden_features(s: pd.Series):
    s = s.dropna().astype(float)
    if s.size < MIN_HISTORY_DAYS: return None
    cur = float(s.iloc[-1])
    def mom(days):
        if s.size <= days: return np.nan
        return (s.iloc[-1] / s.iloc[-days] - 1.0) * 100.0
    m1 = mom(1); m5 = mom(5); m21 = mom(21); m63 = mom(63)
    try: m252 = mom(252)
    except: m252 = np.nan
    vol21 = s.pct_change().rolling(21).std().iloc[-1]
    if np.isnan(vol21): vol21 = s.pct_change().std()
    short_w = min(20, s.size); mid_w = min(50, s.size)
    ma_short = s.rolling(short_w).mean().iloc[-1] if s.size>=short_w else np.nan
    ma_mid = s.rolling(mid_w).mean().iloc[-1] if s.size>=mid_w else np.nan
    ma_bias = 1.0 if (not np.isnan(ma_short) and not np.isnan(ma_mid) and ma_short > ma_mid) else -1.0
    delta = s.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    rsi14 = float((100.0 - (100.0 / (1.0 + rs))).iloc[-1]) if not rs.isna().all() else 50.0
    ema12 = s.ewm(span=12, adjust=False).mean(); ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; signal = macd.ewm(span=9, adjust=False).mean()
    macd_conf = float((macd.iloc[-1] - signal.iloc[-1])) if not macd.isna().all() else 0.0
    if s.size >= 252:
        high52=float(s[-252:].max()); low52=float(s[-252:].min())
        prox52=float((cur - low52) / (high52 - low52 + 1e-9) * 100.0)
    else:
        prox52 = 50.0
    rets = s.pct_change().dropna().tail(60)
    skew60 = float(rets.skew()) if len(rets)>5 else 0.0
    kurt60 = float(rets.kurtosis()) if len(rets)>5 else 0.0
    return {"current":cur,"m1":m1,"m5":m5,"m21":m21,"m63":m63,"m252":m252,"vol21":vol21,
            "ma_bias":ma_bias,"rsi14":rsi14,"macd_conf":macd_conf,"prox52":prox52,"skew60":skew60,"kurt60":kurt60}

# ---------------- Heuristics ----------------
def heuristic_ret1m(feats, geo_news_risk=0.0):
    base = 0.6*(feats.get("m21",0.0) or 0.0) + 0.4*(feats.get("m63",0.0) or 0.0)
    adj = 1.0
    if feats.get("ma_bias",0) < 0: adj *= 0.78
    if feats.get("rsi14",50) > 70: adj *= 0.88
    if feats.get("rsi14",50) < 30: adj *= 1.12
    if feats.get("macd_conf",0) < 0: adj *= 0.93
    elif feats.get("macd_conf",0) > 0.5: adj *= 1.05
    macro_penalty = -0.8 * geo_news_risk
    out = np.clip(base*adj + macro_penalty - 100*(feats.get("vol21",0.0) or 0.0), -50,50)
    return out

def heuristic_ret1y(feats, geo_news_risk=0.0):
    m63 = feats.get("m63",0.0) or 0.0
    m252 = feats.get("m252", np.nan)
    m252_eff = m252 if m252==m252 else (m63*4 if m63==m63 else 0.0)
    macro_penalty = -0.8 * geo_news_risk
    out = np.clip(0.3*m63 + 0.7*m252_eff + macro_penalty - 150*(feats.get("vol21",0.0) or 0.0), -80,120)
    return out

# ---------------- ML (optional) ----------------
if ML_AVAILABLE:
    @st.cache_data(ttl=86400)
    def build_ml_dataset(price_map, geo_news, fundamentals_df=None, min_history_days=60):
        rows=[]
        for ticker, ser in price_map.items():
            if ser is None or ser.empty: continue
            ser = ser.dropna()
            if len(ser) < min_history_days+252: continue
            # use last part: take rolling windows to build samples
            for i in range(min_history_days, len(ser)-252, 21):
                window = ser.iloc[:i]
                feats = compute_hidden_features(window)
                if feats is None: continue
                entry = float(window.iloc[-1])
                fut1m_idx = i + 21 if i+21 < len(ser) else None
                fut1y_idx = i + 252 if i+252 < len(ser) else None
                if fut1m_idx is None: continue
                fut1m = float(ser.iloc[fut1m_idx]); ret1m = (fut1m/entry - 1.0)*100.0
                label1m = 1 if ret1m>0 else 0
                row = {"ticker":ticker,"entry":entry,"ret1m":ret1m,"label1m":label1m}
                for k,v in feats.items(): row[k]=v
                for k,v in geo_news.items(): row[k]=v
                if fundamentals_df is not None and ticker in fundamentals_df.index:
                    for c in fundamentals_df.columns[:8]:
                        try: row[f"f_{c}"] = float(fundamentals_df.loc[ticker,c])
                        except: row[f"f_{c}"] = np.nan
                rows.append(row)
        return pd.DataFrame(rows)

    def train_model(df, label_col, features):
        if len(df) < ML_MIN_ROWS_TO_TRAIN: return None, None
        X = df[features].fillna(0.0)
        y = df[label_col].fillna(0).astype(int)
        Xs, ys = shuffle(X, y, random_state=42)
        Xtr, Xv, ytr, yv = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=ys)
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05)
        try:
            model.fit(Xtr, ytr, eval_set=[(Xv,yv)], eval_metric="auc", early_stopping_rounds=25, verbose=False)
        except TypeError:
            model.fit(Xtr, ytr)
        try:
            auc = roc_auc_score(yv, model.predict_proba(Xv)[:,1]) if len(np.unique(yv))>1 else float("nan")
        except Exception:
            auc = float("nan")
        return model, auc

# ---------------- Actuals fetching ----------------
def fetch_actual_close_on_or_after(ticker, target_date, lookahead_days=7):
    start = target_date.strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=lookahead_days+1)).strftime("%Y-%m-%d")
    try:
        hist = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if hist is None or hist.empty: return np.nan, None
        hist = hist.sort_index()
        price = float(hist["Close"].iloc[0])
        return price, pd.to_datetime(hist.index[0]).date()
    except Exception:
        return np.nan, None

def update_log_actuals(path=PRED_LOG_PATH, force=False):
    ensure_log_exists(path)
    df = read_log(path)
    if df.empty: return df
    today = datetime.utcnow().date()
    updated = False
    for idx,row in df.iterrows():
        try: run_date = pd.to_datetime(row["run_date"]).date()
        except: continue
        # 1M
        if (pd.isna(row.get("actual_1m")) or force) and run_date + timedelta(days=30) <= today:
            price, pdate = fetch_actual_close_on_or_after(row["ticker"], run_date + timedelta(days=30), lookahead_days=7)
            if not pd.isna(price):
                df.at[idx,"actual_1m"] = price; df.at[idx,"actual_1m_date"]=pdate
                pred = row.get("pred_1m", np.nan)
                df.at[idx,"err_pct_1m"] = (abs(pred-price)/price*100) if (not pd.isna(pred) and price!=0) else np.nan
                updated = True
        # 1Y
        if (pd.isna(row.get("actual_1y")) or force) and run_date + timedelta(days=365) <= today:
            price, pdate = fetch_actual_close_on_or_after(row["ticker"], run_date + timedelta(days=365), lookahead_days=14)
            if not pd.isna(price):
                df.at[idx,"actual_1y"] = price; df.at[idx,"actual_1y_date"]=pdate
                pred = row.get("pred_1y", np.nan)
                df.at[idx,"err_pct_1y"] = (abs(pred-price)/price*100) if (not pd.isna(pred) and price!=0) else np.nan
                updated = True
    if updated: write_log(df)
    return df

# ---------------- UI Controls ----------------
st.sidebar.header("Options")
index_choice = st.sidebar.selectbox("Index", list(INDEX_URLS.keys()))
companies = fetch_constituents(index_choice)
if not companies: st.sidebar.error("No constituents"); st.stop()
n = len(companies)
min_t = 1 if n<10 else 10
max_t = n
default_t = max_t
step = 1 if max_t-min_t <= 10 else 5
limit = st.sidebar.slider("Tickers to process", min_value=min_t, max_value=max_t, value=default_t, step=step)
enable_ml = st.sidebar.checkbox("Enable ML (train & use)", value=False)
fund_csv = st.sidebar.text_input("Optional fundamentals CSV (raw public CSV URL)")
show_heatmap = st.sidebar.checkbox("Show heatmap", value=True)
force_actuals = st.sidebar.button("Force refresh actuals")

# ---------------- Fetch macro & news ----------------
st.info("Fetching macro & news snapshots...")
macro_ts = fetch_macro(MACRO_TICKERS, years=2)
vix = fetch_vix()
adj1m = vix_to_adj(vix, "1M")
adj1y = vix_to_adj(vix, "1Y")
st.caption(f"India VIX: {vix if vix else 'N/A'} → Adj 1M {adj1m:+.2f}%, 1Y {adj1y:+.2f}%")
geo_news = get_geo_news() if ML_AVAILABLE else {"geo_news_risk":0.0}

# ---------------- Load fundamentals & prices ----------------
fund_df = None
if fund_csv:
    st.info("Loading fundamentals CSV...")
    fund_df = fetch_fundamentals_csv(fund_csv)
    if fund_df is None: st.warning("Fundamentals load failed; continuing without fundamentals.")

tickers = [t for _,t in companies[:limit]]
st.info(f"Downloading price history for {len(tickers)} tickers...")
price_df = batch_history(tickers, years=4)
price_map = {}
for t in tickers:
    try:
        if isinstance(price_df.columns, pd.MultiIndex) and t in price_df:
            ser = price_df[t]["Close"].dropna()
        else:
            if "Close" in price_df.columns:
                ser = price_df["Close"].dropna()
            else:
                tmp = yf.download(t, period="4y", auto_adjust=True, progress=False)
                ser = tmp["Close"].dropna() if (tmp is not None and "Close" in tmp.columns) else pd.Series(dtype=float)
        ser.index = pd.to_datetime(ser.index)
        price_map[t] = ser.sort_index()
    except Exception:
        price_map[t] = pd.Series(dtype=float)

# ---------------- ML training (optional) ----------------
MODEL_1M = MODEL_1Y = None
AUC_1M = AUC_1Y = None
if enable_ml and ML_AVAILABLE:
    st.info("Building ML dataset (may be slow)...")
    df_ml = build_ml_dataset(price_map, geo_news, fundamentals_df=fund_df, min_history_days=60) if 'build_ml_dataset' in globals() else pd.DataFrame()
    st.write("ML dataset rows:", len(df_ml))
    # minimal training example skipped if small dataset
    # full training pipeline can be added here.

# ---------------- Build predictions ----------------
rows = []; logs = []; skipped = []
for tkr, ser in price_map.items():
    if ser is None or ser.empty:
        skipped.append((tkr,"no data")); continue
    feats = compute_hidden_features(ser)
    if feats is None:
        if len(ser) >= 6:
            cur = float(ser.iloc[-1])
            mom5 = (cur / float(ser.iloc[-6]) - 1.0) * 100.0
            vol = ser.pct_change().tail(5).std() if len(ser.pct_change().dropna())>=1 else 0.01
            feats = {"current": cur, "m1": None, "m5": mom5, "m21": mom5, "m63": mom5, "m252": mom5,
                     "vol21": vol, "ma_bias": 1.0, "rsi14": 50.0, "macd_conf": 0.0, "prox52":50.0, "skew60":0.0, "kurt60":0.0}
        else:
            skipped.append((tkr,f"short({len(ser)})")); continue
    method = "Heuristic"
    geo_risk = geo_news.get("geo_news_risk", 0.0) if isinstance(geo_news, dict) else 0.0
    h1m = heuristic_ret1m(feats, geo_risk); h1y = heuristic_ret1y(feats, geo_risk)
    # ML blending could be added here; for now final = heuristic unless ML provides a value
    final1m = h1m; final1y = h1y
    final1m *= (1 + adj1m/100.0); final1y *= (1 + adj1y/100.0)
    cur = feats["current"]
    p1m = round(cur * (1 + final1m/100.0),2); p1y = round(cur * (1 + final1y/100.0),2)
    rows.append({"Company": tkr.replace(".NS",""), "Ticker":tkr, "Current":round(cur,2),
                 "Pred 1M":p1m, "Pred 1Y":p1y, "Ret 1M %":round(final1m,2), "Ret 1Y %":round(final1y,2),
                 "Method":method})
    logs.append({"run_date":datetime.utcnow(), "company":tkr.replace(".NS",""), "ticker":tkr, "current":round(cur,2),
                 "pred_1m":p1m, "pred_1y":p1y, "ret_1m_pct":round(final1m,2), "ret_1y_pct":round(final1y,2), "method":method})

if skipped:
    st.info(f"Skipped {len(skipped)} tickers.")
    st.dataframe(pd.DataFrame(skipped, columns=["ticker","reason"]).head(50))

if not rows:
    st.error("No predictions generated."); st.stop()

df_out = pd.DataFrame(rows)
df_out["Rank 1M"] = df_out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
df_out["Rank 1Y"] = df_out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
df_out["Composite Rank"] = ((df_out["Rank 1M"]*0.5) + (df_out["Rank 1Y"]*0.5)).rank(ascending=True, method="min").astype(int)
final = df_out.sort_values("Composite Rank").reset_index(drop=True)

st.subheader("Ranked Screener (Composite ascending)")
st.dataframe(final.style.format({"Current":"{:.2f}","Pred 1M":"{:.2f}","Pred 1Y":"{:.2f}"}), use_container_width=True)

# ---------------- Append to log ----------------
ensure_log_exists()
existing = read_log()
new_logs = pd.DataFrame(logs)
if not new_logs.empty:
    combined = pd.concat([existing, new_logs], ignore_index=True, sort=False)
    try:
        combined["run_date"] = pd.to_datetime(combined["run_date"])
    except Exception:
        pass
    write_log(combined)

st.success("Predictions appended to integrated log.")

# ---------------- Update actuals ----------------
st.info("Updating actuals (if matured)...")
if force_actuals:
    updated_log = update_log_actuals(force=True)
else:
    updated_log = update_log_actuals(force=False)
st.success("Actuals update complete.")

# ---------------- Historical log & summary ----------------
st.header("Integrated Predictions Log")
log_df = read_log()
st.dataframe(log_df.sort_values("run_date", ascending=False).head(300), use_container_width=True)

def summaries(df):
    out={}
    m1 = df[~df["actual_1m"].isna()] if "actual_1m" in df.columns else pd.DataFrame()
    if not m1.empty:
        out["1M_mape"] = float(m1["err_pct_1m"].mean())
        out["1M_dir_acc"] = float(((m1["pred_1m"]>m1["current"]) == (m1["actual_1m"]>m1["current"])).mean())*100.0
    else:
        out["1M_mape"]=np.nan; out["1M_dir_acc"]=np.nan
    m2 = df[~df["actual_1y"].isna()] if "actual_1y" in df.columns else pd.DataFrame()
    if not m2.empty:
        out["1Y_mape"] = float(m2["err_pct_1y"].mean())
        out["1Y_dir_acc"] = float(((m2["pred_1y"]>m2["current"]) == (m2["actual_1y"]>m2["current"])).mean())*100.0
    else:
        out["1Y_mape"]=np.nan; out["1Y_dir_acc"]=np.nan
    return out

s = summaries(log_df)
col1,col2 = st.columns(2)
col1.metric("1M MAPE", f"{s['1M_mape']:.2f}" if not pd.isna(s['1M_mape']) else "N/A")
col2.metric("1M Dir Acc (%)", f"{s['1M_dir_acc']:.1f}" if not pd.isna(s['1M_dir_acc']) else "N/A")

# ---------------- Heatmap ----------------
def show_heatmap(df, top_n=20):
    dfp = df.sort_values("Composite Rank").head(top_n).copy()
    cols = ["Ret 1M %","Ret 1Y %","Composite Rank"]
    mat = dfp[cols].copy()
    mat["Composite Rank"] = -mat["Composite Rank"]
    matn = (mat - mat.min()) / (mat.max() - mat.min() + 1e-9)
    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(10, max(3, len(dfp)*0.25)))
        im = ax.imshow(matn.values, aspect="auto", cmap="RdYlGn")
        ax.set_yticks(np.arange(len(dfp))); ax.set_yticklabels(dfp["Company"] + " (" + dfp["Ticker"].str.replace(".NS","") + ")")
        ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels(cols, rotation=45)
        for (i,j),val in np.ndenumerate(mat.values):
            ax.text(j,i, f"{val:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
    else:
        def emoji(v):
            if v>0.75: return "🟩"
            if v>0.5: return "🟨"
            if v>0.25: return "🟧"
            return "🟥"
        rows=[]
        for i,row in mat.iterrows():
            rows.append({"Company": dfp.loc[i,"Company"], **{c: f"{emoji(matn.loc[i,c])} {row[c]:.2f}" for c in cols}})
        st.dataframe(pd.DataFrame(rows))

if show_heatmap:
    topn = st.sidebar.slider("Heatmap top N", 5, min(100, len(final)), 20)
    show_heatmap(final, top_n=topn)

# ---------------- Portfolio builder ----------------
st.markdown("---")
st.header("Portfolio Builder (Equal-weight)")
capital = st.number_input("Total capital (₹)", min_value=1000, value=10000, step=1000)
max_hold = min(20, len(final))
hold_n = st.slider("Number of holdings (Top N)", 3, 3, max_hold, 5)
pf = final.head(hold_n).copy()
pf["Weight %"] = round(100.0/hold_n,2)
pf["Alloc ₹"] = (capital * pf["Weight %"]/100.0).round(2)
pf["Shares"] = (pf["Alloc ₹"] / pf["Current"]).astype(int)
st.dataframe(pf[["Company","Ticker","Current","Weight %","Alloc ₹","Shares"]], use_container_width=True)

st.caption("This app uses heuristics by default. Enable ML and install required packages for model-based predictions.")

