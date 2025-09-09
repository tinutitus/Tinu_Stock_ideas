# app_phase1_final.py
# NSE Screener — Phase 1 Final (Stable Baseline)
# Heuristic predictions with optional ML and geo-news sentiment
# Run: streamlit run app_phase1_final.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta
import os, re, importlib

# Optional libraries detection
_has_feedparser = importlib.util.find_spec("feedparser") is not None
_has_vader = importlib.util.find_spec("vaderSentiment") is not None
_has_lgb = importlib.util.find_spec("lightgbm") is not None
_has_sklearn = importlib.util.find_spec("sklearn") is not None
_has_matplotlib = importlib.util.find_spec("matplotlib") is not None

ML_AVAILABLE = _has_lgb and _has_sklearn and _has_feedparser and _has_vader
HAVE_MPL = _has_matplotlib

if ML_AVAILABLE:
    import feedparser
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
else:
    analyzer = None

if HAVE_MPL:
    import matplotlib.pyplot as plt

st.set_page_config(page_title="NSE Screener - Phase 1", layout="wide")
st.title("NSE Screener — Phase 1 (Stable Baseline)")

# === Utility: Logging ===
PRED_LOG_PATH = "predictions_phase1.csv"
def ensure_log():
    if not os.path.exists(PRED_LOG_PATH):
        cols = ["run_date","company","ticker","current","pred_1m","pred_1y","ret_1m_pct","ret_1y_pct","method"]
        pd.DataFrame(columns=cols).to_csv(PRED_LOG_PATH,index=False)
def read_log():
    ensure_log()
    return pd.read_csv(PRED_LOG_PATH, parse_dates=["run_date"], low_memory=False)
def write_log(df):
    df.to_csv(PRED_LOG_PATH,index=False)

# === Index setup ===
INDEX_URLS = {
    "Midcap100":"https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Smallcap250":"https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"
}
MIDCAP_FALLBACK = ["TATAMOTORS","HAVELLS","VOLTAS","PAGEIND","MINDTREE","MPHASIS"]
SMALLCAP_FALLBACK = ["AARTIIND","AFFLE","DIXON","QUESS","SHREECEM"]

@st.cache_data(ttl=86400)
def fetch_constituents(key):
    url = INDEX_URLS.get(key)
    if not url:
        return [(s,s+".NS") for s in MIDCAP_FALLBACK]
    try:
        r = requests.get(url, timeout=8)
        pairs = []
        try:
            df = pd.read_csv(StringIO(r.text))
            name_col = None; sym_col=None
            for c in df.columns:
                lc=c.lower()
                if "company" in lc or "name" in lc: name_col=c
                if "symbol" in lc or "code" in lc or "ticker" in lc: sym_col=c
            if name_col is None: name_col=df.columns[0]
            if sym_col is None and len(df.columns)>1: sym_col=df.columns[1]
            for _,row in df.iterrows():
                nm=str(row[name_col]).strip(); sym=str(row[sym_col]).strip().upper()
                if not sym.endswith(".NS"): sym = sym + ".NS"
                pairs.append((nm,sym))
            if pairs: return pairs
        except Exception:
            pass
        for ln in r.text.splitlines():
            parts=[p.strip() for p in re.split(r',|	',ln) if p.strip()]
            if len(parts)>=2:
                sym=parts[1].upper()
                if not sym.endswith(".NS"): sym=sym+".NS"
                pairs.append((parts[0],sym))
        if pairs: return pairs
    except Exception:
        pass
    return [(s,s+".NS") for s in MIDCAP_FALLBACK]

@st.cache_data(show_spinner=False)
def download_prices(tickers, years=4):
    return yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False, threads=True, group_by="ticker")

@st.cache_data(ttl=3600)
def fetch_vix():
    try:
        df = yf.download("^INDIAVIX", period="7d", interval="1d", progress=False, auto_adjust=True)
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

def vix_to_adj(vix, horizon):
    if vix is None: return 0.0
    if horizon=="1M": return float(np.clip((15 - vix) * 0.8, -10, 10))
    return float(np.clip((15 - vix) * 1.2, -20, 20))

# === Feature extraction ===
MIN_HISTORY = 30
def compute_feats(s):
    s = s.dropna().astype(float)
    if len(s) < MIN_HISTORY: return None
    cur=float(s.iloc[-1])
    def mom(n):
        if len(s)<=n: return np.nan
        return (s.iloc[-1]/s.iloc[-n]-1.0)*100.0
    m21=mom(21); m63=mom(63)
    vol21 = s.pct_change().rolling(21).std().iloc[-1] if len(s)>1 else 0.0
    ma_short = s.rolling(20).mean().iloc[-1] if len(s)>=20 else np.nan
    ma_long = s.rolling(50).mean().iloc[-1] if len(s)>=50 else np.nan
    ma_bias = 1.0 if (not np.isnan(ma_short) and not np.isnan(ma_long) and ma_short>ma_long) else -1.0
    return {"current":cur,"m21":m21,"m63":m63,"vol21":vol21,"ma_bias":ma_bias}

def heuristic_1m(feats, geo_risk=0.0):
    base = 0.6*(feats.get("m21") or 0.0) + 0.4*(feats.get("m63") or 0.0)
    adj = 1.0
    if feats.get("ma_bias",1)<0: adj*=0.8
    out = base*adj - 100*(feats.get("vol21",0.0)) - 0.8*geo_risk
    return float(np.clip(out, -50,50))

def heuristic_1y(feats, geo_risk=0.0):
    m63 = feats.get("m63") or 0.0
    m252 = feats.get("m252") if "m252" in feats else m63*4
    out = 0.3*m63 + 0.7*m252 - 150*(feats.get("vol21",0.0)) - 0.8*geo_risk
    return float(np.clip(out, -80,120))

# === Geo sentiment snapshot (optional) ===
@st.cache_data(ttl=600)
def get_geo_snapshot():
    if not ML_AVAILABLE:
        return {"geo_news_risk":0.0,"us":0.0,"eu":0.0,"cn":0.0}
    items=[]; now=datetime.utcnow()
    for region, feeds in [("us",["https://www.reuters.com/rssFeed/businessNews"]), ("eu",["https://www.reuters.com/rssFeed/europeNews"]), ("cn",["https://www.reuters.com/places/china/rss"])]:
        for feed in feeds:
            try:
                parsed = feedparser.parse(feed)
                for e in parsed.entries[:50]:
                    txt = (getattr(e,"title","") + " " + getattr(e,"summary","")).lower()
                    pub=None
                    if hasattr(e,"published_parsed") and e.published_parsed:
                        try: pub=datetime(*e.published_parsed[:6])
                        except: pub=None
                    score = analyzer.polarity_scores(txt)["compound"] if txt else 0.0
                    items.append({"region":region,"score":score,"pub":pub})
            except Exception:
                continue
    out={"geo_news_risk":0.0,"us":0.0,"eu":0.0,"cn":0.0}
    for region in ["us","eu","cn"]:
        sel=[it for it in items if it["region"]==region and (it["pub"] is None or it["pub"]>= now - timedelta(days=7))]
        out[region] = float(np.mean([s["score"] for s in sel])) if sel else 0.0
    neg = sum(max(0,-out[r]) for r in ["us","eu","cn"])
    out["geo_news_risk"] = float(min(5.0, neg))
    return out

# === Sidebar ===
st.sidebar.header("Options & Environment")
st.sidebar.write({"feedparser":_has_feedparser,"vader":_has_vader,"lightgbm":_has_lgb,"sklearn":_has_sklearn,"matplotlib":_has_matplotlib})

index_choice = st.sidebar.selectbox("Index", list(INDEX_URLS.keys()))
companies = fetch_constituents(index_choice)

# robust slider
n_companies = len(companies)
index_max = 100 if "Midcap" in index_choice else 250
min_tickers = 1 if n_companies < 10 else 10
max_tickers = min(n_companies, index_max)
default_t = max_tickers
step = 1 if max_tickers - min_tickers <= 10 else 5
limit = st.sidebar.slider("Tickers to process", min_value=min_tickers, max_value=max_tickers, value=default_t, step=step)

enable_ml = st.sidebar.checkbox("Enable ML (train & use)", value=False)
show_heatmap = st.sidebar.checkbox("Show heatmap", value=True)

# === Main info ===
vix = fetch_vix()
adj1m = vix_to_adj(vix,"1M")
adj1y = vix_to_adj(vix,"1Y")
st.caption(f"India VIX: {vix if vix else 'N/A'} → Adj1M {adj1m:+.2f}%, Adj1Y {adj1y:+.2f}%")

geo = get_geo_snapshot() if ML_AVAILABLE else {"geo_news_risk":0.0,"us":0.0,"eu":0.0,"cn":0.0}
st.markdown("### Geo snapshot")
c1,c2,c3,c4 = st.columns(4)
c1.metric("US 7d", f"{geo['us']:+.2f}") ; c2.metric("EU 7d", f"{geo['eu']:+.2f}") ; c3.metric("CN 7d", f"{geo['cn']:+.2f}"); c4.metric("Geo risk", f"{geo['geo_news_risk']:.2f}")

# === Download price data ===
tickers = [t for _,t in companies[:limit]]
price_raw = download_prices(tickers, years=4)
price_map={}
for t in tickers:
    try:
        if isinstance(price_raw.columns, pd.MultiIndex) and t in price_raw:
            ser = price_raw[t]["Close"].dropna()
        else:
            ser = price_raw["Close"].dropna() if "Close" in price_raw.columns else pd.Series(dtype=float)
        ser.index = pd.to_datetime(ser.index)
        price_map[t]=ser.sort_index()
    except Exception:
        price_map[t]=pd.Series(dtype=float)

# === Predictions ===
rows=[]; logs=[]
for t, ser in price_map.items():
    if ser is None or ser.empty: continue
    feats = compute_feats(ser)
    if feats is None: continue
    h1m = heuristic_1m(feats, geo.get("geo_news_risk",0.0))
    h1y = heuristic_1y(feats, geo.get("geo_news_risk",0.0))
    cur = feats["current"]
    p1m = round(cur*(1+h1m/100.0),2); p1y = round(cur*(1+h1y/100.0),2)
    method = "Heuristic"
    rows.append({"Company":t.replace(".NS",""), "Ticker":t, "Current":round(cur,2), "Pred 1M":p1m, "Pred 1Y":p1y, "Ret 1M %":round(h1m,2), "Ret 1Y %":round(h1y,2), "Method":method})
    logs.append({"run_date":datetime.utcnow(), "company":t.replace(".NS",""), "ticker":t, "current":round(cur,2), "pred_1m":p1m, "pred_1y":p1y, "ret_1m_pct":round(h1m,2), "ret_1y_pct":round(h1y,2), "method":method})

df = pd.DataFrame(rows)
df["Rank 1M"] = df["Ret 1M %"].rank(ascending=False, method="min").astype(int)
df["Rank 1Y"] = df["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
df["Composite Rank"] = ((df["Rank 1M"]*0.5)+(df["Rank 1Y"]*0.5)).rank(ascending=True, method="min").astype(int)
final = df.sort_values("Composite Rank").reset_index(drop=True)

st.subheader("Ranked Screener (Composite ascending)")
st.dataframe(final, use_container_width=True)

# === Log ===
ensure_log()
existing = read_log()
new = pd.DataFrame(logs)
if not new.empty:
    combined = pd.concat([existing, new], ignore_index=True, sort=False)
    write_log(combined)
st.header("Integrated Predictions Log (recent)")
log_df = read_log()
st.dataframe(log_df.sort_values("run_date", ascending=False).head(200), use_container_width=True)

# === Heatmap ===
if show_heatmap and HAVE_MPL and not final.empty:
    st.subheader("Heatmap (Top 20)")
    topn = min(20, len(final))
    mat = final.head(topn)[["Ret 1M %","Ret 1Y %","Composite Rank"]].copy()
    mat["Composite Rank"] = -mat["Composite Rank"]
    fig, ax = plt.subplots(figsize=(10, max(3, topn*0.25)))
    im = ax.imshow((mat - mat.min())/(mat.max()-mat.min()+1e-9), aspect="auto", cmap="RdYlGn")
    ax.set_yticks(np.arange(topn)); ax.set_yticklabels(final["Company"].head(topn))
    ax.set_xticks(np.arange(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45)
    for (i,j),val in np.ndenumerate(mat.values):
        ax.text(j,i,f"{val:.1f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

st.caption("Notes: Heuristic predictions are default. Enable ML and install optional packages for model-based predictions.")

