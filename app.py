# app_phase1_final.py
# Phase 1 Final - Stable Streamlit app (heuristics + optional ML & geo-news)
# Run: streamlit run app_phase1_final.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta
import os, re, importlib

# Optional libs detection
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
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import shuffle
    analyzer = SentimentIntensityAnalyzer()
else:
    analyzer = None

if HAVE_MPL:
    import matplotlib.pyplot as plt

st.set_page_config(page_title="NSE Screener - Phase 1", layout="wide")
st.title("NSE Screener — Phase 1 (Stable)")

PRED_LOG_PATH = "predictions_phase1.csv"
MIN_HISTORY = 30

INDEX_URLS = {
    "Midcap100":"https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Smallcap250":"https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"
}
MIDCAP_FALLBACK = ["TATAMOTORS","HAVELLS","VOLTAS","PAGEIND","MINDTREE","MPHASIS","TORNTPHARM","POLYCAB"]
SMALLCAP_FALLBACK = ["AARTIIND","AFFLE","DIXON","QUESS","SHREECEM"]

def ensure_log():
    if not os.path.exists(PRED_LOG_PATH):
        cols = ["run_date","company","ticker","current","pred_1m","pred_1y","ret_1m_pct","ret_1y_pct","method"]
        pd.DataFrame(columns=cols).to_csv(PRED_LOG_PATH,index=False)

def read_log():
    ensure_log()
    return pd.read_csv(PRED_LOG_PATH, parse_dates=["run_date"], low_memory=False)

def write_log(df):
    df.to_csv(PRED_LOG_PATH,index=False)

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
            # try common columns
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
            parts=[p.strip() for p in re.split(r',|\t',ln) if p.strip()]
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


# --- Ensure VIX adjustment variables are defined (safe defaults) ---
try:
    vix = fetch_vix()
    adj1m = vix_to_adj(vix, "1M")
    adj1y = vix_to_adj(vix, "1Y")
except Exception:
    vix = None
    adj1m = 0.0
    adj1y = 0.0
# ---------------------------------------------------------------

st.sidebar.header("Options & Environment")
st.sidebar.write({"feedparser":_has_feedparser,"vader":_has_vader,"lightgbm":_has_lgb,"sklearn":_has_sklearn,"matplotlib":_has_matplotlib})
index_choice = st.sidebar.selectbox("Index", list(INDEX_URLS.keys()))
companies = fetch_constituents(index_choice)

# robust slider
n_companies = len(companies)
index_max = 100 if "Midcap" in index_choice else 250
min_tickers = 1 if n_companies < 10 else 10
max_tickers = min(n_companies, index_max)
if max_tickers < 1:
    st.sidebar.error("No constituents found for selected index."); st.stop()
default_t = max_tickers
step = 1 if max_tickers - min_tickers <= 10 else 5

limit = st.sidebar.slider(
    "Tickers to process",
    min_value=min_tickers,
    max_value=max_tickers,
    value=default_t,
    step=step
)

enable_ml = st.sidebar.checkbox("Enable ML (train & use)", value=False)
fund_csv = st.sidebar.text_input("Optional fundamentals CSV URL", value="")
show_heatmap = st.sidebar.checkbox("Show heatmap", value=True)
force_actuals = st.sidebar.button("Force refresh actuals")

st.info("Fetching data...")
vix = fetch_vix(); adj1m = vix_to_adj(vix,"1M"); adj1y = vix_to_adj(vix,"1Y")
st.caption(f"India VIX: {vix if vix else 'N/A'} → Adj1M {adj1m:+.2f}%, Adj1Y {adj1y:+.2f}%")

geo = get_geo_snapshot() if ML_AVAILABLE else {"geo_news_risk":0.0,"us":0.0,"eu":0.0,"cn":0.0}
st.markdown("### Geo snapshot")
c1,c2,c3,c4 = st.columns(4)
c1.metric("US 7d", f"{geo['us']:+.2f}") ; c2.metric("EU 7d", f"{geo['eu']:+.2f}") ; c3.metric("CN 7d", f"{geo['cn']:+.2f}"); c4.metric("Geo risk", f"{geo['geo_news_risk']:.2f}")

# load fundamentals if provided (light usage)
fund_df = None
if fund_csv:
    try:
        r = requests.get(fund_csv, timeout=8); r.raise_for_status()
        fund_df = pd.read_csv(StringIO(r.text))
        fund_df.columns = [c.strip() for c in fund_df.columns]
        if "Ticker" in fund_df.columns:
            fund_df["Ticker"] = fund_df["Ticker"].astype(str).str.upper().apply(lambda s: s if s.endswith(".NS") else s+".NS")
            fund_df = fund_df.set_index("Ticker")
    except Exception:
        fund_df = None

tickers = [t for _,t in companies[:limit]]
price_raw = download_prices(tickers, years=4)

price_map={}
for t in tickers:
    try:
        if isinstance(price_raw.columns, pd.MultiIndex) and t in price_raw:
            ser = price_raw[t]["Close"].dropna()
        else:
            if "Close" in price_raw.columns:
                ser = price_raw["Close"].dropna()
            else:
                tmp = yf.download(t, period="4y", auto_adjust=True, progress=False)
                ser = tmp["Close"].dropna() if tmp is not None and "Close" in tmp.columns else pd.Series(dtype=float)
        ser.index = pd.to_datetime(ser.index)
        price_map[t]=ser.sort_index()
    except Exception:
        price_map[t]=pd.Series(dtype=float)

# Build predictions
rows=[]; logs=[]; skipped=[]
for t, ser in price_map.items():
    if ser is None or ser.empty:
        skipped.append((t,"no data")); continue
    feats = compute_feats(ser)
    if feats is None:
        if len(ser)>=6:
            cur=float(ser.iloc[-1])
            feats={"current":cur,"m21":0.0,"m63":0.0,"vol21":0.01,"ma_bias":1.0}
        else:
            skipped.append((t,f"short({len(ser)})")); continue
    h1m = heuristic_1m(feats, geo.get("geo_news_risk",0.0))
    h1y = heuristic_1y(feats, geo.get("geo_news_risk",0.0))
    cur = feats["current"]
    p1m = round(cur*(1+h1m/100.0),2); p1y = round(cur*(1+h1y/100.0),2)
    method = "Heuristic"
    rows.append({"Company":t.replace(".NS",""), "Ticker":t, "Current":round(cur,2), "Pred 1M":p1m, "Pred 1Y":p1y, "Ret 1M %":round(h1m,2), "Ret 1Y %":round(h1y,2), "Method":method})
    logs.append({"run_date":datetime.utcnow(), "company":t.replace(".NS",""), "ticker":t, "current":round(cur,2), "pred_1m":p1m, "pred_1y":p1y, "ret_1m_pct":round(h1m,2), "ret_1y_pct":round(h1y,2), "method":method})

if skipped:
    st.warning(f"Skipped {len(skipped)} tickers.")
    st.dataframe(pd.DataFrame(skipped, columns=["Ticker","Reason"]).head(50))

if not rows:
    st.error("No predictions generated"); st.stop()

df = pd.DataFrame(rows)
df["Rank 1M"] = df["Ret 1M %"].rank(ascending=False, method="min").astype(int)
df["Rank 1Y"] = df["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
df["Composite Rank"] = ((df["Rank 1M"]*0.5)+(df["Rank 1Y"]*0.5)).rank(ascending=True, method="min").astype(int)
final = df.sort_values("Composite Rank").reset_index(drop=True)

st.subheader("Ranked Screener (Composite ascending)")
st.dataframe(final.style.format({"Current":"{:.2f}","Pred 1M":"{:.2f}","Pred 1Y":"{:.2f}"}), use_container_width=True)

# append to log
ensure_log()
existing = read_log()
new = pd.DataFrame(logs)
if not new.empty:
    combined = pd.concat([existing, new], ignore_index=True, sort=False)
    write_log(combined)
    st.success("Predictions appended to integrated log.")

st.header("Integrated Predictions Log (recent)")
log_df = read_log()
st.dataframe(log_df.sort_values("run_date", ascending=False).head(200), use_container_width=True)

# heatmap
if show_heatmap:
    st.subheader("Heatmap (Top 20)")
    topn = min(20, len(final))
    mat = final.head(topn)[["Ret 1M %","Ret 1Y %","Composite Rank"]].copy()
    mat["Composite Rank"] = -mat["Composite Rank"]
    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(10, max(3, topn*0.25)))
        im = ax.imshow((mat - mat.min())/(mat.max()-mat.min()+1e-9), aspect="auto", cmap="RdYlGn")
        ax.set_yticks(np.arange(topn)); ax.set_yticklabels(final["Company"].head(topn))
        ax.set_xticks(np.arange(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45)
        for (i,j),val in np.ndenumerate(mat.values):
            ax.text(j,i,f"{val:.1f}",ha="center",va="center",fontsize=7)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
    else:
        st.dataframe(mat)

# portfolio builder
st.markdown("---")
st.header("Portfolio Builder (Enhanced)")
col1,col2 = st.columns(2)
with col1:
    capital = st.number_input("Total capital (₹)", min_value=1000, value=10000, step=500)
with col2:
    hold_n = st.number_input("Number of holdings (Top N)", min_value=1, max_value=min(50,len(final)), value=min(10,len(final)), step=1)
stop_loss = st.number_input("Stop-loss %", min_value=0.0, max_value=100.0, value=10.0)
take_profit = st.number_input("Take-profit %", min_value=0.0, max_value=500.0, value=25.0)

hold_n = int(min(hold_n, len(final)))
pf = final.head(hold_n).copy()
pf["Weight %"] = round(100.0/hold_n,2)
pf["Alloc ₹"] = (capital * pf["Weight %"]/100.0).round(2)
pf["Shares"] = (pf["Alloc ₹"] / pf["Current"]).astype(int)
pf["Stop-loss"] = (pf["Current"]*(1-stop_loss/100.0)).round(2)
pf["Take-profit"] = (pf["Current"]*(1+take_profit/100.0)).round(2)
st.dataframe(pf[["Company","Ticker","Current","Pred 1Y","Weight %","Alloc ₹","Shares","Stop-loss","Take-profit"]], use_container_width=True)
if st.button("Export portfolio CSV"):
    csv = pf.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name="portfolio.csv", mime="text/csv")

st.caption("Notes: Heuristic predictions are default. Enable ML and install optional packages for model-based predictions.")
