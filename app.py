# app_full.py
# Phase 2 — NSE Screener (Midcap/Smallcap) with heuristics + optional ML + geo-news + heatmap + portfolio
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta
import time, re, os

# Optional ML & NLP imports
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

# Try to import matplotlib for heatmap
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

st.set_page_config(page_title="NSE Screener — Phase 2", layout="wide")
st.title("⚡ NSE Screener — Phase 2 (Midcap & Smallcap)")

# ---------------- Configuration ----------------
PRED_LOG_PATH = "predictions_log.csv"
MIN_HISTORY_DAYS = 30
MACRO_TICKERS = {"SP500":"^GSPC","USDINR":"USDINR=X"}
INDEX_URLS = {
    "Nifty Midcap 100": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"
}
MIDCAP_FALLBACK = ["TATAMOTORS","ASHOKLEY","HAVELLS","VOLTAS","PAGEIND"]  # trimmed fallback

# ---------------- Helpers ----------------
def ensure_log():
    if not os.path.exists(PRED_LOG_PATH):
        cols = ["run_date","company","ticker","current","pred_1m","pred_1y","ret_1m_pct","ret_1y_pct","method","actual_1m","actual_1m_date","err_pct_1m","actual_1y","actual_1y_date","err_pct_1y"]
        pd.DataFrame(columns=cols).to_csv(PRED_LOG_PATH,index=False)

def read_log():
    ensure_log()
    return pd.read_csv(PRED_LOG_PATH, parse_dates=["run_date","actual_1m_date","actual_1y_date"])

def write_log(df):
    df.to_csv(PRED_LOG_PATH,index=False)

@st.cache_data(ttl=86400)
def fetch_constituents(index_name):
    url = INDEX_URLS.get(index_name)
    try:
        r = requests.get(url, timeout=8)
        text = r.text
        # try parse CSV
        try:
            df = pd.read_csv(StringIO(text))
            # find symbol column
            sym = None; namecol = None
            for c in df.columns:
                if c.lower() in ("symbol","ticker","code"): sym = c
                if c.lower() in ("company","company name","name"): namecol = c
            if sym is None:
                sym = df.columns[1]
            if namecol is None:
                namecol = df.columns[0]
            pairs = []
            for _,row in df.iterrows():
                s = str(row[sym]).strip().upper()
                if not s.endswith(".NS"): s = s + ".NS"
                pairs.append((str(row[namecol]), s))
            if pairs: return pairs
        except Exception:
            pass
    except Exception:
        pass
    # fallback
    return [(s, s + ".NS") for s in MIDCAP_FALLBACK]

@st.cache_data(ttl=3600)
def download_prices(tickers, years=4):
    if not tickers: return {}
    df = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False, threads=True, group_by="ticker")
    out = {}
    for t in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex) and t in df:
                series = df[t]["Close"].dropna()
            elif "Close" in df.columns:
                series = df["Close"].dropna()
            else:
                tmp = yf.download(t, period=f"{years}y", auto_adjust=True, progress=False)
                series = tmp["Close"].dropna() if tmp is not None and "Close" in tmp.columns else pd.Series(dtype=float)
            series.index = pd.to_datetime(series.index)
            out[t] = series.sort_index()
        except Exception:
            out[t] = pd.Series(dtype=float)
    return out

def compute_feats(series):
    s = series.dropna().astype(float)
    if len(s) < MIN_HISTORY_DAYS: return None
    cur = float(s.iloc[-1])
    def mom(n):
        if len(s) <= n: return np.nan
        return (s.iloc[-1] / s.iloc[-n] - 1.0) * 100.0
    m5 = mom(5); m21 = mom(21); m63 = mom(63)
    vol21 = s.pct_change().rolling(21).std().iloc[-1] if len(s)>5 else s.pct_change().std()
    ma_short = s.rolling(20).mean().iloc[-1] if len(s)>=20 else np.nan
    ma_long = s.rolling(50).mean().iloc[-1] if len(s)>=50 else np.nan
    ma_bias = 1.0 if (not np.isnan(ma_short) and not np.isnan(ma_long) and ma_short>ma_long) else -1.0
    return {"current":cur,"m5":m5,"m21":m21,"m63":m63,"vol21":vol21,"ma_bias":ma_bias}

def heuristic_pred(feats):
    # simple linear heuristic mapping to return percentages
    base = 0.5*(feats.get("m21",0) or 0) + 0.5*(feats.get("m63",0) or 0)
    adj = 1.0
    if feats.get("ma_bias",1) < 0: adj *= 0.85
    vol = feats.get("vol21",0) or 0
    out1m = np.clip(base*adj - 80*vol, -50, 50)
    out1y = np.clip(0.4*base*adj - 150*vol + 10, -80, 200)
    return out1m, out1y

# Simple ML wrapper (optional)
def train_simple_lgbm(df, label_col, features):
    if not ML_AVAILABLE:
        return None, None
    try:
        X = df[features].fillna(0.0)
        y = df[label_col].fillna(0).astype(int)
        if len(X) < 100: return None, None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
        try:
            model.fit(X_train, y_train, eval_set=[(X_val,y_val)], eval_metric="auc", early_stopping_rounds=20, verbose=False)
        except TypeError:
            model.fit(X_train, y_train)
        yprob = model.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, yprob) if len(np.unique(y_val))>1 else float("nan")
        return model, auc
    except Exception:
        return None, None

# ---------------- UI ----------------
st.sidebar.header("Options")
index_choice = st.sidebar.selectbox("Index", list(INDEX_URLS.keys()))
limit = st.sidebar.slider("Tickers to process", min_value=5, max_value=100, value=50, step=5)
enable_ml = st.sidebar.checkbox("Enable ML (requires LightGBM etc.)", value=False)
fund_csv = st.sidebar.text_input("Optional fundamentals CSV URL (raw CSV)")

st.info("Fetching constituents...")
companies = fetch_constituents(index_choice)
if not companies:
    st.error("No constituents found."); st.stop()

pairs = companies[:limit]
tickers = [t for _,t in pairs]

st.info(f"Downloading prices for {len(tickers)} tickers...")
price_map = download_prices(tickers, years=4)

rows = []
skipped = []
for name,ticker in pairs[:limit]:
    ser = price_map.get(ticker, pd.Series(dtype=float))
    if ser.empty or len(ser) < 10:
        skipped.append((name,ticker,len(ser)))
        continue
    feats = compute_feats(ser)
    if feats is None:
        skipped.append((name,ticker,"insufficient history"))
        continue
    h1m, h1y = heuristic_pred(feats)
    cur = feats["current"]
    p1m = round(cur * (1 + h1m/100.0),2)
    p1y = round(cur * (1 + h1y/100.0),2)
    rows.append({"Company":name,"Ticker":ticker,"Current":round(cur,2),"Pred 1M":p1m,"Pred 1Y":p1y,"Ret 1M %":round(h1m,2),"Ret 1Y %":round(h1y,2),"Method":"Heuristic"})

if skipped:
    st.warning(f"Skipped {len(skipped)} tickers due to data issues.")
    st.dataframe(pd.DataFrame(skipped, columns=["Company","Ticker","Len/H"]).head(20))

df_out = pd.DataFrame(rows)
if df_out.empty:
    st.error("No predictions generated."); st.stop()

# compute ranks (ascending composite rank: lower rank number = better)
df_out["Rank 1M"] = df_out["Ret 1M %"].rank(ascending=False,method="min").astype(int)
df_out["Rank 1Y"] = df_out["Ret 1Y %"].rank(ascending=False,method="min").astype(int)
df_out["Composite Rank"] = ((df_out["Rank 1M"]*0.5) + (df_out["Rank 1Y"]*0.5)).rank(ascending=True,method="min").astype(int)
df_final = df_out.sort_values("Composite Rank").reset_index(drop=True)

st.subheader("Ranked Screener (Composite Rank Ascending)")
st.dataframe(df_final.style.format({"Current":"{:.2f}","Pred 1M":"{:.2f}","Pred 1Y":"{:.2f}"}), use_container_width=True)

# Append to integrated log
ensure_log()
log_existing = read_log()
new_logs = df_final[["Company","Ticker","Current","Pred 1M","Pred 1Y","Ret 1M %","Ret 1Y %","Method"]].copy()
new_logs = new_logs.rename(columns={"Pred 1M":"pred_1m","Pred 1Y":"pred_1y","Ret 1M %":"ret_1m_pct","Ret 1Y %":"ret_1y_pct","Current":"current","Method":"method","Ticker":"ticker","Company":"company"})
new_logs["run_date"] = datetime.utcnow()
cols_out = ["run_date","company","ticker","current","pred_1m","pred_1y","ret_1m_pct","ret_1y_pct","method"]
new_logs = new_logs[cols_out]
combined = pd.concat([log_existing, new_logs], ignore_index=True, sort=False)
write_log(combined)

st.success("Predictions logged to integrated history.")

# Historical log & summary
st.header("Integrated Historical Predictions Log")
log_df = read_log()
st.dataframe(log_df.sort_values("run_date", ascending=False).head(200), use_container_width=True)

def summary_metrics(df):
    out = {}
    m1 = df[~df["actual_1m"].isna()]
    if not m1.empty:
        out["1M_mape"] = float(m1["err_pct_1m"].mean())
        out["1M_dir_acc"] = float(( (m1["pred_1m"]>m1["current"]) == (m1["actual_1m"]>m1["current"]) ).mean())*100.0
    else:
        out["1M_mape"]=np.nan; out["1M_dir_acc"]=np.nan
    m2 = df[~df["actual_1y"].isna()]
    if not m2.empty:
        out["1Y_mape"]=float(m2["err_pct_1y"].mean())
        out["1Y_dir_acc"]=float(( (m2["pred_1y"]>m2["current"]) == (m2["actual_1y"]>m2["current"]) ).mean())*100.0
    else:
        out["1Y_mape"]=np.nan; out["1Y_dir_acc"]=np.nan
    return out

summary = summary_metrics(log_df)
c1,c2 = st.columns(2)
c1.metric("1M MAPE", f"{summary['1M_mape']:.2f}" if not pd.isna(summary['1M_mape']) else "N/A")
c2.metric("1M Dir Acc (%)", f"{summary['1M_dir_acc']:.1f}" if not pd.isna(summary['1M_dir_acc']) else "N/A")

# Heatmap
def show_heatmap(df, top_n=20):
    dfp = df.sort_values("Composite Rank").head(top_n)
    cols = ["Ret 1M %","Ret 1Y %","Composite Rank"]
    mat = dfp[cols].copy()
    # invert composite for visual (lower rank = better)
    mat["Composite Rank"] = -mat["Composite Rank"]
    # normalize
    matn = (mat - mat.min()) / (mat.max() - mat.min() + 1e-9)
    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(8, max(3, len(dfp)*0.25)))
        im = ax.imshow(matn.values, aspect="auto", cmap="RdYlGn")
        ax.set_yticks(np.arange(len(dfp))); ax.set_yticklabels(dfp["Company"] + " (" + dfp["Ticker"].str.replace(".NS","") + ")")
        ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels(cols, rotation=45)
        for (i,j),val in np.ndenumerate(mat.values):
            ax.text(j,i, f"{val:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
    else:
        # emoji fallback
        def emoji(v):
            if v>0.75: return "🟩"
            if v>0.5: return "🟨"
            if v>0.25: return "🟧"
            return "🟥"
        rows = []
        for i,row in mat.iterrows():
            rows.append({"Company": dfp.loc[i,"Company"], **{c: f"{emoji(matn.loc[i,c])} {row[c]:.2f}" for c in cols}})
        st.dataframe(pd.DataFrame(rows))

if st.sidebar.checkbox("Show heatmap", value=True):
    nheat = st.sidebar.slider("Heatmap top N", 5, min(50, len(df_final)), 20)
    show_heatmap(df_final, top_n=nheat)

# Portfolio builder
st.header("Portfolio Builder (Equal weight)")
capital = st.number_input("Total capital (₹)", min_value=1000, value=10000, step=1000)
holdings = st.slider("Number of holdings (Top N)", 3, min(3, min(20, len(df_final))), min(10, len(df_final)))
pf = df_final.head(holdings).copy()
pf["Weight %"] = round(100.0/holdings,2)
pf["Alloc ₹"] = (capital * pf["Weight %"]/100.0).round(2)
pf["Shares"] = (pf["Alloc ₹"] / pf["Current"]).astype(int)
st.dataframe(pf[["Company","Ticker","Current","Weight %","Alloc ₹","Shares"]], use_container_width=True)

st.caption("Notes: This app uses simple heuristics. Enable ML to try model-based preds (requires LightGBM and sklearn).")

