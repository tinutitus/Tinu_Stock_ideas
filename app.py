# app.py
# Phase-1 baseline + ETMarkets + Sector-Relative + ML training + Actuals updater
# Single-file Streamlit app. Drop into your Streamlit folder and run with `streamlit run app.py`.

import streamlit as st
st.set_page_config("NSE Screener — Phase1+", layout="wide")

import pandas as pd
import numpy as np
import yfinance as yf
import requests, re, os, time, math
from io import StringIO
from datetime import datetime, timedelta

# Optional imports (safe fallback)
try:
    import feedparser
except Exception:
    feedparser = None
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except Exception:
    vader = None
# ML imports
try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import shuffle
    SKLEARN_OK = True
except Exception:
    lgb = None
    SKLEARN_OK = False

# ------------------------------
#  CONFIG (tweak here if needed)
# ------------------------------
PRED_LOG_PATH = "predictions_log.csv"
MIN_HISTORY_DAYS = 30
ML_MIN_ROWS = 200
DEFAULT_CAPITAL = 10000

# For ETMarkets picks (RSS)
ET_RSS_URL = "https://economictimes.indiatimes.com/markets/stocks/recos/rssfeeds/2146842.cms"

# Macro tickers (for small snapshot)
MACRO_TICKERS = {"SP500":"^GSPC","USDINR":"USDINR=X","CRUDE":"CL=F","US10Y":"^TNX"}

# Heuristic windows
HOLD_1M_DAYS = 21
HOLD_1Y_DAYS = 252

# ML toggle default
ENABLE_ML_DEFAULT = False

# --------------
#  Utilities
# --------------
def ensure_log_exists():
    if not os.path.exists(PRED_LOG_PATH):
        cols = ["run_date","company","ticker","current","pred_1m","pred_1y","ret_1m_pct","ret_1y_pct","method","actual_1m","actual_1m_date","err_pct_1m","actual_1y","actual_1y_date","err_pct_1y"]
        pd.DataFrame(columns=cols).to_csv(PRED_LOG_PATH, index=False)

def read_pred_log():
    ensure_log_exists()
    return pd.read_csv(PRED_LOG_PATH, parse_dates=["run_date","actual_1m_date","actual_1y_date"], infer_datetime_format=True)

def write_pred_log(df):
    df.to_csv(PRED_LOG_PATH, index=False)

# ------------------------------
#  Constituents fetch (robust)
# ------------------------------
INDEX_URLS = {
    "Nifty Midcap 100":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

@st.cache_data(ttl=86400)
def fetch_constituents(index_name):
    url = INDEX_URLS.get(index_name)
    if not url: return []
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        txt = r.text
        # try to parse common CSV with symbol column
        df = pd.read_csv(StringIO(txt))
        cand_sym = None
        cand_name = None
        for c in df.columns:
            lc=c.lower()
            if "symbol" in lc or "code" in lc or "ticker" in lc:
                cand_sym=c; break
        for c in df.columns:
            lc=c.lower()
            if "company" in lc or "name" in lc:
                cand_name=c; break
        if cand_sym:
            if not cand_name: cand_name=cand_sym
            pairs = []
            for _,row in df.iterrows():
                sym = str(row[cand_sym]).strip().upper()
                if sym and not sym.endswith(".NS"): sym = sym + ".NS"
                name = str(row[cand_name]).strip()
                pairs.append((name, sym))
            if pairs: return pairs
    except Exception:
        pass
    # fallback: return empty list (app will warn)
    return []

# ------------------------------
#  Price fetch (batched)
# ------------------------------
@st.cache_data(show_spinner=False)
def batch_history(tickers, years=4):
    if not tickers: return pd.DataFrame()
    try:
        df = yf.download(list(tickers), period=f"{years}y", auto_adjust=True, progress=False, group_by="ticker")
        return df
    except Exception:
        # fallback: download one-by-one
        out={}
        for t in tickers:
            try:
                tmp=yf.download(t, period=f"{years}y", auto_adjust=True, progress=False)
                out[t]=tmp
            except Exception:
                out[t]=pd.DataFrame()
        return out

# ------------------------------
#  Basic feature computation
# ------------------------------
def compute_hidden_features(series: pd.Series):
    s = series.dropna().astype(float)
    if s.size < MIN_HISTORY_DAYS:
        return None
    cur = float(s.iloc[-1])
    def mom(days):
        if s.size<=days: return np.nan
        return (s.iloc[-1]/s.iloc[-days]-1.0)*100.0
    m21 = mom(21); m63=mom(63)
    try: m252 = mom(252)
    except: m252=np.nan
    vol21 = float(s.pct_change().rolling(21).std().iloc[-1]) if len(s.pct_change().dropna())>1 else float(s.pct_change().std())
    # simple moving bias
    ma_short = s.rolling(20).mean().iloc[-1] if len(s)>=20 else np.nan
    ma_mid = s.rolling(50).mean().iloc[-1] if len(s)>=50 else np.nan
    ma_bias = 1.0 if (not np.isnan(ma_short) and not np.isnan(ma_mid) and ma_short>ma_mid) else -1.0
    return {"current":cur,"m21":m21,"m63":m63,"m252":m252,"vol21":vol21,"ma_bias":ma_bias}

# heuristics
def heuristic_ret1m(feats, geo_risk=0.0):
    base = 0.6*(feats.get("m21") or 0.0) + 0.4*(feats.get("m63") or 0.0)
    adj = 1.0
    if feats.get("ma_bias",1) < 0: adj *= 0.85
    if feats.get("vol21",0.0) > 0.05: adj *= 0.9
    out = np.clip(base*adj - 0.8*geo_risk - 50*(feats.get("vol21",0.0) or 0.0), -50,50)
    return out

def heuristic_ret1y(feats, geo_risk=0.0):
    base = 0.3*(feats.get("m63") or 0.0) + 0.7*(feats.get("m252") or (feats.get("m63",0.0)*4))
    out = np.clip(base - 1.2*geo_risk - 150*(feats.get("vol21",0.0) or 0.0), -80,120)
    return out

# ------------------------------
#  ETMarkets simple parser / broker picks (best-effort)
# ------------------------------
@st.cache_data(ttl=600)
def fetch_et_broker_picks():
    picks = {}
    if feedparser is None:
        return picks
    try:
        feed = feedparser.parse(ET_RSS_URL)
        for e in feed.entries[:200]:
            title = e.get("title","") or ""
            # crude heuristic: extract uppercase words as tickers (may be noisy)
            syms = re.findall(r'\b[A-Z]{2,10}\b', title)
            for s in syms:
                if len(s)>1 and not s.isdigit():
                    t = s + ".NS"
                    picks.setdefault(t, []).append("ET")
    except Exception:
        pass
    return picks

# ------------------------------
#  Sector-relative momentum
# ------------------------------
def compute_sector_rel(price_map: dict, sector_map: dict, days=HOLD_1M_DAYS):
    # price_map: ticker->Series ; sector_map: ticker->sector (ticker must match keys)
    ticker_ret = {}
    for t, ser in price_map.items():
        try:
            if ser is None or len(ser)<=days: ticker_ret[t]=np.nan; continue
            ticker_ret[t] = (float(ser.iloc[-1])/float(ser.iloc[-days]) - 1.0)*100.0
        except Exception:
            ticker_ret[t]=np.nan
    sector_avg={}
    grouped={}
    for t, r in ticker_ret.items():
        sec = sector_map.get(t, "ALL")
        grouped.setdefault(sec, []).append(r)
    for sec, lst in grouped.items():
        vals = [x for x in lst if not pd.isna(x)]
        sector_avg[sec] = float(np.mean(vals)) if vals else 0.0
    sector_rel = {}
    for t, r in ticker_ret.items():
        sec = sector_map.get(t,"ALL")
        if pd.isna(r): sector_rel[t] = np.nan
        else: sector_rel[t] = float(r - sector_avg.get(sec, 0.0))
    return sector_rel

# ------------------------------
#  ML dataset builder & training (classification for direction)
# ------------------------------
def build_ml_dataset(price_map: dict, recent_only_years=None, fundamentals_df=None, min_history_days=60):
    rows=[]
    # gather run dates as month-ends present in data
    all_dates = sorted({d for ser in price_map.values() if ser is not None and not ser.empty for d in ser.index})
    if not all_dates: return pd.DataFrame(rows)
    all_dates = pd.DatetimeIndex(all_dates)
    months = sorted({(d.year,d.month) for d in all_dates})
    run_dates=[]
    for y,m in months:
        ds = all_dates[(all_dates.year==y)&(all_dates.month==m)]
        if len(ds): run_dates.append(ds.max())
    HOLD_1M = HOLD_1M_DAYS; HOLD_1Y = HOLD_1Y_DAYS
    for run_date in run_dates:
        if recent_only_years and (datetime.today().year - run_date.year)>recent_only_years:
            continue
        for t, ser in price_map.items():
            if ser is None or ser.empty: continue
            if run_date < ser.index.min(): continue
            ser_up = ser[:run_date]
            if len(ser_up) < min_history_days: continue
            feats = compute_hidden_features(ser_up)
            if feats is None: continue
            idx = ser.index.searchsorted(run_date)
            if idx==len(ser) or ser.index[idx]!=run_date:
                if idx==0: continue
                idx = idx-1
            entry = float(ser.iloc[idx])
            fut1m_idx = idx + HOLD_1M
            fut1y_idx = idx + HOLD_1Y
            if fut1m_idx >= len(ser): continue
            future1m = float(ser.iloc[fut1m_idx])
            real_ret_1m = (future1m/entry - 1.0)*100.0
            label_1m = 1 if real_ret_1m>0 else 0
            if fut1y_idx < len(ser):
                future1y = float(ser.iloc[fut1y_idx])
                real_ret_1y = (future1y/entry - 1.0)*100.0
                label_1y = 1 if real_ret_1y>0 else 0
            else:
                label_1y = np.nan
            row = {"run_date":run_date,"ticker":t,"entry":entry,"real_ret_1m":real_ret_1m,"label_1m":label_1m,"real_ret_1y":np.nan,"label_1y":label_1y}
            # core tech features
            for k in ["m21","m63","vol21","ma_bias"]:
                row[k]=feats.get(k,np.nan)
            # add to rows
            rows.append(row)
    return pd.DataFrame(rows)

def train_lgbm(df_train, features, num_boost_round=300):
    # df_train must have "label"
    try:
        missing = [c for c in features if c not in df_train.columns]
        if missing:
            st.warning(f"train_lgbm: missing features {missing}; model may be weak.")
        X = df_train[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        y = pd.to_numeric(df_train["label"], errors="coerce").fillna(0).astype(int)
        if len(X) < ML_MIN_ROWS:
            st.warning(f"Not enough rows to train ({len(X)}). Need >= {ML_MIN_ROWS}.")
            return None, None
        Xs, ys = shuffle(X, y, random_state=42)
        Xtr, Xval, ytr, yval = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=ys)
        # try sklearn LGBM if available
        try:
            model = lgb.LGBMClassifier(objective="binary", n_estimators=num_boost_round, learning_rate=0.05,
                                       num_leaves=31, random_state=42, n_jobs=1,verbosity=-1)
            # some versions don't accept early_stopping_rounds param; wrap
            try:
                model.fit(Xtr, ytr, eval_set=[(Xval,yval)], eval_metric="auc", early_stopping_rounds=25, verbose=False)
            except TypeError:
                model.fit(Xtr, ytr)
            yproba = model.predict_proba(Xval)[:,1]
            auc = roc_auc_score(yval, yproba) if len(set(yval))>1 else float("nan")
            return model, auc
        except Exception:
            # fallback to booster api
            params = {"objective":"binary","metric":"auc","learning_rate":0.05,"num_leaves":31,"verbose":-1,"seed":42}
            dtrain = lgb.Dataset(Xtr, label=ytr); dval = lgb.Dataset(Xval, label=yval, reference=dtrain)
            booster = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=[dval], early_stopping_rounds=25, verbose_eval=False)
            yproba = booster.predict(Xval)
            auc = roc_auc_score(yval, yproba) if len(set(yval))>1 else float("nan")
            return booster, auc
    except Exception as e:
        st.error(f"train_lgbm failed: {e}")
        return None, None

def ml_predict_prob(model, feats_row, features):
    if model is None:
        raise ValueError("model is None")
    X = pd.DataFrame([feats_row])[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:,1][0])
    else:
        # booster
        return float(model.predict(X)[0])

# ------------------------------
#  Actuals updater
# ------------------------------
def fetch_actual_close_on_or_after(ticker, target_date, lookahead_days=7):
    if not ticker.endswith(".NS"):
        ticker = ticker + ".NS"
    start = target_date.strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=lookahead_days+1)).strftime("%Y-%m-%d")
    try:
        hist = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if hist is None or hist.empty: return np.nan, None
        hist = hist.sort_index()
        return float(hist["Close"].iloc[0]), pd.to_datetime(hist.index[0]).date()
    except Exception:
        return np.nan, None

def update_log_with_actuals(path=PRED_LOG_PATH, force=False):
    ensure_log_exists()
    df = read_pred_log()
    if df.empty: return df
    now = datetime.utcnow().date()
    updated = False
    for idx,row in df.iterrows():
        try:
            run_date = pd.to_datetime(row["run_date"]).date()
        except Exception:
            continue
        # 1M
        needs1m = pd.isna(row.get("actual_1m")) or force
        t1m = run_date + timedelta(days=30)
        if needs1m and t1m <= now:
            price, pdate = fetch_actual_close_on_or_after(row["ticker"], t1m, lookahead_days=14)
            if not pd.isna(price):
                df.at[idx,"actual_1m"] = price
                df.at[idx,"actual_1m_date"] = pdate
                pred = row.get("pred_1m", np.nan)
                df.at[idx,"err_pct_1m"] = (abs(pred-price)/price*100) if (not pd.isna(pred) and price!=0) else np.nan
                updated=True
        # 1Y
        needs1y = pd.isna(row.get("actual_1y")) or force
        t1y = run_date + timedelta(days=365)
        if needs1y and t1y <= now:
            price, pdate = fetch_actual_close_on_or_after(row["ticker"], t1y, lookahead_days=21)
            if not pd.isna(price):
                df.at[idx,"actual_1y"] = price
                df.at[idx,"actual_1y_date"] = pdate
                pred = row.get("pred_1y", np.nan)
                df.at[idx,"err_pct_1y"] = (abs(pred-price)/price*100) if (not pd.isna(pred) and price!=0) else np.nan
                updated=True
    if updated:
        write_pred_log(df)
    return df

# ------------------------------
#  UI - Sidebar
# ------------------------------
st.title("⚡ NSE Screener — Phase1+ (ML, SectorRel, ET Picks, Actuals)")
st.sidebar.header("Options")
index_choice = st.sidebar.selectbox("Index", list(INDEX_URLS.keys()))
companies = fetch_constituents(index_choice)

n_companies = len(companies) if companies else 0
if n_companies > 0:
    min_tickers = 1 if n_companies < 10 else 10
    max_tickers = n_companies
    default_tickers = min(50, n_companies)
    step = 1 if n_companies < 50 else 5
    limit = st.sidebar.slider("Tickers to process",
                              min_value=min_tickers,
                              max_value=max_tickers,
                              value=default_tickers,
                              step=step)
else:
    st.sidebar.warning("⚠️ No companies found for this index.")
    limit = 0
enable_ml = st.sidebar.checkbox("Enable ML training & use", value=ENABLE_ML_DEFAULT)
train_recent = st.sidebar.checkbox("Train using recent N years only (faster)", value=True)
fund_csv_url = st.sidebar.text_input("Optional fundamentals CSV URL (public raw CSV)", value="")
view_mode = st.sidebar.radio("View mode", ["Full Screener","Filtered Investable Picks"])
refresh_news = st.sidebar.button("Refresh ET Picks & News")
force_actuals = st.sidebar.button("Force update actuals")

# sector_map uploader
st.sidebar.markdown("**Sector mapping (optional)**")
st.sidebar.markdown("Upload CSV with columns: Ticker,Sector (Ticker must match e.g. TATAMOTORS.NS)")
f = st.sidebar.file_uploader("Upload sector_map.csv", type=["csv"])
sector_map = {}
if f is not None:
    try:
        dfsec = pd.read_csv(f)
        if "Ticker" in dfsec.columns and "Sector" in dfsec.columns:
            for _,r in dfsec.iterrows():
                t = str(r["Ticker"]).strip().upper()
                sector_map[t] = str(r["Sector"]).strip()
            st.sidebar.success(f"Loaded sector map for {len(sector_map)} tickers.")
        else:
            st.sidebar.error("CSV must have columns: Ticker, Sector")
    except Exception as e:
        st.sidebar.error(f"Failed to read sector CSV: {e}")

# ------------------------------
#  Macro snapshot short
# ------------------------------
try:
    macro_map = {}
    for k,v in MACRO_TICKERS.items():
        try:
            tmp = yf.download(v, period="1y", progress=False, auto_adjust=True)
            if tmp is not None and not tmp.empty:
                macro_map[k] = tmp["Close"].dropna()
        except Exception:
            macro_map[k] = pd.Series(dtype=float)
    vix_val = None
    try:
        vix = yf.download("^INDIAVIX", period="7d", progress=False, auto_adjust=True)
        if vix is not None and not vix.empty: vix_val = float(vix["Close"].iloc[-1])
    except Exception:
        vix_val = None
    st.sidebar.markdown("### Macro snapshot")
    st.sidebar.write({k:(float(v.iloc[-1]) if (isinstance(v,pd.Series) and not v.empty) else None) for k,v in macro_map.items()})
    st.sidebar.markdown(f"India VIX: {vix_val if vix_val else 'N/A'}")
except Exception:
    pass

# ------------------------------
#  Price history fetch for selected tickers
# ------------------------------
tickers_subset = [t for _,t in companies[:limit]]
st.info(f"Fetching price history for {len(tickers_subset)} tickers (this may take a moment)...")
price_data = batch_history(tickers_subset, years=4)

# normalize price_map: ticker -> Series
price_map = {}
for t in tickers_subset:
    try:
        if isinstance(price_data, pd.DataFrame):
            # unexpected shape (if single ticker), handle
            if "Close" in price_data.columns:
                ser = price_data["Close"].dropna()
            else:
                ser = pd.Series(dtype=float)
        elif isinstance(price_data, dict):
            tmp = price_data.get(t, pd.DataFrame())
            ser = tmp["Close"].dropna() if (isinstance(tmp, pd.DataFrame) and "Close" in tmp.columns) else pd.Series(dtype=float)
        else:
            if isinstance(price_data.columns, pd.MultiIndex) and t in price_data.columns.get_level_values(0):
                ser = price_data[t]["Close"].dropna()
            else:
                tmp = yf.download(t, period="4y", progress=False, auto_adjust=True)
                ser = tmp["Close"].dropna() if (tmp is not None and "Close" in tmp.columns) else pd.Series(dtype=float)
        ser.index = pd.to_datetime(ser.index)
        price_map[t] = ser.sort_index()
    except Exception:
        price_map[t] = pd.Series(dtype=float)

# ------------------------------
#  ET Picks (cached)
# ------------------------------
et_picks = fetch_et_broker_picks() if feedparser is not None else {}

# ------------------------------
#  Sector-rel computation
# ------------------------------
sector_rel = compute_sector_rel(price_map, sector_map) if price_map else {}

# ------------------------------
#  Build ML dataset & train if requested
# ------------------------------
model_1m = None; auc1m = None
FEATURES_1M = ["m21","m63","vol21","ma_bias"]
if enable_ml:
    st.info("Building ML dataset and training (this may take some time)...")
    recent_years = 3 if train_recent else None
    df_ml = build_ml_dataset(price_map, recent_only_years=recent_years)
    st.write("ML dataset rows:", len(df_ml))
    if not df_ml.empty:
        df1m = df_ml.dropna(subset=["label_1m"] + [c for c in FEATURES_1M if c in df_ml.columns]) if not df_ml.empty else pd.DataFrame()
        if not df1m.empty:
            df1m = df1m.rename(columns={"label_1m":"label"})
            model_1m, auc1m = train_lgbm(df1m, FEATURES_1M)
            st.success(f"Trained 1M model (AUC ~ {auc1m:.3f} )" if auc1m is not None else "Trained 1M model (AUC N/A)")
        else:
            st.warning("Not enough ML-ready rows for 1M training.")
    else:
        st.warning("ML dataset empty; cannot train.")

# ------------------------------
#  Generate predictions (ML if available else heuristic)
# ------------------------------
rows=[]; log_rows=[]
geo_risk = 0.0  # placeholder; could be derived from news
skipped=[]
for tkr, ser in price_map.items():
    if ser is None or ser.empty:
        skipped.append((tkr,"no_data")); continue
    feats = compute_hidden_features(ser)
    if feats is None:
        skipped.append((tkr,f"insufficient({len(ser)})")); continue
    # sector relative
    srel = sector_rel.get(tkr, np.nan)
    # heuristic fallback
    h_r1m = heuristic_ret1m(feats, geo_risk=geo_risk)
    h_r1y = heuristic_ret1y(feats, geo_risk=geo_risk)
    ml_r1m = None; method="Heuristic"
    prob1m = None
    if model_1m is not None:
        try:
            feats_now = {k:feats.get(k,0.0) for k in ["m21","m63","vol21","ma_bias"]}
            feats_now["sector_rel_1m"] = srel if not pd.isna(srel) else 0.0
            # ensure model expects FEATURES_1M (we didn't add sector_rel into features by default)
            # if you want sector_rel to be used, append it to FEATURES_1M before training
            prob1m = ml_predict_prob(model_1m, feats_now, FEATURES_1M)
            ml_r1m = (prob1m - 0.5) * 40.0
            method="ML"
        except Exception:
            prob1m=None; ml_r1m=None
    final_r1m = ml_r1m if ml_r1m is not None else h_r1m
    final_r1y = h_r1y if (ml_r1m is None) else h_r1y  # we only trained 1M in this pipeline
    # risk adjust using VIX if available
    adj_factor = 1.0
    if vix is not None and vix_val is not None:
        adj_factor = 1 + (15 - (vix_val if vix_val else 15))/100.0
    r1m_adj = final_r1m * adj_factor
    r1y_adj = final_r1y * adj_factor
    cur = feats["current"]
    pred1m = round(cur*(1 + r1m_adj/100.0), 2)
    pred1y = round(cur*(1 + r1y_adj/100.0), 2)
    broker = ", ".join(et_picks.get(tkr, [])) if tkr in et_picks else "—"
    # probability -> confidence
    prob_for_display = round(float(prob1m),2) if prob1m is not None else None
    conf = "Low"
    if prob_for_display is not None:
        if prob_for_display >= 0.70: conf="High"
        elif prob_for_display >= 0.55: conf="Medium"
        else: conf="Low"
    else:
        conf = "Heuristic"
    rows.append({
        "Company": tkr.replace(".NS",""),
        "Ticker": tkr,
        "Current": round(cur,2),
        "Pred 1M": pred1m,
        "Pred 1Y": pred1y,
        "Ret 1M %": round(r1m_adj,2),
        "Ret 1Y %": round(r1y_adj,2),
        "SectorRel1M": round(srel,2) if not pd.isna(srel) else np.nan,
        "Broker Consensus": broker,
        "Model Prob": prob_for_display if prob_for_display is not None else np.nan,
        "Confidence": conf,
        "Method": method
    })
    # log row for integrated log
    log_rows.append({
        "run_date": datetime.utcnow(),
        "company": tkr.replace(".NS",""),
        "ticker": tkr,
        "current": round(cur,2),
        "pred_1m": pred1m,
        "pred_1y": pred1y,
        "ret_1m_pct": round(r1m_adj,2),
        "ret_1y_pct": round(r1y_adj,2),
        "method": method
    })

# show skipped
if skipped:
    st.info(f"Skipped {len(skipped)} tickers (no data or insufficient history).")

# assemble dataframe and ranking
out = pd.DataFrame(rows)
if out.empty:
    st.error("No tickers processed (insufficient data)."); st.stop()

out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
out["Composite Rank"] = ((out["Rank Ret 1M"]*0.5) + (out["Rank Ret 1Y"]*0.5)).rank(ascending=True, method="min").astype(int)
final = out.sort_values("Composite Rank").reset_index(drop=True)

# append new predictions to integrated log
ensure_log_exists()
existing_log = read_pred_log()
new_entries = pd.DataFrame(log_rows)
if not new_entries.empty:
    combined = pd.concat([existing_log, new_entries], ignore_index=True, sort=False)
    # try to compute per-run ranks and write
    try:
        combined["run_date"] = pd.to_datetime(combined["run_date"])
        ranked=[]
        for rd, group in combined.groupby("run_date"):
            grp = group.copy()
            grp["rank_ret_1m"] = grp["ret_1m_pct"].rank(ascending=False, method="min").astype("Int64")
            grp["rank_ret_1y"] = grp["ret_1y_pct"].rank(ascending=False, method="min").astype("Int64")
            grp["composite_rank"] = ((grp["rank_ret_1m"].astype(float) + grp["rank_ret_1y"].astype(float))/2.0).rank(ascending=True, method="min").astype("Int64")
            ranked.append(grp)
        combined = pd.concat(ranked, ignore_index=True, sort=False)
    except Exception:
        pass
    write_pred_log(combined)

st.subheader("📊 Ranked Screener (Phase1+)")
# color style for returns
def style_ret(v):
    if pd.isna(v): return ""
    try:
        v=float(v)
        return "color: green" if v>0 else ("color: red" if v<0 else "")
    except:
        return ""
st.dataframe(final.style.applymap(style_ret, subset=["Ret 1M %","Ret 1Y %"]), use_container_width=True)

# Filtered investable view
if view_mode == "Filtered Investable Picks":
    filt = final[(final["Confidence"]=="High") & (final["Broker Consensus"]!="-")]
    st.subheader("✅ Filtered Investable Picks (High confidence + Broker consensus)")
    if filt.empty:
        st.info("No filtered investable picks found right now.")
    else:
        st.dataframe(filt, use_container_width=True)
        st.download_button("Download Investable Watchlist CSV", data=filt.to_csv(index=False).encode('utf-8'), file_name="investable_watchlist.csv")

# Portfolio builder (keep as-is)
st.markdown("---")
st.header("📦 Portfolio Builder (Equal weight)")
capital = st.number_input("Total capital (₹)", min_value=1000, value=DEFAULT_CAPITAL, step=1000)
top_n = st.slider("Number of holdings (Top N)", 3, min(5, len(final)), min(20, len(final)), step=1)
sl_pct = st.number_input("Stop-loss %", min_value=1, max_value=50, value=10)
tp_pct = st.number_input("Take-profit %", min_value=1, max_value=200, value=25)

pf = final.head(top_n).copy()
pf["Weight %"] = round(100.0/top_n,2)
pf["Alloc ₹"] = (capital * (pf["Weight %"]/100.0)).round(2)
pf["Shares"] = (pf["Alloc ₹"] / pf["Current"]).astype(int)
pf["Stop-loss"] = (pf["Current"] * (1 - sl_pct/100.0)).round(2)
pf["Take-profit"] = (pf["Current"] * (1 + tp_pct/100.0)).round(2)
st.subheader(f"Top {top_n} Portfolio Plan")
st.dataframe(pf[["Company","Ticker","Current","Weight %","Alloc ₹","Shares","Stop-loss","Take-profit"]], use_container_width=True)

# Actuals updater actions
st.markdown("---")
st.header("📜 Integrated Predictions Log & Actuals")
if force_actuals:
    st.info("Forcing actuals update (this can be slow)...")
    updated_log = update_log_with_actuals(force=True)
    st.success("Actuals forced update attempted.")
else:
    st.info("Auto-updating matured actuals (best-effort)...")
    updated_log = update_log_with_actuals(force=False)
st.dataframe(read_pred_log().sort_values("run_date", ascending=False).head(300), use_container_width=True)

st.caption("Phase1+ app: baseline features + ETMarkets picks + sector-rel + ML 1M (optional) + actuals logger. Tune FEATURES_1M to include sector_rel_1m for ML if desired.")
