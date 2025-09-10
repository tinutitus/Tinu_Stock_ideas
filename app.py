# app_phase2_fixed.py
# Phase 2 — Full NSE Screener with Macro Signals, Fundamentals, Geo-News, ML (optional)
# Fixed version: ensures adj1m/adj1y are always defined and fixes syntax issues.
# Run: streamlit run app_phase2_fixed.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta
import time, re, os

# ML & NLP imports (optional at runtime; if missing, ML will be disabled)
try:
    import feedparser
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_NLP = True
except Exception:
    feedparser = None
    SentimentIntensityAnalyzer = None
    _HAS_NLP = False

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import shuffle
    _HAS_ML = True
except Exception:
    lgb = None
    train_test_split = None
    roc_auc_score = None
    shuffle = None
    _HAS_ML = False

# Page config
st.set_page_config(page_title="NSE Screener — Phase 2 (Fixed)", layout="wide")
st.title("⚡ NSE Screener — Phase 2 (Macro + Fundamentals) — Fixed")

# Config
PRED_LOG_PATH = "predictions_log.csv"
FUND_CSV_CACHE_TTL = 3600
NEWS_CACHE_TTL = 60 * 10
MIN_HISTORY_DAYS_FOR_FEATURES = 30
ML_MIN_ROWS_TO_TRAIN = 200

# Macro tickers for yfinance
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

# Helpers for persistence
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
    return pd.read_csv(path, parse_dates=["run_date","actual_1m_date","actual_1y_date"], low_memory=False)

def write_pred_log(df, path=PRED_LOG_PATH):
    df.to_csv(path, index=False)

# Constituents fetching (robust)
INDEX_URLS = {
    "Nifty Midcap 100":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

MIDCAP_FALLBACK = ["TATAMOTORS","HAVELLS","VOLTAS","PAGEIND","MINDTREE","MPHASIS","TORNTPHARM","POLYCAB","MPHASIS","TORNTPHARM"]
SMALLCAP_FALLBACK = ["AARTIIND","AFFLE","DIXON","QUESS","SHREECEM"]

def _csv_to_pairs_fuzzy(csv_text: str):
    csv_text = (csv_text or "").strip()
    if not csv_text:
        return []
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception:
        pairs = []
        lines = [l.strip() for l in csv_text.splitlines() if l.strip()]
        for ln in lines[:500]:
            parts = [p.strip().strip('"').strip("'") for p in re.split(r',|\t', ln) if p.strip()]
            if len(parts) >= 2:
                name, sym = parts[0], parts[1]
                sym = sym.upper()
                if not sym.endswith(".NS"): sym = sym + ".NS"
                pairs.append((name, sym))
        return pairs
    cols = [c.strip() for c in df.columns]
    cand_name = None; cand_sym = None
    for c in cols:
        lc = c.lower()
        if any(x in lc for x in ("company","company name","name")) and cand_name is None:
            cand_name = c
        if any(x in lc for x in ("symbol","code","ticker","token")) and cand_sym is None:
            cand_sym = c
    if cand_sym is not None:
        if cand_name is None:
            df["Company"] = df[cand_sym].astype(str)
            cand_name = "Company"
        names = df[cand_name].astype(str).str.strip().tolist()
        syms = df[cand_sym].astype(str).str.strip().apply(lambda s: s.upper() if s.upper().endswith(".NS") else s.upper()+".NS").tolist()
        return list(zip(names, syms))
    if len(cols) >= 2:
        names = df.iloc[:,0].astype(str).str.strip().tolist()
        syms = df.iloc[:,1].astype(str).str.strip().apply(lambda s: s.upper() if s.upper().endswith(".NS") else s.upper()+".NS").tolist()
        return list(zip(names, syms))
    if len(cols) == 1:
        syms = df.iloc[:,0].astype(str).str.strip().apply(lambda s: s.upper() if s.upper().endswith(".NS") else s.upper()+".NS").tolist()
        return [(s.replace(".NS",""), s) for s in syms]
    return []

@st.cache_data(ttl=86400)
def fetch_constituents(index_name: str):
    url = INDEX_URLS.get(index_name)
    headers = {"User-Agent":"Mozilla/5.0"}
    debug_notes=[]
    if url:
        try:
            r = requests.get(url, headers=headers, timeout=8)
            debug_notes.append(f"status {r.status_code}")
            pairs = _csv_to_pairs_fuzzy(r.text)
            if pairs:
                return pairs
            else:
                debug_notes.append("parsed 0 pairs")
        except Exception as e:
            debug_notes.append(f"error: {type(e).__name__}:{e}")
    # fallback
    if "smallcap" in index_name.lower():
        return [(s, f"{s}.NS") for s in SMALLCAP_FALLBACK]
    return [(s, f"{s}.NS") for s in MIDCAP_FALLBACK]

# Fundamentals ingestion
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

# Macro series fetch
@st.cache_data(ttl=1800)
def fetch_macro_timeseries(tickers: dict, period_years=2):
    out = {}
    syms = [v for v in tickers.values() if v]
    try:
        df = yf.download(list(set(syms)), period=f"{period_years}y", auto_adjust=True, progress=False, threads=True)
    except Exception:
        df = None
    for name, sym in tickers.items():
        ser = pd.Series(dtype=float)
        try:
            if df is not None and isinstance(df.columns, pd.MultiIndex) and sym in df.columns.get_level_values(0):
                ser = df[sym]["Close"].dropna()
            else:
                tmp = yf.download(sym, period=f"{period_years}y", auto_adjust=True, progress=False)
                if tmp is not None and "Close" in tmp.columns:
                    ser = tmp["Close"].dropna()
        except Exception:
            try:
                tmp = yf.download(sym, period=f"{period_years}y", auto_adjust=True, progress=False)
                if tmp is not None and "Close" in tmp.columns:
                    ser = tmp["Close"].dropna()
            except Exception:
                ser = pd.Series(dtype=float)
        if not ser.empty:
            ser.index = pd.to_datetime(ser.index)
            out[name] = ser.sort_index()
        else:
            out[name] = pd.Series(dtype=float)
    return out

def macro_features_from_series(macro_map: dict, ref_date: pd.Timestamp, window_days=30):
    out = {}
    for name, ser in macro_map.items():
        try:
            if ser is None or ser.empty:
                out[f"macro_{name}_pct{window_days}"] = 0.0
                out[f"macro_{name}_vol{window_days}"] = 0.0
                continue
            idx = ser.index.searchsorted(ref_date)
            if idx == len(ser): idx = len(ser)-1
            if idx == 0:
                out[f"macro_{name}_pct{window_days}"] = 0.0
                out[f"macro_{name}_vol{window_days}"] = 0.0
                continue
            cur = float(ser.iloc[idx])
            past_idx = max(0, idx - window_days)
            past = float(ser.iloc[past_idx])
            pct = (cur/past - 1.0) * 100.0 if past != 0 else 0.0
            vol = float(ser.pct_change().tail(window_days).std()) * 100.0 if len(ser.pct_change().dropna())>1 else 0.0
            out[f"macro_{name}_pct{window_days}"] = pct
            out[f"macro_{name}_vol{window_days}"] = vol
        except Exception:
            out[f"macro_{name}_pct{window_days}"] = 0.0
            out[f"macro_{name}_vol{window_days}"] = 0.0
    return out

# VIX fetch and adj
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
    if horizon == "1M":
        return float(np.clip((15 - vix) * 0.8, -10, 10))
    return float(np.clip((15 - vix) * 1.2, -20, 20))

# Geo-news (optional)
if _HAS_NLP:
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
    if not _HAS_NLP:
        return out
    try:
        f = feedparser.parse(url)
        entries = f.entries[:max_items]
        for e in entries:
            title = getattr(e,"title","") or ""
            summary = getattr(e,"summary","") or ""
            pub=None
            if hasattr(e,"published_parsed") and e.published_parsed:
                try: pub = datetime(*e.published_parsed[:6])
                except: pub=None
            elif hasattr(e,"updated_parsed") and e.updated_parsed:
                try: pub = datetime(*e.updated_parsed[:6])
                except: pub=None
            text = _clean_text(title + " " + summary)
            score = analyzer.polarity_scores(text)["compound"] if text and analyzer is not None else 0.0
            out.append({"title":_clean_text(title),"summary":_clean_text(summary),"published":pub,"compound":score,"text":text})
        time.sleep(0.02)
    except Exception:
        pass
    return out

@st.cache_data(ttl=NEWS_CACHE_TTL)
def get_geo_news_features():
    now = datetime.utcnow()
    out = {}
    for region in ["us","eu","cn"]:
        feeds = RSS_FEEDS.get(region,[])
        items=[]
        for url in feeds:
            items.extend(_fetch_feed_items_with_date(url, max_items=60))
        bins = {"3d": now - timedelta(days=3), "7d": now - timedelta(days=7), "30d": now - timedelta(days=30)}
        for key, cutoff in bins.items():
            sel = [itm for itm in items if (itm["published"] is None or itm["published"] >= cutoff)]
            if not sel:
                out[f"news_{region}_avg_{key}"] = 0.0
                out[f"news_{region}_vol_{key}"] = 0
            else:
                out[f"news_{region}_avg_{key}"] = float(np.mean([s["compound"] for s in sel]))
                out[f"news_{region}_vol_{key}"] = len(sel)
            topic_counts = {t:0 for t in TOPIC_KEYWORDS.keys()}
            for s in sel:
                txt = s["text"]
                for t,kws in TOPIC_KEYWORDS.items():
                    for kw in kws:
                        if kw in txt:
                            topic_counts[t] += 1
                            break
            total = max(1,len(sel))
            for t in TOPIC_KEYWORDS.keys():
                out[f"news_{region}_topic_{t}_{key}"] = topic_counts[t]/total
    neg_sum = 0.0
    for r in ["us","eu","cn"]:
        neg = min(0, out.get(f"news_{r}_avg_7d", 0.0))
        vol = out.get(f"news_{r}_vol_7d", 0)
        neg_sum += max(0, -neg) * (1 + vol/20.0)
    out["geo_news_risk"] = float(min(5.0, neg_sum))
    return out

# Price fetch
@st.cache_data(show_spinner=False)
def batch_history(tickers, years=4):
    return yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False, threads=True, group_by="ticker")

# Feature engineering technicals
def compute_hidden_features(s: pd.Series):
    s = s.dropna().astype(float)
    if s.size < MIN_HISTORY_DAYS_FOR_FEATURES:
        return None
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
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
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

# Heuristic fallback functions
def heuristic_ret1m_from_feats(feats, macro_map=None):
    base = 0.6*(feats.get("m21",0.0) or 0.0) + 0.4*(feats.get("m63",0.0) or 0.0)
    adj = 1.0
    if feats.get("ma_bias",0) < 0: adj *= 0.78
    if feats.get("rsi14",50) > 70: adj *= 0.88
    if feats.get("rsi14",50) < 30: adj *= 1.12
    if feats.get("macd_conf",0) < 0: adj *= 0.93
    elif feats.get("macd_conf",0) > 0.5: adj *= 1.05
    macro_penalty = 0.0
    if macro_map and isinstance(macro_map, dict):
        sp = macro_map.get("macro_SP500_pct30", 0.0)
        geo = macro_map.get("geo_news_risk", 0.0)
        macro_penalty = -0.2 * min(0, sp) - 0.8 * geo
    out = np.clip(base*adj + macro_penalty - 100*(feats.get("vol21",0.0) or 0.0), -50,50)
    return out

def heuristic_ret1y_from_feats(feats, macro_map=None):
    m63 = feats.get("m63",0.0) or 0.0
    m252 = feats.get("m252", np.nan)
    m252_eff = m252 if m252==m252 else (m63*4 if m63==m63 else 0.0)
    macro_penalty = 0.0
    if macro_map:
        sp = macro_map.get("macro_SP500_pct30", 0.0)
        usd = macro_map.get("macro_USDINR_pct30", 0.0)
        geo = macro_map.get("geo_news_risk", 0.0)
        macro_penalty = -0.2 * min(0, sp) - 0.1 * abs(usd) - 0.8 * geo
    out = np.clip(0.3*m63 + 0.7*m252_eff + macro_penalty - 150*(feats.get("vol21",0.0) or 0.0), -80,120)
    return out

# ML dataset builder & training (simplified)
@st.cache_data(ttl=86400)
def build_ml_dataset_with_macro_and_fundamentals(price_map: dict, macro_map: dict, fundamentals_df: pd.DataFrame=None, min_history_days=60, recent_only_years=None):
    rows=[]
    all_dates = sorted({d for ser in price_map.values() if ser is not None and not ser.empty for d in ser.index})
    if not all_dates:
        return pd.DataFrame(rows)
    all_dates = pd.DatetimeIndex(all_dates)
    months = sorted({(d.year,d.month) for d in all_dates})
    run_dates=[]
    for y,m in months:
        d = all_dates[(all_dates.year==y)&(all_dates.month==m)]
        if len(d): run_dates.append(d.max())
    HOLD_1M = 21; HOLD_1Y = 252
    for run_date in run_dates:
        if recent_only_years:
            if (datetime.today().year - run_date.year) > recent_only_years: continue
        geo_news = get_geo_news_features()
        macro_feats = macro_features_from_series(macro_map, run_date, window_days=30) if macro_map else {}
        for ticker, ser in price_map.items():
            if ser is None or ser.empty: continue
            if run_date < ser.index.min(): continue
            ser_up = ser[:run_date]
            if len(ser_up) < min_history_days: continue
            feats = compute_hidden_features(ser_up)
            if feats is None: continue
            idx = ser.index.searchsorted(run_date)
            if idx == len(ser) or ser.index[idx] != run_date:
                if idx == 0: continue
                idx = idx - 1
            fut1m = idx + HOLD_1M
            fut1y = idx + HOLD_1Y
            if fut1m >= len(ser): continue
            entry = float(ser.iloc[idx])
            future1m = float(ser.iloc[fut1m])
            real_ret_1m = (future1m / entry - 1.0) * 100.0
            label_1m = 1 if real_ret_1m > 0 else 0
            if fut1y < len(ser):
                future1y = float(ser.iloc[fut1y])
                real_ret_1y = (future1y / entry - 1.0) * 100.0
                label_1y = 1 if real_ret_1y > 0 else 0
            else:
                future1y = np.nan; real_ret_1y = np.nan; label_1y = np.nan
            row = {
                "run_date": run_date, "ticker": ticker,
                "entry": entry, "future1m": future1m, "real_ret_1m": real_ret_1m, "label_1m": label_1m,
                "future1y": future1y, "real_ret_1y": real_ret_1y, "label_1y": label_1y
            }
            row.update({k:feats[k] for k in ["m1","m5","m21","m63","vol21","ma_bias","rsi14","macd_conf","prox52","skew60","kurt60"]})
            row.update(geo_news)
            row.update(macro_feats)
            if fundamentals_df is not None:
                frow = None
                if ticker in fundamentals_df.index:
                    frow = fundamentals_df.loc[ticker]
                else:
                    t_short = ticker.replace(".NS","")
                    alt_idx = [i for i in fundamentals_df.index if i.upper().startswith(t_short)]
                    if alt_idx: frow = fundamentals_df.loc[alt_idx[0]]
                if frow is not None:
                    for col in fundamentals_df.columns:
                        try:
                            val = frow.get(col, np.nan)
                        except Exception:
                            try: val = frow[col]
                            except: val = np.nan
                        try:
                            row[f"f_{col}"] = float(val) if (pd.notna(val) and isinstance(val,(int,float,np.number))) else np.nan
                        except Exception:
                            row[f"f_{col}"] = np.nan
                else:
                    for col in fundamentals_df.columns:
                        row[f"f_{col}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)

@st.cache_data(ttl=86400)
def train_lgbm(df_train, features, num_boost_round=300):
    if not _HAS_ML:
        st.warning("train_lgbm: ML libs not available.")
        return None, None
    try:
        missing = [c for c in features if c not in df_train.columns]
        if missing:
            st.warning(f"train_lgbm: missing features, skipping. Missing: {missing}")
            return None, None
        X = df_train[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        y = pd.to_numeric(df_train["label"], errors="coerce").fillna(0).astype(int)
        if len(X) < ML_MIN_ROWS_TO_TRAIN:
            st.warning(f"Not enough rows to train (found {len(X)}). Need >={ML_MIN_ROWS_TO_TRAIN}.")
            return None, None
        Xs, ys = shuffle(X, y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=ys)
        model = lgb.LGBMClassifier(objective="binary", n_estimators=num_boost_round, learning_rate=0.05,
                                   num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, n_jobs=1, verbosity=-1)
        try:
            model.fit(X_train, y_train, eval_set=[(X_val,y_val)], eval_metric="auc", early_stopping_rounds=25, verbose=False)
        except TypeError:
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                st.error(f"train_lgbm fit failed: {e}")
                return None, None
        try:
            y_proba = model.predict_proba(X_val)[:,1]
            auc = roc_auc_score(y_val, y_proba) if len(np.unique(y_val))>1 else float("nan")
        except Exception:
            auc = float("nan")
        return model, auc
    except Exception as e:
        st.error(f"train_lgbm: {type(e).__name__}: {e}")
        return None, None

def ml_predict_prob(model, feats_row, features):
    X = pd.DataFrame([feats_row])[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:,1][0])
    if hasattr(model, "predict"):
        p = model.predict(X)
        if isinstance(p, np.ndarray): return float(p[0])
        return float(p)
    raise RuntimeError("Model does not support probability prediction")

def prob_to_return_1m(p): return (p - 0.5) * 40.0
def prob_to_return_1y(p): return (p - 0.5) * 200.0

# Actuals helpers
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

def update_log_with_actuals(path=PRED_LOG_PATH, now_date=None, force=False):
    ensure_log_exists(path)
    df = read_pred_log(path)
    if df.empty:
        return df

    now = datetime.utcnow().date() if now_date is None else pd.to_datetime(now_date).date()
    updated = False

    for idx, row in df.iterrows():
        try:
            run_date = pd.to_datetime(row["run_date"]).date()
        except Exception:
            continue

        # --- Handle 1M actual ---
        needs_1m = pd.isna(row.get("actual_1m")) or force
        target_1m = run_date + timedelta(days=30)
        if needs_1m and target_1m is not pd.NaT and target_1m <= now:
            price, price_date = fetch_actual_close_on_or_after(row["ticker"], target_1m, lookahead_days=7)
            if not pd.isna(price):
                df.at[idx,"actual_1m"] = price
                df.at[idx,"actual_1m_date"] = price_date
                pred = row.get("pred_1m", np.nan)
                df.at[idx,"err_pct_1m"] = (abs(pred-price)/price*100) if (not pd.isna(pred) and price!=0) else np.nan
                updated = True

        # --- Handle 1Y actual ---
        needs_1y = pd.isna(row.get("actual_1y")) or force
        target_1y = run_date + timedelta(days=365)
        if needs_1y and target_1y is not pd.NaT and target_1y <= now:
            price, price_date = fetch_actual_close_on_or_after(row["ticker"], target_1y, lookahead_days=14)
            if not pd.isna(price):
                df.at[idx,"actual_1y"] = price
                df.at[idx,"actual_1y_date"] = price_date
                pred = row.get("pred_1y", np.nan)
                df.at[idx,"err_pct_1y"] = (abs(pred-price)/price*100) if (not pd.isna(pred) and price!=0) else np.nan
                updated = True

    if updated:
        write_pred_log(df)
    return df
            price, price_date = fetch_actual_close_on_or_after(row["ticker"], target_1m, lookahead_days=7)
            if not pd.isna(price):
                df.at[idx,"actual_1m"] = price; df.at[idx,"actual_1m_date"] = price_date
                pred = row.get("pred_1m", np.nan)
                df.at[idx,"err_pct_1m"] = (abs(pred-price)/price*100) if (not pd.isna(pred) and price!=0) else np.nan
                updated = True
        needs_1y = pd.isna(row.get("actual_1y")) or force
        target_1y = run_date + timedelta(days=365)
        if needs_1y and target_1y <= now:
            price, price_date = fetch_actual_close_on_or_after(row["ticker"], target_1y, lookahead_days=14)
            if not pd.isna(price):
                df.at[idx,"actual_1y"] = price; df.at[idx,"actual_1y_date"] = price_date
                pred = row.get("pred_1y", np.nan)
                df.at[idx,"err_pct_1y"] = (abs(pred-price)/price*100) if (not pd.isna(pred) and price!=0) else np.nan
                updated = True
    if updated:
        write_pred_log(df)
    return df

# UI controls
st.sidebar.header("Phase 2 Options (fixed)")
index_choice = st.sidebar.selectbox("Index", list(INDEX_URLS.keys()))
companies = fetch_constituents(index_choice)
if not companies:
    st.sidebar.error("No constituents found"); st.stop()

n_companies = len(companies)
min_tickers = 1 if n_companies < 10 else 10
max_tickers = max(1, n_companies)
default_tickers = max_tickers
step = 1 if max_tickers - min_tickers <= 10 else 5

limit = st.sidebar.slider("Tickers to process", min_value=min_tickers, max_value=max_tickers, value=default_tickers, step=step)
enable_ml = st.sidebar.checkbox("Enable ML (train & use)", value=False)
train_recent = st.sidebar.checkbox("Train on recent N years (faster)", value=True)
fund_csv_url = st.sidebar.text_input("Optional fundamentals CSV URL (public raw CSV)", value="")
macro_period_years = st.sidebar.slider("Macro history (years)", 1, 5, 2)
refresh_news = st.sidebar.button("Refresh News Now")
force_actuals_refresh = st.sidebar.button("Force refresh actuals")
# Optional: Show ETMarkets Picks
et_enable = st.sidebar.checkbox("Show ETMarkets Picks", value=False)
# ----- UI controls -----
st.sidebar.header("Phase 1 Options")
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
# Fetch macro series
st.info("Fetching macro time series...")
macro_map_timeseries = fetch_macro_timeseries(MACRO_TICKERS, period_years=macro_period_years)
# Ensure VIX adjustments are defined (avoid NameError)
try:
    vix = fetch_vix()
    adj1m = vix_to_adj(vix, "1M")
    adj1y = vix_to_adj(vix, "1Y")
except Exception:
    vix = None; adj1m = 0.0; adj1y = 0.0

st.caption(f"India VIX = {vix if vix else 'N/A'} → Risk Adj(1M) {adj1m:+.2f}%  (Adj1Y {adj1y:+.2f}%)")

# Geo-news snapshot
geo_snapshot = get_geo_news_features() if _HAS_NLP else {"geo_news_risk":0.0,"news_us_avg_7d":0.0,"news_eu_avg_7d":0.0,"news_cn_avg_7d":0.0}
try:
    st.markdown("### 🔎 Geo-political News Snapshot")
    regions = ["us","eu","cn"]
    cols = st.columns([1,1,1,0.8])
    for i, region in enumerate(regions):
        col = cols[i]
        col.metric("7d avg", f"{geo_snapshot.get(f'news_{region}_avg_7d',0.0):+.2f}", f"vol {int(geo_snapshot.get(f'news_{region}_vol_7d',0))}")
    risk_col = cols[-1]
    risk_col.metric("Geo-news risk", f"{geo_snapshot.get('geo_news_risk',0.0):.2f}")
except Exception:
    st.warning("Geo-news snapshot not available.")

# Load fundamentals
fund_df = None
if fund_csv_url:
    st.info("Loading fundamentals CSV...")
    fund_df = fetch_fundamentals_csv(fund_csv_url)
    if fund_df is None:
        st.warning("Could not load fundamentals CSV; continuing without fundamentals.")
    else:
        st.success(f"Loaded fundamentals for {len(fund_df)} tickers.")

# Fetch price history
tickers_subset = [t for _, t in companies[:limit]]
st.info(f"Fetching price history for {len(tickers_subset)} tickers...")
price_data = batch_history(tickers_subset, years=4)
price_map = {}
for t in tickers_subset:
    try:
        if isinstance(price_data.columns, pd.MultiIndex):
            ser = price_data[t]["Close"].dropna()
        else:
            if "Close" in price_data.columns:
                ser = price_data["Close"].dropna()
            else:
                tmp = yf.download(t, period="4y", auto_adjust=True, progress=False)
                ser = tmp["Close"].dropna() if (tmp is not None and "Close" in tmp.columns) else pd.Series(dtype=float)
        ser.index = pd.to_datetime(ser.index)
        price_map[t] = ser.sort_index()
    except Exception:
        price_map[t] = pd.Series(dtype=float)

# Feature lists
NEWS_KEYS=[]
for r in ["us","eu","cn"]:
    for w in ["3d","7d","30d"]:
        NEWS_KEYS += [f"news_{r}_avg_{w}", f"news_{r}_vol_{w}"]
    for topic in TOPIC_KEYWORDS.keys():
        for w in ["3d","7d","30d"]:
            NEWS_KEYS.append(f"news_{r}_topic_{topic}_{w}")
NEWS_KEYS.append("geo_news_risk")

MACRO_KEYS=[]
for name in MACRO_TICKERS.keys():
    MACRO_KEYS += [f"macro_{name}_pct30", f"macro_{name}_vol30"]

CORE_FEATURES = ["m1","m5","m21","m63","vol21","ma_bias","rsi14","macd_conf","prox52","skew60","kurt60"]
FEATURES_1M = CORE_FEATURES + NEWS_KEYS + MACRO_KEYS
FEATURES_1Y = CORE_FEATURES + NEWS_KEYS + MACRO_KEYS

fund_numeric_cols=[]
if fund_df is not None:
    fund_numeric_cols=[c for c in fund_df.columns if pd.api.types.is_numeric_dtype(fund_df[c])]
    fund_numeric_cols = fund_numeric_cols[:8]
    FUND_FEATS=[f"f_{c}" for c in fund_numeric_cols]
    FEATURES_1M += FUND_FEATS; FEATURES_1Y += FUND_FEATS

# ML training
model_1m = model_1y = None
auc1m = auc1y = None
if enable_ml and _HAS_ML:
    recent_years = 3 if train_recent else None
    st.info("Building ML dataset (macro + fundamentals) — may take time...")
    df_ml = build_ml_dataset_with_macro_and_fundamentals(price_map, macro_map_timeseries, fundamentals_df=fund_df, min_history_days=60, recent_only_years=recent_years)
    st.write("ML dataset shape:", df_ml.shape)
    if not df_ml.empty:
        df1m = df_ml.dropna(subset=["label_1m"] + [c for c in FEATURES_1M if c in df_ml.columns])
        df1y = df_ml.dropna(subset=["label_1y"] + [c for c in FEATURES_1Y if c in df_ml.columns])
        if not df1m.empty:
            df1m = df1m.rename(columns={"label_1m":"label"})
            model_1m, auc1m = train_lgbm(df1m, [c for c in FEATURES_1M if c in df1m.columns])
        if not df1y.empty:
            df1y = df1y.rename(columns={"label_1y":"label"})
            model_1y, auc1y = train_lgbm(df1y, [c for c in FEATURES_1Y if c in df1y.columns])
    st.success(f"ML training finished — AUC 1M: {auc1m if auc1m is not None else 'N/A'}, AUC 1Y: {auc1y if auc1y is not None else 'N/A'}")
elif enable_ml and not _HAS_ML:
    st.warning("ML requested but libraries not installed (lightgbm, scikit-learn).")

# Build predictions
rows=[]; log_rows=[]
geo_news = geo_snapshot if isinstance(geo_snapshot, dict) else {"geo_news_risk":0.0}
skipped_debug=[]

for tkr, ser in price_map.items():
    if ser is None or ser.empty:
        skipped_debug.append((tkr, "no_data")); continue
    feats = compute_hidden_features(ser)
    if feats is None:
        if len(ser) >= 6:
            cur = float(ser.iloc[-1])
            mom5 = (cur / float(ser.iloc[-6]) - 1.0) * 100.0
            vol = ser.pct_change().tail(5).std() if len(ser.pct_change().dropna())>=1 else 0.01
            feats = {"current": cur, "m1": None, "m5": mom5, "m21": mom5, "m63": mom5, "m252": mom5,
                     "vol21": vol, "ma_bias": 1.0, "rsi14": 50.0, "macd_conf": 0.0, "prox52":50.0, "skew60":0.0, "kurt60":0.0}
        else:
            skipped_debug.append((tkr, f"insufficient({len(ser)})")); continue

    macro_feats_now = macro_features_from_series(macro_map_timeseries, ser.index[-1], window_days=30) if macro_map_timeseries else {}
    h_r1m = heuristic_ret1m_from_feats(feats, macro_map={**macro_feats_now, **geo_news})
    h_r1y = heuristic_ret1y_from_feats(feats, macro_map={**macro_feats_now, **geo_news})

    ml_r1m = ml_r1y = None
    method = "Heuristic"

    if model_1m is not None:
        try:
            feats_now = {k: feats.get(k, 0.0) for k in CORE_FEATURES}
            feats_now.update(geo_news); feats_now.update(macro_feats_now)
            if fund_df is not None:
                if tkr in fund_df.index:
                    frow = fund_df.loc[tkr]
                else:
                    alt = tkr.replace(".NS",""); alt_idx=[i for i in fund_df.index if i.upper().startswith(alt)]
                    frow = fund_df.loc[alt_idx[0]] if alt_idx else None
                for col in fund_numeric_cols:
                    feats_now[f"f_{col}"] = float(frow.get(col, 0.0)) if (frow is not None and pd.notna(frow.get(col, np.nan))) else 0.0
            p1m = ml_predict_prob(model_1m, feats_now, [c for c in FEATURES_1M if c in FEATURES_1M])
            ml_r1m = prob_to_return_1m(p1m)
            method = "ML"
        except Exception:
            ml_r1m = None

    if model_1y is not None:
        try:
            feats_now = {k: feats.get(k, 0.0) for k in CORE_FEATURES}
            feats_now.update(geo_news); feats_now.update(macro_feats_now)
            if fund_df is not None:
                if tkr in fund_df.index:
                    frow = fund_df.loc[tkr]
                else:
                    alt = tkr.replace(".NS",""); alt_idx=[i for i in fund_df.index if i.upper().startswith(alt)]
                    frow = fund_df.loc[alt_idx[0]] if alt_idx else None
                for col in fund_numeric_cols:
                    feats_now[f"f_{col}"] = float(frow.get(col, 0.0)) if (frow is not None and pd.notna(frow.get(col, np.nan))) else 0.0
            p1y = ml_predict_prob(model_1y, feats_now, [c for c in FEATURES_1Y if c in FEATURES_1Y])
            ml_r1y = prob_to_return_1y(p1y)
            method = "ML" if method!="ML" else "ML"
        except Exception:
            ml_r1y = None

    final_r1m = ml_r1m if ml_r1m is not None else h_r1m
    final_r1y = ml_r1y if ml_r1y is not None else h_r1y

    # risk adjustment from VIX (safe adj1m/adj1y assumed defined earlier)
    r1m_adj = final_r1m * (1 + adj1m / 100.0)
    r1y_adj = final_r1y * (1 + adj1y / 100.0)

    cur = feats["current"]
    pred1m_price = round(cur * (1 + r1m_adj/100.0), 2)
    pred1y_price = round(cur * (1 + r1y_adj/100.0), 2)

    rows.append({
        "Company": tkr.replace(".NS",""),
        "Ticker": tkr,
        "Current": round(cur,2),
        "Pred 1M": pred1m_price,
        "Pred 1Y": pred1y_price,
        "Ret 1M %": round(r1m_adj,2),
        "Ret 1Y %": round(r1y_adj,2),
        "Method": method
    })

    log_rows.append({
        "run_date": datetime.utcnow(),
        "company": tkr.replace(".NS",""),
        "ticker": tkr,
        "current": round(cur,2),
        "pred_1m": pred1m_price,
        "pred_1y": pred1y_price,
        "ret_1m_pct": round(r1m_adj,2),
        "ret_1y_pct": round(r1y_adj,2),
        "method": method
    })

# skipped debug
if skipped_debug:
    st.info(f"Skipped {len(skipped_debug)} tickers (no data or too short history).")
    st.dataframe(pd.DataFrame(skipped_debug, columns=["ticker","reason"]).head(200))

if not rows:
    st.error("No tickers processed (insufficient data)."); st.stop()

out = pd.DataFrame(rows)
out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
out["Composite Rank"] = ((out["Rank Ret 1M"] * 0.5) + (out["Rank Ret 1Y"] * 0.5)).rank(ascending=True, method="min").astype(int)
final = out[["Company","Ticker","Current","Pred 1M","Pred 1Y","Ret 1M %","Ret 1Y %","Composite Rank","Rank Ret 1M","Rank Ret 1Y","Method"]].sort_values("Composite Rank").reset_index(drop=True)

st.subheader("📊 Phase 2 Ranked Screener (Macro + Fundamentals)")
def color_ret(v):
    if v > 0: return "color: green"
    if v < 0: return "color: red"
    return ""
st.dataframe(final.style.applymap(color_ret, subset=["Ret 1M %","Ret 1Y %"]), use_container_width=True)
# ----- ETMarkets Picks Section -----
if et_enable:
    st.subheader("📌 ETMarkets Expert Picks")
    
    # Temporary placeholder (until we add scraping/API)
    demo_data = {
        "Company": ["HDFC Bank", "Infosys", "Tata Motors"],
        "Ticker": ["HDFCBANK.NS", "INFY.NS", "TATAMOTORS.NS"],
        "ET Rating": ["Buy", "Hold", "Buy"],
        "Target Price (₹)": [1800, 1550, 700]
    }
    et_df = pd.DataFrame(demo_data)
    
    st.dataframe(et_df, use_container_width=True)

# append to integrated log
ensure_log_exists()
existing = read_pred_log()
new_entries = pd.DataFrame(log_rows)
if not new_entries.empty:
    combined = pd.concat([existing, new_entries], ignore_index=True, sort=False)
    try:
        combined["run_date"] = pd.to_datetime(combined["run_date"])
        ranked = []
        for run_date, group in combined.groupby("run_date"):
            grp = group.copy()
            grp["rank_ret_1m"] = grp["ret_1m_pct"].rank(ascending=False, method="min").astype("Int64")
            grp["rank_ret_1y"] = grp["ret_1y_pct"].rank(ascending=False, method="min").astype("Int64")
            grp["composite_rank"] = ((grp["rank_ret_1m"].astype(float) + grp["rank_ret_1y"].astype(float))/2.0).rank(ascending=True, method="min").astype("Int64")
            ranked.append(grp)
        combined = pd.concat(ranked, ignore_index=True, sort=False)
    except Exception:
        pass
    write_pred_log(combined)

st.success("Predictions computed and added to integrated log.")

# update actuals
st.info("Updating integrated log with matured actuals...")
if force_actuals_refresh:
    log_df = update_log_with_actuals(PRED_LOG_PATH, force=True)
else:
    log_df = update_log_with_actuals(PRED_LOG_PATH, force=False)
st.success("Actuals update complete.")

# historical log and summary
st.header("📜 Integrated Historical Predictions Log")
log_df = read_pred_log()
if log_df.empty:
    st.info("No historical predictions yet.")
else:
    display_df = log_df.copy()
    display_df["run_date"] = pd.to_datetime(display_df["run_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(display_df.sort_values("run_date", ascending=False).head(300), use_container_width=True)

    def compute_summary(df):
        out={}
        m1 = df[~pd.isna(df["actual_1m"])].copy()
        if not m1.empty:
            out["1M_mape"] = float(m1["err_pct_1m"].mean())
            m1["pred_dir_up"] = (m1["pred_1m"] > m1["current"]).astype(int)
            m1["actual_dir_up"] = (m1["actual_1m"] > m1["current"]).astype(int)
            out["1M_dir_acc"] = float((m1["pred_dir_up"]==m1["actual_dir_up"]).mean())*100.0
            out["1M_count"] = int(len(m1))
        else:
            out["1M_mape"]=np.nan; out["1M_dir_acc"]=np.nan; out["1M_count"]=0
        m2 = df[~pd.isna(df["actual_1y"])].copy()
        if not m2.empty:
            out["1Y_mape"] = float(m2["err_pct_1y"].mean())
            m2["pred_dir_up"] = (m2["pred_1y"] > m2["current"]).astype(int)
            m2["actual_dir_up"] = (m2["actual_1y"] > m2["current"]).astype(int)
            out["1Y_dir_acc"] = float((m2["pred_dir_up"]==m2["actual_dir_up"]).mean())*100.0
            out["1Y_count"] = int(len(m2))
        else:
            out["1Y_mape"]=np.nan; out["1Y_dir_acc"]=np.nan; out["1Y_count"]=0
        return out

    summary = compute_summary(log_df)
    st.markdown("### 📈 Summary (matured predictions)")
    c1,c2,c3 = st.columns(3)
    c1.metric("1M MAPE (%)", f"{summary['1M_mape']:.2f}" if not pd.isna(summary["1M_mape"]) else "N/A", f"count {summary['1M_count']}")
    c2.metric("1M Dir Acc (%)", f"{summary['1M_dir_acc']:.1f}" if not pd.isna(summary["1M_dir_acc"]) else "N/A")
    c3.metric("1Y MAPE (%)", f"{summary['1Y_mape']:.2f}" if not pd.isna(summary["1Y_mape"]) else "N/A", f"count {summary['1Y_count']}")

# portfolio builder
st.markdown("---")
st.header("📦 Portfolio Builder (Equal-weight)")
col1,col2,col3 = st.columns(3)
with col1:
    capital = st.number_input("Total capital (₹)", min_value=1000, value=10000, step=1000)
with col2:
    top_n = st.slider("Number of holdings (Top N)", 3, min(5, len(final)), min(20, len(final)), step=1)
with col3:
    sl_pct = st.number_input("Stop-loss %", min_value=1, max_value=50, value=10)
tp_pct = st.number_input("Take-profit %", min_value=1, max_value=200, value=25)

pf = final.head(top_n).copy()
pf["Weight %"] = round(100.0/top_n, 2)
pf["Alloc ₹"] = (capital * (pf["Weight %"]/100.0)).round(2)
pf["Shares"] = (pf["Alloc ₹"] / pf["Current"]).astype(int)
pf["Stop-loss"] = (pf["Current"] * (1 - sl_pct/100.0)).round(2)
pf["Take-profit"] = (pf["Current"] * (1 + tp_pct/100.0)).round(2)
st.subheader(f"Top {top_n} Portfolio Plan")
st.dataframe(pf[["Company","Ticker","Current","Weight %","Alloc ₹","Shares","Stop-loss","Take-profit"]], use_container_width=True)

st.caption("Phase 2 implemented: macro signals + fundamentals fused into heuristics and optional ML. Tune FEATURE lists and composite-rank weights in the code as desired.")
