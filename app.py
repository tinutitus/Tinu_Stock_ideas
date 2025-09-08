# app.py
# Full NSE Screener app: robust constituents fetch (midcap/smallcap), geo-news, optional ML,
# predictions 1M & 1Y (heuristic + ML), logging, auto-actuals, portfolio builder.
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta
import time, re, os

# ML & NLP
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

st.set_page_config(page_title="NSE Screener — Full App", layout="wide")
st.title("⚡ NSE Screener — Full App (robust smallcap fetch + ML + Geo-news)")

# ---------------- Config ----------------
PRED_LOG_PATH = "predictions_log.csv"
FUND_CSV_CACHE_TTL = 3600
NEWS_CACHE_TTL = 60 * 10
MIN_HISTORY_DAYS_FOR_FEATURES = 30  # tune to include more tickers if needed

# ---------------- Persistence helpers ----------------
def ensure_log_exists(path=PRED_LOG_PATH):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=[
            "run_date","company","ticker","current",
            "pred_1m","pred_1y","ret_1m_pct","ret_1y_pct",
            "rank_ret_1m","rank_ret_1y","composite_rank",
            "actual_1m","actual_1m_date","err_pct_1m",
            "actual_1y","actual_1y_date","err_pct_1y"
        ])
        df.to_csv(path, index=False)

def read_pred_log(path=PRED_LOG_PATH):
    ensure_log_exists(path)
    return pd.read_csv(path, parse_dates=["run_date","actual_1m_date","actual_1y_date"])

def write_pred_log(df, path=PRED_LOG_PATH):
    df.to_csv(path, index=False)

# ---------------- Robust constituents fetch (smallcap/midcap) ----------------
# Primary CSV URLs (may be blocked occasionally)
INDEX_URLS = {
    "Nifty Midcap 100":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

# Local fallback lists (keeps app functional if remote fetch fails)
MIDCAP_FALLBACK = [
    "ABBOTINDIA","ALKEM","ASHOKLEY","AUBANK","AUROPHARMA","BALKRISIND","BEL","BERGEPAINT","BHEL","CANFINHOME",
    "CUMMINSIND","DALBHARAT","DEEPAKNTR","FEDERALBNK","GODREJPROP","HAVELLS","HINDZINC","IDFCFIRSTB","INDHOTEL",
    "INDIAMART","IPCALAB","JUBLFOOD","LUPIN","MANKIND","MUTHOOTFIN","NMDC","OBEROIRLTY","PAGEIND","PERSISTENT",
    "PFIZER","POLYCAB","RECLTD","SAIL","SRF","SUNTV","TATAELXSI","TATAPOWER","TRENT","TVSMOTOR","UBL","VOLTAS",
    "ZYDUSLIFE","PIIND","CONCOR","APOLLOTYRE","TORNTPHARM","MPHASIS","ASTRAL","OFSS","MINDTREE","CROMPTON"
]

SMALLCAP_FALLBACK = [
    "AARTIIND","AFFLE","AMBER","ANANDRATHI","APLAPOLLO","ARVINDFASN","ASTERDM","ASTRAZEN","BASF","BAJAJHLDNG",
    "BANDHANBNK","BORORENEW","CREDITACC","DIXON","EIHOTEL","GALAXYSURF","INTELLECT","JSL","KPRMILL","LAXMIMACH",
    "MASFIN","NAVINFLUOR","ORIENTELEC","PNCINFRA","QUESS","SHREECEM","TATACHEM","UNIONBANK","VARROC","WELSPUNIND"
]

def _csv_to_pairs_fuzzy(csv_text: str):
    # Try pandas; if fails, simple parsing
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
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115 Safari/537.36"}
    st.sidebar.write(f"Fetching constituents for {index_name} ...")
    debug_notes=[]
    if url:
        try:
            r = requests.get(url, headers=headers, timeout=8)
            debug_notes.append(f"status {r.status_code}")
            pairs = _csv_to_pairs_fuzzy(r.text)
            if pairs:
                st.sidebar.success(f"{index_name}: fetched {len(pairs)} entries from URL")
                return pairs
            else:
                debug_notes.append("parsed 0 pairs")
        except Exception as e:
            debug_notes.append(f"error: {type(e).__name__}:{e}")
    st.sidebar.warning(f"{index_name}: falling back to local list. Debug: {'; '.join(debug_notes)}")
    if "smallcap" in index_name.lower():
        return [(s, f"{s}.NS") for s in SMALLCAP_FALLBACK]
    return [(s, f"{s}.NS") for s in MIDCAP_FALLBACK]

# ---------------- Fundamentals ingestion ----------------
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

# ---------------- VIX & risk ----------------
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
    if horizon=="1D": return float(np.clip((15-vix)*0.4, -5,5))
    if horizon=="1M": return float(np.clip((15-vix)*0.8, -10,10))
    return float(np.clip((15-vix)*1.2, -20,20))

# ---------------- Geo-news (RSS + VADER) ----------------
analyzer = SentimentIntensityAnalyzer()
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
            score = analyzer.polarity_scores(text)["compound"] if text else 0.0
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

# ---------------- Batch price fetch ----------------
@st.cache_data(show_spinner=False)
def batch_history(tickers, years=4):
    return yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False, threads=True, group_by="ticker")

# ---------------- Feature extraction ----------------
def compute_hidden_features(s: pd.Series):
    s = s.dropna().astype(float)
    if s.size < MIN_HISTORY_DAYS_FOR_FEATURES:
        return None
    cur = float(s.iloc[-1])
    def mom(days):
        if s.size <= days: return np.nan
        return (s.iloc[-1]/s.iloc[-days] - 1.0) * 100.0
    m1=mom(1); m5=mom(5); m21=mom(21); m63=mom(63)
    try: m252=mom(252)
    except: m252=np.nan
    vol21 = s.pct_change().rolling(21).std().iloc[-1]
    if np.isnan(vol21): vol21 = s.pct_change().std()
    short_w=min(20,s.size); mid_w=min(50,s.size)
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
        prox52=50.0
    rets = s.pct_change().dropna().tail(60)
    skew60 = float(rets.skew()) if len(rets)>5 else 0.0
    kurt60 = float(rets.kurtosis()) if len(rets)>5 else 0.0
    return {"current":cur,"m1":m1,"m5":m5,"m21":m21,"m63":m63,"m252":m252,"vol21":vol21,
            "ma_bias":ma_bias,"rsi14":rsi14,"macd_conf":macd_conf,"prox52":prox52,"skew60":skew60,"kurt60":kurt60}

# ---------------- Heuristics ----------------
def heuristic_ret1m_from_feats(feats):
    base = 0.6*(feats.get("m21",0.0) or 0.0) + 0.4*(feats.get("m63",0.0) or 0.0)
    adj = 1.0
    if feats.get("ma_bias",0) < 0: adj *= 0.78
    if feats.get("rsi14",50) > 70: adj *= 0.88
    if feats.get("rsi14",50) < 30: adj *= 1.12
    if feats.get("macd_conf",0) < 0: adj *= 0.93
    elif feats.get("macd_conf",0) > 0.5: adj *= 1.05
    out = np.clip(base*adj - 100*(feats.get("vol21",0.0) or 0.0), -50,50)
    return out

def heuristic_ret1y_from_feats(feats):
    m63 = feats.get("m63",0.0) or 0.0
    m252 = feats.get("m252", np.nan)
    m252_eff = m252 if m252==m252 else (m63*4 if m63==m63 else 0.0)
    out = np.clip(0.3*(m63 if m63==m63 else 0.0) + 0.7*m252_eff - 150*(feats.get("vol21",0.0) or 0.0), -80,120)
    return out

# ---------------- ML dataset & training helpers ----------------
@st.cache_data(ttl=86400)
def build_ml_dataset_with_news_and_fundamentals(price_map: dict, fundamentals_df: pd.DataFrame=None, min_history_days=60, recent_only_years=None):
    rows=[]
    all_dates = sorted({d for ser in price_map.values() if ser is not None and not ser.empty for d in ser.index})
    if not all_dates: return pd.DataFrame(rows)
    all_dates = pd.DatetimeIndex(all_dates)
    months = sorted({(d.year,d.month) for d in all_dates})
    run_dates=[]
    for y,m in months:
        d = all_dates[(all_dates.year==y)&(all_dates.month==m)]
        if len(d): run_dates.append(d.max())
    run_dates = sorted(run_dates)
    HOLD_1M=21; HOLD_1Y=252
    for run_date in run_dates:
        if recent_only_years:
            if (datetime.today().year - run_date.year) > recent_only_years: continue
        geo_news = get_geo_news_features()
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
            fut1m = idx + HOLD_1M; fut1y = idx + HOLD_1Y
            if fut1m >= len(ser): continue
            entry = float(ser.iloc[idx]); future1m = float(ser.iloc[fut1m])
            real_ret_1m = (future1m/entry - 1.0) * 100.0; label_1m = 1 if real_ret_1m > 0 else 0
            if fut1y < len(ser):
                future1y = float(ser.iloc[fut1y]); real_ret_1y = (future1y/entry - 1.0)*100.0; label_1y = 1 if real_ret_1y>0 else 0
            else:
                future1y=np.nan; real_ret_1y=np.nan; label_1y=np.nan
            row = {"run_date":run_date,"ticker":ticker,"entry":entry,"future1m":future1m,"real_ret_1m":real_ret_1m,"label_1m":label_1m,
                   "future1y":future1y,"real_ret_1y":real_ret_1y,"label_1y":label_1y}
            row.update({k:feats[k] for k in ["m1","m5","m21","m63","vol21","ma_bias","rsi14","macd_conf","prox52","skew60","kurt60"]})
            row.update(geo_news)
            if fundamentals_df is not None:
                frow = None
                if ticker in fundamentals_df.index: frow = fundamentals_df.loc[ticker]
                else:
                    t_short = ticker.replace(".NS","")
                    alt_idx = [i for i in fundamentals_df.index if i.upper().startswith(t_short)]
                    if alt_idx: frow = fundamentals_df.loc[alt_idx[0]]
                if frow is not None:
                    for col in fundamentals_df.columns:
                        try:
                            val = frow.get(col,np.nan)
                        except:
                            try: val = frow[col]
                            except: val = np.nan
                        row[f"f_{col}"] = float(val) if (pd.notna(val) and isinstance(val,(int,float,np.number))) else (val if isinstance(val,str) else np.nan)
                else:
                    for col in fundamentals_df.columns:
                        row[f"f_{col}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)

@st.cache_data(ttl=86400)
def train_lgbm(df_train, features, num_boost_round=300):
    try:
        missing = [c for c in features if c not in df_train.columns]
        if missing:
            st.warning(f"train_lgbm: missing feature columns, skipping training. Missing: {missing}")
            return None, None
        X = df_train[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        y = pd.to_numeric(df_train["label"], errors="coerce").fillna(0).astype(int)
        if len(X) < 200:
            st.warning(f"Not enough rows to train (found {len(X)}). Need >=200.")
            return None, None
        Xs, ys = shuffle(X, y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=ys)
        try:
            model = lgb.LGBMClassifier(objective="binary", n_estimators=num_boost_round, learning_rate=0.05,
                                       num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                       random_state=42, n_jobs=1, verbosity=-1)
            try:
                model.fit(X_train, y_train, eval_set=[(X_val,y_val)], eval_metric="auc", early_stopping_rounds=25, verbose=False)
            except TypeError:
                try:
                    model.fit(X_train, y_train)
                except Exception:
                    raise RuntimeError("sklearn-fit-failed")
            try:
                y_proba = model.predict_proba(X_val)[:,1]
                auc = roc_auc_score(y_val, y_proba) if len(np.unique(y_val))>1 else float("nan")
            except Exception:
                auc = float("nan")
            return model, auc
        except Exception:
            try:
                lgb_params = {"objective":"binary","metric":"auc","learning_rate":0.05,"num_leaves":31,"min_data_in_leaf":20,
                              "feature_fraction":0.8,"bagging_fraction":0.8,"verbose":-1,"seed":42}
                dtrain = lgb.Dataset(X_train, label=y_train); dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                callbacks = [lgb.early_stopping(stopping_rounds=25), lgb.log_evaluation(period=0)]
                booster = lgb.train(lgb_params, dtrain, num_boost_round=num_boost_round, valid_sets=[dval], callbacks=callbacks)
                y_proba = booster.predict(X_val)
                auc = roc_auc_score(y_val, y_proba) if len(np.unique(y_val))>1 else float("nan")
                return booster, auc
            except Exception as e_raw:
                st.error(f"train_lgbm: failed with raw lgb.train: {type(e_raw).__name__}: {e_raw}")
                return None, None
    except Exception as e:
        st.error(f"ML training failed: {type(e).__name__}: {e}")
        return None, None

def ml_predict_prob(model, feats_row, features):
    if model is None: raise ValueError("ml_predict_prob: model is None")
    X = pd.DataFrame([feats_row])[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    if hasattr(model,"predict_proba"):
        return float(model.predict_proba(X)[:,1][0])
    if hasattr(model,"predict"):
        p = model.predict(X)
        if isinstance(p, np.ndarray): return float(p[0])
        return float(p)
    raise RuntimeError("Model does not support probability prediction")

def prob_to_return_1m(p): return (p - 0.5) * 40.0
def prob_to_return_1y(p): return (p - 0.5) * 200.0

# ---------------- Actuals helpers ----------------
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
    if df.empty: return df
    now = datetime.utcnow().date() if now_date is None else pd.to_datetime(now_date).date()
    updated = False
    for idx,row in df.iterrows():
        try: run_date = pd.to_datetime(row["run_date"]).date()
        except Exception: continue
        needs_1m = pd.isna(row.get("actual_1m")) or force
        target_1m = run_date + timedelta(days=30)
        if needs_1m and target_1m <= now:
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
    if updated: write_pred_log(df)
    return df

# ---------------- UI Controls ----------------
st.sidebar.header("Options")
index_choice = st.sidebar.selectbox("Index", list(INDEX_URLS.keys()))
companies = fetch_constituents(index_choice)
if not companies:
    st.sidebar.error("No constituents available"); st.stop()

# Safe slider (no StreamlitAPIException)
n_companies = len(companies)
min_tickers = 1 if n_companies < 10 else 10
max_tickers = max(1, n_companies)
default_tickers = max_tickers
step = 1 if max_tickers - min_tickers <= 10 else 5

limit = st.sidebar.slider(
    "Tickers to process",
    min_value=min_tickers,
    max_value=max_tickers,
    value=default_tickers,
    step=step
)

enable_ml = st.sidebar.checkbox("Enable ML (train & use)", value=False)
train_recent = st.sidebar.checkbox("Train on recent N years (faster)", value=True)
fund_csv_url = st.sidebar.text_input("Optional fundamentals CSV URL (public raw CSV)", value="")
refresh_news = st.sidebar.button("Refresh News Now")
force_actuals_refresh = st.sidebar.button("Force refresh actuals")

# ---------------- Geo-news snapshot (visible) ----------------
vix = fetch_vix()
adj1m = vix_to_adj(vix, "1M"); adj1y = vix_to_adj(vix, "1Y")
st.caption(f"India VIX = {vix if vix else 'N/A'} → Risk adj: 1M {adj1m:+.2f}%, 1Y {adj1y:+.2f}%")
geo_snapshot = get_geo_news_features()
try:
    st.markdown("### 🔎 Geo-political News Snapshot (rolling windows)")
    st.caption("VADER sentiment avg and headline volume for 3d / 7d / 30d; topic breakdown (7d); geo_news_risk.")
    regions=["us","eu","cn"]
    cols=st.columns([1,1,1,0.8])
    for i, region in enumerate(regions):
        col=cols[i]; col.markdown(f"**{region.upper()}**")
        col.metric("3d avg", f"{geo_snapshot.get(f'news_{region}_avg_3d',0.0):+.2f}", f"vol {int(geo_snapshot.get(f'news_{region}_vol_3d',0))}")
        col.metric("7d avg", f"{geo_snapshot.get(f'news_{region}_avg_7d',0.0):+.2f}", f"vol {int(geo_snapshot.get(f'news_{region}_vol_7d',0))}")
        col.metric("30d avg", f"{geo_snapshot.get(f'news_{region}_avg_30d',0.0):+.2f}", f"vol {int(geo_snapshot.get(f'news_{region}_vol_30d',0))}")
    risk_col=cols[-1]; risk_col.markdown("**Geo-news risk**")
    risk_col.metric("Risk", f"{geo_snapshot.get('geo_news_risk',0.0):.2f}", "Higher → more negative headlines")
    st.markdown("**Topic breakdown (7d)**")
    topic_rows=[]
    for r in regions:
        row={"region":r.upper()}
        for t in TOPIC_KEYWORDS.keys():
            row[t]=round(geo_snapshot.get(f"news_{r}_topic_{t}_7d",0.0),3)
        topic_rows.append(row)
    st.table(pd.DataFrame(topic_rows).set_index("region"))
except Exception:
    st.warning("Geo-news snapshot failed to render (network/RSS may be blocked).")

# ---------------- Load fundamentals if provided ----------------
fund_df=None
if fund_csv_url:
    st.info("Loading fundamentals CSV...")
    fund_df = fetch_fundamentals_csv(fund_csv_url)
    if fund_df is None:
        st.warning("Could not load fundamentals CSV; continuing without fundamentals.")
    else:
        st.success(f"Loaded fundamentals for {len(fund_df)} tickers.")

# ---------------- Fetch price history and build price_map robustly ----------------
tickers_subset = [t for _, t in companies[:limit]]
st.info(f"Fetching price history for {len(tickers_subset)} tickers ...")
price_data = batch_history(tickers_subset, years=4)
price_map={}
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
        try:
            tmp = yf.download(t, period="4y", auto_adjust=True, progress=False)
            ser = tmp["Close"].dropna() if (tmp is not None and "Close" in tmp.columns) else pd.Series(dtype=float)
            ser.index = pd.to_datetime(ser.index)
            price_map[t] = ser.sort_index()
        except Exception:
            price_map[t] = pd.Series(dtype=float)

# ---------------- Prepare feature lists ----------------
NEWS_KEYS=[]
for r in ["us","eu","cn"]:
    for w in ["3d","7d","30d"]:
        NEWS_KEYS += [f"news_{r}_avg_{w}", f"news_{r}_vol_{w}"]
    for topic in TOPIC_KEYWORDS.keys():
        for w in ["3d","7d","30d"]:
            NEWS_KEYS.append(f"news_{r}_topic_{topic}_{w}")
NEWS_KEYS.append("geo_news_risk")
CORE_FEATURES=["m1","m5","m21","m63","vol21","ma_bias","rsi14","macd_conf","prox52","skew60","kurt60"]
FEATURES_1M = CORE_FEATURES + NEWS_KEYS
FEATURES_1Y = CORE_FEATURES + NEWS_KEYS

fund_numeric_cols=[]
if fund_df is not None:
    fund_numeric_cols=[c for c in fund_df.columns if pd.api.types.is_numeric_dtype(fund_df[c])]
    fund_numeric_cols=fund_numeric_cols[:8]
    FUND_FEATS=[f"f_{c}" for c in fund_numeric_cols]
    FEATURES_1M += FUND_FEATS; FEATURES_1Y += FUND_FEATS

# ---------------- ML training (optional) ----------------
model_1m = model_1y = None
auc1m = auc1y = None
if enable_ml and price_map:
    recent_years = 3 if train_recent else None
    st.info("Building ML dataset (may take time)...")
    df_ml = build_ml_dataset_with_news_and_fundamentals(price_map, fundamentals_df=fund_df, min_history_days=60, recent_only_years=recent_years)
    st.write("ML dataset shape:", df_ml.shape)
    if not df_ml.empty:
        if "label_1m" in df_ml.columns:
            st.write("label_1m counts:", df_ml["label_1m"].value_counts(dropna=True).to_dict())
        if "label_1y" in df_ml.columns:
            st.write("label_1y counts:", df_ml["label_1y"].dropna().astype(int).value_counts().to_dict())
    df1m = df_ml.dropna(subset=["label_1m"] + FEATURES_1M) if not df_ml.empty else pd.DataFrame()
    df1y = df_ml.dropna(subset=["label_1y"] + FEATURES_1Y) if not df_ml.empty else pd.DataFrame()
    if not df1m.empty:
        df1m = df1m.rename(columns={"label_1m":"label"})
        model_1m, auc1m = train_lgbm(df1m, FEATURES_1M)
    if not df1y.empty:
        df1y = df1y.rename(columns={"label_1y":"label"})
        model_1y, auc1y = train_lgbm(df1y, FEATURES_1Y)
    st.success(f"ML training done — AUC 1M: {auc1m if auc1m is not None else 'N/A'}, AUC 1Y: {auc1y if auc1y is not None else 'N/A'}")

# ---------------- Build predictions with fallback ----------------
rows=[]; log_rows=[]
geo_news_snapshot = get_geo_news_features()
skipped_debug=[]

for tkr, ser in price_map.items():
    if ser is None or ser.empty:
        skipped_debug.append((tkr,"no_data")); continue
    feats = compute_hidden_features(ser)
    if feats is None:
        if len(ser) >= 6:
            cur = float(ser.iloc[-1])
            mom5 = (cur / float(ser.iloc[-6]) - 1.0) * 100.0
            vol = ser.pct_change().tail(5).std() if len(ser.pct_change().dropna())>=1 else 0.01
            feats = {"current": cur, "m1": None, "m5": mom5, "m21": mom5, "m63": mom5, "m252": mom5,
                     "vol21": vol, "ma_bias": 1.0, "rsi14": 50.0, "macd_conf": 0.0, "prox52":50.0, "skew60":0.0, "kurt60":0.0}
        else:
            skipped_debug.append((tkr, f"insufficient_history({len(ser)})")); continue

    h_r1m = heuristic_ret1m_from_feats(feats); h_r1y = heuristic_ret1y_from_feats(feats)
    ml_r1m = ml_r1y = None

    if model_1m is not None:
        try:
            feats_now = {k: feats.get(k, 0.0) for k in CORE_FEATURES}
            feats_now.update(geo_news_snapshot)
            if fund_df is not None:
                if tkr in fund_df.index: frow = fund_df.loc[tkr]
                else:
                    alt = tkr.replace(".NS",""); alt_idx = [i for i in fund_df.index if i.upper().startswith(alt)]
                    frow = fund_df.loc[alt_idx[0]] if alt_idx else None
                for col in fund_numeric_cols:
                    feats_now[f"f_{col}"] = float(frow.get(col,0.0)) if (frow is not None and pd.notna(frow.get(col,np.nan))) else 0.0
            p1m = ml_predict_prob(model_1m, feats_now, FEATURES_1M)
            ml_r1m = prob_to_return_1m(p1m)
        except Exception:
            ml_r1m = None

    if model_1y is not None:
        try:
            feats_now = {k: feats.get(k, 0.0) for k in CORE_FEATURES}
            feats_now.update(geo_news_snapshot)
            if fund_df is not None:
                if tkr in fund_df.index: frow = fund_df.loc[tkr]
                else:
                    alt = tkr.replace(".NS",""); alt_idx = [i for i in fund_df.index if i.upper().startswith(alt)]
                    frow = fund_df.loc[alt_idx[0]] if alt_idx else None
                for col in fund_numeric_cols:
                    feats_now[f"f_{col}"] = float(frow.get(col,0.0)) if (frow is not None and pd.notna(frow.get(col,np.nan))) else 0.0
            p1y = ml_predict_prob(model_1y, feats_now, FEATURES_1Y)
            ml_r1y = prob_to_return_1y(p1y)
        except Exception:
            ml_r1y = None

    final_r1m = ml_r1m if ml_r1m is not None else h_r1m
    final_r1y = ml_r1y if ml_r1y is not None else h_r1y
    r1m_adj = final_r1m * (1 + adj1m / 100.0)
    r1y_adj = final_r1y * (1 + adj1y / 100.0)
    cur = feats["current"]
    pred1m_price = round(cur * (1 + r1m_adj/100.0), 2)
    pred1y_price = round(cur * (1 + r1y_adj/100.0), 2)

    rows.append({"Company": tkr.replace(".NS",""), "Ticker":tkr, "Current":round(cur,2),
                 "Pred 1M":pred1m_price, "Pred 1Y":pred1y_price, "Ret 1M %":round(r1m_adj,2), "Ret 1Y %":round(r1y_adj,2)})
    log_rows.append({"run_date":datetime.utcnow(), "company":tkr.replace(".NS",""), "ticker":tkr, "current":round(cur,2),
                     "pred_1m":pred1m_price, "pred_1y":pred1y_price, "ret_1m_pct":round(r1m_adj,2), "ret_1y_pct":round(r1y_adj,2)})

# Debug skipped
if skipped_debug:
    st.info(f"Skipped {len(skipped_debug)} tickers due to missing data or insufficient history.")
    st.dataframe(pd.DataFrame(skipped_debug, columns=["ticker","reason"]).head(200))

if not rows:
    st.error("No tickers processed (insufficient data)."); st.stop()

out = pd.DataFrame(rows)
out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
out["Composite Rank"] = ((out["Rank Ret 1M"] + out["Rank Ret 1Y"]) / 2.0).rank(ascending=True, method="min").astype(int)
final = out[["Company","Ticker","Current","Pred 1M","Pred 1Y","Ret 1M %","Ret 1Y %","Composite Rank","Rank Ret 1M","Rank Ret 1Y"]].sort_values("Composite Rank").reset_index(drop=True)

st.subheader("📊 Ranked Screener")
def color_ret(v):
    if v>0: return "color: green"
    if v<0: return "color: red"
    return ""
st.dataframe(final.style.applymap(color_ret, subset=["Ret 1M %","Ret 1Y %"]), use_container_width=True)

# ---------------- Append to predictions log and compute run_date ranks ----------------
ensure_log_exists()
existing = read_pred_log()
new_entries = pd.DataFrame(log_rows)
if not new_entries.empty:
    combined = pd.concat([existing, new_entries], ignore_index=True, sort=False)
    try:
        combined["run_date"] = pd.to_datetime(combined["run_date"])
        ranked=[]
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
st.success("Predictions computed and appended to integrated log.")

# ---------------- Update actuals ----------------
st.info("Updating integrated log with matured actuals...")
if force_actuals_refresh:
    log_df = update_log_with_actuals(PRED_LOG_PATH, force=True)
else:
    log_df = update_log_with_actuals(PRED_LOG_PATH, force=False)
st.success("Historical actuals update complete.")

# ---------------- Historical table & summary ----------------
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

# ---------------- Portfolio builder ----------------
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

st.caption("Notes: If smallcap fetch fails due to provider restrictions the app falls back to a local list (sidebar shows debug info). ML training can be heavy — enable only when required.")
