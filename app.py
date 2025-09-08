# app.py
# NSE Screener with ML (1M & 1Y) + Geo-News features (RSS + VADER)
# Complete single-file application with robust LightGBM training & callback fallback.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
import time
import re

# ML & NLP libs
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

st.set_page_config(page_title="NSE Screener + ML + Geo-News", layout="wide")
st.title("⚡ NSE Midcap / Smallcap Screener — ML (1M & 1Y) + Geo-News")

# -----------------------------
# Index constituent sources & fallbacks
# -----------------------------
INDEX_URLS = {
    "Nifty Midcap 100":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

MIDCAP_FALLBACK = [
    "ABBOTINDIA","ALKEM","ASHOKLEY","AUBANK","AUROPHARMA","BALKRISIND","BEL","BERGEPAINT","BHEL","CANFINHOME",
    "CUMMINSIND","DALBHARAT","DEEPAKNTR","FEDERALBNK","GODREJPROP","HAVELLS","HINDZINC","IDFCFIRSTB","INDHOTEL",
    "INDIAMART","IPCALAB","JUBLFOOD","LUPIN","MANKIND","MUTHOOTFIN","NMDC","OBEROIRLTY","PAGEIND","PERSISTENT",
    "PFIZER","POLYCAB","RECLTD","SAIL","SRF","SUNTV","TATAELXSI","TATAPOWER","TRENT","TVSMOTOR","UBL","VOLTAS",
    "ZYDUSLIFE","PIIND","CONCOR","APOLLOTYRE","TORNTPHARM","MPHASIS","ASTRAL","OFSS","MINDTREE","CROMPTON"
]

SMALLCAP250_FALLBACK = [
    "AARTIIND","AFFLE","AMBER","ANANDRATHI","APLAPOLLO","ARVINDFASN","ASTERDM","ASTRAZEN","BASF","BAJAJHLDNG",
    "BALAMINES","BEML","BECTORFOOD","BLUESTARCO","BSE","CESC","CHEMPLASTS","COFORGE","CYIENT","DATAPATTNS",
    "DCMSHRIRAM","DEEPAKFERT","DEVYANI","EIHOTEL","ENIL","EPL","FDC","GESHIP","GLAND","GLS","GRINDWELL","HAPPSTMNDS",
    "HATHWAY","HGS","IRCON","ISEC","JBCHEPHARM","JCHAC","JKLAKSHMI","JYOTHYLAB","KAJARIACER","KEC","KIRLOSBROS",
    "KIRLOSIND","LAOPALA","LATENTVIEW","LEMONTREE","LUXIND","MAHLIFE","MMTC","NAZARA","NESCO","NOCIL","PAYTM","RADICO",
    "RAILTEL","RAIN","RATEGAIN","REDINGTON","RTNINDIA","SANOFI","SAPPHIRE","SONATSOFTW","STARHEALTH","SUNCLAYLTD",
    "SUNDRMFAST","SUNDARMFIN","SUVENPHAR","SYNGENE","TANLA","TCIEXP","TIDEWATER","TTML","UJJIVAN","VGUARD","VSTIND",
    "WELSPUNIND","ZYDUSWELL"
]

def _csv_to_pairs(csv_text: str):
    try:
        df = pd.read_csv(StringIO(csv_text))
        names = df["Company Name"].astype(str).str.strip().tolist()
        tks = df["Symbol"].astype(str).str.strip().apply(lambda s: f"{s}.NS").tolist()
        return list(zip(names, tks))
    except Exception:
        return []

@st.cache_data(ttl=86400)
def fetch_constituents(index_name: str):
    url = INDEX_URLS[index_name]
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        pairs = _csv_to_pairs(r.text)
        if pairs:
            return pairs
    except Exception:
        pass
    # fallback
    if index_name == "Nifty Midcap 100":
        return [(s, f"{s}.NS") for s in MIDCAP_FALLBACK]
    else:
        return [(s, f"{s}.NS") for s in SMALLCAP250_FALLBACK]

# -----------------------------
# VIX & risk mapping
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_vix():
    try:
        df = yf.download("^INDIAVIX", period="7d", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

def vix_to_adj(vix, horizon):
    if vix is None:
        return 0.0
    if horizon == "1D":
        return float(np.clip((15 - vix) * 0.4, -5, 5))
    elif horizon == "1M":
        return float(np.clip((15 - vix) * 0.8, -10, 10))
    else:
        return float(np.clip((15 - vix) * 1.2, -20, 20))

# -----------------------------
# Batch price fetch (yfinance)
# -----------------------------
@st.cache_data(show_spinner=False)
def batch_history(tickers, years=4):
    return yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False, threads=True, group_by="ticker")

# -----------------------------
# Hidden technical feature extractor (fast)
# -----------------------------
def compute_hidden_features(s: pd.Series):
    s = s.dropna().astype(float)
    if s.size < 60:
        return None
    cur = float(s.iloc[-1])
    def mom(days):
        if s.size <= days:
            return np.nan
        return (s.iloc[-1] / s.iloc[-days] - 1.0) * 100.0
    m1 = mom(1); m5 = mom(5); m21 = mom(21); m63 = mom(63)
    try:
        m252 = mom(252)
    except Exception:
        m252 = np.nan
    vol21 = s.pct_change().rolling(21).std().iloc[-1]
    if np.isnan(vol21):
        vol21 = s.pct_change().std()
    short_w = min(20, s.size); mid_w = min(50, s.size); long_w = min(200, s.size)
    ma_short = s.rolling(short_w).mean().iloc[-1] if s.size >= short_w else np.nan
    ma_mid = s.rolling(mid_w).mean().iloc[-1] if s.size >= mid_w else np.nan
    ma_long = s.rolling(long_w).mean().iloc[-1] if s.size >= long_w else np.nan
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
        high52 = float(s[-252:].max())
        low52 = float(s[-252:].min())
        prox52 = float((cur - low52) / (high52 - low52 + 1e-9) * 100.0)
    else:
        prox52 = 50.0
    rets = s.pct_change().dropna().tail(60)
    skew60 = float(rets.skew()) if len(rets) > 5 else 0.0
    kurt60 = float(rets.kurtosis()) if len(rets) > 5 else 0.0
    return {
        "current": cur, "m1": m1, "m5": m5, "m21": m21, "m63": m63, "m252": m252,
        "vol21": vol21, "ma_bias": ma_bias, "rsi14": rsi14, "macd_conf": macd_conf,
        "prox52": prox52, "skew60": skew60, "kurt60": kurt60
    }

# -----------------------------
# Heuristic 1M & 1Y (used if ML disabled / fallback)
# -----------------------------
def heuristic_ret1m_from_feats(feats):
    base_1m = 0.6 * (feats["m21"] if feats["m21"]==feats["m21"] else 0.0) + 0.4 * (feats["m63"] if feats["m63"]==feats["m63"] else 0.0)
    adj_factor = 1.0
    if feats["ma_bias"] < 0: adj_factor *= 0.78
    if feats["rsi14"] > 70: adj_factor *= 0.88
    if feats["rsi14"] < 30: adj_factor *= 1.12
    if feats["macd_conf"] < 0: adj_factor *= 0.93
    elif feats["macd_conf"] > 0.5: adj_factor *= 1.05
    adj_base = base_1m * adj_factor
    ret1m = np.clip(adj_base - 100 * feats["vol21"], -50, 50)
    return ret1m

def heuristic_ret1y_from_feats(feats):
    m63 = feats.get("m63", 0.0)
    m252 = feats.get("m252", np.nan)
    m252_eff = m252 if m252==m252 else (m63*4 if m63==m63 else 0.0)
    ret1y = np.clip(0.3*(m63 if m63==m63 else 0.0) + 0.7*m252_eff - 150*feats["vol21"], -80, 120)
    return ret1y

# -----------------------------
# Geo-news prototype (RSS + VADER)
# -----------------------------
analyzer = SentimentIntensityAnalyzer()
RSS_FEEDS = {
    "us": [
        "https://www.reuters.com/rssFeed/businessNews",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    ],
    "eu": [
        "https://www.reuters.com/rssFeed/europeNews",
        "https://www.ft.com/?format=rss"
    ],
    "cn": [
        "https://www.reuters.com/places/china/rss",
        "https://www.scmp.com/rss/0/feed"
    ],
}
TOPIC_KEYWORDS = {
    "oil": ["oil","opec","crude","gasoline","petroleum","energy"],
    "fed_rate": ["fed","federal reserve","rate hike","interest rate","fed chair"],
    "china_trade": ["china","export","tariff","trade","supply chain"],
    "sanctions": ["sanction","sanctions"],
    "war": ["war","invasion","conflict","attack","tension"],
    "inflation": ["inflation","cpi","consumer price"]
}

def _clean_text(t: str) -> str:
    return re.sub(r'\s+', ' ', (t or "").strip().lower())

def _fetch_feed_items(url: str, max_items=12):
    try:
        f = feedparser.parse(url)
        entries = f.entries[:max_items]
        items = []
        for e in entries:
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or ""
            items.append({"title": _clean_text(title), "summary": _clean_text(summary)})
        time.sleep(0.05)
        return items
    except Exception:
        return []

def _analyze_headlines(items):
    if not items:
        return {"avg_compound": 0.0, "count": 0, "topic_counts": {}}
    scores = []
    from collections import Counter
    topic_counter = Counter()
    for it in items:
        text = (it["title"] + " " + it["summary"]).strip()
        if not text: continue
        vs = analyzer.polarity_scores(text)
        scores.append(vs["compound"])
        for topic, kws in TOPIC_KEYWORDS.items():
            for kw in kws:
                if kw in text:
                    topic_counter[topic] += 1
                    break
    avg_compound = float(sum(scores)/len(scores)) if scores else 0.0
    return {"avg_compound": avg_compound, "count": len(items), "topic_counts": dict(topic_counter)}

@st.cache_data(ttl=60*60)
def get_geo_news_features():
    out = {}
    for region in ["us","eu","cn"]:
        feeds = RSS_FEEDS.get(region, [])
        items = []
        for url in feeds:
            items.extend(_fetch_feed_items(url, max_items=12))
        stats = _analyze_headlines(items)
        out[f"news_{region}_sentiment"] = stats["avg_compound"]
        out[f"news_{region}_volume"] = stats["count"]
        for topic in TOPIC_KEYWORDS.keys():
            key = f"news_{region}_topic_{topic}"
            out[key] = stats["topic_counts"].get(topic, 0) / max(1, stats["count"])
    neg_sum = 0.0
    for r in ["us","eu","cn"]:
        s = out.get(f"news_{r}_sentiment", 0.0)
        v = out.get(f"news_{r}_volume", 0)
        neg_sum += max(0, -s) * (1 + v/20.0)
    out["geo_news_risk"] = float(min(5.0, neg_sum))
    return out

# -----------------------------
# ML dataset builder (with geo-news) - FIXED version (skips empty series safely)
# -----------------------------
@st.cache_data(ttl=86400)
def build_ml_dataset_with_news(price_map: dict, min_history_days=90, recent_only_years=None):
    rows = []
    # union of dates -> month-ends
    all_dates = sorted({d for ser in price_map.values() if ser is not None and not ser.empty for d in ser.index})
    if not all_dates:
        return pd.DataFrame(rows)
    all_dates = pd.DatetimeIndex(all_dates)
    months = sorted({(d.year, d.month) for d in all_dates})
    run_dates = []
    for y,m in months:
        d = all_dates[(all_dates.year==y) & (all_dates.month==m)]
        if len(d):
            run_dates.append(d.max())
    run_dates = sorted(run_dates)

    HOLD_1M = 21
    HOLD_1Y = 252

    for run_date in run_dates:
        if recent_only_years:
            if (datetime.today().year - run_date.year) > recent_only_years:
                continue

        geo_news = get_geo_news_features()

        for ticker, ser in price_map.items():
            if ser is None or ser.empty:
                continue
            if run_date < ser.index.min():
                continue

            ser_up = ser[:run_date]
            if len(ser_up) < min_history_days:
                continue

            feats = compute_hidden_features(ser_up)
            if feats is None:
                continue

            idx = ser.index.searchsorted(run_date)
            if idx == len(ser) or ser.index[idx] != run_date:
                if idx == 0:
                    continue
                idx = idx - 1

            fut1m = idx + HOLD_1M
            fut1y = idx + HOLD_1Y
            if fut1m >= len(ser):
                continue

            entry = float(ser.iloc[idx])
            future1m = float(ser.iloc[fut1m])
            real_ret_1m = (future1m / entry - 1.0) * 100.0
            label_1m = 1 if real_ret_1m > 0 else 0

            if fut1y < len(ser):
                future1y = float(ser.iloc[fut1y])
                real_ret_1y = (future1y / entry - 1.0) * 100.0
                label_1y = 1 if real_ret_1y > 0 else 0
            else:
                future1y = np.nan
                real_ret_1y = np.nan
                label_1y = np.nan

            row = {
                "run_date": run_date,
                "ticker": ticker,
                "entry": entry,
                "future1m": future1m,
                "real_ret_1m": real_ret_1m,
                "label_1m": label_1m,
                "future1y": future1y,
                "real_ret_1y": real_ret_1y,
                "label_1y": label_1y,
                **{
                    "m1": feats["m1"], "m5": feats["m5"], "m21": feats["m21"], "m63": feats["m63"],
                    "vol21": feats["vol21"], "ma_bias": feats["ma_bias"], "rsi14": feats["rsi14"],
                    "macd_conf": feats["macd_conf"], "prox52": feats["prox52"],
                    "skew60": feats["skew60"], "kurt60": feats["kurt60"]
                },
                **geo_news
            }
            rows.append(row)

    return pd.DataFrame(rows)

# -----------------------------
# Robust ML train & predict helpers (safe dtypes, sklearn wrapper, callback fallback)
# -----------------------------
@st.cache_data(ttl=86400)
def train_lgbm(df_train, features, num_boost_round=300):
    """
    Robust LightGBM training wrapper:
    - coerces feature columns to numeric (NaN -> 0.0)
    - uses LGBMClassifier for predict_proba
    - handles LightGBM versions that don't accept early_stopping_rounds by using callbacks
    - returns (model, auc) or (None, None) on failure
    """
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

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=num_boost_round,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=-1
        )

        # Try early_stopping_rounds; if unsupported, fall back to callback form
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                early_stopping_rounds=25,
                verbose=False
            )
        except TypeError:
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(25)],
                    verbose=False
                )
            except Exception as e2:
                st.error(f"train_lgbm: failed to fit model using callbacks: {type(e2).__name__}: {e2}")
                return None, None

        try:
            y_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else float("nan")
        except Exception:
            auc = float("nan")

        return model, auc

    except Exception as e:
        st.error(f"ML training failed (train_lgbm): {type(e).__name__}: {e}")
        return None, None

def ml_predict_prob(model, feats_row, features):
    """
    Unified predict-prob wrapper for both sklearn LGBMClassifier and raw booster.
    """
    if model is None:
        raise ValueError("ml_predict_prob: model is None")

    X = pd.DataFrame([feats_row])[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1][0]
        return float(proba)

    if hasattr(model, "predict"):
        proba = model.predict(X)
        if isinstance(proba, np.ndarray):
            return float(proba[0])
        return float(proba)

    raise RuntimeError("Model does not support probability prediction")

def prob_to_return_1m(p):
    return (p - 0.5) * 40.0

def prob_to_return_1y(p):
    return (p - 0.5) * 200.0

# -----------------------------
# UI controls
# -----------------------------
index_choice = st.selectbox("Choose Index", list(INDEX_URLS.keys()))
companies = fetch_constituents(index_choice)
if not companies:
    st.stop()

col_a, col_b, col_c = st.columns([2,1,1])
with col_a:
    limit = st.slider("Tickers to process", 10, len(companies), min(50, len(companies)), step=5)
with col_b:
    enable_ml = st.checkbox("Enable ML (train & use)", value=False)
with col_c:
    train_recent = st.checkbox("Train on recent N years (faster)", value=True)

vix = fetch_vix()
adj1d = vix_to_adj(vix, "1D")
adj1m = vix_to_adj(vix, "1M")
adj1y = vix_to_adj(vix, "1Y")
vix_str = f"{vix:.2f}" if vix is not None else "N/A"
st.caption(f"📉 India VIX = {vix_str} → Base Risk Adj: 1D={adj1d:+.1f}%, 1M={adj1m:+.1f}%, 1Y={adj1y:+.1f}%")

# show geo-news summary and scale risk if needed
geo_news = get_geo_news_features()
st.markdown("**Geo-news (latest)**")
c1,c2,c3,c4 = st.columns(4)
c1.metric("US sentiment", f"{geo_news['news_us_sentiment']:+.2f}", f"vol {int(geo_news['news_us_volume'])}")
c2.metric("EU sentiment", f"{geo_news['news_eu_sentiment']:+.2f}", f"vol {int(geo_news['news_eu_volume'])}")
c3.metric("CN sentiment", f"{geo_news['news_cn_sentiment']:+.2f}", f"vol {int(geo_news['news_cn_volume'])}")
c4.metric("Geo-news risk", f"{geo_news['geo_news_risk']:.2f}")

risk_scale = 1.0
if geo_news["geo_news_risk"] >= 2.5:
    risk_scale = 0.5
elif geo_news["geo_news_risk"] >= 1.5:
    risk_scale = 0.75
adj1m *= risk_scale
adj1y *= risk_scale
st.caption(f"(Geo-news scaled risk by {risk_scale:.2f}; adjusted 1M adj now {adj1m:+.2f}%, 1Y adj {adj1y:+.2f}%)")

# -----------------------------
# ML dataset building & training (opt-in)
# -----------------------------
tickers_subset = [t for _, t in companies[:limit]]
st.info(f"Processing {len(tickers_subset)} tickers — this may take a while if ML is enabled.")

data = batch_history(tickers_subset, years=4)
rows = []
price_map = {}
for tkr in tickers_subset:
    try:
        ser = data[tkr]["Close"].dropna()
        ser.index = pd.to_datetime(ser.index)
        price_map[tkr] = ser.sort_index()
    except Exception:
        continue

FEATURES_NEWS = [
    "news_us_sentiment","news_eu_sentiment","news_cn_sentiment",
    "news_us_volume","news_eu_volume","news_cn_volume",
    "news_us_topic_oil","news_us_topic_fed_rate","news_us_topic_china_trade",
    "news_eu_topic_oil","news_eu_topic_fed_rate","news_eu_topic_china_trade",
    "news_cn_topic_oil","news_cn_topic_fed_rate","news_cn_topic_china_trade",
    "geo_news_risk"
]
CORE_FEATURES = ["m1","m5","m21","m63","vol21","ma_bias","rsi14","macd_conf","prox52","skew60","kurt60"]
FEATURES_1M = CORE_FEATURES + FEATURES_NEWS
FEATURES_1Y = CORE_FEATURES + FEATURES_NEWS

model_1m = model_1y = None
auc1m = auc1y = None
if enable_ml and price_map:
    recent_years = 3 if train_recent else None
    st.info("Building ML dataset (may take ~30-120s)...")
    df_ml = build_ml_dataset_with_news(price_map, min_history_days=90, recent_only_years=recent_years)
    df1m = df_ml.dropna(subset=["label_1m"] + FEATURES_1M)
    df1y = df_ml.dropna(subset=["label_1y"] + FEATURES_1Y)
    if not df1m.empty:
        df1m = df1m.rename(columns={"label_1m":"label"})
        model_1m, auc1m = train_lgbm(df1m, FEATURES_1M)
    if not df1y.empty:
        df1y = df1y.rename(columns={"label_1y":"label"})
        model_1y, auc1y = train_lgbm(df1y, FEATURES_1Y)
    st.success(f"ML training done — AUC 1M: {auc1m if auc1m is not None else 'N/A'}, AUC 1Y: {auc1y if auc1y is not None else 'N/A'}")

# -----------------------------
# Build UI rows using ML predictions (if available) or heuristics
# -----------------------------
for tkr, ser in price_map.items():
    feats = compute_hidden_features(ser)
    if feats is None:
        continue
    h_r1m = heuristic_ret1m_from_feats(feats)
    h_r1y = heuristic_ret1y_from_feats(feats)
    ml_r1m = ml_r1y = None
    if model_1m is not None:
        try:
            feats_now = {k: feats[k] for k in CORE_FEATURES}
            feats_now.update(get_geo_news_features())
            p1m = ml_predict_prob(model_1m, feats_now, FEATURES_1M)
            ml_r1m = prob_to_return_1m(p1m)
        except Exception:
            ml_r1m = None
    if model_1y is not None:
        try:
            feats_now = {k: feats[k] for k in CORE_FEATURES}
            feats_now.update(get_geo_news_features())
            p1y = ml_predict_prob(model_1y, feats_now, FEATURES_1Y)
            ml_r1y = prob_to_return_1y(p1y)
        except Exception:
            ml_r1y = None

    final_r1m = ml_r1m if ml_r1m is not None else h_r1m
    final_r1y = ml_r1y if ml_r1y is not None else h_r1y

    r1m_adj = final_r1m * (1 + adj1m / 100.0)
    r1y_adj = final_r1y * (1 + adj1y / 100.0)

    rows.append({
        "Company": tkr.replace(".NS",""),
        "Ticker": tkr,
        "Current": round(feats["current"], 2),
        "Pred 1M": round(feats["current"] * (1 + r1m_adj/100.0), 2),
        "Pred 1Y": round(feats["current"] * (1 + r1y_adj/100.0), 2),
        "Ret 1M %": round(r1m_adj, 2),
        "Ret 1Y %": round(r1y_adj, 2)
    })

if not rows:
    st.error("No rows to show. Increase tickers or check data.")
    st.stop()

out = pd.DataFrame(rows)
out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
out["Composite Rank"] = ((out["Rank Ret 1M"] + out["Rank Ret 1Y"]) / 2.0).rank(ascending=True, method="min").astype(int)
final = out[[
    "Company","Ticker","Current","Pred 1M","Pred 1Y","Ret 1M %","Ret 1Y %","Composite Rank","Rank Ret 1M","Rank Ret 1Y"
]].sort_values("Composite Rank").reset_index(drop=True)

st.subheader("📊 Ranked Screener (ML-enhanced if enabled)")
def color_ret(v):
    if v > 0: return "color: green"
    if v < 0: return "color: red"
    return ""
st.dataframe(final.style.applymap(color_ret, subset=["Ret 1M %","Ret 1Y %"]), use_container_width=True)
st.download_button("Download CSV", final.to_csv(index=False).encode(), f"{index_choice.lower().replace(' ','_')}_screener.csv", "text/csv")

# -----------------------------
# Simple Portfolio Builder (equal-weight)
# -----------------------------
st.markdown("## 📦 Portfolio Builder (Equal-weight)")
col1, col2, col3 = st.columns(3)
with col1:
    capital = st.number_input("Total capital (₹)", min_value=10000, value=1000000, step=10000)
with col2:
    top_n = st.slider("Number of holdings (Top N)", 3, min(5, len(final)), min(20, len(final)), step=1)
with col3:
    sl_pct = st.number_input("Stop-loss %", min_value=1, max_value=50, value=10)
tp_pct = st.number_input("Take-profit %", min_value=1, max_value=200, value=25)

pf = final.head(top_n).copy()
pf["Weight %"] = round(100.0 / top_n, 2)
pf["Alloc ₹"] = (capital * (pf["Weight %"]/100.0)).round(2)
pf["Shares"] = (pf["Alloc ₹"] / pf["Current"]).astype(int)
pf["Stop-loss"] = (pf["Current"] * (1 - sl_pct/100.0)).round(2)
pf["Take-profit"] = (pf["Current"] * (1 + tp_pct/100.0)).round(2)

st.subheader(f"Top {top_n} Portfolio Plan")
st.dataframe(pf[["Company","Ticker","Current","Weight %","Alloc ₹","Shares","Stop-loss","Take-profit"]], use_container_width=True)
st.download_button("Download Portfolio CSV", pf.to_csv(index=False).encode(), "portfolio_plan.csv", "text/csv")

st.caption("ML models are optional. If ML is enabled, the 1M & 1Y predictions are produced by LightGBM models trained on technical + geo-news features. If ML is disabled or fails, heuristics are used as fallback.")
