import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

st.set_page_config(page_title="NSE Mid/Small Screener — Fast + VIX Risk by Horizon", layout="wide")
st.title("⚡ NSE Midcap / Smallcap 250 Screener — Fast + VIX Risk (per horizon)")

# ---------------- Data sources ----------------
INDEX_URLS = {
    "Nifty Midcap 100":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

# --- Fallback lists (compact for speed/reliability) ---
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
    df = pd.read_csv(StringIO(csv_text))
    names = df["Company Name"].astype(str).str.strip().tolist()
    tks = df["Symbol"].astype(str).str.strip().apply(lambda s: f"{s}.NS").tolist()
    return list(zip(names, tks))

@st.cache_data(ttl=86400)
def fetch_constituents(index_name: str):
    url = INDEX_URLS[index_name]
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return _csv_to_pairs(r.text)
    except Exception:
        if index_name == "Nifty Midcap 100":
            return [(s, f"{s}.NS") for s in MIDCAP_FALLBACK]
        else:
            return [(s, f"{s}.NS") for s in SMALLCAP250_FALLBACK]

@st.cache_data(ttl=3600)
def fetch_vix():
    """Fetch latest India VIX close; None on failure."""
    try:
        df = yf.download("^INDIAVIX", period="5d", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

def vix_to_adj(vix, horizon):
    """Map VIX to risk adjustment (%) per horizon."""
    if vix is None:
        return 0.0
    if horizon == "1D":
        return float(np.clip((15 - vix) * 0.4, -5, 5))    # light touch
    elif horizon == "1M":
        return float(np.clip((15 - vix) * 0.8, -10, 10))  # medium
    else:  # "1Y"
        return float(np.clip((15 - vix) * 1.2, -20, 20))  # heavier

@st.cache_data(show_spinner=False)
def batch_history(tickers, years=3):
    return yf.download(
        tickers, period=f"{years}y", auto_adjust=False, progress=False,
        threads=True, group_by="ticker"
    )

def compute_signals(price_series: pd.Series):
    """Fast momentum/volatility heuristics → returns for 1D/1M/1Y."""
    s = price_series.dropna()
    if s.size < 80:
        return None
    y = s.values
    current = float(y[-1])

    def mom(days):
        if s.size <= days:
            return np.nan
        return (y[-1] / y[-days] - 1.0) * 100.0

    m1, m5, m21, m63, m252 = mom(1), mom(5), mom(21), mom(63), mom(252)

    vol21 = s.pct_change().rolling(21).std().iloc[-1]
    if np.isnan(vol21):
        vol21 = s.pct_change().std()

    # Heuristic returns (bounded), with volatility penalty
    ret1d = np.clip(0.7*(m1 if not np.isnan(m1) else 0.0) + 0.3*((m5 if not np.isnan(m5) else 0.0)/5.0) - 30*vol21, -8, 8)
    ret1m = np.clip(0.6*(m21 if not np.isnan(m21) else 0.0) + 0.4*(m63 if not np.isnan(m63) else 0.0) - 100*vol21, -50, 50)
    m252_eff = m252 if not np.isnan(m252) else (m63*4 if not np.isnan(m63) else 0.0)
    ret1y = np.clip(0.3*(m63 if not np.isnan(m63) else 0.0) + 0.7*m252_eff - 150*vol21, -80, 120)

    return dict(current=current, ret1d=ret1d, ret1m=ret1m, ret1y=ret1y)

# ---------------- UI ----------------
index_choice = st.selectbox("Choose Index", list(INDEX_URLS.keys()))
companies = fetch_constituents(index_choice)
if not companies:
    st.stop()

c1, c2 = st.columns([2, 2])
with c1:
    limit = st.slider("Tickers to process", 5, min(100, len(companies)), min(20, len(companies)), step=5)

# Auto risk adjustments from VIX (per horizon)
vix = fetch_vix()
adj1d = vix_to_adj(vix, "1D")
adj1m = vix_to_adj(vix, "1M")
adj1y = vix_to_adj(vix, "1Y")
vix_str = f"{vix:.2f}" if vix is not None else "N/A"
st.caption(f"📉 India VIX = {vix_str} → Auto Risk Adj: 1D={adj1d:+.1f}%, 1M={adj1m:+.1f}%, 1Y={adj1y:+.1f}%")

if st.button("Run (Fast)"):
    subset = companies[:limit]
    name_map = {tkr: name for name, tkr in subset}
    tickers = [tkr for _, tkr in subset]

    data = batch_history(tickers, years=3)
    rows = []

    if isinstance(data.columns, pd.MultiIndex):
        for tkr in tickers:
            cols = list(data[tkr].columns)
            price_col = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else None)
            if not price_col:
                continue

            sig = compute_signals(data[tkr][price_col])
            if not sig:
                continue

            # Apply per-horizon risk adjustments to returns
            r1d = sig["ret1d"] * (1 + adj1d / 100.0)
            r1m = sig["ret1m"] * (1 + adj1m / 100.0)
            r1y = sig["ret1y"] * (1 + adj1y / 100.0)

            rows.append({
                "Company": name_map.get(tkr, tkr.replace(".NS", "")),
                "Ticker": tkr,
                "Current": round(sig["current"], 2),
                "Pred 1D": round(sig["current"] * (1 + r1d / 100.0), 2),
                "Pred 1M": round(sig["current"] * (1 + r1m / 100.0), 2),
                "Pred 1Y": round(sig["current"] * (1 + r1y / 100.0), 2),
                "Ret 1D %": round(r1d, 2),
                "Ret 1M %": round(r1m, 2),
                "Ret 1Y %": round(r1y, 2),
            })
    else:
        # Single symbol case (unlikely with limit >1, but safe-guard)
        cols = list(data.columns)
        price_col = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else None)
        if price_col and tickers:
            tkr = tickers[0]
            sig = compute_signals(data[price_col])
            if sig:
                r1d = sig["ret1d"] * (1 + adj1d / 100.0)
                r1m = sig["ret1m"] * (1 + adj1m / 100.0)
                r1y = sig["ret1y"] * (1 + adj1y / 100.0)
                rows.append({
                    "Company": name_map.get(tkr, tkr.replace(".NS", "")),
                    "Ticker": tkr,
                    "Current": round(sig["current"], 2),
                    "Pred 1D": round(sig["current"] * (1 + r1d / 100.0), 2),
                    "Pred 1M": round(sig["current"] * (1 + r1m / 100.0), 2),
                    "Pred 1Y": round(sig["current"] * (1 + r1y / 100.0), 2),
                    "Ret 1D %": round(r1d, 2),
                    "Ret 1M %": round(r1m, 2),
                    "Ret 1Y %": round(r1y, 2),
                })

    if not rows:
        st.error("No results. Try more tickers.")
        st.stop()

    out = pd.DataFrame(rows)

    # Composite Rank (ascending: 1 = best) using 1M & 1Y returns
    out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
    out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
    out["Composite Rank"] = (out["Rank Ret 1M"] + out["Rank Ret 1Y"]) / 2.0
    out["Composite Rank"] = out["Composite Rank"].rank(ascending=True, method="min").astype(int)

    final = out[[
        "Company","Ticker","Current","Pred 1D","Pred 1M","Pred 1Y",
        "Ret 1D %","Ret 1M %","Ret 1Y %","Composite Rank",
        "Rank Ret 1M","Rank Ret 1Y"
    ]].sort_values("Composite Rank").reset_index(drop=True)

    st.subheader("📊 Ranked Screener")
    def color_ret(v):
        if v > 0: return "color: green"
        if v < 0: return "color: red"
        return ""
    st.dataframe(final.style.applymap(color_ret, subset=["Ret 1D %","Ret 1M %","Ret 1Y %"]),
                 use_container_width=True)

    csv = final.to_csv(index=False).encode()
    st.download_button("Download CSV (Main)", csv, f"{index_choice.lower().replace(' ','_')}_screener.csv", "text/csv")

    # -------- Top-3 under ₹100/₹200/₹300/₹400/₹500 --------
    st.markdown("### 🏆 Top 3 under ₹100 / ₹200 / ₹300 / ₹400 / ₹500 (by horizon)")
    thresholds = [100, 200, 300, 400, 500]

    for thr in thresholds:
        cheap = out[out["Current"] < thr].copy()
        with st.expander(f"Under ₹{thr}"):
            if cheap.empty:
                st.write("None")
            else:
                c1, c2, c3 = st.columns(3)
                top1d = cheap.sort_values("Ret 1D %", ascending=False).head(3)[
                    ["Company","Ticker","Current","Pred 1D","Ret 1D %"]
                ].reset_index(drop=True)
                top1m = cheap.sort_values("Ret 1M %", ascending=False).head(3)[
                    ["Company","Ticker","Current","Pred 1M","Ret 1M %"]
                ].reset_index(drop=True)
                top1y = cheap.sort_values("Ret 1Y %", ascending=False).head(3)[
                    ["Company","Ticker","Current","Pred 1Y","Ret 1Y %"]
                ].reset_index(drop=True)

                with c1:
                    st.caption("Next 1 Day")
                    st.dataframe(top1d, use_container_width=True)
                with c2:
                    st.caption("Next 1 Month")
                    st.dataframe(top1m, use_container_width=True)
                with c3:
                    st.caption("Next 1 Year")
                    st.dataframe(top1y, use_container_width=True)
