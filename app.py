import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

st.set_page_config(page_title="NSE Midcap/Smallcap Screener (Fast)", layout="wide")
st.title("⚡ NSE Midcap / Smallcap Screener — Minimal & Fast (with Auto Risk Adjustment)")

# ---------------- Data sources ----------------
INDEX_URLS = {
    "Nifty Midcap 100":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 100": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

MIDCAP_FALLBACK = ["ABBOTINDIA","ALKEM","ASHOKLEY","AUBANK","AUROPHARMA","BALKRISIND","BEL","BERGEPAINT","BHEL",
    "CANFINHOME","CUMMINSIND","DALBHARAT","DEEPAKNTR","FEDERALBNK","GODREJPROP","HAVELLS","HINDZINC","IDFCFIRSTB",
    "INDHOTEL","INDIAMART","IPCALAB","JUBLFOOD","LUPIN","MANKIND","MUTHOOTFIN","NMDC","OBEROIRLTY","PAGEIND",
    "PERSISTENT","PFIZER","POLYCAB","RECLTD","SAIL","SRF","SUNTV","TATAELXSI","TATAPOWER","TRENT","TVSMOTOR","UBL",
    "VOLTAS","ZYDUSLIFE","PIIND","CONCOR","APOLLOTYRE","TORNTPHARM","MPHASIS","ASTRAL","OFSS","MINDTREE","CROMPTON",
    "ATGL","PETRONET","LTTS","ESCORTS","INDIGO","COLPAL","GILLETTE","BANKBARODA","EXIDEIND","IDBI","INDUSINDBK",
    "LICHSGFIN","MRF","NAVINFLUOR","PFC","PNB","RAMCOCEM","RBLBANK","SHREECEM","TATACHEM","TATACOMM","TORNTPOWER",
    "UNIONBANK","WHIRLPOOL","ZEEL","ABB","ADANIPOWER","AMBUJACEM","BANDHANBNK","CANBK","CHOLAFIN","DLF","EICHERMOT",
    "HINDPETRO","IOC","JINDALSTEL","JSWENERGY","LICI","NTPC","ONGC","POWERGRID","SBICARD","SBILIFE","SBIN",
    "TATAMOTORS","TATASTEEL","TECHM","UPL","VEDL","WIPRO"]

SMALLCAP100_FALLBACK = ["ACE","ANGELONE","ANURAS","APLLTD","BBTC","BIRLACORPN","BLS","CAMS","CREDITACC","DATAPATTNS",
    "DCXINDIA","DEEPAKFERT","EIDPARRY","FINEORG","FORTIS","GNFC","HATSUN","HONAUT","INDIACEM","INGERRAND","JAMNAAUTO",
    "JKCEMENT","KAJARIACER","KEI","KPITTECH","LALPATHLAB","MAHSEAMLES","MCX","METROPOLIS","NATIONALUM","PNBHOUSING",
    "RITES","ROSSARI","SAREGAMA","SIS","SKFINDIA","SUPREMEIND","TATAINVEST","TEAMLEASE","THYROCARE","TRIVENI",
    "UJJIVANSFB","VGUARD","VOLTAMP","VTL","WELSPUNIND"]

SMALLCAP250_FALLBACK = ["AARTIIND","AFFLE","AMBER","ANANDRATHI","APLAPOLLO","ARVINDFASN","ASTERDM","ASTRAZEN","BASF",
    "BAJAJHLDNG","BALAMINES","BEML","BECTORFOOD","BLUESTARCO","BSE","CESC","CHEMPLASTS","COFORGE","CYIENT","DATAPATTNS",
    "DCMSHRIRAM","DEEPAKFERT","DEVYANI","EIHOTEL","ENIL","EPL","FDC","GESHIP","GLAND","GLS","GRINDWELL","HAPPSTMNDS",
    "HATHWAY","HGS","IRCON","ISEC","JBCHEPHARM","JCHAC","JKLAKSHMI","JYOTHYLAB","KAJARIACER","KEC","KIRLOSBROS",
    "KIRLOSIND","LAOPALA","LATENTVIEW","LEMONTREE","LUXIND","MAHLIFE","MMTC","NAZARA","NESCO","NOCIL","PAYTM","RADICO",
    "RAILTEL","RAIN","RATEGAIN","REDINGTON","RTNINDIA","SANOFI","SAPPHIRE","SONATSOFTW","STARHEALTH","SUNCLAYLTD",
    "SUNDRMFAST","SUNDARMFIN","SUVENPHAR","SYNGENE","TANLA","TCIEXP","TIDEWATER","TTML","UJJIVAN","VGUARD","VSTIND",
    "WELSPUNIND","ZYDUSWELL"]

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
        elif index_name == "Nifty Smallcap 100":
            return [(s, f"{s}.NS") for s in SMALLCAP100_FALLBACK]
        else:
            return [(s, f"{s}.NS") for s in SMALLCAP250_FALLBACK]

@st.cache_data(ttl=3600)
def fetch_vix_adjustment():
    """Fetch India VIX from Yahoo and convert to a -20% to +20% risk adjustment."""
    try:
        vix = yf.Ticker("^INDIAVIX")
        hist = vix.history(period="5d")
        latest = hist["Close"].iloc[-1]
        # Base around 15 → calm; scale around ±20 range
        adj = (15 - latest) * 1.2
        adj = max(-20, min(20, adj))
        return round(adj, 2), round(latest, 2)
    except Exception:
        return 0.0, None  # fallback neutral

@st.cache_data(show_spinner=False)
def batch_history(tickers, years=3):
    return yf.download(tickers, period=f"{years}y", auto_adjust=False, progress=False,
                       threads=True, group_by="ticker")

def compute_signals(price_series: pd.Series):
    s = price_series.dropna()
    if s.size < 80:
        return None
    y = s.values
    current = float(y[-1])
    def mom(days):
        if s.size <= days: return np.nan
        return (y[-1]/y[-days]-1.0)*100.0
    m21, m63, m252 = mom(21), mom(63), mom(252)
    vol21 = s.pct_change().rolling(21).std().iloc[-1]
    if np.isnan(vol21): vol21 = s.pct_change().std()
    ret1m = np.clip(0.6*(m21 if not np.isnan(m21) else 0.0) + 0.4*(m63 if not np.isnan(m63) else 0.0) - 100*vol21, -50, 50)
    m252_eff = m252 if not np.isnan(m252) else (m63*4 if not np.isnan(m63) else 0.0)
    ret1y = np.clip(0.3*(m63 if not np.isnan(m63) else 0.0) + 0.7*m252_eff - 150*vol21, -80, 120)
    prob1m = float(np.clip(0.5 + (m21/100.0 if not np.isnan(m21) else 0.0) - (vol21*2), 0.05, 0.95))
    prob1y = float(np.clip(0.5 + ((m252/300.0) if not np.isnan(m252) else (m63/150.0 if not np.isnan(m63) else 0.0)) - (vol21*1.5), 0.05, 0.95))
    pred1m = current * (1 + ret1m/100.0)
    pred1y = current * (1 + ret1y/100.0)
    return dict(current=current, pred1m=pred1m, pred1y=pred1y, ret1m=ret1m, ret1y=ret1y, prob1m=prob1m, prob1y=prob1y)

# ---------------- UI ----------------
index_choice = st.selectbox("Choose Index", list(INDEX_URLS.keys()))
companies = fetch_constituents(index_choice)
if not companies:
    st.stop()

# Auto VIX-based adjustment
auto_adj, vix_value = fetch_vix_adjustment()
c1, c2 = st.columns([2,2])
with c1:
    limit = st.slider("Tickers to process", 5, min(100, len(companies)), min(20, len(companies)), step=5)
with c2:
    manual_adj = st.slider("Manual Override (%)", -20, 20, int(auto_adj))
st.caption(f"📉 India VIX = {vix_value if vix_value else 'N/A'} → Auto Risk Adj = {auto_adj}%")

risk_adj = manual_adj  # use manual if adjusted, default = auto_adj

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
            if not price_col: continue
            sig = compute_signals(data[tkr][price_col])
            if not sig: continue
            r1m = sig["ret1m"] * (1 + risk_adj/100.0)
            r1y = sig["ret1y"] * (1 + risk_adj/100.0)
            rows.append({
                "Company": name_map.get(tkr, tkr.replace(".NS","")),
                "Ticker": tkr,
                "Current": round(sig["current"], 2),
                "Pred 1M": round(sig["current"] * (1 + r1m/100.0), 2),
                "Pred 1Y": round(sig["current"] * (1 + r1y/100.0), 2),
                "Ret 1M %": round(r1m, 2),
                "Ret 1Y %": round(r1y, 2),
                "Prob Up 1M": round(sig["prob1m"], 3),
                "Prob Up 1Y": round(sig["prob1y"], 3),
            })
    if not rows:
        st.error("No results. Try more tickers.")
    else:
        out = pd.DataFrame(rows)
        out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
        out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1M"] = out["Prob Up 1M"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1Y"] = out["Prob Up 1Y"].rank(ascending=False, method="min").astype(int)
        out["Composite Rank"] = (out["Rank Ret 1M"]+out["Rank Ret 1Y"]+out["Rank Prob 1M"]+out["Rank Prob 1Y"])/4.0
        out["Composite Rank"] = out["Composite Rank"].rank(ascending=True, method="min").astype(int)
        final = out[[
            "Company","Ticker","Current","Pred 1M","Pred 1Y","Ret 1M %","Ret 1Y %",
            "Prob Up 1M","Prob Up 1Y","Composite Rank","Rank Ret 1M","Rank Ret 1Y","Rank Prob 1M","Rank Prob 1Y"
        ]].sort_values("Composite Rank").reset_index(drop=True)
        def color_ret(v): return "color: green" if v > 0 else ("color: red" if v < 0 else "")
        styled = final.style.applymap(color_ret, subset=["Ret 1M %","Ret 1Y %"])
        st.subheader("📊 Ranked Screener")
        st.dataframe(styled, use_container_width=True)
        csv = final.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"{index_choice.lower().replace(' ','_')}_screener.csv", "text/csv")
