import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

st.set_page_config(page_title="NSE Midcap / Smallcap Screener (Fast)", layout="wide")
st.title("⚡ NSE Midcap / Smallcap Screener — Minimal & Fast")

# ---------------- Helpers ----------------
INDEX_URLS = {
    "Nifty Midcap 100": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 100": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap100list.csv",
}

MIDCAP_FALLBACK = [
    "ABBOTINDIA","ALKEM","ASHOKLEY","AUBANK","AUROPHARMA","BALKRISIND","BEL","BERGEPAINT","BHEL","CANFINHOME",
    "CUMMINSIND","DALBHARAT","DEEPAKNTR","FEDERALBNK","GODREJPROP","HAVELLS","HINDZINC","IDFCFIRSTB","INDHOTEL",
    "INDIAMART","IPCALAB","JUBLFOOD","LUPIN","MANKIND","MUTHOOTFIN","NMDC","OBEROIRLTY","PAGEIND","PERSISTENT",
    "PFIZER","POLYCAB","RECLTD","SAIL","SRF","SUNTV","TATAELXSI","TATAPOWER","TRENT","TVSMOTOR","UBL","VOLTAS",
    "ZYDUSLIFE","PIIND","CONCOR","APOLLOTYRE","TORNTPHARM","MPHASIS","ASTRAL","OFSS","MINDTREE","CROMPTON",
    "ATGL","PETRONET","LTTS","ESCORTS","INDIGO","COLPAL","GILLETTE","BANKBARODA","EXIDEIND","IDBI","INDUSINDBK",
    "LICHSGFIN","MRF","NAVINFLUOR","PFC","PNB","RAMCOCEM","RBLBANK","SHREECEM","TATACHEM","TATACOMM","TORNTPOWER",
    "UNIONBANK","WHIRLPOOL","ZEEL","ABB","ADANIPOWER","AMBUJACEM","BANDHANBNK","CANBK","CHOLAFIN","DLF","EICHERMOT",
    "HINDPETRO","IOC","JINDALSTEL","JSWENERGY","LICI","NTPC","ONGC","POWERGRID","SBICARD","SBILIFE","SBIN","TATAMOTORS",
    "TATASTEEL","TECHM","UPL","VEDL","WIPRO"
]

# Compact smallcap fallback (not full 100; keeps app small & fast)
SMALLCAP_FALLBACK = [
    "ACE","ANGELONE","ANURAS","APLLTD","BBTC","BIRLACORPN","BLS","CAMS","CREDITACC","DATAPATTNS","DCXINDIA",
    "DEEPAKFERT","EIDPARRY","FINEORG","FORTIS","GNFC","HATSUN","HONAUT","INDIACEM","INGERRAND","JAMNAAUTO",
    "JKCEMENT","KAJARIACER","KEI","KPITTECH","LALPATHLAB","MAHSEAMLES","MCX","METROPOLIS","NATIONALUM","PNBHOUSING",
    "RITES","ROSSARI","SAREGAMA","SIS","SKFINDIA","SUPREMEIND","TATAINVEST","TEAMLEASE","THYROCARE","TRIVENI",
    "UJJIVANSFB","VGUARD","VOLTAMP","VTL","WELSPUNIND"
]

def _csv_to_pairs(csv_text: str):
    df = pd.read_csv(StringIO(csv_text))
    names = df["Company Name"].astype(str).str.strip().tolist()
    tks = df["Symbol"].astype(str).str.strip().apply(lambda s: f"{s}.NS").tolist()
    return list(zip(names, tks))

@st.cache_data(ttl=86400)
def fetch_constituents(index_name: str):
    """Fetch index constituents with 5s timeout; index-specific fallback."""
    url = INDEX_URLS[index_name]
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return _csv_to_pairs(r.text)
    except Exception:
        if index_name == "Nifty Midcap 100":
            return [(s, f"{s}.NS") for s in MIDCAP_FALLBACK]
        else:
            return [(s, f"{s}.NS") for s in SMALLCAP_FALLBACK]

@st.cache_data(show_spinner=False)
def batch_history(tickers, years=3):
    """One fast call for many tickers."""
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

    m21  = mom(21)     # ~1M momentum
    m63  = mom(63)     # ~3M
    m252 = mom(252)    # ~1Y (if available)

    vol21 = s.pct_change().rolling(21).std().iloc[-1]
    if np.isnan(vol21): vol21 = s.pct_change().std()

    # Heuristic returns (fast) with volatility penalty
    ret1m = np.clip(0.6*m21 + 0.4*m63 - 100*vol21, -50, 50)
    m252_eff = m252 if not np.isnan(m252) else (m63*4 if not np.isnan(m63) else 0.0)
    ret1y = np.clip(0.3*m63 + 0.7*m252_eff - 150*vol21, -80, 120)

    prob1m = float(np.clip(0.5 + (m21/100.0) - (vol21*2), 0.05, 0.95))
    prob1y = float(np.clip(0.5 + ((m252 if not np.isnan(m252) else m63)/300.0) - (vol21*1.5), 0.05, 0.95))

    pred1m = current * (1 + ret1m/100.0)
    pred1y = current * (1 + ret1y/100.0)

    return dict(current=current, pred1m=pred1m, pred1y=pred1y, ret1m=ret1m, ret1y=ret1y,
                prob1m=prob1m, prob1y=prob1y)

# ---------------- UI ----------------
index_choice = st.selectbox("Choose Index", ["Nifty Midcap 100", "Nifty Smallcap 100"])
companies = fetch_constituents(index_choice)
if not companies:
    st.stop()

c1, c2, c3 = st.columns([2,2,2])
with c1:
    limit = st.slider("Tickers to process", 5, min(100, len(companies)), min(20, len(companies)), step=5)
with c2:
    risk_adj = st.slider("🌍 Risk Adjustment (%)", -20, 20, 0)
with c3:
    st.write(f"Loaded: **{index_choice}**")

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
            # apply risk adjustment to returns only
            r1m = sig["ret1m"] * (1 + risk_adj/100.0)
            r1y = sig["ret1y"] * (1 + risk_adj/100.0)
            pred1m = sig["current"] * (1 + r1m/100.0)
            pred1y = sig["current"] * (1 + r1y/100.0)

            rows.append({
                "Company": name_map.get(tkr, tkr.replace(".NS","")),
                "Ticker": tkr,
                "Current": round(sig["current"], 2),
                "Pred 1M": round(pred1m, 2),
                "Pred 1Y": round(pred1y, 2),
                "Ret 1M %": round(r1m, 2),
                "Ret 1Y %": round(r1y, 2),
                "Prob Up 1M": round(sig["prob1m"], 3),
                "Prob Up 1Y": round(sig["prob1y"], 3),
            })
    else:
        # single symbol case
        cols = list(data.columns)
        price_col = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else None)
        if price_col and tickers:
            tkr = tickers[0]
            sig = compute_signals(data[price_col])
            if sig:
                r1m = sig["ret1m"] * (1 + risk_adj/100.0)
                r1y = sig["ret1y"] * (1 + risk_adj/100.0)
                pred1m = sig["current"] * (1 + r1m/100.0)
                pred1y = sig["current"] * (1 + r1y/100.0)
                rows.append({
                    "Company": name_map.get(tkr, tkr.replace(".NS","")),
                    "Ticker": tkr,
                    "Current": round(sig["current"], 2),
                    "Pred 1M": round(pred1m, 2),
                    "Pred 1Y": round(pred1y, 2),
                    "Ret 1M %": round(r1m, 2),
                    "Ret 1Y %": round(r1y, 2),
                    "Prob Up 1M": round(sig["prob1m"], 3),
                    "Prob Up 1Y": round(sig["prob1y"], 3),
                })

    if not rows:
        st.error("No results. Try more tickers.")
    else:
        out = pd.DataFrame(rows)

        # Ranks (ascending: 1 = best)
        out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
        out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1M"] = out["Prob Up 1M"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1Y"] = out["Prob Up 1Y"].rank(ascending=False, method="min").astype(int)
        out["Composite Rank"] = (
            out["Rank Ret 1M"] + out["Rank Ret 1Y"] + out["Rank Prob 1M"] + out["Rank Prob 1Y"]
        ) / 4.0
        out["Composite Rank"] = out["Composite Rank"].rank(ascending=True, method="min").astype(int)

        # Final order (as you wanted)
        out = out[[
            "Company","Ticker","Current","Pred 1M","Pred 1Y",
            "Ret 1M %","Ret 1Y %","Prob Up 1M","Prob Up 1Y","Composite Rank",
            "Rank Ret 1M","Rank Ret 1Y","Rank Prob 1M","Rank Prob 1Y"
        ]].sort_values("Composite Rank").reset_index(drop=True)

        # Simple text coloring only (no matplotlib needed)
        def color_ret(v):
            if v > 0: return "color: green"
            if v < 0: return "color: red"
            return ""
        styled = out.style.applymap(color_ret, subset=["Ret 1M %","Ret 1Y %"])
        st.dataframe(styled, use_container_width=True)

        csv = out.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"{index_choice.lower().replace(' ','_')}_screener_fast.csv", "text/csv")

st.caption("Tip: start with ~20 tickers; increase once you confirm speed. No heavy libs, so it should open quickly.")
