# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from math import floor

st.set_page_config(page_title="NSE Midcap / Smallcap Screener — Fast + Hidden 1M Improvement",
                   layout="wide")
st.title("⚡ NSE Midcap / Smallcap 250 Screener — Fast + Hidden 1M Accuracy Boost")

# ---------------- Data sources ----------------
INDEX_URLS = {
    "Nifty Midcap 100":   "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
    "Nifty Smallcap 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

# Fallback lists (compact for speed/reliability)
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

# ---------------- Helpers ----------------
def _csv_to_pairs(csv_text: str):
    df = pd.read_csv(StringIO(csv_text))
    # Guard: some CSVs may vary, try common columns
    if "Company Name" in df.columns and "Symbol" in df.columns:
        names = df["Company Name"].astype(str).str.strip().tolist()
        tks = df["Symbol"].astype(str).str.strip().apply(lambda s: f"{s}.NS").tolist()
        return list(zip(names, tks))
    # fallback simple parse
    return []

@st.cache_data(ttl=86400)
def fetch_constituents(index_name: str):
    url = INDEX_URLS[index_name]
    try:
        r = requests.get(url, timeout=5)
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
    """Map VIX to risk adjustment (%) per horizon.
    1D: lighter; 1M: medium; 1Y: heavier (per prior design).
    """
    if vix is None:
        return 0.0
    if horizon == "1D":
        return float(np.clip((15 - vix) * 0.4, -5, 5))    # light
    elif horizon == "1M":
        return float(np.clip((15 - vix) * 0.8, -10, 10))  # medium
    else:
        return float(np.clip((15 - vix) * 1.2, -20, 20))  # heavy

@st.cache_data(show_spinner=False)
def batch_history(tickers, years=3):
    """Batch download -- auto_adjust=True so Close is adjusted."""
    return yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False,
                       threads=True, group_by="ticker")

# ---------------- Core signal function (with HIDDEN 1M improvement) ----------------
def compute_signals(price_series: pd.Series):
    """
    Returns:
      dict(current, pred1d/pred1m/pred1y not returned directly here),
      but we will compute ret1d, ret1m, ret1y (ret% estimates).
    Hidden technical checks (MA / RSI) are used to nudge ret1m.
    """
    s = price_series.dropna()
    if s.size < 60:
        return None

    # Basic momentum windows
    def mom(arr, days):
        if arr.size <= days:
            return np.nan
        return (arr[-1] / arr[-days] - 1.0) * 100.0

    arr = s.values
    current = float(arr[-1])
    m1   = mom(arr, 1)
    m5   = mom(arr, 5)
    m21  = mom(arr, 21)
    m63  = mom(arr, 63)
    m252 = mom(arr, 252)

    # volatility
    rets = s.pct_change().dropna()
    vol21 = rets.rolling(21).std().iloc[-1] if rets.size >= 21 else rets.std()
    if np.isnan(vol21):
        vol21 = rets.std() if not rets.empty else 0.0

    # Base heuristic returns (same structure as before)
    base_1d = 0.7 * (m1 if not np.isnan(m1) else 0.0) + 0.3 * ((m5 if not np.isnan(m5) else 0.0) / 5.0)
    ret1d = np.clip(base_1d - 30 * vol21, -8, 8)

    base_1m = 0.6 * (m21 if not np.isnan(m21) else 0.0) + 0.4 * (m63 if not np.isnan(m63) else 0.0)
    # hidden features computed locally (not exposed)
    # Moving averages (short slices to be fast)
    series = s.astype(float)
    ma20 = series.rolling(20).mean().iloc[-1] if series.size >= 20 else np.nan
    ma50 = series.rolling(50).mean().iloc[-1] if series.size >= 50 else np.nan
    ma200 = series.rolling(200).mean().iloc[-1] if series.size >= 200 else np.nan

    # RSI(14) - used only to dampen overbought signals
    delta = series.diff().dropna()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    rsi14 = 100 - (100 / (1 + rs.iloc[-1])) if not rs.empty else 50.0

    # Hidden rule adjustments to base_1m to improve 1M accuracy:
    # - If short MA below medium MA → weaken momentum (reduce base_1m)
    # - If RSI > 70 (overbought) → mild dampening
    # - If RSI < 30 (oversold) → mild boost
    adj_factor = 1.0
    try:
        if not np.isnan(ma20) and not np.isnan(ma50):
            if ma20 < ma50:  # weak short-term trend
                adj_factor *= 0.75
            else:
                adj_factor *= 1.05
        # RSI adjustments
        if rsi14 > 70:
            adj_factor *= 0.85
        elif rsi14 < 30:
            adj_factor *= 1.12
    except Exception:
        # on any unexpected issue, keep adj_factor 1.0
        adj_factor = 1.0

    adjusted_1m = base_1m * adj_factor
    ret1m = np.clip(adjusted_1m - 100 * vol21, -50, 50)

    # 1Y return (as before), using m252 if available
    m252_eff = m252 if not np.isnan(m252) else (m63 * 4 if not np.isnan(m63) else 0.0)
    ret1y = np.clip(0.3 * (m63 if not np.isnan(m63) else 0.0) + 0.7 * m252_eff - 150 * vol21, -80, 120)

    # Simple probability proxies (kept light)
    prob1d = float(np.clip(0.5 + (m1 / 25.0 if not np.isnan(m1) else 0.0) - (vol21 * 1.0), 0.05, 0.95))
    prob1m = float(np.clip(0.5 + (adjusted_1m / 100.0) - (vol21 * 1.8), 0.05, 0.95))
    prob1y = float(np.clip(0.5 + ((m252 if not np.isnan(m252) else m63) / 300.0) - (vol21 * 1.5), 0.05, 0.95))

    return {
        "current": current,
        "ret1d": float(ret1d),
        "ret1m": float(ret1m),
        "ret1y": float(ret1y),
        "prob1d": prob1d,
        "prob1m": prob1m,
        "prob1y": prob1y,
        "vol21": float(vol21),
        # hidden technicals used only internally (not surfaced)
        "_hidden": {"rsi14": float(rsi14) if not np.isnan(rsi14) else None,
                    "ma20": float(ma20) if not np.isnan(ma20) else None,
                    "ma50": float(ma50) if not np.isnan(ma50) else None}
    }

# ---------------- UI ----------------
index_choice = st.selectbox("Choose Index", list(INDEX_URLS.keys()))
companies = fetch_constituents(index_choice)
if not companies:
    st.error("No constituents found for selected index.")
    st.stop()

# Layout controls
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    # allow full list length
    limit = st.slider("Tickers to process", 5, len(companies), min(50, len(companies)), step=5)
with c2:
    # keep manual override slider but default to auto VIX mapping
    vix = fetch_vix()
    adj1d_auto = vix_to_adj(vix, "1D")
    adj1m_auto = vix_to_adj(vix, "1M")
    adj1y_auto = vix_to_adj(vix, "1Y")
    vix_str = f"{vix:.2f}" if vix is not None else "N/A"
    st.caption(f"📉 India VIX = {vix_str} → Auto Risk Adj: 1D={adj1d_auto:+.1f}%, 1M={adj1m_auto:+.1f}%, 1Y={adj1y_auto:+.1f}%")
    # allow user to toggle auto/manual
    risk_mode = st.selectbox("Risk adjustment mode", ["Auto (VIX)", "Manual"])
    if risk_mode == "Manual":
        manual_adj1d = st.slider("Manual 1D adj (%)", -20, 20, int(adj1d_auto))
        manual_adj1m = st.slider("Manual 1M adj (%)", -20, 20, int(adj1m_auto))
        manual_adj1y = st.slider("Manual 1Y adj (%)", -20, 20, int(adj1y_auto))
        adj1d, adj1m, adj1y = float(manual_adj1d), float(manual_adj1m), float(manual_adj1y)
    else:
        adj1d, adj1m, adj1y = float(adj1d_auto), float(adj1m_auto), float(adj1y_auto)

with c3:
    run_btn = st.button("Run (Fast)")

# Thresholds for top picks
thresholds = [100, 200, 300, 400, 500]

# Run
if run_btn:
    subset = companies[:limit]
    name_map = {tkr: name for name, tkr in subset}
    tickers = [tkr for _, tkr in subset]

    # batch download
    data = batch_history(tickers, years=3)
    rows = []
    skipped = []

    if isinstance(data.columns, pd.MultiIndex):
        for tkr in tickers:
            try:
                df_t = data[tkr]
            except Exception:
                skipped.append((tkr, "no-data"))
                continue

            cols = list(df_t.columns)
            price_col = "Close" if "Close" in cols else ("Adj Close" if "Adj Close" in cols else None)
            if price_col is None:
                skipped.append((tkr, "no-price-col"))
                continue

            series = df_t[price_col].dropna()
            sig = compute_signals(series)
            if sig is None:
                skipped.append((tkr, "insufficient-history"))
                continue

            # apply per-horizon risk adjustments (user/VIX controlled)
            r1d = sig["ret1d"] * (1 + adj1d / 100.0)
            r1m = sig["ret1m"] * (1 + adj1m / 100.0)
            r1y = sig["ret1y"] * (1 + adj1y / 100.0)

            pred1d = sig["current"] * (1 + r1d / 100.0)
            pred1m = sig["current"] * (1 + r1m / 100.0)
            pred1y = sig["current"] * (1 + r1y / 100.0)

            rows.append({
                "Company": name_map.get(tkr, tkr.replace(".NS", "")),
                "Ticker": tkr,
                "Current": round(sig["current"], 2),
                "Pred 1D": round(pred1d, 2),
                "Pred 1M": round(pred1m, 2),
                "Pred 1Y": round(pred1y, 2),
                "Ret 1D %": round(r1d, 2),
                "Ret 1M %": round(r1m, 2),
                "Ret 1Y %": round(r1y, 2),
                "Prob Up 1D": round(sig["prob1d"], 3),
                "Prob Up 1M": round(sig["prob1m"], 3),
                "Prob Up 1Y": round(sig["prob1y"], 3),
                "Vol21": round(sig.get("vol21", 0.0), 4),
            })
    else:
        # single symbol case (unlikely here)
        cols = list(data.columns)
        price_col = "Close" if "Close" in cols else ("Adj Close" if "Adj Close" in cols else None)
        if price_col and tickers:
            tkr = tickers[0]
            series = data[price_col].dropna()
            sig = compute_signals(series)
            if sig:
                r1d = sig["ret1d"] * (1 + adj1d / 100.0)
                r1m = sig["ret1m"] * (1 + adj1m / 100.0)
                r1y = sig["ret1y"] * (1 + adj1y / 100.0)
                pred1d = sig["current"] * (1 + r1d / 100.0)
                pred1m = sig["current"] * (1 + r1m / 100.0)
                pred1y = sig["current"] * (1 + r1y / 100.0)
                rows.append({
                    "Company": name_map.get(tkr, tkr.replace(".NS", "")),
                    "Ticker": tkr,
                    "Current": round(sig["current"], 2),
                    "Pred 1D": round(pred1d, 2),
                    "Pred 1M": round(pred1m, 2),
                    "Pred 1Y": round(pred1y, 2),
                    "Ret 1D %": round(r1d, 2),
                    "Ret 1M %": round(r1m, 2),
                    "Ret 1Y %": round(r1y, 2),
                    "Prob Up 1D": round(sig["prob1d"], 3),
                    "Prob Up 1M": round(sig["prob1m"], 3),
                    "Prob Up 1Y": round(sig["prob1y"], 3),
                    "Vol21": round(sig.get("vol21", 0.0), 4),
                })

    if not rows:
        st.error("No results. Try more tickers or check network.")
    else:
        out = pd.DataFrame(rows)

        # compute ranks and composite rank (ascending 1 = best)
        out["Rank Ret 1M"] = out["Ret 1M %"].rank(ascending=False, method="min").astype(int)
        out["Rank Ret 1Y"] = out["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1M"] = out["Prob Up 1M"].rank(ascending=False, method="min").astype(int)
        out["Rank Prob 1Y"] = out["Prob Up 1Y"].rank(ascending=False, method="min").astype(int)
        out["Composite Rank"] = (out["Rank Ret 1M"] + out["Rank Ret 1Y"] + out["Rank Prob 1M"] + out["Rank Prob 1Y"]) / 4.0
        out["Composite Rank"] = out["Composite Rank"].rank(ascending=True, method="min").astype(int)

        final = out[[
            "Company", "Ticker", "Current", "Pred 1D", "Pred 1M", "Pred 1Y",
            "Ret 1D %", "Ret 1M %", "Ret 1Y %", "Prob Up 1M", "Prob Up 1Y",
            "Vol21", "Composite Rank", "Rank Ret 1M", "Rank Ret 1Y"
        ]].sort_values("Composite Rank").reset_index(drop=True)

        # display main table (simple color for returns)
        def color_ret(v):
            if v > 0: return "color: green"
            if v < 0: return "color: red"
            return ""

        st.subheader("📊 Ranked Screener (Composite Rank ascending)")
        st.dataframe(final.style.applymap(color_ret, subset=["Ret 1M %", "Ret 1Y %"]), use_container_width=True)

        csv = final.to_csv(index=False).encode()
        st.download_button("Download CSV (Main)", csv, f"{index_choice.lower().replace(' ','_')}_screener.csv", "text/csv")

        # show skipped tickers if any
        if skipped:
            st.info(f"Skipped {len(skipped)} tickers (insufficient data or missing). Toggle below to view.")
            if st.checkbox("Show skipped tickers"):
                st.write(pd.DataFrame(skipped, columns=["Ticker", "Reason"]))

        # -------- Top-3 under thresholds --------
        st.markdown("### 🏆 Top 3 under ₹100 / ₹200 / ₹300 / ₹400 / ₹500 (by horizon)")
        for thr in thresholds:
            cheap = out[out["Current"] < thr].copy()
            with st.expander(f"Under ₹{thr}"):
                if cheap.empty:
                    st.write("None")
                else:
                    c1, c2, c3 = st.columns(3)
                    top1d = cheap.sort_values("Ret 1D %", ascending=False).head(3)[["Company","Ticker","Current","Pred 1D","Ret 1D %"]].reset_index(drop=True)
                    top1m = cheap.sort_values("Ret 1M %", ascending=False).head(3)[["Company","Ticker","Current","Pred 1M","Ret 1M %"]].reset_index(drop=True)
                    top1y = cheap.sort_values("Ret 1Y %", ascending=False).head(3)[["Company","Ticker","Current","Pred 1Y","Ret 1Y %"]].reset_index(drop=True)
                    with c1:
                        st.caption("Next 1 Day")
                        st.dataframe(top1d, use_container_width=True)
                    with c2:
                        st.caption("Next 1 Month")
                        st.dataframe(top1m, use_container_width=True)
                    with c3:
                        st.caption("Next 1 Year")
                        st.dataframe(top1y, use_container_width=True)

        # -------- Simple Portfolio Builder (equal-weight) --------
        st.markdown("### 📦 Build Portfolio (from Composite Rank)")
        with st.expander("Portfolio Builder (click to expand)"):
            capital = st.number_input("Capital to deploy (INR)", min_value=10000, value=1000000, step=50000)
            top_n = st.number_input("Number of holdings (Top N)", min_value=1, max_value=min(50, len(final)), value=min(20, len(final)))
            stop_loss_pct = st.number_input("Stop-loss (%)", min_value=1, max_value=50, value=10)
            take_profit_pct = st.number_input("Take-profit (%)", min_value=1, max_value=200, value=25)
            build_btn = st.button("Build Portfolio")

            if build_btn:
                top_df = final.head(int(top_n)).copy()
                if top_df.empty:
                    st.warning("No top stocks found.")
                else:
                    n = len(top_df)
                    weight = 1.0 / n
                    alloc = capital * weight
                    shares = []
                    allocs = []
                    sl_prices = []
                    tp_prices = []
                    for i, row in top_df.iterrows():
                        price = float(row["Current"])
                        num_shares = int(floor(alloc / price)) if price > 0 else 0
                        actual_alloc = num_shares * price
                        sl_price = round(price * (1 - stop_loss_pct / 100.0), 2)
                        tp_price = round(price * (1 + take_profit_pct / 100.0), 2)
                        shares.append(num_shares)
                        allocs.append(round(actual_alloc, 2))
                        sl_prices.append(sl_price)
                        tp_prices.append(tp_price)

                    top_df["Weight %"] = round(weight * 100.0, 2)
                    top_df["Alloc ₹"] = allocs
                    top_df["Shares"] = shares
                    top_df["Stop-loss"] = sl_prices
                    top_df["Take-profit"] = tp_prices

                    invested = sum(allocs)
                    cash_left = round(capital - invested, 2)
                    st.write(f"Invested: ₹{invested:,}    Cash left (after rounding): ₹{cash_left:,}")
                    st.dataframe(top_df[["Company","Ticker","Current","Ret 1M %","Weight %","Alloc ₹","Shares","Stop-loss","Take-profit"]], use_container_width=True)

                    # CSV download
                    csv_port = top_df.to_csv(index=False).encode()
                    st.download_button("Download Portfolio CSV", csv_port, f"portfolio_{index_choice.lower().replace(' ','_')}.csv", "text/csv")

st.caption("This app applies a hidden, conservative improvement to 1M predictions (MA/RSI-based nudges) to boost short-term accuracy without showing technicals.")
