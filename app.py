# app.py
# NSE Screener — Integrated Predictions + Historical Log + Actuals + Summary (no downloads)
# Prediction log persisted at predictions_log.csv and fully displayed in app UI.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="NSE Screener — Integrated Log & Accuracy", layout="wide")
st.title("⚡ NSE Screener — Integrated Predictions & Accuracy (in-app)")

PRED_LOG_PATH = "predictions_log.csv"

# ---------------------------
# Log helpers
# ---------------------------
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

# ---------------------------
# Fetch actuals helper
# ---------------------------
def fetch_actual_close_on_or_after(ticker, target_date, lookahead_days=7):
    start = target_date.strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=lookahead_days+1)).strftime("%Y-%m-%d")
    try:
        hist = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if hist is None or hist.empty:
            return np.nan, None
        hist = hist.sort_index()
        price = float(hist["Close"].iloc[0])
        return price, pd.to_datetime(hist.index[0]).date()
    except Exception:
        return np.nan, None

def update_log_with_actuals(path=PRED_LOG_PATH, now_date=None, force=False):
    """
    Updates CSV log by filling matured actual 1M/1Y prices and error %.
    If force=True, attempts to update all rows regardless of current values.
    """
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

        # 1M actual
        needs_1m = pd.isna(row.get("actual_1m")) or force
        target_1m = run_date + timedelta(days=30)
        if needs_1m and target_1m <= now:
            price, price_date = fetch_actual_close_on_or_after(row["ticker"], target_1m, lookahead_days=7)
            if not pd.isna(price):
                df.at[idx,"actual_1m"] = price
                df.at[idx,"actual_1m_date"] = price_date
                pred = row.get("pred_1m",np.nan)
                df.at[idx,"err_pct_1m"] = (abs(pred-price)/price*100) if (not pd.isna(pred) and price!=0) else np.nan
                updated = True

        # 1Y actual
        needs_1y = pd.isna(row.get("actual_1y")) or force
        target_1y = run_date + timedelta(days=365)
        if needs_1y and target_1y <= now:
            price, price_date = fetch_actual_close_on_or_after(row["ticker"], target_1y, lookahead_days=14)
            if not pd.isna(price):
                df.at[idx,"actual_1y"] = price
                df.at[idx,"actual_1y_date"] = price_date
                pred = row.get("pred_1y",np.nan)
                df.at[idx,"err_pct_1y"] = (abs(pred-price)/price*100) if (not pd.isna(pred) and price!=0) else np.nan
                updated = True

    if updated:
        write_pred_log(df, path)
    return df

# ---------------------------
# Simple features & prediction (heuristic fallback)
# Replace these with your ML outputs in future if desired
# ---------------------------
def compute_hidden_features(s: pd.Series):
    s = s.dropna().astype(float)
    if s.size < 60: return None
    cur = float(s.iloc[-1])
    def mom(days): return (s.iloc[-1]/s.iloc[-days]-1.0)*100 if s.size>days else np.nan
    return {"current": cur, "m21": mom(21), "m63": mom(63), "m252": mom(252) if s.size>=252 else np.nan, "vol21": s.pct_change().rolling(21).std().iloc[-1]}

def heuristic_ret1m(feats):
    return np.clip(0.6*(feats.get("m21",0.0) or 0.0) + 0.4*(feats.get("m63",0.0) or 0.0) - 100*(feats.get("vol21",0.0) or 0.0), -50,50)

def heuristic_ret1y(feats):
    m63 = feats.get("m63",0.0) or 0.0
    m252 = feats.get("m252", np.nan)
    m252_eff = m252 if not pd.isna(m252) else m63*4
    return np.clip(0.3*m63 + 0.7*m252_eff - 150*(feats.get("vol21",0.0) or 0.0), -80,120)

# ---------------------------
# UI: inputs and main flow
# ---------------------------
st.markdown("### Run predictions (will be logged automatically in-app)")

col1, col2 = st.columns([3,1])
with col1:
    tickers_text = st.text_input("Tickers (comma separated, e.g. RELIANCE.NS,TCS.NS)", value="ABC.NS,DEF.NS,XYZ.NS")
with col2:
    run_button = st.button("Run & Log Predictions")

tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

# Force-refresh actuals
st.markdown("---")
force_col1, force_col2 = st.columns([1,3])
with force_col1:
    force_refresh = st.button("Force refresh actuals (recheck all rows)")
with force_col2:
    show_only_matured = st.checkbox("Show only matured (have actuals) rows", value=False)

# On app load always attempt to update actuals (non-forced)
st.info("Updating prediction log with matured actuals (automatic check)...")
log_df = update_log_with_actuals(PRED_LOG_PATH, force=False)
st.success("Automatic actuals update complete.")

# If user clicked force refresh
if force_refresh:
    st.info("Force-refreshing actuals...")
    log_df = update_log_with_actuals(PRED_LOG_PATH, force=True)
    st.success("Force refresh finished.")

# Run prediction & append to log if user clicks
if run_button and tickers:
    st.info(f"Fetching price data for {len(tickers)} tickers...")
    try:
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False, threads=True, group_by="ticker")
    except Exception as e:
        st.error(f"Price fetch failed: {e}")
        data = {}

    rows = []
    log_rows = []
    for t in tickers:
        try:
            ser = data[t]["Close"].dropna()
            ser.index = pd.to_datetime(ser.index)
        except Exception:
            st.warning(f"No price series for {t}, skipping.")
            continue
        feats = compute_hidden_features(ser)
        if feats is None:
            st.warning(f"Insufficient history for {t}, skipping.")
            continue
        r1m = heuristic_ret1m(feats)
        r1y = heuristic_ret1y(feats)
        cur = feats["current"]
        p1m = round(cur * (1 + r1m/100.0), 2)
        p1y = round(cur * (1 + r1y/100.0), 2)
        rows.append({
            "Company": t.replace(".NS",""),
            "Ticker": t,
            "Current": round(cur,2),
            "Pred 1M": p1m,
            "Pred 1Y": p1y,
            "Ret 1M %": round(r1m,2),
            "Ret 1Y %": round(r1y,2)
        })
        log_rows.append({
            "run_date": datetime.utcnow(),
            "company": t.replace(".NS",""),
            "ticker": t,
            "current": round(cur,2),
            "pred_1m": p1m,
            "pred_1y": p1y,
            "ret_1m_pct": round(r1m,2),
            "ret_1y_pct": round(r1y,2)
        })

    if rows:
        df_run = pd.DataFrame(rows)
        df_run["Rank Ret 1M"] = df_run["Ret 1M %"].rank(ascending=False, method="min").astype(int)
        df_run["Rank Ret 1Y"] = df_run["Ret 1Y %"].rank(ascending=False, method="min").astype(int)
        df_run["Composite Rank"] = ((df_run["Rank Ret 1M"] + df_run["Rank Ret 1Y"]) / 2.0).rank(ascending=True, method="min").astype(int)
        st.subheader("Predictions (just run)")
        st.dataframe(df_run.sort_values("Composite Rank").reset_index(drop=True), use_container_width=True)

        # Append to persistent log
        ensure_log_exists()
        existing = read_pred_log()
        new_log_df = pd.concat([existing, pd.DataFrame(log_rows)], ignore_index=True, sort=False)
        # compute ranks for the new entries inside the log for completeness (do by run_date group)
        try:
            # compute ranks per run_date
            new_log_df["run_date"] = pd.to_datetime(new_log_df["run_date"])
            ranked = []
            for run_date, group in new_log_df.groupby("run_date"):
                grp = group.copy()
                grp["rank_ret_1m"] = grp["ret_1m_pct"].rank(ascending=False, method="min").astype("Int64")
                grp["rank_ret_1y"] = grp["ret_1y_pct"].rank(ascending=False, method="min").astype("Int64")
                grp["composite_rank"] = ((grp["rank_ret_1m"].astype(float) + grp["rank_ret_1y"].astype(float))/2.0).rank(ascending=True, method="min").astype("Int64")
                ranked.append(grp)
            new_log_df = pd.concat(ranked, ignore_index=True, sort=False)
        except Exception:
            pass
        write_pred_log(new_log_df)
        st.success(f"Appended {len(log_rows)} rows to integrated log (in-app).")

# ---------------------------
# Show integrated historical table and summary metrics
# ---------------------------
st.markdown("---")
st.header("📜 Integrated Historical Predictions Log (in-app)")

log_df = read_pred_log()
if log_df.empty:
    st.info("No predictions logged yet.")
else:
    # optionally show only matured (both actual_1m or actual_1y present)
    if show_only_matured:
        mask = (~pd.isna(log_df["actual_1m"])) | (~pd.isna(log_df["actual_1y"]))
        display_df = log_df[mask].copy()
    else:
        display_df = log_df.copy()

    # Convert date columns to readable strings
    if "run_date" in display_df.columns:
        display_df["run_date"] = pd.to_datetime(display_df["run_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if "actual_1m_date" in display_df.columns:
        display_df["actual_1m_date"] = pd.to_datetime(display_df["actual_1m_date"], errors="coerce").dt.date
    if "actual_1y_date" in display_df.columns:
        display_df["actual_1y_date"] = pd.to_datetime(display_df["actual_1y_date"], errors="coerce").dt.date

    # Show top 200 recent rows
    st.dataframe(display_df.sort_values("run_date", ascending=False).head(200), use_container_width=True)

    # Summary metrics: MAPE & Directional accuracy for matured rows
    def compute_summary(df):
        out = {}
        # 1M matured rows
        m1 = df[~pd.isna(df["actual_1m"])].copy()
        if not m1.empty:
            m1["abs_err_pct_1m"] = m1["err_pct_1m"]
            out["1M_mape"] = float(m1["abs_err_pct_1m"].mean())
            # directional accuracy: predicted up? (pred > current) vs actual up? (actual > current)
            m1["pred_dir_up"] = (m1["pred_1m"] > m1["current"]).astype(int)
            m1["actual_dir_up"] = (m1["actual_1m"] > m1["current"]).astype(int)
            out["1M_dir_acc"] = float((m1["pred_dir_up"] == m1["actual_dir_up"]).mean()) * 100.0
            out["1M_count"] = int(len(m1))
        else:
            out["1M_mape"] = np.nan
            out["1M_dir_acc"] = np.nan
            out["1M_count"] = 0

        # 1Y matured rows
        m2 = df[~pd.isna(df["actual_1y"])].copy()
        if not m2.empty:
            m2["abs_err_pct_1y"] = m2["err_pct_1y"]
            out["1Y_mape"] = float(m2["abs_err_pct_1y"].mean())
            m2["pred_dir_up"] = (m2["pred_1y"] > m2["current"]).astype(int)
            m2["actual_dir_up"] = (m2["actual_1y"] > m2["current"]).astype(int)
            out["1Y_dir_acc"] = float((m2["pred_dir_up"] == m2["actual_dir_up"]).mean()) * 100.0
            out["1Y_count"] = int(len(m2))
        else:
            out["1Y_mape"] = np.nan
            out["1Y_dir_acc"] = np.nan
            out["1Y_count"] = 0

        return out

    summary = compute_summary(log_df)

    st.markdown("### 📈 Summary (matured predictions only)")
    col1, col2, col3 = st.columns(3)
    col1.metric("1M MAPE (%)", f"{summary['1M_mape']:.2f}" if not pd.isna(summary["1M_mape"]) else "N/A", f"count {summary['1M_count']}")
    col2.metric("1M Dir Acc (%)", f"{summary['1M_dir_acc']:.1f}" if not pd.isna(summary["1M_dir_acc"]) else "N/A")
    col3.metric("1Y MAPE (%)", f"{summary['1Y_mape']:.2f}" if not pd.isna(summary["1Y_mape"]) else "N/A", f"count {summary['1Y_count']}")
    st.markdown("Directional accuracy = % of matured rows where predicted direction (up/down) matched actual direction.")

st.caption("Everything integrated inside the app UI — predictions logged automatically and matured actuals updated. No need to download unless you want a local copy.")
