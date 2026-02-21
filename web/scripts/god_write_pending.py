# web/scripts/god_write_pending.py
import os
import json
from datetime import timedelta

import numpy as np
import pandas as pd

import wave3_god_core as G

# Paths (repo/web)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../web
OUT_DIR = os.path.join(BASE_DIR, "public")
OUT_PATH = os.path.join(OUT_DIR, "god_state.json")
HIST_PATH = os.path.join(OUT_DIR, "history.json")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_history(hist_path: str, row: dict, keep: int = 50):
    """
    public/history.json に履歴（list[dict]）として保存。
    同一キー(asof,status,action,target)は置換、なければ追加。
    asof降順で keep 件に制限。
    """
    hist = []
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            if not isinstance(hist, list):
                hist = []
        except Exception:
            hist = []

    key = (row.get("asof"), row.get("status"), row.get("action"), row.get("target"))

    def row_key(x):
        return (x.get("asof"), x.get("status"), x.get("action"), x.get("target"))

    replaced = False
    for i in range(len(hist)):
        if row_key(hist[i]) == key:
            hist[i] = row
            replaced = True
            break
    if not replaced:
        hist.append(row)

    # YYYY-MM-DD string sort is safe for chronological ordering
    hist.sort(key=lambda x: x.get("asof", ""), reverse=True)
    hist = hist[:keep]

    save_json(hist_path, hist)


def _to_weekly(df_d: pd.DataFrame) -> pd.DataFrame:
    if hasattr(G, "to_weekly"):
        return G.to_weekly(df_d)

    return df_d.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()


def build_weekly_signals(df_d: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer GodCore implementation if exists.
    Fallback creates minimal weekly frame.
    """
    if hasattr(G, "build_weekly_signals_for_ticker"):
        return G.build_weekly_signals_for_ticker(df_d).copy()

    df_w = _to_weekly(df_d).copy()
    df_w["Zone"] = np.nan

    h = df_w["High"].to_numpy(float)
    for i in range(len(df_w)):
        if hasattr(G, "find_horizontal_wick_zone"):
            df_w.iloc[i, df_w.columns.get_loc("Zone")] = float(
                G.find_horizontal_wick_zone(h[: i + 1])
            )

    df_w["WaveOK"] = True
    df_w["FirstBreak"] = False
    df_w["Retest"] = False
    df_w["Rebreak"] = False
    df_w["Exit"] = False
    if "ATR_PCT" not in df_w.columns:
        df_w["ATR_PCT"] = np.nan
    return df_w


def ensure_cross_events(df_w: pd.DataFrame) -> pd.DataFrame:
    """
    Provide cross-up event columns if GodCore didn't.
    """
    df = df_w.copy()
    if "Zone" not in df.columns:
        df["Zone"] = np.nan

    for col in ["Close", "Low", "High"]:
        if col not in df.columns:
            df[col] = np.nan

    c = df["Close"].to_numpy(float)
    z = df["Zone"].to_numpy(float)

    prev_c = np.roll(c, 1)
    prev_z = np.roll(z, 1)
    prev_c[0] = np.nan
    prev_z[0] = np.nan

    above = (c > z) & np.isfinite(z)
    prev_above = (prev_c > prev_z) & np.isfinite(prev_z)

    cross_up = above & (~prev_above)

    df["FirstBreak"] = cross_up

    if "Retest" not in df.columns:
        tol = float(getattr(G, "RETEST_TOL", 0.003))
        l = df["Low"].to_numpy(float)
        retest = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if not np.isfinite(z[i]):
                continue
            retest[i] = (l[i] <= z[i] * (1.0 + tol)) and (l[i] >= z[i] * (1.0 - 3.0 * tol))
        df["Retest"] = retest

    df["RebreakEvent"] = cross_up
    return df


def compute_entry_event(df: pd.DataFrame) -> np.ndarray:
    """
    Entry event sequence:
    breakout_seen -> retest_seen -> rebreak event
    """
    entry = np.zeros(len(df), dtype=bool)

    breakout_seen = False
    retest_seen = False

    for i in range(len(df)):
        r = df.iloc[i]
        wave_ok = bool(r.get("WaveOK", False))
        first = bool(r.get("FirstBreak", False))
        retest = bool(r.get("Retest", False))
        rebreak_ev = bool(r.get("RebreakEvent", False))

        if wave_ok and first:
            breakout_seen = True
        if breakout_seen and retest:
            retest_seen = True

        ok = wave_ok and breakout_seen and retest_seen and rebreak_ev
        entry[i] = ok

        if ok:
            breakout_seen = False
            retest_seen = False

    return entry


def rank_candidates(cands: list[dict]) -> list[dict]:
    mode = getattr(G, "PRIORITY_MODE", "RISK_ADJ")
    plist = getattr(G, "PRIORITY_LIST", [])

    if mode == "FIXED":
        rank = {t: i for i, t in enumerate(plist)}
        return sorted(cands, key=lambda x: rank.get(x["ticker"], 10**9))

    if mode == "STRENGTH":
        return sorted(cands, key=lambda x: x.get("strength", -np.inf), reverse=True)

    # RISK_ADJ: strength / atr_pct
    out = []
    for x in cands:
        ap = x.get("atr_pct", np.nan)
        st = x.get("strength", -np.inf)
        if ap is None or (not np.isfinite(ap)) or ap <= 0:
            score = -1e99
        else:
            score = float(st) / float(ap)
        y = dict(x)
        y["score"] = score
        out.append(y)
    return sorted(out, key=lambda x: x.get("score", -1e99), reverse=True)


def main():
    ensure_dirs()

    tickers = list(getattr(G, "TICKERS", []))
    if not tickers:
        raise SystemExit("G.TICKERS is empty. Check wave3_god_core.py")

    cash_ticker = getattr(G, "CASH_TICKER", "SHY")
    lev = float(getattr(G, "LEV", 3.0))
    data_start = getattr(G, "DATA_START", "1985-01-01")
    bt_start = getattr(G, "BT_START", "2000-01-01")

    # 1) Build weekly signals for all tickers
    sigs: dict[str, pd.DataFrame] = {}
    for t in tickers + [cash_ticker]:
        df_d = G.download_daily(t, data_start)
        w = build_weekly_signals(df_d)

        w = w[w.index >= pd.Timestamp(bt_start)].copy()
        w = ensure_cross_events(w)

        sigs[t] = w

    # 2) Find common index and asof
    common = None
    for df in sigs.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) == 0:
        raise SystemExit("Common index is empty (signals mismatch).")

    common = common.sort_values()
    dt = common[-1]
    asof = str(dt.date())

    pending_for = (dt + timedelta(days=3)).date().isoformat()  # Fri +3 = Mon

    # 3) Detect entry candidates
    cands = []
    for t in tickers:
        df = sigs[t]
        if dt not in df.index:
            continue

        entry = compute_entry_event(df)
        loc = df.index.get_loc(dt)
        if not isinstance(loc, (int, np.integer)):
            continue
        i = int(loc)

        if not entry[i]:
            continue

        r = df.loc[dt]
        z = float(r["Zone"]) if np.isfinite(r.get("Zone", np.nan)) else np.nan
        cc = float(r["Close"])
        ll = float(r["Low"]) if np.isfinite(r.get("Low", np.nan)) else np.nan  # ★ stop候補（当週Low）
        ap = float(r["ATR_PCT"]) if ("ATR_PCT" in df.columns and np.isfinite(r.get("ATR_PCT", np.nan))) else np.nan
        strength = (cc / z - 1.0) if (np.isfinite(z) and z > 0) else -np.inf

        cands.append(
            {
                "ticker": t,
                "close": cc,
                "low": ll,  # ★追加
                "zone": z,
                "atr_pct": ap,
                "strength": float(strength),
            }
        )

    ranked = rank_candidates(cands) if cands else []
    picked = ranked[0] if ranked else None

    # 4) Output
    if picked:
        stop_px = picked.get("low", np.nan)
        out = {
            "asof": asof,
            "status": "PENDING",
            "action": "BUY",
            "target": picked["ticker"],
            "lev": lev,
            "cash_ticker": cash_ticker,
            "pending_for": pending_for,
            "entry_price": None,
            "stop_price": (None if not np.isfinite(stop_px) else float(stop_px)),
            "reason": f"Entry detected (picked by {getattr(G,'PRIORITY_MODE','RISK_ADJ')}).",
        }
    else:
        out = {
            "asof": asof,
            "status": "PENDING",
            "action": "CASH",
            "target": cash_ticker,
            "lev": lev,
            "cash_ticker": cash_ticker,
            "pending_for": pending_for,
            "entry_price": None,
            "stop_price": None,
            "reason": "No entry candidates this week.",
        }

    save_json(OUT_PATH, out)
    append_history(HIST_PATH, out, keep=50)

    print(f"Saved: {OUT_PATH}")
    print(f"History: {HIST_PATH}")
    print(out)


if __name__ == "__main__":
    main()