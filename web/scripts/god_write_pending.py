# web/scripts/god_write_pending.py
import os
import json
from datetime import timedelta
import numpy as np
import pandas as pd

# 🔒 神コードは凍結：importして参照するだけ
import wave3_god_core as G

# 出力先（Next.js public）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../web
OUT_DIR = os.path.join(BASE_DIR, "public", "data")
OUT_PATH = os.path.join(OUT_DIR, "god_state.json")


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _to_weekly(df_d: pd.DataFrame) -> pd.DataFrame:
    # GodCore側にあればそれを使う（凍結参照）
    if hasattr(G, "to_weekly"):
        return G.to_weekly(df_d)
    # フォールバック（念のため）
    return df_d.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()


def build_weekly_signals(df_d: pd.DataFrame) -> pd.DataFrame:
    """
    GodCoreの build_weekly_signals_for_ticker があればそれを使う（最優先）。
    無い/仕様違いの時は最低限の列を揃える（通常ここは通らない想定）。
    """
    if hasattr(G, "build_weekly_signals_for_ticker"):
        return G.build_weekly_signals_for_ticker(df_d).copy()

    # フォールバック（通常ここは通らない）
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
    過剰点灯を潰す肝：
    - FirstBreak は「CloseがZoneを上抜けた週（クロス）」に正規化
    - RebreakEvent も同様に「クロス週」のみにする

    ※ GodCoreが既に良い定義を持っていても、安全側に統一する。
    """
    df = df_w.copy()
    if "Zone" not in df.columns:
        df["Zone"] = np.nan

    # 必須列が無い場合の保険
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

    # クロス定義で上書き
    df["FirstBreak"] = cross_up

    # Retest が無い場合は作る（通常GodCoreにはある）
    if "Retest" not in df.columns:
        tol = float(getattr(G, "RETEST_TOL", 0.003))
        l = df["Low"].to_numpy(float)
        retest = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            if not np.isfinite(z[i]):
                continue
            retest[i] = (l[i] <= z[i] * (1.0 + tol)) and (l[i] >= z[i] * (1.0 - 3.0 * tol))
        df["Retest"] = retest

    # RebreakEvent = クロス週のみ
    df["RebreakEvent"] = cross_up

    return df


def compute_entry_event(df: pd.DataFrame) -> np.ndarray:
    """
    ticker単体の「点灯週」だけ True になる配列（1シーケンス1回）
    - breakout_seen / retest_seen の状態機械
    - ok したらリセットして過剰点灯を殺す
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

    # RISK_ADJ（st / atr_pct）
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

    # 1) 各ティッカーの週足シグナル構築（GodCore参照）
    sigs: dict[str, pd.DataFrame] = {}
    for t in tickers + [cash_ticker]:
        df_d = G.download_daily(t, data_start)
        w = build_weekly_signals(df_d)

        # BT_START以降に絞る（神コードと同じ期間感）
        w = w[w.index >= pd.Timestamp(bt_start)].copy()

        # 過剰点灯を潰す正規化
        w = ensure_cross_events(w)

        sigs[t] = w

    # 2) 共通の最終週（全銘柄揃う週）
    common = None
    for df in sigs.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) == 0:
        raise SystemExit("Common index is empty (signals mismatch).")

    common = common.sort_values()
    dt = common[-1]
    asof = str(dt.date())

    # 次の「月曜」目安（表示用）
    pending_for = (dt + timedelta(days=3)).date().isoformat()  # Fri +3 = Mon

    # 3) 候補抽出（点灯週のみ）
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
        ap = float(r["ATR_PCT"]) if ("ATR_PCT" in df.columns and np.isfinite(r.get("ATR_PCT", np.nan))) else np.nan
        strength = (cc / z - 1.0) if (np.isfinite(z) and z > 0) else -np.inf

        cands.append(
            {
                "ticker": t,
                "close": cc,
                "zone": z,
                "atr_pct": ap,
                "strength": float(strength),
            }
        )

    ranked = rank_candidates(cands) if cands else []
    pick = ranked[0]["ticker"] if ranked else None

    # 4) ★重要：PENDINGは BUY or CASH しか出さない（HOLD禁止）
    if pick:
        out = {
            "asof": asof,
            "status": "PENDING",
            "action": "BUY",
            "target": pick,
            "lev": lev,
            "cash_ticker": cash_ticker,
            "pending_for": pending_for,
            "entry_price": None,
            "stop_price": None,
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
    print(f"Saved: {OUT_PATH}")
    print(out)


if __name__ == "__main__":
    main()