import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
# 神コード対象（あなたの「一番成績のいいコード」準拠）
TICKERS = ["^GSPC", "GLD", "^N225", "VT", "SLV","XLP","XLV","XLU","IBB","SMH","QQQ","ITA"]

# Idle時に持つ短期国債（検証用：SHY）
CASH_TICKER = "SHY"  # 米国短期国債ETF

DATA_START = "1985-01-01"
BT_START   = "2000-01-01"

# Costs (片道)
FEE_RATE = 0.0002
SLIPPAGE = 0.0

# 神コード週足設定
W_LOOKBACK = 260
MIN_TOUCHES = 3
TOUCH_TOL = 0.005
RETEST_TOL = 0.003
TOUCH_WINDOW = 8
MIN_TOUCHES_IN_WINDOW = 2
LOWER_BODY_BREAK_TOL = 0.0

# Wave3設定
M_LOOKBACK = 12
MA200_PERIOD = 120

# レバ（リスク資産側のみ）
LEV = 3.0

# =========================
# ★エントリー優先順位モード
# =========================
PRIORITY_MODE = "RISK_ADJ"
PRIORITY_LIST = ["^GSPC", "VT", "GLD", "^N225", "SLV"]  # ※あなたのコードに合わせる（^VTは使わない）

# ATR設定（週足）
ATR_PERIOD = 14

# =========================
# DATA HELPERS
# =========================
def download_daily(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, interval="1d", progress=False).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df

def to_weekly(df_d: pd.DataFrame) -> pd.DataFrame:
    return df_d.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

def to_monthly(df_d: pd.DataFrame) -> pd.DataFrame:
    return df_d.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

# =========================
# 神コード：水平ウィックゾーン（逐次）
# =========================
def find_horizontal_wick_zone(high_arr: np.ndarray) -> float:
    n = len(high_arr)
    if n < MIN_TOUCHES:
        return np.nan

    start = max(0, n - W_LOOKBACK)
    seg = high_arr[start:n]

    for i in range(len(seg) - 1, -1, -1):
        level = seg[i]
        rel = np.abs(seg - level) / level
        touches = np.where(rel <= TOUCH_TOL)[0]
        if len(touches) >= MIN_TOUCHES:
            idx = touches + start
            return float(np.max(high_arr[idx]))

    return np.nan

# =========================
# Wave3（月足 + MA）
# =========================
def build_wave3_monthly_gate(df_d: pd.DataFrame) -> pd.Series:
    m = to_monthly(df_d).copy()
    m["MA200"] = m["Close"].rolling(MA200_PERIOD).mean()

    gate = np.zeros(len(m), dtype=bool)
    locked = False
    res = np.nan

    lows = m["Low"].to_numpy(float)
    highs = m["High"].to_numpy(float)
    closes = m["Close"].to_numpy(float)
    ma200 = m["MA200"].to_numpy(float)

    for i in range(len(m)):
        if i < M_LOOKBACK or np.isnan(ma200[i]):
            continue

        s = i - M_LOOKBACK
        prev_low = float(np.min(lows[s:i]))          # 当月除く
        window_high = float(np.max(highs[s:i+1]))    # 当月含む（RES固定）

        # ロック開始
        if (not locked) and (lows[i] < prev_low):
            locked = True
            res = window_high
            continue

        if locked:
            # RES奪還 + MA上で解除
            if (closes[i] > res) and (closes[i] > ma200[i]):
                locked = False
                res = np.nan
                gate[i] = True
        else:
            gate[i] = (closes[i] > ma200[i])

    return pd.Series(gate, index=m.index, name="WAVE3_OK")

def map_monthly_to_weekly(m_gate: pd.Series, w_idx: pd.DatetimeIndex) -> np.ndarray:
    return m_gate.reindex(w_idx, method="ffill").fillna(False).to_numpy(dtype=bool)

# =========================
# ATR（週足）
# =========================
def weekly_atr_pct(df_w: pd.DataFrame, period: int = 14) -> np.ndarray:
    h = df_w["High"].to_numpy(float)
    l = df_w["Low"].to_numpy(float)
    c = df_w["Close"].to_numpy(float)

    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan

    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr, index=df_w.index).rolling(period).mean().to_numpy(float)
    atr_pct = atr / c
    return atr_pct

# =========================
# 各ティッカーの週足シグナル（事前計算）
# =========================
def build_weekly_signals_for_ticker(df_d: pd.DataFrame) -> pd.DataFrame:
    df_w_full = to_weekly(df_d).copy()
    df_w = df_w_full[df_w_full.index >= BT_START].copy()
    if len(df_w) < 60:
        raise ValueError("Not enough weekly bars after BT_START")

    idx = df_w.index
    h = df_w["High"].to_numpy(float)
    l = df_w["Low"].to_numpy(float)
    c = df_w["Close"].to_numpy(float)

    # zone（逐次）
    zone = np.full(len(df_w), np.nan, dtype=float)
    for i in range(len(df_w)):
        zone[i] = find_horizontal_wick_zone(h[:i+1])

    first_break = (c > zone) & ~np.isnan(zone)

    retest = np.full(len(df_w), False, dtype=bool)
    for i in range(len(df_w)):
        z = zone[i]
        if np.isnan(z):
            continue
        retest[i] = (l[i] <= z * (1.0 + RETEST_TOL)) and (l[i] >= z * (1.0 - 3.0 * RETEST_TOL))

    rebreak = (c > zone) & ~np.isnan(zone)

    touch = np.full(len(df_w), False, dtype=bool)
    for i in range(len(df_w)):
        z = zone[i]
        if np.isnan(z):
            continue
        touch[i] = (abs(h[i] - z) / z <= TOUCH_TOL)

    touch_weak = np.full(len(df_w), False, dtype=bool)
    for i in range(len(df_w)):
        s = max(0, i - TOUCH_WINDOW + 1)
        cnt = int(np.sum(touch[s:i+1]))
        touch_weak[i] = (cnt < MIN_TOUCHES_IN_WINDOW)

    lower_body_break = np.full(len(df_w), False, dtype=bool)
    for i in range(len(df_w)):
        z = zone[i]
        if np.isnan(z):
            continue
        lower_body_break[i] = (c[i] < z * (1.0 - LOWER_BODY_BREAK_TOL))

    exit_signal = touch_weak & lower_body_break

    # Wave3 gate
    m_gate = build_wave3_monthly_gate(df_d)
    w_gate = map_monthly_to_weekly(m_gate, idx)

    # ATR%
    atr_pct = weekly_atr_pct(df_w, ATR_PERIOD)

    out = pd.DataFrame(index=idx)
    out["High"] = h
    out["Low"] = l
    out["Close"] = c
    out["Zone"] = zone
    out["ATR_PCT"] = atr_pct
    out["FirstBreak"] = first_break
    out["Retest"] = retest
    out["Rebreak"] = rebreak
    out["Exit"] = exit_signal
    out["WaveOK"] = w_gate
    return out

# =========================
# 優先順位で1つ選ぶ
# =========================
def pick_candidate(candidates: list[dict]) -> dict:
    if PRIORITY_MODE == "FIXED":
        rank = {t: i for i, t in enumerate(PRIORITY_LIST)}
        candidates.sort(key=lambda x: rank.get(x["ticker"], 10**9))
        return candidates[0]

    if PRIORITY_MODE == "STRENGTH":
        candidates.sort(key=lambda x: x["strength"], reverse=True)
        return candidates[0]

    if PRIORITY_MODE == "RISK_ADJ":
        for x in candidates:
            ap = x["atr_pct"]
            x["score"] = x["strength"] / ap if (ap is not None and np.isfinite(ap) and ap > 0) else -np.inf
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[0]

    candidates.sort(key=lambda x: x["strength"], reverse=True)
    return candidates[0]

# =========================
# 検証：IdleをSHYに置き換える 1ポジ運用
# - リスク資産：LEV適用
# - SHY：LEVなし（1倍）
# - スイッチ時のみ手数料を払う（売り/買い）
# =========================
def backtest_single_position_with_cash_rotation(signals_by_ticker: dict[str, pd.DataFrame], shy_w: pd.DataFrame):
    # 共通インデックス（リスク銘柄共通）
    common_idx = None
    for _, df in signals_by_ticker.items():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    common_idx = common_idx.sort_values()

    # SHYの週足も合わせる
    common_idx = common_idx.intersection(shy_w.index).sort_values()
    if len(common_idx) < 60:
        raise ValueError("Common weekly index too short (with CASH_TICKER)")

    breakout_seen = {t: False for t in signals_by_ticker}
    retest_seen   = {t: False for t in signals_by_ticker}

    cash_equity = 1.0  # 「いま持ってる資産」の評価額（SHY時は毎週増減する＝実運用寄り）
    current = None     # NoneならSHY保有中、文字列ならリスク銘柄保有中
    entry_price = np.nan  # リスク銘柄エントリー価格（手数料込み）

    mtm_curve = []
    realized_curve = []

    trades = []
    wins = 0
    trades_by_ticker = {t: 0 for t in signals_by_ticker}

    # SHYの「直前週」Close（SHY保有中の利回り反映用）
    shy_prev_close = float(shy_w.loc[common_idx[0], "Close"])

    in_pos_weeks = 0
    idle_weeks = 0
    longest_idle = 0
    cur_idle = 0

    for dt in common_idx:
        row = {t: signals_by_ticker[t].loc[dt] for t in signals_by_ticker}
        shy_close = float(shy_w.loc[dt, "Close"])

        # =====================
        # 1) SHY保有中（Idleを埋める）
        # =====================
        if current is None:
            # SHYの週次リターンを反映（＝Idleでも資産が増減）
            if np.isfinite(shy_prev_close) and shy_prev_close > 0:
                cash_equity *= (shy_close / shy_prev_close)
            shy_prev_close = shy_close

            idle_weeks += 1
            cur_idle += 1
            longest_idle = max(longest_idle, cur_idle)

            # エントリー探索（神コードはそのまま）
            candidates = []
            for t, r in row.items():
                if bool(r["WaveOK"]) and bool(r["FirstBreak"]):
                    breakout_seen[t] = True
                if breakout_seen[t] and bool(r["Retest"]):
                    retest_seen[t] = True

                if bool(r["WaveOK"]) and breakout_seen[t] and retest_seen[t] and bool(r["Rebreak"]):
                    z = float(r["Zone"])
                    cc = float(r["Close"])
                    atr_pct = float(r["ATR_PCT"]) if np.isfinite(r["ATR_PCT"]) else np.nan
                    strength = (cc / z - 1.0) if (np.isfinite(z) and z > 0) else -np.inf
                    candidates.append({
                        "ticker": t,
                        "close": cc,
                        "zone": z,
                        "atr_pct": atr_pct,
                        "strength": strength
                    })

            if candidates:
                picked = pick_candidate(candidates)

                # SHY売却（手数料）
                cash_equity *= (1.0 - FEE_RATE)

                # リスク資産購入（手数料込みのエントリー価格）
                current = picked["ticker"]
                entry_price = picked["close"] * (1.0 + SLIPPAGE) * (1.0 + FEE_RATE)

                # 監視フラグリセット（乗り換え禁止）
                for t in signals_by_ticker:
                    breakout_seen[t] = False
                    retest_seen[t] = False

                # Idle streak リセット
                cur_idle = 0

            # SHY中は MTM=Realized とみなしてOK（実運用の評価額）
            mtm_curve.append(cash_equity)
            realized_curve.append(cash_equity)
            continue

        # =====================
        # 2) リスク資産保有中（LEVあり）
        # =====================
        in_pos_weeks += 1
        cur_idle = 0

        r = row[current]
        close_now = float(r["Close"])
        r_full_now = close_now / entry_price
        marked = cash_equity * (1.0 + LEV * (r_full_now - 1.0))
        if marked < 0.0:
            marked = 0.0

        if bool(r["Exit"]):
            # リスク資産売却（手数料）
            exit_px = close_now * (1.0 - SLIPPAGE) * (1.0 - FEE_RATE)
            r_full = exit_px / entry_price
            trade_r = 1.0 + LEV * (r_full - 1.0)
            if trade_r < 0.0:
                trade_r = 0.0

            cash_equity *= trade_r
            trades.append(trade_r)
            trades_by_ticker[current] += 1
            if trade_r > 1.0:
                wins += 1

            # すぐSHYを買う（手数料）
            cash_equity *= (1.0 - FEE_RATE)
            current = None
            entry_price = np.nan

            # SHY保有に戻るので、直前Closeを更新（このdtのCloseから次週へ）
            shy_prev_close = shy_close

            # Exit週は確定後に揃える（ズレ防止）
            mtm_curve.append(cash_equity)
            realized_curve.append(cash_equity)
        else:
            mtm_curve.append(marked)
            realized_curve.append(cash_equity)

    mtm = np.array(mtm_curve, dtype=float)
    realized = np.array(realized_curve, dtype=float)

    years = (common_idx[-1] - common_idx[0]).days / 365.25
    final = float(realized[-1])
    cagr = final ** (1.0 / years) - 1.0 if years > 0 else np.nan
    mdd = float(np.min(mtm / np.maximum.accumulate(mtm) - 1.0))
    n_trades = len(trades)
    winrate = (wins / n_trades) if n_trades else np.nan

    total_weeks = len(common_idx)
    exposure = (in_pos_weeks / total_weeks) if total_weeks else np.nan

    return {
        "index": common_idx,
        "mtm_curve": mtm,
        "realized_curve": realized,
        "start": str(common_idx[0].date()),
        "end": str(common_idx[-1].date()),
        "final": final,
        "cagr": float(cagr),
        "mdd": float(mdd),
        "trades": int(n_trades),
        "winrate": float(winrate) if not np.isnan(winrate) else np.nan,
        "trades_by_ticker": trades_by_ticker,
        "idle_stats": {
            "total_weeks": int(total_weeks),
            "in_position_weeks": int(in_pos_weeks),
            "idle_weeks": int(idle_weeks),
            "exposure_ratio": float(exposure),
            "longest_idle_streak": int(longest_idle),
        }
    }

# =========================
# MAIN
# =========================
def main():
    # 1) リスク銘柄のsignals
    signals_by_ticker = {}
    for t in TICKERS:
        try:
            df_d = download_daily(t, DATA_START)
            sig = build_weekly_signals_for_ticker(df_d)
            signals_by_ticker[t] = sig
            print(f"Loaded: {t}  weeks={len(sig)}  ({sig.index[0].date()} -> {sig.index[-1].date()})")
        except Exception as e:
            print("ERROR loading:", t, e)

    if len(signals_by_ticker) < 2:
        raise SystemExit("Not enough tickers loaded.")

    # 2) SHY週足
    try:
        shy_d = download_daily(CASH_TICKER, DATA_START)
        shy_w_full = to_weekly(shy_d).copy()
        shy_w = shy_w_full[shy_w_full.index >= BT_START].copy()
        if len(shy_w) < 60:
            raise ValueError("Not enough weekly bars for CASH_TICKER")
        print(f"Loaded: {CASH_TICKER}  weeks={len(shy_w)}  ({shy_w.index[0].date()} -> {shy_w.index[-1].date()})")
    except Exception as e:
        raise SystemExit(f"ERROR loading CASH_TICKER={CASH_TICKER}: {e}")

    # 3) backtest
    res = backtest_single_position_with_cash_rotation(signals_by_ticker, shy_w)

    print("\n=== SINGLE POSITION PORTFOLIO (Idle -> CASH_TICKER Rotation) ===")
    print(f"CASH_TICKER: {CASH_TICKER} (unlevered)")
    print(f"PRIORITY_MODE: {PRIORITY_MODE}")
    print(f"Period: {res['start']} -> {res['end']}")
    print(f"LEV (risk assets only): {LEV}")
    print(f"Final: {res['final']:.6f}")
    print(f"CAGR: {res['cagr']*100:.2f}%")
    print(f"MaxDD(MTM): {res['mdd']*100:.2f}%")
    print(f"Trades: {res['trades']}")
    if not np.isnan(res["winrate"]):
        print(f"WinRate: {res['winrate']*100:.1f}%")

    st = res["idle_stats"]
    print("\n--- Idle / Exposure ---")
    print(f"Total weeks: {st['total_weeks']}")
    print(f"In-position weeks: {st['in_position_weeks']}")
    print(f"Idle weeks (CASH_TICKER): {st['idle_weeks']}")
    print(f"Exposure ratio: {st['exposure_ratio']*100:.1f}%")
    print(f"Longest idle streak: {st['longest_idle_streak']} weeks")

    print("\n--- Trades by Ticker ---")
    for t, n in sorted(res["trades_by_ticker"].items(), key=lambda x: (-x[1], x[0])):
        print(f"{t}: {n}")

    # Plot
    plt.figure()
    plt.plot(res["index"], res["mtm_curve"], label="Equity (Mark-to-Market)")
    plt.plot(res["index"], res["realized_curve"], label="Equity (Realized-like)")
    plt.title(f"Equity Curve (MTM)  LEV={LEV}  PRIORITY={PRIORITY_MODE}  MA={MA200_PERIOD}  Idle->{CASH_TICKER}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
