# web/scripts/god_apply_monday_fill.py
import os
import json
from datetime import datetime
import pandas as pd
import yfinance as yf

# Paths (repo/web)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../web
STATE_PATH = os.path.join(BASE_DIR, "public", "god_state.json")
HIST_PATH = os.path.join(BASE_DIR, "public", "history.json")

FEE_RATE = 0.0002
SLIPPAGE = 0.0


# ------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------
def load_json(path: str):
    # utf-8-sig: BOM混入耐性
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_history(hist_path: str, row: dict, keep: int = 50):
    """
    public/history.json に履歴(list[dict])を保存。
    同一キー(asof,status,action,target)は置換。
    asof降順でkeep件。
    """
    hist = []
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8-sig") as f:
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

    hist.sort(key=lambda x: x.get("asof", ""), reverse=True)
    hist = hist[:keep]
    save_json(hist_path, hist)


# ------------------------------------------------------------
# Market data helpers
# ------------------------------------------------------------
def fetch_open_price(ticker: str) -> float | None:
    """
    直近7日の日足から最新行のOpenを取得
    """
    df = yf.download(ticker, period="7d", interval="1d", progress=False).dropna()
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    o = float(df.iloc[-1]["Open"])
    return None if pd.isna(o) else o


def fetch_week_low_for_asof(ticker: str, asof_ymd: str) -> float | None:
    """
    asof(YYYY-MM-DD, 金曜想定)の週足Lowを返す。
    PENDING側でstop_priceが無い場合の保険。
    """
    try:
        asof = pd.Timestamp(asof_ymd)
    except Exception:
        return None

    start = (asof - pd.Timedelta(days=21)).date().isoformat()
    end = (asof + pd.Timedelta(days=3)).date().isoformat()

    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False).dropna()
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].copy()
    df.index = pd.to_datetime(df.index)

    w = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()

    if asof not in w.index:
        w = w[w.index <= asof]
        if w.empty:
            return None
        low = float(w.iloc[-1]["Low"])
        return None if pd.isna(low) else low

    low = float(w.loc[asof, "Low"])
    return None if pd.isna(low) else low


# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
def main():
    if not os.path.exists(STATE_PATH):
        raise SystemExit(f"Missing: {STATE_PATH}")

    st = load_json(STATE_PATH)

    # ------------------------------------------------------------
    # 安全パッチ：必須キーを常に保持
    # ------------------------------------------------------------
    st.setdefault("entry_price", None)
    st.setdefault("stop_price", None)
    st.setdefault("take_profit", None)
    st.setdefault("valid_until", st.get("pending_for", None))
    st.setdefault("updated_at", None)

    status = str(st.get("status", "")).upper()
    action = str(st.get("action", "")).upper()
    target = st.get("target", None)

    if status != "PENDING":
        print("No pending. Skip.")
        return

    now = datetime.now().isoformat(timespec="seconds")
    st["updated_at"] = now

    # ------------------------------------------------------------
    # BUY約定
    # ------------------------------------------------------------
    if action == "BUY" and target:
        op = fetch_open_price(str(target))
        if op is None:
            st["reason"] = f"Pending BUY but open price not available. ({now})"
            save_json(STATE_PATH, st)
            append_history(HIST_PATH, st, keep=50)
            print("Open price unavailable. Kept PENDING.")
            return

        entry = op * (1.0 + SLIPPAGE) * (1.0 + FEE_RATE)

        st["status"] = "IN_POSITION"
        st["action"] = "HOLD"
        st["entry_price"] = float(entry)

        # --------------------------------------------------------
        # stop_price維持（Noneのときのみ補完）
        # --------------------------------------------------------
        if st.get("stop_price") is None:
            asof_ymd = st.get("asof")
            if asof_ymd:
                sp = fetch_week_low_for_asof(str(target), str(asof_ymd))
                if sp is not None:
                    st["stop_price"] = float(sp)

        st["reason"] = f"Filled at open. ({now})"

        save_json(STATE_PATH, st)
        append_history(HIST_PATH, st, keep=50)
        print("Applied fill:", st)
        return

    # ------------------------------------------------------------
    # BUY以外（CASH想定）
    # ------------------------------------------------------------
    st["status"] = "CASH"
    st["action"] = "HOLD"
    st["entry_price"] = None
    st["stop_price"] = None
    st["take_profit"] = None
    st["reason"] = f"No position. ({now})"

    save_json(STATE_PATH, st)
    append_history(HIST_PATH, st, keep=50)
    print("Set CASH:", st)


if __name__ == "__main__":
    main()