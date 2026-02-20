# web/scripts/god_apply_monday_fill.py
import os
import json
from datetime import datetime
import pandas as pd
import yfinance as yf

STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # .../web
    "public", "data", "god_state.json"
)

FEE_RATE = 0.0002
SLIPPAGE = 0.0


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def fetch_open_price(ticker: str) -> float | None:
    """
    月曜の寄り値（ざっくり）：
    直近数日の日足を取り、最後の行のOpenを使う。
    """
    df = yf.download(ticker, period="7d", interval="1d", progress=False).dropna()
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    o = float(df.iloc[-1]["Open"])
    if not pd.isna(o):
        return o
    return None


def main():
    if not os.path.exists(STATE_PATH):
        raise SystemExit(f"Missing: {STATE_PATH}")

    st = load_json(STATE_PATH)
    status = str(st.get("status", "")).upper()
    action = str(st.get("action", "")).upper()
    target = st.get("target", None)

    # すでに確定済みなら何もしない
    if status != "PENDING":
        print("No pending. Skip.")
        return

    now = datetime.now().isoformat(timespec="seconds")

    if action == "BUY" and target:
        op = fetch_open_price(str(target))
        if op is None:
            # 寄り値取れないなら pending のまま（事故防止）
            st["reason"] = f"Pending BUY but open price not available. ({now})"
            save_json(STATE_PATH, st)
            print("Open price unavailable. Kept PENDING.")
            return

        entry = op * (1.0 + SLIPPAGE) * (1.0 + FEE_RATE)
        st["status"] = "IN_POSITION"
        st["action"] = "HOLD"
        st["entry_price"] = float(entry)
        st["reason"] = f"Filled at open. ({now})"
        save_json(STATE_PATH, st)
        print("Applied fill:", st)
        return

    # CASH（またはSELL運用にしたいならここを拡張）
    st["status"] = "CASH"
    st["action"] = "HOLD"
    st["entry_price"] = None
    st["reason"] = f"No position. ({now})"
    save_json(STATE_PATH, st)
    print("Set CASH:", st)


if __name__ == "__main__":
    main()
