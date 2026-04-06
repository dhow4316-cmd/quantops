"""
QUANTOPS Options Alert Agent — Single Scan Mode (GitHub Actions)
Runs once per trigger, uses a state file committed to repo for persistence.
"""

import os
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
BYBIT_API_KEY    = os.environ["BYBIT_API_KEY"]
BYBIT_API_SECRET = os.environ["BYBIT_API_SECRET"]

BASE_URL  = "https://api.bybit.com"
ASSETS    = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
STATE_FILE = "options_state.json"

# Alert thresholds
THETA_DANGER_TIME_VALUE = 20.0   # USDT
MARK_PRICE_DROP_PCT     = 0.50   # 50% from entry
DELTA_ATM_THRESHOLD     = 0.45
IV_SPIKE_THRESHOLD      = 0.20
STATUS_EVERY_N_SCANS    = 6      # ~30 min heartbeat


# ── State persistence (JSON file in repo) ─────────────────────────────────────
def load_state() -> dict:
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"scan_count": 0, "positions": {}}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Bybit helpers ─────────────────────────────────────────────────────────────
def bybit_signed_request(endpoint, params=None):
    params = params or {}
    timestamp   = str(int(time.time() * 1000))
    recv_window = "5000"
    param_str   = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    sign_str    = f"{timestamp}{BYBIT_API_KEY}{recv_window}{param_str}"
    signature   = hmac.new(
        BYBIT_API_SECRET.encode(), sign_str.encode(), hashlib.sha256
    ).hexdigest()
    headers = {
        "X-BAPI-API-KEY":     BYBIT_API_KEY,
        "X-BAPI-TIMESTAMP":   timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN":        signature,
    }
    r = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()


def bybit_public(endpoint, params=None):
    r = requests.get(f"{BASE_URL}{endpoint}", params=params or {}, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
        r.raise_for_status()
        print(f"  [TG] {msg[:60]}...")
    except Exception as e:
        print(f"  [TG ERROR] {e}")


# ── Data fetchers ─────────────────────────────────────────────────────────────
def get_positions():
    all_pos = []
    for asset in ASSETS:
        try:
            data = bybit_signed_request(
                "/v5/position/list",
                {"category": "option", "baseCoin": asset, "limit": "50"}
            )
            if data.get("retCode") == 0:
                for p in data["result"].get("list", []):
                    if float(p.get("size", 0)) > 0:
                        all_pos.append(p)
        except Exception as e:
            print(f"  [POS ERROR] {asset}: {e}")
    return all_pos


def get_ticker(symbol: str):
    try:
        data = bybit_public("/v5/market/tickers", {"category": "option", "symbol": symbol})
        if data.get("retCode") == 0:
            lst = data["result"].get("list", [])
            return lst[0] if lst else None
    except Exception as e:
        print(f"  [TICKER ERROR] {symbol}: {e}")
    return None


def get_spot(asset: str) -> float:
    try:
        data = bybit_public("/v5/market/tickers", {"category": "spot", "symbol": f"{asset}USDT"})
        if data.get("retCode") == 0:
            lst = data["result"].get("list", [])
            if lst:
                return float(lst[0]["lastPrice"])
    except Exception as e:
        print(f"  [SPOT ERROR] {asset}: {e}")
    return 0.0


def hours_to_expiry(symbol: str) -> float:
    try:
        parts    = symbol.split("-")
        expiry   = datetime.strptime(parts[1], "%d%b%y").replace(
            hour=8, minute=0, second=0, tzinfo=timezone.utc
        )
        return max(0.0, (expiry - datetime.now(timezone.utc)).total_seconds() / 3600)
    except Exception:
        return 999.0


# ── Alert logic ───────────────────────────────────────────────────────────────
def check_position(pos: dict, ps: dict) -> list[str]:
    """ps = per-symbol state dict (mutated in place)"""
    alerts  = []
    symbol  = pos["symbol"]
    asset   = symbol.split("-")[0]
    parts   = symbol.split("-")
    strike  = float(parts[2]) if len(parts) >= 3 else 0
    opt_type = parts[3] if len(parts) >= 4 else ""

    entry_px = float(pos.get("avgPrice", 0) or 0)

    ticker = get_ticker(symbol)
    if not ticker:
        return alerts

    mark_px   = float(ticker.get("markPrice", 0) or 0)
    delta     = float(ticker.get("delta", 0) or 0)
    iv        = float(ticker.get("markIv", 0) or 0)
    spot      = get_spot(asset)
    hrs_left  = hours_to_expiry(symbol)
    intrinsic = max(0.0, spot - strike) if opt_type == "C" else max(0.0, strike - spot)
    time_val  = max(0.0, mark_px - intrinsic)
    roi_pct   = ((mark_px - entry_px) / entry_px * 100) if entry_px > 0 else 0

    # 1 — ITM Flip
    is_itm   = intrinsic > 0
    prev_itm = ps.get("itm")
    if prev_itm is not None and prev_itm != is_itm:
        label = "🟢 NOW IN THE MONEY" if is_itm else "🔴 NOW OUT OF THE MONEY"
        alerts.append(
            f"⚡ <b>ITM FLIP — {symbol}</b>\n{label}\n"
            f"Spot: ${spot:,.2f} | Strike: ${strike:,.0f}\n"
            f"Intrinsic: ${intrinsic:.2f} | Mark: ${mark_px:.2f} | ROI: {roi_pct:+.1f}%"
        )
    ps["itm"] = is_itm

    # 2 — Theta Danger
    is_theta = time_val < THETA_DANGER_TIME_VALUE and hrs_left < 12
    if is_theta and not ps.get("theta_danger"):
        alerts.append(
            f"⏰ <b>THETA DANGER — {symbol}</b>\n"
            f"Time value: ${time_val:.2f} | {hrs_left:.1f}h to expiry\n"
            f"Mark: ${mark_px:.2f} | Intrinsic: ${intrinsic:.2f}\n⚠️ Act now or accept loss"
        )
    ps["theta_danger"] = is_theta

    # 3 — Expiry Warnings (6h, 2h, 1h)
    for warn_h in [6, 2, 1]:
        key = f"exp_warned_{warn_h}"
        if hrs_left <= warn_h and not ps.get(key):
            status = "ITM ✅" if is_itm else "OTM ❌ (expires worthless)"
            alerts.append(
                f"🔔 <b>EXPIRY {warn_h}H WARNING — {symbol}</b>\n"
                f"⏳ {hrs_left:.1f}h remaining | {status}\n"
                f"Spot: ${spot:,.2f} | Strike: ${strike:,.0f}\n"
                f"Mark: ${mark_px:.2f} | ROI: {roi_pct:+.1f}%"
            )
            ps[key] = True

    # 4 — Stop Loss
    is_stop = entry_px > 0 and mark_px < entry_px * (1 - MARK_PRICE_DROP_PCT)
    if is_stop and not ps.get("stop"):
        alerts.append(
            f"🛑 <b>STOP LOSS — {symbol}</b>\n"
            f"Down {abs(roi_pct):.0f}% from entry\n"
            f"Entry: ${entry_px:.2f} → Now: ${mark_px:.2f}\nConsider closing."
        )
    ps["stop"] = is_stop

    # 5 — Delta ATM Cross
    prev_d = ps.get("delta")
    if prev_d is not None:
        crossed = (prev_d < DELTA_ATM_THRESHOLD and abs(delta) >= DELTA_ATM_THRESHOLD) or \
                  (prev_d >= DELTA_ATM_THRESHOLD and abs(delta) < DELTA_ATM_THRESHOLD)
        if crossed:
            direction = "above" if abs(delta) >= DELTA_ATM_THRESHOLD else "below"
            alerts.append(
                f"📊 <b>DELTA ATM CROSS — {symbol}</b>\n"
                f"Delta moved {direction} 0.50 → now {delta:.3f}\n"
                f"Spot: ${spot:,.2f} | Mark: ${mark_px:.2f}"
            )
    ps["delta"] = abs(delta)

    # 6 — IV Spike
    prev_iv = ps.get("iv")
    if prev_iv and prev_iv > 0:
        iv_chg = (iv - prev_iv) / prev_iv
        if abs(iv_chg) >= IV_SPIKE_THRESHOLD:
            tag = "📈 SPIKE" if iv_chg > 0 else "📉 CRUSH"
            alerts.append(
                f"🌪️ <b>IV {tag} — {symbol}</b>\n"
                f"IV {iv_chg*100:+.1f}%: {prev_iv:.1%} → {iv:.1%}\n"
                f"Mark: ${mark_px:.2f}"
            )
    ps["iv"] = iv

    return alerts


def status_report(positions: list):
    if not positions:
        send_telegram(
            "🤖 <b>QUANTOPS Options Agent</b>\n"
            "No open options positions found.\n"
            f"Watching: {', '.join(ASSETS)}"
        )
        return
    lines = ["🤖 <b>QUANTOPS — Options Summary</b>\n"]
    for pos in positions:
        sym      = pos["symbol"]
        mark_px  = float(pos.get("markPrice", 0) or 0)
        entry_px = float(pos.get("avgPrice", 0) or 0)
        unreal   = float(pos.get("unrealisedPnl", 0) or 0)
        size     = float(pos.get("size", 0) or 0)
        hrs      = hours_to_expiry(sym)
        roi      = ((mark_px - entry_px) / entry_px * 100) if entry_px > 0 else 0
        e        = "🟢" if unreal >= 0 else "🔴"
        lines.append(
            f"{e} <b>{sym}</b>\n"
            f"   Size: {size} | Entry: ${entry_px:.2f} | Mark: ${mark_px:.2f}\n"
            f"   ROI: {roi:+.1f}% | P&L: {unreal:+.2f} USDT | ⏳{hrs:.1f}h\n"
        )
    send_telegram("\n".join(lines))


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    state      = load_state()
    scan_count = state.get("scan_count", 0) + 1
    state["scan_count"] = scan_count
    pos_states = state.get("positions", {})

    print(f"[SCAN #{scan_count}] {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    positions = get_open_options_positions() if False else get_positions()
    print(f"  Open options positions: {len(positions)}")

    # Periodic status report
    if scan_count % STATUS_EVERY_N_SCANS == 1:
        status_report(positions)

    # Check each position
    for pos in positions:
        sym = pos["symbol"]
        if sym not in pos_states:
            pos_states[sym] = {}
        alerts = check_position(pos, pos_states[sym])
        for alert in alerts:
            send_telegram(alert)
            time.sleep(1)

    # Clean up closed positions from state
    active_syms = {p["symbol"] for p in positions}
    for sym in list(pos_states.keys()):
        if sym not in active_syms:
            del pos_states[sym]

    state["positions"] = pos_states
    save_state(state)
    print(f"  State saved. Done.")


if __name__ == "__main__":
    main()
