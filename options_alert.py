"""
QUANTOPS Options Alert Agent
Monitors BTC, ETH, SOL, XRP, DOGE options positions on Bybit
Sends Telegram alerts on key conditions
"""

import os
import time
import hmac
import hashlib
import requests
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
BYBIT_API_KEY    = os.environ["BYBIT_API_KEY"]
BYBIT_API_SECRET = os.environ["BYBIT_API_SECRET"]

BASE_URL = "https://api.bybit.com"
ASSETS   = ["BTC", "ETH", "SOL", "XRP", "DOGE"]

# Alert thresholds
THETA_DANGER_TIME_VALUE   = 20.0   # USDT — alert when time value bleeds below this
MARK_PRICE_DROP_PCT       = 0.50   # 50% drop from entry triggers stop alert
DELTA_ATM_THRESHOLD       = 0.45   # Alert when delta crosses near 0.5 (ATM)
IV_SPIKE_THRESHOLD        = 0.20   # 20% IV spike vs prev reading
HOURS_TO_EXPIRY_WARNING   = 6      # Alert when < 6 hours to expiry

# ── Bybit Auth ────────────────────────────────────────────────────────────────
def bybit_signed_request(endpoint, params=None):
    """Make authenticated GET request to Bybit V5 private endpoints."""
    params = params or {}
    timestamp  = str(int(time.time() * 1000))
    recv_window = "5000"

    param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    sign_str  = f"{timestamp}{BYBIT_API_KEY}{recv_window}{param_str}"
    signature = hmac.new(
        BYBIT_API_SECRET.encode(), sign_str.encode(), hashlib.sha256
    ).hexdigest()

    headers = {
        "X-BAPI-API-KEY":     BYBIT_API_KEY,
        "X-BAPI-TIMESTAMP":   timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN":        signature,
    }
    url = f"{BASE_URL}{endpoint}"
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()


def bybit_public_request(endpoint, params=None):
    """Make unauthenticated GET request to Bybit V5 public endpoints."""
    url = f"{BASE_URL}{endpoint}"
    r = requests.get(url, params=params or {}, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(message: str):
    """Send message to Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        print(f"[TELEGRAM] Sent: {message[:80]}...")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")


# ── Fetch open options positions ───────────────────────────────────────────────
def get_open_options_positions():
    """Fetch all open option positions across watched assets."""
    positions = []
    for asset in ASSETS:
        try:
            data = bybit_signed_request(
                "/v5/position/list",
                {"category": "option", "baseCoin": asset, "limit": "50"}
            )
            if data.get("retCode") == 0:
                for pos in data["result"].get("list", []):
                    if float(pos.get("size", 0)) > 0:
                        positions.append(pos)
        except Exception as e:
            print(f"[POSITION FETCH ERROR] {asset}: {e}")
    return positions


# ── Fetch ticker data for a specific option symbol ────────────────────────────
def get_option_ticker(symbol: str):
    """Fetch live ticker for a specific option contract."""
    try:
        data = bybit_public_request(
            "/v5/market/tickers",
            {"category": "option", "symbol": symbol}
        )
        if data.get("retCode") == 0:
            items = data["result"].get("list", [])
            return items[0] if items else None
    except Exception as e:
        print(f"[TICKER ERROR] {symbol}: {e}")
    return None


# ── Fetch spot price ──────────────────────────────────────────────────────────
def get_spot_price(asset: str) -> float:
    """Get current spot price for an asset."""
    try:
        data = bybit_public_request(
            "/v5/market/tickers",
            {"category": "spot", "symbol": f"{asset}USDT"}
        )
        if data.get("retCode") == 0:
            items = data["result"].get("list", [])
            if items:
                return float(items[0]["lastPrice"])
    except Exception as e:
        print(f"[SPOT PRICE ERROR] {asset}: {e}")
    return 0.0


# ── Parse expiry from symbol ───────────────────────────────────────────────────
def hours_to_expiry(symbol: str) -> float:
    """
    Parse expiry from Bybit option symbol format: BTC-5APR26-67500-C
    Returns hours remaining to 08:00 UTC expiry date.
    """
    try:
        parts = symbol.split("-")
        # parts[1] = e.g. "5APR26"
        date_str = parts[1]
        expiry_dt = datetime.strptime(date_str, "%d%b%y").replace(
            hour=8, minute=0, second=0, tzinfo=timezone.utc
        )
        now = datetime.now(timezone.utc)
        delta = (expiry_dt - now).total_seconds() / 3600
        return max(0.0, delta)
    except Exception:
        return 999.0  # Unknown — don't trigger expiry alerts


# ── Check position and generate alerts ───────────────────────────────────────
def check_position(pos: dict, prev_state: dict) -> list[str]:
    """Evaluate a position and return list of alert messages."""
    alerts = []
    symbol     = pos["symbol"]
    size       = float(pos.get("size", 0))
    entry_px   = float(pos.get("avgPrice", 0))
    unrealised = float(pos.get("unrealisedPnl", 0))
    side       = pos.get("side", "")

    # Derive asset from symbol (e.g. BTC-5APR26-67500-C → BTC)
    asset = symbol.split("-")[0]

    # Live ticker data
    ticker = get_option_ticker(symbol)
    if not ticker:
        return alerts

    mark_price  = float(ticker.get("markPrice", 0))
    bid         = float(ticker.get("bid1Price", 0) or 0)
    ask         = float(ticker.get("ask1Price", 0) or 0)
    delta       = float(ticker.get("delta", 0) or 0)
    iv          = float(ticker.get("markIv", 0) or 0)  # as decimal e.g. 0.65

    # Spot price
    spot_price  = get_spot_price(asset)

    # Parse strike and type from symbol
    parts       = symbol.split("-")
    strike      = float(parts[2]) if len(parts) >= 3 else 0
    opt_type    = parts[3] if len(parts) >= 4 else ""  # C or P

    # Hours to expiry
    hrs_left    = hours_to_expiry(symbol)

    # Intrinsic vs time value
    if opt_type == "C":
        intrinsic = max(0.0, spot_price - strike)
    else:
        intrinsic = max(0.0, strike - spot_price)
    time_value = max(0.0, mark_price - intrinsic)

    # ROI from entry
    roi_pct = ((mark_price - entry_px) / entry_px * 100) if entry_px > 0 else 0

    state_key = symbol

    # ── ALERT 1: ITM Flip ─────────────────────────────────────────────────────
    was_itm = prev_state.get(f"{state_key}_itm", None)
    is_itm  = intrinsic > 0
    if was_itm is not None and was_itm != is_itm:
        direction = "🟢 NOW IN THE MONEY" if is_itm else "🔴 NOW OUT OF THE MONEY"
        alerts.append(
            f"⚡ <b>ITM FLIP — {symbol}</b>\n"
            f"{direction}\n"
            f"Spot: ${spot_price:,.2f} | Strike: ${strike:,.0f}\n"
            f"Intrinsic: ${intrinsic:.2f} | Mark: ${mark_price:.2f}\n"
            f"ROI: {roi_pct:+.1f}%"
        )
    prev_state[f"{state_key}_itm"] = is_itm

    # ── ALERT 2: Theta Danger Zone ────────────────────────────────────────────
    was_theta_danger = prev_state.get(f"{state_key}_theta_danger", False)
    is_theta_danger  = time_value < THETA_DANGER_TIME_VALUE and hrs_left < 12
    if is_theta_danger and not was_theta_danger:
        alerts.append(
            f"⏰ <b>THETA DANGER — {symbol}</b>\n"
            f"Time value bleeding: ${time_value:.2f} remaining\n"
            f"Hours to expiry: {hrs_left:.1f}h\n"
            f"Mark: ${mark_price:.2f} | Intrinsic: ${intrinsic:.2f}\n"
            f"⚠️ Act now or accept loss"
        )
    prev_state[f"{state_key}_theta_danger"] = is_theta_danger

    # ── ALERT 3: Expiry Warning ────────────────────────────────────────────────
    for warn_hours in [6, 2, 1]:
        key = f"{state_key}_expiry_warned_{warn_hours}"
        if hrs_left <= warn_hours and not prev_state.get(key, False):
            status = "ITM ✅" if is_itm else "OTM ❌ (worthless at expiry)"
            alerts.append(
                f"🔔 <b>EXPIRY WARNING — {symbol}</b>\n"
                f"⏳ {hrs_left:.1f} hours remaining!\n"
                f"Status: {status}\n"
                f"Spot: ${spot_price:,.2f} | Strike: ${strike:,.0f}\n"
                f"Mark: ${mark_price:.2f} | ROI: {roi_pct:+.1f}%"
            )
            prev_state[key] = True

    # ── ALERT 4: Stop Loss (mark dropped 50%+ from entry) ─────────────────────
    was_stop = prev_state.get(f"{state_key}_stop_triggered", False)
    is_stop  = entry_px > 0 and mark_price < entry_px * (1 - MARK_PRICE_DROP_PCT)
    if is_stop and not was_stop:
        alerts.append(
            f"🛑 <b>STOP LOSS ALERT — {symbol}</b>\n"
            f"Mark dropped {abs(roi_pct):.0f}% from entry\n"
            f"Entry: ${entry_px:.2f} | Now: ${mark_price:.2f}\n"
            f"Consider closing to protect capital"
        )
    prev_state[f"{state_key}_stop_triggered"] = is_stop

    # ── ALERT 5: Delta ATM crossover ──────────────────────────────────────────
    prev_delta = prev_state.get(f"{state_key}_delta", None)
    if prev_delta is not None:
        crossed_atm = (
            (prev_delta < DELTA_ATM_THRESHOLD and abs(delta) >= DELTA_ATM_THRESHOLD) or
            (prev_delta >= DELTA_ATM_THRESHOLD and abs(delta) < DELTA_ATM_THRESHOLD)
        )
        if crossed_atm:
            direction = "above" if abs(delta) >= DELTA_ATM_THRESHOLD else "below"
            alerts.append(
                f"📊 <b>DELTA ATM CROSS — {symbol}</b>\n"
                f"Delta moved {direction} 0.50 → now {delta:.2f}\n"
                f"Spot: ${spot_price:,.2f} | Strike: ${strike:,.0f}\n"
                f"Mark: ${mark_price:.2f}"
            )
    prev_state[f"{state_key}_delta"] = abs(delta)

    # ── ALERT 6: IV Spike ─────────────────────────────────────────────────────
    prev_iv = prev_state.get(f"{state_key}_iv", None)
    if prev_iv and prev_iv > 0:
        iv_change = (iv - prev_iv) / prev_iv
        if abs(iv_change) >= IV_SPIKE_THRESHOLD:
            direction = "📈 SPIKE" if iv_change > 0 else "📉 CRUSH"
            alerts.append(
                f"🌪️ <b>IV {direction} — {symbol}</b>\n"
                f"IV moved {iv_change*100:+.1f}%: {prev_iv:.1%} → {iv:.1%}\n"
                f"Mark: ${mark_price:.2f}"
            )
    prev_state[f"{state_key}_iv"] = iv

    return alerts


# ── Status heartbeat ──────────────────────────────────────────────────────────
def send_status_report(positions: list):
    """Send a summary of all watched positions."""
    if not positions:
        send_telegram("🤖 <b>QUANTOPS Options Agent</b>\nNo open options positions found across BTC/ETH/SOL/XRP/DOGE.")
        return

    lines = ["🤖 <b>QUANTOPS Options Agent — Position Summary</b>\n"]
    for pos in positions:
        symbol    = pos["symbol"]
        mark_px   = float(pos.get("markPrice", 0) or 0)
        entry_px  = float(pos.get("avgPrice", 0) or 0)
        unreal    = float(pos.get("unrealisedPnl", 0) or 0)
        size      = float(pos.get("size", 0) or 0)
        hrs_left  = hours_to_expiry(symbol)
        roi_pct   = ((mark_px - entry_px) / entry_px * 100) if entry_px > 0 else 0
        emoji     = "🟢" if unreal >= 0 else "🔴"
        lines.append(
            f"{emoji} <b>{symbol}</b>\n"
            f"   Size: {size} | Entry: ${entry_px:.2f} | Mark: ${mark_px:.2f}\n"
            f"   ROI: {roi_pct:+.1f}% | P&L: {unreal:+.2f} USDT\n"
            f"   ⏳ {hrs_left:.1f}h to expiry\n"
        )
    send_telegram("\n".join(lines))


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("[QUANTOPS] Options Alert Agent starting...")
    send_telegram(
        "🚀 <b>QUANTOPS Options Agent ONLINE</b>\n"
        f"Watching: {', '.join(ASSETS)}\n"
        f"Scan interval: every 5 minutes\n"
        f"Alerts: ITM flip | Theta danger | Expiry | Stop loss | Delta ATM | IV spike"
    )

    prev_state    = {}
    scan_count    = 0
    STATUS_EVERY  = 6  # Send full status every 6 scans = ~30 mins

    while True:
        try:
            scan_count += 1
            print(f"[SCAN #{scan_count}] {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

            positions = get_open_options_positions()
            print(f"  Found {len(positions)} open option position(s)")

            # Send status heartbeat periodically
            if scan_count % STATUS_EVERY == 1:
                send_status_report(positions)

            # Check each position for alerts
            for pos in positions:
                alerts = check_position(pos, prev_state)
                for alert in alerts:
                    send_telegram(alert)
                    time.sleep(1)  # Avoid Telegram rate limit

        except Exception as e:
            print(f"[MAIN ERROR] {e}")
            send_telegram(f"⚠️ <b>QUANTOPS Agent Error</b>\n{str(e)[:200]}")

        print(f"  Sleeping 5 minutes...")
        time.sleep(300)  # 5 minute scan interval


if __name__ == "__main__":
    main()
