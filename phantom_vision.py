"""
PHANTOM Vision Agent - QUANTOPS
================================
Ascending triangle detection using Claude Vision API.
Replaces the geometric detection approach with AI-powered chart analysis.

Scans: BTC, ETH, SOL, XRP, DOGE + any symbols passed via env
Timeframes: 15m, 1H, 4H
Triggers Telegram alert when:
  - Ascending triangle confirmed (confidence >= 0.75)
  - Howrie Band is BLUE (red = STAND_ASIDE, no alert)
  - Pattern is confirmed (not partial)

Data source: Bybit V5 Kline API
NOTE: Requires BYBIT_API_KEY and BYBIT_API_SECRET GitHub Secrets.

GitHub Secrets required:
  TELEGRAM_TOKEN
  TELEGRAM_CHAT_ID
  ANTHROPIC_API_KEY
  BYBIT_API_KEY
  BYBIT_API_SECRET
"""

import os
import io
import json
import base64
import logging
import requests
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timezone
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PHANTOM] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("PHANTOM")

# ── Config ───────────────────────────────────────────────────────────────────
BINANCE_BASE     = "https://api.binance.com"
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT    = os.environ["TELEGRAM_CHAT_ID"]
ANTHROPIC_KEY    = os.environ["ANTHROPIC_API_KEY"]
BYBIT_API_KEY    = os.environ.get("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET", "")

SYMBOLS = os.environ.get(
    "PHANTOM_SYMBOLS",
    "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT"
).split(",")

TIMEFRAMES = os.environ.get("PHANTOM_TIMEFRAMES", "15,60,240").split(",")

CONFIDENCE_THRESHOLD = float(os.environ.get("PHANTOM_CONFIDENCE", "0.75"))
CANDLE_LIMIT         = int(os.environ.get("PHANTOM_CANDLES", "80"))

# Howrie Band EMA periods
HB_FAST = int(os.environ.get("HB_FAST", "8"))
HB_SLOW = int(os.environ.get("HB_SLOW", "21"))

# Binance kline interval map — phantom timeframe (minutes) → Binance interval string
# Binance accepts: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
BINANCE_INTERVAL_MAP = {
    "15":  "15m",
    "60":  "1h",
    "240": "4h",
}

# ── Binance Data ──────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, interval: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance Kline API.
    Endpoint: GET /api/v3/klines
    No API key required for market data.
    No IP restrictions from GitHub Actions runners.
    Returns up to 1000 candles per call (Binance max).
    """
    binance_interval = BINANCE_INTERVAL_MAP.get(interval, interval)

    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol":   symbol.upper(),
        "interval": binance_interval,
        "limit":    min(limit, 1000),   # Binance max is 1000
    }

    headers = {
        "Accept":     "application/json",
        "User-Agent": "QUANTOPS-PHANTOM/1.0",
    }

    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()

    raw_list = resp.json()
    if not raw_list:
        raise ValueError(f"Binance returned empty kline list for {symbol}")

    # Binance returns rows:
    # [openTime, open, high, low, close, volume, closeTime,
    #  quoteVolume, trades, takerBuyBase, takerBuyQuote, ignore]
    df = pd.DataFrame(raw_list, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df.astype({
        "timestamp": "int64",
        "open":      "float64",
        "high":      "float64",
        "low":       "float64",
        "close":     "float64",
        "volume":    "float64",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    df.index.name = "Date"

    # Keep only OHLCV
    df = df[["open", "high", "low", "close", "volume"]]

    # Trim to requested limit
    df = df.iloc[-limit:]
    return df

# ── Howrie Band ───────────────────────────────────────────────────────────────

def compute_howrie_band(df: pd.DataFrame) -> tuple:
    """
    Compute Howrie Band using dual EMA crossover.
    Returns (band_series, color, fast_ema, slow_ema).
    Fast EMA above Slow EMA = blue (long/hold).
    Fast EMA below Slow EMA = red (exit immediately).
    """
    fast_ema = df["close"].ewm(span=HB_FAST, adjust=False).mean()
    slow_ema = df["close"].ewm(span=HB_SLOW, adjust=False).mean()

    band = fast_ema

    last_fast = fast_ema.iloc[-1]
    last_slow = slow_ema.iloc[-1]
    color = "blue" if last_fast >= last_slow else "red"

    return band, color, fast_ema, slow_ema

# ── Chart Renderer ────────────────────────────────────────────────────────────

def render_chart_png(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    howrie_band: pd.Series,
    slow_ema: pd.Series,
    band_color: str
) -> bytes:
    """Render candlestick chart with Howrie Band overlay to PNG bytes."""
    hb_color   = "#2196F3" if band_color == "blue" else "#F44336"
    slow_color = "#FF9800"

    apds = [
        mpf.make_addplot(howrie_band, color=hb_color, width=2.5,
                         label=f"Howrie Band ({band_color.upper()})"),
        mpf.make_addplot(slow_ema, color=slow_color, width=1.5,
                         linestyle="--", label=f"EMA{HB_SLOW}"),
    ]

    tf_label = {
        "15": "15m", "60": "1H", "240": "4H",
        "D": "Daily", "W": "Weekly"
    }.get(interval, interval)

    title = f"PHANTOM | {symbol} {tf_label} | Howrie Band: {band_color.upper()}"

    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        gridstyle=":",
        gridcolor="#333333",
        facecolor="#0d0d0d",
        edgecolor="#333333",
        figcolor="#0d0d0d",
        y_on_right=True,
        rc={
            "axes.labelcolor": "#cccccc",
            "xtick.color":     "#cccccc",
            "ytick.color":     "#cccccc",
            "text.color":      "#cccccc",
        }
    )

    buf = io.BytesIO()
    fig, axes = mpf.plot(
        df,
        type="candle",
        style=style,
        addplot=apds,
        title=title,
        volume=True,
        returnfig=True,
        figsize=(14, 8),
        tight_layout=True,
    )

    patch_color = "#2196F3" if band_color == "blue" else "#F44336"
    patch = mpatches.Patch(color=patch_color,
                           label=f"Howrie Band: {band_color.upper()}")
    axes[0].legend(handles=[patch], loc="upper left",
                   facecolor="#1a1a1a", edgecolor="#444444",
                   labelcolor="#ffffff", fontsize=9)

    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ── Claude Vision ─────────────────────────────────────────────────────────────

PHANTOM_SYSTEM = """You are PHANTOM, the chart pattern recognition agent inside the QUANTOPS trading system.
Your role is to detect ascending triangle patterns with high precision.
You are analytical, direct, and pattern-focused.
You understand that the Howrie Band color (blue/red) is the highest-priority signal and overrides all patterns."""

PHANTOM_PROMPT = """Analyze this candlestick chart for an ascending triangle pattern.

An ascending triangle has:
- A FLAT or near-flat horizontal resistance zone (allow minor wick violations)
- A series of HIGHER LOWS forming a rising support trendline
- CONVERGING price action toward an apex
- Ideally DECLINING volume into the apex, then expansion on breakout

Also identify:
- The Howrie Band line color (blue line = bullish/hold, red line = bearish/exit)
- Whether a breakout has already occurred

Respond ONLY with valid JSON, no markdown, no explanation, exactly this structure:

{
  "pattern_detected": true or false,
  "pattern_type": "ascending_triangle" or "partial_ascending_triangle" or "none",
  "confidence": 0.0 to 1.0,
  "flat_resistance_level": "approximate price as string or null",
  "rising_support_slope": "strong" or "moderate" or "weak" or "none",
  "apex_proximity": "close" or "mid" or "early" or "none",
  "volume_behaviour": "declining" or "neutral" or "expanding",
  "howrie_band_color": "blue" or "red" or "unclear",
  "breakout_occurred": true or false,
  "breakout_direction": "up" or "down" or "none",
  "wick_violations": "none" or "minor" or "significant",
  "pattern_quality": "clean" or "moderate" or "messy",
  "phantom_action": "ALERT" or "WATCH" or "STAND_ASIDE" or "INVALID",
  "phantom_note": "brief one-sentence reason"
}

CRITICAL RULES for phantom_action:
- If howrie_band_color is "red" → phantom_action MUST be "STAND_ASIDE"
- If confidence < 0.75 → phantom_action MUST be "WATCH" (unless red band)
- If pattern_type is "ascending_triangle" AND confidence >= 0.75 AND band is "blue" → "ALERT"
- If pattern_type is "partial_ascending_triangle" → "WATCH"
- If pattern_type is "none" → "INVALID"
"""

def claude_vision_analyze(png_bytes: bytes) -> dict:
    """Send chart PNG to Claude Vision API and get PHANTOM analysis."""
    b64 = base64.standard_b64encode(png_bytes).decode("utf-8")

    headers = {
        "x-api-key":         ANTHROPIC_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }

    payload = {
        "model":      "claude-sonnet-4-6",   # Sonnet 4.6
        "max_tokens": 1024,
        "system":     PHANTOM_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/png",
                            "data":       b64,
                        }
                    },
                    {
                        "type": "text",
                        "text": PHANTOM_PROMPT,
                    }
                ]
            }
        ]
    }

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=60
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()

    # Strip any accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)

# ── Alert Logic ───────────────────────────────────────────────────────────────

def should_alert(result: dict) -> tuple:
    """Apply QUANTOPS rules to determine if Telegram alert fires."""
    action = result.get("phantom_action", "INVALID")
    band   = result.get("howrie_band_color", "unclear")
    conf   = result.get("confidence", 0.0)
    note   = result.get("phantom_note", "")

    # Locked QUANTOPS rule: red band overrides everything
    if band == "red":
        return False, f"STAND_ASIDE — Red Howrie Band active. {note}"

    if action == "ALERT":
        return True, note

    if action == "WATCH":
        return False, f"WATCH — confidence {conf:.0%}. {note}"

    return False, f"{action} — {note}"

# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(message: str, png_bytes: Optional[bytes] = None) -> None:
    """Send Telegram message, optionally with chart image."""
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

    if png_bytes:
        files = {"photo": ("chart.png", io.BytesIO(png_bytes), "image/png")}
        data  = {"chat_id": TELEGRAM_CHAT, "caption": message, "parse_mode": "HTML"}
        resp  = requests.post(f"{base_url}/sendPhoto", data=data, files=files, timeout=30)
    else:
        data  = {"chat_id": TELEGRAM_CHAT, "text": message, "parse_mode": "HTML"}
        resp  = requests.post(f"{base_url}/sendMessage", data=data, timeout=30)

    resp.raise_for_status()
    log.info("Telegram sent OK")

def format_alert(symbol: str, interval: str, result: dict) -> str:
    """Format the Telegram alert message."""
    tf_label = {"15": "15m", "60": "1H", "240": "4H"}.get(interval, interval)

    band      = result.get("howrie_band_color", "unclear").upper()
    band_icon = "🔵" if band == "BLUE" else "🔴" if band == "RED" else "⚪"
    conf      = result.get("confidence", 0.0)
    conf_pct  = f"{conf:.0%}"
    quality   = result.get("pattern_quality", "").upper()
    slope     = result.get("rising_support_slope", "").upper()
    apex      = result.get("apex_proximity", "").upper()
    vol       = result.get("volume_behaviour", "").upper()
    resist    = result.get("flat_resistance_level", "N/A")
    wick      = result.get("wick_violations", "none").upper()
    breakout  = result.get("breakout_occurred", False)
    note      = result.get("phantom_note", "")
    ts        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    breakout_line = ""
    if breakout:
        direction = result.get("breakout_direction", "unknown").upper()
        breakout_line = f"\n⚡ <b>BREAKOUT DETECTED:</b> {direction}"

    return (
        f"🔺 <b>PHANTOM — ASCENDING TRIANGLE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>{symbol}</b> | {tf_label}\n"
        f"{band_icon} Howrie Band: <b>{band}</b>\n"
        f"🎯 Confidence: <b>{conf_pct}</b>\n"
        f"✨ Pattern Quality: {quality}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📐 Resistance Zone: <code>{resist}</code>\n"
        f"📈 Support Slope: {slope}\n"
        f"⏳ Apex Proximity: {apex}\n"
        f"📦 Volume: {vol}\n"
        f"🪄 Wick Violations: {wick}"
        f"{breakout_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 <i>{note}</i>\n"
        f"🕐 {ts}"
    )

def format_summary_line(symbol: str, interval: str, result: dict, fired: bool) -> str:
    """One-line scan summary for the run report."""
    tf    = {"15": "15m", "60": "1H", "240": "4H"}.get(interval, interval)
    band  = result.get("howrie_band_color", "?")[0].upper()
    conf  = result.get("confidence", 0.0)
    ptype = result.get("pattern_type", "none")
    icon  = "🔺" if fired else ("👁" if ptype != "none" else "⬜")
    return f"{icon} {symbol:<12} {tf:<4} | Band:{band} Conf:{conf:.0%} | {ptype}"

# ── Main Scanner ──────────────────────────────────────────────────────────────

def scan_symbol_timeframe(symbol: str, interval: str) -> tuple:
    """Full pipeline for one symbol/timeframe. Returns (result, alerted, reason)."""
    log.info(f"Scanning {symbol} {interval}m ...")

    # 1. Fetch OHLCV from Bybit
    df = fetch_ohlcv(symbol, interval)
    if df.empty or len(df) < 30:
        log.warning(f"{symbol} {interval}: insufficient data")
        return {}, False, "insufficient data"

    # 2. Compute Howrie Band
    band_series, band_color, fast_ema, slow_ema = compute_howrie_band(df)

    # 3. Render chart PNG
    png_bytes = render_chart_png(df, symbol, interval, band_series, slow_ema, band_color)

    # 4. Claude Vision analysis
    result = claude_vision_analyze(png_bytes)
    result["symbol"]               = symbol
    result["interval"]             = interval
    result["howrie_band_computed"] = band_color

    log.info(
        f"{symbol} {interval} → pattern={result.get('pattern_type')} "
        f"conf={result.get('confidence')} band={result.get('howrie_band_color')} "
        f"action={result.get('phantom_action')}"
    )

    # 5. Alert decision
    alert, reason = should_alert(result)

    if alert:
        msg = format_alert(symbol, interval, result)
        send_telegram(msg, png_bytes)
        log.info(f"ALERT sent for {symbol} {interval}")
    else:
        log.info(f"No alert: {reason}")

    return result, alert, reason


def main():
    log.info("=" * 60)
    log.info("PHANTOM Vision Scan starting")
    log.info(f"Symbols    : {SYMBOLS}")
    log.info(f"Timeframes : {TIMEFRAMES}")
    log.info(f"Confidence : {CONFIDENCE_THRESHOLD}")
    log.info("=" * 60)

    summary_lines = []
    alerts_fired  = 0
    errors        = 0

    for symbol in SYMBOLS:
        symbol = symbol.strip()
        for interval in TIMEFRAMES:
            interval = interval.strip()
            try:
                result, fired, reason = scan_symbol_timeframe(symbol, interval)
                if result:
                    line = format_summary_line(symbol, interval, result, fired)
                    summary_lines.append(line)
                    if fired:
                        alerts_fired += 1
            except Exception as e:
                log.error(f"ERROR {symbol} {interval}: {e}")
                summary_lines.append(f"❌ {symbol:<12} {interval:<4} | ERROR: {e}")
                errors += 1

    # Send run summary to Telegram
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    summary_body = "\n".join(summary_lines) if summary_lines else "No results."
    summary_msg = (
        f"🤖 <b>PHANTOM — Scan Complete</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{summary_body}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔺 Alerts fired : {alerts_fired}\n"
        f"❌ Errors       : {errors}\n"
        f"🕐 {ts}"
    )
    try:
        send_telegram(summary_msg)
        log.info("Scan complete — summary sent to Telegram.")
    except Exception as e:
        log.error(f"Failed to send Telegram summary: {e}")
        log.error("Check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID secrets are correctly set.")

    if errors == len(SYMBOLS) * len(TIMEFRAMES):
        log.error("All scans failed — exiting with code 1.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
