"""
╔══════════════════════════════════════════════════════════╗
║       S&R BREAKOUT TRADING BOT — by @kapoy111           ║
║  Strategy : Support & Resistance Breakout               ║
║  Assets   : Forex | Gold (XAUUSD) | Crypto              ║
║  Signals  : Telegram                                    ║
║  Timeframe: 1H (configurable)                           ║
╚══════════════════════════════════════════════════════════╝

Usage:
  python bot.py            → run forever (loop every SCAN_INTERVAL seconds)
  python bot.py --once     → scan once then exit  ← used by GitHub Actions
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import ccxt
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════
#  ⚙️  CONFIGURATION  —  edit these to your liking
# ══════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Assets ──────────────────────────────────────────────
FOREX_PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
}

GOLD = {
    "XAUUSD": "GC=F",   # Gold futures (close to spot)
}

CRYPTO_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
]

# ── Strategy parameters ─────────────────────────────────
PIVOT_WINDOW    = 5     # bars each side when detecting swing highs/lows
MIN_TOUCHES     = 2     # minimum times price must touch a level for it to count
ZONE_ATR_MULT   = 0.3   # S&R zone width = this × ATR  (tighter = more precise)
ATR_PERIOD      = 14
RSI_PERIOD      = 14
EMA_PERIOD      = 200   # trend filter — long only above, short only below
RISK_REWARD     = 2.0   # Take-profit = SL distance × this
SCAN_INTERVAL   = 300   # seconds between full scans (300 = 5 min)
SIGNAL_COOLDOWN = 3600  # ignore repeat signals on same pair within this window (s)

# ══════════════════════════════════════════════════════════
#  📋  LOGGING
# ══════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("SR_Bot")

# ══════════════════════════════════════════════════════════
#  🔁  DUPLICATE SIGNAL GUARD
# ══════════════════════════════════════════════════════════

_last_signal: dict[str, float] = {}   # { symbol: unix_timestamp }

def _is_duplicate(symbol: str) -> bool:
    return (time.time() - _last_signal.get(symbol, 0)) < SIGNAL_COOLDOWN

def _mark_sent(symbol: str) -> None:
    _last_signal[symbol] = time.time()

# ══════════════════════════════════════════════════════════
#  📨  TELEGRAM
# ══════════════════════════════════════════════════════════

def send_telegram(msg: str) -> None:
    """Push a message to Telegram. Falls back to console if not configured."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n── SIGNAL (Telegram not configured) ──")
        print(msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
        log.info("✉️  Telegram message sent.")
    except Exception as exc:
        log.error(f"Telegram error: {exc}")

# ══════════════════════════════════════════════════════════
#  📥  DATA FETCHING
# ══════════════════════════════════════════════════════════

def fetch_forex_gold(ticker: str) -> pd.DataFrame:
    """Fetch 7-day hourly OHLCV for a Forex/Gold ticker via yfinance."""
    try:
        raw = yf.download(ticker, period="7d", interval="1h",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        # yfinance may return MultiIndex columns when using auto_adjust
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        df = raw[["open", "high", "low", "close", "volume"]].dropna()
        return df
    except Exception as exc:
        log.error(f"Fetch error [{ticker}]: {exc}")
        return pd.DataFrame()


def fetch_crypto(symbol: str) -> pd.DataFrame:
    """Fetch the last 300 hourly candles for a crypto pair from KuCoin."""
    try:
        exchange = ccxt.kucoin({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=300)
        df = pd.DataFrame(
            ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df.index = pd.to_datetime(df["ts"], unit="ms")
        df.drop(columns="ts", inplace=True)
        return df.dropna()
    except Exception as exc:
        log.error(f"Crypto fetch error [{symbol}]: {exc}")
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════
#  📐  INDICATORS
# ══════════════════════════════════════════════════════════

def calc_atr(df: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def calc_rsi(df: pd.DataFrame, n: int = RSI_PERIOD) -> pd.Series:
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0.0).ewm(span=n, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(span=n, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_ema(df: pd.DataFrame, n: int = EMA_PERIOD) -> pd.Series:
    return df["close"].ewm(span=n, adjust=False).mean()

# ══════════════════════════════════════════════════════════
#  🏔️  S&R LEVEL DETECTION
# ══════════════════════════════════════════════════════════

def find_pivot_highs(df: pd.DataFrame, w: int = PIVOT_WINDOW) -> list[float]:
    """Return prices of all local swing highs in the dataframe."""
    highs  = df["high"].values
    pivots = []
    for i in range(w, len(highs) - w):
        window_max = max(highs[i - w : i + w + 1])
        if highs[i] == window_max:
            pivots.append(float(highs[i]))
    return pivots


def find_pivot_lows(df: pd.DataFrame, w: int = PIVOT_WINDOW) -> list[float]:
    """Return prices of all local swing lows in the dataframe."""
    lows   = df["low"].values
    pivots = []
    for i in range(w, len(lows) - w):
        window_min = min(lows[i - w : i + w + 1])
        if lows[i] == window_min:
            pivots.append(float(lows[i]))
    return pivots


def cluster_levels(
    prices: list[float], threshold: float
) -> list[tuple[float, int]]:
    """
    Group nearby pivot prices into zones.
    Returns list of (mean_price, touch_count) sorted ascending.
    Only zones with touch_count >= MIN_TOUCHES are returned.
    """
    if not prices:
        return []
    arr  = np.array(sorted(prices))
    used = [False] * len(arr)
    zones: list[tuple[float, int]] = []

    for i, p in enumerate(arr):
        if used[i]:
            continue
        grp = [p]
        used[i] = True
        for j in range(i + 1, len(arr)):
            if not used[j] and abs(arr[j] - p) <= threshold:
                grp.append(arr[j])
                used[j] = True
        if len(grp) >= MIN_TOUCHES:
            zones.append((float(np.mean(grp)), len(grp)))

    return sorted(zones, key=lambda x: x[0])


def get_sr_levels(df: pd.DataFrame) -> dict:
    """
    Full pipeline: detect pivot highs/lows → cluster into zones →
    split into resistance (above price) and support (below price).
    """
    current_price = float(df["close"].iloc[-1])
    _atr          = float(calc_atr(df).iloc[-1])
    threshold     = _atr * ZONE_ATR_MULT

    all_res = cluster_levels(find_pivot_highs(df), threshold)
    all_sup = cluster_levels(find_pivot_lows(df),  threshold)

    # Only keep levels on the correct side of current price
    resistance = sorted(
        [(lvl, cnt) for lvl, cnt in all_res if lvl > current_price],
        key=lambda x: x[0],       # nearest first
    )
    support = sorted(
        [(lvl, cnt) for lvl, cnt in all_sup if lvl < current_price],
        key=lambda x: x[0], reverse=True,  # nearest first
    )

    return {
        "resistance": resistance,
        "support":    support,
        "atr":        _atr,
        "current":    current_price,
    }

# ══════════════════════════════════════════════════════════
#  💥  BREAKOUT DETECTION
# ══════════════════════════════════════════════════════════

def detect_breakout(df: pd.DataFrame, sr: dict) -> dict | None:
    """
    Check if the latest candle broke through the nearest S or R level.

    LONG  → previous close was BELOW resistance, current close is ABOVE.
    SHORT → previous close was ABOVE support,    current close is BELOW.

    Confirmation filters applied:
      • RSI not in extreme zone (avoids chasing momentum tops/bottoms)
      • Price on the correct side of 200 EMA (trend alignment)
    """
    c0   = float(df["close"].iloc[-1])   # current (just-closed) candle
    c1   = float(df["close"].iloc[-2])   # previous candle
    _atr = sr["atr"]

    _rsi_val = float(calc_rsi(df).iloc[-1])
    _ema_val = float(calc_ema(df).iloc[-1]) if len(df) >= EMA_PERIOD else None

    # ── LONG signal ─────────────────────────────────────
    if sr["resistance"]:
        lvl, touches = sr["resistance"][0]          # nearest resistance
        broke_above  = (c1 < lvl) and (c0 > lvl)   # candle closed above it

        if broke_above and _rsi_val < 75:           # not overbought
            trend_ok = (_ema_val is None) or (c0 > _ema_val)
            if trend_ok:
                sl = lvl - _atr                     # SL just below broken level
                tp = c0 + (c0 - sl) * RISK_REWARD
                return {
                    "side":    "LONG 📈",
                    "entry":   c0,
                    "sl":      sl,
                    "tp":      tp,
                    "level":   lvl,
                    "touches": touches,
                    "rsi":     _rsi_val,
                    "label":   "Resistance Breakout 🔓",
                }

    # ── SHORT signal ────────────────────────────────────
    if sr["support"]:
        lvl, touches = sr["support"][0]             # nearest support
        broke_below  = (c1 > lvl) and (c0 < lvl)   # candle closed below it

        if broke_below and _rsi_val > 25:           # not oversold
            trend_ok = (_ema_val is None) or (c0 < _ema_val)
            if trend_ok:
                sl = lvl + _atr                     # SL just above broken level
                tp = c0 - (sl - c0) * RISK_REWARD
                return {
                    "side":    "SHORT 📉",
                    "entry":   c0,
                    "sl":      sl,
                    "tp":      tp,
                    "level":   lvl,
                    "touches": touches,
                    "rsi":     _rsi_val,
                    "label":   "Support Breakdown 🔓",
                }

    return None

# ══════════════════════════════════════════════════════════
#  📣  SIGNAL FORMATTER
# ══════════════════════════════════════════════════════════

def _decimals(symbol: str) -> int:
    """Pick decimal places for display based on asset type."""
    if any(k in symbol for k in ["XAU", "BTC", "ETH", "SOL", "BNB"]):
        return 2
    if "JPY" in symbol:
        return 3
    return 5


def format_signal(symbol: str, sig: dict, asset_type: str) -> str:
    d   = _decimals(symbol)
    fmt = f"{{:.{d}f}}"
    ico = "🟢" if "LONG" in sig["side"] else "🔴"

    lines = [
        f"{ico} *S\\&R BREAKOUT — {symbol}* {ico}",
        f"",
        f"🏷 Asset: `{asset_type}`",
        f"📌 Direction: {sig['side']}",
        f"🔓 Broken level: `{fmt.format(sig['level'])}` ({sig['touches']} touches)",
        f"",
        f"💵 Entry:        `{fmt.format(sig['entry'])}`",
        f"🛑 Stop Loss:   `{fmt.format(sig['sl'])}`",
        f"🎯 Take Profit: `{fmt.format(sig['tp'])}`",
        f"",
        f"📊 RSI: `{sig['rsi']:.1f}` | R:R `1:{RISK_REWARD}`",
        f"📋 {sig['label']}",
        f"⏰ `{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}`",
        f"",
        f"⚠️ _Risk max 1-2% per trade. Always use a SL._",
    ]
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════
#  🔍  SCANNER  (one asset at a time)
# ══════════════════════════════════════════════════════════

def scan(symbol: str, df: pd.DataFrame, asset_type: str) -> None:
    if df.empty or len(df) < 60:
        log.warning(f"[{symbol}] Skipped — insufficient data ({len(df)} bars)")
        return

    sr  = get_sr_levels(df)
    sig = detect_breakout(df, sr)

    log.info(
        f"[{symbol:10s}] "
        f"price={sr['current']:.5g}  "
        f"res={len(sr['resistance'])}  "
        f"sup={len(sr['support'])}"
        + (f"  🚨 BREAKOUT!" if sig else "")
    )

    if sig:
        if _is_duplicate(symbol):
            log.info(f"[{symbol}] Duplicate signal — cooldown active, skipped.")
            return
        msg = format_signal(symbol, sig, asset_type)
        send_telegram(msg)
        _mark_sent(symbol)

# ══════════════════════════════════════════════════════════
#  🚀  MAIN LOOP
# ══════════════════════════════════════════════════════════

def run_scan() -> None:
    """Execute one full scan across all assets."""
    log.info(f"═══ SCAN @ {datetime.utcnow().strftime('%H:%M UTC')} ═══")

    # ── Forex ────────────────────────────────────────────
    for name, ticker in FOREX_PAIRS.items():
        scan(name, fetch_forex_gold(ticker), "Forex")
        time.sleep(1)

    # ── Gold ─────────────────────────────────────────────
    for name, ticker in GOLD.items():
        scan(name, fetch_forex_gold(ticker), "Gold")
        time.sleep(1)

    # ── Crypto ───────────────────────────────────────────
    for pair in CRYPTO_PAIRS:
        scan(pair, fetch_crypto(pair), "Crypto")
        time.sleep(2)   # Binance rate-limit safety


def main() -> None:
    parser = argparse.ArgumentParser(description="S&R Breakout Trading Bot")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scan once then exit (used by GitHub Actions cron)",
    )
    args = parser.parse_args()

    log.info("🤖 S&R Breakout Bot starting …")

    if args.once:
        # ── GitHub Actions mode: scan once and exit ──────
        log.info("📋 Mode: --once  (GitHub Actions)")
        send_telegram(
            f"🔍 *Bot scan running …*\n"
            f"⏰ `{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}`"
        )
        run_scan()
        log.info("✅ Scan complete. Exiting.")

    else:
        # ── Local / server mode: loop forever ────────────
        log.info(f"📋 Mode: continuous loop (every {SCAN_INTERVAL // 60} min)")
        send_telegram(
            "🤖 *S\\&R Breakout Bot — LIVE*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📡 Scanning: Forex | Gold | Crypto\n"
            "📐 Strategy: Support \\& Resistance Breakout\n"
            f"⏱ Timeframe: 1H | R:R 1:{RISK_REWARD}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        while True:
            run_scan()
            log.info(f"⏳  Next scan in {SCAN_INTERVAL // 60} min …\n")
            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
