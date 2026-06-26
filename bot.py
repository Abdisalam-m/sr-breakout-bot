"""
╔══════════════════════════════════════════════════════════╗
║       S&R BREAKOUT TRADING BOT — by @kapoy111           ║
║  Strategy : Support & Resistance Breakout               ║
║  Assets   : Forex | Gold (XAUUSD) | Crypto              ║
║  Signals  : Telegram (24/7)                             ║
║  Execution: OANDA (opt) | Binance/KuCoin (opt)          ║
║  Timeframe: Weekly > Daily > 4H > 1H confluence         ║
╚══════════════════════════════════════════════════════════╝

Usage:
  python bot.py          → run forever (loop every SCAN_INTERVAL seconds)
  python bot.py --once   → scan once then exit  ← used by GitHub Actions
"""

import os
import json
import time
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import ccxt
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════
#  ⚙️  CONFIGURATION
# ══════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

OANDA_API_KEY     = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID  = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_DEMO        = os.getenv("OANDA_DEMO", "true").lower() == "true"

BINANCE_API_KEY   = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET    = os.getenv("BINANCE_SECRET", "")

KUCOIN_API_KEY    = os.getenv("KUCOIN_API_KEY", "")
KUCOIN_SECRET     = os.getenv("KUCOIN_SECRET", "")
KUCOIN_PASSPHRASE = os.getenv("KUCOIN_PASSPHRASE", "")

CRYPTO_ORDER_SIZE = float(os.getenv("CRYPTO_ORDER_SIZE", "0.001"))

# ── Assets ──────────────────────────────────────────────
FOREX_PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
}
GOLD         = {"XAUUSD": "GC=F"}
CRYPTO_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

SYMBOL_CURRENCIES = {
    "EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"], "AUDUSD": ["AUD", "USD"],
    "USDCAD": ["USD", "CAD"], "XAUUSD": ["USD"],
}
OANDA_INSTRUMENTS = {
    "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY", "AUDUSD": "AUD_USD",
    "USDCAD": "USD_CAD", "XAUUSD": "XAU_USD",
}
CORRELATED_GROUPS = [
    {"EURUSD", "GBPUSD"},
    {"BTC/USDT", "ETH/USDT", "BNB/USDT"},
]

# Max spread (price units) to allow trading — widens during news/off-hours
MAX_SPREAD = {
    "EURUSD": 0.0004, "GBPUSD": 0.0006, "USDJPY": 0.05,
    "AUDUSD": 0.0005, "USDCAD": 0.0006, "XAUUSD": 0.60,
}

# ── Strategy parameters ─────────────────────────────────
PIVOT_WINDOW     = 5
MIN_TOUCHES_1H   = 3
MIN_TOUCHES_1D   = 2    # daily levels need fewer touches to be valid
ZONE_ATR_MULT    = 0.3
ATR_PERIOD       = 14
RSI_PERIOD       = 14
EMA_1H_PERIOD    = 200
EMA_4H_PERIOD    = 50
RISK_REWARD      = 2.0
RISK_PCT         = 0.01
SCAN_INTERVAL    = 300
SIGNAL_COOLDOWN  = 3600
NEWS_BLACKOUT    = 1800
RETEST_EXPIRY    = 86400   # pending retests expire after 24 h
DAILY            = 23 * 3600  # guard for once-per-day scheduled posts

STATE_FILE = Path("state.json")

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
#  💾  STATE PERSISTENCE
# ══════════════════════════════════════════════════════════

_state: dict = {}


def load_state() -> None:
    global _state
    if STATE_FILE.exists():
        try:
            _state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            log.info(f"📂 State loaded — {len(_state.get('signals', {}))} signals")
        except Exception as exc:
            log.warning(f"State load failed ({exc}) — starting fresh")
            _state = {}
    _state.setdefault("signals",          {})
    _state.setdefault("calendar",         {})
    _state.setdefault("news_sent",        {})
    _state.setdefault("fear_greed",       {})
    _state.setdefault("open_trades",      {})
    _state.setdefault("pending_retests",  {})   # { symbol: [{ level, direction, ts }] }
    _state.setdefault("trade_results",    {"week": "", "wins": 0, "losses": 0})
    _state.setdefault("last_weekly_post", "")
    _state.setdefault("last_update_id",   0)     # Telegram command polling
    _state.setdefault("eod_signals",      [])
    _state.setdefault("eod_date",         "")
    _state.setdefault("paused",           False)  # manual pause via /pause command


def save_state() -> None:
    try:
        STATE_FILE.write_text(json.dumps(_state, indent=2), encoding="utf-8")
    except Exception as exc:
        log.error(f"State save failed: {exc}")


def _is_duplicate(key: str) -> bool:
    return (time.time() - _state["signals"].get(key, 0)) < SIGNAL_COOLDOWN


def _posted_within(key: str, seconds: int) -> bool:
    return (time.time() - _state["signals"].get(key, 0)) < seconds


def _mark_sent(key: str) -> None:
    _state["signals"][key] = time.time()
    save_state()


def _reset_eod_if_new_day() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _state.get("eod_date") != today:
        _state["eod_signals"] = []
        _state["eod_date"]    = today
        save_state()

# ══════════════════════════════════════════════════════════
#  📨  TELEGRAM
# ══════════════════════════════════════════════════════════

def send_telegram(msg: str, chat_id: str | None = None) -> None:
    cid = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_TOKEN or not cid:
        print("\n── SIGNAL ──\n", msg)
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": cid, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
        log.info("✉️  Telegram sent.")
    except Exception as exc:
        log.error(f"Telegram error: {exc}")

# ══════════════════════════════════════════════════════════
#  🤖  TELEGRAM BOT COMMANDS
#  Handled in --commands-only mode (runs every 1 min via separate workflow)
#  Commands: /help /status /signals /trades /price /levels /fear /news
#            /watchlist /week /pause /resume
# ══════════════════════════════════════════════════════════

def process_telegram_commands() -> None:
    if not TELEGRAM_TOKEN:
        return
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"offset": _state["last_update_id"] + 1, "timeout": 0},
            timeout=10,
        )
        r.raise_for_status()
        updates = r.json().get("result", [])
    except Exception as exc:
        log.warning(f"getUpdates failed: {exc}")
        return

    for upd in updates:
        _state["last_update_id"] = upd["update_id"]
        msg  = upd.get("message", {})
        text = msg.get("text", "").strip()
        cid  = str(msg.get("chat", {}).get("id", ""))
        if not text.startswith("/"):
            continue
        parts = text.split()
        cmd   = parts[0].lower().split("@")[0]
        args  = parts[1:]
        log.info(f"📩 {cmd} from {cid}")
        if   cmd == "/help":      _cmd_help(cid)
        elif cmd == "/status":    _cmd_status(cid)
        elif cmd == "/signals":   _cmd_signals(cid)
        elif cmd == "/trades":    _cmd_trades(cid)
        elif cmd == "/price":     _cmd_price(cid, args)
        elif cmd == "/levels":    _cmd_levels(cid, args)
        elif cmd == "/fear":      _cmd_fear(cid)
        elif cmd == "/news":      _cmd_news(cid)
        elif cmd == "/watchlist": _cmd_watchlist_now(cid)
        elif cmd == "/week":      _cmd_week(cid)
        elif cmd == "/pause":     _cmd_pause(cid)
        elif cmd == "/resume":    _cmd_resume(cid)

    save_state()


# ── Command handlers ─────────────────────────────────────

def _cmd_help(cid: str) -> None:
    send_telegram(
        "🤖 *Bot Commands*\n━━━━━━━━━━━━━━━━━━━━\n"
        "/status      — bot health & stats\n"
        "/signals     — today's signals\n"
        "/trades      — open trades\n"
        "/price EURUSD — live price + levels\n"
        "/levels EURUSD — full S\\&R levels\n"
        "/fear        — Crypto Fear \\& Greed\n"
        "/news        — upcoming high-impact news\n"
        "/watchlist   — all key levels now\n"
        "/week        — this week's performance\n"
        "/pause       — pause all signals\n"
        "/resume      — resume signals",
        chat_id=cid,
    )


def _cmd_status(cid: str) -> None:
    sigs         = len(_state.get("eod_signals", []))
    open_count   = len(_state.get("open_trades", {}))
    tr           = _state.get("trade_results", {})
    wins, losses = tr.get("wins", 0), tr.get("losses", 0)
    total        = wins + losses
    winrate      = f"{wins/total*100:.0f}%" if total else "—"
    paused       = "⏸ PAUSED" if _state.get("paused") else "🟢 Active"
    send_telegram(
        f"*Bot Status*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"{paused}\n"
        f"📊 Signals today: `{sigs}`\n"
        f"⏳ Open trades:   `{open_count}`\n"
        f"📈 This week:     `{wins}W / {losses}L` ({winrate})\n"
        f"⏰ `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`",
        chat_id=cid,
    )


def _cmd_signals(cid: str) -> None:
    sigs = _state.get("eod_signals", [])
    if not sigs:
        send_telegram("📭 No signals today yet.", chat_id=cid)
        return
    rows = "\n".join(f"• `{s['symbol']}` {s['side']} @ `{s['entry']}`" for s in sigs)
    send_telegram(f"📋 *Today's Signals*\n━━━━━━━━━━━━━━━━━━━━\n{rows}", chat_id=cid)


def _cmd_trades(cid: str) -> None:
    trades = _state.get("open_trades", {})
    if not trades:
        send_telegram("📭 No open trades.", chat_id=cid)
        return
    lines = ["⏳ *Open Trades*\n━━━━━━━━━━━━━━━━━━━━"]
    for t in trades.values():
        d    = _decimals(t["symbol"])
        tp1h = "✅" if t.get("tp1_hit") else "⏳"
        tp2h = "✅" if t.get("tp2_hit") else "⏳"
        lines.append(
            f"`{t['symbol']}` {t['side']}\n"
            f"  Entry `{t['entry']:.{d}f}` | SL `{t['sl']:.{d}f}`\n"
            f"  TP1 {tp1h} `{t['tp1']:.{d}f}` | TP2 {tp2h} `{t['tp2']:.{d}f}` | TP3 ⏳ `{t['tp3']:.{d}f}`"
        )
    send_telegram("\n".join(lines), chat_id=cid)


def _cmd_price(cid: str, args: list[str]) -> None:
    if not args:
        send_telegram("Usage: `/price EURUSD` or `/price BTC/USDT`", chat_id=cid)
        return
    symbol = args[0].upper()
    # Normalise crypto shorthand (BTC → BTC/USDT)
    if symbol in ("BTC","ETH","SOL","BNB"):
        symbol = symbol + "/USDT"

    # Try cached price first, otherwise fetch live
    price = _prices.get(symbol)
    df    = pd.DataFrame()
    if price is None:
        if symbol in FOREX_PAIRS:
            df = fetch_forex_gold(FOREX_PAIRS[symbol])
        elif symbol in GOLD:
            df = fetch_forex_gold(GOLD[symbol])
        elif symbol in CRYPTO_PAIRS:
            df = fetch_crypto(symbol)
        if not df.empty:
            price = float(df["close"].iloc[-1])

    if price is None:
        send_telegram(f"❌ `{symbol}` not found. Try EURUSD, XAUUSD, BTC/USDT…", chat_id=cid)
        return

    d   = _decimals(symbol)
    sr  = _sr_data.get(symbol)
    res = f"`{sr['resistance'][0][0]:.{d}f}` ({sr['resistance'][0][1]}x)" if sr and sr["resistance"] else "—"
    sup = f"`{sr['support'][0][0]:.{d}f}` ({sr['support'][0][1]}x)"       if sr and sr["support"]    else "—"
    send_telegram(
        f"💵 *{symbol}* — `{price:.{d}f}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔴 Resistance: {res}\n"
        f"🟢 Support:    {sup}\n"
        f"⏰ `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`",
        chat_id=cid,
    )


def _cmd_levels(cid: str, args: list[str]) -> None:
    if not args:
        send_telegram("Usage: `/levels EURUSD`", chat_id=cid)
        return
    symbol = args[0].upper()
    if symbol in ("BTC","ETH","SOL","BNB"):
        symbol = symbol + "/USDT"
    sr = _sr_data.get(symbol)
    if not sr:
        send_telegram(f"❌ No level data for `{symbol}`. Wait for next scan.", chat_id=cid)
        return
    d     = _decimals(symbol)
    lines = [f"📐 *S\\&R Levels — {symbol}*", "━━━━━━━━━━━━━━━━━━━━",
             f"💵 Price: `{sr['current']:.{d}f}`", ""]
    lines.append("*Resistance:*")
    for lvl, cnt in sr["resistance"][:5]:
        lines.append(f"  🔴 `{lvl:.{d}f}` — {cnt} touches")
    lines.append("\n*Support:*")
    for lvl, cnt in sr["support"][:5]:
        lines.append(f"  🟢 `{lvl:.{d}f}` — {cnt} touches")
    send_telegram("\n".join(lines), chat_id=cid)


def _cmd_fear(cid: str) -> None:
    fg = get_fear_greed()
    if not fg:
        send_telegram("❌ Fear & Greed data unavailable.", chat_id=cid)
        return
    v   = fg["value"]
    ico = _fg_icon(v)
    bar = "█" * (v // 10) + "░" * (10 - v // 10)
    send_telegram(
        f"🧠 *Crypto Fear \\& Greed Index*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"{ico} `{v}` — *{fg['label']}*\n"
        f"`{bar}`\n\n"
        f"{'⚠️ Risky to LONG — market overheated' if v >= 75 else '✅ Neutral to positive for LONG' if v >= 45 else '🎯 Fear = opportunity — good for LONG'}\n"
        f"⏰ `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`",
        chat_id=cid,
    )


def _cmd_news(cid: str) -> None:
    events  = _fetch_calendar()
    now     = datetime.now(timezone.utc)
    today   = now.date()
    lines   = ["📰 *Today's High-Impact News*\n━━━━━━━━━━━━━━━━━━━━"]
    found   = False
    for evt in events:
        if evt.get("impact", "").lower() != "high":
            continue
        dt = _parse_event_time(evt.get("date", ""))
        if dt is None or dt.date() != today:
            continue
        diff    = (dt - now).total_seconds()
        status  = "✅ Done" if diff < 0 else f"In {int(diff/60)} min"
        lines.append(
            f"⚡ `{evt.get('country','')}` *{evt.get('title','')}*\n"
            f"   {status} | F: `{evt.get('forecast','—')}` P: `{evt.get('previous','—')}`"
        )
        found = True
    if not found:
        lines.append("No high-impact events today 🟢")
    send_telegram("\n".join(lines), chat_id=cid)


def _cmd_watchlist_now(cid: str) -> None:
    if not _sr_data:
        send_telegram("❌ No data yet. Wait for next scan.", chat_id=cid)
        return
    lines = ["📋 *Current Watchlist*\n━━━━━━━━━━━━━━━━━━━━"]
    for sym, sr in _sr_data.items():
        d = _decimals(sym)
        parts = [f"`{sym}` @ `{sr['current']:.{d}f}`"]
        if sr["resistance"]:
            parts.append(f"🔴`{sr['resistance'][0][0]:.{d}f}`")
        if sr["support"]:
            parts.append(f"🟢`{sr['support'][0][0]:.{d}f}`")
        lines.append("  ".join(parts))
    fg = get_fear_greed()
    if fg:
        lines.append(f"\n🧠 F\\&G: `{fg['value']} — {fg['label']}` {_fg_icon(fg['value'])}")
    send_telegram("\n".join(lines), chat_id=cid)


def _cmd_week(cid: str) -> None:
    tr           = _state.get("trade_results", {})
    wins, losses = tr.get("wins", 0), tr.get("losses", 0)
    total        = wins + losses
    winrate      = f"{wins/total*100:.0f}%" if total else "—"
    open_count   = len(_state.get("open_trades", {}))
    send_telegram(
        f"📈 *This Week*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Wins:   `{wins}`\n"
        f"❌ Losses: `{losses}`\n"
        f"📊 Win rate: `{winrate}`\n"
        f"⏳ Open trades: `{open_count}`",
        chat_id=cid,
    )


def _cmd_pause(cid: str) -> None:
    _state["paused"] = True
    save_state()
    send_telegram("⏸ *Bot paused* — no new signals until /resume", chat_id=cid)


def _cmd_resume(cid: str) -> None:
    _state["paused"] = False
    save_state()
    send_telegram("▶️ *Bot resumed* — scanning for signals", chat_id=cid)

# ══════════════════════════════════════════════════════════
#  📅  ECONOMIC CALENDAR  (ForexFactory — free)
# ══════════════════════════════════════════════════════════

_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def _fetch_calendar() -> list[dict]:
    now, cached = time.time(), _state.get("calendar", {})
    if cached.get("fetched_at", 0) > now - 3600:
        return cached.get("events", [])
    try:
        r = requests.get(_CALENDAR_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        events = r.json()
        _state["calendar"] = {"fetched_at": now, "events": events}
        save_state()
        log.info(f"📅 Calendar refreshed — {len(events)} events")
        return events
    except Exception as exc:
        log.warning(f"Calendar fetch failed: {exc}")
        return cached.get("events", [])


def _parse_event_time(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def is_news_blackout(symbol: str) -> bool:
    currencies = SYMBOL_CURRENCIES.get(symbol, [])
    if not currencies:
        return False
    now = datetime.now(timezone.utc)
    for evt in _fetch_calendar():
        if evt.get("impact", "").lower() != "high":
            continue
        if evt.get("country", "").upper() not in currencies:
            continue
        dt = _parse_event_time(evt.get("date", ""))
        if dt and abs((dt - now).total_seconds()) <= NEWS_BLACKOUT:
            return True
    return False


def post_upcoming_news() -> None:
    now, alerts = datetime.now(timezone.utc), []
    for evt in _fetch_calendar():
        if evt.get("impact", "").lower() != "high":
            continue
        dt = _parse_event_time(evt.get("date", ""))
        if dt is None:
            continue
        diff = (dt - now).total_seconds()
        if not (900 < diff <= 3600):
            continue
        key = f"{evt.get('country')}_{evt.get('title')}_{evt.get('date')}"
        if _state["news_sent"].get(key, 0) > time.time() - 7200:
            continue
        alerts.append((diff, evt, key))
    if not alerts:
        return
    alerts.sort(key=lambda x: x[0])
    lines = ["📰 *UPCOMING HIGH-IMPACT NEWS*", "━━━━━━━━━━━━━━━━━━━━", ""]
    for diff, evt, key in alerts[:5]:
        lines.append(
            f"⚡ `{evt.get('country', '')}` — *{evt.get('title', '')}*\n"
            f"   ⏰ In {int(diff/60)} min | "
            f"Forecast: `{evt.get('forecast','—')}` | Prev: `{evt.get('previous','—')}`"
        )
        _state["news_sent"][key] = time.time()
    lines += ["", "⚠️ _Bot pauses signals 30 min before each event_"]
    send_telegram("\n".join(lines))
    save_state()

# ══════════════════════════════════════════════════════════
#  😱  CRYPTO FEAR & GREED  (alternative.me — free)
# ══════════════════════════════════════════════════════════

def get_fear_greed() -> dict | None:
    now, cached = time.time(), _state.get("fear_greed", {})
    if cached.get("fetched_at", 0) > now - 3600:
        return cached.get("data")
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        d    = r.json()["data"][0]
        data = {"value": int(d["value"]), "label": d["value_classification"]}
        _state["fear_greed"] = {"fetched_at": now, "data": data}
        save_state()
        return data
    except Exception as exc:
        log.warning(f"Fear & Greed fetch failed: {exc}")
        return cached.get("data")


def _fg_icon(v: int) -> str:
    return "😱" if v >= 75 else "🤑" if v >= 55 else "😐" if v >= 45 else "😰" if v >= 25 else "😨"

# ══════════════════════════════════════════════════════════
#  📥  DATA FETCHING
# ══════════════════════════════════════════════════════════

def fetch_forex_gold(ticker: str) -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period="30d", interval="1h",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]
        return raw[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as exc:
        log.error(f"Fetch error [{ticker}]: {exc}")
        return pd.DataFrame()


def fetch_forex_gold_daily(ticker: str) -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period="90d", interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]
        return raw[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as exc:
        log.error(f"Fetch daily error [{ticker}]: {exc}")
        return pd.DataFrame()


def fetch_crypto(symbol: str) -> pd.DataFrame:
    try:
        ex    = ccxt.kucoin({"enableRateLimit": True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1h", limit=720)
        df    = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.drop(columns="ts").dropna()
    except Exception as exc:
        log.error(f"Crypto fetch error [{symbol}]: {exc}")
        return pd.DataFrame()


def fetch_crypto_daily(symbol: str) -> pd.DataFrame:
    try:
        ex    = ccxt.kucoin({"enableRateLimit": True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1d", limit=90)
        df    = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.drop(columns="ts").dropna()
    except Exception as exc:
        log.error(f"Crypto daily fetch error [{symbol}]: {exc}")
        return pd.DataFrame()


def resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    try:
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            return pd.DataFrame()
        return df_1h.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna()
    except Exception:
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════
#  📐  INDICATORS
# ══════════════════════════════════════════════════════════

def calc_atr(df: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hpc, lpc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()


def calc_rsi(df: pd.DataFrame, n: int = RSI_PERIOD) -> pd.Series:
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0.0).ewm(span=n, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(span=n, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_ema(df: pd.DataFrame, n: int = EMA_1H_PERIOD) -> pd.Series:
    return df["close"].ewm(span=n, adjust=False).mean()


def calc_macd_hist(df: pd.DataFrame) -> float:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    line  = ema12 - ema26
    return float((line - line.ewm(span=9, adjust=False).mean()).iloc[-1])

# ══════════════════════════════════════════════════════════
#  📊  MARKET STRUCTURE  (Higher Highs / Lower Lows)
# ══════════════════════════════════════════════════════════

def detect_market_structure(df: pd.DataFrame) -> str:
    """Returns 'bullish', 'bearish', or 'ranging' based on last 2 swing H/L."""
    w = PIVOT_WINDOW
    h, lo = df["high"].values, df["low"].values
    highs = [float(h[i]) for i in range(w, len(h) - w)
              if h[i] == max(h[i - w : i + w + 1])]
    lows  = [float(lo[i]) for i in range(w, len(lo) - w)
              if lo[i] == min(lo[i - w : i + w + 1])]
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1] > highs[-2]
        hl = lows[-1]  > lows[-2]
        lh = highs[-1] < highs[-2]
        ll = lows[-1]  < lows[-2]
        if hh and hl: return "bullish"
        if lh and ll: return "bearish"
    return "ranging"

# ══════════════════════════════════════════════════════════
#  📉  RSI DIVERGENCE
# ══════════════════════════════════════════════════════════

def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> str | None:
    """
    Returns 'bullish' or 'bearish' if RSI diverges from price over last N bars.
    Minimum 5-point RSI gap required to avoid noise.
    """
    if len(df) < lookback + 5:
        return None
    rsi    = calc_rsi(df)
    c_now  = float(df["close"].iloc[-1])
    c_prev = float(df["close"].iloc[-1 - lookback])
    r_now  = float(rsi.iloc[-1])
    r_prev = float(rsi.iloc[-1 - lookback])
    gap    = abs(r_now - r_prev)
    if gap < 5:
        return None
    if c_now > c_prev and r_now < r_prev:   # price higher, RSI lower → bearish div
        return "bearish"
    if c_now < c_prev and r_now > r_prev:   # price lower, RSI higher → bullish div
        return "bullish"
    return None

# ══════════════════════════════════════════════════════════
#  🏔️  S&R DETECTION
# ══════════════════════════════════════════════════════════

def _pivot_highs(df: pd.DataFrame, w: int = PIVOT_WINDOW) -> list[float]:
    h = df["high"].values
    return [float(h[i]) for i in range(w, len(h) - w)
            if h[i] == max(h[i - w : i + w + 1])]


def _pivot_lows(df: pd.DataFrame, w: int = PIVOT_WINDOW) -> list[float]:
    lo = df["low"].values
    return [float(lo[i]) for i in range(w, len(lo) - w)
            if lo[i] == min(lo[i - w : i + w + 1])]


def _cluster(prices: list[float], threshold: float,
             min_touches: int) -> list[tuple[float, int]]:
    if not prices:
        return []
    arr  = np.array(sorted(prices))
    used = [False] * len(arr)
    zones: list[tuple[float, int]] = []
    for i, p in enumerate(arr):
        if used[i]:
            continue
        grp, used[i] = [p], True
        for j in range(i + 1, len(arr)):
            if not used[j] and abs(arr[j] - p) <= threshold:
                grp.append(arr[j])
                used[j] = True
        if len(grp) >= min_touches:
            zones.append((float(np.mean(grp)), len(grp)))
    return sorted(zones, key=lambda x: x[0])


def get_sr_levels(df: pd.DataFrame, min_touches: int = None) -> dict:
    mt      = min_touches or MIN_TOUCHES_1H
    current = float(df["close"].iloc[-1])
    atr     = float(calc_atr(df).iloc[-1])
    thresh  = atr * ZONE_ATR_MULT
    res = [(l, c) for l, c in _cluster(_pivot_highs(df), thresh, mt) if l > current]
    sup = [(l, c) for l, c in _cluster(_pivot_lows(df),  thresh, mt) if l < current]
    return {
        "resistance": sorted(res, key=lambda x: x[0]),
        "support":    sorted(sup, key=lambda x: x[0], reverse=True),
        "atr": atr, "current": current,
    }


def get_daily_levels(df_daily: pd.DataFrame) -> list[float]:
    """Return daily S&R level prices (min 2 touches). Used for confluence check."""
    if df_daily.empty or len(df_daily) < 10:
        return []
    try:
        atr    = float(calc_atr(df_daily).iloc[-1])
        thresh = atr * 0.5
        levels = []
        for lvl, _ in _cluster(_pivot_highs(df_daily), thresh, MIN_TOUCHES_1D):
            levels.append(lvl)
        for lvl, _ in _cluster(_pivot_lows(df_daily), thresh, MIN_TOUCHES_1D):
            levels.append(lvl)
        return levels
    except Exception:
        return []


def is_daily_confluence(level: float, daily_levels: list[float], atr: float) -> bool:
    """True if a 1H level sits within 0.5 ATR of a daily S&R level."""
    return any(abs(level - dl) <= atr * 0.5 for dl in daily_levels)

# ══════════════════════════════════════════════════════════
#  🕯️  CANDLESTICK CONFIRMATION
# ══════════════════════════════════════════════════════════

def is_strong_candle(df: pd.DataFrame, direction: str) -> bool:
    c    = df.iloc[-1]
    body = abs(float(c["close"]) - float(c["open"]))
    rng  = float(c["high"]) - float(c["low"])
    if rng == 0:
        return False
    if (body / rng) < 0.6:
        return False
    return (float(c["close"]) > float(c["open"])) if direction == "long" \
           else (float(c["close"]) < float(c["open"]))

# ══════════════════════════════════════════════════════════
#  ⭐  SIGNAL GRADING & VOLUME
# ══════════════════════════════════════════════════════════

def grade_signal(touches: int, multi_tf: bool, divergence: bool,
                 daily_conf: bool) -> str:
    if touches >= 4 or (touches >= 3 and daily_conf):
        base = "⭐⭐⭐ A+"
    elif touches >= 3 or daily_conf:
        base = "⭐⭐ B"
    else:
        base = "⭐ C"
    extras = []
    if multi_tf:   extras.append("4H✅")
    if divergence: extras.append("Div✅")
    if daily_conf: extras.append("Daily✅")
    return base + (" " + " ".join(extras) if extras else "")


def volume_confirmed(df: pd.DataFrame) -> bool:
    if "volume" not in df.columns or df["volume"].iloc[-20:].sum() == 0:
        return False
    return float(df["volume"].iloc[-1]) >= df["volume"].iloc[-20:].mean() * 1.5

# ══════════════════════════════════════════════════════════
#  🕐  SESSION & FILTERS
# ══════════════════════════════════════════════════════════

def is_active_session(asset_type: str) -> bool:
    if asset_type == "Crypto":
        return True
    h = datetime.now(timezone.utc).hour
    return (8 <= h < 17) or (13 <= h < 22)


def is_correlated_active(symbol: str) -> bool:
    for group in CORRELATED_GROUPS:
        if symbol not in group:
            continue
        for other in group:
            if other != symbol and _is_duplicate(other):
                log.info(f"[{symbol}] Correlated pair {other} active — skipped")
                return True
    return False


def is_spread_ok(symbol: str) -> bool:
    """Check current OANDA spread. Only runs when OANDA keys are set."""
    if not OANDA_API_KEY or symbol not in OANDA_INSTRUMENTS:
        return True
    max_spread = MAX_SPREAD.get(symbol)
    if max_spread is None:
        return True
    try:
        instrument = OANDA_INSTRUMENTS[symbol]
        r = requests.get(
            f"{_oanda_base()}/v3/accounts/{OANDA_ACCOUNT_ID}/pricing",
            headers={"Authorization": f"Bearer {OANDA_API_KEY}"},
            params={"instruments": instrument},
            timeout=10,
        )
        r.raise_for_status()
        p      = r.json().get("prices", [{}])[0]
        bid    = float(p.get("bids", [{}])[0].get("price", 0))
        ask    = float(p.get("asks", [{}])[0].get("price", 0))
        spread = ask - bid
        if spread > max_spread:
            log.info(f"[{symbol}] Spread too wide: {spread:.5f} > {max_spread}")
            return False
        return True
    except Exception:
        return True   # can't check → allow

# ══════════════════════════════════════════════════════════
#  📅  PREVIOUS DAY HIGH / LOW BREAKOUT
# ══════════════════════════════════════════════════════════

def detect_prev_day_breakout(df: pd.DataFrame) -> dict | None:
    try:
        idx      = df.index
        date_arr = (idx.tz_convert("UTC").date if idx.tz is not None else idx.date)
        today    = date_arr[-1]
        prev_data = None
        for d in range(1, 5):
            mask = date_arr == (today - timedelta(days=d))
            if mask.any():
                prev_data = df[mask]
                break
        if prev_data is None or prev_data.empty:
            return None
        prev_high = float(prev_data["high"].max())
        prev_low  = float(prev_data["low"].min())
        c0, c1    = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
        rsi       = float(calc_rsi(df).iloc[-1])
        atr       = float(calc_atr(df).iloc[-1])
        if c1 < prev_high < c0 and rsi < 75:
            sl = prev_high - atr
            return {"side": "LONG 📈", "entry": c0, "sl": sl,
                    "tp": c0 + (c0 - sl) * RISK_REWARD,
                    "level": prev_high, "touches": None, "rsi": rsi,
                    "label": "Previous Day High Breakout 📅", "grade": "⭐⭐ B",
                    "vol_ok": volume_confirmed(df), "multi_tf": False}
        if c1 > prev_low > c0 and rsi > 25:
            sl = prev_low + atr
            return {"side": "SHORT 📉", "entry": c0, "sl": sl,
                    "tp": c0 - (sl - c0) * RISK_REWARD,
                    "level": prev_low, "touches": None, "rsi": rsi,
                    "label": "Previous Day Low Breakdown 📅", "grade": "⭐⭐ B",
                    "vol_ok": volume_confirmed(df), "multi_tf": False}
        return None
    except Exception as exc:
        log.warning(f"Prev day breakout: {exc}")
        return None

# ══════════════════════════════════════════════════════════
#  💥  BREAKOUT DETECTION
# ══════════════════════════════════════════════════════════

def detect_breakout(df_1h: pd.DataFrame, sr: dict,
                    daily_levels: list[float]) -> dict | None:
    c0, c1    = float(df_1h["close"].iloc[-1]), float(df_1h["close"].iloc[-2])
    atr       = sr["atr"]
    rsi       = float(calc_rsi(df_1h).iloc[-1])
    ema_1h    = float(calc_ema(df_1h, EMA_1H_PERIOD).iloc[-1]) if len(df_1h) >= EMA_1H_PERIOD else None
    macd_hist = calc_macd_hist(df_1h)
    structure = detect_market_structure(df_1h)

    df_4h    = resample_4h(df_1h)
    trend_4h = None
    if len(df_4h) >= EMA_4H_PERIOD:
        ema_4h   = float(calc_ema(df_4h, EMA_4H_PERIOD).iloc[-1])
        trend_4h = "bull" if float(df_4h["close"].iloc[-1]) > ema_4h else "bear"

    multi_tf  = trend_4h is not None
    divergence = detect_rsi_divergence(df_1h)

    # ── LONG ──────────────────────────────────────────────
    if sr["resistance"]:
        lvl, touches = sr["resistance"][0]
        if (c1 < lvl < c0) and rsi < 75 and macd_hist > 0:
            ok_1h      = ema_1h is None or c0 > ema_1h
            ok_4h      = trend_4h in (None, "bull")
            ok_struct  = structure in ("bullish", "ranging")
            ok_candle  = is_strong_candle(df_1h, "long")
            div_match  = divergence == "bullish"
            daily_conf = is_daily_confluence(lvl, daily_levels, atr)
            if ok_1h and ok_4h and ok_struct and ok_candle:
                sl = lvl - atr
                tp = c0 + (c0 - sl) * RISK_REWARD
                return {
                    "side": "LONG 📈", "entry": c0, "sl": sl, "tp": tp,
                    "level": lvl, "touches": touches, "rsi": rsi,
                    "label": "Resistance Breakout 🔓",
                    "grade": grade_signal(touches, multi_tf, div_match, daily_conf),
                    "vol_ok": volume_confirmed(df_1h), "multi_tf": multi_tf,
                    "divergence": div_match, "daily_conf": daily_conf,
                    "structure": structure,
                }

    # ── SHORT ─────────────────────────────────────────────
    if sr["support"]:
        lvl, touches = sr["support"][0]
        if (c0 < lvl < c1) and rsi > 25 and macd_hist < 0:
            ok_1h      = ema_1h is None or c0 < ema_1h
            ok_4h      = trend_4h in (None, "bear")
            ok_struct  = structure in ("bearish", "ranging")
            ok_candle  = is_strong_candle(df_1h, "short")
            div_match  = divergence == "bearish"
            daily_conf = is_daily_confluence(lvl, daily_levels, atr)
            if ok_1h and ok_4h and ok_struct and ok_candle:
                sl = lvl + atr
                tp = c0 - (sl - c0) * RISK_REWARD
                return {
                    "side": "SHORT 📉", "entry": c0, "sl": sl, "tp": tp,
                    "level": lvl, "touches": touches, "rsi": rsi,
                    "label": "Support Breakdown 🔓",
                    "grade": grade_signal(touches, multi_tf, div_match, daily_conf),
                    "vol_ok": volume_confirmed(df_1h), "multi_tf": multi_tf,
                    "divergence": div_match, "daily_conf": daily_conf,
                    "structure": structure,
                }

    return None

# ══════════════════════════════════════════════════════════
#  🔄  BREAKOUT RETEST
# ══════════════════════════════════════════════════════════

def register_pending_retest(symbol: str, level: float, direction: str) -> None:
    """After a breakout, watch for price to retest the broken level."""
    pr = _state.setdefault("pending_retests", {})
    pr.setdefault(symbol, [])
    # Expire old retests
    pr[symbol] = [r for r in pr[symbol]
                  if time.time() - r.get("ts", 0) < RETEST_EXPIRY]
    pr[symbol].append({"level": level, "direction": direction, "ts": time.time()})
    save_state()


def check_retest_setups(symbol: str, df: pd.DataFrame) -> dict | None:
    """Return a retest signal if price pulls back to a broken level and bounces."""
    retests = _state.get("pending_retests", {}).get(symbol, [])
    if not retests:
        return None
    c0, c1  = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
    atr     = float(calc_atr(df).iloc[-1])
    rsi     = float(calc_rsi(df).iloc[-1])
    keep    = []
    result  = None

    for rt in retests:
        lvl       = rt["level"]
        direction = rt["direction"]
        near      = abs(c1 - lvl) <= atr * 0.5

        if direction == "long" and near and c0 > c1 and c0 > lvl and rsi < 70:
            sl = lvl - atr
            result = {
                "side": "LONG 📈", "entry": c0, "sl": sl,
                "tp": c0 + (c0 - sl) * RISK_REWARD,
                "level": lvl, "touches": None, "rsi": rsi,
                "label": "Resistance → Support Retest 🔄",
                "grade": "⭐⭐⭐ A+ Retest",
                "vol_ok": volume_confirmed(df), "multi_tf": False,
            }
        elif direction == "short" and near and c0 < c1 and c0 < lvl and rsi > 30:
            sl = lvl + atr
            result = {
                "side": "SHORT 📉", "entry": c0, "sl": sl,
                "tp": c0 - (sl - c0) * RISK_REWARD,
                "level": lvl, "touches": None, "rsi": rsi,
                "label": "Support → Resistance Retest 🔄",
                "grade": "⭐⭐⭐ A+ Retest",
                "vol_ok": volume_confirmed(df), "multi_tf": False,
            }
        else:
            keep.append(rt)

    _state["pending_retests"][symbol] = keep
    save_state()
    return result

# ══════════════════════════════════════════════════════════
#  🏦  BROKER EXECUTION
# ══════════════════════════════════════════════════════════

def _oanda_base() -> str:
    return f"https://{'api-fxpractice' if OANDA_DEMO else 'api-fxtrade'}.oanda.com"


def oanda_execute(symbol: str, sig: dict) -> str | None:
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        return None
    instrument = OANDA_INSTRUMENTS.get(symbol)
    if not instrument:
        return None
    hdrs = {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}
    base = _oanda_base()
    try:
        r = requests.get(f"{base}/v3/accounts/{OANDA_ACCOUNT_ID}", headers=hdrs, timeout=10)
        r.raise_for_status()
        balance = float(r.json()["account"]["balance"])
    except Exception as exc:
        log.error(f"OANDA balance: {exc}")
        return None
    sl_dist = abs(sig["entry"] - sig["sl"])
    if sl_dist == 0:
        return None
    units = int((balance * RISK_PCT) / sl_dist)
    if "SHORT" in sig["side"]:
        units = -units
    if units == 0:
        return None
    payload = {"order": {
        "type": "MARKET", "instrument": instrument, "units": str(units),
        "timeInForce": "FOK",
        "stopLossOnFill":   {"price": f"{sig['sl']:.5f}"},
       "takeProfitOnFill": {"price": f"{sig.get('tp2') or sig.get('tp1', sig['sl']):.5f}"},
   # use TP2 (2R) for broker order
    }}
    try:
        r = requests.post(f"{base}/v3/accounts/{OANDA_ACCOUNT_ID}/orders",
                          headers=hdrs, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        oid  = (data.get("orderFillTransaction") or
                data.get("orderCreateTransaction") or {}).get("id")
        log.info(f"✅ OANDA: {oid} | {units} {instrument}")
        return oid
    except Exception as exc:
        log.error(f"OANDA order failed: {exc}")
        return None


def crypto_execute(symbol: str, sig: dict) -> str | None:
    try:
        if BINANCE_API_KEY and BINANCE_SECRET:
            ex = ccxt.binance({"apiKey": BINANCE_API_KEY, "secret": BINANCE_SECRET,
                               "enableRateLimit": True})
        elif KUCOIN_API_KEY and KUCOIN_SECRET:
            ex = ccxt.kucoin({"apiKey": KUCOIN_API_KEY, "secret": KUCOIN_SECRET,
                              "password": KUCOIN_PASSPHRASE, "enableRateLimit": True})
        else:
            return None
        side  = "buy" if "LONG" in sig["side"] else "sell"
        order = ex.create_market_order(symbol, side, CRYPTO_ORDER_SIZE)
        oid   = str(order.get("id", ""))
        log.info(f"✅ Crypto: {oid} {side} {symbol}")
        return oid
    except Exception as exc:
        log.error(f"Crypto order failed [{symbol}]: {exc}")
        return None


def execute_trade(symbol: str, sig: dict, asset_type: str) -> str | None:
    return crypto_execute(symbol, sig) if asset_type == "Crypto" else oanda_execute(symbol, sig)

# ══════════════════════════════════════════════════════════
#  📊  OPEN TRADE TRACKER  (multi-TP with BE move)
# ══════════════════════════════════════════════════════════

def register_open_trade(symbol: str, sig: dict, asset_type: str) -> None:
    sl_dist  = abs(sig["entry"] - sig["sl"])
    is_long  = "LONG" in sig["side"]
    tp1 = sig["entry"] + sl_dist       if is_long else sig["entry"] - sl_dist
    tp2 = sig["entry"] + sl_dist * 2   if is_long else sig["entry"] - sl_dist * 2
    tp3 = sig["entry"] + sl_dist * 3   if is_long else sig["entry"] - sl_dist * 3

    _state["open_trades"][f"{symbol.replace('/','_')}_{int(time.time())}"] = {
        "symbol": symbol, "side": sig["side"], "entry": sig["entry"],
        "sl": sig["sl"], "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "tp1_hit": False, "tp2_hit": False, "asset_type": asset_type,
        "opened_at": time.time(),
    }
    save_state()


def _update_trade_result(result: str) -> None:
    week = datetime.now(timezone.utc).strftime("%Y-W%W")
    tr   = _state.get("trade_results", {"week": "", "wins": 0, "losses": 0})
    if tr.get("week") != week:
        tr = {"week": week, "wins": 0, "losses": 0}
    tr["wins" if result == "win" else "losses"] += 1
    _state["trade_results"] = tr
    save_state()


def track_open_trades() -> None:
    open_trades = _state.get("open_trades", {})
    if not open_trades:
        return
    closed = []

    for tid, t in open_trades.items():
        sym   = t["symbol"]
        price = _prices.get(sym)
        if price is None:
            continue
        d       = _decimals(sym)
        is_long = "LONG" in t["side"]

        # TP1 hit → partial profit, move SL to breakeven
        if not t.get("tp1_hit"):
            hit = price >= t["tp1"] if is_long else price <= t["tp1"]
            if hit:
                t["tp1_hit"] = True
                t["sl"]      = t["entry"]   # move SL to BE
                send_telegram(
                    f"🟡 *TP1 HIT — {sym}* 🎯\n\n"
                    f"📌 {t['side']} | Entry `{t['entry']:.{d}f}`\n"
                    f"✅ TP1 `{t['tp1']:.{d}f}` hit — `+1R`\n"
                    f"🛡 SL moved to breakeven\n"
                    f"🎯 TP2 `{t['tp2']:.{d}f}` | TP3 `{t['tp3']:.{d}f}` still running\n"
                    f"⏰ `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`"
                )
                _state["open_trades"][tid] = t
                save_state()
                continue

        # TP2 hit
        if t.get("tp1_hit") and not t.get("tp2_hit"):
            hit = price >= t["tp2"] if is_long else price <= t["tp2"]
            if hit:
                t["tp2_hit"] = True
                send_telegram(
                    f"🟢 *TP2 HIT — {sym}* 🎯\n\n"
                    f"📌 {t['side']} | Entry `{t['entry']:.{d}f}`\n"
                    f"✅ TP2 `{t['tp2']:.{d}f}` hit — `+2R`\n"
                    f"🎯 TP3 `{t['tp3']:.{d}f}` still running — max profit 📈\n"
                    f"⏰ `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`"
                )
                _state["open_trades"][tid] = t
                save_state()
                continue

        # TP3 hit → full close
        if t.get("tp2_hit"):
            hit = price >= t["tp3"] if is_long else price <= t["tp3"]
            if hit:
                send_telegram(
                    f"✅ *TP3 HIT — {sym}* 🏆\n\n"
                    f"📌 {t['side']} | Full `+3R` profit\n"
                    f"💰 All targets hit — trade closed\n"
                    f"⏰ `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`"
                )
                _update_trade_result("win")
                closed.append(tid)
                continue

        # SL hit
        sl_hit = price <= t["sl"] if is_long else price >= t["sl"]
        if sl_hit:
            result = "BE" if t.get("tp1_hit") else "-1R loss"
            emoji  = "🟡" if t.get("tp1_hit") else "❌"
            send_telegram(
                f"{emoji} *SL HIT — {sym}*\n\n"
                f"📌 {t['side']} | Entry `{t['entry']:.{d}f}`\n"
                f"🛑 SL `{t['sl']:.{d}f}` hit\n"
                f"💸 Result: `{result}`\n"
                f"⏰ `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`"
            )
            if not t.get("tp1_hit"):
                _update_trade_result("loss")
            closed.append(tid)

    for tid in closed:
        del open_trades[tid]
    if closed:
        _state["open_trades"] = open_trades
        save_state()

# ══════════════════════════════════════════════════════════
#  📣  SIGNAL FORMATTER
# ══════════════════════════════════════════════════════════

def _decimals(symbol: str) -> int:
    if any(k in symbol for k in ["XAU", "BTC", "ETH", "SOL", "BNB"]):
        return 2
    return 3 if "JPY" in symbol else 5


def format_signal(symbol: str, sig: dict, asset_type: str,
                  order_id: str | None) -> str:
    d         = _decimals(symbol)
    fmt       = f"{{:.{d}f}}"
    ico       = "🟢" if "LONG" in sig["side"] else "🔴"
    vol       = "✅ High volume" if sig.get("vol_ok") else "⚠️ Normal volume"
    sl_dist   = abs(sig["entry"] - sig["sl"])
    is_long   = "LONG" in sig["side"]
    tp1       = sig["entry"] + sl_dist     if is_long else sig["entry"] - sl_dist
    tp2       = sig["entry"] + sl_dist * 2 if is_long else sig["entry"] - sl_dist * 2
    tp3       = sig["entry"] + sl_dist * 3 if is_long else sig["entry"] - sl_dist * 3
    touches   = f" ({sig['touches']} touches)" if sig.get("touches") else ""
    exec_note = (f"\n💼 Order: `{order_id}` ✅" if order_id
                 else "\n💡 _Set broker keys for auto-execution_")

    extras = []
    if sig.get("multi_tf"):   extras.append("4H+1H ✅")
    if sig.get("divergence"): extras.append("RSI Div 🚨")
    if sig.get("daily_conf"): extras.append("Daily Level 📆")
    extras_str = ("  |  " + "  |  ".join(extras)) if extras else ""

    return "\n".join([
        f"{ico} *S\\&R BREAKOUT — {symbol}* {ico}",
        "",
        f"🏆 Grade: `{sig['grade']}`",
        f"🏷 Asset: `{asset_type}` | Structure: `{sig.get('structure','—')}`",
        f"📌 Direction: {sig['side']}",
        f"🔓 Level: `{fmt.format(sig['level'])}`{touches}",
        "",
        f"💵 Entry:        `{fmt.format(sig['entry'])}`",
        f"🛑 Stop Loss:   `{fmt.format(sig['sl'])}`",
        f"🎯 TP1 (1R):   `{fmt.format(tp1)}`",
        f"🎯 TP2 (2R):   `{fmt.format(tp2)}`",
        f"🎯 TP3 (3R):   `{fmt.format(tp3)}`",
        "",
        f"📊 RSI: `{sig['rsi']:.1f}` | R:R `1:{RISK_REWARD}`{extras_str}",
        f"📉 Volume: {vol}" + exec_note,
        f"📋 {sig['label']}",
        f"⏰ `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`",
        "",
        "⚠️ _TP1 moves SL to breakeven. Risk max 1-2%._",
    ])

# ══════════════════════════════════════════════════════════
#  🌐  FREE MARKET DATA  (CoinGecko · Binance public · CryptoCompare)
# ══════════════════════════════════════════════════════════

def fetch_coingecko_global() -> dict | None:
    """BTC/ETH dominance + total market cap from CoinGecko (no key needed)."""
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        r.raise_for_status()
        d = r.json().get("data", {})
        return {
            "btc_dom":  round(d.get("market_cap_percentage", {}).get("btc", 0), 1),
            "eth_dom":  round(d.get("market_cap_percentage", {}).get("eth", 0), 1),
            "mcap_usd": d.get("total_market_cap", {}).get("usd", 0),
            "mcap_chg": round(d.get("market_cap_change_percentage_24h_usd", 0), 2),
        }
    except Exception as exc:
        log.warning(f"CoinGecko global: {exc}")
        return None


def fetch_coingecko_trending() -> list[dict]:
    """Top 5 trending coins on CoinGecko (no key needed)."""
    try:
        r = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=10)
        r.raise_for_status()
        coins = r.json().get("coins", [])[:5]
        return [{"name": c["item"]["name"], "symbol": c["item"]["symbol"].upper(),
                 "rank": c["item"].get("market_cap_rank")} for c in coins]
    except Exception as exc:
        log.warning(f"CoinGecko trending: {exc}")
        return []


def fetch_binance_top_movers() -> tuple[list, list]:
    """Top 3 gainers + top 3 losers from Binance 24h ticker (no key needed)."""
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=15)
        r.raise_for_status()
        tickers = [t for t in r.json()
                   if t["symbol"].endswith("USDT") and float(t["quoteVolume"]) > 5_000_000]
        tickers.sort(key=lambda t: float(t["priceChangePercent"]), reverse=True)
        gainers = [{"symbol": t["symbol"], "pct": float(t["priceChangePercent"])}
                   for t in tickers[:3]]
        losers  = [{"symbol": t["symbol"], "pct": float(t["priceChangePercent"])}
                   for t in tickers[-3:]]
        return gainers, losers
    except Exception as exc:
        log.warning(f"Binance movers: {exc}")
        return [], []


def fetch_binance_funding() -> list[dict]:
    """Extreme funding rates from Binance perpetual futures (no key needed)."""
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10)
        r.raise_for_status()
        rates = [{"symbol": t["symbol"], "rate": float(t["lastFundingRate"]) * 100}
                 for t in r.json() if t.get("lastFundingRate")]
        extremes = [x for x in rates if abs(x["rate"]) > 0.05]
        extremes.sort(key=lambda x: abs(x["rate"]), reverse=True)
        return extremes[:5]
    except Exception as exc:
        log.warning(f"Binance funding: {exc}")
        return []


def fetch_crypto_news() -> list[dict]:
    """Latest crypto news from CryptoCompare (no key needed for basic use)."""
    try:
        r = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/",
            params={"lang": "EN", "sortOrder": "latest"},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json().get("Data", [])[:5]
        return [{"title": i["title"], "source": i["source_info"]["name"]}
                for i in items]
    except Exception as exc:
        log.warning(f"Crypto news: {exc}")
        return []


# ══════════════════════════════════════════════════════════
#  📢  CHANNEL CONTENT
# ══════════════════════════════════════════════════════════

_prices:  dict[str, float] = {}
_sr_data: dict[str, dict]  = {}


def _collect(symbol: str, price: float, sr: dict) -> None:
    _prices[symbol]  = price
    _sr_data[symbol] = sr


def post_price_update() -> None:
    if not _prices:
        return
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    def fmt(sym: str, p: float) -> str:
        return f"`{sym:<10}` `{p:.{_decimals(sym)}f}`"
    fx   = [fmt(s, _prices[s]) for s in FOREX_PAIRS  if s in _prices]
    gold = [fmt(s, _prices[s]) for s in GOLD         if s in _prices]
    cry  = [fmt(s, _prices[s]) for s in CRYPTO_PAIRS if s in _prices]
    send_telegram(
        f"📊 *MARKET UPDATE — {now}*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"\n💱 *Forex*\n"    + "\n".join(fx)   +
        f"\n\n🥇 *Gold*\n"   + "\n".join(gold) +
        f"\n\n🪙 *Crypto*\n" + "\n".join(cry)  +
        f"\n\n🤖 _Scanning 24/7 every 15 min_"
    )


def post_session_open() -> None:
    h, m = datetime.now(timezone.utc).hour, datetime.now(timezone.utc).minute
    if m >= 15:
        return
    msgs = {
        0:  "🇯🇵 *TOKYO SESSION OPEN*\n━━━━━━━━━━━━━━━━━━━━\n00:00 UTC — Asian session 🌏\nJPY pairs in focus 🎯",
        8:  "🇬🇧 *LONDON SESSION OPEN*\n━━━━━━━━━━━━━━━━━━━━\n08:00 UTC — Market active 🔥\nEUR, GBP, Gold setups forming 📊",
        13: "🇺🇸 *NEW YORK SESSION OPEN*\n━━━━━━━━━━━━━━━━━━━━\n13:00 UTC — London+NY overlap 🔥\nBest signals of the day 📈",
    }
    if h in msgs:
        send_telegram(msgs[h])


def post_morning_watchlist() -> None:
    now = datetime.now(timezone.utc)
    if not (now.hour == 7 and now.minute >= 45) or not _sr_data:
        return
    lines = [f"📋 *WATCHLIST — {now.strftime('%A %d %B')}*",
             "━━━━━━━━━━━━━━━━━━━━", "Key S\\&R levels today:", ""]
    for sym, sr in _sr_data.items():
        d = _decimals(sym)
        if sr["resistance"]:
            lvl, cnt = sr["resistance"][0]
            lines.append(f"🔴 `{sym}` res `{lvl:.{d}f}` ({cnt}x)")
        if sr["support"]:
            lvl, cnt = sr["support"][0]
            lines.append(f"🟢 `{sym}` sup `{lvl:.{d}f}` ({cnt}x)")
    fg = get_fear_greed()
    if fg:
        lines.append(f"\n🧠 *Crypto Sentiment:* `{fg['value']} — {fg['label']}` {_fg_icon(fg['value'])}")
    lines += ["", "🔔 Bot alerts on breakout 🔓", "London open in 15 min ⏰"]
    send_telegram("\n".join(lines))


def post_eod_summary() -> None:
    now = datetime.now(timezone.utc)
    if not (now.hour == 21 and now.minute < 15):
        return
    sigs, open_count = _state.get("eod_signals", []), len(_state.get("open_trades", {}))
    if not sigs:
        send_telegram(
            "🌙 *END OF DAY SUMMARY*\n━━━━━━━━━━━━━━━━━━━━\n"
            "No breakout signals today.\nPatience pays 💪\n"
            f"📅 `{now.strftime('%Y-%m-%d')}`"
        )
    else:
        rows = "\n".join(f"• `{s['symbol']}` {s['side']} @ `{s['entry']}`" for s in sigs)
        send_telegram(
            f"🌙 *END OF DAY SUMMARY*\n━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ {len(sigs)} signal(s):\n\n{rows}\n\n"
            f"⏳ Open: `{open_count}` | 📅 `{now.strftime('%Y-%m-%d')}`"
        )


def post_weekly_summary() -> None:
    now  = datetime.now(timezone.utc)
    if not (now.weekday() == 0 and now.hour == 8 and now.minute < 15):
        return
    week = now.strftime("%Y-W%W")
    if _state.get("last_weekly_post") == week:
        return
    tr         = _state.get("trade_results", {})
    wins, losses = tr.get("wins", 0), tr.get("losses", 0)
    total      = wins + losses
    winrate    = f"{wins/total*100:.0f}%" if total else "—"
    send_telegram(
        f"📈 *WEEKLY PERFORMANCE*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"Week of `{now.strftime('%d %B %Y')}`\n\n"
        f"📊 Closed: `{total}` | ✅ `{wins}W` | ❌ `{losses}L` | Win rate: `{winrate}`\n"
        f"⏳ Still open: `{len(_state.get('open_trades', {}))}`\n\n"
        f"🤖 _Bot continues scanning 24/7_"
    )
    _state["last_weekly_post"] = week
    _state["trade_results"]    = {"week": week, "wins": 0, "losses": 0}
    save_state()


def post_pre_london_briefing() -> None:
    """06:00 UTC — BTC dominance, market cap, Fear & Greed, news count."""
    now = datetime.now(timezone.utc)
    if now.hour != 6 or now.minute >= 15:
        return
    key = "daily_pre_london"
    if _posted_within(key, DAILY):
        return
    gdata    = fetch_coingecko_global()
    fg       = get_fear_greed()
    today    = now.strftime("%Y-%m-%d")
    upcoming = sum(1 for e in _state.get("calendar", {}).values()
                   if e.get("date", "").startswith(today) and e.get("impact") == "High")
    lines = [
        f"☀️ *PRE-LONDON BRIEFING — {now.strftime('%d %b')}*",
        "━━━━━━━━━━━━━━━━━━━━",
        "London opens in 2 hours 🕗", "",
    ]
    if gdata:
        mcap_b = gdata["mcap_usd"] / 1e12
        arrow  = "📈" if gdata["mcap_chg"] >= 0 else "📉"
        lines += [
            f"🌍 *Crypto Market Cap:* `${mcap_b:.2f}T` {arrow} `{gdata['mcap_chg']:+.2f}%`",
            f"₿ *BTC Dom:* `{gdata['btc_dom']}%`  |  Ξ *ETH Dom:* `{gdata['eth_dom']}%`",
        ]
    if fg:
        lines.append(f"🧠 *Sentiment:* `{fg['value']} — {fg['label']}` {_fg_icon(fg['value'])}")
    lines += [
        f"📰 *High-impact events today:* `{upcoming}`",
        "", "🔔 Bot watching for London breakouts — alerts on fire 🔓",
    ]
    send_telegram("\n".join(lines))
    _mark_sent(key)


def post_midmorning_update() -> None:
    """10:00 UTC — CoinGecko trending coins + Binance extreme funding rates."""
    now = datetime.now(timezone.utc)
    if now.hour != 10 or now.minute >= 15:
        return
    key = "daily_midmorning"
    if _posted_within(key, DAILY):
        return
    trending = fetch_coingecko_trending()
    funding  = fetch_binance_funding()
    if not trending and not funding:
        return
    lines = [
        f"🔥 *MID-MORNING UPDATE — {now.strftime('%H:%M UTC')}*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    if trending:
        lines.append("\n🚀 *Trending on CoinGecko:*")
        for c in trending:
            rank = f"#{c['rank']}" if c["rank"] else "—"
            lines.append(f"  • `{c['symbol']}` {c['name']} ({rank})")
    if funding:
        lines.append("\n⚡ *Extreme Funding Rates \\(Binance\\):*")
        for f in funding:
            side = "longs paying 🔴" if f["rate"] > 0 else "shorts paying 🟢"
            lines.append(f"  • `{f['symbol']}` `{f['rate']:+.4f}%` — {side}")
        lines.append("_High positive = crowded longs = reversal risk_")
    send_telegram("\n".join(lines))
    _mark_sent(key)


def post_midday_pulse() -> None:
    """12:00 UTC — Binance 24h top movers ahead of NY open."""
    now = datetime.now(timezone.utc)
    if now.hour != 12 or now.minute >= 15:
        return
    key = "daily_midday"
    if _posted_within(key, DAILY):
        return
    gainers, losers = fetch_binance_top_movers()
    if not gainers:
        return
    lines = [
        f"📊 *MIDDAY PULSE — {now.strftime('%H:%M UTC')}*",
        "━━━━━━━━━━━━━━━━━━━━",
        "NY opens in 1 hour 🕐", "",
        "📈 *Top Gainers \\(24h\\):*",
    ]
    for g in gainers:
        lines.append(f"  🟢 `{g['symbol']}` `+{g['pct']:.2f}%`")
    lines.append("\n📉 *Top Losers \\(24h\\):*")
    for l in losers:
        lines.append(f"  🔴 `{l['symbol']}` `{l['pct']:.2f}%`")
    lines.append("\n🎯 Best signals fire during London\\+NY overlap \\(13–17 UTC\\)")
    send_telegram("\n".join(lines))
    _mark_sent(key)


def post_london_close() -> None:
    """16:00 UTC — London session recap: signals + open trades."""
    now = datetime.now(timezone.utc)
    if now.hour != 16 or now.minute >= 15:
        return
    key = "daily_london_close"
    if _posted_within(key, DAILY):
        return
    sigs_today  = _state.get("eod_signals", [])
    open_count  = len(_state.get("open_trades", {}))
    forex_sigs  = [s for s in sigs_today if "/" not in s["symbol"]]
    lines = [
        f"🇬🇧 *LONDON CLOSE — {now.strftime('%H:%M UTC')}*",
        "━━━━━━━━━━━━━━━━━━━━",
        "London session closing now 🔔", "",
        f"📊 Forex/Gold signals today: `{len(forex_sigs)}`",
        f"⏳ Open trades: `{open_count}`",
        "",
        "🇺🇸 New York session active until 22:00 UTC",
        "🪙 Crypto markets open 24/7 — bot never sleeps 🤖",
    ]
    send_telegram("\n".join(lines))
    _mark_sent(key)


def post_afternoon_crypto() -> None:
    """18:00 UTC — Latest crypto news headlines from CryptoCompare."""
    now = datetime.now(timezone.utc)
    if now.hour != 18 or now.minute >= 15:
        return
    key = "daily_afternoon_crypto"
    if _posted_within(key, DAILY):
        return
    news = fetch_crypto_news()
    if not news:
        return
    lines = [
        f"📰 *CRYPTO NEWS — {now.strftime('%H:%M UTC')}*",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    for i, item in enumerate(news[:4], 1):
        title = item["title"].replace("*", "").replace("_", "").replace("`", "")
        lines.append(f"{i}\\. {title}\n   _— {item['source']}_")
    lines.append("\n🤖 _Scanning for crypto breakouts 24/7_")
    send_telegram("\n".join(lines))
    _mark_sent(key)


def post_ny_check() -> None:
    """20:00 UTC — NY session check: open trades + day's signal recap."""
    now = datetime.now(timezone.utc)
    if now.hour != 20 or now.minute >= 15:
        return
    key = "daily_ny_check"
    if _posted_within(key, DAILY):
        return
    sigs_today  = _state.get("eod_signals", [])
    open_trades = _state.get("open_trades", {})
    lines = [
        f"🇺🇸 *NY SESSION CHECK — {now.strftime('%H:%M UTC')}*",
        "━━━━━━━━━━━━━━━━━━━━",
        "2 hours until NY close 🕙", "",
        f"📊 Signals today: `{len(sigs_today)}`",
    ]
    for s in sigs_today[-3:]:
        lines.append(f"  • `{s['symbol']}` {s['side']} @ `{s['entry']}`")
    lines.append(f"\n⏳ Open trades: `{len(open_trades)}`")
    for t in list(open_trades.values())[:3]:
        d = _decimals(t["symbol"])
        lines.append(f"  • `{t['symbol']}` {t['side']} @ `{t['entry']:.{d}f}`")
    lines.append("\n🌙 EOD summary at 21:00 UTC")
    send_telegram("\n".join(lines))
    _mark_sent(key)


def check_near_level(symbol: str, sr: dict) -> None:
    current, pct, d, key = sr["current"], 0.003, _decimals(symbol), f"{symbol}_near"
    def _alert(tag: str, lvl: float, cnt: int, dist: float) -> None:
        if not _is_duplicate(key):
            send_telegram(
                f"⚠️ *LEVEL APPROACHING — {symbol}*\n\n"
                f"📍 {tag}: `{lvl:.{d}f}` ({cnt} touches)\n"
                f"💵 Current: `{current:.{d}f}`\n"
                f"📏 Distance: `{dist:.2f}%` away\n"
                f"👀 Breakout possible soon\n"
                f"⏰ `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`"
            )
            _mark_sent(key)
    if sr["resistance"]:
        lvl, cnt = sr["resistance"][0]
        if cnt >= 3 and 0 < (lvl - current) / current <= pct:
            _alert("Resistance", lvl, cnt, (lvl - current) / current * 100)
    if sr["support"]:
        lvl, cnt = sr["support"][0]
        if cnt >= 3 and 0 < (current - lvl) / current <= pct:
            _alert("Support", lvl, cnt, (current - lvl) / current * 100)

# ══════════════════════════════════════════════════════════
#  🔍  SCANNER
# ══════════════════════════════════════════════════════════

def scan(symbol: str, df: pd.DataFrame, asset_type: str,
         df_daily: pd.DataFrame) -> None:
    if df.empty or len(df) < 60:
        log.warning(f"[{symbol}] Skipped — only {len(df)} bars")
        return
    if not is_active_session(asset_type):
        log.info(f"[{symbol}] Outside session")
        return

    sr           = get_sr_levels(df)
    daily_levels = get_daily_levels(df_daily)
    sig          = detect_breakout(df, sr, daily_levels)
    sig_key      = symbol

    # Retest check if no primary breakout
    if sig is None:
        retest = check_retest_setups(symbol, df)
        if retest:
            sig     = retest
            sig_key = f"{symbol}_rt"

    # Previous day high/low if still no signal
    if sig is None:
        pd_sig = detect_prev_day_breakout(df)
        if pd_sig:
            sig     = pd_sig
            sig_key = f"{symbol}_pd"

    _collect(symbol, sr["current"], sr)
    check_near_level(symbol, sr)

    log.info(
        f"[{symbol:10s}] {sr['current']:.5g}  "
        f"res={len(sr['resistance'])} sup={len(sr['support'])}"
        + ("  🚨 " + sig.get("label","SIGNAL") if sig else "")
    )

    if not sig:
        return

    if _state.get("paused"):
        log.info(f"[{symbol}] Bot paused — signal skipped")
        return

    if is_correlated_active(symbol):
        return

    # Crypto Fear & Greed — block LONG in Extreme Greed
    if asset_type == "Crypto" and "LONG" in sig["side"]:
        fg = get_fear_greed()
        if fg and fg["value"] >= 80:
            log.info(f"[{symbol}] Extreme Greed ({fg['value']}) — LONG blocked")
            send_telegram(
                f"⚠️ *{symbol} LONG skipped*\n"
                f"😱 Fear \\& Greed: `{fg['value']} — {fg['label']}`\n"
                f"Too risky to buy into extreme greed 🚫"
            )
            return

    # Spread filter
    if not is_spread_ok(symbol):
        log.info(f"[{symbol}] Spread too wide — signal skipped")
        return

    if is_news_blackout(symbol):
        log.info(f"[{symbol}] News blackout")
        send_telegram(f"⏸ *{symbol} signal paused* — high-impact news nearby 📰")
        return

    if _is_duplicate(sig_key):
        log.info(f"[{symbol}] Duplicate — cooldown active")
        return

    order_id = execute_trade(symbol, sig, asset_type)
    send_telegram(format_signal(symbol, sig, asset_type, order_id))
    _mark_sent(sig_key)
    register_open_trade(symbol, sig, asset_type)

    # Register broken level as a pending retest to watch
    if sig.get("touches") and "Breakout" in sig.get("label", ""):
        direction = "long" if "LONG" in sig["side"] else "short"
        register_pending_retest(symbol, sig["level"], direction)

    _state["eod_signals"].append({
        "symbol": symbol, "side": sig["side"],
        "entry": f"{sig['entry']:.{_decimals(symbol)}f}",
    })
    save_state()

# ══════════════════════════════════════════════════════════
#  🚀  MAIN LOOP
# ══════════════════════════════════════════════════════════

def run_scan() -> None:
    log.info(f"═══ SCAN @ {datetime.now(timezone.utc).strftime('%H:%M UTC')} ═══")
    _reset_eod_if_new_day()

    for name, ticker in FOREX_PAIRS.items():
        df_1h    = fetch_forex_gold(ticker)
        df_daily = fetch_forex_gold_daily(ticker)
        scan(name, df_1h, "Forex", df_daily)
        time.sleep(1)

    for name, ticker in GOLD.items():
        df_1h    = fetch_forex_gold(ticker)
        df_daily = fetch_forex_gold_daily(ticker)
        scan(name, df_1h, "Gold", df_daily)
        time.sleep(1)

    for pair in CRYPTO_PAIRS:
        df_1h    = fetch_crypto(pair)
        df_daily = fetch_crypto_daily(pair)
        scan(pair, df_1h, "Crypto", df_daily)
        time.sleep(2)

    track_open_trades()
    process_telegram_commands()
    post_upcoming_news()
    post_morning_watchlist()
    post_session_open()
    post_price_update()
    post_pre_london_briefing()
    post_midmorning_update()
    post_midday_pulse()
    post_london_close()
    post_afternoon_crypto()
    post_ny_check()
    post_eod_summary()
    post_weekly_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="S&R Breakout Trading Bot")
    parser.add_argument("--once", action="store_true",
                        help="Scan once then exit (GitHub Actions mode)")
    parser.add_argument("--commands-only", action="store_true",
                        help="Only process Telegram commands — no market scan (fast, runs every 1 min)")
    args = parser.parse_args()

    load_state()

    modes = []
    if OANDA_API_KEY:   modes.append(f"OANDA ({'demo' if OANDA_DEMO else 'LIVE'})")
    if BINANCE_API_KEY: modes.append("Binance")
    if KUCOIN_API_KEY:  modes.append("KuCoin")
    exec_str = " | ".join(modes) or "Signal-only"
    log.info(f"🤖 S&R Bot | {exec_str}")

    if getattr(args, "commands_only", False):
        process_telegram_commands()
        log.info("✅ Commands processed.")

    elif args.once:
        run_scan()
        log.info("✅ Done.")
    else:
        send_telegram(
            "🤖 *S\\&R Breakout Bot — LIVE 24/7*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📡 Forex | Gold | Crypto\n"
            "📐 S\\&R | 4H+1H | MACD | Candle | Divergence\n"
            f"⚡ {exec_str}\n"
            f"⏱ R:R 1:{RISK_REWARD} | TP1/TP2/TP3 | Risk {int(RISK_PCT*100)}%\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        while True:
            run_scan()
            log.info(f"⏳ Next scan in {SCAN_INTERVAL // 60} min")
            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
