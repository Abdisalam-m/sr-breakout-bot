# sr-breakout-bot

**Owner:** @kapoy111  
**Goal:** Real S&R breakout trading bot — live trade execution + Telegram signals + 24/7 market updates

## What this bot does
- Detects Support & Resistance breakouts on Forex, Gold (XAUUSD), and Crypto (1H timeframe)
- Sends Telegram signals with entry, stop loss, take profit, grade, and volume confirmation
- Runs via GitHub Actions every 15 min using `python bot.py --once`
- Session filter: Forex/Gold only fire during London (08:00–17:00 UTC) and New York (13:00–22:00 UTC)

## Credentials (stored in GitHub Secrets + .env)
- `TELEGRAM_TOKEN` — Telegram bot token
- `TELEGRAM_CHAT_ID` — Telegram channel/chat ID
- Broker API keys to be added (OANDA or similar)

## Roadmap (in progress)
- [ ] Connect to OANDA or similar broker API for live trade execution
- [ ] Add real-time economic calendar alerts (high-impact news)
- [ ] Add 24/7 Telegram market update schedule (session opens, news, price levels)
- [ ] Add position/trade tracker (open trades, P&L updates)
- [ ] Add multi-timeframe confirmation (4H + 1H)
- [ ] Improve signal persistence across GitHub Actions runs (currently in-memory only)

## Key files
- `bot.py` — main bot logic (fetch, S&R detection, breakout scan, Telegram)
- `.github/workflows/scan.yml` — GitHub Actions cron (every 15 min)
- `requirements.txt` — Python dependencies

## Tech stack
- Python, yfinance (Forex/Gold), ccxt/KuCoin (Crypto)
- Telegram Bot API for signals
- GitHub Actions for scheduling
