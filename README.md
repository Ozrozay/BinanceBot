# OzBot — Binance Futures Trading Bot

An automated crypto trading bot for Binance Futures with Telegram control, multi-channel signal monitoring, and smart risk management.

---

## Features

- **Auto-trades** signals from Telegram channels instantly
- **Paste any signal** directly to your bot — it parses and executes it
- **Multiple channels** monitored simultaneously
- **Partial TP closes** — 50% at TP1, 25% at TP2, 25% at TP3
- **Breakeven stop** — moves stop to entry after TP1 hits
- **Smart leverage** — auto steps down if Binance rejects your requested leverage
- **Multiple simultaneous positions** — no single-position limit
- **Auto-pause** after X consecutive losses
- **Daily & weekly P&L reports** sent to Telegram
- **Morning health check** every day at 08:30 UTC
- **Drawdown alerts** when balance drops too far
- **AI chat** — ask the bot "what's a good trade now?" (requires Gemini API key)

### Telegram Commands
| Command | What it does |
|---|---|
| `/status` | Current position and bot state |
| `/positions` | All open positions with unrealised P&L |
| `/balance` | USDT balance and open position value |
| `/pnl` | Today's P&L |
| `/stats` | All-time win rate, R:R, best/worst trade |
| `/history` | Last 10 closed trades |
| `/close` | Close bot-tracked position |
| `/close SYMBOL` | Close a specific position (e.g. `/close SOL`) |
| `/closeall` | Close ALL open positions |
| `/cancel SYMBOL` | Cancel TP/stop orders without closing |
| `/scan` | Scan all pairs for best setup right now |
| `/auto off` | Pause automatic trading |
| `/auto on` | Resume automatic trading |
| `/help` | Full command list |

---

## Requirements

- A VPS running **Ubuntu 22.04** (DigitalOcean, Vultr, Linode — ~$6/month)
- A **Binance account** with Futures enabled
- A **Telegram account**

---

## Quick Setup (Recommended)

### Step 1 — Get a VPS
Sign up at [Vultr](https://vultr.com) or [DigitalOcean](https://digitalocean.com).  
Create an **Ubuntu 22.04** server (the $6/month plan is fine).

### Step 2 — SSH into your VPS
```bash
ssh root@YOUR_VPS_IP
```

### Step 3 — Download the bot
```bash
git clone https://github.com/YOUR_USERNAME/binancebot.git /root/binancebot
cd /root/binancebot
```

### Step 4 — Run the setup wizard
```bash
bash setup.sh
```

The wizard will ask you for all your API keys and configure everything automatically.

### Step 5 — Log into Telegram (one time only)
```bash
cd /root/binancebot && .venv/bin/python -c "
from telethon.sync import TelegramClient
import os; from dotenv import load_dotenv; load_dotenv()
c = TelegramClient('session', int(os.environ['TELEGRAM_API_ID']), os.environ['TELEGRAM_API_HASH'])
c.start()
print('Login successful!')
"
```
Enter your phone number and the code Telegram sends you.

### Step 6 — Start the bot
```bash
systemctl start binancebot
tail -f /root/binancebot/logs/bot.log
```

Open Telegram and send `/help` to your bot. 🚀

---

## Manual Setup (if you prefer)

### 1. Install dependencies
```bash
apt-get update && apt-get install -y python3 python3-pip python3-venv
cd /root/binancebot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Configure your environment
```bash
cp .env.example .env
nano .env   # fill in your API keys
```

### 3. Install the systemd service
```bash
cp binancebot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable binancebot
systemctl start binancebot
```

---

## Getting Your API Keys

### Binance API Key
1. Go to [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Create a new API key
3. Enable **Futures Trading** permission
4. Add your VPS IP to the whitelist (recommended)
5. Copy the API Key and Secret into `.env`

### Telegram Bot Token
1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`
3. Follow the prompts — choose a name and username
4. Copy the token into `.env` as `TELEGRAM_TOKEN`

### Telegram Chat ID
1. Start your bot on Telegram (send it any message)
2. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Find `"chat":{"id": 123456789}` — that number is your chat ID
4. Copy it into `.env` as `TELEGRAM_CHAT_ID`

### Telegram API ID & Hash (for channel monitoring)
1. Go to [https://my.telegram.org](https://my.telegram.org)
2. Log in with your phone number
3. Click **API development tools**
4. Create an app (name/description can be anything)
5. Copy `App api_id` → `TELEGRAM_API_ID`
6. Copy `App api_hash` → `TELEGRAM_API_HASH`

### Gemini API Key (optional — for AI chat)
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a free API key
3. Copy it into `.env` as `GEMINI_API_KEY`

---

## Useful Commands

```bash
# View live logs
tail -f /root/binancebot/logs/bot.log

# Restart the bot
systemctl restart binancebot

# Stop the bot
systemctl stop binancebot

# Check bot status
systemctl status binancebot

# Push code updates from your Mac
rsync -avz --exclude='.git' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='data/' --exclude='.venv' \
  --exclude='*.session' /path/to/binancebot/ root@YOUR_VPS_IP:/root/binancebot/
```

---

## How to Send Signals

Paste any signal message directly into your Telegram bot chat:

```
SOL/USDT SHORT
Entry at market price
Leverage 10X
TP1: 106.10
TP2: 98.90
TP3: 85.00
🛑 Stoploss: 114.90
Use only 1% of your wallet.
```

The bot parses it and executes immediately — no confirmation needed.

---

## Risk Warning

⚠️ **This bot trades real money on your Binance account.**  
Crypto futures trading is high risk. You can lose your entire balance.  
Start with `TRADING_ENV=testnet` to test with paper money before going live.  
Never trade more than you can afford to lose.

---

## Configuration Reference

See [`.env.example`](.env.example) for all available settings with descriptions.

Key settings to tune:
- `RISK_PER_TRADE_PCT` — % of balance risked per trade (default: 1%)
- `MAX_LEVERAGE` — max leverage for strategy signals (channel signals use their own)
- `MAX_CONSECUTIVE_LOSSES` — auto-pause after this many losses in a row (default: 3)
- `DRAWDOWN_ALERT_PCT` — alert when balance drops this % from peak (default: 10%)
