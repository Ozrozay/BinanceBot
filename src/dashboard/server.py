"""
Dashboard server — observer only.

A lightweight Flask app that reads from the SQLite database and serves a
single-page dashboard. It has zero write access to the database and cannot
affect the trading loop in any way.

Run separately from the bot:
    .venv/bin/python -m src.dashboard.server

Or run alongside the bot in a second terminal.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template_string

from src.database.schema import get_connection

app = Flask(__name__)

# Shared state — populated by the bot process via DB reads
_db_path: str = "data/trading_bot.db"
_config_summary: dict = {}
_lock = threading.Lock()


def configure(db_path: str, config_summary: dict) -> None:
    global _db_path, _config_summary
    _db_path = db_path
    _config_summary = config_summary


def _conn() -> sqlite3.Connection:
    return get_connection(_db_path)


# ------------------------------------------------------------------
# HTML template (self-contained, no external CDN dependencies)
# ------------------------------------------------------------------

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>OzBinanceBot Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Courier New', monospace; background: #0d1117; color: #e6edf3; padding: 20px; }
  h1 { color: #58a6ff; font-size: 1.4rem; margin-bottom: 20px; }
  h2 { color: #8b949e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card .label { color: #8b949e; font-size: 0.75rem; margin-bottom: 4px; }
  .card .value { font-size: 1.3rem; font-weight: bold; }
  .green { color: #3fb950; }
  .red { color: #f85149; }
  .yellow { color: #d29922; }
  .blue { color: #58a6ff; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th { color: #8b949e; text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d; }
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
  tr:hover td { background: #161b22; }
  .section { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
  .env-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; }
  .env-mainnet_readonly { background: #1f2d1f; color: #3fb950; border: 1px solid #3fb950; }
  .env-testnet { background: #2d2a1f; color: #d29922; border: 1px solid #d29922; }
  .env-live { background: #2d1f1f; color: #f85149; border: 1px solid #f85149; }
  .footer { color: #484f58; font-size: 0.72rem; margin-top: 20px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #3fb950; margin-right: 6px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
</style>
</head>
<body>
<h1><span class="dot"></span>OzBinanceBot Dashboard</h1>

<div class="grid">
  <div class="card">
    <div class="label">Environment</div>
    <div class="value">
      <span class="env-badge env-{{ config.trading_env }}">{{ config.trading_env }}</span>
    </div>
  </div>
  <div class="card">
    <div class="label">Symbol / Timeframe</div>
    <div class="value blue">{{ config.symbol }} {{ config.timeframe }}</div>
  </div>
  <div class="card">
    <div class="label">Total Signals (all time)</div>
    <div class="value">{{ stats.total_signals }}</div>
  </div>
  <div class="card">
    <div class="label">Trades (closed)</div>
    <div class="value">{{ stats.total_trades }}</div>
  </div>
  <div class="card">
    <div class="label">Today's P&L</div>
    <div class="value {% if stats.daily_pnl >= 0 %}green{% else %}red{% endif %}">
      {% if stats.daily_pnl >= 0 %}+{% endif %}{{ "%.2f"|format(stats.daily_pnl) }} USDT
    </div>
  </div>
  <div class="card">
    <div class="label">All-time P&L</div>
    <div class="value {% if stats.total_pnl >= 0 %}green{% else %}red{% endif %}">
      {% if stats.total_pnl >= 0 %}+{% endif %}{{ "%.2f"|format(stats.total_pnl) }} USDT
    </div>
  </div>
</div>

{% if open_trades %}
<div class="section">
  <h2>Open Positions</h2>
  <table>
    <tr><th>Symbol</th><th>Direction</th><th>Entry Price</th><th>Size</th><th>Stop</th><th>TP</th><th>Opened</th></tr>
    {% for t in open_trades %}
    <tr>
      <td>{{ t.symbol }}</td>
      <td class="{% if t.direction == 'long' %}green{% else %}red{% endif %}">{{ t.direction|upper }}</td>
      <td>{{ "%.2f"|format(t.entry_price) }}</td>
      <td>{{ "%.4f"|format(t.position_size) }}</td>
      <td class="red">{{ "%.2f"|format(t.stop_price) if t.stop_price else '—' }}</td>
      <td class="green">{{ "%.2f"|format(t.tp_price) if t.tp_price else '—' }}</td>
      <td>{{ t.entry_ts[:16] }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

<div class="section">
  <h2>Recent Signals</h2>
  <table>
    <tr><th>Time</th><th>Type</th><th>Direction</th><th>Entry</th><th>Stop</th><th>TP</th><th>Reason</th></tr>
    {% for s in signals %}
    <tr>
      <td>{{ s.ts[:16] }}</td>
      <td class="{% if s.signal_type == 'entry' %}green{% elif s.signal_type == 'exit' %}red{% else %}{% endif %}">
        {{ s.signal_type|upper }}
      </td>
      <td>{{ s.direction }}</td>
      <td>{{ "%.2f"|format(s.entry_price) if s.entry_price else '—' }}</td>
      <td>{{ "%.2f"|format(s.stop_price) if s.stop_price else '—' }}</td>
      <td>{{ "%.2f"|format(s.tp_price) if s.tp_price else '—' }}</td>
      <td style="color:#8b949e;font-size:0.78rem">{{ s.reason }}</td>
    </tr>
    {% endfor %}
  </table>
</div>

{% if closed_trades %}
<div class="section">
  <h2>Recent Closed Trades</h2>
  <table>
    <tr><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th><th>Closed</th></tr>
    {% for t in closed_trades %}
    <tr>
      <td>{{ t.symbol }}</td>
      <td class="{% if t.direction == 'long' %}green{% else %}red{% endif %}">{{ t.direction|upper }}</td>
      <td>{{ "%.2f"|format(t.entry_price) }}</td>
      <td>{{ "%.2f"|format(t.exit_price) if t.exit_price else '—' }}</td>
      <td class="{% if t.pnl_usd and t.pnl_usd >= 0 %}green{% else %}red{% endif %}">
        {% if t.pnl_usd %}{% if t.pnl_usd >= 0 %}+{% endif %}{{ "%.2f"|format(t.pnl_usd) }}{% else %}—{% endif %}
      </td>
      <td>{{ t.exit_reason or '—' }}</td>
      <td>{{ t.exit_ts[:16] if t.exit_ts else '—' }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

<div class="footer">
  Auto-refreshes every 30s &nbsp;|&nbsp; {{ now }} UTC &nbsp;|&nbsp;
  Strategy: EMA{{ config.ema_period }} + RSI{{ config.rsi_period }} |
  Stop {{ config.atr_stop_multiplier }}×ATR | TP {{ config.atr_trail_multiplier }}×ATR |
  Risk {{ config.risk_pct }}% per trade
</div>
</body>
</html>
"""


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/")
def index():
    try:
        conn = _conn()
    except Exception:
        return "Database not found — start the bot first.", 503

    # Stats
    total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    total_trades  = conn.execute("SELECT COUNT(*) FROM trades WHERE exit_ts IS NOT NULL").fetchone()[0]
    daily_pnl     = conn.execute(
        "SELECT COALESCE(SUM(pnl_usd),0) FROM trades WHERE exit_ts LIKE ?",
        (datetime.now(timezone.utc).strftime("%Y-%m-%d") + "%",)
    ).fetchone()[0]
    total_pnl = conn.execute("SELECT COALESCE(SUM(pnl_usd),0) FROM trades WHERE exit_ts IS NOT NULL").fetchone()[0]

    signals      = conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT 20").fetchall()
    open_trades  = conn.execute("SELECT * FROM trades WHERE exit_ts IS NULL").fetchall()
    closed_trades = conn.execute("SELECT * FROM trades WHERE exit_ts IS NOT NULL ORDER BY id DESC LIMIT 10").fetchall()
    conn.close()

    return render_template_string(
        TEMPLATE,
        config=_config_summary,
        stats=dict(
            total_signals=total_signals,
            total_trades=total_trades,
            daily_pnl=float(daily_pnl or 0),
            total_pnl=float(total_pnl or 0),
        ),
        signals=[dict(s) for s in signals],
        open_trades=[dict(t) for t in open_trades],
        closed_trades=[dict(t) for t in closed_trades],
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.route("/api/status")
def api_status():
    """JSON endpoint for programmatic access."""
    try:
        conn = _conn()
        total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        total_trades  = conn.execute("SELECT COUNT(*) FROM trades WHERE exit_ts IS NOT NULL").fetchone()[0]
        open_count    = conn.execute("SELECT COUNT(*) FROM trades WHERE exit_ts IS NULL").fetchone()[0]
        conn.close()
        return jsonify({"ok": True, "total_signals": total_signals,
                        "total_trades": total_trades, "open_positions": open_count})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 503


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

def run(db_path: str, config_summary: dict, host: str = "127.0.0.1", port: int = 5000) -> None:
    configure(db_path, config_summary)
    app.run(host=host, port=port, debug=False, use_reloader=False)
