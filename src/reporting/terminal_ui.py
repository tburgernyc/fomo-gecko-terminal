import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, TextIO
from urllib.parse import parse_qs, urlparse

import httpx
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import Settings
from src.db import cursor_scope

logger = logging.getLogger(__name__)


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>FOMO Gecko Terminal</title>
<script src="/lightweight-charts.js"></script>
<style>body{font:14px sans-serif;background:#10151d;color:#dbe4ef;margin:20px}#chart{height:65vh}</style></head>
<body><h2>FOMO + GeckoTerminal</h2><p>Live chart dashboard: <a href="http://localhost:8050">localhost:8050</a></p>
<div id="chart"></div><pre id="orders"></pre>
<script>
const chart=LightweightCharts.createChart(document.getElementById('chart'),{
  layout:{background:{color:'#10151d'},textColor:'#dbe4ef'},
  grid:{vertLines:{color:'#263241'},horzLines:{color:'#263241'}},
  timeScale:{timeVisible:true,secondsVisible:false}
});
const candles=chart.addCandlestickSeries();
async function refresh(){
  const token=new URLSearchParams(location.search).get('token')||'';
  const response=await fetch('/api/state?token='+encodeURIComponent(token)); if(!response.ok) throw new Error('state request failed');
  const state=await response.json(); candles.setData(state.candles);
  candles.setMarkers(state.markers);
  document.getElementById('orders').textContent=JSON.stringify(state.orders,null,2);
}
refresh().catch(console.error); setInterval(()=>refresh().catch(console.error),5000);
</script></body></html>"""


def _read_state(connection: Any, token_address: str | None = None) -> dict[str, Any]:
    snapshot_factory = getattr(connection, "get_readonly_snapshot_conn", None)
    if callable(snapshot_factory):
        with snapshot_factory() as snapshot:
            return _read_state(snapshot, token_address)

    with cursor_scope(connection) as cur:
        tokens = cur.execute(
            """
            SELECT token_address
            FROM market_snapshots
            GROUP BY token_address
            ORDER BY MAX(captured_at) DESC
            LIMIT 50
            """
        ).fetchall()
        if not token_address:
            row = cur.execute(
                """
                SELECT token_address FROM market_candles
                ORDER BY candle_ts DESC LIMIT 1
                """
            ).fetchone()
            token_address = str(row[0]) if row else (str(tokens[0][0]) if tokens else "")
        candles = cur.execute(
            """
            SELECT candle_ts, open, high, low, close, volume
            FROM market_candles
            WHERE token_address = ?
            ORDER BY candle_ts DESC LIMIT 200
            """,
            [token_address],
        ).fetchall()
        orders = cur.execute(
            """
            SELECT client_order_id, trader_handle, network, token_address, side,
                   status, requested_usd, filled_usd, entry_price_usd, created_at,
                   realized_pnl_usd, fee_usd, execution_detail
            FROM execution_orders
            WHERE token_address = ?
            ORDER BY created_at DESC LIMIT 25
            """,
            [token_address],
        ).fetchall()
        traders = cur.execute(
            """
            SELECT user_handle, rank_24h, pnl_24h_usd, win_rate, sleeve_weight
            FROM leaderboard_traders
            ORDER BY rank_24h
            LIMIT 10
            """
        ).fetchall()
        risk = cur.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN created_at >= CURRENT_TIMESTAMP - INTERVAL 24 HOUR
                                  THEN realized_pnl_usd ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN created_at >= CURRENT_TIMESTAMP - INTERVAL 24 HOUR
                                  THEN fee_usd ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN side = 'BUY'
                                       AND status IN ('OPEN', 'DRY_RUN_OPEN')
                                  THEN unrealized_pnl_usd ELSE 0 END), 0),
                COUNT(CASE WHEN side = 'BUY'
                                AND status IN ('OPEN', 'DRY_RUN_OPEN') THEN 1 END),
                COUNT(CASE WHEN status IN ('INTENT', 'SUBMITTED', 'UNCERTAIN') THEN 1 END)
            FROM execution_orders
            """
        ).fetchone()

    candles.reverse()
    candle_rows = [
        {
            "time": int(timestamp.timestamp()),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume or 0.0,
        }
        for timestamp, open_price, high_price, low_price, close_price, volume in candles
    ]
    markers = []
    for row in orders:
        timestamp, side, status, price, trader = row[9], row[4], row[5], row[8], row[1]
        if timestamp and price:
            markers.append({
                "time": int(timestamp.timestamp()),
                "position": "belowBar" if side == "BUY" else "aboveBar",
                "color": "#26a69a" if side == "BUY" else "#ef5350",
                "shape": "arrowUp" if side == "BUY" else "arrowDown",
                "text": f"{side} {trader} {status}",
                "side": side,
                "trader": trader,
                "price": price,
            })
    order_rows = [
        {
            "client_order_id": row[0],
            "trader": row[1],
            "network": row[2],
            "token": row[3],
            "side": row[4],
            "status": row[5],
            "requested_usd": row[6],
            "filled_usd": row[7],
            "entry_price_usd": row[8],
            "created_at": row[9].isoformat() if row[9] else None,
            "realized_pnl_usd": row[10],
            "fee_usd": row[11],
            "execution_detail": row[12],
        }
        for row in orders
    ]
    markers.sort(key=lambda marker: marker["time"])
    realized, fees, unrealized, open_positions, unresolved = risk
    return {
        "token": token_address,
        "tokens": [str(row[0]) for row in tokens],
        "candles": candle_rows,
        "markers": markers,
        "orders": order_rows,
        "traders": [
            {
                "handle": row[0],
                "rank": row[1],
                "pnl_24h_usd": row[2],
                "win_rate": row[3],
                "sleeve_weight": row[4],
            }
            for row in traders
        ],
        "risk": {
            "realized_pnl_24h_usd": float(realized),
            "fees_24h_usd": float(fees),
            "unrealized_pnl_usd": float(unrealized),
            "net_pnl_24h_usd": float(realized) + float(unrealized) - float(fees),
            "open_positions": int(open_positions),
            "unresolved_orders": int(unresolved),
        },
    }


def start_terminal_ui(
    connection: Any, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    chart_library = (
        Path(__file__).parent / "static" / "lightweight-charts.standalone.production.js"
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = _HTML.encode()
                content_type = "text/html; charset=utf-8"
            elif parsed.path == "/lightweight-charts.js":
                body = chart_library.read_bytes()
                content_type = "text/javascript; charset=utf-8"
            elif parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                token = query.get("token", [None])[0]
                body = json.dumps(_read_state(connection, token), default=str).encode()
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("Terminal UI: " + format, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    Thread(target=server.serve_forever, name="terminal-ui-api", daemon=True).start()
    logger.info("Terminal state API listening at http://%s:%d", host, server.server_port)
    return server


@dataclass
class ChartDashboardProcess:
    process: subprocess.Popen[str]
    output_thread: Thread

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.output_thread.join(timeout=1)


def _forward_chart_server_logs(output: TextIO) -> None:
    for line in output:
        message = line.rstrip()
        if message:
            logger.info("Chart dashboard: %s", message)


async def start_chart_dashboard(
    host: str = "127.0.0.1",
    port: int = 8050,
    state_api_url: str = "http://127.0.0.1:8765",
    startup_timeout_seconds: float = 30,
) -> ChartDashboardProcess:
    script = Path(__file__).with_name("chart_dashboard.py")
    inherited_environment = {
        "PATH", "HOME", "USER", "LANG", "LC_ALL", "VIRTUAL_ENV",
        "PYTHONPATH", "PYTHONHOME", "TMPDIR",
    }
    environment = {
        key: value for key, value in os.environ.items() if key in inherited_environment
    }
    environment["FOMO_STATE_API_URL"] = state_api_url.rstrip("/")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(script),
            f"--server.address={host}",
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    if process.stdout is None:
        process.terminate()
        raise RuntimeError("Unable to capture chart dashboard server output")
    output_thread = Thread(
        target=_forward_chart_server_logs,
        args=(process.stdout,),
        name="chart-dashboard-logs",
        daemon=True,
    )
    output_thread.start()
    dashboard = ChartDashboardProcess(process, output_thread)
    deadline = asyncio.get_running_loop().time() + startup_timeout_seconds
    try:
        async with httpx.AsyncClient(timeout=1) as client:
            while asyncio.get_running_loop().time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Chart dashboard exited during startup with status {process.returncode}"
                    )
                try:
                    response = await client.get(f"http://{host}:{port}/_stcore/health")
                    if response.is_success:
                        logger.info("Chart dashboard listening at http://%s:%d", host, port)
                        return dashboard
                except httpx.RequestError:
                    pass
                await asyncio.sleep(0.25)
        raise TimeoutError(
            f"Chart dashboard did not become ready at http://{host}:{port} "
            f"within {startup_timeout_seconds:g} seconds"
        )
    except BaseException:
        dashboard.close()
        raise


def render_risk_dashboard(
    connection: Any,
    settings: Settings,
    chart_url: str = "http://localhost:8050",
) -> RenderableType:
    state = _read_state(connection)
    risk = state["risk"]
    net_pnl = risk["net_pnl_24h_usd"]
    loss_limit = settings.starting_bankroll_usd * settings.max_daily_loss_pct
    pnl_style = "red" if net_pnl <= -loss_limit else ("green" if net_pnl >= 0 else "yellow")

    overview = Table.grid(expand=True)
    overview.add_column()
    overview.add_column(justify="right")
    overview.add_row("Execution mode", "DRY RUN" if settings.dry_run else "LIVE")
    overview.add_row("Rolling 24h net P&L", Text(f"${net_pnl:,.2f}", style=pnl_style))
    overview.add_row("24h realized P&L", f"${risk['realized_pnl_24h_usd']:,.2f}")
    overview.add_row("24h fees", f"${risk['fees_24h_usd']:,.2f}")
    overview.add_row("Open positions", str(risk["open_positions"]))
    overview.add_row("Unresolved orders", str(risk["unresolved_orders"]))
    overview.add_row(
        "Circuit breaker",
        Text("TRIPPED" if net_pnl <= -loss_limit else "ARMED",
             style="red bold" if net_pnl <= -loss_limit else "green"),
    )
    overview.add_row("Browser chart", chart_url)

    traders = Table(expand=True, title="FOMO Top-10")
    traders.add_column("#", justify="right", width=3)
    traders.add_column("Trader")
    traders.add_column("24h P&L", justify="right")
    traders.add_column("Win %", justify="right")
    traders.add_column("Sleeve", justify="right")
    for trader in state["traders"]:
        traders.add_row(
            str(trader["rank"] or "-"),
            trader["handle"],
            f"${float(trader['pnl_24h_usd'] or 0):,.2f}",
            f"{float(trader['win_rate'] or 0) * 100:.1f}%",
            f"{float(trader['sleeve_weight'] or 0) * 100:.1f}%",
        )
    if not state["traders"]:
        traders.add_row("-", "No persisted leaderboard", "-", "-", "-")

    orders = Table(expand=True, title="Recent order lifecycle")
    orders.add_column("Time", no_wrap=True)
    orders.add_column("Trader")
    orders.add_column("Side")
    orders.add_column("Token")
    orders.add_column("Status")
    orders.add_column("Notional", justify="right")
    orders.add_column("P&L", justify="right")
    for order in state["orders"][:8]:
        timestamp = order["created_at"] or ""
        orders.add_row(
            timestamp[11:19] if len(timestamp) >= 19 else timestamp,
            order["trader"],
            order["side"],
            order["token"][:12],
            order["status"],
            f"${float(order['requested_usd'] or 0):,.2f}",
            f"${float(order['realized_pnl_usd'] or 0):,.2f}",
        )
    if not state["orders"]:
        orders.add_row("-", "-", "-", "-", "No orders", "-", "-")

    return Group(
        Panel(overview, title="FOMO + GeckoTerminal | Risk Monitor"),
        traders,
        orders,
    )


async def run_terminal_dashboard(
    connection: Any,
    settings: Settings,
    stop_event: asyncio.Event,
    *,
    chart_url: str = "http://localhost:8050",
    refresh_seconds: float = 2,
    console: Console | None = None,
) -> None:
    dashboard_console = console or Console(stderr=True)
    with Live(
        render_risk_dashboard(connection, settings, chart_url),
        console=dashboard_console,
        refresh_per_second=1,
        screen=False,
    ) as live:
        while not stop_event.is_set():
            live.update(render_risk_dashboard(connection, settings, chart_url), refresh=True)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=refresh_seconds)
            except asyncio.TimeoutError:
                continue
