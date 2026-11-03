import os
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from lightweight_charts.widgets import StreamlitChart


STATE_API_URL = os.environ.get("FOMO_STATE_API_URL", "http://127.0.0.1:8765").rstrip("/")


def fetch_state(token: str | None = None) -> dict[str, Any]:
    with httpx.Client(timeout=5) as client:
        response = client.get(
            f"{STATE_API_URL}/api/state",
            params={"token": token} if token else None,
        )
        response.raise_for_status()
        state = response.json()
    if not isinstance(state, dict):
        raise ValueError("state API response must be a JSON object")
    return state


def candle_frame(candles: list[dict[str, Any]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(candles)
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    return frame[["time", "open", "high", "low", "close", "volume"]]


def add_trade_markers(chart: Any, markers: list[dict[str, Any]]) -> None:
    for marker in markers:
        chart.marker(
            time=datetime.fromtimestamp(int(marker["time"]), timezone.utc),
            position="below" if marker["side"] == "BUY" else "above",
            shape="arrow_up" if marker["side"] == "BUY" else "arrow_down",
            color=marker["color"],
            text=f"{marker['side']} | {marker['trader']}",
        )


def render_dashboard() -> None:
    st.title("FOMO + GeckoTerminal")
    st.caption("Live five-minute candles, copied-trade markers, and Top-10 risk context.")

    try:
        state = fetch_state()
    except (httpx.HTTPError, ValueError) as exc:
        st.error(f"Cannot read the local FOMO state API: {exc}")
        st.stop()

    tokens = state.get("tokens", [])
    if not tokens:
        st.info("Waiting for the first verified GeckoTerminal market snapshot.")
        return
    initial_token = state.get("token")
    selected_token = st.selectbox(
        "Token",
        options=tokens,
        index=tokens.index(initial_token) if initial_token in tokens else 0,
    )
    if selected_token != initial_token:
        try:
            state = fetch_state(selected_token)
        except (httpx.HTTPError, ValueError) as exc:
            st.error(f"Cannot load {selected_token}: {exc}")
            st.stop()

    risk = state.get("risk", {})
    net_pnl = float(risk.get("net_pnl_24h_usd") or 0)
    realized = float(risk.get("realized_pnl_24h_usd") or 0)
    fees = float(risk.get("fees_24h_usd") or 0)
    unrealized = float(risk.get("unrealized_pnl_usd") or 0)
    open_positions = int(risk.get("open_positions") or 0)
    unresolved = int(risk.get("unresolved_orders") or 0)
    metrics = st.columns(6)
    metrics[0].metric("Rolling 24h net P&L", f"${net_pnl:,.2f}")
    metrics[1].metric("Realized P&L", f"${realized:,.2f}")
    metrics[2].metric("Unrealized P&L", f"${unrealized:,.2f}")
    metrics[3].metric("Fees", f"${fees:,.2f}")
    metrics[4].metric("Open positions", str(open_positions))
    metrics[5].metric("Unresolved orders", str(unresolved))

    candles = state.get("candles", [])
    if candles:
        chart = StreamlitChart(height=600, toolbox=False)
        chart.set(candle_frame(candles))
        add_trade_markers(chart, state.get("markers", []))
        chart.load()
    else:
        st.info("No five-minute candles are stored for this token yet.")

    leaders, orders = st.columns((1, 2))
    with leaders:
        st.subheader("FOMO Top-10")
        traders = state.get("traders", [])
        if traders:
            st.dataframe(
                pd.DataFrame(traders),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No persisted leaderboard data.")
    with orders:
        st.subheader("Recent executions")
        recent_orders = state.get("orders", [])
        if recent_orders:
            st.dataframe(
                pd.DataFrame(recent_orders),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No orders for this token.")


@st.fragment(run_every="2s")
def live_dashboard() -> None:
    render_dashboard()


def main() -> None:
    st.set_page_config(
        page_title="FOMO Gecko Market Dashboard",
        layout="wide",
    )
    live_dashboard()


if __name__ == "__main__":
    main()
