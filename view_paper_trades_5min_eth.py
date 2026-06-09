"""
Paper Trading Viewer
View and analyze simulation trades
"""
import json
from datetime import datetime
from pathlib import Path

PAPER_TRADES_JSONL = Path("paper_trades_eth_5min.jsonl")
LEGACY_PAPER_TRADES_JSON = Path("paper_trades_eth_5min.json")
TABLE_WIDTH = 124


def _load_json_array(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return rows if isinstance(rows, list) else []


def _load_jsonl(path):
    if not path.exists():
        return []
    trades = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trades.append(json.loads(line))
    return trades


def _dedupe_trades(trades):
    seen = set()
    deduped = []
    for trade in trades:
        key = (
            trade.get("timestamp"),
            trade.get("direction"),
            trade.get("size_usd"),
            trade.get("price"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(trade)
    return sorted(deduped, key=lambda t: t.get("timestamp", ""))


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _calculate_trade_pnl(trade):
    pnl_usd = trade.get("pnl_usd")
    if pnl_usd is not None:
        return _as_float(pnl_usd)

    size_usd = _as_float(trade.get("size_usd"))
    entry_price = _as_float(trade.get("price", trade.get("entry_price")))

    exit_price = trade.get("exit_price")
    if exit_price is not None and entry_price > 0:
        return size_usd * (_as_float(exit_price) - entry_price) / entry_price

    outcome = str(trade.get("outcome", "")).upper()
    if outcome == "WIN" and entry_price > 0:
        return size_usd * (1.0 - entry_price) / entry_price
    if outcome == "LOSS":
        return -size_usd
    return None


def calculate_bot_summary(trades):
    settled = [
        (trade, pnl)
        for trade in trades
        for pnl in [_calculate_trade_pnl(trade)]
        if pnl is not None
    ]
    total_invested = sum(_as_float(trade.get("size_usd")) for trade, _ in settled)
    total_pnl = sum(pnl for _, pnl in settled)
    roi = (total_pnl / total_invested * 100.0) if total_invested else 0.0
    return total_invested, total_pnl, roi


def _calculate_trade_roi(trade):
    pnl = _calculate_trade_pnl(trade)
    size_usd = _as_float(trade.get("size_usd"))
    if pnl is None or size_usd <= 0:
        return None
    return pnl / size_usd * 100.0


def _format_signed_usd(value):
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.2f}"


def _format_trade_pnl(value):
    return _format_signed_usd(value) if value is not None else "N/A"


def _format_trade_roi(value):
    return f"{value:+.2f}%" if value is not None else "N/A"


def load_paper_trades():
    """Load paper trades from file."""
    try:
        trades = _load_json_array(LEGACY_PAPER_TRADES_JSON)
        trades.extend(_load_jsonl(PAPER_TRADES_JSONL))
        if trades:
            return _dedupe_trades(trades)
        print("No paper trades file found.")
        return []
    except Exception as e:
        print(f"Error loading paper trades: {e}")
        return []


def display_paper_trades(trades):
    """Display paper trades in a nice format."""
    if not trades:
        print("\nNo paper trades recorded yet.")
        return
    
    print("\n" + "=" * TABLE_WIDTH)
    print("PAPER TRADING RESULTS (SIMULATION)")
    print("=" * TABLE_WIDTH)
    print()
    
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if str(t.get('outcome', '')).upper() == 'WIN')
    losing_trades = sum(1 for t in trades if str(t.get('outcome', '')).upper() == 'LOSS')
    pending_trades = sum(1 for t in trades if str(t.get('outcome', 'PENDING')).upper() == 'PENDING')
    
    print(f"Total Trades: {total_trades}")
    print(f"Winning: {winning_trades}")
    print(f"Losing: {losing_trades}")
    print(f"Pending: {pending_trades}")
    
    if winning_trades + losing_trades > 0:
        win_rate = winning_trades / (winning_trades + losing_trades) * 100
        print(f"Win Rate: {win_rate:.1f}%")
    
    print()
    print("-" * TABLE_WIDTH)
    print(f"{'#':<4} {'Time':<20} {'Token':<8} {'Size':<10} {'Price':<10} {'Model':<8} {'Edge':<8} {'Kelly':<8} {'Outcome':<10} {'PNL':<12} {'ROI':<10}")
    print("-" * TABLE_WIDTH)
    
    for i, trade in enumerate(trades, 1):
        timestamp = datetime.fromisoformat(trade['timestamp']).strftime('%Y-%m-%d %H:%M')
        direction = trade['direction']
        size = f"${trade['size_usd']:.2f}"
        price = f"${trade['price']:.4f}"
        model = f"{trade.get('model_probability', trade.get('signal_confidence', 0)):.1%}"
        edge = f"{trade.get('edge', 0):.1%}" if trade.get('edge') is not None else "N/A"
        kelly = f"{trade.get('kelly_fraction', 0):.1%}" if trade.get('kelly_fraction') is not None else "N/A"
        outcome = trade.get('outcome', 'PENDING')
        pnl = _format_trade_pnl(_calculate_trade_pnl(trade))
        roi = _format_trade_roi(_calculate_trade_roi(trade))
        
        print(f"{i:<4} {timestamp:<20} {direction:<8} {size:<10} {price:<10} {model:<8} {edge:<8} {kelly:<8} {outcome:<10} {pnl:<12} {roi:<10}")
    
    print("-" * TABLE_WIDTH)
    print()


def print_bot_summary(trades):
    total_invested, total_pnl, roi = calculate_bot_summary(trades)
    print(
        f"Bot Summary: Settled Invested ${total_invested:.2f} | "
        f"Total P&L {_format_signed_usd(total_pnl)} | ROI {roi:+.2f}%"
    )


def main():
    """Main entry point."""
    trades = load_paper_trades()
    display_paper_trades(trades)
    
    if trades:
        print_bot_summary(trades)
        print()
        print("NOTE: These are SIMULATION trades only - no real money involved!")
        print("Pending outcomes are settled automatically after market resolution.")


if __name__ == "__main__":
    main()
