"""
Local trade journal for dry-run and live Markov decisions.

The journal is append-only JSONL by design: each line is one sanitized event,
safe to feed into a later review loop without exposing credentials.
"""
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger


class TradeJournal:
    """Append and summarize sanitized trade records."""

    def __init__(self, path: str = "trade_journal.jsonl"):
        self.path = Path(path)

    def append(self, event: Dict[str, Any]) -> None:
        payload = self._sanitize(event)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append trade journal event: {e}")

    def recent(self, limit: int = 200) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in lines[-limit:] if line.strip()]
            return rows
        except Exception as e:
            logger.warning(f"Failed to read trade journal: {e}")
            return []

    def summarize(self, limit: int = 200) -> Dict[str, Any]:
        rows = [r for r in self.recent(limit) if r.get("event") in {"paper_trade", "live_order"}]
        completed = [r for r in rows if r.get("pnl_usd") is not None]
        wins = [r for r in completed if float(r.get("pnl_usd", 0)) > 0]
        losses = [r for r in completed if float(r.get("pnl_usd", 0)) < 0]
        total_pnl = sum(float(r.get("pnl_usd", 0)) for r in completed)

        by_state: Dict[str, Dict[str, Any]] = {}
        for row in completed:
            state = row.get("markov", {}).get("current_state", "unknown")
            bucket = by_state.setdefault(state, {"trades": 0, "wins": 0, "pnl_usd": 0.0})
            bucket["trades"] += 1
            bucket["wins"] += 1 if float(row.get("pnl_usd", 0)) > 0 else 0
            bucket["pnl_usd"] += float(row.get("pnl_usd", 0))

        for bucket in by_state.values():
            trades = bucket["trades"]
            bucket["win_rate"] = bucket["wins"] / trades if trades else 0.0

        return {
            "journal_path": str(self.path),
            "events": len(rows),
            "completed_trades": len(completed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(completed) if completed else 0.0,
            "pnl_usd": round(total_pnl, 4),
            "by_state": by_state,
        }

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(k): self._sanitize(v)
                for k, v in value.items()
                if not self._is_secret_key(str(k))
            }
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize(v) for v in value]
        return value

    def _is_secret_key(self, key: str) -> bool:
        lowered = key.lower()
        secret_markers: Iterable[str] = (
            "private_key",
            "secret",
            "passphrase",
            "api_key",
            "password",
            "pk",
        )
        return any(marker in lowered for marker in secret_markers)


_trade_journal_instance: Optional[TradeJournal] = None


def get_trade_journal(path: str = "trade_journal.jsonl") -> TradeJournal:
    global _trade_journal_instance
    if _trade_journal_instance is None or str(_trade_journal_instance.path) != path:
        _trade_journal_instance = TradeJournal(path)
    return _trade_journal_instance
