import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


def parse_utc_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        text = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_paper_trade_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        rows = json.loads(text)
        return rows if isinstance(rows, list) else [rows]
    except json.JSONDecodeError:
        return [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]


def write_paper_trade_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _trade_key(trade: Dict[str, Any]) -> tuple:
    trade_id = trade.get("trade_id")
    if trade_id:
        return ("trade_id", trade_id)

    return (
        "legacy",
        trade.get("timestamp"),
        trade.get("direction"),
        trade.get("size_usd"),
        trade.get("price"),
        trade.get("market_slug"),
    )


def upsert_paper_trade(path: Path, trade_data: Dict[str, Any]) -> None:
    rows = load_paper_trade_rows(path)
    incoming_key = _trade_key(trade_data)
    updated = False

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = _trade_key(row)
        if key == incoming_key:
            deduped.append(trade_data)
            seen.add(key)
            updated = True
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    if not updated:
        deduped.append(trade_data)

    write_paper_trade_rows(path, deduped)


def _json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _normalize_market_result(label: Any) -> Optional[str]:
    text = str(label or "").strip().lower()
    if text in {"up", "yes", "y"}:
        return "UP"
    if text in {"down", "no", "n"}:
        return "DOWN"
    return None


def _infer_market_result(market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    outcomes = _json_list(market.get("outcomes"))
    prices = _json_list(market.get("outcomePrices"))
    events = market.get("events") if isinstance(market.get("events"), list) else []
    metadata = {}
    if events and isinstance(events[0], dict):
        metadata = events[0].get("eventMetadata") or {}
    if not metadata:
        metadata = market.get("eventMetadata") or {}

    start_price = None
    end_price = None
    try:
        end_price = float(metadata["finalPrice"])
        start_price = float(metadata["priceToBeat"])
    except (KeyError, TypeError, ValueError):
        pass

    parsed_prices: List[float] = []
    for price in prices:
        try:
            parsed_prices.append(float(price))
        except (TypeError, ValueError):
            parsed_prices.append(0.0)

    if outcomes and parsed_prices and len(outcomes) == len(parsed_prices):
        winner_index = max(range(len(parsed_prices)), key=lambda i: parsed_prices[i])
        if parsed_prices[winner_index] >= 0.99:
            market_result = _normalize_market_result(outcomes[winner_index])
            if market_result:
                return {
                    "market_result": market_result,
                    "settlement_source": "gamma_outcome_prices",
                    "start_price": start_price,
                    "end_price": end_price,
                }

    if start_price is None or end_price is None:
        return None

    return {
        "market_result": "UP" if end_price >= start_price else "DOWN",
        "settlement_source": "gamma_event_metadata",
        "start_price": start_price,
        "end_price": end_price,
    }


def _market_slug(trade: Dict[str, Any]) -> Optional[str]:
    slug = trade.get("market_slug")
    if slug:
        return str(slug)

    market = trade.get("market")
    if isinstance(market, dict) and market.get("slug"):
        return str(market["slug"])

    return None


def _market_end_time(trade: Dict[str, Any], interval_seconds: int) -> Optional[datetime]:
    for value in (
        trade.get("market_end_time"),
        trade.get("market", {}).get("end_time") if isinstance(trade.get("market"), dict) else None,
    ):
        dt = parse_utc_datetime(value)
        if dt:
            return dt

    timestamp = trade.get("market_timestamp")
    if timestamp is None:
        slug = _market_slug(trade)
        if slug:
            try:
                timestamp = int(slug.split("-")[-1])
            except (TypeError, ValueError):
                timestamp = None

    if timestamp is None:
        return None

    try:
        return datetime.fromtimestamp(int(timestamp) + interval_seconds, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _selected_token_won(selected_token: Any, market_result: str) -> bool:
    token = str(selected_token or "").strip().upper()
    if token in {"YES", "UP", "LONG"}:
        return market_result == "UP"
    if token in {"NO", "DOWN", "SHORT"}:
        return market_result == "DOWN"
    return False


def _decimal_from_trade(trade: Dict[str, Any], *keys: str) -> Decimal:
    for key in keys:
        value = trade.get(key)
        if value is not None:
            try:
                return Decimal(str(value))
            except Exception:
                continue
    return Decimal("0")


def _settle_trade(
    trade: Dict[str, Any],
    resolution: Dict[str, Any],
    settled_at: datetime,
) -> Optional[Dict[str, Any]]:
    market_result = resolution["market_result"]
    won = _selected_token_won(trade.get("direction"), market_result)
    exit_price = Decimal("1.0") if won else Decimal("0.0")
    entry_price = _decimal_from_trade(trade, "price", "entry_price")
    size_usd = _decimal_from_trade(trade, "size_usd")

    if entry_price <= 0 or size_usd <= 0:
        return None

    pnl = size_usd * (exit_price - entry_price) / entry_price

    updated = dict(trade)
    updated["outcome"] = "WIN" if won else "LOSS"
    updated["exit_price"] = float(exit_price)
    updated["pnl_usd"] = float(pnl)
    updated["settlement_time"] = settled_at.isoformat()
    updated["settlement_source"] = resolution.get("settlement_source")
    updated["market_result"] = market_result
    if resolution.get("start_price") is not None:
        updated["start_price"] = resolution["start_price"]
    if resolution.get("end_price") is not None:
        updated["end_price"] = resolution["end_price"]
    return updated


async def _fetch_gamma_market(api_base_url: str, slug: str) -> Optional[Dict[str, Any]]:
    base_url = api_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{base_url}/markets/keyset",
            params={"closed": "true", "slug": slug},
        )
        response.raise_for_status()
        payload = response.json()
        markets = payload.get("markets", []) if isinstance(payload, dict) else payload

        if not markets:
            response = await client.get(
                f"{base_url}/markets",
                params={"closed": "true", "slug": slug},
            )
            response.raise_for_status()
            payload = response.json()
            markets = payload.get("markets", []) if isinstance(payload, dict) else payload

    if not isinstance(markets, list):
        return None

    for market in markets:
        if isinstance(market, dict) and market.get("slug") == slug:
            return market
    return None


async def settle_pending_paper_trades(
    path: Path,
    market_interval_seconds: int,
    gamma_api_base_url: str,
    settlement_delay_seconds: int,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    rows = load_paper_trade_rows(path)
    if not rows:
        return []

    now = now or datetime.now(timezone.utc)
    changed = False
    settled: List[Dict[str, Any]] = []
    resolution_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    updated_rows: List[Dict[str, Any]] = []

    for row in rows:
        if str(row.get("outcome", "PENDING")).upper() != "PENDING":
            updated_rows.append(row)
            continue

        end_time = _market_end_time(row, market_interval_seconds)
        slug = _market_slug(row)
        if not end_time or not slug:
            updated_rows.append(row)
            continue

        settle_after = end_time + timedelta(seconds=settlement_delay_seconds)
        if now < settle_after:
            updated_rows.append(row)
            continue

        if slug not in resolution_cache:
            market = await _fetch_gamma_market(gamma_api_base_url, slug)
            resolution_cache[slug] = _infer_market_result(market) if market else None

        resolution = resolution_cache[slug]
        if not resolution:
            updated_rows.append(row)
            continue

        settled_row = _settle_trade(row, resolution, now)
        if settled_row is None:
            updated_rows.append(row)
            continue

        updated_rows.append(settled_row)
        settled.append(settled_row)
        changed = True

    if changed:
        write_paper_trade_rows(path, updated_rows)

    return settled
