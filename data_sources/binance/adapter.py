"""
Binance Data Source Adapter
Fetches spot market data from Binance Global API (not Binance US).
"""
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import httpx
from loguru import logger


_GRANULARITY_TO_INTERVAL = {
    1: "1s",
    60: "1m",
    180: "3m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    7200: "2h",
    14400: "4h",
    21600: "6h",
    28800: "8h",
    43200: "12h",
    86400: "1d",
    259200: "3d",
    604800: "1w",
}


class BinanceDataSource:
    """
    Binance Global REST API data source for spot crypto price data.

    Uses the global Binance host (api.binance.com), not Binance US.

    Provides:
    - Real-time spot price
    - Order book data
    - Recent trades
    - 24h statistics
    - Historical candles
    """

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        symbol: str = "BTCUSDT",
    ):
        """
        Initialize Binance data source.

        Args:
            base_url: Binance Global API base URL. Do not use api.binance.us.
            symbol: Trading pair (default: BTCUSDT)
        """
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol.upper()
        self.session: Optional[httpx.AsyncClient] = None

        self._last_price: Optional[Decimal] = None
        self._last_update: Optional[datetime] = None

        logger.info(
            f"Initialized Binance Global data source for {self.symbol} "
            f"via {self.base_url}"
        )

    async def connect(self) -> bool:
        """
        Connect to Binance Global API.

        Returns:
            True if connection successful
        """
        try:
            self.session = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                headers={
                    "User-Agent": "PolymarketBot/1.0",
                    "Accept": "application/json",
                },
            )

            response = await self.session.get(
                "/api/v3/exchangeInfo",
                params={"symbol": self.symbol},
            )
            response.raise_for_status()

            logger.info("Connected to Binance Global API")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Binance Global API: {e}")
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Close connection."""
        if self.session:
            await self.session.aclose()
            self.session = None
            logger.info("Disconnected from Binance Global API")

    async def get_current_price(self) -> Optional[Decimal]:
        """
        Get current spot price.

        Returns:
            Current price or None if error
        """
        try:
            response = await self.session.get(
                "/api/v3/ticker/price",
                params={"symbol": self.symbol},
            )
            response.raise_for_status()

            data = response.json()
            price = Decimal(str(data["price"]))

            self._last_price = price
            self._last_update = datetime.now()

            logger.debug(f"Binance {self.symbol} price: ${price:,.2f}")
            return price

        except Exception as e:
            logger.error(f"Error fetching Binance price: {e}")
            return None

    async def get_order_book(self, level: int = 2) -> Optional[Dict[str, Any]]:
        """
        Get order book data.

        Args:
            level: Coinbase-compatible level. 1=best levels, 2=top 100, 3=top 1000.
                Binance depth limits (5, 10, 20, 50, 100, 500, 1000, 5000) are
                also accepted directly.

        Returns:
            Order book dict with bids and asks
        """
        try:
            limit = self._depth_limit(level)
            response = await self.session.get(
                "/api/v3/depth",
                params={"symbol": self.symbol, "limit": limit},
            )
            response.raise_for_status()

            data = response.json()

            return {
                "timestamp": datetime.now(),
                "last_update_id": data.get("lastUpdateId"),
                "bids": [
                    {
                        "price": Decimal(str(b[0])),
                        "size": Decimal(str(b[1])),
                        "quantity": Decimal(str(b[1])),
                    }
                    for b in data.get("bids", [])
                ],
                "asks": [
                    {
                        "price": Decimal(str(a[0])),
                        "size": Decimal(str(a[1])),
                        "quantity": Decimal(str(a[1])),
                    }
                    for a in data.get("asks", [])
                ],
            }

        except Exception as e:
            logger.error(f"Error fetching Binance order book: {e}")
            return None

    async def get_24h_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get 24-hour statistics.

        Returns:
            Dict with open, high, low, volume, etc.
        """
        try:
            response = await self.session.get(
                "/api/v3/ticker/24hr",
                params={"symbol": self.symbol},
            )
            response.raise_for_status()

            data = response.json()

            return {
                "timestamp": datetime.now(),
                "open": Decimal(str(data["openPrice"])),
                "high": Decimal(str(data["highPrice"])),
                "low": Decimal(str(data["lowPrice"])),
                "volume": Decimal(str(data["volume"])),
                "quote_volume": Decimal(str(data["quoteVolume"])),
                "last": Decimal(str(data["lastPrice"])),
                "price_change": Decimal(str(data["priceChange"])),
                "price_change_percent": Decimal(str(data["priceChangePercent"])),
            }

        except Exception as e:
            logger.error(f"Error fetching Binance 24h stats: {e}")
            return None

    async def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trades.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of recent trades
        """
        try:
            response = await self.session.get(
                "/api/v3/trades",
                params={"symbol": self.symbol, "limit": min(limit, 1000)},
            )
            response.raise_for_status()

            data = response.json()

            trades = []
            for trade in data[:limit]:
                buyer_is_maker = bool(trade["isBuyerMaker"])
                size = Decimal(str(trade["qty"]))
                trades.append({
                    "timestamp": datetime.fromtimestamp(trade["time"] / 1000),
                    "trade_id": trade["id"],
                    "price": Decimal(str(trade["price"])),
                    "size": size,
                    "quantity": size,
                    "buyer_is_maker": buyer_is_maker,
                    "side": "sell" if buyer_is_maker else "buy",
                })

            return trades

        except Exception as e:
            logger.error(f"Error fetching Binance trades: {e}")
            return []

    async def get_candles(
        self,
        granularity: Union[int, str] = 300,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get historical candles (OHLCV data).

        Args:
            granularity: Candle size in seconds, or a Binance interval string.
            limit: Number of candles to return

        Returns:
            List of candle data ordered oldest to newest
        """
        try:
            interval = self._interval_from_granularity(granularity)
            response = await self.session.get(
                "/api/v3/klines",
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "limit": min(limit, 1000),
                },
            )
            response.raise_for_status()

            data = response.json()

            candles = []
            for candle in data[:limit]:
                candles.append({
                    "timestamp": datetime.fromtimestamp(candle[0] / 1000),
                    "open": Decimal(str(candle[1])),
                    "high": Decimal(str(candle[2])),
                    "low": Decimal(str(candle[3])),
                    "close": Decimal(str(candle[4])),
                    "volume": Decimal(str(candle[5])),
                    "close_time": datetime.fromtimestamp(candle[6] / 1000),
                    "quote_volume": Decimal(str(candle[7])),
                    "trade_count": int(candle[8]),
                })

            return candles

        except Exception as e:
            logger.error(f"Error fetching Binance candles: {e}")
            return []

    @property
    def last_price(self) -> Optional[Decimal]:
        """Get cached last price."""
        return self._last_price

    @property
    def last_update(self) -> Optional[datetime]:
        """Get time of last price update."""
        return self._last_update

    async def health_check(self) -> bool:
        """
        Check if data source is healthy.

        Returns:
            True if healthy
        """
        try:
            price = await self.get_current_price()
            return price is not None
        except Exception:
            return False

    def _interval_from_granularity(self, granularity: Union[int, str]) -> str:
        if isinstance(granularity, str):
            return granularity

        interval = _GRANULARITY_TO_INTERVAL.get(granularity)
        if not interval:
            supported = ", ".join(str(g) for g in sorted(_GRANULARITY_TO_INTERVAL))
            raise ValueError(
                f"Unsupported Binance candle granularity {granularity}. "
                f"Supported second values: {supported}"
            )
        return interval

    def _depth_limit(self, level: int) -> int:
        if level in {5, 10, 20, 50, 100, 500, 1000, 5000}:
            return level
        if level == 1:
            return 5
        if level == 2:
            return 100
        return 1000


_binance_data_instance: Optional[BinanceDataSource] = None


def get_binance_data_source() -> BinanceDataSource:
    """Get singleton instance of Binance Global REST data source."""
    global _binance_data_instance
    if _binance_data_instance is None:
        _binance_data_instance = BinanceDataSource()
    return _binance_data_instance
