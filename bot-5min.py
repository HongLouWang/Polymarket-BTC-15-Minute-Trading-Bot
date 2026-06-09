import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import math
from decimal import Decimal
import time
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Dict
import random

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


try:
    from patch_gamma_markets import apply_gamma_markets_patch, verify_patch
    patch_applied = apply_gamma_markets_patch()
    if patch_applied:
        verify_patch()
    else:
        print("ERROR: Failed to apply gamma_market patch")
        sys.exit(1)
except ImportError as e:
    print(f"ERROR: Could not import patch module: {e}")
    print("Make sure patch_gamma_markets.py is in the same directory")
    sys.exit(1)

# Now import Nautilus
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LiveDataEngineConfig,
    LiveExecEngineConfig,
    LiveRiskEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.adapters.polymarket import POLYMARKET
from nautilus_trader.adapters.polymarket import (
    PolymarketDataClientConfig,
    PolymarketExecClientConfig,
)
from nautilus_trader.adapters.polymarket.factories import (
    PolymarketLiveDataClientFactory,
    PolymarketLiveExecClientFactory,
)
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId, ClientOrderId
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.data import QuoteTick

from dotenv import load_dotenv
from loguru import logger
import redis

# Import our phases
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.deribit_pcr_processor import DeribitPCRProcessor
from core.strategy_brain.signal_processors.markov_persistence_processor import MarkovPersistenceProcessor
from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine
from execution.risk_engine import get_risk_engine
from monitoring.performance_tracker import get_performance_tracker
from monitoring.grafana_exporter import get_grafana_exporter
from feedback.learning_engine import get_learning_engine
from feedback.trade_journal import get_trade_journal
from paper_trade_settlement import (
    parse_utc_datetime,
    settle_pending_paper_trades,
    upsert_paper_trade,
)
load_dotenv()
from patch_market_orders import apply_market_order_patch
patch_applied = apply_market_order_patch()
if patch_applied:
    logger.info("Market order patch applied successfully")
else:
    logger.warning("Market order patch failed - orders may be rejected")


# =============================================================================
# CONSTANTS
# =============================================================================
QUOTE_STABILITY_REQUIRED = 3      # Need only 3 valid ticks to be stable (faster startup)
QUOTE_MIN_SPREAD = 0.001          # Both bid AND ask must be at least this
MARKET_INTERVAL_SECONDS = int(os.getenv("MARKET_5M_INTERVAL_SECONDS", "300"))
MARKET_SLUG_PREFIX = os.getenv("MARKET_5M_SLUG_PREFIX", "btc-updown-5m").lower()
MARKET_LABEL = "5-MIN"
TRADE_JOURNAL_PATH = os.getenv("TRADE_JOURNAL_5M_PATH", "trade_journal_5min.jsonl")
PAPER_TRADES_PATH = os.getenv("PAPER_TRADES_5M_PATH", "paper_trades_5min.json")
SPOT_SYMBOL = os.getenv("BINANCE_SPOT_SYMBOL", "BTCUSDT").upper()

# Local strategy/test configuration. Keep these in code so threshold changes are
# explicit and not dependent on shell or system environment variables.
DRY_RUN = True
MIN_EDGE = 0.0
MIN_PROB = 0.2
EDGE_CHECK_INTERVAL_SECONDS = 1
MIN_BET = Decimal("1.00")
MAX_BET = Decimal("50.00")
BANKROLL = Decimal("100.00")
KELLY_SCALE = 1.00
FEE_BPS = 0.0
MARKET_WARMUP_SECONDS = 5
MARKET_CLOSE_BUFFER_SECONDS = 1
MAX_TRADES_PER_MARKET = 3
REQUIRE_FUSION_CONFIRMATION = False
MAX_TOTAL_EXPOSURE = BANKROLL
PAPER_SETTLEMENT_DELAY_SECONDS = int(os.getenv("PAPER_SETTLEMENT_DELAY_SECONDS", "20"))
PAPER_SETTLEMENT_CHECK_SECONDS = int(os.getenv("PAPER_SETTLEMENT_CHECK_SECONDS", "10"))
GAMMA_API_BASE_URL = os.getenv("GAMMA_API_BASE_URL", "https://gamma-api.polymarket.com")

# Lowered Markov sample requirements for dry-run system testing.
MARKOV_MIN_TRANSITIONS = 100
MARKOV_MIN_STATE_OBSERVATIONS = 10
MARKOV_RETURN_THRESHOLD = 0.0001
MARKOV_STRONG_RETURN_THRESHOLD = 0.0005
MARKOV_PROBABILITY_RETURN_THRESHOLD = 0.005
MARKOV_STRONG_PROBABILITY_RETURN_THRESHOLD = 0.015


def polymarket_private_key() -> Optional[str]:
    return os.getenv("POLYMARKET_PK") or os.getenv("PRIVATE_KEY")


def polymarket_funder_address() -> Optional[str]:
    return (
        os.getenv("POLYMARKET_ADDRESS")
        or os.getenv("SAFE_ADDRESS")
        or os.getenv("POLYMARKET_FUNDER")
    )


def clob_host() -> str:
    return os.getenv("CLOB_HOST", "https://clob.polymarket.com")


def polymarket_signature_type() -> int:
    return int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))


@dataclass
class PaperTrade:
    """Track paper/simulation trades"""
    timestamp: datetime
    direction: str
    size_usd: float
    price: float
    signal_score: float
    signal_confidence: float
    outcome: str = "PENDING"
    trade_id: Optional[str] = None
    model_probability: Optional[float] = None
    market_probability: Optional[float] = None
    edge: Optional[float] = None
    kelly_fraction: Optional[float] = None
    market_slug: Optional[str] = None
    market_timestamp: Optional[int] = None
    market_end_time: Optional[str] = None
    yes_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl_usd: Optional[float] = None
    settlement_time: Optional[str] = None
    settlement_source: Optional[str] = None
    market_result: Optional[str] = None
    start_price: Optional[float] = None
    end_price: Optional[float] = None

    def to_dict(self):
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'direction': self.direction,
            'size_usd': self.size_usd,
            'price': self.price,
            'signal_score': self.signal_score,
            'signal_confidence': self.signal_confidence,
            'outcome': self.outcome,
            'model_probability': self.model_probability,
            'market_probability': self.market_probability,
            'edge': self.edge,
            'kelly_fraction': self.kelly_fraction,
            'market_slug': self.market_slug,
            'market_timestamp': self.market_timestamp,
            'market_end_time': self.market_end_time,
            'yes_price': self.yes_price,
            'exit_price': self.exit_price,
            'pnl_usd': self.pnl_usd,
            'settlement_time': self.settlement_time,
            'settlement_source': self.settlement_source,
            'market_result': self.market_result,
            'start_price': self.start_price,
            'end_price': self.end_price,
        }


def init_redis():
    """Initialize Redis connection for simulation mode control."""
    try:
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 2)),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True
        )
        redis_client.ping()
        logger.info("Redis connection established")
        return redis_client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        logger.warning("Simulation mode will be static (from .env)")
        return None


class IntegratedBTCStrategy(Strategy):
    """
    Integrated BTC Strategy - FIXED VERSION
    - Subscribes immediately at startup
    - Forces stability for first trade
    - Correct timing for market switching
    """

    def __init__(self, redis_client=None, enable_grafana=True, test_mode=False):
        super().__init__()

        self.bot_start_time = datetime.now(timezone.utc)
        self.restart_after_minutes = 90

        # Nautilus
        self.instrument_id = None
        self.redis_client = redis_client
        self.current_simulation_mode = False

        # Store ALL BTC instruments
        self.all_btc_instruments: List[Dict] = []
        self.current_instrument_index: int = -1
        self.next_switch_time: Optional[datetime] = None

        # Quote-stability tracking
        self._stable_tick_count = 0
        self._market_stable = False
        self._last_instrument_switch = None
        
        # Markov/Kelly configuration is intentionally local to bot-5min.py for
        # repeatable dry-run testing.
        self.dry_run = DRY_RUN
        self.min_edge = MIN_EDGE
        self.min_prob = MIN_PROB
        self.min_bet = MIN_BET
        self.max_bet = MAX_BET
        self.bankroll = BANKROLL
        self.kelly_scale = KELLY_SCALE
        self.fee_bps = FEE_BPS
        self.edge_check_interval_seconds = EDGE_CHECK_INTERVAL_SECONDS
        self.market_warmup_seconds = MARKET_WARMUP_SECONDS
        self.market_close_buffer_seconds = MARKET_CLOSE_BUFFER_SECONDS
        self.max_trades_per_market = MAX_TRADES_PER_MARKET
        self.require_fusion_confirmation = REQUIRE_FUSION_CONFIRMATION
        self.trade_journal = get_trade_journal(TRADE_JOURNAL_PATH)

        self.last_trade_time = -1
        self._last_edge_check_at: Optional[datetime] = None
        self._decision_in_flight = False
        self._market_trade_counts: Dict[int, int] = {}
        self._waiting_for_market_open = False  # True when waiting for a future market to open
        self._last_bid_ask = None  # (bid_decimal, ask_decimal) from last tick, for liquidity checks
        self._last_paper_settlement_check: Optional[datetime] = None

        # Tick buffer: rolling 90s of ticks for TickVelocityProcessor
        from collections import deque
        self._tick_buffer: deque = deque(maxlen=500)  # ~500 ticks = well over 90s
        self._spot_price_history: deque = deque(maxlen=500)
        self._spot_candles_cache: List[Dict] = []
        self._spot_candles_cache_time: Optional[datetime] = None

        # YES token id for the current market (set in _load_all_btc_instruments)
        self._yes_token_id: Optional[str] = None

        # Phase 4: Signal Processors
        self.spike_detector = SpikeDetectionProcessor(
            spike_threshold=0.05,       # FIXED: was 0.15 (too high for probabilities)
            lookback_periods=20,
        )
        self.sentiment_processor = SentimentProcessor(
            extreme_fear_threshold=25,
            extreme_greed_threshold=75,
        )
        self.divergence_processor = PriceDivergenceProcessor(
            divergence_threshold=0.05,
        )
        self.orderbook_processor = OrderBookImbalanceProcessor(
            imbalance_threshold=0.30,   # 30% skew to signal
            min_book_volume=50.0,       # ignore illiquid books
        )
        self.tick_velocity_processor = TickVelocityProcessor(
            velocity_threshold_60s=0.015,  # 1.5% move in 60s
            velocity_threshold_30s=0.010,  # 1.0% move in 30s
        )
        self.deribit_pcr_processor = DeribitPCRProcessor(
            bullish_pcr_threshold=1.20,
            bearish_pcr_threshold=0.70,
            max_days_to_expiry=2,
            cache_seconds=300,          # refresh every 5 min
        )
        self.markov_processor = MarkovPersistenceProcessor(
            min_probability=self.min_prob,
            min_edge=self.min_edge,
            fee_bps=self.fee_bps,
            bankroll=self.bankroll,
            min_bet=self.min_bet,
            max_bet=self.max_bet,
            kelly_scale=self.kelly_scale,
            min_transitions=MARKOV_MIN_TRANSITIONS,
            min_state_observations=MARKOV_MIN_STATE_OBSERVATIONS,
            return_threshold=MARKOV_RETURN_THRESHOLD,
            strong_return_threshold=MARKOV_STRONG_RETURN_THRESHOLD,
            probability_return_threshold=MARKOV_PROBABILITY_RETURN_THRESHOLD,
            strong_probability_return_threshold=MARKOV_STRONG_PROBABILITY_RETURN_THRESHOLD,
        )

        # Phase 4: Signal Fusion. Markov is the entry gate; other signals are context.
        self.fusion_engine = get_fusion_engine()
        self.fusion_engine.set_weight("MarkovPersistence",  0.45)
        self.fusion_engine.set_weight("OrderBookImbalance", 0.18)
        self.fusion_engine.set_weight("TickVelocity",       0.15)
        self.fusion_engine.set_weight("PriceDivergence",    0.10)
        self.fusion_engine.set_weight("SpikeDetection",     0.06)
        self.fusion_engine.set_weight("DeribitPCR",         0.04)
        self.fusion_engine.set_weight("SentimentAnalysis",  0.02)

        # Phase 5: Risk Management
        self.risk_engine = get_risk_engine()
        self.risk_engine.limits.max_position_size = self.max_bet
        self.risk_engine.limits.max_total_exposure = MAX_TOTAL_EXPOSURE
        self.risk_engine.min_position_size = self.min_bet

        # Phase 6: Performance Tracking
        self.performance_tracker = get_performance_tracker()

        # Phase 7: Learning Engine
        self.learning_engine = get_learning_engine()

        # Phase 6: Grafana (optional)
        if enable_grafana:
            self.grafana_exporter = get_grafana_exporter()
        else:
            self.grafana_exporter = None

        # Price history
        self.price_history = []
        self.max_history = 100

        # Paper trading tracker
        self.paper_trades: List[PaperTrade] = []

        self.test_mode = test_mode

        if test_mode:
            logger.info("=" * 80)
            logger.info("  TEST MODE ACTIVE - Trading every minute!")
            logger.info("=" * 80)

        logger.info("=" * 80)
        logger.info("INTEGRATED BTC STRATEGY INITIALIZED - FIXED VERSION")
        logger.info("  Phase 4: Signal processors ready")
        logger.info("  Phase 5: Risk engine ready")
        logger.info("  Phase 6: Performance tracking ready")
        logger.info("  Phase 7: Learning engine ready")
        logger.info(f"  DRY_RUN: {self.dry_run}")
        logger.info(f"  Markov gate: p >= {self.min_prob:.0%}, edge >= {self.min_edge:.0%}")
        logger.info(f"  Kelly sizing: bankroll=${self.bankroll:.2f}, bet=${self.min_bet:.2f}-${self.max_bet:.2f}")
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seconds_to_next_market_boundary(self) -> float:
        """Return seconds until the next market interval boundary."""
        now_ts = datetime.now(timezone.utc).timestamp()
        next_boundary = (math.floor(now_ts / MARKET_INTERVAL_SECONDS) + 1) * MARKET_INTERVAL_SECONDS
        return next_boundary - now_ts

    def _is_quote_valid(self, bid, ask) -> bool:
        """Return True only when BOTH bid and ask are present and make sense."""
        if bid is None or ask is None:
            return False
        try:
            b = float(bid)
            a = float(ask)
        except (TypeError, ValueError):
            return False
        if b < QUOTE_MIN_SPREAD or a < QUOTE_MIN_SPREAD:
            return False
        if b > 0.999 or a > 0.999:
            return False
        return True

    def _reset_stability(self, reason: str = ""):
        """Mark the market as unstable and reset the counter."""
        if self._market_stable:
            logger.warning(f"Market stability RESET{' – ' + reason if reason else ''}")
        self._market_stable = False
        self._stable_tick_count = 0

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------

    async def check_simulation_mode(self) -> bool:
        """Check Redis for current simulation mode."""
        if self.dry_run:
            self.current_simulation_mode = True
            return True
        if not self.redis_client:
            return self.current_simulation_mode
        try:
            sim_mode = self.redis_client.get('btc_trading:simulation_mode')
            if sim_mode is not None:
                redis_simulation = sim_mode == '1'
                if redis_simulation != self.current_simulation_mode:
                    self.current_simulation_mode = redis_simulation
                    mode_text = "SIMULATION" if redis_simulation else "LIVE TRADING"
                    logger.warning(f"Trading mode changed to: {mode_text}")
                    if not redis_simulation:
                        logger.warning("LIVE TRADING ACTIVE - Real money at risk!")
                return redis_simulation
        except Exception as e:
            logger.warning(f"Failed to check Redis simulation mode: {e}")
        return self.current_simulation_mode

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def on_start(self):
        """Called when strategy starts - LOAD ALL MARKETS AND SUBSCRIBE IMMEDIATELY"""
        logger.info("=" * 80)
        logger.info("INTEGRATED BTC STRATEGY STARTED - FIXED VERSION")
        logger.info("=" * 80)

        # =========================================================================
        # FIX 2: Load ALL BTC instruments at startup
        # =========================================================================
        self._load_all_btc_instruments()

        # =========================================================================
        # FIX 3: Force subscribe to current market IMMEDIATELY
        # =========================================================================
        if self.instrument_id:
            self.subscribe_quote_ticks(self.instrument_id)
            logger.info(f"✓ SUBSCRIBED to market: {self.instrument_id}")
            
            # Try to get current price from cache
            try:
                quote = self.cache.quote_tick(self.instrument_id)
                if quote and quote.bid_price and quote.ask_price:
                    current_price = (quote.bid_price + quote.ask_price) / 2
                    self.price_history.append(current_price)
                    logger.info(f"✓ Initial price: ${float(current_price):.4f}")
            except Exception as e:
                logger.debug(f"No initial price yet: {e}")

        # Generate synthetic history if needed
        if len(self.price_history) < 20:
            self._generate_synthetic_history(target_count=20, existing_count=len(self.price_history))

        # =========================================================================
        # FIX 4: Start the timer loop (but don't rely on it for trading)
        # =========================================================================
        self.run_in_executor(self._start_timer_loop)

        if self.grafana_exporter:
            import threading
            threading.Thread(target=self._start_grafana_sync, daemon=True).start()

        logger.info("=" * 80)
        logger.info(f"Strategy active - will trade every {MARKET_INTERVAL_SECONDS // 60} minutes")
        logger.info(f"Price history: {len(self.price_history)} points")
        if len(self.price_history) >= 20:
            logger.info("✓ READY TO TRADE NOW!")
        else:
            logger.warning(f"⚠ Need more history ({len(self.price_history)}/20)")
        logger.info("=" * 80)

    def _generate_synthetic_history(self, target_count: int = 20, existing_count: int = 0):
        """Generate synthetic price history for testing"""
        if self.price_history:
            base_price = self.price_history[-1]
        else:
            base_price = Decimal("0.5")
        needed = target_count - existing_count
        if needed <= 0:
            return
        for _ in range(needed):
            change = Decimal(str(random.uniform(-0.03, 0.03)))
            new_price = base_price * (Decimal("1.0") + change)
            new_price = max(Decimal("0.01"), min(Decimal("0.99"), new_price))
            self.price_history.append(new_price)
            base_price = new_price

    # ------------------------------------------------------------------
    # Load all BTC instruments at once
    # ------------------------------------------------------------------

    def _load_all_btc_instruments(self):
        """Load ALL BTC instruments from cache and sort by start time"""
        instruments = self.cache.instruments()
        logger.info(f"Loading ALL BTC instruments from {len(instruments)} total...")
        
        now = datetime.now(timezone.utc)
        current_timestamp = int(now.timestamp())
        
        btc_instruments = []
        
        for instrument in instruments:
            try:
                if hasattr(instrument, 'info') and instrument.info:
                    question = instrument.info.get('question', '').lower()
                    slug = instrument.info.get('market_slug', '').lower()
                    
                    if ('btc' in question or 'btc' in slug) and MARKET_SLUG_PREFIX in slug:
                        try:
                            timestamp_part = slug.split('-')[-1]
                            market_timestamp = int(timestamp_part)
                            
                            # The slug timestamp IS the market start time (Unix, no offset).
                            # end_date_iso is a DATE-only string (e.g. "2026-02-20"), NOT a datetime,
                            # so parsing it gives midnight UTC which is wrong for intraday markets.
                            # Always derive end_timestamp from the slug: start + market interval.
                            real_start_ts = market_timestamp
                            end_timestamp = market_timestamp + MARKET_INTERVAL_SECONDS
                            time_diff = real_start_ts - current_timestamp
                            
                            # Only include markets that haven't ended yet
                            if end_timestamp > current_timestamp:
                                # Extract YES token ID for CLOB order book API.
                                # Nautilus instrument ID format:
                                #   {condition_id}-{token_id}.POLYMARKET
                                # The CLOB /book endpoint only accepts the token_id
                                # (the part after the dash, before .POLYMARKET).
                                raw_id = str(instrument.id)
                                # Strip .POLYMARKET suffix first
                                without_suffix = raw_id.split('.')[0] if '.' in raw_id else raw_id
                                # Then take the token_id after the condition_id dash
                                yes_token_id = without_suffix.split('-')[-1] if '-' in without_suffix else without_suffix

                                btc_instruments.append({
                                    'instrument': instrument,
                                    'slug': slug,
                                    'start_time': datetime.fromtimestamp(real_start_ts, tz=timezone.utc),
                                    'end_time': datetime.fromtimestamp(end_timestamp, tz=timezone.utc),
                                    'market_timestamp': market_timestamp,
                                    'end_timestamp': end_timestamp,
                                    'time_diff_minutes': time_diff / 60,
                                    'yes_token_id': yes_token_id,
                                })
                        except (ValueError, IndexError):
                            continue
            except Exception:
                continue
        
        # Pair YES and NO tokens by slug.
        # Each Polymarket market has two tokens loaded as separate Nautilus instruments.
        # The first instrument found for a slug is stored as the primary (YES/UP).
        # The second instrument found for the same slug is the NO/DOWN token.
        seen_slugs = {}
        deduped = []
        for inst in btc_instruments:
            slug = inst['slug']
            if slug not in seen_slugs:
                # First token seen = YES (UP)
                inst['yes_instrument_id'] = inst['instrument'].id
                inst['no_instrument_id'] = None  # will be filled when second token found
                seen_slugs[slug] = inst
                deduped.append(inst)
            else:
                # Second token seen = NO (DOWN); store it on the existing entry.
                seen_slugs[slug]['no_instrument_id'] = inst['instrument'].id
        btc_instruments = deduped
        
        # Sort by start time (absolute timestamp, not time-of-day)
        btc_instruments.sort(key=lambda x: x['market_timestamp'])
        
        logger.info("=" * 80)
        logger.info(f"FOUND {len(btc_instruments)} BTC {MARKET_LABEL} MARKETS:")
        for i, inst in enumerate(btc_instruments):
            # A market is ACTIVE if it has started AND not yet ended
            is_active = inst['time_diff_minutes'] <= 0 and inst['end_timestamp'] > current_timestamp
            status = "ACTIVE" if is_active else "FUTURE" if inst['time_diff_minutes'] > 0 else "PAST"
            logger.info(f"  [{i}] {inst['slug']}: {status} (starts at {inst['start_time'].strftime('%H:%M:%S')}, ends at {inst['end_time'].strftime('%H:%M:%S')})")
        logger.info("=" * 80)
        
        self.all_btc_instruments = btc_instruments
        
        # Find current market and SUBSCRIBE IMMEDIATELY
        # FIXED: A market is current if it has STARTED and not yet ENDED (use end_time, not a hardcoded interval window)
        for i, inst in enumerate(btc_instruments):
            is_active = inst['time_diff_minutes'] <= 0 and inst['end_timestamp'] > current_timestamp
            if is_active:
                self.current_instrument_index = i
                self.instrument_id = inst['instrument'].id
                self.next_switch_time = inst['end_time']
                self._yes_token_id = inst.get('yes_token_id')
                self._yes_instrument_id = inst.get('yes_instrument_id', inst['instrument'].id)
                self._no_instrument_id = inst.get('no_instrument_id')
                logger.info(f"✓ CURRENT MARKET: {inst['slug']} (index {i})")
                logger.info(f"  Next switch at: {self.next_switch_time.strftime('%H:%M:%S')}")
                logger.info(f"  YES token: {self._yes_token_id[:16]}…" if self._yes_token_id else "  YES token: unknown")
                
                # =========================================================================
                # CRITICAL FIX: Subscribe immediately!
                # =========================================================================
                self.subscribe_quote_ticks(self.instrument_id)
                logger.info(f"  ✓ SUBSCRIBED to current market")
                break
        
        if self.current_instrument_index == -1 and btc_instruments:
                # No currently-active market; find the NEAREST upcoming one.
            # (smallest positive time_diff_minutes = starts soonest)
            future_markets = [inst for inst in btc_instruments if inst['time_diff_minutes'] > 0]
            if future_markets:
                nearest = min(future_markets, key=lambda x: x['time_diff_minutes'])
                nearest_idx = btc_instruments.index(nearest)
            else:
                # All markets are in the past; use the last one.
                nearest = btc_instruments[-1]
                nearest_idx = len(btc_instruments) - 1

            self.current_instrument_index = nearest_idx
            inst = nearest
            self.instrument_id = inst['instrument'].id
            self._yes_token_id = inst.get('yes_token_id')
            self._yes_instrument_id = inst.get('yes_instrument_id', inst['instrument'].id)
            self._no_instrument_id = inst.get('no_instrument_id')
            self.next_switch_time = inst['start_time']  # switch_time = when it OPENS
            logger.info(f"⚠ NO CURRENT MARKET - WAITING FOR NEAREST FUTURE: {inst['slug']}")
            logger.info(f"  Starts in {inst['time_diff_minutes']:.1f} min at {self.next_switch_time.strftime('%H:%M:%S')} UTC")

            # Subscribe so we get ticks when it opens
            self.subscribe_quote_ticks(self.instrument_id)
            logger.info(f"  ✓ SUBSCRIBED to future market")
            # Block trading until the market actually opens (timer loop sets _market_open flag)
            self._waiting_for_market_open = True
            
    def _switch_to_next_market(self):
        """Switch to the next market in the pre-loaded list"""
        if not self.all_btc_instruments:
            logger.error("No instruments loaded!")
            return False
        
        next_index = self.current_instrument_index + 1
        if next_index >= len(self.all_btc_instruments):
            logger.warning("No more markets available - will restart bot")
            return False
        
        next_market = self.all_btc_instruments[next_index]
        now = datetime.now(timezone.utc)
        
        # Check if next market is ready
        if now < next_market['start_time']:
            logger.info(f"Waiting for next market at {next_market['start_time'].strftime('%H:%M:%S')}")
            return False
        
        # Switch to next market
        self.current_instrument_index = next_index
        self.instrument_id = next_market['instrument'].id
        self.next_switch_time = next_market['end_time']
        self._yes_token_id = next_market.get('yes_token_id')
        self._yes_instrument_id = next_market.get('yes_instrument_id', next_market['instrument'].id)
        self._no_instrument_id = next_market.get('no_instrument_id')
        
        logger.info("=" * 80)
        logger.info(f"SWITCHING TO NEXT MARKET: {next_market['slug']}")
        logger.info(f"  Current time: {now.strftime('%H:%M:%S')}")
        logger.info(f"  Market ends at: {self.next_switch_time.strftime('%H:%M:%S')}")
        logger.info("=" * 80)
        
        # =========================================================================
        # FIX 5: Force stability for new market and reset trade timer correctly
        # =========================================================================
        self._stable_tick_count = QUOTE_STABILITY_REQUIRED  # Force stable immediately
        self._market_stable = True
        self._waiting_for_market_open = False  # Market is now active
        
        # Reset trade timer so we trade at the NEXT quote we receive
        # Use -1 so any interval will trigger (same as startup)
        self.last_trade_time = -1
        self._last_edge_check_at = None
        self._decision_in_flight = False
        logger.info(f"  Edge timer reset; will evaluate on next eligible tick")
        
        self.subscribe_quote_ticks(self.instrument_id)
        return True

    # ------------------------------------------------------------------
    # Timer loop - SIMPLIFIED
    # ------------------------------------------------------------------

    def _start_timer_loop(self):
        """Start timer loop in executor"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._timer_loop())
        finally:
            loop.close()

    async def _timer_loop(self):
        """
        Timer loop: checks every 10 seconds if it's time to switch markets.
        Also handles the case where we're waiting for a future market to open.
        """
        while True:
            # --- auto-restart check ---
            uptime_minutes = (datetime.now(timezone.utc) - self.bot_start_time).total_seconds() / 60
            if uptime_minutes >= self.restart_after_minutes:
                logger.warning("AUTO-RESTART TIME - Loading fresh filters")
                import signal as _signal
                os.kill(os.getpid(), _signal.SIGTERM)
                return

            now = datetime.now(timezone.utc)

            if (
                self._last_paper_settlement_check is None
                or (now - self._last_paper_settlement_check).total_seconds()
                >= PAPER_SETTLEMENT_CHECK_SECONDS
            ):
                self._last_paper_settlement_check = now
                await self._settle_pending_paper_trades()

            if self.next_switch_time and now >= self.next_switch_time:
                if self._waiting_for_market_open:
                    # The future market we were waiting for has now opened
                    # Treat it like a market switch so trade timer resets
                    logger.info("=" * 80)
                    logger.info(f"⏰ WAITING MARKET NOW OPEN: {now.strftime('%H:%M:%S')} UTC")
                    logger.info("=" * 80)
                    # Update next_switch_time to the market's END time
                    if (self.current_instrument_index >= 0 and
                            self.current_instrument_index < len(self.all_btc_instruments)):
                        current_market = self.all_btc_instruments[self.current_instrument_index]
                        self.next_switch_time = current_market['end_time']
                        logger.info(f"  Market ends at {self.next_switch_time.strftime('%H:%M:%S')} UTC")
                    self._waiting_for_market_open = False
                    self._market_stable = True
                    self._stable_tick_count = QUOTE_STABILITY_REQUIRED
                    self.last_trade_time = -1
                    self._last_edge_check_at = None
                    self._decision_in_flight = False
                    logger.info("  ✓ MARKET OPEN - ready to evaluate Markov edge")
                else:
                    # Normal market switch
                    self._switch_to_next_market()

            await asyncio.sleep(10)

    # ------------------------------------------------------------------
    # Quote tick handler - SIMPLIFIED
    # ------------------------------------------------------------------

    def on_quote_tick(self, tick: QuoteTick):
        """Handle quote ticks and periodically evaluate Markov edge."""
        try:
            # Only process ticks from current instrument
            if self.instrument_id is None or tick.instrument_id != self.instrument_id:
                return

            now = datetime.now(timezone.utc)
            bid = tick.bid_price
            ask = tick.ask_price

            if bid is None or ask is None:
                return
                
            try:
                bid_decimal = bid.as_decimal()
                ask_decimal = ask.as_decimal()
            except:
                return

            # Always store price history
            mid_price = (bid_decimal + ask_decimal) / 2
            self.price_history.append(mid_price)
            if len(self.price_history) > self.max_history:
                self.price_history.pop(0)
            
            # Store latest bid/ask for liquidity check before order placement
            self._last_bid_ask = (bid_decimal, ask_decimal)

            # Tick buffer for TickVelocityProcessor (rolling 90s window)
            self._tick_buffer.append({'ts': now, 'price': mid_price})

            # Stability gate
            if not self._market_stable:
                self._stable_tick_count += 1
                if self._stable_tick_count >= 1:
                    self._market_stable = True
                    logger.info(f"✓ Market STABLE immediately")
                else:
                    return

            # Markov edge checks:
            #   1. wait for the market to open and gather a little quote history
            #   2. evaluate at most once every EDGE_CHECK_INTERVAL_SECONDS
            #   3. let the Markov processor decide whether p - q >= MIN_EDGE
            if self._waiting_for_market_open:
                return

            if (self.current_instrument_index < 0 or
                    self.current_instrument_index >= len(self.all_btc_instruments)):
                return

            current_market = self.all_btc_instruments[self.current_instrument_index]
            market_start_ts = current_market['market_timestamp']  # Slug timestamp = market start (Unix)

            elapsed_secs = now.timestamp() - market_start_ts
            if elapsed_secs < self.market_warmup_seconds:
                return

            seconds_until_close = current_market['end_timestamp'] - now.timestamp()
            if seconds_until_close <= self.market_close_buffer_seconds:
                return

            if self._market_trade_counts.get(market_start_ts, 0) >= self.max_trades_per_market:
                return

            if self._decision_in_flight:
                return

            if self._last_edge_check_at:
                since_last = (now - self._last_edge_check_at).total_seconds()
                if since_last < self.edge_check_interval_seconds:
                    return

            self._last_edge_check_at = now
            self._decision_in_flight = True

            logger.info("=" * 80)
            logger.info(f"MARKOV EDGE CHECK: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            logger.info(f"  Market: {current_market['slug']}")
            logger.info(f"  Elapsed: {elapsed_secs:.1f}s | Close in: {seconds_until_close:.1f}s")
            logger.info(f"  YES price: ${float(mid_price):,.4f} | Bid: ${float(bid_decimal):,.4f} | Ask: ${float(ask_decimal):,.4f}")
            logger.info(f"  Price history: {len(self.price_history)} points")
            logger.info("=" * 80)

            self.run_in_executor(lambda: self._make_trading_decision_sync(float(mid_price)))

        except Exception as e:
            logger.error(f"Error processing quote tick: {e}")

    # ------------------------------------------------------------------
    # Trading decision
    # ------------------------------------------------------------------

    def _make_trading_decision_sync(self, current_price):
        """Synchronous wrapper for trading decision (called from executor)."""
        # Convert float back to Decimal for processing
        from decimal import Decimal
        price_decimal = Decimal(str(current_price))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._make_trading_decision(price_decimal))
        finally:
            loop.close()
            self._decision_in_flight = False
            
    async def _fetch_market_context(self, current_price: Decimal) -> dict:
        """
        Fetch REAL external data to populate signal processor metadata.

        Returns a dict with:
          - sentiment_score (float 0-100): live Fear & Greed index, or None
          - spot_price (float): live BTCUSDT from Binance Global, or None
          - spot_candles (list): recent BTC candles for Markov state transitions
          - deviation (float): polymarket price vs SMA-20 (always computed)
          - momentum (float): 5-period rate of change (always computed)
          - volatility (float): price std-dev over last 20 ticks (always computed)
        """
        current_price_float = float(current_price)

        # --- Always-available stats from local price_history ---
        recent_prices = [float(p) for p in self.price_history[-20:]]
        sma_20 = sum(recent_prices) / len(recent_prices)
        deviation = (current_price_float - sma_20) / sma_20
        momentum = (
            (current_price_float - float(self.price_history[-5])) / float(self.price_history[-5])
            if len(self.price_history) >= 5 else 0.0
        )
        variance = sum((p - sma_20) ** 2 for p in recent_prices) / len(recent_prices)
        volatility = math.sqrt(variance)

        metadata = {
            "deviation": deviation,
            "momentum": momentum,
            "volatility": volatility,
            # Tick buffer for TickVelocityProcessor
            "tick_buffer": list(self._tick_buffer),
            # YES token id for OrderBookImbalanceProcessor
            "yes_token_id": self._yes_token_id,
            # Approximate NO probability when we only subscribe to the YES token.
            "no_price": max(0.0, min(1.0, 1.0 - current_price_float)),
        }

        # --- Real sentiment: Fear & Greed Index via NewsSocialDataSource ---
        try:
            from data_sources.news_social.adapter import NewsSocialDataSource
            news_source = NewsSocialDataSource()
            await news_source.connect()
            fg = await news_source.get_fear_greed_index()
            await news_source.disconnect()
            if fg and "value" in fg:
                metadata["sentiment_score"] = float(fg["value"])
                metadata["sentiment_classification"] = fg.get("classification", "")
                logger.info(
                    f"Fear & Greed: {metadata['sentiment_score']:.0f} "
                    f"({metadata['sentiment_classification']})"
                )
            else:
                logger.warning("Fear & Greed fetch returned no data - sentiment processor skipped")
        except Exception as e:
            logger.warning(f"Could not fetch Fear & Greed index: {e} - sentiment processor skipped")

        # --- Real spot price and candles: Binance Global BTCUSDT REST API ---
        try:
            from data_sources.binance.adapter import BinanceDataSource
            binance = BinanceDataSource(symbol=SPOT_SYMBOL)
            try:
                connected = await binance.connect()
                spot = await binance.get_current_price() if connected else None
                if spot:
                    metadata["spot_price"] = float(spot)
                    metadata["spot_source"] = "binance"
                    metadata["spot_symbol"] = SPOT_SYMBOL
                    self._spot_price_history.append({"ts": datetime.now(timezone.utc), "price": spot})
                    metadata["spot_price_history"] = list(self._spot_price_history)
                    logger.info(f"Binance Global {SPOT_SYMBOL} spot price: ${float(spot):,.2f}")
                else:
                    logger.warning("Binance price fetch returned None - divergence processor skipped")

                cache_seconds = int(os.getenv("MARKOV_CANDLES_CACHE_SECONDS", "60"))
                cache_stale = (
                    self._spot_candles_cache_time is None or
                    (datetime.now(timezone.utc) - self._spot_candles_cache_time).total_seconds() >= cache_seconds
                )
                if connected and cache_stale:
                    granularity = int(os.getenv("MARKOV_CANDLE_GRANULARITY", "60"))
                    candles = await binance.get_candles(granularity=granularity, limit=120)
                    if candles:
                        self._spot_candles_cache = candles
                        self._spot_candles_cache_time = datetime.now(timezone.utc)
                        logger.info(f"Loaded {len(candles)} Binance candles for Markov model")
                    else:
                        logger.warning("Binance candle fetch returned no data - Markov will use fallback history")
            finally:
                await binance.disconnect()

            if self._spot_candles_cache:
                metadata["spot_candles"] = self._spot_candles_cache
        except Exception as e:
            logger.warning(f"Could not fetch Binance market context: {e} - Markov may use fallback history")

        logger.info(
            f"Market context - deviation={deviation:.2%}, "
            f"momentum={momentum:.2%}, volatility={volatility:.4f}, "
            f"sentiment={'%.0f' % metadata['sentiment_score'] if 'sentiment_score' in metadata else 'N/A'}, "
            f"spot=${'%.2f' % metadata['spot_price'] if 'spot_price' in metadata else 'N/A'}, "
            f"spot_source={metadata.get('spot_source', 'N/A')}"
        )
        return metadata

    async def _make_trading_decision(self, current_price: Decimal):
        """
        Make trading decision using the Markov + edge + Kelly rule.
        """
        is_simulation = await self.check_simulation_mode()
        logger.info(f"Mode: {'SIMULATION' if is_simulation else 'LIVE TRADING'}")

        if len(self.price_history) < 20:
            logger.warning(f"Not enough price history ({len(self.price_history)}/20)")
            return

        logger.info(f"Current YES price: ${float(current_price):,.4f}")

        metadata = await self._fetch_market_context(current_price)
        signals = self._process_signals(current_price, metadata)

        if not signals:
            logger.info("No signals generated - no trade this interval")
            return

        logger.info(f"Generated {len(signals)} signal(s):")
        for sig in signals:
            logger.info(
                f"  [{sig.source}] {sig.direction.value}: "
                f"score={sig.score:.1f}, confidence={sig.confidence:.2%}"
            )

        markov_signal = next((sig for sig in signals if sig.source == "MarkovPersistence"), None)
        if not markov_signal:
            markov_eval = getattr(self.markov_processor, "last_evaluation", {}) or {}
            model_probability = markov_eval.get("model_probability")
            market_probability = markov_eval.get("market_probability")
            fee_aware_edge = markov_eval.get("fee_aware_edge")
            reason = markov_eval.get("reason", "unknown")
            model_probability_text = (
                f"{float(model_probability):.2%}"
                if model_probability is not None
                else "N/A"
            )
            market_probability_text = (
                f"{float(market_probability):.2%}"
                if market_probability is not None
                else "N/A"
            )
            fee_aware_edge_text = (
                f"{float(fee_aware_edge):.2%}"
                if fee_aware_edge is not None
                else "N/A"
            )
            logger.info(
                f"Markov gate not passed: requires p >= {self.min_prob:.0%} "
                f"and edge >= {self.min_edge:.0%}; reason={reason}, "
                f"model_probability={model_probability_text}, "
                f"market_probability={market_probability_text}, "
                f"fee_aware_edge={fee_aware_edge_text}"
            )
            return

        fused = self.fusion_engine.fuse_signals(signals, min_signals=1, min_score=40.0)
        if fused:
            logger.info(
                f"FUSED CONTEXT: {fused.direction.value} "
                f"(score={fused.score:.1f}, confidence={fused.confidence:.2%})"
            )

            if self.require_fusion_confirmation:
                fused_dir = str(fused.direction).upper()
                markov_dir = str(markov_signal.direction).upper()
                if ("BULLISH" in fused_dir) != ("BULLISH" in markov_dir):
                    logger.info("Fusion disagrees with Markov and confirmation is required - skipping")
                    return
        else:
            logger.info("Fusion produced no consensus; Markov signal remains the trade gate")

        markov_meta = markov_signal.metadata or {}
        position_size = Decimal(str(markov_meta.get("position_size_usd", float(self.min_bet))))
        position_size = min(max(position_size, self.min_bet), self.max_bet, self.bankroll)

        is_bullish = "BULLISH" in str(markov_signal.direction).upper()
        direction = "long" if is_bullish else "short"
        selected_token = "YES" if is_bullish else "NO"
        selected_entry_price = current_price if is_bullish else Decimal("1.0") - current_price

        logger.info("=" * 80)
        logger.info("MARKOV ENTRY APPROVED")
        logger.info(f"  Direction: buy {selected_token} ({direction.upper()})")
        logger.info(f"  Model probability: {markov_meta.get('model_probability', 0):.2%}")
        logger.info(f"  Market probability: {markov_meta.get('market_probability', 0):.2%}")
        logger.info(f"  Fee-aware edge: {markov_meta.get('fee_aware_edge', 0):.2%}")
        logger.info(f"  Kelly fraction: {markov_meta.get('kelly_fraction', 0):.2%}")
        logger.info(f"  Position size: ${float(position_size):.2f}")
        logger.info("=" * 80)

        is_valid, error = self.risk_engine.validate_new_position(
            size=position_size,
            direction=direction,
            current_price=selected_entry_price,
        )
        if not is_valid:
            logger.warning(f"Risk engine blocked trade: {error}")
            return

        last_tick = getattr(self, '_last_bid_ask', None)
        if last_tick:
            last_bid, last_ask = last_tick
            min_liquidity = Decimal("0.02")
            if direction == "long" and last_ask <= min_liquidity:
                logger.warning(
                    f"No YES liquidity: ask=${float(last_ask):.4f} <= "
                    f"{float(min_liquidity):.2f}; retrying on next check"
                )
                self._last_edge_check_at = None
                return
            if direction == "short":
                no_ask_estimate = Decimal("1.0") - last_bid
                if no_ask_estimate <= min_liquidity:
                    logger.warning(
                        f"No estimated NO liquidity: ask~${float(no_ask_estimate):.4f} <= "
                        f"{float(min_liquidity):.2f}; retrying on next check"
                    )
                    self._last_edge_check_at = None
                    return

        if is_simulation:
            executed = await self._record_paper_trade(markov_signal, position_size, current_price, direction)
        else:
            executed = await self._place_real_order(markov_signal, position_size, current_price, direction)

        if executed and 0 <= self.current_instrument_index < len(self.all_btc_instruments):
            market_ts = self.all_btc_instruments[self.current_instrument_index]['market_timestamp']
            self._market_trade_counts[market_ts] = self._market_trade_counts.get(market_ts, 0) + 1
            logger.info(
                f"Market trade count: {self._market_trade_counts[market_ts]}/"
                f"{self.max_trades_per_market}"
            )
            
    async def _record_paper_trade(self, signal, position_size, current_price, direction):
        entry_time = datetime.now(timezone.utc)
        markov_meta = signal.metadata or {}
        market_snapshot = self._current_market_snapshot()

        selected_token = "YES" if direction == "long" else "NO"
        entry_price = current_price if direction == "long" else Decimal("1.0") - current_price
        if entry_price <= Decimal("0"):
            logger.warning("Paper trade skipped: invalid selected token entry price")
            return False

        trade_id = f"paper_{int(time.time() * 1000)}"
        model_probability = float(markov_meta.get("model_probability", signal.confidence))
        market_end_time = market_snapshot.get("end_time")
        if isinstance(market_end_time, datetime):
            market_end_time = market_end_time.isoformat()

        paper_trade = PaperTrade(
            trade_id=trade_id,
            timestamp=entry_time,
            direction=selected_token,
            size_usd=float(position_size),
            price=float(entry_price),
            signal_score=signal.score,
            signal_confidence=signal.confidence,
            outcome="PENDING",
            model_probability=model_probability,
            market_probability=float(markov_meta.get("market_probability", entry_price)),
            edge=float(markov_meta.get("fee_aware_edge", 0.0)),
            kelly_fraction=float(markov_meta.get("kelly_fraction", 0.0)),
            market_slug=market_snapshot.get("slug"),
            market_timestamp=market_snapshot.get("market_timestamp"),
            market_end_time=market_end_time,
            yes_price=float(current_price),
        )
        self.paper_trades.append(paper_trade)

        self.trade_journal.append({
            "event": "paper_trade",
            "trade_id": trade_id,
            "mode": "DRY_RUN",
            "settlement_mode": "market_resolution",
            "market": market_snapshot,
            "direction": direction,
            "selected_token": selected_token,
            "yes_price": current_price,
            "entry_price": entry_price,
            "exit_price": None,
            "size_usd": position_size,
            "pnl_usd": None,
            "outcome": "PENDING",
            "markov": markov_meta,
        })

        logger.info("=" * 80)
        logger.info("[SIMULATION] PAPER TRADE RECORDED - PENDING SETTLEMENT")
        logger.info(f"  Direction: {direction.upper()}")
        logger.info(f"  Selected Token: {selected_token}")
        logger.info(f"  Size: ${float(position_size):.2f}")
        logger.info(f"  Entry Price: ${float(entry_price):,.4f}")
        logger.info(f"  Market: {market_snapshot.get('slug', 'unknown')}")
        logger.info(f"  Market End: {market_end_time or 'unknown'}")
        logger.info("  Outcome: PENDING")
        logger.info(f"  Model probability used: {model_probability:.2%}")
        logger.info(f"  Total Paper Trades: {len(self.paper_trades)}")
        logger.info("=" * 80)

        self._save_paper_trades()
        return True

    def _current_market_snapshot(self) -> Dict[str, Any]:
        if 0 <= self.current_instrument_index < len(self.all_btc_instruments):
            market = self.all_btc_instruments[self.current_instrument_index]
            return {
                "slug": market.get("slug"),
                "start_time": market.get("start_time"),
                "end_time": market.get("end_time"),
                "market_timestamp": market.get("market_timestamp"),
                "yes_instrument_id": str(market.get("yes_instrument_id")),
                "no_instrument_id": str(market.get("no_instrument_id")),
            }
        return {}

    def _save_paper_trades(self):
        try:
            if not self.paper_trades:
                return
            trade_data = self.paper_trades[-1].to_dict()
            upsert_paper_trade(Path(PAPER_TRADES_PATH), trade_data)
        except Exception as e:
            logger.error(f"Failed to save paper trades: {e}")

    async def _settle_pending_paper_trades(self):
        try:
            settled_trades = await settle_pending_paper_trades(
                path=Path(PAPER_TRADES_PATH),
                market_interval_seconds=MARKET_INTERVAL_SECONDS,
                gamma_api_base_url=GAMMA_API_BASE_URL,
                settlement_delay_seconds=PAPER_SETTLEMENT_DELAY_SECONDS,
            )
        except Exception as e:
            logger.warning(f"Paper trade settlement check failed: {e}")
            return

        for trade in settled_trades:
            self._record_paper_settlement(trade)

    def _record_paper_settlement(self, trade: Dict[str, Any]):
        trade_id = trade.get("trade_id") or f"paper_settled_{int(time.time() * 1000)}"
        entry_value = trade.get("price")
        if entry_value is None:
            entry_value = trade.get("entry_price", 0)
        entry_price = Decimal(str(entry_value))
        exit_price = Decimal(str(trade.get("exit_price") or 0))
        size = Decimal(str(trade.get("size_usd") or 0))
        entry_time = parse_utc_datetime(trade.get("timestamp")) or datetime.now(timezone.utc)
        exit_time = parse_utc_datetime(trade.get("settlement_time")) or datetime.now(timezone.utc)
        won = trade.get("outcome") == "WIN"

        if entry_price > 0 and size > 0:
            self.performance_tracker.record_trade(
                trade_id=trade_id,
                direction="long",
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                entry_time=entry_time,
                exit_time=exit_time,
                signal_score=float(trade.get("signal_score") or 0.0),
                signal_confidence=float(trade.get("signal_confidence") or 0.0),
                metadata={
                    "simulated": True,
                    "settlement_mode": "market_resolution",
                    "market_result": trade.get("market_result"),
                    "market_slug": trade.get("market_slug"),
                    "selected_token": trade.get("direction"),
                    "yes_price": trade.get("yes_price"),
                },
            )

        if hasattr(self, 'grafana_exporter') and self.grafana_exporter:
            self.grafana_exporter.increment_trade_counter(won=won)
            self.grafana_exporter.record_trade_duration(
                max(0.0, (exit_time - entry_time).total_seconds())
            )

        self.trade_journal.append({
            "event": "paper_trade",
            "trade_id": trade_id,
            "mode": "DRY_RUN",
            "settlement_mode": "market_resolution",
            "market_slug": trade.get("market_slug"),
            "market_result": trade.get("market_result"),
            "selected_token": trade.get("direction"),
            "yes_price": trade.get("yes_price"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size_usd": size,
            "pnl_usd": trade.get("pnl_usd"),
            "outcome": trade.get("outcome"),
            "settlement_source": trade.get("settlement_source"),
        })

        for paper_trade in self.paper_trades:
            if paper_trade.trade_id == trade.get("trade_id"):
                paper_trade.outcome = trade.get("outcome", paper_trade.outcome)
                paper_trade.exit_price = trade.get("exit_price")
                paper_trade.pnl_usd = trade.get("pnl_usd")
                paper_trade.settlement_time = trade.get("settlement_time")
                paper_trade.settlement_source = trade.get("settlement_source")
                paper_trade.market_result = trade.get("market_result")
                paper_trade.start_price = trade.get("start_price")
                paper_trade.end_price = trade.get("end_price")
                break

        logger.info("=" * 80)
        logger.info("[SIMULATION] PAPER TRADE SETTLED")
        logger.info(f"  Trade ID: {trade_id}")
        logger.info(f"  Market: {trade.get('market_slug')}")
        logger.info(f"  Market Result: {trade.get('market_result')}")
        logger.info(f"  Selected Token: {trade.get('direction')}")
        logger.info(f"  Outcome: {trade.get('outcome')}")
        logger.info(f"  P&L: ${float(trade.get('pnl_usd') or 0):+.2f}")
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Real order
    # ------------------------------------------------------------------

    async def _place_real_order(self, signal, position_size, current_price, direction):
        if not self.instrument_id:
            logger.error("No instrument available")
            return False

        try:
            logger.info("=" * 80)
            logger.info("LIVE MODE - PLACING REAL ORDER!")
            logger.info("=" * 80)

            # On Polymarket, both UP and DOWN are BUY orders.
            # Bullish = buy YES token (self._yes_instrument_id)
            # Bearish = buy NO token  (self._no_instrument_id)
            # There is NO sell; you always buy whichever side you want.
            side = OrderSide.BUY

            if direction == "long":
                trade_instrument_id = getattr(self, '_yes_instrument_id', self.instrument_id)
                trade_label = "YES (UP)"
            else:
                no_id = getattr(self, '_no_instrument_id', None)
                if no_id is None:
                    logger.warning(
                        "NO token instrument not found for this market - "
                        "cannot bet DOWN. Skipping trade."
                    )
                    return False
                trade_instrument_id = no_id
                trade_label = "NO (DOWN)"

            instrument = self.cache.instrument(trade_instrument_id)
            if not instrument:
                logger.error(f"Instrument not in cache: {trade_instrument_id}")
                return False

            logger.info(f"Buying {trade_label} token: {trade_instrument_id}")

            selected_entry_price = current_price if direction == "long" else Decimal("1.0") - current_price
            trade_price = float(selected_entry_price)
            max_usd_amount = float(position_size)
            os.environ["MARKET_BUY_USD"] = f"{max_usd_amount:.2f}"

            precision = instrument.size_precision

            # Always BUY: the market-order patch converts this to a USD amount.
            # Pass dummy qty=5 (minimum) so Nautilus risk engine doesn't deny it.
            min_qty_val = float(getattr(instrument, 'min_quantity', None) or 5.0)
            token_qty = max(min_qty_val, 5.0)
            token_qty = round(token_qty, precision)
            logger.info(
                f"BUY {trade_label}: dummy qty={token_qty:.6f} "
                f"(patch converts to ${max_usd_amount:.2f} USD)"
            )

            qty = Quantity(token_qty, precision=precision)
            timestamp_ms = int(time.time() * 1000)
            unique_id = f"BTC-MARKOV-${max_usd_amount:.0f}-{timestamp_ms}"

            order = self.order_factory.market(
                instrument_id=trade_instrument_id,
                order_side=side,
                quantity=qty,
                client_order_id=ClientOrderId(unique_id),
                quote_quantity=False,
                time_in_force=TimeInForce.IOC,
            )

            self.submit_order(order)

            logger.info(f"REAL ORDER SUBMITTED!")
            logger.info(f"  Order ID: {unique_id}")
            logger.info(f"  Direction: {trade_label}")
            logger.info(f"  Side: BUY")
            logger.info(f"  Token Quantity: {token_qty:.6f}")
            logger.info(f"  Estimated Cost: ~${max_usd_amount:.2f}")
            logger.info(f"  Price: ${trade_price:.4f}")
            logger.info("=" * 80)

            self._track_order_event("placed")
            self.trade_journal.append({
                "event": "live_order",
                "order_id": unique_id,
                "mode": "LIVE",
                "market": self._current_market_snapshot(),
                "direction": direction,
                "selected_token": "YES" if direction == "long" else "NO",
                "yes_price": current_price,
                "entry_price": selected_entry_price,
                "size_usd": position_size,
                "pnl_usd": None,
                "markov": signal.metadata or {},
            })
            return True

        except Exception as e:
            logger.error(f"Error placing real order: {e}")
            import traceback
            traceback.print_exc()
            self._track_order_event("rejected")
            return False

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def _process_signals(self, current_price, metadata=None):
        signals = []
        if metadata is None:
            metadata = {}

        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, float):
                processed_metadata[key] = Decimal(str(value))
            else:
                processed_metadata[key] = value

        markov_signal = self.markov_processor.process(
            current_price=current_price,
            historical_prices=self.price_history,
            metadata=processed_metadata,
        )
        if markov_signal:
            signals.append(markov_signal)

        spike_signal = self.spike_detector.process(
            current_price=current_price,
            historical_prices=self.price_history,
            metadata=processed_metadata,
        )
        if spike_signal:
            signals.append(spike_signal)

        if 'sentiment_score' in processed_metadata:
            sentiment_signal = self.sentiment_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if sentiment_signal:
                signals.append(sentiment_signal)

        if 'spot_price' in processed_metadata:
            divergence_signal = self.divergence_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if divergence_signal:
                signals.append(divergence_signal)

        # --- Order Book Imbalance (real-time Polymarket CLOB depth) ---
        if processed_metadata.get('yes_token_id'):
            ob_signal = self.orderbook_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if ob_signal:
                signals.append(ob_signal)

        # --- Tick Velocity (last 60s of Polymarket probability movement) ---
        if processed_metadata.get('tick_buffer'):
            tv_signal = self.tick_velocity_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if tv_signal:
                signals.append(tv_signal)

        # --- Deribit Put/Call Ratio (institutional options sentiment) ---
        pcr_signal = self.deribit_pcr_processor.process(
            current_price=current_price,
            historical_prices=self.price_history,
            metadata=processed_metadata,
        )
        if pcr_signal:
            signals.append(pcr_signal)

        return signals

    # ------------------------------------------------------------------
    # Order events
    # ------------------------------------------------------------------

    def _track_order_event(self, event_type: str) -> None:
        """
        Safely track an order event on the performance tracker.

        PerformanceTracker does not expose `increment_order_counter`, so we
        use whichever method is actually available, or fall back to a no-op.
        Supported event_type values: "placed", "filled", "rejected".
        """
        try:
            pt = self.performance_tracker
            # Try the method that actually exists first
            if hasattr(pt, 'record_order_event'):
                pt.record_order_event(event_type)
            elif hasattr(pt, 'increment_counter'):
                pt.increment_counter(event_type)
            elif hasattr(pt, 'increment_order_counter'):
                pt.increment_order_counter(event_type)
            else:
                # No suitable method found – log and carry on
                logger.debug(
                    f"PerformanceTracker has no order-counter method; "
                    f"ignoring event '{event_type}'"
                )
        except Exception as e:
            logger.warning(f"Failed to track order event '{event_type}': {e}")

    def on_order_filled(self, event):
        logger.info("=" * 80)
        logger.info(f"ORDER FILLED!")
        logger.info(f"  Order: {event.client_order_id}")
        logger.info(f"  Fill Price: ${float(event.last_px):.4f}")
        logger.info(f"  Quantity: {float(event.last_qty):.6f}")
        logger.info("=" * 80)
        self._track_order_event("filled")

    def on_order_denied(self, event):
        logger.error("=" * 80)
        logger.error(f"ORDER DENIED!")
        logger.error(f"  Order: {event.client_order_id}")
        logger.error(f"  Reason: {event.reason}")
        logger.error("=" * 80)
        self._track_order_event("rejected")

    def on_order_rejected(self, event):
        """Handle order rejection and reset the edge timer so we can retry."""
        reason = str(getattr(event, 'reason', ''))
        reason_lower = reason.lower()
        if 'no orders found' in reason_lower or 'fak' in reason_lower or 'no match' in reason_lower:
            logger.warning(
                f"FAK rejected (no liquidity) - resetting edge timer to retry\n"
                f"  Reason: {reason}"
            )
            self.last_trade_time = -1
            self._last_edge_check_at = None
        else:
            logger.warning(f"Order rejected: {reason}")

    # ------------------------------------------------------------------
    # Grafana / stop
    # ------------------------------------------------------------------

    def _start_grafana_sync(self):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.grafana_exporter.start())
            logger.info("Grafana metrics started on port 8000")
        except Exception as e:
            logger.error(f"Failed to start Grafana: {e}")

    def on_stop(self):
        logger.info("Integrated BTC strategy stopped")
        logger.info(f"Total paper trades recorded: {len(self.paper_trades)}")
        if self.grafana_exporter:
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.grafana_exporter.stop())
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_integrated_bot(simulation: bool = False, enable_grafana: bool = True, test_mode: bool = False):
    """Run the integrated BTC Up/Down trading bot."""
    dry_run = DRY_RUN
    if dry_run:
        simulation = True

    funder_address = polymarket_funder_address()
    signer_address = os.getenv("POLYMARKET_SIGNER_ADDRESS") or os.getenv("SIGNER_ADDRESS")
    if funder_address:
        os.environ.setdefault("SAFE_ADDRESS", funder_address)
        os.environ.setdefault("POLYMARKET_FUNDER", funder_address)
        os.environ.setdefault("POLYMARKET_ADDRESS", funder_address)
    if signer_address:
        os.environ.setdefault("SIGNER_ADDRESS", signer_address)
        os.environ.setdefault("POLYMARKET_SIGNER_ADDRESS", signer_address)
    relayer_api_key = os.getenv("RELAYER_API_KEY") or os.getenv("POLYMARKET_RELAYER_API_KEY")
    if relayer_api_key:
        os.environ.setdefault("RELAYER_API_KEY", relayer_api_key)
        os.environ.setdefault("POLYMARKET_RELAYER_API_KEY", relayer_api_key)
    
    print("=" * 80)
    print(f"INTEGRATED POLYMARKET BTC {MARKET_LABEL} TRADING BOT")
    print("Nautilus + Markov Edge + Kelly Sizing")
    print("=" * 80)

    redis_client = init_redis()

    if redis_client:
        try:
            # ALWAYS overwrite Redis with the current session mode.
            # This prevents a stale value from a previous --live run
            # silently overriding --test-mode or --simulation runs.
            mode_value = '1' if simulation else '0'
            redis_client.set('btc_trading:simulation_mode', mode_value)
            mode_label = 'SIMULATION' if simulation else 'LIVE'
            logger.info(f"Redis simulation_mode forced to: {mode_label} ({mode_value})")
        except Exception as e:
            logger.warning(f"Could not set Redis simulation mode: {e}")

    print(f"\nConfiguration:")
    print(f"  Initial Mode: {'SIMULATION' if simulation else 'LIVE TRADING'}")
    print(f"  DRY_RUN: {dry_run}")
    print(f"  Redis Control: {'Enabled' if redis_client else 'Disabled'}")
    print(f"  Grafana: {'Enabled' if enable_grafana else 'Disabled'}")
    print(f"  Market interval: {MARKET_INTERVAL_SECONDS}s")
    print(f"  Market slug prefix: {MARKET_SLUG_PREFIX}")
    print(f"  Spot data source: Binance Global ({SPOT_SYMBOL})")
    print(f"  Markov: MIN_PROB={MIN_PROB:.2f} MIN_EDGE={MIN_EDGE:.2f}")
    print(f"  Bankroll: ${BANKROLL:.2f}")
    print(f"  Bet range: ${MIN_BET:.2f} - ${MAX_BET:.2f}")
    print(f"  Address/funder: {'configured' if funder_address else 'not configured'}")
    print(f"  Signer address: {'configured' if signer_address else 'not configured'}")
    print(f"  Relayer API key: {'configured' if relayer_api_key else 'not configured'}")
    print(f"  Quote stability gate: {QUOTE_STABILITY_REQUIRED} valid ticks")
    print(f"  Execution client: {'disabled for dry run' if simulation else 'enabled for live trading'}")
    print()

    now = datetime.now(timezone.utc)
    
    # =========================================================================
    # Slug timestamps are standard Unix timestamps aligned to the market interval.
    # =========================================================================
    now = datetime.now(timezone.utc)
    unix_interval_start = (int(now.timestamp()) // MARKET_INTERVAL_SECONDS) * MARKET_INTERVAL_SECONDS

    btc_slugs = []
    for i in range(-1, 97):  # include 1 prior interval (in case we're just after boundary)
        timestamp = unix_interval_start + (i * MARKET_INTERVAL_SECONDS)
        btc_slugs.append(f"{MARKET_SLUG_PREFIX}-{timestamp}")

    filters = {
        "active": True,
        "closed": False,
        "archived": False,
        "slug": tuple(btc_slugs),
        "limit": 100,
    }

    logger.info("=" * 80)
    logger.info(f"LOADING BTC {MARKET_LABEL} MARKETS BY SLUG")
    logger.info(f"  Interval start: {unix_interval_start} | Count: {len(btc_slugs)}")
    logger.info(f"  First: {btc_slugs[0]}  Last: {btc_slugs[-1]}")
    logger.info("=" * 80)

    instrument_cfg = InstrumentProviderConfig(
        load_all=True,
        filters=filters,
        use_gamma_markets=True,
    )

    poly_data_cfg = PolymarketDataClientConfig(
        private_key=polymarket_private_key(),
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
        signature_type=polymarket_signature_type(),
        funder=funder_address,
        base_url_http=clob_host(),
        base_url_ws=os.getenv("CLOB_WS_HOST"),
        instrument_provider=instrument_cfg,
    )

    poly_exec_cfg = None
    exec_clients = {}
    if simulation:
        logger.info(
            "Simulation mode: Polymarket execution client disabled; "
            "paper trading will not subscribe to user trades or reconcile account state."
        )
    else:
        poly_exec_cfg = PolymarketExecClientConfig(
            private_key=polymarket_private_key(),
            api_key=os.getenv("POLYMARKET_API_KEY"),
            api_secret=os.getenv("POLYMARKET_API_SECRET"),
            passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
            signature_type=polymarket_signature_type(),
            funder=funder_address,
            base_url_http=clob_host(),
            base_url_ws=os.getenv("CLOB_WS_HOST"),
            instrument_provider=instrument_cfg,
        )
        exec_clients[POLYMARKET] = poly_exec_cfg

    config = TradingNodeConfig(
        environment="live",
        trader_id="BTC-MARKOV-INTEGRATED-001",
        logging=LoggingConfig(
            log_level="INFO",
            log_directory="./logs/nautilus",
        ),
        data_engine=LiveDataEngineConfig(qsize=6000),
        exec_engine=LiveExecEngineConfig(qsize=6000),
        risk_engine=LiveRiskEngineConfig(bypass=simulation),
        data_clients={POLYMARKET: poly_data_cfg},
        exec_clients=exec_clients,
    )

    strategy = IntegratedBTCStrategy(
        redis_client=redis_client,
        enable_grafana=enable_grafana,
        test_mode=test_mode,
    )

    print("\nBuilding Nautilus node...")
    node = TradingNode(config=config)
    node.add_data_client_factory(POLYMARKET, PolymarketLiveDataClientFactory)
    if poly_exec_cfg is not None:
        node.add_exec_client_factory(POLYMARKET, PolymarketLiveExecClientFactory)
    node.trader.add_strategy(strategy)
    node.build()
    logger.info("Nautilus node built successfully")

    print()
    print("=" * 80)
    print("BOT STARTING")
    print("=" * 80)

    try:
        node.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.dispose()
        logger.info("Bot stopped")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Integrated BTC Up/Down Markov Trading Bot")
    parser.add_argument("--live", action="store_true",
                        help="Run in LIVE mode (real money at risk!). Default is simulation.")
    parser.add_argument("--no-grafana", action="store_true", help="Disable Grafana metrics")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in TEST MODE (trade every minute for faster testing)")

    args = parser.parse_args()
    enable_grafana = not args.no_grafana
    test_mode = args.test_mode

    # --test-mode ALWAYS forces simulation even if --live is also passed
    if args.test_mode:
        simulation = True
    else:
        simulation = not args.live

    if DRY_RUN:
        if args.live:
            logger.warning("DRY_RUN=True in bot-5min.py; --live will still run in simulation mode.")
        simulation = True

    if not simulation:
        logger.warning("=" * 80)
        logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK!")
        logger.warning("=" * 80)
    else:
        logger.info("=" * 80)
        logger.info(f"SIMULATION MODE - {'TEST MODE (fast clock)' if test_mode else 'paper trading only'}")
        logger.info("No real orders will be placed.")
        logger.info("=" * 80)

    run_integrated_bot(simulation=simulation, enable_grafana=enable_grafana, test_mode=test_mode)


if __name__ == "__main__":
    main()
