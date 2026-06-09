"""
Markov persistence signal processor.

The processor models short-term BTC price movement as discrete states, then
trades only when the current directional state has a high probability of
persisting and Polymarket has not fully priced that probability in.
"""
from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    SignalDirection,
    SignalStrength,
    SignalType,
    TradingSignal,
)


class MarkovPersistenceProcessor(BaseSignalProcessor):
    """
    Generate entries from Markov state persistence.

    Entry rule:
      model_probability - market_probability - fee_adjustment >= min_edge
      and model_probability >= min_probability

    For an UP state, market_probability is the YES price. For a DOWN state, it
    is approximated as 1 - YES price unless a richer NO quote is provided.
    """

    DIRECTIONAL_STATES = {"strong_up", "up", "down", "strong_down"}

    def __init__(
        self,
        min_probability: float = 0.87,
        min_edge: float = 0.05,
        return_threshold: float = 0.0002,
        strong_return_threshold: float = 0.0010,
        probability_return_threshold: float = 0.0100,
        strong_probability_return_threshold: float = 0.0250,
        min_transitions: int = 30,
        min_state_observations: int = 5,
        fee_bps: float = 0.0,
        bankroll: Decimal = Decimal("100.00"),
        min_bet: Decimal = Decimal("1.00"),
        max_bet: Decimal = Decimal("50.00"),
        kelly_scale: float = 1.0,
    ):
        super().__init__("MarkovPersistence")
        self.min_probability = min_probability
        self.min_edge = min_edge
        self.return_threshold = return_threshold
        self.strong_return_threshold = strong_return_threshold
        self.probability_return_threshold = probability_return_threshold
        self.strong_probability_return_threshold = strong_probability_return_threshold
        self.min_transitions = min_transitions
        self.min_state_observations = min_state_observations
        self.fee_bps = fee_bps
        self.bankroll = bankroll
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.kelly_scale = kelly_scale
        self.last_evaluation: Dict[str, Any] = {}

        logger.info(
            "Initialized Markov Persistence Processor: "
            f"min_prob={min_probability:.0%}, min_edge={min_edge:.1%}, "
            f"bankroll=${bankroll:.2f}, bet=${min_bet:.2f}-${max_bet:.2f}, "
            f"kelly_scale={kelly_scale:.2f}"
        )

    def process(
        self,
        current_price: Decimal,
        historical_prices: list,
        metadata: Dict[str, Any] = None,
    ) -> Optional[TradingSignal]:
        if not self.is_enabled:
            self.last_evaluation = {"reason": "disabled"}
            return None

        metadata = metadata or {}
        prices, source = self._extract_price_series(historical_prices, metadata)
        if len(prices) < self.min_transitions + 1:
            self.last_evaluation = {
                "reason": "insufficient_history",
                "state_source": source,
                "price_count": len(prices),
                "required_price_count": self.min_transitions + 1,
            }
            logger.info(
                f"MarkovPersistence: insufficient history "
                f"({len(prices)} prices < {self.min_transitions + 1})"
            )
            return None

        threshold, strong_threshold = self._thresholds_for_series(prices, source)
        states = self._states_from_prices(prices, threshold, strong_threshold)
        if len(states) < self.min_transitions:
            self.last_evaluation = {
                "reason": "insufficient_transitions",
                "state_source": source,
                "transition_count": len(states),
                "required_transition_count": self.min_transitions,
                "return_threshold": threshold,
                "strong_return_threshold": strong_threshold,
            }
            return None

        current_state = states[-1]
        if current_state not in self.DIRECTIONAL_STATES:
            self.last_evaluation = {
                "reason": "non_directional_state",
                "state_source": source,
                "current_state": current_state,
                "total_transitions": len(states) - 1,
                "return_threshold": threshold,
                "strong_return_threshold": strong_threshold,
            }
            logger.info(f"MarkovPersistence: current state is {current_state}; no directional edge")
            return None

        transition_counts, state_counts = self._transition_counts(states)
        current_state_count = state_counts.get(current_state, 0)
        if current_state_count < self.min_state_observations:
            self.last_evaluation = {
                "reason": "insufficient_state_observations",
                "state_source": source,
                "current_state": current_state,
                "state_observations": current_state_count,
                "required_state_observations": self.min_state_observations,
                "total_transitions": len(states) - 1,
                "return_threshold": threshold,
                "strong_return_threshold": strong_threshold,
            }
            logger.info(
                f"MarkovPersistence: state {current_state} has only "
                f"{current_state_count} observations (< {self.min_state_observations})"
            )
            return None

        persistence = self._transition_probability(
            transition_counts,
            state_counts,
            current_state,
            current_state,
        )
        self.last_evaluation = {
            "reason": "evaluated",
            "state_source": source,
            "current_state": current_state,
            "model_probability": round(persistence, 6),
            "state_observations": current_state_count,
            "total_transitions": len(states) - 1,
            "return_threshold": threshold,
            "strong_return_threshold": strong_threshold,
        }
        if persistence < self.min_probability:
            self.last_evaluation["reason"] = "model_probability_below_min"
            logger.info(
                f"MarkovPersistence: p({current_state}->{current_state})="
                f"{persistence:.2%} below {self.min_probability:.2%}"
            )
            return None

        direction = (
            SignalDirection.BULLISH
            if current_state in {"up", "strong_up"}
            else SignalDirection.BEARISH
        )
        yes_price = float(current_price)
        market_probability = self._market_probability_for_direction(direction, yes_price, metadata)
        self.last_evaluation.update({
            "direction": direction.value,
            "market_probability": round(market_probability, 6),
        })
        if market_probability <= 0.0 or market_probability >= 1.0:
            self.last_evaluation["reason"] = "invalid_market_probability"
            logger.info(f"MarkovPersistence: invalid market probability {market_probability:.4f}")
            return None

        fee_adjustment = self.fee_bps / 10000.0
        raw_edge = persistence - market_probability
        fee_aware_edge = raw_edge - fee_adjustment
        self.last_evaluation.update({
            "raw_edge": round(raw_edge, 6),
            "fee_aware_edge": round(fee_aware_edge, 6),
            "fee_bps": self.fee_bps,
        })
        if fee_aware_edge < self.min_edge:
            self.last_evaluation["reason"] = "edge_below_min"
            logger.info(
                f"MarkovPersistence: edge={fee_aware_edge:.2%} below "
                f"{self.min_edge:.2%} (model={persistence:.2%}, market={market_probability:.2%})"
            )
            return None

        kelly_fraction = self._kelly_fraction(persistence, market_probability)
        position_size = self._kelly_position_size(kelly_fraction)
        self.last_evaluation.update({
            "kelly_fraction": round(kelly_fraction, 6),
            "kelly_scale": self.kelly_scale,
            "position_size_usd": float(position_size),
        })
        if position_size <= Decimal("0"):
            self.last_evaluation["reason"] = "zero_kelly_size"
            logger.info("MarkovPersistence: Kelly size is zero; no trade")
            return None

        self.last_evaluation["reason"] = "signal_generated"
        strength = self._strength_for_edge(fee_aware_edge)
        confidence = min(0.99, max(0.0, persistence))

        signal = TradingSignal(
            timestamp=datetime.now(),
            source=self.name,
            signal_type=SignalType.MOMENTUM,
            direction=direction,
            strength=strength,
            confidence=confidence,
            current_price=current_price,
            metadata={
                "state_source": source,
                "current_state": current_state,
                "model_probability": round(persistence, 6),
                "market_probability": round(market_probability, 6),
                "raw_edge": round(raw_edge, 6),
                "fee_aware_edge": round(fee_aware_edge, 6),
                "fee_bps": self.fee_bps,
                "kelly_fraction": round(kelly_fraction, 6),
                "kelly_scale": self.kelly_scale,
                "position_size_usd": float(position_size),
                "state_observations": current_state_count,
                "total_transitions": len(states) - 1,
                "return_threshold": threshold,
                "strong_return_threshold": strong_threshold,
            },
        )
        self._record_signal(signal)

        logger.info(
            f"Generated {direction.value.upper()} Markov signal: "
            f"state={current_state}, p={persistence:.2%}, "
            f"market={market_probability:.2%}, edge={fee_aware_edge:.2%}, "
            f"kelly={kelly_fraction:.2%}, size=${position_size:.2f}"
        )
        return signal

    def _extract_price_series(
        self,
        historical_prices: list,
        metadata: Dict[str, Any],
    ) -> Tuple[List[float], str]:
        candles = metadata.get("spot_candles") or []
        if candles:
            sorted_candles = sorted(candles, key=lambda c: c.get("timestamp", datetime.min))
            closes = [
                float(c["close"])
                for c in sorted_candles
                if c.get("close") is not None and float(c["close"]) > 0
            ]
            if len(closes) >= self.min_transitions + 1:
                return closes, "spot_candles"

        spot_history = metadata.get("spot_price_history") or []
        if spot_history:
            sorted_history = sorted(spot_history, key=lambda p: p.get("ts", datetime.min))
            spot_prices = [
                float(p["price"])
                for p in sorted_history
                if p.get("price") is not None and float(p["price"]) > 0
            ]
            if len(spot_prices) >= self.min_transitions + 1:
                return spot_prices, "spot_price_history"

        fallback = [float(p) for p in historical_prices if float(p) > 0]
        fallback_source = "polymarket_probability" if fallback and max(fallback) <= 2.0 else "price_history"
        return fallback, fallback_source

    def _thresholds_for_series(self, prices: Sequence[float], source: str) -> Tuple[float, float]:
        if source == "polymarket_probability" or max(prices) <= 2.0:
            return self.probability_return_threshold, self.strong_probability_return_threshold
        return self.return_threshold, self.strong_return_threshold

    def _states_from_prices(
        self,
        prices: Sequence[float],
        threshold: float,
        strong_threshold: float,
    ) -> List[str]:
        states = []
        for prev, curr in zip(prices, prices[1:]):
            if prev <= 0:
                continue
            ret = (curr - prev) / prev
            if ret >= strong_threshold:
                states.append("strong_up")
            elif ret >= threshold:
                states.append("up")
            elif ret <= -strong_threshold:
                states.append("strong_down")
            elif ret <= -threshold:
                states.append("down")
            else:
                states.append("flat")
        return states

    def _transition_counts(self, states: Sequence[str]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
        transition_counts: Dict[str, Dict[str, int]] = {}
        state_counts: Dict[str, int] = {}
        for state, next_state in zip(states, states[1:]):
            transition_counts.setdefault(state, {})
            transition_counts[state][next_state] = transition_counts[state].get(next_state, 0) + 1
            state_counts[state] = state_counts.get(state, 0) + 1
        return transition_counts, state_counts

    def _transition_probability(
        self,
        transition_counts: Dict[str, Dict[str, int]],
        state_counts: Dict[str, int],
        state: str,
        next_state: str,
    ) -> float:
        total = state_counts.get(state, 0)
        if total <= 0:
            return 0.0
        return transition_counts.get(state, {}).get(next_state, 0) / total

    def _market_probability_for_direction(
        self,
        direction: SignalDirection,
        yes_price: float,
        metadata: Dict[str, Any],
    ) -> float:
        if direction == SignalDirection.BULLISH:
            return yes_price

        no_price = metadata.get("no_price")
        if no_price is not None:
            return float(no_price)
        return 1.0 - yes_price

    def _kelly_fraction(self, probability: float, market_probability: float) -> float:
        odds_profit = (1.0 - market_probability) / market_probability
        if odds_profit <= 0:
            return 0.0
        fraction = probability - ((1.0 - probability) / odds_profit)
        return max(0.0, min(1.0, fraction))

    def _kelly_position_size(self, kelly_fraction: float) -> Decimal:
        scaled_fraction = Decimal(str(kelly_fraction * self.kelly_scale))
        raw_size = self.bankroll * scaled_fraction
        if raw_size <= Decimal("0"):
            return Decimal("0")
        size = max(self.min_bet, raw_size)
        size = min(self.max_bet, size, self.bankroll)
        return size.quantize(Decimal("0.01"))

    def _strength_for_edge(self, edge: float) -> SignalStrength:
        if edge >= 0.15:
            return SignalStrength.VERY_STRONG
        if edge >= 0.10:
            return SignalStrength.STRONG
        return SignalStrength.MODERATE
