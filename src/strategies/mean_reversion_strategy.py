"""
Mean Reversion Trading Strategy

Implements a mean reversion strategy based on statistical analysis of price movements.
Uses RSI, Bollinger Bands, and Z-score analysis to identify oversold/overbought conditions.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config_manager import config
from src.screening.universe import UniverseManager
from src.screening.filters import TechnicalIndicators, TechnicalFilters
from src.screening.mean_reversion_scanner import MeanReversionScanner
from .base_strategy import BaseStrategy, PositionType


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Trading Strategy.
    
    This strategy identifies stocks that have moved significantly away from their
    statistical mean and bets on their return to the mean. It uses multiple
    technical indicators to confirm signals and manage risk.
    """
    
    def __init__(
        self,
        strategy_id: str = "mean_reversion_v1",
        name: str = "Mean Reversion Strategy",
        universe_manager: Optional[UniverseManager] = None,
        data_manager=None
    ):
        """
        Initialize Mean Reversion Strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable strategy name
            universe_manager: Universe manager for stock selection
            data_manager: Data manager for market data access
        """
        super().__init__(strategy_id, name, universe_manager, data_manager, "MEAN_REVERSION")
        
        # Strategy-specific configuration
        self.lookback_period = self.config.get('LOOKBACK_PERIOD', 20)
        self.z_score_entry = self.config.get('Z_SCORE_ENTRY', 2.0)
        self.z_score_exit = self.config.get('Z_SCORE_EXIT', 0.5)
        self.rsi_oversold = self.config.get('RSI_OVERSOLD', 30)
        self.rsi_overbought = self.config.get('RSI_OVERBOUGHT', 70)
        self.min_volume_ratio = self.config.get('MIN_VOLUME_RATIO', 1.2)
        self.max_position_hold_days = self.config.get('MAX_POSITION_HOLD_DAYS', 10)
        self.stop_loss_percent = self.config.get('STOP_LOSS_PERCENT', 0.05)  # 5%
        self.take_profit_percent = self.config.get('TAKE_PROFIT_PERCENT', 0.08)  # 8%
        
        # Initialize components
        self.universe_manager = universe_manager or UniverseManager()
        self.technical_filters = TechnicalFilters()
        self.mean_reversion_scanner = MeanReversionScanner()
        
        # Strategy state
        self.last_scan_time: Optional[datetime] = None
        self.scan_interval = timedelta(hours=1)  # Scan every hour
        self.signal_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"Mean Reversion Strategy initialized: {self.strategy_id}")
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate mean reversion trading signals.
        
        Args:
            market_data: Dictionary of symbol -> OHLCV dataframes
            
        Returns:
            List of trading signals with entry/exit recommendations
        """
        current_time = datetime.now()
        
        # Check if it's time to scan for new opportunities
        if (self.last_scan_time is None or 
            current_time - self.last_scan_time >= self.scan_interval):
            
            self.last_scan_time = current_time
            return self._scan_for_opportunities(market_data)
        
        # Check for exit signals on existing positions
        return self._check_exit_signals(market_data)
    
    def _scan_for_opportunities(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Scan universe for mean reversion opportunities."""
        signals = []
        
        # Get universe of tradeable symbols
        if self.universe_manager:
            universe = self.universe_manager.get_universe(apply_filters=True)
        else:
            universe = list(market_data.keys())
        
        self.logger.info(f"Scanning {len(universe)} symbols for mean reversion opportunities")
        
        for symbol in universe:
            if symbol not in market_data or market_data[symbol].empty:
                continue
            
            try:
                signal = self._analyze_symbol(symbol, market_data[symbol])
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        self.logger.info(f"Generated {len(signals)} mean reversion signals")
        return signals
    
    def _analyze_symbol(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze a single symbol for mean reversion opportunities.
        
        Args:
            symbol: Stock symbol to analyze
            data: OHLCV data for the symbol
            
        Returns:
            Trading signal dictionary or None
        """
        if len(data) < self.lookback_period:
            return None
        
        # Calculate technical indicators
        prices = data['Close']
        volumes = data['Volume']
        
        # Z-score analysis
        recent_prices = prices.tail(self.lookback_period)
        price_mean = recent_prices.mean()
        price_std = recent_prices.std()
        
        if price_std == 0:
            return None
        
        current_price = prices.iloc[-1]
        z_score = (current_price - price_mean) / price_std
        
        # RSI analysis
        rsi_values = TechnicalIndicators.rsi(prices)
        current_rsi = rsi_values.iloc[-1] if not rsi_values.empty else 50
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(prices)
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # Volume analysis
        recent_volume = volumes.tail(10).mean()
        current_volume = volumes.iloc[-1]
        volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1
        
        # Signal generation logic
        signal = None
        
        # Long signal (oversold conditions)
        if (z_score <= -self.z_score_entry and 
            current_rsi <= self.rsi_oversold and
            bb_position <= 0.2 and  # Near lower Bollinger Band
            volume_ratio >= self.min_volume_ratio):
            
            signal = {
                'symbol': symbol,
                'action': 'buy',
                'signal_type': 'mean_reversion_long',
                'entry_price': current_price,
                'confidence': self._calculate_confidence(z_score, current_rsi, bb_position, volume_ratio),
                'z_score': z_score,
                'rsi': current_rsi,
                'bb_position': bb_position,
                'volume_ratio': volume_ratio,
                'stop_loss': current_price * (1 - self.stop_loss_percent),
                'take_profit': current_price * (1 + self.take_profit_percent),
                'timestamp': datetime.now()
            }
        
        # Short signal (overbought conditions)
        elif (z_score >= self.z_score_entry and 
              current_rsi >= self.rsi_overbought and
              bb_position >= 0.8 and  # Near upper Bollinger Band
              volume_ratio >= self.min_volume_ratio):
            
            signal = {
                'symbol': symbol,
                'action': 'sell_short',
                'signal_type': 'mean_reversion_short',
                'entry_price': current_price,
                'confidence': self._calculate_confidence(abs(z_score), 100 - current_rsi, 1 - bb_position, volume_ratio),
                'z_score': z_score,
                'rsi': current_rsi,
                'bb_position': bb_position,
                'volume_ratio': volume_ratio,
                'stop_loss': current_price * (1 + self.stop_loss_percent),
                'take_profit': current_price * (1 - self.take_profit_percent),
                'timestamp': datetime.now()
            }
        
        if signal:
            self.signal_history.append(signal)
            self.logger.debug(f"Mean reversion signal: {symbol} {signal['action']} (Z={z_score:.2f}, RSI={current_rsi:.1f})")
        
        return signal
    
    def _calculate_confidence(self, z_score: float, rsi_component: float, bb_component: float, volume_ratio: float) -> float:
        """
        Calculate confidence score for a mean reversion signal.
        
        Args:
            z_score: Absolute Z-score value
            rsi_component: RSI-based component (adjusted for direction)
            bb_component: Bollinger Band position component
            volume_ratio: Volume ratio component
            
        Returns:
            Confidence score between 0 and 1
        """
        # Normalize components
        z_score_norm = min(abs(z_score) / 3.0, 1.0)  # Cap at 3 standard deviations
        rsi_norm = max(0, (50 - rsi_component) / 50.0) if rsi_component < 50 else max(0, (rsi_component - 50) / 50.0)
        bb_norm = abs(bb_component - 0.5) * 2  # Distance from middle band
        volume_norm = min((volume_ratio - 1.0) / 2.0, 1.0)  # Cap at 3x average volume
        
        # Weighted confidence calculation
        confidence = (
            z_score_norm * 0.4 +      # 40% weight on statistical deviation
            rsi_norm * 0.3 +          # 30% weight on momentum
            bb_norm * 0.2 +           # 20% weight on Bollinger position
            volume_norm * 0.1         # 10% weight on volume confirmation
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def _check_exit_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Check for exit signals on existing positions."""
        exit_signals = []
        
        for symbol, position in self.positions.items():
            if symbol not in market_data or market_data[symbol].empty:
                continue
            
            data = market_data[symbol]
            current_price = data['Close'].iloc[-1]
            
            # Calculate current Z-score
            prices = data['Close'].tail(self.lookback_period)
            price_mean = prices.mean()
            price_std = prices.std()
            
            if price_std > 0:
                z_score = (current_price - price_mean) / price_std
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                # Mean reversion exit (Z-score normalized)
                if position.position_type == PositionType.LONG and z_score >= -self.z_score_exit:
                    should_exit = True
                    exit_reason = "mean_reversion_target"
                elif position.position_type == PositionType.SHORT and z_score <= self.z_score_exit:
                    should_exit = True
                    exit_reason = "mean_reversion_target"
                
                # Time-based exit
                elif (datetime.now() - position.entry_time).days >= self.max_position_hold_days:
                    should_exit = True
                    exit_reason = "max_hold_period"
                
                if should_exit:
                    action = 'sell' if position.position_type == PositionType.LONG else 'cover'
                    exit_signals.append({
                        'symbol': symbol,
                        'action': action,
                        'signal_type': 'mean_reversion_exit',
                        'exit_price': current_price,
                        'reason': exit_reason,
                        'z_score': z_score,
                        'timestamp': datetime.now()
                    })
        
        return exit_signals
    
    def calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> float:
        """
        Calculate position size based on confidence and risk management.
        
        Args:
            symbol: Stock symbol
            signal: Trading signal information
            
        Returns:
            Position size in shares
        """
        if symbol not in signal or 'confidence' not in signal:
            return 0
        
        confidence = signal['confidence']
        entry_price = signal['entry_price']
        
        # Base position value as percentage of portfolio
        portfolio_value = max(self.get_portfolio_value(), self.max_portfolio_value)
        base_position_value = portfolio_value * self.position_size_percent
        
        # Adjust by confidence (min 50% of base size, max 150%)
        confidence_multiplier = 0.5 + (confidence * 1.0)
        adjusted_position_value = base_position_value * confidence_multiplier
        
        # Calculate shares
        shares = int(adjusted_position_value / entry_price)
        
        # Risk-based position sizing (limit loss to 1% of portfolio on stop loss)
        stop_loss = signal.get('stop_loss', entry_price * 0.95)
        max_loss_per_share = abs(entry_price - stop_loss)
        max_shares_by_risk = int((portfolio_value * 0.01) / max_loss_per_share) if max_loss_per_share > 0 else shares
        
        # Take the minimum of confidence-based and risk-based sizing
        final_shares = min(shares, max_shares_by_risk)
        
        self.logger.debug(f"Position sizing for {symbol}: {final_shares} shares "
                         f"(confidence={confidence:.2f}, risk_limit={max_shares_by_risk})")
        
        return final_shares
    
    def manage_risk(self, positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Implement risk management for mean reversion strategy.
        
        Args:
            positions: Current positions
            
        Returns:
            List of risk management actions
        """
        risk_actions = []
        current_time = datetime.now()
        
        for symbol, position in positions.items():
            # Check for trailing stop adjustments
            if position.position_type == PositionType.LONG:
                # Trailing stop for long positions
                if position.unrealized_pnl > 0:
                    # Move stop loss up if profitable
                    new_stop_loss = position.current_price * (1 - self.stop_loss_percent * 0.5)  # Tighter stop when profitable
                    if new_stop_loss > position.stop_loss:
                        risk_actions.append({
                            'type': 'adjust_stop_loss',
                            'symbol': symbol,
                            'new_stop_loss': new_stop_loss,
                            'reason': 'trailing_stop_up'
                        })
            
            elif position.position_type == PositionType.SHORT:
                # Trailing stop for short positions
                if position.unrealized_pnl > 0:
                    # Move stop loss down if profitable
                    new_stop_loss = position.current_price * (1 + self.stop_loss_percent * 0.5)  # Tighter stop when profitable
                    if new_stop_loss < position.stop_loss:
                        risk_actions.append({
                            'type': 'adjust_stop_loss',
                            'symbol': symbol,
                            'new_stop_loss': new_stop_loss,
                            'reason': 'trailing_stop_down'
                        })
            
            # Force close positions held too long
            hold_duration = current_time - position.entry_time
            if hold_duration.days > self.max_position_hold_days:
                risk_actions.append({
                    'type': 'close_position',
                    'symbol': symbol,
                    'reason': 'max_hold_period_exceeded'
                })
        
        # Portfolio-level risk management
        total_portfolio_value = self.get_portfolio_value()
        if total_portfolio_value > self.max_portfolio_value * 1.1:  # 10% over limit
            # Close least profitable positions
            sorted_positions = sorted(
                positions.items(),
                key=lambda x: x[1].unrealized_pnl
            )
            
            positions_to_close = len(positions) // 4  # Close 25% of positions
            for symbol, _ in sorted_positions[:positions_to_close]:
                risk_actions.append({
                    'type': 'close_position',
                    'symbol': symbol,
                    'reason': 'portfolio_size_limit'
                })
        
        return risk_actions
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy-specific performance metrics."""
        base_metrics = self.get_performance_summary()
        
        # Add mean reversion specific metrics
        mean_reversion_metrics = {
            'avg_hold_duration_hours': 0.0,
            'successful_reversions': 0,
            'failed_reversions': 0,
            'avg_z_score_entry': 0.0,
            'signal_accuracy': 0.0
        }
        
        # Calculate metrics from signal history
        if self.signal_history:
            z_scores = [s['z_score'] for s in self.signal_history]
            mean_reversion_metrics['avg_z_score_entry'] = sum(abs(z) for z in z_scores) / len(z_scores)
        
        # Calculate hold duration from closed positions
        closed_positions = [pos for pos in self.positions.values() if pos.status.value == 'closed']
        if closed_positions:
            total_duration = sum((pos.last_update - pos.entry_time).total_seconds() for pos in closed_positions)
            mean_reversion_metrics['avg_hold_duration_hours'] = total_duration / (3600 * len(closed_positions))
        
        base_metrics['mean_reversion_metrics'] = mean_reversion_metrics
        return base_metrics
