"""
Pairs Trading Strategy

Implements a market-neutral pairs trading strategy that identifies correlated stocks
and trades their relative price movements. Uses statistical arbitrage principles
to profit from temporary divergences in historically correlated securities.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config_manager import config
from src.screening.universe import UniverseManager
from src.screening.pairs_scanner import PairsTradingScanner, TradingPair
from .base_strategy import BaseStrategy, PositionType, Position


class PairPosition:
    """Represents a pairs trading position with two legs."""
    
    def __init__(
        self,
        pair: TradingPair,
        leg_a_position: Position,
        leg_b_position: Position,
        entry_spread: float,
        entry_time: datetime,
        strategy_id: str
    ):
        self.pair = pair
        self.leg_a_position = leg_a_position
        self.leg_b_position = leg_b_position
        self.entry_spread = entry_spread
        self.entry_time = entry_time
        self.strategy_id = strategy_id
        
        # Pair tracking
        self.current_spread = entry_spread
        self.spread_z_score = 0.0
        self.unrealized_pnl = 0.0
        self.last_update = entry_time
        
        # Performance tracking
        self.max_favorable_spread = entry_spread
        self.max_adverse_spread = entry_spread
    
    def update_spread(self, price_a: float, price_b: float, timestamp: datetime) -> None:
        """Update the pair with current prices."""
        # Calculate current spread
        self.current_spread = price_a - (self.pair.hedge_ratio * price_b)
        
        # Calculate Z-score based on historical spread statistics
        self.spread_z_score = (self.current_spread - self.pair.spread_mean) / self.pair.spread_std
        
        # Update individual position prices
        self.leg_a_position.update_price(price_a, timestamp)
        self.leg_b_position.update_price(price_b, timestamp)
        
        # Calculate combined P&L
        self.unrealized_pnl = self.leg_a_position.unrealized_pnl + self.leg_b_position.unrealized_pnl
        self.last_update = timestamp
        
        # Track spread extremes
        if self.leg_a_position.position_type == PositionType.LONG:
            # Long A, Short B - profit when spread decreases
            self.max_favorable_spread = min(self.max_favorable_spread, self.current_spread)
            self.max_adverse_spread = max(self.max_adverse_spread, self.current_spread)
        else:
            # Short A, Long B - profit when spread increases
            self.max_favorable_spread = max(self.max_favorable_spread, self.current_spread)
            self.max_adverse_spread = min(self.max_adverse_spread, self.current_spread)
    
    def check_exit_conditions(self, z_score_exit: float) -> Tuple[bool, str]:
        """Check if pair should be closed."""
        # Mean reversion exit
        if abs(self.spread_z_score) <= z_score_exit:
            return True, "mean_reversion"
        
        # Individual position stop losses
        leg_a_exit, leg_a_reason = self.leg_a_position.check_exit_conditions()
        if leg_a_exit:
            return True, f"leg_a_{leg_a_reason}"
        
        leg_b_exit, leg_b_reason = self.leg_b_position.check_exit_conditions()
        if leg_b_exit:
            return True, f"leg_b_{leg_b_reason}"
        
        return False, "none"
    
    def close_pair(self, price_a: float, price_b: float, exit_time: datetime, reason: str) -> float:
        """Close the pair and calculate total P&L."""
        # Close individual positions
        pnl_a = self.leg_a_position.close_position(price_a, exit_time, reason)
        pnl_b = self.leg_b_position.close_position(price_b, exit_time, reason)
        
        total_pnl = pnl_a + pnl_b
        return total_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pair position to dictionary."""
        return {
            'pair': self.pair.to_dict(),
            'leg_a_position': self.leg_a_position.to_dict(),
            'leg_b_position': self.leg_b_position.to_dict(),
            'entry_spread': self.entry_spread,
            'current_spread': self.current_spread,
            'spread_z_score': self.spread_z_score,
            'unrealized_pnl': self.unrealized_pnl,
            'entry_time': self.entry_time,
            'last_update': self.last_update
        }


class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading Strategy.
    
    This market-neutral strategy trades pairs of correlated stocks by taking
    opposing positions when their price relationship diverges significantly
    from the historical norm, betting on mean reversion of the spread.
    """
    
    def __init__(
        self,
        strategy_id: str = "pairs_trading_v1",
        name: str = "Pairs Trading Strategy",
        universe_manager: Optional[UniverseManager] = None,
        data_manager=None
    ):
        """
        Initialize Pairs Trading Strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable strategy name
            universe_manager: Universe manager for stock selection
            data_manager: Data manager for market data access
        """
        super().__init__(strategy_id, name, universe_manager, data_manager, "PAIRS_TRADING")
        
        # Strategy-specific configuration
        self.min_correlation = self.config.get('MIN_CORRELATION', 0.7)
        self.correlation_lookback = self.config.get('CORRELATION_LOOKBACK', 126)  # 6 months
        self.z_score_entry = self.config.get('Z_SCORE_ENTRY', 2.0)
        self.z_score_exit = self.config.get('Z_SCORE_EXIT', 0.5)
        self.max_pairs = self.config.get('MAX_PAIRS', 5)
        self.rebalance_frequency = self.config.get('REBALANCE_FREQUENCY_HOURS', 24)
        self.max_pair_hold_days = self.config.get('MAX_PAIR_HOLD_DAYS', 14)
        self.pair_dollar_neutral = self.config.get('PAIR_DOLLAR_NEUTRAL', True)
        
        # Risk management
        self.max_individual_position_loss = self.config.get('MAX_INDIVIDUAL_POSITION_LOSS', 0.02)  # 2%
        self.max_pair_loss = self.config.get('MAX_PAIR_LOSS', 0.03)  # 3%
        
        # Initialize components
        self.universe_manager = universe_manager or UniverseManager()
        self.pairs_scanner = PairsTradingScanner(self.universe_manager, data_manager)
        
        # Strategy state
        self.pair_positions: Dict[str, PairPosition] = {}
        self.last_pair_scan: Optional[datetime] = None
        self.last_rebalance: Optional[datetime] = None
        self.correlation_cache: Dict[Tuple[str, str], TradingPair] = {}
        self.pair_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"Pairs Trading Strategy initialized: {self.strategy_id}")
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate pairs trading signals.
        
        Args:
            market_data: Dictionary of symbol -> OHLCV dataframes
            
        Returns:
            List of trading signals for pair entries and exits
        """
        current_time = datetime.now()
        signals = []
        
        # Check for pair exit signals first
        exit_signals = self._check_pair_exits(market_data)
        signals.extend(exit_signals)
        
        # Check if it's time to scan for new pairs
        scan_interval = timedelta(hours=self.rebalance_frequency)
        if (self.last_pair_scan is None or 
            current_time - self.last_pair_scan >= scan_interval):
            
            self.last_pair_scan = current_time
            entry_signals = self._scan_for_pair_opportunities(market_data)
            signals.extend(entry_signals)
        
        return signals
    
    def _scan_for_pair_opportunities(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Scan for new pairs trading opportunities."""
        if len(self.pair_positions) >= self.max_pairs:
            self.logger.debug("Maximum pairs reached, skipping new opportunities scan")
            return []
        
        signals = []
        
        # Get universe for pair analysis
        if self.universe_manager:
            universe = list(self.universe_manager.get_universe(apply_filters=True))
        else:
            universe = list(market_data.keys())
        
        # Filter symbols with sufficient data
        valid_symbols = []
        for symbol in universe:
            if (symbol in market_data and 
                not market_data[symbol].empty and 
                len(market_data[symbol]) >= self.correlation_lookback):
                valid_symbols.append(symbol)
        
        if len(valid_symbols) < 2:
            self.logger.warning("Insufficient symbols for pairs analysis")
            return []
        
        self.logger.info(f"Scanning {len(valid_symbols)} symbols for pairs opportunities")
        
        # Find correlated pairs
        correlated_pairs = self.pairs_scanner.find_correlated_pairs(
            valid_symbols, 
            self.min_correlation
        )
        
        # Analyze each pair for trading opportunities
        for pair in correlated_pairs:
            # Skip pairs we already have positions in
            pair_key = f"{pair.symbol_a}_{pair.symbol_b}"
            if pair_key in self.pair_positions:
                continue
            
            # Check if spread is significantly diverged
            if abs(pair.z_score) >= self.z_score_entry:
                signal = self._create_pair_entry_signal(pair, market_data)
                if signal:
                    signals.append(signal)
        
        self.logger.info(f"Generated {len(signals)} pairs entry signals")
        return signals
    
    def _create_pair_entry_signal(self, pair: TradingPair, market_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Create a pairs trading entry signal."""
        symbol_a, symbol_b = pair.symbol_a, pair.symbol_b
        
        # Get current prices
        if (symbol_a not in market_data or symbol_b not in market_data or
            market_data[symbol_a].empty or market_data[symbol_b].empty):
            return None
        
        price_a = market_data[symbol_a]['Close'].iloc[-1]
        price_b = market_data[symbol_b]['Close'].iloc[-1]
        
        # Determine trade direction based on z-score
        if pair.z_score > self.z_score_entry:
            # Spread is too high: short A, long B
            action_a, action_b = 'sell_short', 'buy'
            signal_type = 'pairs_short_spread'
        elif pair.z_score < -self.z_score_entry:
            # Spread is too low: long A, short B
            action_a, action_b = 'buy', 'sell_short'
            signal_type = 'pairs_long_spread'
        else:
            return None
        
        # Calculate position sizes for dollar neutrality
        if self.pair_dollar_neutral:
            # Make positions dollar neutral
            total_capital = self.max_portfolio_value * self.position_size_percent
            capital_per_leg = total_capital / 2
            
            shares_a = int(capital_per_leg / price_a)
            shares_b = int(capital_per_leg / price_b)
        else:
            # Use hedge ratio for share neutrality
            base_shares = int((self.max_portfolio_value * self.position_size_percent) / price_a)
            shares_a = base_shares
            shares_b = int(base_shares * pair.hedge_ratio)
        
        return {
            'signal_type': signal_type,
            'pair': pair,
            'symbol_a': symbol_a,
            'symbol_b': symbol_b,
            'action_a': action_a,
            'action_b': action_b,
            'price_a': price_a,
            'price_b': price_b,
            'shares_a': shares_a,
            'shares_b': shares_b,
            'z_score': pair.z_score,
            'correlation': pair.correlation,
            'hedge_ratio': pair.hedge_ratio,
            'confidence': min(abs(pair.z_score) / 3.0, 1.0),  # Higher z-score = higher confidence
            'timestamp': datetime.now()
        }
    
    def _check_pair_exits(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Check existing pairs for exit signals."""
        exit_signals = []
        current_time = datetime.now()
        
        for pair_key, pair_position in self.pair_positions.items():
            symbol_a = pair_position.pair.symbol_a
            symbol_b = pair_position.pair.symbol_b
            
            # Get current prices
            if (symbol_a not in market_data or symbol_b not in market_data or
                market_data[symbol_a].empty or market_data[symbol_b].empty):
                continue
            
            price_a = market_data[symbol_a]['Close'].iloc[-1]
            price_b = market_data[symbol_b]['Close'].iloc[-1]
            
            # Update pair position
            pair_position.update_spread(price_a, price_b, current_time)
            
            # Check exit conditions
            should_exit, reason = pair_position.check_exit_conditions(self.z_score_exit)
            
            # Additional exit conditions
            if not should_exit:
                # Time-based exit
                hold_duration = current_time - pair_position.entry_time
                if hold_duration.days >= self.max_pair_hold_days:
                    should_exit = True
                    reason = "max_hold_period"
                
                # Risk-based exit
                elif pair_position.unrealized_pnl < -self.max_pair_loss * self.max_portfolio_value:
                    should_exit = True
                    reason = "max_loss_exceeded"
            
            if should_exit:
                exit_signals.append({
                    'signal_type': 'pairs_exit',
                    'pair_key': pair_key,
                    'symbol_a': symbol_a,
                    'symbol_b': symbol_b,
                    'action_a': 'sell' if pair_position.leg_a_position.position_type == PositionType.LONG else 'cover',
                    'action_b': 'sell' if pair_position.leg_b_position.position_type == PositionType.LONG else 'cover',
                    'price_a': price_a,
                    'price_b': price_b,
                    'reason': reason,
                    'z_score': pair_position.spread_z_score,
                    'pnl': pair_position.unrealized_pnl,
                    'timestamp': current_time
                })
        
        return exit_signals
    
    def calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> float:
        """
        Calculate position size for pairs trading.
        
        For pairs trading, position sizes are calculated as part of the signal
        generation to ensure proper hedging ratios.
        """
        if signal.get('signal_type', '').startswith('pairs_'):
            # Position sizes already calculated in signal
            if symbol == signal.get('symbol_a'):
                return signal.get('shares_a', 0)
            elif symbol == signal.get('symbol_b'):
                return signal.get('shares_b', 0)
        
        return 0
    
    def manage_risk(self, positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Implement risk management for pairs trading.
        
        Args:
            positions: Current positions (not used directly, using pair_positions)
            
        Returns:
            List of risk management actions
        """
        risk_actions = []
        current_time = datetime.now()
        
        # Portfolio-level risk management
        total_exposure = sum(abs(pp.unrealized_pnl) for pp in self.pair_positions.values())
        max_allowed_exposure = self.max_portfolio_value * 0.10  # 10% max total exposure
        
        if total_exposure > max_allowed_exposure:
            # Close least profitable pairs
            sorted_pairs = sorted(
                self.pair_positions.items(),
                key=lambda x: x[1].unrealized_pnl
            )
            
            pairs_to_close = len(self.pair_positions) // 3  # Close 1/3 of pairs
            for pair_key, _ in sorted_pairs[:pairs_to_close]:
                risk_actions.append({
                    'type': 'close_pair',
                    'pair_key': pair_key,
                    'reason': 'portfolio_exposure_limit'
                })
        
        # Individual pair risk management
        for pair_key, pair_position in self.pair_positions.items():
            # Check individual position risk
            leg_a_loss_pct = (pair_position.leg_a_position.unrealized_pnl / 
                             (pair_position.leg_a_position.entry_price * pair_position.leg_a_position.quantity))
            leg_b_loss_pct = (pair_position.leg_b_position.unrealized_pnl / 
                             (pair_position.leg_b_position.entry_price * pair_position.leg_b_position.quantity))
            
            if (abs(leg_a_loss_pct) > self.max_individual_position_loss or 
                abs(leg_b_loss_pct) > self.max_individual_position_loss):
                risk_actions.append({
                    'type': 'close_pair',
                    'pair_key': pair_key,
                    'reason': 'individual_position_risk'
                })
        
        return risk_actions
    
    def _execute_pair_entry(self, signal: Dict[str, Any]) -> None:
        """Execute a pairs trading entry signal."""
        pair = signal['pair']
        symbol_a, symbol_b = signal['symbol_a'], signal['symbol_b']
        
        # Create individual positions
        current_time = datetime.now()
        
        # Position A
        pos_type_a = PositionType.LONG if signal['action_a'] == 'buy' else PositionType.SHORT
        position_a = Position(
            symbol=symbol_a,
            position_type=pos_type_a,
            quantity=signal['shares_a'],
            entry_price=signal['price_a'],
            entry_time=current_time,
            strategy_id=self.strategy_id
        )
        
        # Position B
        pos_type_b = PositionType.LONG if signal['action_b'] == 'buy' else PositionType.SHORT
        position_b = Position(
            symbol=symbol_b,
            position_type=pos_type_b,
            quantity=signal['shares_b'],
            entry_price=signal['price_b'],
            entry_time=current_time,
            strategy_id=self.strategy_id
        )
        
        # Create pair position
        current_spread = signal['price_a'] - (pair.hedge_ratio * signal['price_b'])
        pair_position = PairPosition(
            pair=pair,
            leg_a_position=position_a,
            leg_b_position=position_b,
            entry_spread=current_spread,
            entry_time=current_time,
            strategy_id=self.strategy_id
        )
        
        # Store pair position
        pair_key = f"{symbol_a}_{symbol_b}"
        self.pair_positions[pair_key] = pair_position
        
        # Add to individual positions for base class tracking
        self.positions[f"{symbol_a}_pair"] = position_a
        self.positions[f"{symbol_b}_pair"] = position_b
        
        self.logger.info(f"Opened pairs trade: {symbol_a} {signal['action_a']} x{signal['shares_a']} @ ${signal['price_a']:.2f}, "
                        f"{symbol_b} {signal['action_b']} x{signal['shares_b']} @ ${signal['price_b']:.2f} "
                        f"(Z-score: {signal['z_score']:.2f})")
    
    def _execute_pair_exit(self, signal: Dict[str, Any]) -> None:
        """Execute a pairs trading exit signal."""
        pair_key = signal['pair_key']
        
        if pair_key not in self.pair_positions:
            self.logger.warning(f"Pair {pair_key} not found for exit")
            return
        
        pair_position = self.pair_positions[pair_key]
        symbol_a, symbol_b = signal['symbol_a'], signal['symbol_b']
        
        # Close the pair
        total_pnl = pair_position.close_pair(
            signal['price_a'],
            signal['price_b'],
            signal['timestamp'],
            signal['reason']
        )
        
        # Remove from positions
        self.positions.pop(f"{symbol_a}_pair", None)
        self.positions.pop(f"{symbol_b}_pair", None)
        
        # Store in history
        self.pair_history.append({
            'pair_key': pair_key,
            'entry_time': pair_position.entry_time,
            'exit_time': signal['timestamp'],
            'total_pnl': total_pnl,
            'exit_reason': signal['reason'],
            'hold_duration_hours': (signal['timestamp'] - pair_position.entry_time).total_seconds() / 3600
        })
        
        # Remove from active pairs
        del self.pair_positions[pair_key]
        
        self.logger.info(f"Closed pairs trade: {pair_key} P&L: ${total_pnl:.2f}, Reason: {signal['reason']}")
    
    def update(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Override base update to handle pairs-specific logic."""
        if self.state.value != 'active':
            return
        
        try:
            self.last_update = datetime.now()
            
            # Update pair positions first
            self._update_pair_positions(market_data)
            
            # Generate signals (includes pair-specific logic)
            signals = self.generate_signals(market_data)
            
            # Execute signals
            for signal in signals:
                if signal.get('signal_type') == 'pairs_exit':
                    self._execute_pair_exit(signal)
                elif signal.get('signal_type', '').startswith('pairs_'):
                    self._execute_pair_entry(signal)
            
            # Manage risk
            risk_actions = self.manage_risk(self.positions)
            for action in risk_actions:
                if action.get('type') == 'close_pair':
                    pair_key = action.get('pair_key')
                    if pair_key in self.pair_positions:
                        # Create exit signal for risk management
                        pair_pos = self.pair_positions[pair_key]
                        risk_exit_signal = {
                            'signal_type': 'pairs_exit',
                            'pair_key': pair_key,
                            'symbol_a': pair_pos.pair.symbol_a,
                            'symbol_b': pair_pos.pair.symbol_b,
                            'price_a': pair_pos.leg_a_position.current_price,
                            'price_b': pair_pos.leg_b_position.current_price,
                            'reason': action.get('reason', 'risk_management'),
                            'timestamp': datetime.now()
                        }
                        self._execute_pair_exit(risk_exit_signal)
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating pairs strategy {self.name}: {e}")
            self.state = self.state.ERROR
    
    def _update_pair_positions(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update all pair positions with current market data."""
        current_time = datetime.now()
        
        for pair_position in self.pair_positions.values():
            symbol_a = pair_position.pair.symbol_a
            symbol_b = pair_position.pair.symbol_b
            
            if (symbol_a in market_data and symbol_b in market_data and
                not market_data[symbol_a].empty and not market_data[symbol_b].empty):
                
                price_a = market_data[symbol_a]['Close'].iloc[-1]
                price_b = market_data[symbol_b]['Close'].iloc[-1]
                
                pair_position.update_spread(price_a, price_b, current_time)
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get pairs trading specific performance metrics."""
        base_metrics = self.get_performance_summary()
        
        # Add pairs-specific metrics
        pairs_metrics = {
            'active_pairs': len(self.pair_positions),
            'completed_pairs': len(self.pair_history),
            'avg_pair_hold_hours': 0.0,
            'avg_pair_pnl': 0.0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'avg_correlation': 0.0,
            'avg_z_score_entry': 0.0
        }
        
        # Calculate metrics from pair history
        if self.pair_history:
            total_hold_time = sum(p['hold_duration_hours'] for p in self.pair_history)
            pairs_metrics['avg_pair_hold_hours'] = total_hold_time / len(self.pair_history)
            
            total_pnl = sum(p['total_pnl'] for p in self.pair_history)
            pairs_metrics['avg_pair_pnl'] = total_pnl / len(self.pair_history)
            
            pairs_metrics['successful_pairs'] = sum(1 for p in self.pair_history if p['total_pnl'] > 0)
            pairs_metrics['failed_pairs'] = len(self.pair_history) - pairs_metrics['successful_pairs']
        
        # Calculate active pairs metrics
        if self.pair_positions:
            correlations = [pp.pair.correlation for pp in self.pair_positions.values()]
            pairs_metrics['avg_correlation'] = sum(correlations) / len(correlations)
            
            z_scores = [abs(pp.spread_z_score) for pp in self.pair_positions.values()]
            pairs_metrics['avg_z_score_entry'] = sum(z_scores) / len(z_scores)
        
        base_metrics['pairs_metrics'] = pairs_metrics
        return base_metrics
