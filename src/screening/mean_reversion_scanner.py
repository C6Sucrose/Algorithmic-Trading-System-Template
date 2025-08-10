"""
Mean Reversion Scanner

Scans the universe for mean reversion trading opportunities.
Identifies oversold/overbought stocks using technical indicators.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config_manager import config
from src.screening.universe import UniverseManager
from src.screening.filters import TechnicalFilters
from src.data_management.historical_provider import HistoricalDataManager


class MeanReversionOpportunity:
    """Represents a mean reversion trading opportunity."""
    
    def __init__(
        self,
        symbol: str,
        direction: str,  # 'long' or 'short'
        confidence: float,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        signals: Dict[str, Any],
        timestamp: datetime
    ):
        self.symbol = symbol
        self.direction = direction
        self.confidence = confidence
        self.entry_price = entry_price
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.signals = signals
        self.timestamp = timestamp
        
        # Calculate risk/reward metrics
        self.risk = abs(entry_price - stop_loss)
        self.reward = abs(target_price - entry_price)
        self.risk_reward_ratio = self.reward / self.risk if self.risk > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert opportunity to dictionary."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'risk': self.risk,
            'reward': self.reward,
            'risk_reward_ratio': self.risk_reward_ratio,
            'signals': self.signals,
            'timestamp': self.timestamp
        }


class MeanReversionScanner:
    """Scans for mean reversion trading opportunities."""
    
    def __init__(
        self,
        universe_manager: Optional[UniverseManager] = None,
        data_manager: Optional[HistoricalDataManager] = None
    ):
        """
        Initialize mean reversion scanner.
        
        Args:
            universe_manager: Universe manager for stock selection
            data_manager: Data manager for historical data access
        """
        self.universe_manager = universe_manager or UniverseManager()
        self.data_manager = data_manager or HistoricalDataManager()
        self.technical_filters = TechnicalFilters()
        
        # Configuration from config file
        self.config = config.get('MEAN_REVERSION', {})
        self.min_confidence = self.config.get('MIN_CONFIDENCE', 0.7)
        self.min_risk_reward = self.config.get('MIN_RISK_REWARD', 1.5)
        self.stop_loss_pct = self.config.get('STOP_LOSS_PCT', 0.02)
        self.target_profit_pct = self.config.get('TARGET_PROFIT_PCT', 0.05)
        self.max_hold_days = self.config.get('MAX_HOLD_DAYS', 5)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def scan_universe(
        self,
        symbols: Optional[List[str]] = None,
        min_confidence: Optional[float] = None
    ) -> List[MeanReversionOpportunity]:
        """
        Scan universe for mean reversion opportunities.
        
        Args:
            symbols: Optional list of symbols to scan (uses universe if None)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of mean reversion opportunities
        """
        # Get symbols to scan
        if symbols is None:
            symbols = list(self.universe_manager.get_universe(apply_filters=True))
        
        if not symbols:
            self.logger.warning("No symbols to scan")
            return []
        
        confidence_threshold = min_confidence or self.min_confidence
        opportunities = []
        
        self.logger.info(f"Scanning {len(symbols)} symbols for mean reversion opportunities")
        
        for symbol in symbols:
            try:
                opportunity = self._analyze_symbol(symbol)
                if opportunity and opportunity.confidence >= confidence_threshold:
                    opportunities.append(opportunity)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by confidence (descending)
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        
        self.logger.info(f"Found {len(opportunities)} mean reversion opportunities")
        
        return opportunities
    
    def _analyze_symbol(self, symbol: str) -> Optional[MeanReversionOpportunity]:
        """
        Analyze a single symbol for mean reversion opportunities.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            MeanReversionOpportunity if found, None otherwise
        """
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 60 days of data
        
        data = self.data_manager.get_symbol_data(symbol, start_date, end_date)
        if data is None or len(data) < 30:  # Need at least 30 days
            return None
        
        # Calculate technical indicators
        data_with_indicators = self.technical_filters.calculate_all_indicators(data)
        
        # Get signals
        signals = self.technical_filters.get_mean_reversion_signals(data_with_indicators)
        
        # Determine direction and confidence
        if signals['bullish_score'] > signals['bearish_score']:
            direction = 'long'
            confidence = signals['bullish_score']
            entry_condition = self._check_long_entry_conditions(signals)
        else:
            direction = 'short'
            confidence = signals['bearish_score']
            entry_condition = self._check_short_entry_conditions(signals)
        
        # Check if entry conditions are met
        if not entry_condition:
            return None
        
        # Calculate entry, target, and stop loss prices
        current_price = data['close'].iloc[-1]
        entry_price = current_price
        
        if direction == 'long':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            target_price = entry_price * (1 + self.target_profit_pct)
        else:
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            target_price = entry_price * (1 - self.target_profit_pct)
        
        # Create opportunity
        opportunity = MeanReversionOpportunity(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            signals=signals,
            timestamp=datetime.now()
        )
        
        # Check risk/reward ratio
        if opportunity.risk_reward_ratio < self.min_risk_reward:
            return None
        
        return opportunity
    
    def _check_long_entry_conditions(self, signals: Dict[str, Any]) -> bool:
        """
        Check if long entry conditions are met.
        
        Args:
            signals: Technical signals dictionary
            
        Returns:
            True if long entry conditions are met
        """
        conditions = []
        
        # RSI oversold
        if not pd.isna(signals.get('rsi_value', np.nan)):
            conditions.append(signals.get('rsi_oversold', False))
        
        # Near lower Bollinger Band
        conditions.append(signals.get('near_bb_lower', False))
        
        # Not in strong uptrend (avoid buying into momentum)
        conditions.append(not signals.get('trending_up', False))
        
        # Volume confirmation (optional but preferred)
        volume_ok = signals.get('volume_spike', False) or signals.get('volume_ratio', 1.0) > 0.8
        conditions.append(volume_ok)
        
        # Need at least 75% of conditions to be True
        true_conditions = sum(conditions)
        return true_conditions >= len(conditions) * 0.75
    
    def _check_short_entry_conditions(self, signals: Dict[str, Any]) -> bool:
        """
        Check if short entry conditions are met.
        
        Args:
            signals: Technical signals dictionary
            
        Returns:
            True if short entry conditions are met
        """
        conditions = []
        
        # RSI overbought
        if not pd.isna(signals.get('rsi_value', np.nan)):
            conditions.append(signals.get('rsi_overbought', False))
        
        # Near upper Bollinger Band
        conditions.append(signals.get('near_bb_upper', False))
        
        # Not in strong downtrend (avoid shorting into momentum)
        conditions.append(not signals.get('trending_down', False))
        
        # Volume confirmation
        volume_ok = signals.get('volume_spike', False) or signals.get('volume_ratio', 1.0) > 0.8
        conditions.append(volume_ok)
        
        # Need at least 75% of conditions to be True
        true_conditions = sum(conditions)
        return true_conditions >= len(conditions) * 0.75
    
    def get_top_opportunities(
        self,
        limit: int = 10,
        direction: Optional[str] = None
    ) -> List[MeanReversionOpportunity]:
        """
        Get top mean reversion opportunities.
        
        Args:
            limit: Maximum number of opportunities to return
            direction: Filter by direction ('long', 'short', or None for both)
            
        Returns:
            List of top opportunities
        """
        opportunities = self.scan_universe()
        
        # Filter by direction if specified
        if direction:
            opportunities = [op for op in opportunities if op.direction == direction]
        
        # Return top N opportunities
        return opportunities[:limit]
    
    def get_opportunities_summary(self) -> Dict[str, Any]:
        """Get summary of current opportunities."""
        opportunities = self.scan_universe()
        
        if not opportunities:
            return {
                'total_opportunities': 0,
                'long_opportunities': 0,
                'short_opportunities': 0,
                'avg_confidence': 0.0,
                'avg_risk_reward': 0.0,
                'top_opportunities': []
            }
        
        long_ops = [op for op in opportunities if op.direction == 'long']
        short_ops = [op for op in opportunities if op.direction == 'short']
        
        return {
            'total_opportunities': len(opportunities),
            'long_opportunities': len(long_ops),
            'short_opportunities': len(short_ops),
            'avg_confidence': np.mean([op.confidence for op in opportunities]),
            'avg_risk_reward': np.mean([op.risk_reward_ratio for op in opportunities]),
            'top_opportunities': [op.to_dict() for op in opportunities[:5]]
        }
    
    def scan_for_exits(
        self,
        current_positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Scan current positions for exit signals.
        
        Args:
            current_positions: List of current position dictionaries
            
        Returns:
            List of exit recommendations
        """
        exit_signals = []
        
        for position in current_positions:
            symbol = position['symbol']
            direction = position['direction']
            entry_price = position['entry_price']
            entry_date = position['entry_date']
            
            # Get current data
            data = self.data_manager.get_symbol_data(symbol)
            if data is None or data.empty:
                continue
            
            current_price = data['close'].iloc[-1]
            days_held = (datetime.now() - entry_date).days
            
            # Calculate current P&L
            if direction == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check exit conditions
            exit_reason = None
            
            # Time-based exit
            if days_held >= self.max_hold_days:
                exit_reason = 'max_hold_period'
            
            # Profit target hit
            elif pnl_pct >= self.target_profit_pct:
                exit_reason = 'profit_target'
            
            # Stop loss hit
            elif pnl_pct <= -self.stop_loss_pct:
                exit_reason = 'stop_loss'
            
            # Technical reversal
            else:
                signals = self.technical_filters.get_mean_reversion_signals(data)
                if direction == 'long' and signals['bearish_score'] > 0.8:
                    exit_reason = 'technical_reversal'
                elif direction == 'short' and signals['bullish_score'] > 0.8:
                    exit_reason = 'technical_reversal'
            
            if exit_reason:
                exit_signals.append({
                    'symbol': symbol,
                    'direction': direction,
                    'current_price': current_price,
                    'pnl_pct': pnl_pct,
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                    'urgency': 'high' if exit_reason in ['stop_loss', 'technical_reversal'] else 'medium'
                })
        
        return exit_signals
