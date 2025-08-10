"""
Pairs Trading Scanner

Identifies correlated stock pairs for pairs trading strategies.
Analyzes correlations, spread characteristics, and entry/exit signals.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from itertools import combinations

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config_manager import config
from src.screening.universe import UniverseManager
from src.data_management.historical_provider import HistoricalDataManager


class TradingPair:
    """Represents a trading pair with correlation and spread analysis."""
    
    def __init__(
        self,
        symbol_a: str,
        symbol_b: str,
        correlation: float,
        spread_mean: float,
        spread_std: float,
        current_spread: float,
        z_score: float,
        hedge_ratio: float,
        data_points: int,
        last_updated: datetime
    ):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.correlation = correlation
        self.spread_mean = spread_mean
        self.spread_std = spread_std
        self.current_spread = current_spread
        self.z_score = z_score
        self.hedge_ratio = hedge_ratio
        self.data_points = data_points
        self.last_updated = last_updated
        
        # Trading signal determination
        self.signal = self._determine_signal()
        self.signal_strength = abs(self.z_score)
    
    def _determine_signal(self) -> str:
        """Determine trading signal based on z-score."""
        z_threshold = config.get('PAIRS_TRADING', {}).get('Z_SCORE_THRESHOLD', 2.0)
        
        if self.z_score > z_threshold:
            return 'short_a_long_b'  # Short A, Long B
        elif self.z_score < -z_threshold:
            return 'long_a_short_b'  # Long A, Short B
        else:
            return 'no_signal'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pair to dictionary."""
        return {
            'symbol_a': self.symbol_a,
            'symbol_b': self.symbol_b,
            'correlation': self.correlation,
            'spread_mean': self.spread_mean,
            'spread_std': self.spread_std,
            'current_spread': self.current_spread,
            'z_score': self.z_score,
            'hedge_ratio': self.hedge_ratio,
            'signal': self.signal,
            'signal_strength': self.signal_strength,
            'data_points': self.data_points,
            'last_updated': self.last_updated
        }


class PairsOpportunity:
    """Represents a pairs trading opportunity."""
    
    def __init__(
        self,
        pair: TradingPair,
        entry_signal: str,
        confidence: float,
        position_size_a: float,
        position_size_b: float,
        expected_return: float,
        max_risk: float,
        timestamp: datetime
    ):
        self.pair = pair
        self.entry_signal = entry_signal
        self.confidence = confidence
        self.position_size_a = position_size_a
        self.position_size_b = position_size_b
        self.expected_return = expected_return
        self.max_risk = max_risk
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert opportunity to dictionary."""
        return {
            'pair': self.pair.to_dict(),
            'entry_signal': self.entry_signal,
            'confidence': self.confidence,
            'position_size_a': self.position_size_a,
            'position_size_b': self.position_size_b,
            'expected_return': self.expected_return,
            'max_risk': self.max_risk,
            'timestamp': self.timestamp
        }


class PairsTradingScanner:
    """Scans for pairs trading opportunities."""
    
    def __init__(
        self,
        universe_manager: Optional[UniverseManager] = None,
        data_manager: Optional[HistoricalDataManager] = None
    ):
        """
        Initialize pairs trading scanner.
        
        Args:
            universe_manager: Universe manager for stock selection
            data_manager: Data manager for historical data access
        """
        self.universe_manager = universe_manager or UniverseManager()
        self.data_manager = data_manager or HistoricalDataManager()
        
        # Configuration
        self.config = config.get('PAIRS_TRADING', {})
        self.correlation_threshold = self.config.get('CORRELATION_THRESHOLD', 0.7)
        self.correlation_lookback = self.config.get('CORRELATION_LOOKBACK', 126)  # 6 months
        self.z_score_threshold = self.config.get('Z_SCORE_THRESHOLD', 2.0)
        self.min_data_points = self.config.get('MIN_DATA_POINTS', 100)
        self.spread_lookback = self.config.get('SPREAD_LOOKBACK', 60)  # 60 days for spread analysis
        
        # Cache for pairs analysis
        self._pairs_cache: Dict[Tuple[str, str], TradingPair] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)  # Cache for 1 hour
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def find_correlated_pairs(
        self,
        symbols: Optional[List[str]] = None,
        min_correlation: Optional[float] = None
    ) -> List[TradingPair]:
        """
        Find correlated pairs in the universe.
        
        Args:
            symbols: Optional list of symbols to analyze (uses universe if None)
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of correlated trading pairs
        """
        # Get symbols to analyze
        if symbols is None:
            symbols = list(self.universe_manager.get_universe(apply_filters=True))
        
        if len(symbols) < 2:
            self.logger.warning("Need at least 2 symbols for pairs analysis")
            return []
        
        correlation_threshold = min_correlation or self.correlation_threshold
        pairs = []
        
        # Get historical data for all symbols
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.correlation_lookback)
        
        price_data = {}
        self.logger.info(f"Loading price data for {len(symbols)} symbols")
        
        for symbol in symbols:
            data = self.data_manager.get_symbol_data(symbol, start_date, end_date)
            if data is not None and len(data) >= self.min_data_points:
                price_data[symbol] = data['close']
            else:
                self.logger.debug(f"Insufficient data for {symbol}")
        
        if len(price_data) < 2:
            self.logger.warning("Insufficient price data for pairs analysis")
            return []
        
        # Align all price series
        price_df = pd.DataFrame(price_data).ffill().dropna()
        
        if len(price_df) < self.min_data_points:
            self.logger.warning("Insufficient aligned data points")
            return []
        
        # Calculate correlations for all pairs
        self.logger.info(f"Analyzing {len(list(combinations(price_df.columns, 2)))} potential pairs")
        
        for symbol_a, symbol_b in combinations(price_df.columns, 2):
            try:
                pair = self._analyze_pair(symbol_a, symbol_b, price_df)
                if pair and pair.correlation >= correlation_threshold:
                    pairs.append(pair)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing pair {symbol_a}-{symbol_b}: {e}")
                continue
        
        # Sort by correlation (descending)
        pairs.sort(key=lambda x: x.correlation, reverse=True)
        
        self.logger.info(f"Found {len(pairs)} correlated pairs (correlation >= {correlation_threshold:.2f})")
        
        return pairs
    
    def _analyze_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        price_df: pd.DataFrame
    ) -> Optional[TradingPair]:
        """
        Analyze a specific pair for correlation and spread characteristics.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            price_df: DataFrame with price data for both symbols
            
        Returns:
            TradingPair object if analysis successful, None otherwise
        """
        prices_a = price_df[symbol_a]
        prices_b = price_df[symbol_b]
        
        # Calculate correlation
        correlation = prices_a.corr(prices_b)
        
        if pd.isna(correlation) or abs(correlation) < self.correlation_threshold:
            return None
        
        # Calculate hedge ratio using simple linear regression
        # We want to find beta such that: price_a = alpha + beta * price_b + error
        
        # Simple implementation without sklearn dependency
        # Calculate correlation and variance for hedge ratio
        covariance = np.cov(prices_a, prices_b)[0, 1]
        variance_b = np.var(prices_b)
        
        if variance_b > 0:
            hedge_ratio = covariance / variance_b
        else:
            hedge_ratio = 1.0  # Default to 1:1 ratio
        
        # Remove any NaN values for data quality check
        valid_mask = ~(pd.isna(prices_a) | pd.isna(prices_b))
        valid_points = np.sum(valid_mask)
        
        if valid_points < self.min_data_points:
            return None
        
        # Calculate spread (residuals)
        spread = prices_a - hedge_ratio * prices_b
        
        # Use only recent data for spread analysis
        recent_spread = spread.tail(self.spread_lookback)
        
        # Calculate spread statistics
        spread_mean = recent_spread.mean()
        spread_std = recent_spread.std()
        current_spread = spread.iloc[-1]
        
        # Calculate z-score
        if spread_std > 0:
            z_score = (current_spread - spread_mean) / spread_std
        else:
            z_score = 0
        
        return TradingPair(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            correlation=correlation,
            spread_mean=spread_mean,
            spread_std=spread_std,
            current_spread=current_spread,
            z_score=z_score,
            hedge_ratio=hedge_ratio,
            data_points=valid_points,
            last_updated=datetime.now()
        )
    
    def scan_for_opportunities(
        self,
        symbols: Optional[List[str]] = None,
        min_signal_strength: float = 2.0
    ) -> List[PairsOpportunity]:
        """
        Scan for pairs trading opportunities.
        
        Args:
            symbols: Optional list of symbols to scan
            min_signal_strength: Minimum signal strength (z-score threshold)
            
        Returns:
            List of pairs trading opportunities
        """
        # Get correlated pairs
        pairs = self.find_correlated_pairs(symbols)
        
        if not pairs:
            return []
        
        opportunities = []
        
        for pair in pairs:
            if pair.signal != 'no_signal' and pair.signal_strength >= min_signal_strength:
                opportunity = self._create_opportunity(pair)
                if opportunity:
                    opportunities.append(opportunity)
        
        # Sort by signal strength (descending)
        opportunities.sort(key=lambda x: x.pair.signal_strength, reverse=True)
        
        self.logger.info(f"Found {len(opportunities)} pairs trading opportunities")
        
        return opportunities
    
    def _create_opportunity(self, pair: TradingPair) -> Optional[PairsOpportunity]:
        """
        Create a pairs trading opportunity from a trading pair.
        
        Args:
            pair: TradingPair object
            
        Returns:
            PairsOpportunity if viable, None otherwise
        """
        # Get current prices
        data_a = self.data_manager.get_symbol_data(pair.symbol_a)
        data_b = self.data_manager.get_symbol_data(pair.symbol_b)
        
        if data_a is None or data_b is None:
            return None
        
        price_a = data_a['close'].iloc[-1]
        price_b = data_b['close'].iloc[-1]
        
        # Calculate position sizes
        # For a market-neutral strategy, we want equal dollar amounts
        # Position size A: number of shares of A
        # Position size B: number of shares of B
        
        if pair.signal == 'long_a_short_b':
            # Long A, Short B
            direction_a = 1  # Buy
            direction_b = -1  # Sell
        elif pair.signal == 'short_a_long_b':
            # Short A, Long B
            direction_a = -1  # Sell
            direction_b = 1  # Buy
        else:
            return None
        
        # Calculate equal dollar position sizes
        base_amount = 10000  # $10,000 base position
        
        # For market neutral: |shares_a * price_a| = |shares_b * price_b|
        # And we use the hedge ratio: shares_a = hedge_ratio * shares_b
        
        shares_b = base_amount / (price_b * (1 + abs(pair.hedge_ratio * price_a / price_b)))
        shares_a = pair.hedge_ratio * shares_b
        
        position_size_a = direction_a * abs(shares_a)
        position_size_b = direction_b * abs(shares_b)
        
        # Estimate expected return based on mean reversion
        # Assuming spread will revert to mean
        expected_spread_move = -pair.current_spread + pair.spread_mean
        expected_return = abs(expected_spread_move) * abs(shares_a) / base_amount
        
        # Estimate maximum risk (2 standard deviations)
        max_risk = 2 * pair.spread_std * abs(shares_a) / base_amount
        
        # Calculate confidence based on signal strength
        confidence = min(0.95, pair.signal_strength / 3.0)  # Cap at 95%
        
        return PairsOpportunity(
            pair=pair,
            entry_signal=pair.signal,
            confidence=confidence,
            position_size_a=position_size_a,
            position_size_b=position_size_b,
            expected_return=expected_return,
            max_risk=max_risk,
            timestamp=datetime.now()
        )
    
    def get_top_opportunities(
        self,
        limit: int = 10,
        min_signal_strength: float = 2.0
    ) -> List[PairsOpportunity]:
        """
        Get top pairs trading opportunities.
        
        Args:
            limit: Maximum number of opportunities to return
            min_signal_strength: Minimum signal strength threshold
            
        Returns:
            List of top opportunities
        """
        opportunities = self.scan_for_opportunities(min_signal_strength=min_signal_strength)
        return opportunities[:limit]
    
    def monitor_existing_pairs(
        self,
        current_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Monitor existing pairs positions for exit signals.
        
        Args:
            current_pairs: List of current pairs positions
            
        Returns:
            List of exit signals
        """
        exit_signals = []
        
        for position in current_pairs:
            symbol_a = position['symbol_a']
            symbol_b = position['symbol_b']
            entry_z_score = position['entry_z_score']
            entry_date = position['entry_date']
            
            # Get current pair analysis
            pairs = self.find_correlated_pairs([symbol_a, symbol_b])
            
            if not pairs:
                # Can't analyze - suggest exit due to data issues
                exit_signals.append({
                    'symbol_a': symbol_a,
                    'symbol_b': symbol_b,
                    'exit_reason': 'data_unavailable',
                    'urgency': 'high',
                    'current_z_score': None
                })
                continue
            
            pair = pairs[0]  # Should only be one pair
            current_z_score = pair.z_score
            
            # Check exit conditions
            exit_reason = None
            urgency = 'low'
            
            # Z-score mean reversion (main exit signal)
            if abs(current_z_score) < 0.5:  # Close to mean
                exit_reason = 'mean_reversion'
                urgency = 'medium'
            
            # Z-score reversal (against original direction)
            elif (entry_z_score > 0 and current_z_score < -1.0) or \
                 (entry_z_score < 0 and current_z_score > 1.0):
                exit_reason = 'reversal'
                urgency = 'high'
            
            # Time-based exit (pairs positions shouldn't be held too long)
            elif (datetime.now() - entry_date).days > 30:
                exit_reason = 'time_limit'
                urgency = 'medium'
            
            # Correlation breakdown
            elif abs(pair.correlation) < 0.5:
                exit_reason = 'correlation_breakdown'
                urgency = 'high'
            
            if exit_reason:
                exit_signals.append({
                    'symbol_a': symbol_a,
                    'symbol_b': symbol_b,
                    'exit_reason': exit_reason,
                    'urgency': urgency,
                    'current_z_score': current_z_score,
                    'entry_z_score': entry_z_score
                })
        
        return exit_signals
    
    def get_pairs_summary(self) -> Dict[str, Any]:
        """Get summary of pairs trading analysis."""
        pairs = self.find_correlated_pairs()
        opportunities = self.scan_for_opportunities()
        
        if not pairs:
            return {
                'total_pairs': 0,
                'avg_correlation': 0.0,
                'opportunities': 0,
                'strong_signals': 0,
                'top_pairs': []
            }
        
        strong_signals = len([op for op in opportunities if op.pair.signal_strength >= 2.5])
        
        return {
            'total_pairs': len(pairs),
            'avg_correlation': np.mean([p.correlation for p in pairs]),
            'opportunities': len(opportunities),
            'strong_signals': strong_signals,
            'top_pairs': [pair.to_dict() for pair in pairs[:5]]
        }
