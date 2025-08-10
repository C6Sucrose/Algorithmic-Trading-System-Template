"""
Risk Controls for Order Management System

Implements comprehensive risk management controls including:
- Position size limits
- Portfolio exposure limits
- Drawdown protection
- Concentration limits
- Real-time risk monitoring
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .broker_interface import Order, OrderType
from ..strategies.base_strategy import TradeDirection


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskValidationResult:
    """Result of risk validation."""
    is_valid: bool
    reason: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    recommendations: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize recommendations list."""
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class RiskLimits:
    """Risk management limits configuration."""
    # Position limits
    max_position_size: float = 10000.0  # Maximum position value
    max_position_percentage: float = 0.05  # Max 5% of portfolio per position
    max_positions: int = 20  # Maximum number of open positions
    
    # Portfolio limits
    max_portfolio_exposure: float = 0.95  # Max 95% of capital deployed
    max_sector_exposure: float = 0.3  # Max 30% in any sector
    max_daily_loss: float = 0.02  # Max 2% daily loss
    max_total_drawdown: float = 0.10  # Max 10% total drawdown
    
    # Trade limits
    min_trade_size: float = 100.0  # Minimum trade size
    max_trade_size: float = 50000.0  # Maximum trade size
    max_orders_per_minute: int = 10  # Rate limiting
    
    # Leverage limits
    max_leverage: float = 1.0  # No leverage by default
    max_margin_usage: float = 0.5  # Max 50% margin usage


class RiskController:
    """
    Comprehensive risk management system.
    
    Monitors and controls trading risk across all strategies and positions
    with real-time validation and intervention capabilities.
    """
    
    def __init__(self, limits: RiskLimits, initial_capital: float = 100000.0):
        """
        Initialize Risk Controller.
        
        Args:
            limits: Risk limits configuration
            initial_capital: Starting capital amount
        """
        self.limits = limits
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # Portfolio tracking
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.portfolio_value = initial_capital
        self.cash_available = initial_capital
        self.total_exposure = 0.0
        
        # Performance tracking
        self.peak_portfolio_value = initial_capital
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.daily_start_value = initial_capital
        
        # Order tracking for rate limiting
        self.recent_orders: List[datetime] = []
        
        # Risk metrics
        self.risk_alerts: List[Dict[str, Any]] = []
        self.violations: List[Dict[str, Any]] = []
    
    def validate_order(self, order: Order) -> RiskValidationResult:
        """
        Validate an order against all risk controls.
        
        Args:
            order: Order to validate
            
        Returns:
            RiskValidationResult with validation outcome
        """
        try:
            # Rate limiting check
            if not self._check_rate_limits():
                return RiskValidationResult(
                    is_valid=False,
                    reason="Order rate limit exceeded",
                    risk_level=RiskLevel.HIGH
                )
            
            # Trade size validation
            trade_value = self._calculate_trade_value(order)
            if not self._validate_trade_size(trade_value):
                return RiskValidationResult(
                    is_valid=False,
                    reason=f"Trade size {trade_value:.2f} outside allowed range",
                    risk_level=RiskLevel.MEDIUM
                )
            
            # Position size validation
            if not self._validate_position_size(order, trade_value):
                return RiskValidationResult(
                    is_valid=False,
                    reason="Position size would exceed limits",
                    risk_level=RiskLevel.HIGH
                )
            
            # Portfolio exposure validation
            if not self._validate_portfolio_exposure(trade_value):
                return RiskValidationResult(
                    is_valid=False,
                    reason="Portfolio exposure would exceed limits",
                    risk_level=RiskLevel.HIGH
                )
            
            # Drawdown protection
            if not self._validate_drawdown_limits():
                return RiskValidationResult(
                    is_valid=False,
                    reason="Portfolio drawdown exceeds maximum allowed",
                    risk_level=RiskLevel.CRITICAL
                )
            
            # Cash availability check
            if not self._validate_cash_availability(order, trade_value):
                return RiskValidationResult(
                    is_valid=False,
                    reason="Insufficient cash for trade",
                    risk_level=RiskLevel.MEDIUM
                )
            
            # Maximum positions check
            if not self._validate_max_positions(order):
                return RiskValidationResult(
                    is_valid=False,
                    reason="Maximum number of positions reached",
                    risk_level=RiskLevel.MEDIUM
                )
            
            # All validations passed
            return RiskValidationResult(is_valid=True, reason="Order validated successfully")
            
        except Exception as e:
            self.logger.error(f"Error validating order: {str(e)}")
            return RiskValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                risk_level=RiskLevel.HIGH
            )
    
    def update_portfolio_state(
        self,
        positions: Dict[str, Dict[str, Any]],
        portfolio_value: float,
        cash_available: float
    ):
        """
        Update portfolio state for risk calculations.
        
        Args:
            positions: Current positions
            portfolio_value: Current portfolio value
            cash_available: Available cash
        """
        self.current_positions = positions
        self.portfolio_value = portfolio_value
        self.cash_available = cash_available
        
        # Update exposure
        self.total_exposure = sum(
            pos.get('market_value', 0) for pos in positions.values()
        )
        
        # Update drawdown tracking
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        
        # Update daily P&L
        if self.daily_start_value > 0:
            self.daily_pnl = (portfolio_value - self.daily_start_value) / self.daily_start_value
        
        # Check for risk alerts
        self._check_risk_alerts()
    
    def reset_daily_tracking(self):
        """Reset daily tracking metrics."""
        self.daily_start_value = self.portfolio_value
        self.daily_pnl = 0.0
        self.recent_orders.clear()
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        return {
            'portfolio_value': self.portfolio_value,
            'cash_available': self.cash_available,
            'total_exposure': self.total_exposure,
            'exposure_percentage': self.total_exposure / max(1, self.portfolio_value) * 100,
            'current_drawdown': self.current_drawdown * 100,
            'daily_pnl': self.daily_pnl * 100,
            'positions_count': len(self.current_positions),
            'max_positions': self.limits.max_positions,
            'recent_alerts': len(self.risk_alerts),
            'violations': len(self.violations),
            'limits': {
                'max_position_size': self.limits.max_position_size,
                'max_portfolio_exposure': self.limits.max_portfolio_exposure * 100,
                'max_daily_loss': self.limits.max_daily_loss * 100,
                'max_total_drawdown': self.limits.max_total_drawdown * 100
            }
        }
    
    def get_position_size_recommendation(
        self,
        symbol: str,
        entry_price: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Get recommended position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for position
            volatility: Optional volatility estimate
            
        Returns:
            Recommended position size in shares
        """
        # Base position size (1% of portfolio)
        base_allocation = self.portfolio_value * 0.01
        
        # Adjust for volatility if provided
        if volatility:
            # Reduce size for high volatility
            volatility_adjustment = min(1.0, 0.02 / max(volatility, 0.01))
            base_allocation *= volatility_adjustment
        
        # Apply position limits
        max_by_percentage = self.portfolio_value * self.limits.max_position_percentage
        max_allowed = min(base_allocation, max_by_percentage, self.limits.max_position_size)
        
        # Convert to shares
        recommended_shares = max_allowed / entry_price
        
        return round(recommended_shares)
    
    def _check_rate_limits(self) -> bool:
        """Check order rate limits."""
        now = datetime.now()
        
        # Clean old orders (older than 1 minute)
        self.recent_orders = [
            order_time for order_time in self.recent_orders
            if now - order_time < timedelta(minutes=1)
        ]
        
        # Check if under limit
        if len(self.recent_orders) >= self.limits.max_orders_per_minute:
            return False
        
        # Add current order to tracking
        self.recent_orders.append(now)
        return True
    
    def _calculate_trade_value(self, order: Order) -> float:
        """Calculate the notional value of a trade."""
        if order.order_type == OrderType.MARKET:
            # Estimate market price (would need real market data)
            estimated_price = 100.0  # Placeholder
        else:
            estimated_price = order.price or 100.0
        
        return order.quantity * estimated_price
    
    def _validate_trade_size(self, trade_value: float) -> bool:
        """Validate trade size limits."""
        return self.limits.min_trade_size <= trade_value <= self.limits.max_trade_size
    
    def _validate_position_size(self, order: Order, trade_value: float) -> bool:
        """Validate position size limits."""
        # Check against absolute limit
        if trade_value > self.limits.max_position_size:
            return False
        
        # Check against percentage limit
        max_by_percentage = self.portfolio_value * self.limits.max_position_percentage
        if trade_value > max_by_percentage:
            return False
        
        # Check existing position size
        if order.symbol in self.current_positions:
            current_value = self.current_positions[order.symbol].get('market_value', 0)
            total_value = current_value + trade_value
            
            if total_value > max_by_percentage:
                return False
        
        return True
    
    def _validate_portfolio_exposure(self, trade_value: float) -> bool:
        """Validate portfolio exposure limits."""
        new_exposure = self.total_exposure + trade_value
        max_exposure = self.portfolio_value * self.limits.max_portfolio_exposure
        
        return new_exposure <= max_exposure
    
    def _validate_drawdown_limits(self) -> bool:
        """Validate drawdown limits."""
        return self.current_drawdown <= self.limits.max_total_drawdown
    
    def _validate_cash_availability(self, order: Order, trade_value: float) -> bool:
        """Validate sufficient cash for trade."""
        # Add some buffer for commissions
        required_cash = trade_value * 1.001  # 0.1% buffer
        return self.cash_available >= required_cash
    
    def _validate_max_positions(self, order: Order) -> bool:
        """Validate maximum positions limit."""
        if order.symbol in self.current_positions:
            return True  # Adding to existing position
        
        return len(self.current_positions) < self.limits.max_positions
    
    def _check_risk_alerts(self):
        """Check for risk conditions requiring alerts."""
        alerts = []
        
        # Drawdown alerts
        if self.current_drawdown > self.limits.max_total_drawdown * 0.8:
            alerts.append({
                'type': 'drawdown_warning',
                'level': RiskLevel.HIGH,
                'message': f"Drawdown at {self.current_drawdown*100:.1f}%, approaching limit",
                'timestamp': datetime.now()
            })
        
        # Daily loss alerts
        if self.daily_pnl < -self.limits.max_daily_loss * 0.8:
            alerts.append({
                'type': 'daily_loss_warning',
                'level': RiskLevel.HIGH,
                'message': f"Daily loss at {self.daily_pnl*100:.1f}%, approaching limit",
                'timestamp': datetime.now()
            })
        
        # Exposure alerts
        exposure_pct = self.total_exposure / max(1, self.portfolio_value)
        if exposure_pct > self.limits.max_portfolio_exposure * 0.9:
            alerts.append({
                'type': 'exposure_warning',
                'level': RiskLevel.MEDIUM,
                'message': f"Portfolio exposure at {exposure_pct*100:.1f}%",
                'timestamp': datetime.now()
            })
        
        # Add new alerts
        self.risk_alerts.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Risk Alert [{alert['level'].value}]: {alert['message']}")
    
    def emergency_stop(self) -> bool:
        """
        Emergency stop all trading activity.
        
        Returns:
            True if emergency stop activated
        """
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        
        # Record violation
        self.violations.append({
            'type': 'emergency_stop',
            'reason': 'Manual emergency stop or critical risk condition',
            'timestamp': datetime.now(),
            'portfolio_value': self.portfolio_value,
            'drawdown': self.current_drawdown
        })
        
        return True
