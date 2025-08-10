"""
Performance Analysis Module for Backtesting

Provides comprehensive performance metrics, risk analysis, and visualization
capabilities for strategy backtesting results.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

from .backtesting_engine import BacktestResult, BacktestTrade


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results.
    
    Calculates advanced metrics, generates reports, and creates
    visualizations for strategy performance evaluation.
    """
    
    def __init__(self):
        """Initialize the performance analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_results(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of backtest results.
        
        Args:
            result: BacktestResult to analyze
            
        Returns:
            Dictionary containing detailed analysis
        """
        if result.portfolio_history.empty:
            self.logger.warning("No portfolio history to analyze")
            return {}
        
        analysis = {
            'basic_metrics': self._calculate_basic_metrics(result),
            'risk_metrics': self._calculate_risk_metrics(result),
            'trade_analysis': self._analyze_trades(result.trade_history),
            'drawdown_analysis': self._analyze_drawdowns(result.portfolio_history),
            'monthly_returns': self._calculate_monthly_returns(result.portfolio_history),
            'rolling_metrics': self._calculate_rolling_metrics(result.portfolio_history)
        }
        
        return analysis
    
    def _calculate_basic_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        portfolio_values = result.portfolio_history['portfolio_value']
        initial_value = result.config.initial_capital
        final_value = portfolio_values.iloc[-1]
        
        # Calculate returns
        total_return = (final_value - initial_value) / initial_value
        
        # Daily returns
        daily_returns = portfolio_values.pct_change().dropna()
        
        # Time-based metrics
        trading_days = len(result.portfolio_history)
        years = trading_days / 252  # Assuming 252 trading days per year
        
        metrics = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return_pct': total_return * 100,
            'total_return_abs': final_value - initial_value,
            'annualized_return': (1 + total_return) ** (1/years) - 1 if years > 0 else 0,
            'trading_days': trading_days,
            'years': years
        }
        
        return metrics
    
    def _calculate_risk_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        portfolio_values = result.portfolio_history['portfolio_value']
        daily_returns = portfolio_values.pct_change().dropna()
        
        if len(daily_returns) == 0:
            return {}
        
        # Volatility metrics
        daily_vol = daily_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = daily_returns - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns.mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (daily_returns.mean() / downside_vol * np.sqrt(252)) if downside_vol > 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(daily_returns, 5)
        var_99 = np.percentile(daily_returns, 1)
        
        # Expected Shortfall (CVaR)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        cvar_99 = daily_returns[daily_returns <= var_99].mean()
        
        # Maximum drawdown
        cumulative_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        annual_return = self._calculate_basic_metrics(result)['annualized_return']
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        metrics = {
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }
        
        return metrics
    
    def _analyze_trades(self, trades: List[BacktestTrade]) -> Dict[str, Any]:
        """Analyze trade statistics."""
        if not trades:
            return {}
        
        # Basic trade stats
        total_trades = len(trades)
        closed_trades = [t for t in trades if t.is_closed]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        # P&L analysis
        total_pnl = sum(t.pnl for t in closed_trades)
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        
        # Trade metrics
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
        avg_trade = total_pnl / len(closed_trades) if closed_trades else 0
        
        # Trade duration analysis
        durations = []
        for trade in closed_trades:
            if trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                durations.append(duration)
        
        analysis = {
            'total_trades': total_trades,
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'avg_duration_hours': np.mean(durations) if durations else 0,
            'max_duration_hours': max(durations) if durations else 0,
            'min_duration_hours': min(durations) if durations else 0
        }
        
        return analysis
    
    def _analyze_drawdowns(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Analyze drawdown characteristics."""
        portfolio_values = portfolio_history['portfolio_value']
        
        # Calculate drawdown series
        cumulative_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)
        
        # Analyze individual drawdowns
        drawdown_periods = []
        start_idx = None
        
        for i, (is_start, is_end) in enumerate(zip(drawdown_starts, drawdown_ends)):
            if is_start:
                start_idx = i
            elif is_end and start_idx is not None:
                period_drawdown = drawdown.iloc[start_idx:i+1]
                max_dd = abs(period_drawdown.min())
                duration = i - start_idx
                
                drawdown_periods.append({
                    'start_date': portfolio_history.iloc[start_idx]['date'],
                    'end_date': portfolio_history.iloc[i]['date'],
                    'duration_days': duration,
                    'max_drawdown': max_dd
                })
                start_idx = None
        
        # Summary statistics
        if drawdown_periods:
            avg_drawdown = np.mean([dd['max_drawdown'] for dd in drawdown_periods])
            max_drawdown_period = max(drawdown_periods, key=lambda x: x['max_drawdown'])
            longest_drawdown = max(drawdown_periods, key=lambda x: x['duration_days'])
        else:
            avg_drawdown = 0
            max_drawdown_period = None
            longest_drawdown = None
        
        analysis = {
            'num_drawdown_periods': len(drawdown_periods),
            'avg_drawdown': avg_drawdown,
            'max_drawdown_overall': abs(drawdown.min()),
            'max_drawdown_period': max_drawdown_period,
            'longest_drawdown_period': longest_drawdown,
            'current_drawdown': abs(drawdown.iloc[-1]) if drawdown.iloc[-1] < 0 else 0
        }
        
        return analysis
    
    def _calculate_monthly_returns(self, portfolio_history: pd.DataFrame) -> Dict[str, Any]:
        """Calculate monthly return statistics."""
        if 'date' not in portfolio_history.columns:
            return {}
        
        # Set date as index and resample to monthly
        df = portfolio_history.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Get month-end values
        monthly_values = df['portfolio_value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        if len(monthly_returns) == 0:
            return {}
        
        analysis = {
            'num_months': len(monthly_returns),
            'avg_monthly_return': monthly_returns.mean(),
            'monthly_volatility': monthly_returns.std(),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'positive_months': (monthly_returns > 0).sum(),
            'negative_months': (monthly_returns < 0).sum(),
            'monthly_returns_series': monthly_returns.to_dict()
        }
        
        return analysis
    
    def _calculate_rolling_metrics(self, portfolio_history: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
        """Calculate rolling performance metrics."""
        portfolio_values = portfolio_history['portfolio_value']
        
        if len(portfolio_values) < window:
            return {}
        
        # Rolling returns
        rolling_returns = portfolio_values.rolling(window).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
        )
        
        # Rolling volatility
        daily_returns = portfolio_values.pct_change()
        rolling_vol = daily_returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe
        risk_free_rate = 0.02
        excess_returns = daily_returns - (risk_free_rate / 252)
        rolling_sharpe = (
            excess_returns.rolling(window).mean() / 
            daily_returns.rolling(window).std() * np.sqrt(252)
        )
        
        analysis = {
            'window_days': window,
            'avg_rolling_return': rolling_returns.mean(),
            'max_rolling_return': rolling_returns.max(),
            'min_rolling_return': rolling_returns.min(),
            'avg_rolling_volatility': rolling_vol.mean(),
            'avg_rolling_sharpe': rolling_sharpe.mean(),
            'rolling_returns': rolling_returns.dropna().to_list(),
            'rolling_volatility': rolling_vol.dropna().to_list(),
            'rolling_sharpe': rolling_sharpe.dropna().to_list()
        }
        
        return analysis
    
    def generate_report(self, result: BacktestResult, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive text report.
        
        Args:
            result: BacktestResult to analyze
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        analysis = self.analyze_results(result)
        
        report_lines = [
            "=" * 60,
            "BACKTEST PERFORMANCE REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Strategy: {result.strategy_results.get('strategy_name', 'Unknown')}",
            f"Backtest Period: {result.config.start_date.strftime('%Y-%m-%d')} to {result.config.end_date.strftime('%Y-%m-%d')}",
            "",
            "BASIC METRICS",
            "-" * 30
        ]
        
        if 'basic_metrics' in analysis:
            basic = analysis['basic_metrics']
            report_lines.extend([
                f"Initial Capital: ${basic.get('initial_capital', 0):,.2f}",
                f"Final Value: ${basic.get('final_value', 0):,.2f}",
                f"Total Return: {basic.get('total_return_pct', 0):.2f}%",
                f"Annualized Return: {basic.get('annualized_return', 0)*100:.2f}%",
                f"Trading Days: {basic.get('trading_days', 0)}",
                ""
            ])
        
        if 'risk_metrics' in analysis:
            risk = analysis['risk_metrics']
            report_lines.extend([
                "RISK METRICS",
                "-" * 30,
                f"Annual Volatility: {risk.get('annual_volatility', 0)*100:.2f}%",
                f"Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}",
                f"Sortino Ratio: {risk.get('sortino_ratio', 0):.2f}",
                f"Maximum Drawdown: {risk.get('max_drawdown', 0)*100:.2f}%",
                f"Calmar Ratio: {risk.get('calmar_ratio', 0):.2f}",
                f"VaR (95%): {risk.get('var_95', 0)*100:.2f}%",
                ""
            ])
        
        if 'trade_analysis' in analysis:
            trade = analysis['trade_analysis']
            report_lines.extend([
                "TRADE ANALYSIS",
                "-" * 30,
                f"Total Trades: {trade.get('total_trades', 0)}",
                f"Closed Trades: {trade.get('closed_trades', 0)}",
                f"Win Rate: {trade.get('win_rate', 0):.1f}%",
                f"Profit Factor: {trade.get('profit_factor', 0):.2f}",
                f"Average Trade: ${trade.get('avg_trade', 0):.2f}",
                f"Average Win: ${trade.get('avg_win', 0):.2f}",
                f"Average Loss: ${trade.get('avg_loss', 0):.2f}",
                ""
            ])
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {save_path}")
        
        return report_text
