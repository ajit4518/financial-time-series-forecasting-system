"""
Trading Strategy Backtester

This module implements backtesting for trading strategies based on model predictions.
It calculates financial metrics like returns, Sharpe ratio, and drawdown.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def setup_logger():
    """Set up a logger for the backtester."""
    logger = logging.getLogger('backtester')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str):
        """Initialize the trading strategy."""
        self.name = name
        self.logger = setup_logger()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            data: DataFrame with price data and predictions
            
        Returns:
            DataFrame with signals added
        """
        raise NotImplementedError("This method should be implemented by subclasses")

class SimplePredictionStrategy(TradingStrategy):
    """
    A simple strategy that buys when predicted price is higher than current
    and sells when predicted price is lower than current.
    """
    
    def __init__(self, prediction_col: str = 'Predicted', 
                 price_col: str = 'close', 
                 threshold_pct: float = 0.0):
        """
        Initialize the strategy.
        
        Args:
            prediction_col: Column with price predictions
            price_col: Column with current price
            threshold_pct: Threshold percentage for signals (0.01 = 1%)
        """
        super().__init__("Simple Prediction Strategy")
        self.prediction_col = prediction_col
        self.price_col = price_col
        self.threshold_pct = threshold_pct
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on predictions."""
        # Create a copy of the data
        data_with_signals = data.copy()
        
        # Calculate predicted percent change
        data_with_signals['predicted_change_pct'] = (
            data_with_signals[self.prediction_col] / data_with_signals[self.price_col] - 1
        )
        
        # Generate signals (1 = buy, -1 = sell, 0 = hold)
        data_with_signals['signal'] = 0
        
        # Buy signal when predicted change > threshold
        data_with_signals.loc[data_with_signals['predicted_change_pct'] > self.threshold_pct, 'signal'] = 1
        
        # Sell signal when predicted change < -threshold
        data_with_signals.loc[data_with_signals['predicted_change_pct'] < -self.threshold_pct, 'signal'] = -1
        
        return data_with_signals

class MACDStrategy(TradingStrategy):
    """
    MACD-based strategy enhanced with predictions.
    Combines MACD signals with model predictions for better timing.
    """
    
    def __init__(self, prediction_col: str = 'Predicted', 
                 price_col: str = 'close',
                 prediction_weight: float = 0.5,
                 fast_period: int = 12, 
                 slow_period: int = 26, 
                 signal_period: int = 9):
        """
        Initialize the MACD strategy.
        
        Args:
            prediction_col: Column with price predictions
            price_col: Column with current price
            prediction_weight: Weight to give predictions vs. MACD (0-1)
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        """
        super().__init__("MACD Strategy with Predictions")
        self.prediction_col = prediction_col
        self.price_col = price_col
        self.prediction_weight = prediction_weight
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MACD and predictions."""
        # Create a copy of the data
        data_with_signals = data.copy()
        
        # Calculate MACD components if not already present
        if 'macd' not in data_with_signals.columns:
            # Fast EMA
            data_with_signals['ema_fast'] = data_with_signals[self.price_col].ewm(
                span=self.fast_period, adjust=False).mean()
            
            # Slow EMA
            data_with_signals['ema_slow'] = data_with_signals[self.price_col].ewm(
                span=self.slow_period, adjust=False).mean()
            
            # MACD Line
            data_with_signals['macd'] = data_with_signals['ema_fast'] - data_with_signals['ema_slow']
            
            # Signal Line
            data_with_signals['macd_signal'] = data_with_signals['macd'].ewm(
                span=self.signal_period, adjust=False).mean()
            
            # MACD Histogram
            data_with_signals['macd_hist'] = data_with_signals['macd'] - data_with_signals['macd_signal']
        
        # Calculate predicted percent change
        data_with_signals['predicted_change_pct'] = (
            data_with_signals[self.prediction_col] / data_with_signals[self.price_col] - 1
        )
        
        # Generate signals (1 = buy, -1 = sell, 0 = hold)
        data_with_signals['macd_signal_value'] = 0
        
        # MACD buy signal: MACD crosses above signal line
        data_with_signals.loc[
            (data_with_signals['macd'] > data_with_signals['macd_signal']) & 
            (data_with_signals['macd'].shift(1) <= data_with_signals['macd_signal'].shift(1)),
            'macd_signal_value'
        ] = 1
        
        # MACD sell signal: MACD crosses below signal line
        data_with_signals.loc[
            (data_with_signals['macd'] < data_with_signals['macd_signal']) & 
            (data_with_signals['macd'].shift(1) >= data_with_signals['macd_signal'].shift(1)),
            'macd_signal_value'
        ] = -1
        
        # Prediction signal
        data_with_signals['pred_signal_value'] = 0
        data_with_signals.loc[data_with_signals['predicted_change_pct'] > 0.01, 'pred_signal_value'] = 1
        data_with_signals.loc[data_with_signals['predicted_change_pct'] < -0.01, 'pred_signal_value'] = -1
        
        # Combined signal (weighted average)
        data_with_signals['signal'] = (
            (1 - self.prediction_weight) * data_with_signals['macd_signal_value'] + 
            self.prediction_weight * data_with_signals['pred_signal_value']
        )
        
        # Discretize the signal
        data_with_signals['signal'] = np.sign(data_with_signals['signal'])
        
        return data_with_signals

class Backtester:
    """
    Backtester for evaluating trading strategies.
    
    Calculates performance metrics like returns, Sharpe ratio, and drawdown.
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Initial capital
            commission: Commission rate per trade (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.logger = setup_logger()
        
    def backtest(self, data: pd.DataFrame, strategy: TradingStrategy, 
                 price_col: str = 'close') -> Dict:
        """
        Backtest a trading strategy.
        
        Args:
            data: DataFrame with price data
            strategy: Trading strategy
            price_col: Column with price data
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Backtesting strategy: {strategy.name}")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data)
        
        # Initialize positions and portfolio value
        data_with_signals['position'] = data_with_signals['signal'].fillna(0)
        
        # Calculate position changes
        data_with_signals['position_change'] = data_with_signals['position'].diff()
        
        # Calculate transaction costs
        data_with_signals['transaction_cost'] = abs(data_with_signals['position_change']) * \
                                               data_with_signals[price_col] * self.commission
        
        # Calculate strategy returns
        data_with_signals['strategy_returns'] = data_with_signals['position'].shift(1) * \
                                              (data_with_signals[price_col] / data_with_signals[price_col].shift(1) - 1) - \
                                              data_with_signals['transaction_cost'] / data_with_signals[price_col].shift(1)
        
        # Calculate cumulative returns
        data_with_signals['cumulative_returns'] = (1 + data_with_signals['strategy_returns']).cumprod()
        
        # Calculate portfolio value
        data_with_signals['portfolio_value'] = data_with_signals['cumulative_returns'] * self.initial_capital
        
        # Calculate drawdown
        data_with_signals['peak'] = data_with_signals['portfolio_value'].cummax()
        data_with_signals['drawdown'] = (data_with_signals['portfolio_value'] - data_with_signals['peak']) / data_with_signals['peak']
        
        # Calculate performance metrics
        total_return = data_with_signals['cumulative_returns'].iloc[-1] - 1
        annualized_return = self._calculate_annualized_return(data_with_signals)
        sharpe_ratio = self._calculate_sharpe_ratio(data_with_signals)
        max_drawdown = data_with_signals['drawdown'].min()
        win_rate = self._calculate_win_rate(data_with_signals)
        profit_factor = self._calculate_profit_factor(data_with_signals)
        
        # Calculate trade statistics
        trade_stats = self._calculate_trade_statistics(data_with_signals)
        
        self.logger.info(f"Backtest completed with total return: {total_return:.2%}")
        
        # Prepare results
        results = {
            'strategy_name': strategy.name,
            'data_with_signals': data_with_signals,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trade_stats': trade_stats
        }
        
        return results
    
    def _calculate_annualized_return(self, data: pd.DataFrame) -> float:
        """Calculate annualized return."""
        # Assuming daily data
        total_return = data['cumulative_returns'].iloc[-1] - 1
        num_days = len(data)
        
        # Convert to annualized return
        if 'timestamp' in data.columns and pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            # Calculate actual days between first and last timestamp
            total_days = (data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]).days
            if total_days > 0:
                years = total_days / 365.25
                return (1 + total_return) ** (1 / years) - 1
        
        # Default: assume daily data
        years = num_days / 252  # Trading days in a year
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_sharpe_ratio(self, data: pd.DataFrame) -> float:
        """Calculate Sharpe ratio."""
        if len(data) < 2:
            return 0.0
            
        # Calculate excess returns (assuming risk-free rate = 0)
        excess_returns = data['strategy_returns']
        
        # Calculate Sharpe ratio (annualized)
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        return sharpe_ratio
    
    def _calculate_win_rate(self, data: pd.DataFrame) -> float:
        """Calculate win rate."""
        trades = data[data['position_change'] != 0]
        if len(trades) == 0:
            return 0.0
            
        # Calculate returns for each trade
        trades['trade_return'] = trades['strategy_returns']
        
        # Calculate win rate
        win_rate = len(trades[trades['trade_return'] > 0]) / len(trades)
        
        return win_rate
    
    def _calculate_profit_factor(self, data: pd.DataFrame) -> float:
        """Calculate profit factor."""
        trades = data[data['position_change'] != 0]
        if len(trades) == 0:
            return 0.0
            
        # Calculate returns for each trade
        trades['trade_return'] = trades['strategy_returns']
        
        # Calculate profit factor
        winning_trades = trades[trades['trade_return'] > 0]
        losing_trades = trades[trades['trade_return'] < 0]
        
        if len(losing_trades) == 0 or abs(losing_trades['trade_return'].sum()) == 0:
            return float('inf')  # No losing trades
            
        profit_factor = abs(winning_trades['trade_return'].sum() / losing_trades['trade_return'].sum())
        
        return profit_factor
    
    def _calculate_trade_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate trade statistics."""
        position_changes = data[data['position_change'] != 0]
        
        if len(position_changes) == 0:
            return {
                'num_trades': 0,
                'avg_trade_duration': 0,
                'avg_profit_per_trade': 0,
                'largest_winner': 0,
                'largest_loser': 0
            }
        
        # Count trades (entry + exit = 1 trade)
        num_trades = len(position_changes) / 2
        
        # Calculate average trade duration
        # This is approximate and assumes consecutive position changes form a trade
        if 'timestamp' in data.columns and pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            trade_durations = []
            entry_time = None
            
            for idx, row in position_changes.iterrows():
                if entry_time is None:
                    entry_time = row['timestamp']
                else:
                    exit_time = row['timestamp']
                    duration = (exit_time - entry_time).total_seconds() / (60 * 60 * 24)  # in days
                    trade_durations.append(duration)
                    entry_time = None
            
            avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        else:
            avg_trade_duration = 0
        
        # Calculate trade profitability
        trade_returns = data['strategy_returns']
        avg_profit_per_trade = trade_returns.mean() if len(trade_returns) > 0 else 0
        largest_winner = trade_returns.max() if len(trade_returns) > 0 else 0
        largest_loser = trade_returns.min() if len(trade_returns) > 0 else 0
        
        return {
            'num_trades': num_trades,
            'avg_trade_duration': avg_trade_duration,
            'avg_profit_per_trade': avg_profit_per_trade,
            'largest_winner': largest_winner,
            'largest_loser': largest_loser
        }
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results.
        
        Args:
            results: Backtest results dictionary
            save_path: Path to save the plot
        """
        data = results['data_with_signals']
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot portfolio value
        ax1.plot(data.index, data['portfolio_value'], label='Portfolio Value')
        ax1.set_title(f"Strategy: {results['strategy_name']}")
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot drawdown
        ax2.fill_between(data.index, 0, data['drawdown'] * 100, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(data['drawdown'].min() * 100 - 5, 5)
        ax2.grid(True)
        
        # Plot positions
        ax3.plot(data.index, data['position'], label='Position', color='green')
        ax3.set_ylabel('Position')
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_xlabel('Time')
        ax3.grid(True)
        
        # Add performance metrics as text
        performance_text = (
            f"Total Return: {results['total_return']:.2%}\n"
            f"Annualized Return: {results['annualized_return']:.2%}\n"
            f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {results['max_drawdown']:.2%}\n"
            f"Win Rate: {results['win_rate']:.2%}\n"
            f"Profit Factor: {results['profit_factor']:.2f}\n"
            f"Number of Trades: {results['trade_stats']['num_trades']:.0f}"
        )
        
        fig.text(0.15, 0.01, performance_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def run_backtest(prediction_file_path: str, symbol: str = None):
    """
    Run a backtest using predictions.
    
    Args:
        prediction_file_path: Path to prediction file
        symbol: Symbol to use in the plot title
    """
    logger = setup_logger()
    
    # Load prediction data
    try:
        predictions = pd.read_csv(prediction_file_path)
        logger.info(f"Loaded predictions from {prediction_file_path}")
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        return
    
    # Ensure we have required columns
    required_cols = ['Actual', 'Predicted']
    if not all(col in predictions.columns for col in required_cols):
        logger.error(f"Missing required columns in predictions file. Need: {required_cols}")
        return
    
    # Add close column if it doesn't exist (use Actual)
    if 'close' not in predictions.columns:
        predictions['close'] = predictions['Actual']
    
    # Add timestamp if it doesn't exist
    if 'timestamp' not in predictions.columns:
        predictions['timestamp'] = pd.date_range(start='2023-01-01', periods=len(predictions), freq='D')
    
    # Create backtester
    backtester = Backtester(initial_capital=100000, commission=0.001)
    
    # Create strategies
    simple_strategy = SimplePredictionStrategy(threshold_pct=0.005)
    macd_strategy = MACDStrategy(prediction_weight=0.7)
    
    # Run backtests
    simple_results = backtester.backtest(predictions, simple_strategy)
    macd_results = backtester.backtest(predictions, macd_strategy)
    
    # Create results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    results_dir = os.path.join(project_root, 'results')
    plots_dir = os.path.join(project_root, 'plots')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save backtest results
    symbol_text = f"{symbol}_" if symbol else ""
    
    # Plot and save results
    simple_plot_path = os.path.join(plots_dir, f"{symbol_text}simple_strategy_backtest_{timestamp}.png")
    backtester.plot_results(simple_results, save_path=simple_plot_path)
    
    macd_plot_path = os.path.join(plots_dir, f"{symbol_text}macd_strategy_backtest_{timestamp}.png")
    backtester.plot_results(macd_results, save_path=macd_plot_path)
    
    # Save strategy results to CSV
    simple_results_path = os.path.join(results_dir, f"{symbol_text}simple_strategy_results_{timestamp}.csv")
    simple_results['data_with_signals'].to_csv(simple_results_path, index=False)
    
    macd_results_path = os.path.join(results_dir, f"{symbol_text}macd_strategy_results_{timestamp}.csv")
    macd_results['data_with_signals'].to_csv(macd_results_path, index=False)
    
    # Print summary
    print("\nBacktesting Results:")
    print("\nSimple Prediction Strategy:")
    print(f"Total Return: {simple_results['total_return']:.2%}")
    print(f"Annualized Return: {simple_results['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {simple_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {simple_results['max_drawdown']:.2%}")
    print(f"Win Rate: {simple_results['win_rate']:.2%}")
    print(f"Number of Trades: {simple_results['trade_stats']['num_trades']:.0f}")
    
    print("\nMACD Strategy with Predictions:")
    print(f"Total Return: {macd_results['total_return']:.2%}")
    print(f"Annualized Return: {macd_results['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {macd_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {macd_results['max_drawdown']:.2%}")
    print(f"Win Rate: {macd_results['win_rate']:.2%}")
    print(f"Number of Trades: {macd_results['trade_stats']['num_trades']:.0f}")
    
    print(f"\nResults saved to {results_dir}")
    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run backtest on prediction results')
    parser.add_argument('--predictions', type=str, help='Path to prediction results CSV file')
    parser.add_argument('--symbol', type=str, help='Symbol for the backtest (optional)')
    
    args = parser.parse_args()
    
    # If no prediction file is provided, try to find one
    if not args.predictions:
        # Get the current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        results_dir = os.path.join(project_root, 'results')
        
        if os.path.exists(results_dir):
            # Find most recent inference results
            result_files = [f for f in os.listdir(results_dir) if f.endswith('_inference_results.csv') 
                           or 'inference_results' in f]
            
            if result_files:
                # Sort by modification time (most recent first)
                result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
                args.predictions = os.path.join(results_dir, result_files[0])
                
                # Extract symbol from filename
                if not args.symbol and '_' in result_files[0]:
                    args.symbol = result_files[0].split('_')[0]
    
    if args.predictions:
        run_backtest(args.predictions, args.symbol)
    else:
        print("Error: No prediction file provided or found automatically.")
        print("Please specify a prediction file with --predictions or run model inference first.")