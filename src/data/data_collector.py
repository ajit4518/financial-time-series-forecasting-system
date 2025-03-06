"""
Data Collector Module

This module generates synthetic market data for testing the time series forecasting engine.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def setup_logger():
    """Set up a basic logger for the data collector."""
    logger = logging.getLogger('data_collector')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class DataCollector:
    """
    A class for collecting and generating synthetic market data.
    """
    
    def __init__(self):
        """Initialize the DataCollector."""
        self.logger = setup_logger()
        
        # Get the current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create paths using absolute paths instead of relative
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        self.raw_dir = os.path.join(project_root, 'data', 'raw')
        self.processed_dir = os.path.join(project_root, 'data', 'processed')
        
        # Create directories with proper permissions
        try:
            os.makedirs(self.raw_dir, exist_ok=True)
            os.makedirs(self.processed_dir, exist_ok=True)
            self.logger.info(f"DataCollector initialized with raw data directory: {self.raw_dir}")
        except PermissionError:
            # Fallback to a directory in the user's home folder if permission issues persist
            user_home = os.path.expanduser("~")
            project_dir = os.path.join(user_home, "time_series_forecasting_data")
            
            self.raw_dir = os.path.join(project_dir, 'raw')
            self.processed_dir = os.path.join(project_dir, 'processed')
            
            os.makedirs(self.raw_dir, exist_ok=True)
            os.makedirs(self.processed_dir, exist_ok=True)
            self.logger.info(f"Permission error on original path. Using alternative directory: {project_dir}")
    
    def generate_synthetic_data(self, 
                               symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                               start_date='2022-01-01',
                               end_date='2022-12-31',
                               interval='1h'):
        """
        Generate synthetic market data for testing.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date (format: YYYY-MM-DD)
            end_date: End date (format: YYYY-MM-DD)
            interval: Time interval ('1m', '5m', '1h', '1d')
            
        Returns:
            DataFrame with synthetic market data
        """
        self.logger.info(f"Generating synthetic data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Determine time increment based on interval
        if interval == '1m':
            time_increment = timedelta(minutes=1)
            points_per_day = 390  # 6.5 hours of trading
        elif interval == '5m':
            time_increment = timedelta(minutes=5)
            points_per_day = 78
        elif interval == '1h':
            time_increment = timedelta(hours=1)
            points_per_day = 7
        else:  # Daily
            time_increment = timedelta(days=1)
            points_per_day = 1
        
        all_data = []
        
        for symbol in symbols:
            self.logger.info(f"Generating data for {symbol}")
            
            # Base price and volatility specific to the symbol
            base_price = 100 + (hash(symbol) % 900)  # $100-$1000
            volatility = 0.01 + (hash(symbol) % 10) / 100  # 1-10% volatility
            
            price = base_price
            current_dt = start_dt
            
            symbol_data = []
            
            while current_dt <= end_dt:
                # Only generate data for weekdays (Monday to Friday)
                if current_dt.weekday() < 5:
                    # Set to market open for this day
                    market_open = current_dt.replace(hour=9, minute=30, second=0)
                    
                    # Generate data points for this trading day
                    for i in range(points_per_day):
                        timestamp = market_open + i * time_increment
                        
                        # Random price movement based on volatility
                        price_change = np.random.normal(0, volatility)
                        price = price * (1 + price_change)
                        
                        # Generate OHLC data with realistic relationships
                        open_price = price * (1 + np.random.normal(0, volatility/3))
                        high_price = max(price, open_price) * (1 + abs(np.random.normal(0, volatility/2)))
                        low_price = min(price, open_price) * (1 - abs(np.random.normal(0, volatility/2)))
                        close_price = price
                        
                        # Generate volume with some randomness
                        volume = int(np.random.gamma(2.0, 100000) * (price / 100))
                        
                        # Add data point
                        symbol_data.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': volume
                        })
                
                # Move to next day
                current_dt += timedelta(days=1)
            
            # Convert to DataFrame and add to the list
            symbol_df = pd.DataFrame(symbol_data)
            all_data.append(symbol_df)
        
        # Combine all symbols' data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(['symbol', 'timestamp'])
        
        self.logger.info(f"Generated {len(combined_data)} data points for {len(symbols)} symbols")
        return combined_data
    
    def save_data(self, data, filename=None):
        """
        Save data to disk.
        
        Args:
            data: DataFrame to save
            filename: Optional filename
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"market_data_{timestamp}"
        
        # Save as CSV
        csv_path = os.path.join(self.raw_dir, f"{filename}.csv")
        data.to_csv(csv_path, index=False)
        self.logger.info(f"Data saved to {csv_path}")
        
        # Save as Parquet for better performance
        parquet_path = os.path.join(self.raw_dir, f"{filename}.parquet")
        data.to_parquet(parquet_path, index=False)
        self.logger.info(f"Data saved to {parquet_path}")
        
        return parquet_path
    
    def load_data(self, filepath):
        """
        Load data from a file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        self.logger.info(f"Loading data from {filepath}")
        
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
            # Convert timestamp to datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
        elif filepath.endswith('.parquet'):
            data = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        self.logger.info(f"Loaded data with shape: {data.shape}")
        return data

def main():
    """Main function to demonstrate data collection."""
    collector = DataCollector()
    
    # Check if data already exists
    data_path = os.path.join(collector.raw_dir, 'market_data.parquet')
    
    if os.path.exists(data_path):
        # Load existing data
        data = collector.load_data(data_path)
        print(f"File exists and contains data with shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()[:5]}... (showing first 5)")
        print("First few rows:")
        print(data.head())
    else:
        # Generate new data
        print("Generating new synthetic market data...")
        data = collector.generate_synthetic_data(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            start_date='2022-01-01',
            end_date='2022-12-31',
            interval='1h'
        )
        
        # Save the data
        collector.save_data(data, 'market_data')
        
        print(f"Generated data with shape: {data.shape}")
        print("First few rows:")
        print(data.head())

if __name__ == "__main__":
    main()
