"""
Feature Engineering Module - Fixed version with better path handling and performance

This module creates meaningful features from financial time series data,
including technical indicators, statistical features, and time-based features.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def setup_logger():
    """Set up a basic logger for the feature engineering module."""
    logger = logging.getLogger('feature_engineering')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class FeatureEngineer:
    """
    A class for engineering features from financial time series data.
    
    Creates features like:
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Statistical features (rolling stats)
    - Time-based features (time of day, day of week)
    - Volatility measures
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.logger = setup_logger()
        
        # Use absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        self.raw_dir = os.path.join(project_root, 'data', 'raw')
        self.processed_dir = os.path.join(project_root, 'data', 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Feature configuration
        self.short_windows = [5, 10, 20]
        self.medium_windows = [50]
        self.long_windows = [200]
        
        self.logger.info(f"FeatureEngineer initialized with processed dir: {os.path.abspath(self.processed_dir)}")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to create all features.
        
        Args:
            data: Input DataFrame with time series data
            
        Returns:
            DataFrame with additional engineered features
        """
        self.logger.info(f"Creating features for {len(data)} rows")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Sort by symbol and timestamp
        if 'symbol' in df.columns and 'timestamp' in df.columns:
            df = df.sort_values(['symbol', 'timestamp'])
        elif 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Process by symbol if multiple symbols exist
        if 'symbol' in df.columns:
            all_feature_dfs = []
            symbols = df['symbol'].unique()
            
            for symbol in symbols:
                self.logger.info(f"Processing features for symbol: {symbol}")
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df_with_features = self._process_single_dataframe(symbol_df)
                all_feature_dfs.append(symbol_df_with_features)
            
            # Combine all processed data
            result_df = pd.concat(all_feature_dfs, ignore_index=True)
        else:
            # Single dataframe processing
            result_df = self._process_single_dataframe(df)
        
        # Drop rows with NaN values (due to rolling calculations)
        original_len = len(result_df)
        result_df = result_df.dropna()
        
        self.logger.info(f"Dropped {original_len - len(result_df)} rows with NaN values")
        self.logger.info(f"Feature engineering complete. Total features: {len(result_df.columns)}")
        
        return result_df
    
    def _process_single_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a single dataframe (for one symbol or the whole dataset)."""
        # Initialize all features that will be created
        feature_dict = {}
        
        # Price features
        feature_dict['daily_return'] = df['close'].pct_change()
        feature_dict['log_return'] = np.log(df['close'] / df['close'].shift(1))
        feature_dict['high_low_range'] = df['high'] - df['low']
        feature_dict['close_open_range'] = df['close'] - df['open']
        feature_dict['body_size'] = abs(df['close'] - df['open'])
        feature_dict['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        feature_dict['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        feature_dict['high_low_ratio'] = df['high'] / df['low']
        feature_dict['close_open_ratio'] = df['close'] / df['open']
        
        # Technical indicators
        # Moving Averages
        for window in self.short_windows + self.medium_windows + self.long_windows:
            feature_dict[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            feature_dict[f'close_to_sma_{window}'] = (df['close'] / feature_dict[f'sma_{window}'] - 1) * 100
        
        # Exponential Moving Averages
        for window in self.short_windows + self.medium_windows:
            feature_dict[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            feature_dict[f'close_to_ema_{window}'] = (df['close'] / feature_dict[f'ema_{window}'] - 1) * 100
        
        # MACD
        feature_dict['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        feature_dict['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        feature_dict['macd'] = feature_dict['ema_12'] - feature_dict['ema_26']
        feature_dict['macd_signal'] = feature_dict['macd'].ewm(span=9, adjust=False).mean()
        feature_dict['macd_hist'] = feature_dict['macd'] - feature_dict['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Handle first values
        if len(gain) >= 14:
            avg_gain.iloc[13] = gain.iloc[1:15].mean()
            avg_loss.iloc[13] = loss.iloc[1:15].mean()
            
            # Calculate RSI using EMA
            for i in range(14, len(gain)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
                
        rs = avg_gain / avg_loss
        feature_dict['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for window in [20]:
            feature_dict[f'bb_middle_{window}'] = df['close'].rolling(window=window).mean()
            feature_dict[f'bb_std_{window}'] = df['close'].rolling(window=window).std()
            feature_dict[f'bb_upper_{window}'] = feature_dict[f'bb_middle_{window}'] + 2 * feature_dict[f'bb_std_{window}']
            feature_dict[f'bb_lower_{window}'] = feature_dict[f'bb_middle_{window}'] - 2 * feature_dict[f'bb_std_{window}']
            feature_dict[f'bb_width_{window}'] = (feature_dict[f'bb_upper_{window}'] - feature_dict[f'bb_lower_{window}']) / feature_dict[f'bb_middle_{window}']
            feature_dict[f'bb_pct_{window}'] = (df['close'] - feature_dict[f'bb_lower_{window}']) / (feature_dict[f'bb_upper_{window}'] - feature_dict[f'bb_lower_{window}'])
        
        # Volume indicators
        feature_dict['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        feature_dict['volume_ratio'] = df['volume'] / feature_dict['volume_sma_10']
        
        # On-Balance Volume (OBV)
        obv_direction = np.where(df['close'] > df['close'].shift(1), 1, 
                               np.where(df['close'] < df['close'].shift(1), -1, 0))
        obv_volume = obv_direction * df['volume']
        feature_dict['obv'] = obv_volume.cumsum()
        
        # Volatility features
        for window in self.short_windows + self.medium_windows:
            feature_dict[f'return_std_{window}'] = feature_dict['daily_return'].rolling(window=window).std()
            feature_dict[f'volatility_{window}'] = feature_dict[f'return_std_{window}'] * np.sqrt(252)  # Annualized
        
        # Average True Range (ATR)
        tr1 = df['high'] - df['low']  # Current high - current low
        tr2 = abs(df['high'] - df['close'].shift(1))  # Current high - previous close
        tr3 = abs(df['low'] - df['close'].shift(1))  # Current low - previous close
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        feature_dict['atr_14'] = tr.rolling(window=14).mean()
        feature_dict['atr_pct'] = feature_dict['atr_14'] / df['close'] * 100
        
        # Parkinson's Volatility (uses high-low range)
        for window in [10, 20]:
            feature_dict[f'parkinsons_vol_{window}'] = (1.0 / (4.0 * np.log(2.0))) * (np.log(df['high'] / df['low'])**2).rolling(window=window).mean() * np.sqrt(252)
        
        # Time features
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract time components
            feature_dict['hour'] = df['timestamp'].dt.hour
            feature_dict['minute'] = df['timestamp'].dt.minute
            feature_dict['day_of_week'] = df['timestamp'].dt.dayofweek
            feature_dict['day_of_month'] = df['timestamp'].dt.day
            feature_dict['month'] = df['timestamp'].dt.month
            
            # Trading session (0-4: pre-market, morning, mid-day, afternoon, after-hours)
            time_minutes = feature_dict['hour'] * 60 + feature_dict['minute']
            
            conditions = [
                (time_minutes < 9*60 + 30),  # Before 9:30 AM
                (time_minutes >= 9*60 + 30) & (time_minutes < 12*60),  # 9:30 AM - 12:00 PM
                (time_minutes >= 12*60) & (time_minutes < 14*60),  # 12:00 PM - 2:00 PM
                (time_minutes >= 14*60) & (time_minutes < 16*60),  # 2:00 PM - 4:00 PM
                (time_minutes >= 16*60)  # After 4:00 PM
            ]
            session_values = [0, 1, 2, 3, 4]
            feature_dict['session'] = np.select(conditions, session_values, default=0)
            
            # Cyclic encoding of time
            feature_dict['hour_sin'] = np.sin(2 * np.pi * feature_dict['hour'] / 24)
            feature_dict['hour_cos'] = np.cos(2 * np.pi * feature_dict['hour'] / 24)
            feature_dict['day_of_week_sin'] = np.sin(2 * np.pi * feature_dict['day_of_week'] / 7)
            feature_dict['day_of_week_cos'] = np.cos(2 * np.pi * feature_dict['day_of_week'] / 7)
            feature_dict['month_sin'] = np.sin(2 * np.pi * feature_dict['month'] / 12)
            feature_dict['month_cos'] = np.cos(2 * np.pi * feature_dict['month'] / 12)
        
        # Lag features
        # Define features to lag
        price_cols = ['close', 'high', 'low', 'daily_return', 'volume']
        tech_cols = ['rsi_14', 'macd', 'bb_pct_20'] if all(col in feature_dict for col in ['rsi_14', 'macd', 'bb_pct_20']) else []
        
        lag_cols = price_cols + tech_cols
        
        # Create lagged features
        for col in lag_cols:
            for lag in [1, 2, 3, 5, 10]:
                if col in feature_dict:
                    feature_dict[f'{col}_lag_{lag}'] = feature_dict[col].shift(lag)
                elif col in df.columns:
                    feature_dict[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Rolling mean of target
        for window in [5, 10, 20]:
            feature_dict[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            feature_dict[f'return_rolling_mean_{window}'] = feature_dict['daily_return'].rolling(window=window).mean()
        
        # Return momentum (rate of change)
        for period in [5, 10, 20]:
            feature_dict[f'momentum_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100
        
        # Create a single DataFrame at once (more efficient than adding columns one by one)
        feature_df = pd.DataFrame(feature_dict)
        
        # Add original columns
        for col in df.columns:
            if col not in feature_df:
                feature_df[col] = df[col]
        
        return feature_df
    
    def save_features(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save engineered features to disk.
        
        Args:
            df: DataFrame with features
            filename: Base filename (without extension)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"features_{timestamp}"
        
        output_path = os.path.join(self.processed_dir, f"{filename}.parquet")
        df.to_parquet(output_path, index=False)
        
        self.logger.info(f"Features saved to {output_path}")
        return output_path

def main():
    """Main function to demonstrate feature engineering."""
    engineer = FeatureEngineer()
    
    # Try to find the market data
    data_files = [f for f in os.listdir(engineer.raw_dir) if f.endswith('.parquet') or f.endswith('.csv')]
    
    if data_files:
        # Use the first available data file
        data_path = os.path.join(engineer.raw_dir, data_files[0])
        print(f"Found data file: {data_path}")
        
        # Load data
        if data_path.endswith('.parquet'):
            try:
                data = pd.read_parquet(data_path)
            except ImportError:
                print("Error: Missing pyarrow or fastparquet. Install with 'pip install pyarrow fastparquet'")
                # Fall back to CSV if possible
                if data_path[:-8] + '.csv' in data_files:
                    data_path = data_path[:-8] + '.csv'
                    data = pd.read_csv(data_path)
                    print(f"Falling back to CSV file: {data_path}")
                else:
                    raise
        else:  # CSV file
            data = pd.read_csv(data_path)
            # Convert timestamp to datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        print(f"Loaded data with shape: {data.shape}")
    else:
        print("No existing data files found. Creating dummy data.")
        # Create dummy data
        from datetime import datetime, timedelta
        
        # Create dummy data
        dates = [datetime(2023, 1, 1) + timedelta(minutes=5*i) for i in range(1000)]
        symbols = ['AAPL', 'MSFT']
        
        data = []
        for symbol in symbols:
            base_price = 150 if symbol == 'AAPL' else 250
            for date in dates:
                if date.weekday() < 5 and 9 <= date.hour < 16:  # Weekdays, trading hours
                    price = base_price + np.random.normal(0, 2)
                    data.append({
                        'timestamp': date,
                        'symbol': symbol,
                        'open': price,
                        'high': price * (1 + np.random.uniform(0, 0.01)),
                        'low': price * (1 - np.random.uniform(0, 0.01)),
                        'close': price * (1 + np.random.normal(0, 0.005)),
                        'volume': int(np.random.uniform(1000, 10000))
                    })
        
        data = pd.DataFrame(data)
    
    # Create feature engineer and generate features
    featured_data = engineer.create_features(data)
    
    # Save and print summary
    engineer.save_features(featured_data)
    print(f"Generated {len(featured_data.columns)} features for {len(featured_data)} data points")
    print("Feature columns:", featured_data.columns.tolist())

if __name__ == "__main__":
    main()