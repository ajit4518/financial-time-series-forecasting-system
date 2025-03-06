"""
Main Pipeline Script

This script ties together all components of the time series forecasting engine,
from data collection to feature engineering and data preparation for modeling.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt

def setup_logger():
    """Set up a logger for the main pipeline."""
    logger = logging.getLogger('main_pipeline')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f"pipeline_{timestamp}.log")
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger

def run_pipeline():
    """Run the complete data pipeline from collection to feature engineering."""
    logger = setup_logger()
    logger.info("Starting the time series forecasting pipeline")
    
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, '..'))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Add the src directory to the Python path
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Import project modules using direct imports from subdirectories
    try:
        from data.data_collector import DataCollector
        from data.data_preprocessor import DataPreprocessor
        from features.feature_engineering import FeatureEngineer
        
        logger.info("Successfully imported project modules")
    except ImportError as e:
        logger.error(f"Error importing modules: {str(e)}")
        sys.exit(1)
    
    # Set up directories
    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    # Ensure directories exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    logger.info(f"Working with project root: {project_root}")
    
    # Step 1: Data Collection
    logger.info("Step 1: Data Collection")
    collector = DataCollector()
    
    # Check if data already exists
    data_files = [f for f in os.listdir(raw_dir) if f.endswith('.parquet') or f.endswith('.csv')]
    
    if data_files:
        # Use the first available data file
        data_path = os.path.join(raw_dir, data_files[0])
        logger.info(f"Loading existing data from {data_path}")
        
        if data_path.endswith('.parquet'):
            try:
                data = pd.read_parquet(data_path)
            except ImportError:
                logger.warning("Missing pyarrow or fastparquet. Install with 'pip install pyarrow fastparquet'")
                # Fall back to CSV if possible
                csv_path = data_path.replace('.parquet', '.csv')
                if os.path.exists(csv_path):
                    data = pd.read_csv(csv_path)
                    logger.info(f"Falling back to CSV file: {csv_path}")
                else:
                    raise
        else:  # CSV file
            data = pd.read_csv(data_path)
    else:
        # Generate new data
        logger.info("No existing data found. Generating new synthetic market data.")
        data = collector.generate_synthetic_data(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            start_date='2022-01-01',
            end_date='2022-12-31',
            interval='1h'
        )
        collector.save_data(data, 'market_data')
    
    logger.info(f"Data collected with shape: {data.shape}")
    
    # Ensure timestamp is datetime
    if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Step 2: Data Preprocessing
    logger.info("Step 2: Data Preprocessing")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process(data)
    logger.info(f"Data preprocessed with shape: {processed_data.shape}")
    
    # Step 3: Feature Engineering
    logger.info("Step 3: Feature Engineering")
    engineer = FeatureEngineer()
    featured_data = engineer.create_features(processed_data)
    
    logger.info(f"Features engineered with shape: {featured_data.shape}")
    logger.info(f"Number of features created: {len(featured_data.columns)}")
    
    # Save the featured data
    features_path = os.path.join(processed_dir, f"featured_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
    featured_data.to_parquet(features_path)
    logger.info(f"Featured data saved to {features_path}")
    
    # Step 4: Data Splitting (prepare for model training)
    logger.info("Step 4: Preparing Data for Model Training")
    
    # Get list of symbols
    symbols = featured_data['symbol'].unique().tolist()
    
    # For each symbol, prepare a prediction-ready dataset
    for symbol in symbols:
        logger.info(f"Preparing training data for {symbol}")
        
        # Filter data for this symbol
        symbol_data = featured_data[featured_data['symbol'] == symbol].copy()
        
        # Sort by timestamp
        symbol_data = symbol_data.sort_values('timestamp')
        
        # Add target variable (next period's close price)
        symbol_data['target_next_close'] = symbol_data['close'].shift(-1)
        symbol_data['target_return'] = symbol_data['target_next_close'] / symbol_data['close'] - 1
        
        # Drop rows with NaN target (last row)
        symbol_data = symbol_data.dropna(subset=['target_next_close'])
        
        # Train/validation split (80/20)
        # This is time series data, so we split by time
        train_size = int(len(symbol_data) * 0.8)
        train_data = symbol_data.iloc[:train_size]
        val_data = symbol_data.iloc[train_size:]
        
        # Save training and validation data
        train_path = os.path.join(processed_dir, f"{symbol}_train.parquet")
        val_path = os.path.join(processed_dir, f"{symbol}_val.parquet")
        
        train_data.to_parquet(train_path)
        val_data.to_parquet(val_path)
        
        logger.info(f"Saved training data for {symbol}: {train_data.shape} rows")
        logger.info(f"Saved validation data for {symbol}: {val_data.shape} rows")
        
        # Create a basic visualization of the symbol's price and prediction target
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(symbol_data['timestamp'], symbol_data['close'], label='Close Price')
            plt.title(f'{symbol} Stock Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            
            # Save the figure
            plot_dir = os.path.join(project_root, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"{symbol}_price.png"))
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create visualization: {str(e)}")
    
    logger.info("Pipeline completed successfully!")
    return featured_data

def print_summary(featured_data):
    """Print a summary of the data and features."""
    print("\nPipeline completed! Summary statistics:")
    print(f"Total data points: {len(featured_data)}")
    print(f"Total features: {len(featured_data.columns)}")
    print(f"Symbols in dataset: {featured_data['symbol'].unique().tolist()}")
    
    # Print memory usage
    memory_usage = featured_data.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB
    print(f"Dataset memory usage: {memory_usage:.2f} MB")
    
    # Group features by category for better overview
    column_groups = {
        "Identifiers": ['timestamp', 'symbol'],
        "Price data": [col for col in featured_data.columns if col in ['open', 'high', 'low', 'close', 'volume']],
        "Moving averages": [col for col in featured_data.columns if 'sma_' in col or 'ema_' in col],
        "Technical indicators": [col for col in featured_data.columns if any(ind in col for ind in ['rsi', 'macd', 'bb_', 'obv'])],
        "Time features": [col for col in featured_data.columns if any(time_feat in col for time_feat in ['hour', 'day', 'month', 'week', 'session'])],
        "Lag features": [col for col in featured_data.columns if 'lag_' in col]
    }
    
    print("\nFeature categories:")
    for category, cols in column_groups.items():
        print(f"  {category}: {len(cols)} features")
    
    # Print correlation of some key features with the target (if available)
    if 'target_return' in featured_data.columns:
        print("\nTop correlations with target return:")
        correlations = featured_data.corr()['target_return'].abs().sort_values(ascending=False)
        print(correlations.head(10))

if __name__ == "__main__":
    try:
        # Run the pipeline
        featured_data = run_pipeline()
        
        # Print summary
        if featured_data is not None:
            print_summary(featured_data)
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()