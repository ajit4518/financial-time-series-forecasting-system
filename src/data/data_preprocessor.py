"""
Data Preprocessor Module

This module handles preprocessing of time series data, including cleaning,
normalization, and handling of missing values.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Optional

def setup_logger():
    """Set up a basic logger for the data preprocessor."""
    logger = logging.getLogger('data_preprocessor')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class DataPreprocessor:
    """
    A class for preprocessing time series data for machine learning models.
    
    Handles:
    - Data cleaning
    - Feature normalization
    - Missing value imputation
    - Outlier detection and handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataPreprocessor with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing settings
        """
        self.config = config or {}
        self.logger = setup_logger()
        
        # Get the current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        self.raw_dir = os.path.join(project_root, 'data', 'raw')
        self.processed_dir = os.path.join(project_root, 'data', 'processed')
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.logger.info(f"DataPreprocessor initialized")
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing method that applies the complete preprocessing pipeline.
        
        Args:
            data: Raw input data as a DataFrame
            
        Returns:
            Processed DataFrame
        """
        self.logger.info(f"Starting preprocessing for {len(data)} rows")
        
        # Clone the data to avoid modifying the original
        processed_data = data.copy()
        
        # Apply preprocessing steps
        processed_data = self._clean_data(processed_data)
        processed_data = self._handle_missing_values(processed_data)
        processed_data = self._handle_outliers(processed_data)
        
        # Ensure timestamp is datetime
        if 'timestamp' in processed_data.columns and not pd.api.types.is_datetime64_any_dtype(processed_data['timestamp']):
            processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
        
        # For time series, ensure data is properly sorted
        if 'timestamp' in processed_data.columns:
            if 'symbol' in processed_data.columns:
                processed_data = processed_data.sort_values(['symbol', 'timestamp'])
            else:
                processed_data = processed_data.sort_values('timestamp')
        
        self.logger.info(f"Preprocessing completed, resulting in {len(processed_data)} rows")
        return processed_data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by removing duplicates and handling obvious errors.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        original_rows = len(data)
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Check for and remove rows with invalid values
        numeric_cols = data.select_dtypes(include=np.number).columns
        
        for col in numeric_cols:
            # Remove rows with infinite values
            data = data[~data[col].isin([np.inf, -np.inf])]
        
        # Remove rows where price values don't make sense (e.g., negative prices)
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # Prices should be positive
            data = data[(data['open'] > 0) & (data['high'] > 0) & 
                         (data['low'] > 0) & (data['close'] > 0)]
            
            # High should be >= low
            data = data[data['high'] >= data['low']]
            
            # High should be >= open and close
            data = data[(data['high'] >= data['open']) & (data['high'] >= data['close'])]
            
            # Low should be <= open and close
            data = data[(data['low'] <= data['open']) & (data['low'] <= data['close'])]
        
        rows_removed = original_rows - len(data)
        if rows_removed > 0:
            self.logger.info(f"Removed {rows_removed} rows during cleaning")
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # Check for missing values
        missing_values = data.isnull().sum()
        total_missing = missing_values.sum()
        
        if total_missing == 0:
            self.logger.info("No missing values found")
            return data
        
        self.logger.info(f"Handling {total_missing} missing values")
        
        # For time series data, forward fill then backward fill is often a good strategy
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining missing values in numeric columns, use mean
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].mean())
        
        # For non-numeric columns, fill with the most common value
        categorical_cols = data.select_dtypes(exclude=np.number).columns
        for col in categorical_cols:
            if col != 'timestamp' and data[col].isnull().any():
                data[col] = data[col].fillna(data[col].mode()[0])
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        # Only handle outliers in the specified columns
        # For financial data, we may want to be careful not to remove price spikes
        # that could be legitimate market movements
        outlier_cols = self.config.get('outlier_columns', [])
        
        if not outlier_cols:
            return data
            
        original_rows = len(data)
        
        for col in outlier_cols:
            if col in data.columns:
                # Use IQR method for outlier detection
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Replace outliers with bounds instead of removing
                data[col] = data[col].clip(lower_bound, upper_bound)
        
        return data

def main():
    """Main function to demonstrate data preprocessing."""
    preprocessor = DataPreprocessor()
    
    # Try to find market data
    data_files = [f for f in os.listdir(preprocessor.raw_dir) if f.endswith('.parquet') or f.endswith('.csv')]
    
    if data_files:
        # Use the first available data file
        data_path = os.path.join(preprocessor.raw_dir, data_files[0])
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
        
        # Preprocess the data
        processed_data = preprocessor.process(data)
        
        # Save processed data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(preprocessor.processed_dir, f"processed_data_{timestamp}.parquet")
        processed_data.to_parquet(output_path, index=False)
        
        print(f"Preprocessed data saved to {output_path}")
        print(f"Processed data shape: {processed_data.shape}")
    else:
        print("No data files found in the raw directory. Please run data_collector.py first.")

if __name__ == "__main__":
    main()