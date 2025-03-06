import pandas as pd

# Path to your features file
file_path = "time_series_forecasting_engine/data/processed/processed_data_20250301_145158.parquet"

try:
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Print information about the DataFrame
    print(f"File exists and contains data with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]}... (showing first 5)")
    print(f"First few rows: \n{df.head(100)}")
    
except Exception as e:
    print(f"Error reading file: {e}")