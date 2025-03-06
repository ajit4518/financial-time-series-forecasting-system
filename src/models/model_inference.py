"""
Model Inference Script

This script demonstrates how to load a trained LSTM model and use it for making predictions
on test data.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import argparse

def setup_logger():
    """Set up a logger for the inference script."""
    logger = logging.getLogger('model_inference')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def run_inference(model_path=None, data_path=None):
    """
    Run inference using a trained LSTM model.
    
    Args:
        model_path: Path to trained model file (optional)
        data_path: Path to data file for inference (optional)
    """
    logger = setup_logger()
    logger.info("Starting model inference")
    
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Import the LSTM model module
    try:
        from lstm_model import LSTMForecaster, TimeSeriesDataset, prepare_data_for_model
        logger.info("Successfully imported LSTM model")
    except ImportError as e:
        logger.error(f"Error importing LSTM model: {str(e)}")
        # Try adding the current directory to the path
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        try:
            from lstm_model import LSTMForecaster, TimeSeriesDataset, prepare_data_for_model
            logger.info("Successfully imported LSTM model (alternative path)")
        except ImportError:
            logger.error("Failed to import LSTM model. Make sure lstm_model.py is in the same directory.")
            return
    
    # Set up directories
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Auto-find model if not provided
    if model_path is None:
        if not os.path.exists(models_dir):
            logger.error(f"Models directory not found: {models_dir}")
            return
            
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        if not model_files:
            logger.error("No model files found. Train models first.")
            return
            
        # Use the latest model
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(models_dir, latest_model)
        logger.info(f"Using latest model: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Auto-find data if not provided
    if data_path is None:
        val_files = [f for f in os.listdir(processed_dir) if f.endswith('_val.parquet')]
        if not val_files:
            logger.error("No validation data files found. Run the pipeline first.")
            return
            
        # Use the first validation file
        data_path = os.path.join(processed_dir, val_files[0])
        logger.info(f"Using validation data: {data_path}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Model hyperparameters
    sequence_length = 10
    hidden_dim = 64
    num_layers = 2
    dropout = 0.2
    batch_size = 32
    target_column = 'target_next_close'
    
    try:
        # Load and prepare data
        data, target_col, feature_cols = prepare_data_for_model(
            data_path, target_col=target_column, sequence_length=sequence_length
        )
        
        # Get symbol from filename if available
        symbol = os.path.basename(data_path).split('_')[0] if '_' in os.path.basename(data_path) else "Unknown"
        
        # Create dataset
        dataset = TimeSeriesDataset(
            data=data, 
            target_col=target_col, 
            feature_cols=feature_cols, 
            sequence_length=sequence_length
        )
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset.to_torch_dataset(), 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # Initialize model with the same architecture as the trained model
        input_dim = dataset.get_feature_dim()
        model = LSTMForecaster(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout
        )
        
        # Load model weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        
        # Make predictions
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                # Move data to device
                X_batch = X_batch.to(device)
                
                # Forward pass
                y_pred = model(X_batch)
                
                # Store predictions and actuals
                predictions.extend(y_pred.cpu().numpy())
                actuals.extend(y_batch.numpy())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        logger.info(f"Inference completed with metrics:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Actual': actuals.flatten(),
            'Predicted': predictions.flatten()
        })
        
        # Add timestamp if available
        if 'timestamp' in data.columns:
            # Account for sequence length
            timestamps = data['timestamp'].iloc[sequence_length:].reset_index(drop=True)
            if len(timestamps) == len(results):
                results['timestamp'] = timestamps
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(results_dir, f"{symbol}_inference_results_{timestamp}.csv")
        results.to_csv(results_path, index=False)
        
        logger.info(f"Results saved to {results_path}")
        
        # Generate plot
        plt.figure(figsize=(12, 6))
        # Take a subset for better visualization
        subset_size = min(len(results), 100)
        
        plt.plot(range(subset_size), results['Actual'].iloc[:subset_size], label='Actual')
        plt.plot(range(subset_size), results['Predicted'].iloc[:subset_size], label='Predicted')
        plt.title(f'{symbol} - Model Predictions')
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(results_dir, f"{symbol}_inference_plot_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Plot saved to {plot_path}")
        
        print("\nInference Results:")
        print(f"Symbol: {symbol}")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"Results saved to {results_path}")
        print(f"Plot saved to {plot_path}")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with a trained LSTM model')
    
    parser.add_argument('--model', type=str, help='Path to trained model file')
    parser.add_argument('--data', type=str, help='Path to data file for inference')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Run inference
    run_inference(args.model, args.data)