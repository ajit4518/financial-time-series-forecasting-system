"""
Model Runner Script

This script handles the training and evaluation of LSTM models for time series forecasting.
It serves as an entry point for model training in the pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def setup_logger():
    """Set up a logger for the model training runner."""
    logger = logging.getLogger('model_runner')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f"model_training_{timestamp}.log")
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger

def train_models():
    """Train LSTM models for each symbol."""
    logger = setup_logger()
    logger.info("Starting model training")
    
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Import the LSTM model - using a direct import since we're in the same directory
    try:
        # Direct import from the same directory
        from lstm_model import LSTMForecaster, ModelTrainer, TimeSeriesDataset, prepare_data_for_model
        logger.info("Successfully imported LSTM model")
    except ImportError as e:
        logger.error(f"Error importing LSTM model: {str(e)}")
        logger.info("Trying alternative import...")
        
        try:
            # Add the current directory to the Python path
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # Try alternative import
            import lstm_model
            from lstm_model import LSTMForecaster, ModelTrainer, TimeSeriesDataset, prepare_data_for_model
            logger.info("Alternative import successful")
        except ImportError as e2:
            logger.error(f"Alternative import also failed: {str(e2)}")
            print("Please ensure lstm_model.py is in the same directory as this script")
            return
    
    # Set up directories
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    plots_dir = os.path.join(project_root, 'plots')
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info(f"Working with processed_dir: {processed_dir}")
    logger.info(f"Models will be saved to: {models_dir}")
    
    # Find training data files
    try:
        train_files = [f for f in os.listdir(processed_dir) if f.endswith('_train.parquet')]
        logger.info(f"Found {len(train_files)} training data files")
    except Exception as e:
        logger.error(f"Error finding training files: {str(e)}")
        print(f"Could not find training files in {processed_dir}")
        return
    
    if not train_files:
        logger.error("No training data files found. Run the pipeline first.")
        print(f"No training files found in {processed_dir}")
        return
    
    # Model hyperparameters
    sequence_length = 10
    hidden_dim = 64
    num_layers = 2
    dropout = 0.2
    batch_size = 32
    epochs = 30
    learning_rate = 0.001
    patience = 10
    target_column = 'target_next_close'
    
    # Track overall results
    all_results = []
    
    # Train a model for each symbol
    for train_file in train_files:
        symbol = train_file.split('_')[0]
        logger.info(f"Training model for {symbol}")
        
        # Get validation file
        val_file = train_file.replace('_train.parquet', '_val.parquet')
        val_path = os.path.join(processed_dir, val_file)
        
        if not os.path.exists(val_path):
            logger.error(f"Validation file not found for {symbol}")
            continue
        
        # Prepare data
        train_path = os.path.join(processed_dir, train_file)
        
        try:
            # Load and prepare data
            logger.info(f"Loading training data from {train_path}")
            train_data, target_col, feature_cols = prepare_data_for_model(
                train_path, target_col=target_column, sequence_length=sequence_length
            )
            
            logger.info(f"Loading validation data from {val_path}")
            val_data, _, _ = prepare_data_for_model(
                val_path, target_col=target_column, sequence_length=sequence_length
            )
            
            # Create datasets
            train_dataset = TimeSeriesDataset(
                data=train_data, 
                target_col=target_col, 
                feature_cols=feature_cols, 
                sequence_length=sequence_length
            )
            
            val_dataset = TimeSeriesDataset(
                data=val_data, 
                target_col=target_col, 
                feature_cols=feature_cols, 
                sequence_length=sequence_length
            )
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset.to_torch_dataset(), 
                batch_size=batch_size, 
                shuffle=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset.to_torch_dataset(), 
                batch_size=batch_size, 
                shuffle=False
            )
            
            # Initialize model
            input_dim = train_dataset.get_feature_dim()
            logger.info(f"Creating LSTM model with input dim: {input_dim}")
            model = LSTMForecaster(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout
            )
            
            # Initialize trainer
            trainer = ModelTrainer(
                model=model, 
                learning_rate=learning_rate
            )
            
            # Train model
            logger.info(f"Starting training for {symbol}")
            history = trainer.train(
                train_loader=train_loader, 
                val_loader=val_loader, 
                epochs=epochs, 
                patience=patience, 
                model_dir=models_dir
            )
            
            # Plot loss curves
            loss_plot_path = os.path.join(plots_dir, f"{symbol}_loss.png")
            trainer.plot_loss(save_path=loss_plot_path)
            
            # Evaluate on validation set
            val_metrics = trainer.evaluate(val_loader)
            
            # Store results
            result = {
                'symbol': symbol,
                'best_val_loss': history['best_val_loss'],
                'rmse': val_metrics['rmse'],
                'mae': val_metrics['mae'],
                'model_path': history['best_model_path']
            }
            all_results.append(result)
            
            # Generate prediction plots
            try:
                plt.figure(figsize=(12, 6))
                # Take a subset for better visualization
                subset_size = min(len(val_metrics['actuals']), 100) 
                
                plt.plot(range(subset_size), val_metrics['actuals'][:subset_size], label='Actual')
                plt.plot(range(subset_size), val_metrics['predictions'][:subset_size], label='Predicted')
                plt.title(f'{symbol} - Validation Set Predictions')
                plt.xlabel('Time')
                plt.ylabel(f'{target_col}')
                plt.legend()
                plt.grid(True)
                
                pred_plot_path = os.path.join(plots_dir, f"{symbol}_predictions.png")
                plt.savefig(pred_plot_path)
                plt.close()
            except Exception as e:
                logger.warning(f"Could not create prediction plot: {str(e)}")
            
            logger.info(f"Model training completed for {symbol}")
            logger.info(f"Best validation loss: {history['best_val_loss']:.6f}")
            logger.info(f"RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}")
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save overall results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"All model results saved to {results_path}")
        
        # Print summary
        print("\nModel Training Summary:")
        print(f"Total models trained: {len(all_results)}")
        
        # Show best model
        best_model = results_df.loc[results_df['rmse'].idxmin()]
        print(f"Best model: {best_model['symbol']} with RMSE: {best_model['rmse']:.6f}")
        
        # Show metrics for all models
        print("\nMetrics for all models:")
        for _, row in results_df.iterrows():
            print(f"  {row['symbol']}: RMSE = {row['rmse']:.6f}, MAE = {row['mae']:.6f}")
    
    logger.info("Model training completed")

if __name__ == "__main__":
    # Train models
    train_models()