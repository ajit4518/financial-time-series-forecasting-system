"""
LSTM Model for Time Series Forecasting

This module implements an LSTM-based neural network model for financial
time series forecasting.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

def setup_logger():
    """Set up a basic logger for the LSTM model."""
    logger = logging.getLogger('lstm_model')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class TimeSeriesDataset:
    """
    Dataset for preparing time series data for PyTorch models.
    
    This class handles the creation of sequences from time series data
    and preparing them for LSTM training.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str, feature_cols: List[str], 
                 sequence_length: int, target_horizon: int = 1):
        """
        Initialize the time series dataset.
        
        Args:
            data: DataFrame containing the time series data
            target_col: Name of the target column to predict
            feature_cols: List of feature column names to use
            sequence_length: Length of input sequences (lookback period)
            target_horizon: Number of steps ahead to predict
        """
        self.data = data
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        
        # Create sequences and targets
        self.X, self.y = self._create_sequences()
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from the time series data.
        
        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        features = self.data[self.feature_cols].values
        targets = self.data[self.target_col].values
        
        X, y = [], []
        
        for i in range(len(self.data) - self.sequence_length - self.target_horizon + 1):
            # Extract sequence of features
            X.append(features[i:i+self.sequence_length])
            
            # Extract target (future value)
            y.append(targets[i+self.sequence_length+self.target_horizon-1])
        
        return np.array(X), np.array(y)
    
    def to_torch_dataset(self) -> TensorDataset:
        """
        Convert the sequences to PyTorch TensorDataset.
        
        Returns:
            PyTorch TensorDataset
        """
        X_tensor = torch.FloatTensor(self.X)
        y_tensor = torch.FloatTensor(self.y).unsqueeze(1)  # Add dimension for output
        
        return TensorDataset(X_tensor, y_tensor)
    
    def get_feature_dim(self) -> int:
        """Get the number of input features."""
        return len(self.feature_cols)

class LSTMForecaster(nn.Module):
    """
    LSTM-based neural network for time series forecasting.
    
    Features:
    - Multi-layer LSTM architecture
    - Dropout for regularization
    - Configurable for different prediction tasks
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            output_dim: Output dimension size (typically 1 for regression)
            dropout: Dropout probability
        """
        super(LSTMForecaster, self).__init__()
        
        # Store model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_len, hidden_dim)
        
        # Take the last time step output
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        
        return out

class ModelTrainer:
    """
    Trainer class for LSTM models.
    
    Handles:
    - Model training process
    - Model evaluation
    - Learning rate scheduling
    - Early stopping
    - Model saving
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None, 
                 learning_rate: float = 0.001, weight_decay: float = 0.0001):
        """
        Initialize the model trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on (CPU or GPU)
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
        """
        self.logger = setup_logger()
        self.model = model
        
        # Use GPU if available
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.logger.info(f"Training on device: {self.device}")
        
        # Optimizer and criterion
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
        # For tracking training progress
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, patience: int = 10, model_dir: str = 'models'):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            patience: Patience for early stopping
            model_dir: Directory to save models
            
        Returns:
            Dictionary with training history
        """
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize variables for early stopping
        no_improve_epochs = 0
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                # Move data to device
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            # Calculate average training loss
            train_loss = train_loss / len(train_loader.dataset)
            self.train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    # Move data to device
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    # Forward pass
                    y_pred = self.model(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    
                    val_loss += loss.item() * X_batch.size(0)
            
            # Calculate average validation loss
            val_loss = val_loss / len(val_loader.dataset)
            self.val_losses.append(val_loss)
            
            # Print progress
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check if this is the best model so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                
                # Save the model
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = os.path.join(model_dir, f"lstm_model_{timestamp}.pt")
                torch.save(self.model.state_dict(), model_path)
                
                if self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)  # Remove previous best model
                
                self.best_model_path = model_path
                no_improve_epochs = 0
                
                self.logger.info(f"New best model saved to {model_path}")
            else:
                no_improve_epochs += 1
            
            # Early stopping
            if no_improve_epochs >= patience:
                self.logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        self.logger.info("Training completed")
        
        # Load the best model
        if self.best_model_path:
            self.model.load_state_dict(torch.load(self.best_model_path))
            self.logger.info(f"Loaded best model from {self.best_model_path}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_model_path': self.best_model_path
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        test_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # Move data to device
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                
                test_loss += loss.item() * X_batch.size(0)
                
                # Store predictions and actuals
                predictions.extend(y_pred.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())
        
        # Calculate average test loss
        test_loss = test_loss / len(test_loader.dataset)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        self.logger.info(f"Test Loss: {test_loss:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
        
        return {
            'test_loss': test_loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def plot_loss(self, save_path: str = None):
        """
        Plot the training and validation loss curves.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Loss plot saved to {save_path}")
        
        plt.close()
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the model.
        
        Args:
            X: Input tensor
            
        Returns:
            Predicted values
        """
        self.model.eval()
        X = X.to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(X)
        
        return y_pred.cpu()


def prepare_data_for_model(data_path: str, target_col: str = 'target_next_close', sequence_length: int = 10):
    """
    Load and prepare data for the LSTM model.
    
    Args:
        data_path: Path to the data file
        target_col: Name of the target column
        sequence_length: Length of input sequences
        
    Returns:
        Tuple of (features, target, feature_cols)
    """
    logger = setup_logger()
    
    # Load data
    try:
        if data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            data = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    logger.info(f"Loaded data with shape: {data.shape}")
    
    # Ensure timestamp is datetime if it exists
    if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Sort by timestamp if available
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp')
    
    # Select features (exclude non-numeric columns and target)
    exclude_cols = ['timestamp', 'symbol', target_col, 'target_return']
    if 'target_next_close' in data.columns and target_col != 'target_next_close':
        exclude_cols.append('target_next_close')
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    logger.info(f"Selected {len(feature_cols)} features")
    
    # Ensure target column exists
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    return data, target_col, feature_cols

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("LSTM Model for Time Series Forecasting")
    logger.info("Run this module directly to test the implementation.")