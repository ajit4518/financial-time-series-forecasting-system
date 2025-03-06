"""
Model Monitoring and Retraining System

This module implements monitoring for model performance, drift detection,
and automated retraining pipelines.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import shutil
from scipy import stats

def setup_logger():
    """Set up a logger for the MLOps system."""
    logger = logging.getLogger('mlops')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def json_serializable(obj):
    """
    Helper function to make objects JSON serializable.
    Handles datetime objects by converting them to ISO format strings.
    
    Args:
        obj: Object to make JSON serializable
        
    Returns:
        JSON serializable object
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [json_serializable(item) for item in obj]
    else:
        return obj

class LSTMForecaster(torch.nn.Module):
    """
    Simplified LSTM model structure for loading saved models without needing the original class.
    This is a placeholder that matches the structure of your trained model.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int = 1, dropout: float = 0.2):
        """Initialize the LSTM model."""
        super(LSTMForecaster, self).__init__()
        
        # Store model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM layer
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step output
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        
        return out

class ModelRegistry:
    """
    Model Registry for versioning and managing models.
    
    Stores model metadata, performance metrics, and enables model versioning.
    """
    
    def __init__(self, registry_dir: str = None):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory to store the model registry
        """
        self.logger = setup_logger()
        
        # Get current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        # Set registry directory
        self.registry_dir = registry_dir or os.path.join(project_root, 'model_registry')
        os.makedirs(self.registry_dir, exist_ok=True)
        
        # Initialize registry
        self.registry_file = os.path.join(self.registry_dir, 'registry.json')
        self.registry = self._load_registry()
        
        self.logger.info(f"Model registry initialized at {self.registry_dir}")
    
    def _load_registry(self) -> Dict:
        """Load the registry from disk."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning(f"Error decoding registry file. Creating new registry.")
                registry = {
                    'models': {},
                    'active_models': {}
                }
                self._save_registry(registry)
                return registry
        else:
            # Initialize empty registry
            registry = {
                'models': {},
                'active_models': {}
            }
            self._save_registry(registry)
            return registry
    
    def _save_registry(self, registry: Dict = None) -> None:
        """Save the registry to disk."""
        registry = registry or self.registry
        with open(self.registry_file, 'w') as f:
            # Use the helper function to ensure JSON serializable objects
            serializable_registry = json_serializable(registry)
            json.dump(serializable_registry, f, indent=4)
    
    def register_model(self, model_path: str, model_type: str, metrics: Dict, 
                       params: Dict, symbol: str = None) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (e.g., 'lstm', 'gru')
            metrics: Model performance metrics
            params: Model parameters
            symbol: Symbol associated with the model
            
        Returns:
            Model ID
        """
        # Generate model ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol_text = f"{symbol}_" if symbol else ""
        model_id = f"{symbol_text}{model_type}_{timestamp}"
        
        # Create model directory
        model_dir = os.path.join(self.registry_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy model file to registry
        model_filename = os.path.basename(model_path)
        registry_model_path = os.path.join(model_dir, model_filename)
        shutil.copy2(model_path, registry_model_path)
        
        # Create model metadata
        model_metadata = {
            'id': model_id,
            'type': model_type,
            'symbol': symbol,
            'created_at': timestamp,
            'metrics': metrics,
            'params': params,
            'path': registry_model_path,
            'status': 'registered'
        }
        
        # Update registry
        self.registry['models'][model_id] = model_metadata
        self._save_registry()
        
        self.logger.info(f"Model registered with ID: {model_id}")
        return model_id
    
    def activate_model(self, model_id: str, symbol: str) -> None:
        """
        Activate a model for a symbol.
        
        Args:
            model_id: Model ID
            symbol: Symbol
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"Model ID {model_id} not found in registry")
            
        # Update model status
        self.registry['models'][model_id]['status'] = 'active'
        
        # Set as active model for symbol
        if symbol:
            self.registry['active_models'][symbol] = model_id
        
        self._save_registry()
        self.logger.info(f"Model {model_id} activated for symbol {symbol}")
    
    def get_active_model(self, symbol: str) -> Optional[Dict]:
        """
        Get the active model for a symbol.
        
        Args:
            symbol: Symbol
            
        Returns:
            Model metadata or None if no active model
        """
        if symbol in self.registry['active_models']:
            model_id = self.registry['active_models'][symbol]
            return self.registry['models'].get(model_id)
        
        return None
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict]:
        """
        Get model metadata.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model metadata or None if not found
        """
        return self.registry['models'].get(model_id)
    
    def list_models(self, symbol: str = None) -> List[Dict]:
        """
        List models in the registry.
        
        Args:
            symbol: Filter by symbol
            
        Returns:
            List of model metadata
        """
        models = list(self.registry['models'].values())
        
        if symbol:
            models = [model for model in models if model.get('symbol') == symbol]
            
        return models
    
    def load_model(self, model_id: str, device: torch.device = None) -> torch.nn.Module:
        """
        Load a model from the registry.
        
        Args:
            model_id: Model ID
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        model_metadata = self.get_model_metadata(model_id)
        if not model_metadata:
            raise ValueError(f"Model ID {model_id} not found in registry")
            
        model_path = model_metadata['path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model instance using parameters from metadata
        model_params = model_metadata['params']
        
        try:
            # Create a basic model with default parameters if metadata doesn't have them
            input_dim = model_params.get('input_dim', 100)
            hidden_dim = model_params.get('hidden_dim', 64)
            num_layers = model_params.get('num_layers', 2)
            dropout = model_params.get('dropout', 0.2)
            
            model = LSTMForecaster(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
            
            # Load model state
            device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            self.logger.info(f"Model {model_id} loaded from registry")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

class DriftDetector:
    """
    Drift detector for monitoring data and concept drift.
    
    Uses statistical tests to detect changes in data distribution.
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: Reference data for comparison
        """
        self.logger = setup_logger()
        self.reference_data = reference_data
        self.reference_stats = {}
        
        if reference_data is not None:
            self._compute_reference_statistics()
    
    def _compute_reference_statistics(self) -> None:
        """Compute statistics for reference data."""
        if self.reference_data is None:
            return
            
        # Compute statistics for numeric columns
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = self.reference_data[col].dropna().values
            if len(values) == 0:
                continue
                
            self.reference_stats[col] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'median': np.median(values),
                'q75': np.percentile(values, 75),
                'skew': stats.skew(values),
                'kurtosis': stats.kurtosis(values)
            }
    
    def set_reference_data(self, reference_data: pd.DataFrame) -> None:
        """
        Set reference data.
        
        Args:
            reference_data: Reference data
        """
        self.reference_data = reference_data
        self._compute_reference_statistics()
        self.logger.info(f"Reference data set with {len(reference_data)} samples")
    
    def detect_drift(self, current_data: pd.DataFrame, 
                    threshold: float = 0.05) -> Dict[str, Dict[str, Union[float, bool]]]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current data
            threshold: P-value threshold for statistical tests
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            self.logger.warning("Reference data not set for drift detection")
            return {'summary': {'any_distribution_drift': False, 'num_features_with_drift': 0}}
            
        # Ensure we have the same columns
        common_cols = set(self.reference_data.columns).intersection(set(current_data.columns))
        numeric_cols = [col for col in common_cols if col in self.reference_stats]
        
        drift_results = {}
        
        for col in numeric_cols:
            ref_values = self.reference_data[col].dropna().values
            cur_values = current_data[col].dropna().values
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
            
            # Kolmogorov-Smirnov test for distribution shift
            try:
                ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)
            except Exception:
                # Handle cases where KS test might fail
                ks_stat, ks_pvalue = 0, 1.0
            
            # T-test for mean shift (if enough samples and variance > 0)
            try:
                t_stat, t_pvalue = stats.ttest_ind(ref_values, cur_values, equal_var=False)
                if np.isnan(t_pvalue):
                    t_stat, t_pvalue = 0, 1.0
            except Exception:
                t_stat, t_pvalue = 0, 1.0
            
            # Mann-Whitney U test for median shift
            try:
                u_stat, u_pvalue = stats.mannwhitneyu(ref_values, cur_values, alternative='two-sided')
            except Exception:
                u_stat, u_pvalue = 0, 1.0
            
            # F-test for variance shift
            try:
                ref_var = np.var(ref_values, ddof=1)
                cur_var = np.var(cur_values, ddof=1)
                if ref_var > 0 and cur_var > 0:
                    f_stat = ref_var / cur_var
                    f_pvalue = stats.f.sf(f_stat, len(ref_values) - 1, len(cur_values) - 1) * 2
                else:
                    f_stat, f_pvalue = 0, 1.0
            except Exception:
                f_stat, f_pvalue = 0, 1.0
            
            # Compute current statistics
            cur_stats = {
                'mean': np.mean(cur_values),
                'std': np.std(cur_values),
                'min': np.min(cur_values),
                'max': np.max(cur_values),
                'median': np.median(cur_values)
            }
            
            # Calculate relative changes
            ref_stats = self.reference_stats[col]
            rel_changes = {}
            
            for stat in ['mean', 'std', 'median']:
                if abs(ref_stats[stat]) > 1e-10:  # Avoid division by zero or very small numbers
                    rel_changes[f'{stat}_rel_change'] = (cur_stats[stat] - ref_stats[stat]) / abs(ref_stats[stat])
                else:
                    rel_changes[f'{stat}_rel_change'] = float('inf') if abs(cur_stats[stat]) > 1e-10 else 0
            
            # Determine if there's a drift
            is_distribution_drift = ks_pvalue < threshold
            is_mean_drift = t_pvalue < threshold
            is_median_drift = u_pvalue < threshold
            is_variance_drift = f_pvalue < threshold
            
            # Combine results
            drift_results[col] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                't_statistic': float(t_stat) if not np.isnan(t_stat) else 0.0,
                't_pvalue': float(t_pvalue) if not np.isnan(t_pvalue) else 1.0,
                'u_statistic': float(u_stat),
                'u_pvalue': float(u_pvalue),
                'f_statistic': float(f_stat),
                'f_pvalue': float(f_pvalue),
                'is_distribution_drift': is_distribution_drift,
                'is_mean_drift': is_mean_drift,
                'is_median_drift': is_median_drift,
                'is_variance_drift': is_variance_drift,
                'reference_stats': ref_stats,
                'current_stats': cur_stats,
                'relative_changes': rel_changes
            }
        
        # Overall drift status
        any_distribution_drift = any(result['is_distribution_drift'] for result in drift_results.values())
        any_mean_drift = any(result['is_mean_drift'] for result in drift_results.values())
        any_median_drift = any(result['is_median_drift'] for result in drift_results.values())
        any_variance_drift = any(result['is_variance_drift'] for result in drift_results.values())
        
        drift_summary = {
            'any_distribution_drift': any_distribution_drift,
            'any_mean_drift': any_mean_drift,
            'any_median_drift': any_median_drift,
            'any_variance_drift': any_variance_drift,
            'num_features_with_drift': sum(1 for result in drift_results.values() 
                                         if result['is_distribution_drift'])
        }
        
        drift_results['summary'] = drift_summary
        
        return drift_results

class PerformanceMonitor:
    """
    Monitor model performance over time.
    
    Tracks prediction errors, detects performance degradation,
    and triggers retraining when needed.
    """
    
    def __init__(self, model_registry: ModelRegistry = None):
        """
        Initialize the performance monitor.
        
        Args:
            model_registry: Model registry
        """
        self.logger = setup_logger()
        self.model_registry = model_registry or ModelRegistry()
        
        # Get current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        # Set monitor directory
        self.monitor_dir = os.path.join(project_root, 'monitoring')
        os.makedirs(self.monitor_dir, exist_ok=True)
        
        # Performance history by model ID
        self.performance_history = {}
        self._load_performance_history()
        
        self.logger.info(f"Performance monitor initialized at {self.monitor_dir}")
    
    def _load_performance_history(self) -> None:
        """Load performance history from disk."""
        history_file = os.path.join(self.monitor_dir, 'performance_history.pkl')
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'rb') as f:
                    self.performance_history = pickle.load(f)
            except Exception as e:
                self.logger.error(f"Error loading performance history: {str(e)}")
                self.performance_history = {}
    
    def _save_performance_history(self) -> None:
        """Save performance history to disk."""
        history_file = os.path.join(self.monitor_dir, 'performance_history.pkl')
        
        try:
            with open(history_file, 'wb') as f:
                pickle.dump(self.performance_history, f)
        except Exception as e:
            self.logger.error(f"Error saving performance history: {str(e)}")
    
    def log_prediction_performance(self, model_id: str, timestamp: datetime,
                                  metrics: Dict[str, float]) -> None:
        """
        Log prediction performance.
        
        Args:
            model_id: Model ID
            timestamp: Timestamp of predictions
            metrics: Performance metrics
        """
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
            
        # Add prediction performance
        self.performance_history[model_id].append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Save performance history
        self._save_performance_history()
        
        self.logger.info(f"Logged prediction performance for model {model_id}")
    
    def get_performance_trend(self, model_id: str, metric: str = 'rmse',
                            window: int = 10) -> Optional[pd.DataFrame]:
        """
        Get performance trend for a model.
        
        Args:
            model_id: Model ID
            metric: Metric to track
            window: Window size for rolling average
            
        Returns:
            DataFrame with performance trend
        """
        if model_id not in self.performance_history:
            self.logger.warning(f"No performance history for model {model_id}")
            return None
            
        # Create DataFrame from performance history
        history = self.performance_history[model_id]
        
        if not history:
            return None
            
        df = pd.DataFrame([{
            'timestamp': entry['timestamp'],
            **entry['metrics']
        } for entry in history])
        
        if metric not in df.columns:
            self.logger.warning(f"Metric {metric} not found in performance history")
            return None
            
        # Calculate rolling average
        if len(df) >= window:
            df[f'{metric}_rolling'] = df[metric].rolling(window=window).mean()
            
        return df
    
    def should_retrain(self, model_id: str, metric: str = 'rmse',
                      threshold: float = 0.2, window: int = 10) -> bool:
        """
        Determine if a model should be retrained.
        
        Args:
            model_id: Model ID
            metric: Metric to track
            threshold: Threshold for performance degradation
            window: Window size for rolling average
            
        Returns:
            True if model should be retrained, False otherwise
        """
        trend = self.get_performance_trend(model_id, metric, window)
        
        if trend is None or len(trend) < window * 2:
            return False
            
        # Get initial and current performance
        initial_perf = trend[metric].iloc[:window].mean()
        current_perf = trend[metric].iloc[-window:].mean()
        
        # Calculate relative degradation
        degradation = (current_perf - initial_perf) / initial_perf
        
        # For metrics where lower is better (like RMSE, MAE)
        should_retrain = degradation > threshold
        
        if should_retrain:
            self.logger.info(f"Model {model_id} should be retrained")
            self.logger.info(f"Initial {metric}: {initial_perf:.4f}, Current {metric}: {current_perf:.4f}")
            self.logger.info(f"Degradation: {degradation:.2%}")
            
        return should_retrain
    
    def plot_performance_trend(self, model_id: str, metric: str = 'rmse',
                              window: int = 10, save_path: Optional[str] = None) -> None:
        """
        Plot performance trend for a model.
        
        Args:
            model_id: Model ID
            metric: Metric to track
            window: Window size for rolling average
            save_path: Path to save the plot
        """
        trend = self.get_performance_trend(model_id, metric, window)
        
        if trend is None:
            self.logger.warning(f"No performance history for model {model_id}")
            return
            
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot raw metric
        plt.plot(trend['timestamp'], trend[metric], 'o-', alpha=0.5, label=metric)
        
        # Plot rolling average if available
        rolling_col = f'{metric}_rolling'
        if rolling_col in trend.columns:
            plt.plot(trend['timestamp'], trend[rolling_col], 'r-', linewidth=2,
                   label=f'{metric} ({window}-point rolling avg)')
            
        plt.title(f'Model {model_id} Performance Trend')
        plt.xlabel('Time')
        plt.ylabel(metric.upper())
        plt.grid(True)
        plt.legend()
        
        # Add annotations
        if len(trend) >= window * 2:
            initial_perf = trend[metric].iloc[:window].mean()
            current_perf = trend[metric].iloc[-window:].mean()
            degradation = (current_perf - initial_perf) / initial_perf
            
            plt.axhline(y=initial_perf, color='g', linestyle='--', alpha=0.7,
                      label=f'Initial: {initial_perf:.4f}')
            plt.axhline(y=current_perf, color='b', linestyle='--', alpha=0.7,
                      label=f'Current: {current_perf:.4f}')
            
            plt.text(trend['timestamp'].iloc[-1], current_perf,
                   f' Current: {current_perf:.4f}\n Change: {degradation:.2%}',
                   verticalalignment='center')
            
            plt.legend()
            
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class ModelAPI:
    """
    Simple API for model serving.
    
    Provides a consistent interface for making predictions with models.
    """
    
    def __init__(self, model_registry: ModelRegistry = None):
        """
        Initialize the model API.
        
        Args:
            model_registry: Model registry
        """
        self.logger = setup_logger()
        self.model_registry = model_registry or ModelRegistry()
        
        # Cache for loaded models
        self.model_cache = {}
        
        # Default device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.info(f"Model API initialized with device: {self.device}")
    
    def load_model(self, model_id: str = None, symbol: str = None) -> torch.nn.Module:
        """
        Load a model.
        
        Args:
            model_id: Model ID
            symbol: Symbol (to load active model)
            
        Returns:
            Loaded model
        """
        # If symbol is provided, get active model for the symbol
        if model_id is None and symbol is not None:
            active_model = self.model_registry.get_active_model(symbol)
            if active_model:
                model_id = active_model['id']
            else:
                raise ValueError(f"No active model found for symbol {symbol}")
                
        if model_id is None:
            raise ValueError("Either model_id or symbol must be provided")
            
        # Check if model is already loaded
        if model_id in self.model_cache:
            return self.model_cache[model_id]
            
        # Load model from registry
        model = self.model_registry.load_model(model_id, self.device)
        
        # Cache the model
        self.model_cache[model_id] = model
        
        return model
    
    def preprocess_features(self, features: pd.DataFrame, model_id: str) -> torch.Tensor:
        """
        Preprocess features for prediction.
        
        Args:
            features: Raw features
            model_id: Model ID
            
        Returns:
            Preprocessed features as torch.Tensor
        """
        # Get model metadata
        model_metadata = self.model_registry.get_model_metadata(model_id)
        if not model_metadata:
            raise ValueError(f"Model ID {model_id} not found in registry")
            
        # Get model parameters
        model_params = model_metadata['params']
        
        # Ensure we have the expected feature columns
        feature_cols = model_params.get('feature_cols', [])
        if not feature_cols:
            # Assume all numeric columns except known non-features
            exclude_cols = ['timestamp', 'symbol', 'target_next_close', 'target_return']
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Ensure minimal required features are present
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in features.columns:
                # Create dummy values for missing columns
                features[col] = features.select_dtypes(include=[np.number]).mean(axis=1)
        
        # Extract features that exist in the dataframe
        available_cols = [col for col in feature_cols if col in features.columns]
        if not available_cols:
            # If no matching columns, use all numeric columns
            available_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        
        X = features[available_cols].values
        
        # Reshape for sequence input if necessary
        sequence_length = model_params.get('sequence_length', 1)
        if len(X.shape) == 2:  # (samples, features)
            if sequence_length > 1:
                # Create sequences
                if len(X) < sequence_length:
                    # Handle case where there aren't enough samples
                    # Repeat the first sample to fill the sequence
                    padding = np.repeat(X[:1], sequence_length - len(X), axis=0)
                    X_padded = np.vstack([padding, X])
                    sequences = [X_padded[-sequence_length:]]
                else:
                    # Regular sequence creation
                    sequences = []
                    for i in range(len(X) - sequence_length + 1):
                        sequences.append(X[i:i+sequence_length])
                X = np.array(sequences)
            else:
                # Add sequence dimension
                X = X.reshape(-1, 1, X.shape[1])
        
        # Convert to torch.Tensor
        X_tensor = torch.FloatTensor(X)
        
        return X_tensor
    
    def predict(self, features: pd.DataFrame, model_id: str = None, 
               symbol: str = None) -> np.ndarray:
        """
        Make predictions with a model.
        
        Args:
            features: Features for prediction
            model_id: Model ID
            symbol: Symbol (to use active model)
            
        Returns:
            Predictions
        """
        # Load model
        model = self.load_model(model_id, symbol)
        
        # Get model ID (in case symbol was provided)
        if model_id is None and symbol is not None:
            active_model = self.model_registry.get_active_model(symbol)
            model_id = active_model['id']
        
        # Preprocess features
        X = self.preprocess_features(features, model_id)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            predictions = model(X)
            
        # Convert to numpy array
        predictions = predictions.cpu().numpy()
        
        return predictions

class AutoMLOps:
    """
    Automated MLOps system.
    
    Combines monitoring, drift detection, and retraining in a single system.
    """
    
    def __init__(self, model_registry: ModelRegistry = None,
                performance_monitor: PerformanceMonitor = None,
                drift_detector: DriftDetector = None,
                model_api: ModelAPI = None):
        """
        Initialize the AutoMLOps system.
        
        Args:
            model_registry: Model registry
            performance_monitor: Performance monitor
            drift_detector: Drift detector
            model_api: Model API
        """
        self.logger = setup_logger()
        
        # Initialize components
        self.model_registry = model_registry or ModelRegistry()
        self.performance_monitor = performance_monitor or PerformanceMonitor(self.model_registry)
        self.drift_detector = drift_detector or DriftDetector()
        self.model_api = model_api or ModelAPI(self.model_registry)
        
        # Get current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        # Set directories
        self.mlops_dir = os.path.join(project_root, 'mlops')
        os.makedirs(self.mlops_dir, exist_ok=True)
        
        self.logger.info("AutoMLOps system initialized")
    
    def auto_set_reference_data(self, symbol: str) -> None:
        """
        Automatically set reference data for drift detection based on validation data.
        
        Args:
            symbol: Symbol to set reference data for
        """
        # Get current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        processed_dir = os.path.join(project_root, 'data', 'processed')
        
        val_file = os.path.join(processed_dir, f"{symbol}_val.parquet")
        
        if os.path.exists(val_file):
            try:
                validation_data = pd.read_parquet(val_file)
                self.drift_detector.set_reference_data(validation_data)
                self.logger.info(f"Set reference data for drift detection using {val_file}")
                return True
            except Exception as e:
                self.logger.error(f"Error setting reference data: {str(e)}")
        else:
            self.logger.warning(f"Validation data file not found: {val_file}")
        
        return False
    
    def analyze_latest_predictions(self, symbol: str) -> Dict:
        """
        Analyze latest prediction results for a symbol.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Analysis results
        """
        # Get current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        results_dir = os.path.join(project_root, 'results')
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            self.logger.warning(f"Results directory created: {results_dir}")
            return {}
        
        # Find latest prediction results
        pattern = f"{symbol}_inference_results_*.csv"
        matching_files = []
        
        for filename in os.listdir(results_dir):
            if filename.startswith(f"{symbol}_inference_results_") and filename.endswith(".csv"):
                matching_files.append(os.path.join(results_dir, filename))
        
        if not matching_files:
            self.logger.warning(f"No prediction results found for symbol {symbol}")
            return {}
            
        # Sort by modification time (most recent first)
        latest_file = sorted(matching_files, key=os.path.getmtime)[-1]
        
        # Load prediction results
        predictions = pd.read_csv(latest_file)
        
        # Calculate metrics
        metrics = {}
        
        if 'Actual' in predictions.columns and 'Predicted' in predictions.columns:
            actuals = predictions['Actual'].values
            predicted = predictions['Predicted'].values
            
            # Calculate metrics
            mse = np.mean((predicted - actuals) ** 2)
            mae = np.mean(np.abs(predicted - actuals))
            rmse = np.sqrt(mse)
            
            # Calculate directional accuracy
            if len(actuals) > 1:
                actual_direction = np.sign(actuals[1:] - actuals[:-1])
                predicted_direction = np.sign(predicted[1:] - predicted[:-1])
                directional_accuracy = np.mean(actual_direction == predicted_direction)
            else:
                directional_accuracy = np.nan
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'directional_accuracy': float(directional_accuracy) if not np.isnan(directional_accuracy) else 0.0,
                'timestamp': datetime.now()
            }
        
        # Get active model for the symbol
        active_model = self.model_registry.get_active_model(symbol)
        
        if active_model:
            model_id = active_model['id']
            
            # Log performance
            if metrics:
                self.performance_monitor.log_prediction_performance(model_id, datetime.now(), metrics)
            
            # Check if retraining is needed
            retrain_recommended = self.performance_monitor.should_retrain(model_id)
        else:
            self.logger.warning(f"No active model found for symbol {symbol}")
            model_id = None
            retrain_recommended = False
        
        # Check for data drift
        drift_detected = False
        if active_model:
            # First ensure we have reference data
            has_reference = isinstance(self.drift_detector.reference_data, pd.DataFrame)
            
            if not has_reference:
                self.auto_set_reference_data(symbol)
                has_reference = isinstance(self.drift_detector.reference_data, pd.DataFrame)
            
            if has_reference and len(predictions) > 0:
                try:
                    drift_results = self.drift_detector.detect_drift(predictions)
                    drift_detected = drift_results['summary']['any_distribution_drift']
                except Exception as e:
                    self.logger.error(f"Error detecting drift: {str(e)}")
                    drift_detected = False
        
        # Prepare analysis results
        analysis = {
            'symbol': symbol,
            'model_id': model_id,
            'metrics': metrics,
            'retrain_recommended': retrain_recommended,
            'drift_detected': drift_detected,
            'timestamp': datetime.now()
        }
        
        # Save analysis results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_file = os.path.join(self.mlops_dir, f"{symbol}_analysis_{timestamp}.json")
        
        with open(analysis_file, 'w') as f:
            # Convert the entire analysis dictionary to be JSON serializable
            serializable_analysis = json_serializable(analysis)
            json.dump(serializable_analysis, f, indent=4)
        
        self.logger.info(f"Analysis for symbol {symbol} saved to {analysis_file}")
        
        return analysis
    
    def register_model_from_file(self, model_path: str, symbol: str, 
                                model_type: str = 'lstm', activate: bool = True) -> str:
        """
        Register a model from file.
        
        Args:
            model_path: Path to model file
            symbol: Symbol associated with the model
            model_type: Model type
            activate: Whether to activate the model
            
        Returns:
            Model ID
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Gather basic model info
        model_info = os.path.basename(model_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load model parameters - use fixed parameters since we can't extract them
        params = {
            'input_dim': 100,  # Default
            'hidden_dim': 64,  # Default
            'num_layers': 2,   # Default
            'sequence_length': 10  # Default
        }
        
        # Register model
        model_id = self.model_registry.register_model(
            model_path=model_path,
            model_type=model_type,
            metrics={},  # No metrics available
            params=params,
            symbol=symbol
        )
        
        # Activate model if requested
        if activate:
            self.model_registry.activate_model(model_id, symbol)
        
        return model_id
    
    def auto_register_latest_models(self) -> Dict[str, str]:
        """
        Automatically register the latest models for each symbol.
        
        Returns:
            Dictionary mapping symbols to model IDs
        """
        # Get current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        models_dir = os.path.join(project_root, 'models')
        
        if not os.path.exists(models_dir):
            self.logger.warning(f"Models directory not found: {models_dir}")
            return {}
        
        # Find model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt') or f.endswith('.pth')]
        
        if not model_files:
            self.logger.warning("No model files found")
            return {}
        
        # Get processed directory to find symbols
        processed_dir = os.path.join(project_root, 'data', 'processed')
        symbol_files = []
        
        if os.path.exists(processed_dir):
            symbol_files = [f for f in os.listdir(processed_dir) if f.endswith('_train.parquet')]
        
        # Extract symbols from filenames
        symbols = set([f.split('_')[0] for f in symbol_files])
        
        # If no symbols found, use default list
        if not symbols:
            symbols = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'}
        
        # Register latest model for each symbol
        registered_models = {}
        
        for symbol in symbols:
            # Find the latest model for this symbol
            symbol_models = [f for f in model_files if symbol.lower() in f.lower()]
            
            if not symbol_models:
                # If no symbol-specific models, use the latest model
                if model_files:
                    latest_model = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))[-1]
                else:
                    self.logger.warning(f"No models found for symbol {symbol}")
                    continue
            else:
                # Use the latest model for this symbol
                latest_model = sorted(symbol_models, 
                                     key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))[-1]
            
            model_path = os.path.join(models_dir, latest_model)
            
            try:
                model_id = self.register_model_from_file(model_path, symbol)
                registered_models[symbol] = model_id
            except Exception as e:
                self.logger.error(f"Error registering model for {symbol}: {str(e)}")
        
        return registered_models
    
    def generate_mlops_dashboard(self, symbols: List[str] = None) -> None:
        """
        Generate MLOps dashboard for monitoring models.
        
        Args:
            symbols: List of symbols to include (default: all active models)
        """
        # Get all active models if symbols not provided
        if symbols is None:
            active_models = self.model_registry.registry.get('active_models', {})
            symbols = list(active_models.keys())
        
        if not symbols:
            self.logger.warning("No symbols to include in dashboard")
            return
        
        # Create dashboard directory
        dashboard_dir = os.path.join(self.mlops_dir, 'dashboard')
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Generate dashboard for each symbol
        all_reports = []
        
        for symbol in symbols:
            # Get active model
            active_model = self.model_registry.get_active_model(symbol)
            
            if not active_model:
                self.logger.warning(f"No active model for symbol {symbol}")
                continue
            
            model_id = active_model['id']
            
            # Make sure we have reference data for drift detection
            self.auto_set_reference_data(symbol)
            
            # Generate performance plot (try-except to handle errors)
            performance_plot_path = None
            try:
                performance_plot_path = os.path.join(dashboard_dir, f"{symbol}_performance.png")
                self.performance_monitor.plot_performance_trend(
                    model_id, save_path=performance_plot_path
                )
            except Exception as e:
                self.logger.error(f"Error generating performance plot for {symbol}: {str(e)}")
            
            # Analyze latest predictions (try-except to handle errors)
            analysis = {}
            try:
                analysis = self.analyze_latest_predictions(symbol)
            except Exception as e:
                self.logger.error(f"Error analyzing predictions for {symbol}: {str(e)}")
                analysis = {
                    'symbol': symbol,
                    'model_id': model_id,
                    'metrics': {},
                    'retrain_recommended': False,
                    'drift_detected': False,
                    'timestamp': datetime.now()
                }
            
            # Generate report
            report = {
                'symbol': symbol,
                'model_id': model_id,
                'model_type': active_model.get('type', 'unknown'),
                'created_at': active_model.get('created_at', 'unknown'),
                'metrics': analysis.get('metrics', {}),
                'retrain_recommended': analysis.get('retrain_recommended', False),
                'drift_detected': analysis.get('drift_detected', False),
                'plots': {
                    'performance': os.path.basename(performance_plot_path) if performance_plot_path and os.path.exists(performance_plot_path) else None
                }
            }
            
            all_reports.append(report)
            
            # Save report
            report_path = os.path.join(dashboard_dir, f"{symbol}_report.json")
            
            with open(report_path, 'w') as f:
                serializable_report = json_serializable(report)
                json.dump(serializable_report, f, indent=4)
        
        # Generate summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols,
            'num_models': len(symbols),
            'retraining_needed': []
        }
        
        for report in all_reports:
            if report.get('retrain_recommended', False) or report.get('drift_detected', False):
                summary['retraining_needed'].append(report['symbol'])
        
        # Save summary
        summary_path = os.path.join(dashboard_dir, 'summary.json')
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        self.logger.info(f"MLOps dashboard generated at {dashboard_dir}")
        self.logger.info(f"Summary report saved to {summary_path}")

def run_mlops():
    """Run MLOps system for model monitoring and management."""
    logger = setup_logger()
    logger.info("Starting MLOps system")
    
    # Initialize MLOps components
    model_registry = ModelRegistry()
    performance_monitor = PerformanceMonitor(model_registry)
    drift_detector = DriftDetector()
    model_api = ModelAPI(model_registry)
    
    # Create AutoMLOps system
    mlops = AutoMLOps(
        model_registry=model_registry,
        performance_monitor=performance_monitor,
        drift_detector=drift_detector,
        model_api=model_api
    )
    
    # Auto-register latest models
    registered_models = mlops.auto_register_latest_models()
    
    if registered_models:
        logger.info(f"Registered {len(registered_models)} models")
        
        # Set reference data for each symbol for drift detection
        for symbol in registered_models.keys():
            mlops.auto_set_reference_data(symbol)
        
        # Generate MLOps dashboard
        mlops.generate_mlops_dashboard(symbols=list(registered_models.keys()))
    else:
        logger.warning("No models registered")
    
    logger.info("MLOps system completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MLOps system')
    parser.add_argument('--symbol', type=str, help='Symbol to analyze')
    parser.add_argument('--model', type=str, help='Path to model file to register')
    parser.add_argument('--dashboard', action='store_true', help='Generate MLOps dashboard')
    
    args = parser.parse_args()
    
    if args.model and args.symbol:
        # Register specific model
        registry = ModelRegistry()
        model_id = registry.register_model(
            model_path=args.model,
            model_type='lstm',
            metrics={},
            params={},
            symbol=args.symbol
        )
        registry.activate_model(model_id, args.symbol)
        print(f"Model registered with ID: {model_id}")
    elif args.dashboard:
        # Generate dashboard
        mlops = AutoMLOps()
        mlops.generate_mlops_dashboard()
    else:
        # Run full MLOps system
        run_mlops()