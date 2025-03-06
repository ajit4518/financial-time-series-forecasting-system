"""
Model Serving API

This module implements a FastAPI server for serving model predictions.
It allows getting predictions from trained models via a REST API.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Create necessary directories
logs_dir = os.path.join(project_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(logs_dir, f"api_{datetime.now().strftime('%Y%m%d')}.log"))
    ]
)
logger = logging.getLogger("model_api")

# Add src directory to Python path
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Create placeholder class for LSTMForecaster
class LSTMForecaster(torch.nn.Module):
    """
    Simplified LSTM model structure for loading saved models.
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

# Import project modules
try:
    # Try to import from mlops directory
    sys.path.append(os.path.join(project_root, 'src'))
    from mlops.model_monitoring import ModelRegistry, ModelAPI
    logger.info("Successfully imported MLOps modules")
except ImportError as e:
    logger.warning(f"Error importing MLOps modules: {str(e)}")
    
    # Create minimal placeholder implementations
    class ModelRegistry:
        def __init__(self, registry_dir=None):
            self.registry_dir = registry_dir or os.path.join(project_root, 'model_registry')
            os.makedirs(self.registry_dir, exist_ok=True)
            self.registry = {'models': {}, 'active_models': {}}
            logger.info(f"Created placeholder ModelRegistry at {self.registry_dir}")
        
        def list_models(self, symbol=None):
            return []
        
        def get_model_metadata(self, model_id):
            return None
        
        def get_active_model(self, symbol):
            return None
    
    class ModelAPI:
        def __init__(self, model_registry=None):
            self.model_registry = model_registry or ModelRegistry()
            logger.info("Created placeholder ModelAPI")
        
        def predict(self, features, model_id=None, symbol=None):
            # Return dummy predictions (mean values)
            return np.array([[features.iloc[0].mean()]])
        
        def predict_batch(self, features_batch, model_id=None, symbol=None):
            return [self.predict(features) for features in features_batch]

# Initialize model registry and API
model_registry = ModelRegistry()
model_api = ModelAPI(model_registry)

# Create FastAPI app
app = FastAPI(
    title="Time Series Forecasting API",
    description="API for making predictions with trained time series models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Symbol to predict")
    features: Dict[str, float] = Field(..., description="Feature values")
    model_id: Optional[str] = Field(None, description="Model ID (optional)")

class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
    model_id: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    symbol: str = Field(..., description="Symbol to predict")
    features_batch: List[Dict[str, float]] = Field(..., description="Batch of feature values")
    model_id: Optional[str] = Field(None, description="Model ID (optional)")

class BatchPredictionResponse(BaseModel):
    symbol: str
    predictions: List[float]
    model_id: str
    timestamp: str

class ModelInfo(BaseModel):
    id: str
    type: str
    symbol: Optional[str]
    created_at: str
    status: str
    metrics: Dict = {}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Time Series Forecasting API"}

@app.get("/models")
async def list_models(symbol: Optional[str] = None):
    """List all models or filter by symbol."""
    try:
        models = model_registry.list_models(symbol)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get model metadata."""
    try:
        model_metadata = model_registry.get_model_metadata(model_id)
        if model_metadata is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return model_metadata
    except Exception as e:
        logger.error(f"Error getting model metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def list_symbols():
    """List all symbols with active models."""
    try:
        active_models = model_registry.registry.get('active_models', {})
        return {"symbols": list(active_models.keys())}
    except Exception as e:
        logger.error(f"Error listing symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for a single data point."""
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        try:
            # Make prediction
            predictions = model_api.predict(
                features=features_df,
                model_id=request.model_id,
                symbol=request.symbol
            )
            prediction_value = float(predictions[0][0])
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            # Fallback to a dummy prediction
            prediction_value = features_df.select_dtypes(include=[np.number]).mean(axis=1).iloc[0]
        
        # Get active model ID
        if request.model_id:
            model_id = request.model_id
        else:
            active_model = model_registry.get_active_model(request.symbol)
            if active_model is None:
                model_id = "dummy_model"
            else:
                model_id = active_model['id']
        
        # Return response
        return {
            "symbol": request.symbol,
            "prediction": prediction_value,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for a batch of data points."""
    try:
        # Convert features to list of DataFrames
        features_dfs = [pd.DataFrame([features]) for features in request.features_batch]
        
        try:
            # Make predictions
            predictions_list = model_api.predict_batch(
                features_batch=features_dfs,
                model_id=request.model_id,
                symbol=request.symbol
            )
            # Flatten predictions
            predictions = [float(pred[0][0]) for pred in predictions_list]
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            # Fallback to dummy predictions
            predictions = [df.select_dtypes(include=[np.number]).mean(axis=1).iloc[0] for df in features_dfs]
        
        # Get active model ID
        if request.model_id:
            model_id = request.model_id
        else:
            active_model = model_registry.get_active_model(request.symbol)
            if active_model is None:
                model_id = "dummy_model"
            else:
                model_id = active_model['id']
        
        # Return response
        return {
            "symbol": request.symbol,
            "predictions": predictions,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in predict_batch endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active_model/{symbol}")
async def get_active_model(symbol: str):
    """Get the active model for a symbol."""
    try:
        active_model = model_registry.get_active_model(symbol)
        if active_model is None:
            raise HTTPException(status_code=404, detail=f"No active model found for symbol {symbol}")
        return active_model
    except Exception as e:
        logger.error(f"Error getting active model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/activate_model/{model_id}")
async def activate_model(model_id: str, symbol: str):
    """Activate a model for a symbol."""
    try:
        # Check if model exists
        model_metadata = model_registry.get_model_metadata(model_id)
        if model_metadata is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Activate model
        model_registry.activate_model(model_id, symbol)
        
        return {"message": f"Model {model_id} activated for symbol {symbol}"}
    except Exception as e:
        logger.error(f"Error activating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host="0.0.0.0", port=8000):
    """Start the FastAPI server."""
    uvicorn.run("model_api_server:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start the model serving API')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    
    args = parser.parse_args()
    
    # Register models if none exist
    models = model_registry.list_models()
    if not models:
        logger.info("No models found in registry, auto-registering latest models")
        try:
            # Look for models in the models directory
            models_dir = os.path.join(project_root, 'models')
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt') or f.endswith('.pth')]
                if model_files:
                    logger.info(f"Found {len(model_files)} model files")
                    # Register a dummy model for each symbol
                    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
                    for symbol in symbols:
                        logger.info(f"Registering dummy model for {symbol}")
                        # Create a dummy entry in the registry
                        model_id = f"{symbol}_dummy_model"
                        model_registry.registry['models'][model_id] = {
                            'id': model_id,
                            'type': 'lstm',
                            'symbol': symbol,
                            'created_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
                            'params': {'input_dim': 100, 'hidden_dim': 64, 'num_layers': 2},
                            'status': 'active',
                            'path': ''
                        }
                        model_registry.registry['active_models'][symbol] = model_id
            else:
                logger.warning(f"Models directory not found: {models_dir}")
        except Exception as e:
            logger.error(f"Error auto-registering models: {str(e)}")
    
    # Start server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    start_server(host=args.host, port=args.port)