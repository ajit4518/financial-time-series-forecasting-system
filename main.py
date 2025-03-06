"""
Time Series Forecasting Engine - Main Runner

This script orchestrates the entire time series forecasting project,
from data collection to model training, evaluation, and deployment.
"""

import os
import sys
import subprocess
import time
import logging
import importlib
import platform
import shutil
from datetime import datetime
import argparse
from pathlib import Path

# Set up logging
def setup_logger():
    """Set up a logger for the main runner."""
    logger = logging.getLogger('main_runner')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Add file handler
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Helper function to run shell commands
def run_command(command, description=None, check=True):
    """
    Run a shell command and log the output.
    
    Args:
        command: Command to run
        description: Description of the command
        check: Whether to raise an exception if the command fails
        
    Returns:
        True if the command succeeded, False otherwise
    """
    logger = setup_logger()
    
    if description:
        logger.info(f"Step: {description}")
    
    logger.info(f"Running command: {command}")
    
    try:
        # Fix for Windows paths with spaces - avoid shell=True
        if platform.system() == 'Windows':
            # For Windows, use list form instead of string to avoid shell parsing issues
            if isinstance(command, str):
                # Split by spaces but respect quoted strings
                import shlex
                args = shlex.split(command)
            else:
                args = command
            
            result = subprocess.run(args, check=check, capture_output=True, text=True)
        else:
            # For Unix-like systems, shell=True is generally fine
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        
        if result.stdout:
            logger.info(f"Command output: {result.stdout}")
        
        if result.stderr:
            logger.warning(f"Command error: {result.stderr}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {e}")
        logger.error(f"Error output: {e.stderr}")
        
        if check:
            raise
        
        return False
    except Exception as e:
        logger.error(f"Exception while running command: {str(e)}")
        if check:
            raise
        return False

# Check environment
def check_environment():
    """
    Check if the required packages are installed and directories exist.
    
    Returns:
        True if the environment is ready, False otherwise
    """
    logger = setup_logger()
    logger.info("Checking environment setup...")
    
    # Check Python version
    python_version = platform.python_version()
    logger.info(f"Python version: {python_version}")
    
    # Check required packages
    required_packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'torch',
        'scipy'
    ]
    
    optional_packages = [
        'fastapi',  # For API server
        'uvicorn',  # For API server
        'pyarrow',  # For parquet files
        'fastparquet'  # Alternative for parquet files
    ]
    
    missing_required = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_required.append(package)
    
    if missing_required:
        logger.error(f"Missing required packages: {', '.join(missing_required)}")
        logger.error("Please install these packages before continuing")
        return False
    else:
        logger.info("All core packages are available")
    
    # Check optional packages
    missing_optional = []
    for package in optional_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_optional:
        logger.warning(f"Missing optional packages: {', '.join(missing_optional)}")
        logger.info("Consider installing these packages for full functionality")
    
    return True

# Run data pipeline
def run_data_pipeline():
    """
    Run the data pipeline to collect, preprocess, and engineer features.
    
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logger()
    logger.info("Running data pipeline...")
    
    # Get the current script directory
    current_dir = os.path.abspath(os.getcwd())
    
    # Direct module execution without shell
    try:
        # Try importing and running the module directly first
        sys.path.insert(0, current_dir)
        
        try:
            from src.pipeline.main_pipeline import run_pipeline
            logger.info("Executing pipeline module directly")
            result = run_pipeline()
            logger.info("Data pipeline completed successfully")
            return True
        except ImportError:
            logger.warning("Could not import pipeline module directly, falling back to subprocess")
        
        # Find the main_pipeline.py script
        pipeline_script = os.path.join(current_dir, "src", "pipeline", "main_pipeline.py")
        
        # If the script doesn't exist, try to copy the template
        if not os.path.exists(pipeline_script):
            template_dir = os.path.join(current_dir, "templates")
            if os.path.exists(os.path.join(template_dir, "main_pipeline.py")):
                logger.info("Copying main_pipeline.py template")
                os.makedirs(os.path.dirname(pipeline_script), exist_ok=True)
                shutil.copy(os.path.join(template_dir, "main_pipeline.py"), pipeline_script)
        
        # Run the script
        if os.path.exists(pipeline_script):
            # Use Python executable path to avoid issues with spaces
            python_exe = sys.executable
            success = run_command(
                [python_exe, pipeline_script],
                "Running the data pipeline to collect, process, and feature engineer the data"
            )
            return success
        else:
            logger.error(f"Pipeline script not found: {pipeline_script}")
            return False
            
    except Exception as e:
        logger.error(f"Error running data pipeline: {str(e)}")
        return False

# Run model training
def run_model_training():
    """
    Run model training on the prepared data.
    
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logger()
    logger.info("Running model training...")
    
    # Get the current script directory
    current_dir = os.path.abspath(os.getcwd())
    
    # Direct module execution without shell
    try:
        # Try importing and running the module directly first
        sys.path.insert(0, current_dir)
        
        try:
            from src.models.model_runner import train_models
            logger.info("Executing model training module directly")
            train_models()
            logger.info("Model training completed successfully")
            return True
        except ImportError:
            logger.warning("Could not import model training module directly, falling back to subprocess")
        
        # Find the model_runner.py script
        model_script = os.path.join(current_dir, "src", "models", "model_runner.py")
        
        # If the script doesn't exist, try to copy the template
        if not os.path.exists(model_script):
            template_dir = os.path.join(current_dir, "templates")
            if os.path.exists(os.path.join(template_dir, "model_runner.py")):
                logger.info("Copying model_runner.py template")
                os.makedirs(os.path.dirname(model_script), exist_ok=True)
                shutil.copy(os.path.join(template_dir, "model_runner.py"), model_script)
        
        # Run the script
        if os.path.exists(model_script):
            # Use Python executable path to avoid issues with spaces
            python_exe = sys.executable
            success = run_command(
                [python_exe, model_script],
                "Running model training"
            )
            return success
        else:
            logger.error(f"Model training script not found: {model_script}")
            return False
            
    except Exception as e:
        logger.error(f"Error running model training: {str(e)}")
        return False

# Run model inference
def run_model_inference():
    """
    Run model inference on validation data.
    
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logger()
    logger.info("Running model inference...")
    
    # Get the current script directory
    current_dir = os.path.abspath(os.getcwd())
    
    # Direct module execution without shell
    try:
        # Try importing and running the module directly first
        sys.path.insert(0, current_dir)
        
        try:
            from src.models.model_inference import run_inference
            logger.info("Executing model inference module directly")
            run_inference()
            logger.info("Model inference completed successfully")
            return True
        except ImportError:
            logger.warning("Could not import model inference module directly, falling back to subprocess")
        
        # Find the model_inference.py script
        inference_script = os.path.join(current_dir, "src", "models", "model_inference.py")
        
        # If the script doesn't exist, try to copy the template
        if not os.path.exists(inference_script):
            template_dir = os.path.join(current_dir, "templates")
            if os.path.exists(os.path.join(template_dir, "model_inference.py")):
                logger.info("Copying model_inference.py template")
                os.makedirs(os.path.dirname(inference_script), exist_ok=True)
                shutil.copy(os.path.join(template_dir, "model_inference.py"), inference_script)
        
        # Run the script
        if os.path.exists(inference_script):
            # Use Python executable path to avoid issues with spaces
            python_exe = sys.executable
            success = run_command(
                [python_exe, inference_script],
                "Running model inference"
            )
            return success
        else:
            logger.error(f"Model inference script not found: {inference_script}")
            return False
            
    except Exception as e:
        logger.error(f"Error running model inference: {str(e)}")
        return False

# Run backtesting
def run_backtesting():
    """
    Run backtesting for trading strategies.
    
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logger()
    logger.info("Running trading strategy backtesting...")
    
    # Get the current script directory
    current_dir = os.path.abspath(os.getcwd())
    
    # Direct module execution without shell
    try:
        # Try importing and running the module directly first
        sys.path.insert(0, current_dir)
        
        try:
            from src.backtesting.trading_strategy import run_backtest
            logger.info("Executing backtesting module directly")
            # Find the latest inference results
            results_dir = os.path.join(current_dir, "results")
            if os.path.exists(results_dir):
                inference_files = [f for f in os.listdir(results_dir) if "inference_results" in f]
                if inference_files:
                    # Use the latest file
                    latest_file = max(inference_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
                    symbol = latest_file.split('_')[0] if '_' in latest_file else None
                    run_backtest(os.path.join(results_dir, latest_file), symbol)
                    logger.info("Backtesting completed successfully")
                    return True
            
            logger.warning("No inference results found for backtesting")
            return False
            
        except ImportError:
            logger.warning("Could not import backtesting module directly, falling back to subprocess")
        
        # Find the trading_strategy.py script
        backtest_script = os.path.join(current_dir, "src", "backtesting", "trading_strategy.py")
        
        # If the script doesn't exist, try to copy the template
        if not os.path.exists(backtest_script):
            template_dir = os.path.join(current_dir, "templates")
            if os.path.exists(os.path.join(template_dir, "trading_strategy.py")):
                logger.info("Copying trading_strategy.py template")
                os.makedirs(os.path.dirname(backtest_script), exist_ok=True)
                shutil.copy(os.path.join(template_dir, "trading_strategy.py"), backtest_script)
        
        # Run the script
        if os.path.exists(backtest_script):
            # Use Python executable path to avoid issues with spaces
            python_exe = sys.executable
            success = run_command(
                [python_exe, backtest_script],
                "Running trading strategy backtesting",
                check=False  # Don't fail if this step has issues
            )
            return success
        else:
            logger.error(f"Backtesting script not found: {backtest_script}")
            return False
            
    except Exception as e:
        logger.error(f"Error running backtesting: {str(e)}")
        return False

# Run MLOps system
def run_mlops():
    """
    Run MLOps monitoring and model management.
    
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logger()
    logger.info("Running MLOps system...")
    
    # Get the current script directory
    current_dir = os.path.abspath(os.getcwd())
    
    # Direct module execution without shell
    try:
        # Try importing and running the module directly first
        sys.path.insert(0, current_dir)
        
        try:
            from src.mlops.model_monitoring import run_mlops
            logger.info("Executing MLOps module directly")
            run_mlops()
            logger.info("MLOps system completed successfully")
            return True
        except ImportError:
            logger.warning("Could not import MLOps module directly, falling back to subprocess")
        
        # Find the model_monitoring.py script
        mlops_script = os.path.join(current_dir, "src", "mlops", "model_monitoring.py")
        
        # If the script doesn't exist, try to copy the template
        if not os.path.exists(mlops_script):
            template_dir = os.path.join(current_dir, "templates")
            if os.path.exists(os.path.join(template_dir, "model_monitoring.py")):
                logger.info("Copying model_monitoring.py template")
                os.makedirs(os.path.dirname(mlops_script), exist_ok=True)
                shutil.copy(os.path.join(template_dir, "model_monitoring.py"), mlops_script)
        
        # Run the script
        if os.path.exists(mlops_script):
            # Use Python executable path to avoid issues with spaces
            python_exe = sys.executable
            success = run_command(
                [python_exe, mlops_script],
                "Running MLOps monitoring and model management",
                check=False  # Don't fail if this step has issues
            )
            return success
        else:
            logger.error(f"MLOps script not found: {mlops_script}")
            return False
            
    except Exception as e:
        logger.error(f"Error running MLOps system: {str(e)}")
        return False

# Run API server
def run_api_server():
    """
    Run the model serving API server.
    
    Returns:
        Process object for the API server
    """
    logger = setup_logger()
    logger.info("Starting model API server...")
    
    # Check if required packages are installed
    try:
        import fastapi
        import uvicorn
    except ImportError:
        logger.error("FastAPI and/or uvicorn packages not installed. Cannot start API server.")
        logger.error("Install with: pip install fastapi uvicorn")
        return None
    
    # Get the current script directory
    current_dir = os.path.abspath(os.getcwd())
    
    # Find the model_api_server.py script
    api_script = os.path.join(current_dir, "src", "api", "model_api_server.py")
    
    # If the script doesn't exist, try to copy the template
    if not os.path.exists(api_script):
        template_dir = os.path.join(current_dir, "templates")
        if os.path.exists(os.path.join(template_dir, "model_api_server.py")):
            logger.info("Copying model_api_server.py template")
            os.makedirs(os.path.dirname(api_script), exist_ok=True)
            shutil.copy(os.path.join(template_dir, "model_api_server.py"), api_script)
    
    # Run the API server
    if os.path.exists(api_script):
        try:
            # Use Python executable path to avoid issues with spaces
            python_exe = sys.executable
            
            # Start API server as a separate process
            api_process = subprocess.Popen(
                [python_exe, api_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give the server a moment to start
            time.sleep(3)
            
            # Check if the process is still running
            if api_process.poll() is None:
                logger.info("API server started successfully")
                return api_process
            else:
                stdout, stderr = api_process.communicate()
                logger.error(f"API server failed to start: {stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error starting API server: {str(e)}")
            return None
    else:
        logger.error(f"API server script not found: {api_script}")
        return None

# Test API with a sample request
def test_api():
    """
    Test the API with a sample request.
    
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logger()
    logger.info("Testing model API...")
    
    # Prepare sample request
    sample_request = {
        "symbol": "AAPL",
        "features": {
            "open": 150.0,
            "high": 152.0,
            "low": 149.0,
            "close": 151.0,
            "volume": 10000000
        }
    }
    
    # Create a temporary request file
    import json
    request_file = os.path.join(os.getcwd(), "request.json")
    with open(request_file, 'w') as f:
        json.dump(sample_request, f)
    
    # Test with curl if available
    curl_cmd = 'curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @{}'.format(request_file)
    
    try:
        success = run_command(curl_cmd, "Testing API with a sample prediction request", check=False)
        
        # Clean up
        if os.path.exists(request_file):
            os.remove(request_file)
            
        return success
    except Exception as e:
        logger.error(f"Error testing API: {str(e)}")
        
        # Clean up
        if os.path.exists(request_file):
            os.remove(request_file)
            
        return False

# Run the complete pipeline
def run_full_pipeline():
    """
    Run the complete pipeline from data collection to model serving.
    
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logger()
    
    # Step 1: Run data pipeline
    if not run_data_pipeline():
        logger.error("Data pipeline failed. Stopping the process.")
        return False
    
    # Step 2: Run model training
    if not run_model_training():
        logger.error("Model training failed. Stopping the process.")
        return False
    
    # Step 3: Run model inference
    if not run_model_inference():
        logger.error("Model inference failed. Stopping the process.")
        return False
    
    # Step 4: Run backtesting
    run_backtesting()  # Continue even if this step fails
    
    # Step 5: Run MLOps
    run_mlops()  # Continue even if this step fails
    
    # Step 6: Start API server
    api_process = run_api_server()
    
    # Step 7: Test API
    if api_process is not None:
        test_api()
        
        # Terminate the API server
        api_process.terminate()
    
    logger.info("Full pipeline completed successfully!")
    return True

def show_banner():
    """Show an ASCII banner for the project."""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║               TIME SERIES FORECASTING ENGINE                   ║
║                                                                ║
║  A comprehensive ML system for high-frequency trading data     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Main function to run the project."""
    parser = argparse.ArgumentParser(description='Time Series Forecasting Engine')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    parser.add_argument('--data', action='store_true', help='Run only the data pipeline')
    parser.add_argument('--train', action='store_true', help='Run only model training')
    parser.add_argument('--inference', action='store_true', help='Run only model inference')
    parser.add_argument('--backtest', action='store_true', help='Run only backtesting')
    parser.add_argument('--mlops', action='store_true', help='Run only MLOps system')
    parser.add_argument('--api', action='store_true', help='Run only API server')
    
    args = parser.parse_args()
    
    # Show banner
    show_banner()
    
    # Set up logger
    logger = setup_logger()
    logger.info("Starting the complete time series forecasting pipeline")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed. Please fix the issues and try again.")
        return
    
    # Run selected components
    if args.all:
        run_full_pipeline()
    elif args.data:
        run_data_pipeline()
    elif args.train:
        run_model_training()
    elif args.inference:
        run_model_inference()
    elif args.backtest:
        run_backtesting()
    elif args.mlops:
        run_mlops()
    elif args.api:
        api_process = run_api_server()
        if api_process is not None:
            try:
                print("API server running at http://localhost:8000")
                print("Press Ctrl+C to stop...")
                api_process.wait()
            except KeyboardInterrupt:
                api_process.terminate()
                print("API server stopped")
    else:
        # If no specific component is selected, run the full pipeline
        run_full_pipeline()

if __name__ == "__main__":
    main()