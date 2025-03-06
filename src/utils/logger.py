"""
Logger Module - Provides logging functionality for the application.
"""
import logging
import os
from datetime import datetime

def setup_logger(name, log_level=logging.INFO):
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d')
    file_handler = logging.FileHandler(f'logs/app_{timestamp}.log')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger
