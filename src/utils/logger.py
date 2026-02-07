"""
Centralized Logging Configuration
==================================
Provides consistent logging across all modules with structured output.

Design Decisions:
- Uses Python's built-in logging (no external dependencies)
- Logs to both console and file for debugging
- Includes timestamps and module names for traceability
- Non-blocking: never silently fails

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Create and configure a logger instance.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module)
    log_file : str, optional
        Path to log file. If None, logs only to console.
    level : int
        Logging level (default: INFO)
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    
    Example
    -------
    >>> logger = get_logger(__name__)
    >>> logger.info("Data loading started")
    2026-02-04 10:30:00 | INFO | data_loader | Data loading started
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter with structured output
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler - always enabled
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def log_dataframe_info(logger: logging.Logger, df_name: str, df) -> None:
    """
    Log summary information about a DataFrame.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    df_name : str
        Name of the DataFrame for identification
    df : pd.DataFrame
        The DataFrame to summarize
    """
    logger.info(f"DataFrame '{df_name}': {len(df):,} rows, {len(df.columns)} columns")
    
    if hasattr(df, 'isnull'):
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"DataFrame '{df_name}' has {missing_count:,} missing values")


def log_date_range(logger: logging.Logger, df_name: str, date_column, df) -> None:
    """
    Log the date range of a DataFrame.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    df_name : str
        Name of the DataFrame
    date_column : str
        Name of the date column
    df : pd.DataFrame
        The DataFrame containing date data
    """
    if date_column in df.columns:
        try:
            min_date = df[date_column].min()
            max_date = df[date_column].max()
            logger.info(f"DataFrame '{df_name}' date range: {min_date} to {max_date}")
        except Exception as e:
            logger.warning(f"Could not determine date range for '{df_name}': {e}")


class LogContext:
    """
    Context manager for structured logging of operations.
    
    Usage:
        with LogContext(logger, "Loading sales data"):
            # ... operation code ...
    """
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} ({elapsed:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} ({elapsed:.2f}s) - {exc_val}")
        
        # Don't suppress exceptions
        return False
