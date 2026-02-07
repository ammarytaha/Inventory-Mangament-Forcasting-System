"""
Data Loading Service
=====================
Robust data loading with schema validation, timestamp conversion, and logging.

Design Principles:
- Never silently fail
- Validate schemas before processing
- Convert UNIX timestamps automatically
- Log record counts and date ranges
- Handle missing values safely

Usage:
    loader = DataLoader(data_dir="path/to/csvs")
    result = loader.load_all()
    
    sales_data = result.data["fct_orders.csv"]
    validation = result.validation_results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from utils.logger import get_logger, LogContext
from utils.validators import SchemaValidator, ValidationResult
from utils.constants import (
    SALES_SCHEMA,
    INVENTORY_SCHEMA,
    EXPIRY_SCHEMA,
    MENU_SCHEMA
)

logger = get_logger(__name__)


@dataclass
class LoadResult:
    """
    Result of a data loading operation.
    
    Attributes
    ----------
    data : Dict[str, pd.DataFrame]
        Loaded DataFrames keyed by file name
    validation_results : Dict[str, ValidationResult]
        Validation results for each file
    summary : Dict[str, Any]
        Loading summary statistics
    """
    data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    validation_results: Dict[str, ValidationResult] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


class DataLoader:
    """
    Robust data loading service with validation and transformation.
    
    This service handles:
    - Loading CSVs with proper encoding
    - Schema validation
    - UNIX timestamp conversion
    - Missing value handling
    - Comprehensive logging
    
    Assumptions (documented for hackathon):
    - All input files are UTF-8 encoded CSVs
    - UNIX timestamps are in seconds (not milliseconds)
    - ID columns should be integers (nullable)
    - Date columns contain either UNIX timestamps or ISO strings
    
    Example
    -------
    >>> loader = DataLoader("./data")
    >>> result = loader.load_all()
    >>> orders = result.data["fct_orders.csv"]
    >>> print(f"Loaded {len(orders)} orders")
    """
    
    # Files that are critical for the system
    CRITICAL_FILES = [
        "fct_orders.csv",
        "fct_order_items.csv",
        "dim_menu_items.csv"
    ]
    
    # Files that enhance analysis but are optional
    OPTIONAL_FILES = [
        "dim_skus.csv",
        "dim_places.csv",
        "dim_bill_of_materials.csv",
        "fct_inventory_reports.csv",
        "dim_campaigns.csv",
        "fct_campaigns.csv"
    ]
    
    def __init__(self, data_dir: str, validate_schemas: bool = True):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        data_dir : str
            Path to directory containing CSV files
        validate_schemas : bool
            Whether to validate schemas on load (default: True)
        """
        self.data_dir = Path(data_dir)
        self.validate_schemas = validate_schemas
        self.validator = SchemaValidator()
        self._data: Dict[str, pd.DataFrame] = {}
        self._validation_results: Dict[str, ValidationResult] = {}
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info(f"DataLoader initialized with data_dir: {self.data_dir}")
    
    def load_all(self) -> LoadResult:
        """
        Load all CSV files from the data directory.
        
        Returns
        -------
        LoadResult
            Contains loaded data, validation results, and summary
        """
        result = LoadResult()
        
        with LogContext(logger, "Loading all data files"):
            # Find all CSV files
            csv_files = list(self.data_dir.glob("*.csv"))
            logger.info(f"Found {len(csv_files)} CSV files")
            
            # Load each file
            for csv_path in csv_files:
                try:
                    df = self._load_single_file(csv_path)
                    if df is not None:
                        result.data[csv_path.name] = df
                except Exception as e:
                    logger.error(f"Failed to load {csv_path.name}: {e}")
                    result.validation_results[csv_path.name] = ValidationResult(
                        is_valid=False,
                        errors=[str(e)]
                    )
            
            # Copy validation results
            result.validation_results = self._validation_results.copy()
            
            # Generate summary
            result.summary = self._generate_summary(result.data)
        
        return result
    
    def load_file(self, file_name: str) -> Optional[pd.DataFrame]:
        """
        Load a single CSV file.
        
        Parameters
        ----------
        file_name : str
            Name of the file to load (e.g., "fct_orders.csv")
        
        Returns
        -------
        pd.DataFrame or None
            Loaded DataFrame, or None if loading failed
        """
        file_path = self.data_dir / file_name
        return self._load_single_file(file_path)
    
    # Size threshold for chunked loading (10MB)
    LARGE_FILE_THRESHOLD_BYTES = 10 * 1024 * 1024
    CHUNK_SIZE = 50000  # Rows per chunk
    
    def _load_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load and process a single CSV file.
        
        Parameters
        ----------
        file_path : Path
            Full path to the CSV file
        
        Returns
        -------
        pd.DataFrame or None
            Processed DataFrame, or None if loading failed
        """
        file_name = file_path.name
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        with LogContext(logger, f"Loading {file_name}"):
            # Check file size - use chunked loading for large files
            file_size = file_path.stat().st_size
            use_chunked = file_size > self.LARGE_FILE_THRESHOLD_BYTES
            
            if use_chunked:
                logger.info(f"Large file detected ({file_size / 1024 / 1024:.1f} MB), using chunked loading")
            
            # Load CSV
            try:
                if use_chunked:
                    df = self._load_large_file(file_path, 'utf-8')
                else:
                    df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
            except UnicodeDecodeError:
                # Fallback to latin-1 if UTF-8 fails
                if use_chunked:
                    df = self._load_large_file(file_path, 'latin-1')
                else:
                    df = pd.read_csv(file_path, low_memory=False, encoding='latin-1')
                logger.warning(f"Used latin-1 encoding for {file_name}")
            except MemoryError:
                # If still fails, try chunked loading
                logger.warning(f"Memory error, falling back to chunked loading for {file_name}")
                df = self._load_large_file(file_path, 'utf-8')
            
            if len(df) == 0:
                logger.warning(f"File is empty: {file_name}")
                self._data[file_name] = df
                return df
            
            # Log basic stats
            logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Convert timestamps (skip for very large files to save memory)
            if len(df) < 500000:
                df = self._convert_timestamps(df, file_name)
                # Standardize ID columns (skip for large files)
                df = self._standardize_ids(df)
            else:
                logger.info(f"Skipping timestamp conversion for large file ({len(df):,} rows)")
            
            # Validate schema
            if self.validate_schemas:
                self._validate_file(df, file_name)
            
            # Log date range if applicable
            self._log_date_range(df, file_name)
            
            self._data[file_name] = df
            return df
    
    def _load_large_file(self, file_path: Path, encoding: str) -> pd.DataFrame:
        """
        Load a large CSV file using chunked reading.
        
        This approach:
        - Reads file in chunks to avoid memory issues
        - Uses memory-efficient dtypes
        - Concatenates chunks at the end
        
        Parameters
        ----------
        file_path : Path
            Path to the CSV file
        encoding : str
            File encoding to use
        
        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
        """
        chunks = []
        chunk_count = 0
        
        # Read in chunks
        for chunk in pd.read_csv(
            file_path,
            encoding=encoding,
            chunksize=self.CHUNK_SIZE,
            low_memory=True
        ):
            chunks.append(chunk)
            chunk_count += 1
            
            if chunk_count % 10 == 0:
                logger.info(f"Loaded {chunk_count} chunks ({chunk_count * self.CHUNK_SIZE:,} rows)")
        
        # Concatenate all chunks
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Combined {chunk_count} chunks into {len(df):,} rows")
        
        return df
    
    def _convert_timestamps(
        self,
        df: pd.DataFrame,
        file_name: str
    ) -> pd.DataFrame:
        """
        Convert UNIX timestamps to datetime objects.
        
        Detection Logic:
        - Check if column name suggests a date (created, updated, etc.)
        - Check if values are in typical UNIX timestamp range
        - Convert using seconds (not milliseconds)
        """
        timestamp_keywords = ['created', 'updated', 'date', 'time', 'at', 'expires']
        
        for col in df.columns:
            # Check if column name suggests a timestamp
            is_timestamp_name = any(
                keyword in col.lower() for keyword in timestamp_keywords
            )
            
            if not is_timestamp_name:
                continue
            
            # Check if values look like UNIX timestamps
            sample = df[col].dropna()
            if len(sample) == 0:
                continue
            
            if sample.dtype in ['int64', 'float64']:
                min_val = sample.min()
                max_val = sample.max()
                
                # UNIX timestamp range check (2020-01-01 to 2030-01-01)
                # Seconds: 1577836800 to 1893456000
                # Milliseconds would be 1000x larger
                if 1000000000 < min_val < 2000000000 and max_val < 2000000000:
                    # It's a UNIX timestamp in seconds
                    df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
                    logger.info(f"Converted '{col}' from UNIX timestamp (seconds)")
                elif min_val > 1000000000000:
                    # Might be milliseconds
                    df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')
                    logger.info(f"Converted '{col}' from UNIX timestamp (milliseconds)")
        
        return df
    
    def _standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize ID columns to nullable integers.
        
        Why nullable integers?
        - IDs should be integers, not floats
        - But we need to handle missing IDs gracefully
        - pd.Int64Dtype() allows NaN in integer columns
        """
        id_columns = [col for col in df.columns if col.endswith('_id') or col == 'id']
        
        for col in id_columns:
            if df[col].dtype == 'float64':
                try:
                    df[col] = df[col].astype('Int64')  # Nullable integer
                except (ValueError, TypeError):
                    # Keep as-is if conversion fails
                    pass
        
        return df
    
    def _validate_file(self, df: pd.DataFrame, file_name: str) -> None:
        """
        Validate a file against appropriate schema.
        """
        # Determine which schema to use
        schema = None
        
        if file_name in ['fct_orders.csv', 'fct_order_items.csv']:
            schema = SALES_SCHEMA
        elif file_name in ['dim_skus.csv', 'fct_inventory_reports.csv']:
            schema = INVENTORY_SCHEMA
        elif file_name in ['dim_menu_items.csv', 'dim_bill_of_materials.csv']:
            schema = MENU_SCHEMA
        
        if schema:
            result = self.validator.validate(df, schema, file_name)
            self._validation_results[file_name] = result
        else:
            # Basic validation for files without specific schemas
            result = ValidationResult()
            result.info["file_name"] = file_name
            result.info["row_count"] = len(df)
            result.info["column_count"] = len(df.columns)
            self._validation_results[file_name] = result
    
    def _log_date_range(self, df: pd.DataFrame, file_name: str) -> None:
        """
        Log the date range of a DataFrame if it contains date columns.
        """
        date_columns = ['created', 'updated', 'date', 'order_date']
        
        for col in date_columns:
            if col in df.columns:
                try:
                    # Check if already datetime
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        min_date = df[col].min()
                        max_date = df[col].max()
                        logger.info(
                            f"{file_name} date range ({col}): "
                            f"{min_date} to {max_date}"
                        )
                        break
                except Exception:
                    pass
    
    def _generate_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate a summary of loaded data.
        """
        summary = {
            "load_timestamp": datetime.now().isoformat(),
            "total_files": len(data),
            "total_rows": sum(len(df) for df in data.values()),
            "files": {}
        }
        
        for file_name, df in data.items():
            summary["files"][file_name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns)
            }
        
        # Check for critical files
        missing_critical = [
            f for f in self.CRITICAL_FILES 
            if f not in data or len(data[f]) == 0
        ]
        
        if missing_critical:
            summary["warnings"] = [
                f"Critical file missing or empty: {f}" 
                for f in missing_critical
            ]
            logger.warning(f"Missing critical files: {missing_critical}")
        
        return summary
    
    def get_sales_data(self, columns_only: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get order and order item data for demand analysis.
        
        Parameters
        ----------
        columns_only : bool
            If True (default), only return the columns needed for demand analysis.
            This reduces memory usage significantly for large datasets.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (orders DataFrame, order_items DataFrame)
        
        Raises
        ------
        ValueError
            If required sales data is not loaded
        """
        if "fct_orders.csv" not in self._data:
            raise ValueError("Orders data not loaded. Call load_all() first.")
        
        if "fct_order_items.csv" not in self._data:
            raise ValueError("Order items data not loaded. Call load_all() first.")
        
        orders = self._data["fct_orders.csv"]
        order_items = self._data["fct_order_items.csv"]
        
        # For large datasets, avoid copying and only return needed columns
        if len(order_items) > 500000 or columns_only:
            order_cols = ['id', 'created', 'place_id', 'status']
            order_cols = [c for c in order_cols if c in orders.columns]
            
            item_cols = ['item_id', 'order_id', 'created', 'quantity', 'price']
            item_cols = [c for c in item_cols if c in order_items.columns]
            
            logger.info(f"Returning minimal columns to reduce memory: {len(order_cols)} order cols, {len(item_cols)} item cols")
            return (orders[order_cols], order_items[item_cols])
        
        return (orders.copy(), order_items.copy())
    
    def get_inventory_data(self) -> pd.DataFrame:
        """
        Get SKU inventory data.
        
        Returns
        -------
        pd.DataFrame
            SKU inventory data
        """
        if "dim_skus.csv" not in self._data:
            logger.warning("SKU data not available")
            return pd.DataFrame()
        
        return self._data["dim_skus.csv"].copy()
    
    def get_menu_data(self) -> pd.DataFrame:
        """
        Get menu item data.
        
        Returns
        -------
        pd.DataFrame
            Menu items data
        """
        if "dim_menu_items.csv" not in self._data:
            logger.warning("Menu data not available")
            return pd.DataFrame()
        
        return self._data["dim_menu_items.csv"].copy()


def aggregate_daily_demand(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    date_column: str = 'created'
) -> pd.DataFrame:
    """
    Aggregate sales data to daily demand per product.
    
    This function:
    1. Merges orders with order items
    2. Extracts date from timestamp
    3. Aggregates quantity by item and date
    4. Fills missing dates with zero demand (if dataset is small enough)
    5. Sorts by date
    
    Parameters
    ----------
    orders : pd.DataFrame
        Orders table with order dates
    order_items : pd.DataFrame
        Order items table with quantities
    date_column : str
        Name of the date column in orders table
    
    Returns
    -------
    pd.DataFrame
        Daily demand per item with columns:
        - item_id
        - date
        - demand (quantity sold)
        - revenue (if price available)
    
    Assumptions (documented):
    - Each order_item has a quantity
    - Quantity represents demand (not returns/cancellations)
    - Missing dates mean zero demand
    """
    logger.info("Aggregating daily demand from sales data")
    
    # Validate inputs
    if len(orders) == 0 or len(order_items) == 0:
        logger.warning("Empty orders or order_items provided")
        return pd.DataFrame(columns=['item_id', 'date', 'demand', 'revenue'])
    
    # Check for required columns
    if 'id' not in orders.columns or 'order_id' not in order_items.columns:
        logger.error("Missing required columns for merge")
        return pd.DataFrame(columns=['item_id', 'date', 'demand', 'revenue'])
    
    if 'item_id' not in order_items.columns:
        logger.error("'item_id' column not found in order_items")
        return pd.DataFrame(columns=['item_id', 'date', 'demand', 'revenue'])
    
    # For large datasets, avoid the expensive merge approach
    # Instead, use the 'created' column directly from order_items if available
    if len(order_items) > 500000:
        logger.info(f"Using direct aggregation for large dataset ({len(order_items):,} rows)")
        
        # Use the created column from order_items directly
        if 'created' in order_items.columns:
            # Create a working copy with only needed columns
            work_df = order_items[['item_id', 'created', 'quantity']].copy()
            
            # Add price if available
            if 'price' in order_items.columns:
                work_df['price'] = order_items['price']
            
            # Convert created to date
            if work_df['created'].dtype in ['int64', 'float64']:
                # UNIX timestamp
                work_df['date'] = pd.to_datetime(work_df['created'], unit='s', errors='coerce').dt.date
            else:
                work_df['date'] = pd.to_datetime(work_df['created'], errors='coerce').dt.date
            
            # Drop rows with invalid dates
            work_df = work_df.dropna(subset=['date'])
            
            # Calculate line total if price available
            if 'price' in work_df.columns:
                work_df['line_total'] = work_df['quantity'].fillna(0) * work_df['price'].fillna(0)
                agg_dict = {'quantity': 'sum', 'line_total': 'sum'}
            else:
                agg_dict = {'quantity': 'sum'}
            
            # Aggregate by item and date
            daily_demand = work_df.groupby(['item_id', 'date']).agg(agg_dict).reset_index()
            
            # Rename columns
            daily_demand = daily_demand.rename(columns={
                'quantity': 'demand',
                'line_total': 'revenue'
            })
            
            # Ensure revenue column exists
            if 'revenue' not in daily_demand.columns:
                daily_demand['revenue'] = 0
            
            # Sort by item and date
            daily_demand = daily_demand.sort_values(['item_id', 'date']).reset_index(drop=True)
            
            logger.info(
                f"Generated daily demand: {len(daily_demand):,} rows, "
                f"{daily_demand['item_id'].nunique()} items"
            )
            
            return daily_demand
    
    # Standard approach for smaller datasets
    merged = order_items.merge(
        orders[['id', date_column]],
        left_on='order_id',
        right_on='id',
        how='left',
        suffixes=('', '_order')
    )
    
    # Extract date
    if date_column not in merged.columns:
        logger.error(f"Date column '{date_column}' not found after merge")
        return pd.DataFrame(columns=['item_id', 'date', 'demand', 'revenue'])
    
    merged['date'] = pd.to_datetime(merged[date_column]).dt.date
    
    # Aggregate by item and date
    agg_dict = {
        'quantity': 'sum'
    }
    
    # Include revenue if price column exists
    if 'price' in merged.columns:
        merged['line_total'] = merged['quantity'] * merged['price']
        agg_dict['line_total'] = 'sum'
    
    daily_demand = merged.groupby(['item_id', 'date']).agg(agg_dict).reset_index()
    
    # Rename columns
    daily_demand = daily_demand.rename(columns={
        'quantity': 'demand',
        'line_total': 'revenue'
    })
    
    # Fill missing dates with zero demand
    daily_demand = _fill_missing_dates(daily_demand)
    
    # Sort by item and date
    daily_demand = daily_demand.sort_values(['item_id', 'date']).reset_index(drop=True)
    
    logger.info(
        f"Generated daily demand: {len(daily_demand):,} rows, "
        f"{daily_demand['item_id'].nunique()} items"
    )
    
    return daily_demand


def _fill_missing_dates(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing dates with zero demand for each item.
    
    This ensures continuous time series for forecasting.
    
    Why fill with zeros?
    - Forecasting models expect continuous time series
    - Missing dates = no sales = zero demand
    - This is a reasonable assumption for retail
    
    Note: For very large datasets with many items over long periods,
    filling all missing dates can create billions of rows. In such cases,
    we skip filling to avoid memory issues and let the forecaster handle
    sparse data.
    """
    if len(demand_df) == 0:
        return demand_df
    
    # Get date range
    min_date = demand_df['date'].min()
    max_date = demand_df['date'].max()
    
    # Calculate potential size
    date_range_days = (pd.Timestamp(max_date) - pd.Timestamp(min_date)).days + 1
    num_items = demand_df['item_id'].nunique()
    potential_rows = date_range_days * num_items
    
    # Skip filling if it would create too many rows (>5M rows)
    MAX_ROWS = 5_000_000
    if potential_rows > MAX_ROWS:
        logger.warning(
            f"Skipping date filling: would create {potential_rows:,} rows "
            f"({num_items:,} items × {date_range_days:,} days). "
            f"Max allowed: {MAX_ROWS:,}. Forecaster will handle sparse data."
        )
        return demand_df
    
    # Create complete date range
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D').date
    
    # Get all items
    all_items = demand_df['item_id'].unique()
    
    # Create complete index
    complete_index = pd.MultiIndex.from_product(
        [all_items, all_dates],
        names=['item_id', 'date']
    )
    
    # Reindex and fill missing with 0
    demand_indexed = demand_df.set_index(['item_id', 'date'])
    demand_complete = demand_indexed.reindex(complete_index, fill_value=0)
    demand_complete = demand_complete.reset_index()
    
    # Fill NaN in revenue column if it exists
    if 'revenue' in demand_complete.columns:
        demand_complete['revenue'] = demand_complete['revenue'].fillna(0)
    
    filled_count = len(demand_complete) - len(demand_df)
    if filled_count > 0:
        logger.info(f"Filled {filled_count:,} missing date-item combinations with zero demand")
    
    return demand_complete
