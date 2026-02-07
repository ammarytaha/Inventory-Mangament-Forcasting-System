"""
Data Cleaning Pipeline
======================
Reusable pipeline for cleaning and preparing data for analytics.
Handles missing values, standardizes dates/IDs, removes outliers, produces clean tables.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataCleaningPipeline:
    """Reusable data cleaning pipeline."""
    
    def __init__(self, data_dir: str):
        """
        Initialize cleaning pipeline.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.raw_data = {}
        self.cleaned_data = {}
        self.cleaning_log = []
        
    def load_raw_data(self, file_name: str) -> pd.DataFrame:
        """Load raw CSV file."""
        file_path = self.data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path, low_memory=False)
        self.raw_data[file_name] = df.copy()
        return df
    
    def standardize_timestamps(self, df: pd.DataFrame, timestamp_cols: List[str]) -> pd.DataFrame:
        """Convert Unix timestamps to datetime objects."""
        df_clean = df.copy()
        
        for col in timestamp_cols:
            if col in df_clean.columns:
                # Check if values are Unix timestamps
                sample = df_clean[col].dropna()
                if len(sample) > 0:
                    if sample.dtype in ['int64', 'float64']:
                        min_val = sample.min()
                        max_val = sample.max()
                        # Unix timestamp range check
                        if min_val > 1000000000 and max_val < 2147483647:
                            df_clean[col] = pd.to_datetime(df_clean[col], unit='s', errors='coerce')
                            self.cleaning_log.append(f"Converted {col} from Unix timestamp to datetime")
        
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, table_name: str, strategy: str = 'smart') -> pd.DataFrame:
        """
        Handle missing values based on column type and context.
        
        Strategies:
        - 'smart': Use context-aware imputation
        - 'drop': Drop rows/columns with missing values
        - 'zero': Fill with 0 for numeric, empty string for text
        - 'forward_fill': Forward fill for time series
        """
        df_clean = df.copy()
        missing_before = df_clean.isnull().sum().sum()
        
        if strategy == 'smart':
            # ID columns: cannot impute, flag but keep
            id_cols = [col for col in df_clean.columns if col.endswith('_id') or col == 'id']
            for col in id_cols:
                if df_clean[col].isnull().sum() > 0:
                    self.cleaning_log.append(f"WARNING: {col} has {df_clean[col].isnull().sum()} missing values (cannot impute IDs)")
            
            # Amount/quantity columns: fill with 0 if reasonable
            amount_cols = [col for col in df_clean.columns 
                          if any(x in col.lower() for x in ['amount', 'quantity', 'price', 'cost', 'points'])]
            for col in amount_cols:
                if df_clean[col].dtype in ['int64', 'float64']:
                    missing_count = df_clean[col].isnull().sum()
                    if missing_count > 0:
                        df_clean[col] = df_clean[col].fillna(0)
                        self.cleaning_log.append(f"Filled {missing_count} missing values in {col} with 0")
            
            # Status/categorical: fill with 'Unknown' or most common
            cat_cols = [col for col in df_clean.columns 
                       if any(x in col.lower() for x in ['status', 'type', 'channel', 'method'])]
            for col in cat_cols:
                if df_clean[col].dtype == 'object':
                    missing_count = df_clean[col].isnull().sum()
                    if missing_count > 0:
                        mode_value = df_clean[col].mode()
                        fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                        df_clean[col] = df_clean[col].fillna(fill_value)
                        self.cleaning_log.append(f"Filled {missing_count} missing values in {col} with '{fill_value}'")
            
            # Text columns: fill with empty string
            text_cols = [col for col in df_clean.columns 
                        if col not in id_cols + amount_cols + cat_cols and df_clean[col].dtype == 'object']
            for col in text_cols:
                missing_count = df_clean[col].isnull().sum()
                if missing_count > 0:
                    df_clean[col] = df_clean[col].fillna('')
        
        missing_after = df_clean.isnull().sum().sum()
        if missing_before > missing_after:
            self.cleaning_log.append(f"Reduced missing values from {missing_before} to {missing_after}")
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, table_name: str, key_col: str = 'id') -> pd.DataFrame:
        """Remove duplicate records based on primary key."""
        df_clean = df.copy()
        
        if key_col in df_clean.columns:
            duplicates = df_clean[key_col].duplicated().sum()
            if duplicates > 0:
                df_clean = df_clean.drop_duplicates(subset=[key_col], keep='first')
                self.cleaning_log.append(f"Removed {duplicates} duplicate records based on {key_col}")
        
        return df_clean
    
    def handle_negative_values(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Handle negative values in quantity/amount columns."""
        df_clean = df.copy()
        
        # For inventory/SKU tables, negative quantities might be valid (backorders)
        # For order tables, negative amounts might indicate refunds
        # We'll flag them but keep them unless they're clearly errors
        
        amount_cols = [col for col in df_clean.columns 
                      if any(x in col.lower() for x in ['quantity', 'amount', 'price', 'cost'])]
        
        for col in amount_cols:
            if df_clean[col].dtype in ['int64', 'float64']:
                negative_count = (df_clean[col] < 0).sum()
                if negative_count > 0:
                    # For dim_skus, negative quantities might be data entry errors
                    if 'dim_skus' in table_name and 'quantity' in col.lower():
                        # Set negative quantities to 0 (likely data entry error)
                        df_clean.loc[df_clean[col] < 0, col] = 0
                        self.cleaning_log.append(f"Fixed {negative_count} negative values in {col} (set to 0)")
                    else:
                        # Flag but keep (might be refunds, adjustments, etc.)
                        self.cleaning_log.append(f"WARNING: {negative_count} negative values in {col} (kept as-is)")
        
        return df_clean
    
    def standardize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure ID columns are integers (remove decimals, handle NaN)."""
        df_clean = df.copy()
        
        id_cols = [col for col in df_clean.columns if col.endswith('_id') or col == 'id']
        
        for col in id_cols:
            if df_clean[col].dtype == 'float64':
                # Convert to Int64 (nullable integer)
                df_clean[col] = df_clean[col].astype('Int64')
                self.cleaning_log.append(f"Standardized {col} to integer type")
        
        return df_clean
    
    def flag_outliers(self, df: pd.DataFrame, table_name: str, method: str = 'iqr') -> pd.DataFrame:
        """
        Flag outliers without removing them (add outlier_flag column).
        Useful for analysis but preserving data integrity.
        """
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_clean[col].notna().sum() > 10:  # Need enough data
                flag_col = f"{col}_outlier_flag"
                
                if method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower_bound = Q1 - 3 * IQR
                        upper_bound = Q3 + 3 * IQR
                        df_clean[flag_col] = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))
                        
                        outlier_count = df_clean[flag_col].sum()
                        if outlier_count > 0:
                            self.cleaning_log.append(f"Flagged {outlier_count} outliers in {col}")
        
        return df_clean
    
    def clean_table(self, file_name: str, 
                   timestamp_cols: Optional[List[str]] = None,
                   missing_strategy: str = 'smart',
                   remove_duplicates: bool = True,
                   handle_negatives: bool = True,
                   standardize_ids: bool = True,
                   flag_outliers: bool = True) -> pd.DataFrame:
        """
        Complete cleaning pipeline for a single table.
        
        Parameters:
        -----------
        file_name : str
            Name of CSV file to clean
        timestamp_cols : list, optional
            List of timestamp column names. If None, auto-detect.
        missing_strategy : str
            Strategy for handling missing values
        remove_duplicates : bool
            Whether to remove duplicate records
        handle_negatives : bool
            Whether to handle negative values
        standardize_ids : bool
            Whether to standardize ID columns
        flag_outliers : bool
            Whether to flag outliers
        """
        self.cleaning_log = []  # Reset log for this table
        
        # Load raw data
        df = self.load_raw_data(file_name)
        self.cleaning_log.append(f"Loaded {len(df)} rows from {file_name}")
        
        # Auto-detect timestamp columns if not provided
        if timestamp_cols is None:
            timestamp_cols = [col for col in df.columns 
                            if any(x in col.lower() for x in ['created', 'updated', 'time', 'date'])]
        
        # Apply cleaning steps
        df_clean = df.copy()
        
        # 1. Standardize timestamps
        if timestamp_cols:
            df_clean = self.standardize_timestamps(df_clean, timestamp_cols)
        
        # 2. Handle missing values
        df_clean = self.handle_missing_values(df_clean, file_name, strategy=missing_strategy)
        
        # 3. Remove duplicates
        if remove_duplicates:
            df_clean = self.remove_duplicates(df_clean, file_name)
        
        # 4. Handle negative values
        if handle_negatives:
            df_clean = self.handle_negative_values(df_clean, file_name)
        
        # 5. Standardize IDs
        if standardize_ids:
            df_clean = self.standardize_ids(df_clean)
        
        # 6. Flag outliers
        if flag_outliers:
            df_clean = self.flag_outliers(df_clean, file_name)
        
        # Store cleaned data
        self.cleaned_data[file_name] = df_clean
        
        print(f"✓ Cleaned {file_name}: {len(df)} -> {len(df_clean)} rows")
        
        return df_clean
    
    def clean_all_tables(self, file_list: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Clean all tables or specified list of tables."""
        if file_list is None:
            # Get all CSV files
            csv_files = list(self.data_dir.glob("*.csv"))
            file_list = [f.name for f in csv_files]
        
        cleaned = {}
        for file_name in file_list:
            try:
                df_clean = self.clean_table(file_name)
                cleaned[file_name] = df_clean
            except Exception as e:
                print(f"⚠ Error cleaning {file_name}: {str(e)}")
        
        return cleaned
    
    def save_cleaned_data(self, output_dir: str, prefix: str = 'cleaned_'):
        """Save cleaned data to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for file_name, df in self.cleaned_data.items():
            output_file = output_path / f"{prefix}{file_name}"
            df.to_csv(output_file, index=False)
            print(f"✓ Saved cleaned data to {output_file}")
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of cleaning operations."""
        return {
            'tables_cleaned': list(self.cleaned_data.keys()),
            'cleaning_log': self.cleaning_log,
            'summary': {
                'total_tables': len(self.cleaned_data),
                'total_rows_before': sum(len(df) for df in self.raw_data.values()),
                'total_rows_after': sum(len(df) for df in self.cleaned_data.values())
            }
        }


if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).parent
    pipeline = DataCleaningPipeline(data_dir)
    
    # Clean key tables
    key_tables = ['fct_orders.csv', 'fct_order_items.csv', 'dim_skus.csv', 'dim_menu_items.csv']
    cleaned = pipeline.clean_all_tables(key_tables)
    
    print("\nCleaning Summary:")
    summary = pipeline.get_cleaning_summary()
    print(f"Tables cleaned: {summary['summary']['total_tables']}")
