"""
Data Validation Utilities
==========================
Schema validation, data quality checks, and input verification.

Design Principles:
- Never silently fail - always log issues
- Return structured validation results
- Support partial validation (warn but continue)
- Provide actionable error messages
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from utils.logger import get_logger
from utils.constants import DATA_QUALITY_THRESHOLDS

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """
    Structured result of a validation operation.
    
    Attributes
    ----------
    is_valid : bool
        Overall validation status
    errors : List[str]
        Critical issues that prevent processing
    warnings : List[str]
        Non-critical issues to be aware of
    info : Dict[str, Any]
        Additional validation metadata
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info
        }


class SchemaValidator:
    """
    Validates DataFrames against defined schemas.
    
    Usage
    -----
    validator = SchemaValidator()
    result = validator.validate(df, SALES_SCHEMA, "fct_orders.csv")
    
    if not result.is_valid:
        print(f"Validation failed: {result.errors}")
    """
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize validator with optional custom thresholds.
        
        Parameters
        ----------
        thresholds : dict, optional
            Custom thresholds for validation. Uses defaults if not provided.
        """
        self.thresholds = thresholds or DATA_QUALITY_THRESHOLDS
    
    def validate(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        file_name: str
    ) -> ValidationResult:
        """
        Validate a DataFrame against a schema.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate
        schema : dict
            Schema definition with required/optional columns
        file_name : str
            Name of the file being validated (for error messages)
        
        Returns
        -------
        ValidationResult
            Structured validation result with errors/warnings
        """
        result = ValidationResult()
        result.info["file_name"] = file_name
        result.info["row_count"] = len(df)
        result.info["column_count"] = len(df.columns)
        
        # Check required columns
        if file_name in schema.get("required_columns", {}):
            required = schema["required_columns"][file_name]
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                result.add_error(
                    f"Missing required columns in {file_name}: {missing}"
                )
            else:
                result.info["required_columns_present"] = True
        
        # Check optional columns (just log which are present)
        if file_name in schema.get("optional_columns", {}):
            optional = schema["optional_columns"][file_name]
            present = [col for col in optional if col in df.columns]
            result.info["optional_columns_present"] = present
        
        # Validate timestamps if present
        if "timestamp_columns" in schema:
            for col in schema["timestamp_columns"]:
                if col in df.columns:
                    ts_result = self._validate_timestamps(df, col)
                    result.warnings.extend(ts_result.warnings)
                    result.info[f"{col}_date_range"] = ts_result.info.get("date_range")
        
        # Validate numeric columns if present
        if "numeric_columns" in schema:
            for col in schema["numeric_columns"]:
                if col in df.columns:
                    num_result = self._validate_numeric(df, col)
                    result.warnings.extend(num_result.warnings)
        
        # Check for missing values
        missing_result = self._validate_missing_values(df, file_name)
        result.warnings.extend(missing_result.warnings)
        result.info["missing_value_summary"] = missing_result.info.get("summary")
        
        # Check for duplicates
        dup_result = self._validate_duplicates(df, file_name)
        result.warnings.extend(dup_result.warnings)
        result.info["duplicate_count"] = dup_result.info.get("duplicate_count", 0)
        
        # Log validation result
        if result.is_valid:
            logger.info(f"Validation PASSED for {file_name}")
        else:
            logger.error(f"Validation FAILED for {file_name}: {result.errors}")
        
        for warning in result.warnings:
            logger.warning(warning)
        
        return result
    
    def _validate_timestamps(
        self,
        df: pd.DataFrame,
        column: str
    ) -> ValidationResult:
        """Validate a timestamp column."""
        result = ValidationResult()
        
        try:
            # Check if values are Unix timestamps
            sample = df[column].dropna()
            if len(sample) == 0:
                result.add_warning(f"Timestamp column '{column}' is empty")
                return result
            
            # Determine if Unix timestamp or datetime string
            if sample.dtype in ['int64', 'float64']:
                min_val = sample.min()
                max_val = sample.max()
                
                # Unix timestamp range (roughly 2020-2030)
                if min_val > 1577836800 and max_val < 1893456000:
                    # Convert to datetime for range check
                    min_date = datetime.fromtimestamp(min_val)
                    max_date = datetime.fromtimestamp(max_val)
                    result.info["date_range"] = {
                        "min": min_date.isoformat(),
                        "max": max_date.isoformat()
                    }
                    result.info["is_unix_timestamp"] = True
                else:
                    result.add_warning(
                        f"Column '{column}' has unusual numeric values for timestamps"
                    )
            else:
                # Try parsing as datetime string
                parsed = pd.to_datetime(sample, errors='coerce')
                valid_count = parsed.notna().sum()
                
                if valid_count < len(sample):
                    invalid_pct = (len(sample) - valid_count) / len(sample) * 100
                    result.add_warning(
                        f"Column '{column}' has {invalid_pct:.1f}% unparseable dates"
                    )
                
                if valid_count > 0:
                    result.info["date_range"] = {
                        "min": str(parsed.min()),
                        "max": str(parsed.max())
                    }
        except Exception as e:
            result.add_warning(f"Error validating timestamp column '{column}': {e}")
        
        return result
    
    def _validate_numeric(
        self,
        df: pd.DataFrame,
        column: str
    ) -> ValidationResult:
        """Validate a numeric column."""
        result = ValidationResult()
        
        try:
            if df[column].dtype not in ['int64', 'float64', 'Int64', 'Float64']:
                result.add_warning(
                    f"Column '{column}' expected to be numeric but is {df[column].dtype}"
                )
                return result
            
            sample = df[column].dropna()
            if len(sample) == 0:
                result.add_warning(f"Numeric column '{column}' is empty")
                return result
            
            # Check for negative values in typically positive columns
            negative_cols = ['quantity', 'price', 'cost', 'stock']
            if any(keyword in column.lower() for keyword in negative_cols):
                neg_count = (sample < 0).sum()
                if neg_count > 0:
                    neg_pct = neg_count / len(sample) * 100
                    result.add_warning(
                        f"Column '{column}' has {neg_count} negative values ({neg_pct:.1f}%)"
                    )
            
            # Check for extreme outliers
            q1, q3 = sample.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                extreme_low = sample < (q1 - 3 * iqr)
                extreme_high = sample > (q3 + 3 * iqr)
                outlier_count = extreme_low.sum() + extreme_high.sum()
                
                if outlier_count > 0:
                    outlier_pct = outlier_count / len(sample) * 100
                    result.add_warning(
                        f"Column '{column}' has {outlier_count} extreme outliers ({outlier_pct:.1f}%)"
                    )
            
            result.info["stats"] = {
                "min": float(sample.min()),
                "max": float(sample.max()),
                "mean": float(sample.mean()),
                "median": float(sample.median())
            }
        except Exception as e:
            result.add_warning(f"Error validating numeric column '{column}': {e}")
        
        return result
    
    def _validate_missing_values(
        self,
        df: pd.DataFrame,
        file_name: str
    ) -> ValidationResult:
        """Check for missing values."""
        result = ValidationResult()
        
        missing = df.isnull().sum()
        missing_pct = missing / len(df)
        
        summary = {}
        for col in df.columns:
            if missing[col] > 0:
                summary[col] = {
                    "count": int(missing[col]),
                    "percentage": float(missing_pct[col] * 100)
                }
                
                # Check thresholds
                if missing_pct[col] > self.thresholds["missing_critical_pct"]:
                    result.add_warning(
                        f"CRITICAL: Column '{col}' in {file_name} has "
                        f"{missing_pct[col]*100:.1f}% missing values"
                    )
                elif missing_pct[col] > self.thresholds["missing_warning_pct"]:
                    result.add_warning(
                        f"Column '{col}' in {file_name} has "
                        f"{missing_pct[col]*100:.1f}% missing values"
                    )
        
        result.info["summary"] = summary
        return result
    
    def _validate_duplicates(
        self,
        df: pd.DataFrame,
        file_name: str
    ) -> ValidationResult:
        """Check for duplicate records."""
        result = ValidationResult()
        
        # Check for complete duplicates
        dup_count = df.duplicated().sum()
        result.info["duplicate_count"] = int(dup_count)
        
        if dup_count > self.thresholds["duplicate_critical_count"]:
            result.add_warning(
                f"CRITICAL: {file_name} has {dup_count} duplicate rows"
            )
        elif dup_count > self.thresholds["duplicate_warning_count"]:
            result.add_warning(
                f"{file_name} has {dup_count} duplicate rows"
            )
        
        # Check for duplicate IDs if 'id' column exists
        if 'id' in df.columns:
            id_dup_count = df['id'].duplicated().sum()
            if id_dup_count > 0:
                result.add_warning(
                    f"{file_name} has {id_dup_count} duplicate IDs"
                )
        
        return result


def validate_file_exists(file_path: str) -> bool:
    """
    Check if a file exists and log appropriately.
    
    Parameters
    ----------
    file_path : str
        Path to the file to check
    
    Returns
    -------
    bool
        True if file exists, False otherwise
    """
    path = Path(file_path)
    if path.exists():
        logger.info(f"File found: {file_path}")
        return True
    else:
        logger.error(f"File not found: {file_path}")
        return False


def validate_dataframe_not_empty(df: pd.DataFrame, name: str) -> bool:
    """
    Check if a DataFrame is not empty.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check
    name : str
        Name for logging purposes
    
    Returns
    -------
    bool
        True if DataFrame has data, False if empty
    """
    if df is None or len(df) == 0:
        logger.warning(f"DataFrame '{name}' is empty")
        return False
    
    logger.info(f"DataFrame '{name}' has {len(df):,} rows")
    return True
