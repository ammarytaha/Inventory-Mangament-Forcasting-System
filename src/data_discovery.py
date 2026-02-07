"""
Data Discovery Module
======================
Automatically loads all CSV files and performs comprehensive data profiling.
Identifies fact tables, dimension tables, and data quality issues.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataDiscovery:
    """Comprehensive data discovery and profiling engine."""
    
    def __init__(self, data_dir: str):
        """
        Initialize data discovery engine.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.files = {}
        self.profiles = {}
        self.fact_tables = []
        self.dimension_tables = []
        
    def discover_files(self) -> Dict[str, str]:
        """Discover all CSV files in the directory."""
        csv_files = list(self.data_dir.glob("*.csv"))
        file_dict = {}
        
        for file_path in csv_files:
            file_name = file_path.name
            file_dict[file_name] = str(file_path)
            
        self.files = file_dict
        print(f"✓ Discovered {len(file_dict)} CSV files")
        return file_dict
    
    def classify_table_type(self, file_name: str) -> str:
        """Classify table as fact or dimension based on naming convention."""
        if file_name.startswith("fct_"):
            return "fact"
        elif file_name.startswith("dim_"):
            return "dimension"
        else:
            return "other"
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with error handling."""
        try:
            df = pd.read_csv(file_path, low_memory=False)
            return df
        except Exception as e:
            print(f"⚠ Error loading {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect and categorize data types."""
        type_mapping = {}
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            
            # Check for timestamp columns (Unix timestamps)
            if 'created' in col.lower() or 'updated' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                if df[col].dtype in ['int64', 'float64']:
                    # Check if values look like Unix timestamps
                    sample = df[col].dropna().head(100)
                    if len(sample) > 0:
                        if sample.min() > 1000000000 and sample.max() < 2147483647:
                            type_mapping[col] = "timestamp"
                            continue
            
            # Check for ID columns
            if col.endswith('_id') or col == 'id':
                type_mapping[col] = "id"
            # Check for amount/price columns
            elif any(x in col.lower() for x in ['amount', 'price', 'cost', 'quantity', 'points']):
                type_mapping[col] = "numeric"
            # Check for status/enum columns
            elif any(x in col.lower() for x in ['status', 'type', 'channel', 'method']):
                type_mapping[col] = "categorical"
            # Default based on pandas dtype
            else:
                if dtype.startswith('int'):
                    type_mapping[col] = "integer"
                elif dtype.startswith('float'):
                    type_mapping[col] = "float"
                elif dtype == 'object':
                    type_mapping[col] = "string"
                else:
                    type_mapping[col] = dtype
        
        return type_mapping
    
    def detect_data_quality_issues(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Detect various data quality issues."""
        issues = {
            'negative_amounts': [],
            'invalid_dates': [],
            'missing_keys': [],
            'duplicate_keys': [],
            'outliers': [],
            'inconsistent_values': []
        }
        
        # Check for negative quantities/amounts
        amount_cols = [col for col in df.columns if any(x in col.lower() for x in ['quantity', 'amount', 'price', 'cost'])]
        for col in amount_cols:
            if df[col].dtype in ['int64', 'float64']:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues['negative_amounts'].append({
                        'column': col,
                        'count': int(negative_count),
                        'min_value': float(df[col].min()),
                        'sample_values': df[df[col] < 0][col].head(5).tolist()
                    })
        
        # Check for duplicate primary keys
        if 'id' in df.columns:
            duplicate_ids = df['id'].duplicated().sum()
            if duplicate_ids > 0:
                issues['duplicate_keys'].append({
                    'column': 'id',
                    'count': int(duplicate_ids)
                })
        
        # Check for missing foreign keys in fact tables
        if table_name.startswith('fct_'):
            fk_cols = [col for col in df.columns if col.endswith('_id') and col != 'id']
            for col in fk_cols:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    issues['missing_keys'].append({
                        'column': col,
                        'count': int(missing_count),
                        'pct': round(100 * missing_count / len(df), 2)
                    })
        
        # Detect outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].notna().sum() > 10:  # Need enough data
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers > 0:
                        issues['outliers'].append({
                            'column': col,
                            'count': int(outliers),
                            'pct': round(100 * outliers / len(df), 2)
                        })
        
        return issues
    
    def profile_table(self, file_name: str, file_path: str) -> Dict[str, Any]:
        """Create comprehensive profile for a single table."""
        print(f"\n📊 Profiling {file_name}...")
        
        df = self.load_file(file_path)
        
        if df.empty:
            return {
                'file_name': file_name,
                'status': 'error',
                'error': 'Failed to load file'
            }
        
        table_type = self.classify_table_type(file_name)
        
        profile = {
            'file_name': file_name,
            'table_type': table_type,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'data_types': self.detect_data_types(df),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_pct': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_pct': round(100 * df.duplicated().sum() / len(df), 2),
            'data_quality_issues': self.detect_data_quality_issues(df, file_name),
            'sample_data': df.head(3).to_dict('records') if len(df) > 0 else [],
            'date_range': self._get_date_range(df),
            'numeric_summary': self._get_numeric_summary(df)
        }
        
        # Store classification
        if table_type == 'fact':
            self.fact_tables.append(file_name)
        elif table_type == 'dimension':
            self.dimension_tables.append(file_name)
        
        return profile
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract date range from timestamp columns."""
        date_info = {}
        timestamp_cols = [col for col in df.columns if 'created' in col.lower() or 'updated' in col.lower()]
        
        for col in timestamp_cols:
            if df[col].dtype in ['int64', 'float64']:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    try:
                        min_ts = int(non_null.min())
                        max_ts = int(non_null.max())
                        if min_ts > 1000000000:  # Looks like Unix timestamp
                            date_info[col] = {
                                'min': datetime.fromtimestamp(min_ts).strftime('%Y-%m-%d'),
                                'max': datetime.fromtimestamp(max_ts).strftime('%Y-%m-%d'),
                                'min_timestamp': min_ts,
                                'max_timestamp': max_ts
                            }
                    except:
                        pass
        
        return date_info
    
    def _get_numeric_summary(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Get summary statistics for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = {}
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                summary[col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'non_null_count': int(df[col].notna().sum())
                }
        
        return summary
    
    def discover_all(self) -> Dict[str, Any]:
        """Run complete discovery process."""
        print("=" * 80)
        print("DATA DISCOVERY ENGINE")
        print("=" * 80)
        
        # Discover files
        self.discover_files()
        
        # Profile each file
        for file_name, file_path in self.files.items():
            profile = self.profile_table(file_name, file_path)
            self.profiles[file_name] = profile
        
        # Generate summary
        summary = self._generate_summary()
        
        return {
            'profiles': self.profiles,
            'summary': summary,
            'fact_tables': self.fact_tables,
            'dimension_tables': self.dimension_tables
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall summary statistics."""
        total_rows = sum(p['row_count'] for p in self.profiles.values() if 'row_count' in p)
        total_columns = sum(p['column_count'] for p in self.profiles.values() if 'column_count' in p)
        
        tables_with_issues = []
        for file_name, profile in self.profiles.items():
            issues = profile.get('data_quality_issues', {})
            has_issues = any(len(v) > 0 for v in issues.values())
            if has_issues:
                tables_with_issues.append(file_name)
        
        return {
            'total_files': len(self.profiles),
            'fact_tables': len(self.fact_tables),
            'dimension_tables': len(self.dimension_tables),
            'total_rows': total_rows,
            'total_columns': total_columns,
            'tables_with_issues': tables_with_issues,
            'tables_with_issues_count': len(tables_with_issues)
        }
    
    def print_summary_report(self):
        """Print human-readable summary report."""
        print("\n" + "=" * 80)
        print("DATA DISCOVERY SUMMARY REPORT")
        print("=" * 80)
        
        print(f"\n📁 Files Discovered: {self.summary['total_files']}")
        print(f"📊 Fact Tables: {self.summary['fact_tables']}")
        print(f"📋 Dimension Tables: {self.summary['dimension_tables']}")
        print(f"📈 Total Rows: {self.summary['total_rows']:,}")
        print(f"📑 Total Columns: {self.summary['total_columns']}")
        
        print("\n" + "-" * 80)
        print("TABLE DETAILS")
        print("-" * 80)
        
        for file_name, profile in sorted(self.profiles.items()):
            if 'status' in profile and profile['status'] == 'error':
                print(f"\n❌ {file_name}: ERROR - {profile.get('error', 'Unknown error')}")
                continue
            
            table_type_icon = "📊" if profile['table_type'] == 'fact' else "📋"
            print(f"\n{table_type_icon} {file_name}")
            print(f"   Type: {profile['table_type'].upper()}")
            print(f"   Rows: {profile['row_count']:,}")
            print(f"   Columns: {profile['column_count']}")
            
            # Show missing values
            missing_cols = {k: v for k, v in profile['missing_pct'].items() if v > 0}
            if missing_cols:
                print(f"   ⚠ Missing Values: {len(missing_cols)} columns")
                for col, pct in list(missing_cols.items())[:3]:
                    print(f"      - {col}: {pct}%")
            
            # Show data quality issues
            issues = profile['data_quality_issues']
            issue_count = sum(len(v) for v in issues.values())
            if issue_count > 0:
                print(f"   ⚠ Data Quality Issues: {issue_count}")
                if issues['negative_amounts']:
                    print(f"      - Negative amounts: {len(issues['negative_amounts'])} columns")
                if issues['duplicate_keys']:
                    print(f"      - Duplicate keys: {len(issues['duplicate_keys'])}")
                if issues['missing_keys']:
                    print(f"      - Missing foreign keys: {len(issues['missing_keys'])}")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).parent
    discovery = DataDiscovery(data_dir)
    results = discovery.discover_all()
    discovery.summary = results['summary']
    discovery.print_summary_report()
