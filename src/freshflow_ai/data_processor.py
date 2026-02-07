"""
FreshFlow AI - Data Processor Module
=====================================

Handles all data loading, transformation, and preprocessing.
Works with both raw CSV files and pre-processed parquet files.

Features:
- Automatic data source detection (CSV/Parquet)
- Place-level filtering for personalization
- Weekly aggregation at place-item level
- Demand classification (SBC methodology)
- Feature engineering for forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging

from .config import Config, DEFAULT_CONFIG

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processing engine for FreshFlow AI.
    
    Handles all data operations including:
    - Loading raw or pre-processed data
    - Place-specific filtering
    - Weekly aggregation
    - Feature engineering
    - Demand classification
    
    Usage:
        processor = DataProcessor(config)
        data = processor.load_data()
        place_data = processor.get_place_data(place_id=94025)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or DEFAULT_CONFIG
        self._orders_df = None
        self._order_items_df = None
        self._items_df = None
        self._places_df = None
        self._features_df = None
        self._demand_classification = None
        self._weekly_data = None
        
    def load_data(self, force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load all required data files.
        
        Automatically detects whether to use pre-processed parquet files
        or raw CSV files.
        
        Args:
            force_reload: If True, reload data even if already cached.
            
        Returns:
            Dictionary of DataFrames with keys:
            - 'orders', 'order_items', 'items', 'places', 
            - 'features', 'demand_classification'
        """
        if not force_reload and self._orders_df is not None:
            return self._get_cached_data()
        
        logger.info("Loading data files...")
        
        # Check for pre-processed parquet files first
        analysis_data_path = self.config.analysis_path / 'data'
        
        if (analysis_data_path / 'features_place_item_week.parquet').exists():
            self._load_preprocessed_data(analysis_data_path)
        else:
            self._load_raw_data()
            
        return self._get_cached_data()
    
    def _load_preprocessed_data(self, analysis_data_path: Path):
        """Load pre-processed parquet files from Data Analysis folder"""
        logger.info("Loading pre-processed parquet files...")
        
        try:
            # Load features dataset (main forecasting dataset)
            features_path = analysis_data_path / 'features_place_item_week.parquet'
            if features_path.exists():
                self._features_df = pd.read_parquet(features_path)
                logger.info(f"Loaded features: {len(self._features_df):,} rows")
                
            # Load demand classification
            classification_path = analysis_data_path / 'demand_classification.csv'
            if classification_path.exists():
                self._demand_classification = pd.read_csv(classification_path)
                logger.info(f"Loaded demand classification: {len(self._demand_classification):,} rows")
                
            # Load dimension tables
            if (analysis_data_path / 'dim_places_clean.parquet').exists():
                self._places_df = pd.read_parquet(analysis_data_path / 'dim_places_clean.parquet')
            if (analysis_data_path / 'dim_items_clean.parquet').exists():
                self._items_df = pd.read_parquet(analysis_data_path / 'dim_items_clean.parquet')
                
        except Exception as e:
            logger.warning(f"Error loading parquet files: {e}. Falling back to raw CSV.")
            self._load_raw_data()
    
    def _load_raw_data(self):
        """Load and process raw CSV files"""
        logger.info("Loading raw CSV files...")
        
        data_path = self.config.data_path
        
        # Load orders
        orders_path = data_path / 'fct_orders.csv'
        if orders_path.exists():
            self._orders_df = pd.read_csv(orders_path, low_memory=False)
            self._clean_orders()
            logger.info(f"Loaded orders: {len(self._orders_df):,} rows")
            
        # Load order items
        order_items_path = data_path / 'fct_order_items.csv'
        if order_items_path.exists():
            self._order_items_df = pd.read_csv(order_items_path, low_memory=False)
            self._clean_order_items()
            logger.info(f"Loaded order items: {len(self._order_items_df):,} rows")
            
        # Load items
        items_path = data_path / 'dim_items.csv'
        if items_path.exists():
            self._items_df = pd.read_csv(items_path, low_memory=False)
            logger.info(f"Loaded items: {len(self._items_df):,} rows")
            
        # Load places
        places_path = data_path / 'dim_places.csv'
        if places_path.exists():
            self._places_df = pd.read_csv(places_path, low_memory=False)
            self._clean_places()
            logger.info(f"Loaded places: {len(self._places_df):,} rows")
        
        # Load menu items for product names
        menu_items_path = data_path / 'dim_menu_items.csv'
        if menu_items_path.exists():
            self._menu_items_df = pd.read_csv(menu_items_path, low_memory=False)
            logger.info(f"Loaded menu items: {len(self._menu_items_df):,} rows")
            
        # Load demand classification if exists
        classification_path = self.config.analysis_path / 'data' / 'demand_classification.csv'
        if classification_path.exists():
            self._demand_classification = pd.read_csv(classification_path)
            
        # Create weekly aggregates
        self._create_weekly_aggregates()
            
    def _clean_orders(self):
        """Clean orders dataframe"""
        df = self._orders_df
        
        # Convert timestamp
        df['created_dt'] = pd.to_datetime(df['created'], unit='s', utc=True)
        
        # Remove demo transactions
        if 'demo_mode' in df.columns:
            df = df[df['demo_mode'] != 1]
            
        # Remove cancelled orders
        if 'status' in df.columns:
            df = df[~df['status'].isin(['cancelled', 'Cancelled', 'CANCELLED'])]
            
        self._orders_df = df
        
    def _clean_order_items(self):
        """Clean order items dataframe"""
        df = self._order_items_df
        
        # Convert timestamp
        df['created_dt'] = pd.to_datetime(df['created'], unit='s', utc=True)
        
        # Remove negative quantities
        if 'quantity' in df.columns:
            df = df[df['quantity'] >= 0]
            
        # Remove orphan items
        df = df.dropna(subset=['order_id'])
        
        self._order_items_df = df
        
    def _clean_places(self):
        """Clean places dataframe and identify active/inactive"""
        df = self._places_df
        
        # Identify active places based on recent orders
        if self._orders_df is not None:
            cutoff_date = self._orders_df['created_dt'].max() - timedelta(days=90)
            recent_orders = self._orders_df[self._orders_df['created_dt'] >= cutoff_date]
            active_places = recent_orders['place_id'].unique()
            df['is_active'] = df['id'].isin(active_places)
        else:
            df['is_active'] = True
            
        self._places_df = df
        
    def _create_weekly_aggregates(self):
        """Create weekly demand aggregates at place-item level"""
        if self._order_items_df is None or self._orders_df is None:
            return
            
        logger.info("Creating weekly aggregates...")
        
        # Merge to get place_id
        df = self._order_items_df.merge(
            self._orders_df[['id', 'place_id']],
            left_on='order_id',
            right_on='id',
            suffixes=('', '_order')
        )
        
        # Create week column (Monday start)
        df['week_start'] = df['created_dt'].dt.to_period('W-SUN').dt.start_time
        
        # Aggregate
        weekly = df.groupby(['place_id', 'item_id', 'week_start']).agg({
            'quantity': 'sum',
            'price': 'mean',
            'id': 'count'
        }).reset_index()
        
        weekly.columns = ['place_id', 'item_id', 'week_start', 'demand', 'avg_price', 'transaction_count']
        
        self._weekly_data = weekly
        
    def _get_cached_data(self) -> Dict[str, pd.DataFrame]:
        """Return cached dataframes"""
        return {
            'orders': self._orders_df,
            'order_items': self._order_items_df,
            'items': self._items_df,
            'places': self._places_df,
            'features': self._features_df,
            'demand_classification': self._demand_classification,
            'weekly': self._weekly_data
        }
    
    def get_active_places(self) -> pd.DataFrame:
        """
        Get list of active places with basic stats.
        
        Returns:
            DataFrame with place_id, title, country, order_count, last_order
        """
        if self._places_df is None:
            self.load_data()
            
        places = self._places_df.copy()
        
        # Add order statistics
        if self._orders_df is not None:
            order_stats = self._orders_df.groupby('place_id').agg({
                'id': 'count',
                'created_dt': 'max'
            }).reset_index()
            order_stats.columns = ['place_id', 'order_count', 'last_order']
            places = places.merge(order_stats, left_on='id', right_on='place_id', how='left')
            
        # Filter to active places
        if 'is_active' in places.columns:
            places = places[places['is_active'] == True]
            
        return places.sort_values('order_count', ascending=False)
    
    def get_place_data(
        self, 
        place_id: int,
        include_features: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific place (location).
        
        This is the main method for location-based personalization.
        
        Args:
            place_id: The unique identifier of the location
            include_features: Whether to include feature-engineered data
            
        Returns:
            Dictionary containing:
            - place_info: Place metadata
            - weekly_demand: Weekly demand by item
            - items: Items available at this place
            - features: Feature-engineered data (if available)
            - demand_classification: Demand types for items
            - summary_stats: Summary statistics
        """
        self.load_data()
        
        result = {}
        
        # Place info
        if self._places_df is not None:
            result['place_info'] = self._places_df[
                self._places_df['id'] == place_id
            ].iloc[0].to_dict() if len(self._places_df[self._places_df['id'] == place_id]) > 0 else {}
            
        # Weekly demand data
        if self._weekly_data is not None:
            result['weekly_demand'] = self._weekly_data[
                self._weekly_data['place_id'] == place_id
            ].copy()
        elif self._features_df is not None:
            result['weekly_demand'] = self._features_df[
                self._features_df['place_id'] == place_id
            ].copy()
            
        # Items for this place
        if result.get('weekly_demand') is not None and self._items_df is not None:
            place_items = result['weekly_demand']['item_id'].unique()
            result['items'] = self._items_df[
                self._items_df['id'].isin(place_items)
            ].copy()
            
        # Demand classification for place items
        if self._demand_classification is not None:
            result['demand_classification'] = self._demand_classification[
                self._demand_classification['place_id'] == place_id
            ].copy()
            
        # Calculate summary stats
        result['summary_stats'] = self._calculate_place_stats(place_id, result)
        
        return result
    
    def _calculate_place_stats(
        self, 
        place_id: int, 
        place_data: Dict
    ) -> Dict:
        """Calculate summary statistics for a place"""
        stats = {
            'place_id': place_id,
            'total_items': 0,
            'total_demand': 0,
            'avg_weekly_demand': 0,
            'weeks_active': 0,
            'demand_type_distribution': {}
        }
        
        # Weekly demand stats
        weekly = place_data.get('weekly_demand')
        if weekly is not None and len(weekly) > 0:
            stats['total_items'] = weekly['item_id'].nunique()
            stats['total_demand'] = weekly['demand'].sum() if 'demand' in weekly.columns else 0
            stats['weeks_active'] = weekly['week_start'].nunique() if 'week_start' in weekly.columns else 0
            if stats['weeks_active'] > 0:
                stats['avg_weekly_demand'] = stats['total_demand'] / stats['weeks_active']
                
        # Demand type distribution
        classification = place_data.get('demand_classification')
        if classification is not None and len(classification) > 0:
            stats['demand_type_distribution'] = classification['demand_type'].value_counts().to_dict()
            
        return stats
    
    def get_item_history(
        self, 
        place_id: int, 
        item_id: int,
        weeks: int = 52
    ) -> pd.DataFrame:
        """
        Get historical demand data for a specific item at a place.
        
        Args:
            place_id: Location identifier
            item_id: Product identifier
            weeks: Number of weeks of history to retrieve
            
        Returns:
            DataFrame with week_start, demand, price columns
        """
        self.load_data()
        
        # Try features data first
        if self._features_df is not None:
            df = self._features_df[
                (self._features_df['place_id'] == place_id) &
                (self._features_df['item_id'] == item_id)
            ].copy()
            if len(df) > 0:
                return df.sort_values('week_start').tail(weeks)
                
        # Fall back to weekly data
        if self._weekly_data is not None:
            df = self._weekly_data[
                (self._weekly_data['place_id'] == place_id) &
                (self._weekly_data['item_id'] == item_id)
            ].copy()
            return df.sort_values('week_start').tail(weeks)
            
        return pd.DataFrame()
    
    def get_demand_type(self, place_id: int, item_id: int) -> str:
        """
        Get the demand classification type for a place-item combination.
        
        Returns one of: 'Smooth', 'Erratic', 'Intermittent', 'Lumpy', 'Insufficient Data'
        """
        if self._demand_classification is None:
            self.load_data()
            
        if self._demand_classification is not None:
            match = self._demand_classification[
                (self._demand_classification['place_id'] == place_id) &
                (self._demand_classification['item_id'] == item_id)
            ]
            if len(match) > 0:
                return match.iloc[0]['demand_type']
                
        return 'Insufficient Data'
    
    def classify_demand(
        self, 
        series: pd.Series,
        adi_threshold: float = 1.32,
        cv2_threshold: float = 0.49
    ) -> str:
        """
        Classify demand pattern using SBC (Syntetos-Boylan) methodology.
        
        Args:
            series: Demand time series
            adi_threshold: ADI threshold (default 1.32)
            cv2_threshold: CV² threshold (default 0.49)
            
        Returns:
            Classification: 'Smooth', 'Erratic', 'Intermittent', 'Lumpy', or 'Insufficient Data'
        """
        non_zero = series[series > 0]
        
        if len(non_zero) < 2:
            return 'Insufficient Data'
            
        # Calculate ADI (Average Demand Interval)
        demand_frequency = len(non_zero) / len(series)
        adi = 1 / demand_frequency if demand_frequency > 0 else float('inf')
        
        # Calculate CV² (Coefficient of Variation squared)
        cv = non_zero.std() / non_zero.mean() if non_zero.mean() > 0 else 0
        cv2 = cv ** 2
        
        # Classify
        if adi <= adi_threshold:
            return 'Smooth' if cv2 <= cv2_threshold else 'Erratic'
        else:
            return 'Intermittent' if cv2 <= cv2_threshold else 'Lumpy'
    
    def get_top_items(
        self, 
        place_id: int,
        n: int = 20,
        metric: str = 'demand'
    ) -> pd.DataFrame:
        """
        Get top N items for a place by specified metric.
        
        Args:
            place_id: Location identifier
            n: Number of items to return
            metric: Sorting metric ('demand', 'revenue', 'frequency')
            
        Returns:
            DataFrame with item details and ranks
        """
        place_data = self.get_place_data(place_id)
        weekly = place_data.get('weekly_demand')
        
        if weekly is None or len(weekly) == 0:
            return pd.DataFrame()
            
        # Aggregate by item
        if metric == 'demand':
            item_stats = weekly.groupby('item_id')['demand'].sum().reset_index()
            item_stats.columns = ['item_id', 'total_demand']
            item_stats = item_stats.sort_values('total_demand', ascending=False)
        elif metric == 'revenue':
            weekly['revenue'] = weekly['demand'] * weekly['avg_price']
            item_stats = weekly.groupby('item_id')['revenue'].sum().reset_index()
            item_stats = item_stats.sort_values('revenue', ascending=False)
        else:  # frequency
            item_stats = weekly.groupby('item_id')['demand'].count().reset_index()
            item_stats.columns = ['item_id', 'weeks_with_demand']
            item_stats = item_stats.sort_values('weeks_with_demand', ascending=False)
            
        # Add item names
        items_df = place_data.get('items')
        if items_df is not None:
            item_stats = item_stats.merge(
                items_df[['id', 'title']], 
                left_on='item_id', 
                right_on='id',
                how='left'
            )
            
        return item_stats.head(n)
    
    def export_place_data(
        self, 
        place_id: int, 
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export all data for a place to CSV files.
        
        Args:
            place_id: Location identifier
            output_path: Output directory (uses config default if not provided)
            
        Returns:
            Path to output directory
        """
        output_dir = output_path or (self.config.output_path / f'place_{place_id}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        place_data = self.get_place_data(place_id)
        
        for key, df in place_data.items():
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                df.to_csv(output_dir / f'{key}.csv', index=False)
                
        logger.info(f"Exported place {place_id} data to {output_dir}")
        return output_dir
