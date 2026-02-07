"""
Data Model Documentation
========================
Infers and documents relationships between tables.
Creates logical star schema diagram documentation.
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path


class DataModelDocumenter:
    """Document data model and relationships."""
    
    def __init__(self, discovery_results: Dict[str, Any]):
        """
        Initialize with discovery results.
        
        Parameters:
        -----------
        discovery_results : dict
            Results from DataDiscovery.discover_all()
        """
        self.profiles = discovery_results['profiles']
        self.fact_tables = discovery_results['fact_tables']
        self.dimension_tables = discovery_results['dimension_tables']
        self.relationships = []
        self.model_doc = {}
        
    def infer_relationships(self) -> List[Dict[str, Any]]:
        """Infer relationships between tables based on column names and structure."""
        relationships = []
        
        # Core fact table relationships
        if 'fct_orders' in self.profiles:
            orders_profile = self.profiles['fct_orders']
            orders_cols = orders_profile['columns']
            
            # fct_orders -> dim_places
            if 'place_id' in orders_cols:
                relationships.append({
                    'from_table': 'fct_orders',
                    'to_table': 'dim_places',
                    'foreign_key': 'place_id',
                    'relationship_type': 'many_to_one',
                    'description': 'Orders belong to a Place/Location',
                    'cardinality': 'N:1'
                })
            
            # fct_orders -> dim_users
            if 'user_id' in orders_cols:
                relationships.append({
                    'from_table': 'fct_orders',
                    'to_table': 'dim_users',
                    'foreign_key': 'user_id',
                    'relationship_type': 'many_to_one',
                    'description': 'Orders placed by Users',
                    'cardinality': 'N:1'
                })
        
        # fct_order_items relationships
        if 'fct_order_items' in self.profiles:
            order_items_profile = self.profiles['fct_order_items']
            order_items_cols = order_items_profile['columns']
            
            # fct_order_items -> fct_orders
            if 'order_id' in order_items_cols:
                relationships.append({
                    'from_table': 'fct_order_items',
                    'to_table': 'fct_orders',
                    'foreign_key': 'order_id',
                    'relationship_type': 'many_to_one',
                    'description': 'Order Items belong to Orders',
                    'cardinality': 'N:1'
                })
            
            # fct_order_items -> dim_menu_items
            if 'item_id' in order_items_cols:
                relationships.append({
                    'from_table': 'fct_order_items',
                    'to_table': 'dim_menu_items',
                    'foreign_key': 'item_id',
                    'relationship_type': 'many_to_one',
                    'description': 'Order Items reference Menu Items',
                    'cardinality': 'N:1'
                })
        
        # Inventory relationships
        if 'fct_inventory_reports' in self.profiles:
            inv_profile = self.profiles['fct_inventory_reports']
            inv_cols = inv_profile['columns']
            
            if 'place_id' in inv_cols:
                relationships.append({
                    'from_table': 'fct_inventory_reports',
                    'to_table': 'dim_places',
                    'foreign_key': 'place_id',
                    'relationship_type': 'many_to_one',
                    'description': 'Inventory Reports belong to Places',
                    'cardinality': 'N:1'
                })
        
        # SKU relationships
        if 'dim_skus' in self.profiles:
            sku_profile = self.profiles['dim_skus']
            sku_cols = sku_profile['columns']
            
            # dim_skus -> dim_stock_categories
            if 'stock_category_id' in sku_cols:
                relationships.append({
                    'from_table': 'dim_skus',
                    'to_table': 'dim_stock_categories',
                    'foreign_key': 'stock_category_id',
                    'relationship_type': 'many_to_one',
                    'description': 'SKUs belong to Stock Categories',
                    'cardinality': 'N:1'
                })
            
            # dim_skus -> dim_items (if exists)
            if 'item_id' in sku_cols:
                relationships.append({
                    'from_table': 'dim_skus',
                    'to_table': 'dim_items',
                    'foreign_key': 'item_id',
                    'relationship_type': 'many_to_one',
                    'description': 'SKUs may reference Items',
                    'cardinality': 'N:1'
                })
        
        # Bill of Materials (BOM) relationships
        if 'dim_bill_of_materials' in self.profiles:
            bom_profile = self.profiles['dim_bill_of_materials']
            bom_cols = bom_profile['columns']
            
            if 'parent_sku_id' in bom_cols and 'sku_id' in bom_cols:
                relationships.append({
                    'from_table': 'dim_bill_of_materials',
                    'to_table': 'dim_skus',
                    'foreign_key': 'parent_sku_id',
                    'relationship_type': 'many_to_one',
                    'description': 'BOM defines parent SKU (composite items)',
                    'cardinality': 'N:1'
                })
                
                relationships.append({
                    'from_table': 'dim_bill_of_materials',
                    'to_table': 'dim_skus',
                    'foreign_key': 'sku_id',
                    'relationship_type': 'many_to_one',
                    'description': 'BOM defines component SKUs',
                    'cardinality': 'N:1'
                })
        
        # Campaign relationships
        if 'fct_campaigns' in self.profiles:
            camp_profile = self.profiles['fct_campaigns']
            camp_cols = camp_profile['columns']
            
            if 'campaign_id' in camp_cols:
                relationships.append({
                    'from_table': 'fct_campaigns',
                    'to_table': 'dim_campaigns',
                    'foreign_key': 'campaign_id',
                    'relationship_type': 'many_to_one',
                    'description': 'Campaign facts reference Campaign dimensions',
                    'cardinality': 'N:1'
                })
        
        # Bonus codes -> Orders
        if 'fct_bonus_codes' in self.profiles:
            bonus_profile = self.profiles['fct_bonus_codes']
            bonus_cols = bonus_profile['columns']
            
            # Note: Would need to check if orders reference bonus codes
            relationships.append({
                'from_table': 'fct_bonus_codes',
                'to_table': 'fct_orders',
                'foreign_key': 'code (inferred)',
                'relationship_type': 'one_to_many',
                'description': 'Bonus codes can be applied to multiple orders',
                'cardinality': '1:N',
                'note': 'Relationship inferred from business logic'
            })
        
        self.relationships = relationships
        return relationships
    
    def create_star_schema_documentation(self) -> Dict[str, Any]:
        """Create star schema documentation."""
        
        # Identify fact tables and their dimensions
        star_schema = {}
        
        # Core Sales Star Schema
        if 'fct_orders' in self.fact_tables:
            star_schema['sales_star'] = {
                'fact_table': 'fct_orders',
                'dimensions': [
                    {'table': 'dim_places', 'key': 'place_id', 'role': 'Location'},
                    {'table': 'dim_users', 'key': 'user_id', 'role': 'Customer'},
                    {'table': 'dim_menu_items', 'key': 'item_id (via fct_order_items)', 'role': 'Product'}
                ],
                'grain': 'One row per order',
                'measures': ['total_amount', 'items_amount', 'discount_amount', 'vat_amount']
            }
        
        # Order Items Star Schema
        if 'fct_order_items' in self.fact_tables:
            star_schema['order_items_star'] = {
                'fact_table': 'fct_order_items',
                'dimensions': [
                    {'table': 'fct_orders', 'key': 'order_id', 'role': 'Order'},
                    {'table': 'dim_menu_items', 'key': 'item_id', 'role': 'Product'},
                    {'table': 'dim_places', 'key': 'place_id (via fct_orders)', 'role': 'Location'}
                ],
                'grain': 'One row per order item',
                'measures': ['quantity', 'price', 'cost', 'discount_amount']
            }
        
        # Inventory Star Schema
        if 'fct_inventory_reports' in self.fact_tables:
            star_schema['inventory_star'] = {
                'fact_table': 'fct_inventory_reports',
                'dimensions': [
                    {'table': 'dim_places', 'key': 'place_id', 'role': 'Location'},
                    {'table': 'dim_skus', 'key': 'sku_id (inferred)', 'role': 'Product'}
                ],
                'grain': 'One row per inventory report snapshot',
                'measures': ['quantity (inferred)', 'stock_levels']
            }
        
        # SKU Dimension Details
        if 'dim_skus' in self.dimension_tables:
            star_schema['sku_dimension'] = {
                'dimension_table': 'dim_skus',
                'attributes': ['id', 'title', 'quantity', 'low_stock_threshold', 'unit', 'type'],
                'hierarchies': [
                    {
                        'level': 'Stock Category',
                        'table': 'dim_stock_categories',
                        'key': 'stock_category_id'
                    }
                ]
            }
        
        return star_schema
    
    def generate_model_documentation(self) -> Dict[str, Any]:
        """Generate complete data model documentation."""
        print("\n" + "=" * 80)
        print("DATA MODEL DOCUMENTATION")
        print("=" * 80)
        
        relationships = self.infer_relationships()
        star_schema = self.create_star_schema_documentation()
        
        self.model_doc = {
            'generated_at': datetime.now().isoformat(),
            'fact_tables': self.fact_tables,
            'dimension_tables': self.dimension_tables,
            'relationships': relationships,
            'star_schemas': star_schema,
            'business_logic': {
                'order_flow': [
                    'Customer places order (fct_orders)',
                    'Order contains items (fct_order_items)',
                    'Items reference menu items (dim_menu_items)',
                    'Menu items may require SKUs (dim_skus via BOM)',
                    'Inventory tracked separately (fct_inventory_reports, dim_skus)'
                ],
                'inventory_flow': [
                    'SKUs tracked in dim_skus',
                    'Composite items defined via dim_bill_of_materials',
                    'Inventory snapshots in fct_inventory_reports',
                    'Stock levels updated in dim_skus.quantity'
                ],
                'campaign_flow': [
                    'Campaigns defined in dim_campaigns',
                    'Campaign usage tracked in fct_campaigns',
                    'Bonus codes tracked in fct_bonus_codes',
                    'Orders may reference campaigns/bonus codes'
                ]
            }
        }
        
        return self.model_doc
    
    def print_model_summary(self):
        """Print human-readable model summary."""
        if not self.model_doc:
            self.generate_model_documentation()
        
        doc = self.model_doc
        
        print("\n" + "=" * 80)
        print("DATA MODEL SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Fact Tables ({len(doc['fact_tables'])}):")
        for fact in doc['fact_tables']:
            print(f"   • {fact}")
        
        print(f"\n📋 Dimension Tables ({len(doc['dimension_tables'])}):")
        for dim in doc['dimension_tables']:
            print(f"   • {dim}")
        
        print(f"\n🔗 Relationships ({len(doc['relationships'])}):")
        for rel in doc['relationships'][:10]:  # Show first 10
            print(f"   • {rel['from_table']} --[{rel['foreign_key']}]--> {rel['to_table']}")
            print(f"     {rel['description']} ({rel['cardinality']})")
        
        print(f"\n⭐ Star Schemas:")
        for schema_name, schema_info in doc['star_schemas'].items():
            print(f"\n   {schema_name.upper()}:")
            print(f"   Fact: {schema_info.get('fact_table', schema_info.get('dimension_table', 'N/A'))}")
            if 'dimensions' in schema_info:
                print(f"   Dimensions:")
                for dim in schema_info['dimensions']:
                    print(f"     - {dim['table']} ({dim['role']})")
            if 'grain' in schema_info:
                print(f"   Grain: {schema_info['grain']}")
        
        print("\n" + "=" * 80)
    
    def save_documentation(self, output_path: str):
        """Save model documentation to JSON."""
        if not self.model_doc:
            self.generate_model_documentation()
        
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(self.model_doc, f, indent=2, default=str)
        
        print(f"\n✓ Model documentation saved to: {output_path}")


if __name__ == "__main__":
    # This will be called from main script
    pass
