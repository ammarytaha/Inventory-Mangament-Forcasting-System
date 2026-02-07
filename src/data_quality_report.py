"""
Data Quality Report Generator
==============================
Generates comprehensive data quality assessment reports.
Answers: Which tables are usable? Which need cleaning? What are the issues?
"""

import pandas as pd
import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path


class DataQualityReport:
    """Generate actionable data quality reports."""
    
    def __init__(self, discovery_results: Dict[str, Any]):
        """
        Initialize with discovery results.
        
        Parameters:
        -----------
        discovery_results : dict
            Results from DataDiscovery.discover_all()
        """
        self.profiles = discovery_results['profiles']
        self.summary = discovery_results['summary']
        self.fact_tables = discovery_results['fact_tables']
        self.dimension_tables = discovery_results['dimension_tables']
        self.report = {}
        
    def assess_table_readiness(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess if a table is ready for use.
        Returns: {'status': 'ready'|'needs_cleaning'|'critical_issues', 'issues': [], 'score': 0-100}
        """
        issues = []
        score = 100
        
        # Critical issues (reduce score significantly)
        if profile['row_count'] == 0:
            issues.append("CRITICAL: Table is empty")
            score -= 50
        
        dq_issues = profile.get('data_quality_issues', {})
        
        # Duplicate primary keys
        if dq_issues.get('duplicate_keys'):
            issues.append("CRITICAL: Duplicate primary keys detected")
            score -= 30
        
        # High missing values in key columns
        missing_pct = profile.get('missing_pct', {})
        id_cols = [col for col in profile['columns'] if col.endswith('_id') or col == 'id']
        for col in id_cols:
            if col in missing_pct and missing_pct[col] > 50:
                issues.append(f"CRITICAL: {col} has {missing_pct[col]}% missing values")
                score -= 20
        
        # Warning issues (reduce score moderately)
        high_missing_cols = {k: v for k, v in missing_pct.items() if v > 20 and k not in id_cols}
        if len(high_missing_cols) > 5:
            issues.append(f"WARNING: {len(high_missing_cols)} columns have >20% missing values")
            score -= 10
        
        # Negative amounts
        if dq_issues.get('negative_amounts'):
            issues.append(f"WARNING: {len(dq_issues['negative_amounts'])} columns have negative values")
            score -= 5
        
        # Missing foreign keys in fact tables
        if profile['table_type'] == 'fact' and dq_issues.get('missing_keys'):
            missing_fks = dq_issues['missing_keys']
            critical_fks = [fk for fk in missing_fks if fk['pct'] > 30]
            if critical_fks:
                issues.append(f"WARNING: {len(critical_fks)} foreign keys have >30% missing values")
                score -= 10
        
        # Determine status
        if score >= 80:
            status = 'ready'
        elif score >= 50:
            status = 'needs_cleaning'
        else:
            status = 'critical_issues'
        
        return {
            'status': status,
            'score': max(0, score),
            'issues': issues,
            'issue_count': len(issues)
        }
    
    def check_relationships(self) -> Dict[str, Any]:
        """Check referential integrity between fact and dimension tables."""
        relationship_issues = []
        
        # Check fct_orders -> dim_places
        if 'fct_orders' in self.profiles and 'dim_places' in self.profiles:
            orders = self.profiles['fct_orders']
            places = self.profiles['dim_places']
            
            if 'place_id' in orders['columns']:
                # This would require loading the actual data, so we note it for later
                relationship_issues.append({
                    'type': 'referential_integrity',
                    'fact_table': 'fct_orders',
                    'dimension_table': 'dim_places',
                    'foreign_key': 'place_id',
                    'check_needed': True
                })
        
        # Check fct_order_items -> dim_menu_items
        if 'fct_order_items' in self.profiles and 'dim_menu_items' in self.profiles:
            relationship_issues.append({
                'type': 'referential_integrity',
                'fact_table': 'fct_order_items',
                'dimension_table': 'dim_menu_items',
                'foreign_key': 'item_id',
                'check_needed': True
            })
        
        # Check fct_order_items -> fct_orders
        if 'fct_order_items' in self.profiles and 'fct_orders' in self.profiles:
            relationship_issues.append({
                'type': 'fact_to_fact',
                'fact_table': 'fct_order_items',
                'parent_fact': 'fct_orders',
                'foreign_key': 'order_id',
                'check_needed': True
            })
        
        return relationship_issues
    
    def identify_suspicious_columns(self) -> List[Dict[str, Any]]:
        """Identify columns that are unreliable or suspicious."""
        suspicious = []
        
        for file_name, profile in self.profiles.items():
            # Check for columns with very high missing rates
            missing_pct = profile.get('missing_pct', {})
            for col, pct in missing_pct.items():
                if pct > 80:
                    suspicious.append({
                        'table': file_name,
                        'column': col,
                        'issue': f'{pct}% missing values',
                        'severity': 'high'
                    })
            
            # Check for columns with all same value (no variance)
            # This would require loading data, so we note it
            
            # Check for negative quantities in inventory
            if 'dim_skus' in file_name:
                dq_issues = profile.get('data_quality_issues', {})
                if dq_issues.get('negative_amounts'):
                    for neg_issue in dq_issues['negative_amounts']:
                        if 'quantity' in neg_issue['column'].lower():
                            suspicious.append({
                                'table': file_name,
                                'column': neg_issue['column'],
                                'issue': f"Negative quantities detected (min: {neg_issue['min_value']})",
                                'severity': 'critical'
                            })
        
        return suspicious
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        print("\n" + "=" * 80)
        print("DATA QUALITY ASSESSMENT REPORT")
        print("=" * 80)
        
        # Assess each table
        table_assessments = {}
        for file_name, profile in self.profiles.items():
            if 'status' in profile and profile['status'] == 'error':
                continue
            table_assessments[file_name] = self.assess_table_readiness(profile)
        
        # Identify suspicious columns
        suspicious_columns = self.identify_suspicious_columns()
        
        # Check relationships
        relationship_issues = self.check_relationships()
        
        # Categorize tables
        ready_tables = [t for t, a in table_assessments.items() if a['status'] == 'ready']
        needs_cleaning = [t for t, a in table_assessments.items() if a['status'] == 'needs_cleaning']
        critical_issues = [t for t, a in table_assessments.items() if a['status'] == 'critical_issues']
        
        self.report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_tables': len(table_assessments),
                'ready_tables': len(ready_tables),
                'needs_cleaning': len(needs_cleaning),
                'critical_issues': len(critical_issues)
            },
            'table_assessments': table_assessments,
            'ready_tables': ready_tables,
            'needs_cleaning': needs_cleaning,
            'critical_issues': critical_issues,
            'suspicious_columns': suspicious_columns,
            'relationship_checks': relationship_issues
        }
        
        return self.report
    
    def print_report(self):
        """Print human-readable quality report."""
        if not self.report:
            self.generate_report()
        
        report = self.report
        
        print("\n" + "=" * 80)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 80)
        
        print(f"\n📊 Summary:")
        print(f"   Total Tables Assessed: {report['summary']['total_tables']}")
        print(f"   ✅ Ready to Use: {report['summary']['ready_tables']}")
        print(f"   ⚠️  Needs Cleaning: {report['summary']['needs_cleaning']}")
        print(f"   ❌ Critical Issues: {report['summary']['critical_issues']}")
        
        # Ready tables
        if report['ready_tables']:
            print(f"\n✅ READY TO USE ({len(report['ready_tables'])} tables):")
            for table in report['ready_tables']:
                score = report['table_assessments'][table]['score']
                print(f"   • {table} (Quality Score: {score}/100)")
        
        # Needs cleaning
        if report['needs_cleaning']:
            print(f"\n⚠️  NEEDS CLEANING ({len(report['needs_cleaning'])} tables):")
            for table in report['needs_cleaning']:
                assessment = report['table_assessments'][table]
                print(f"   • {table} (Score: {assessment['score']}/100)")
                for issue in assessment['issues'][:3]:  # Show first 3 issues
                    print(f"     - {issue}")
        
        # Critical issues
        if report['critical_issues']:
            print(f"\n❌ CRITICAL ISSUES ({len(report['critical_issues'])} tables):")
            for table in report['critical_issues']:
                assessment = report['table_assessments'][table]
                print(f"   • {table} (Score: {assessment['score']}/100)")
                for issue in assessment['issues']:
                    print(f"     - {issue}")
        
        # Suspicious columns
        if report['suspicious_columns']:
            print(f"\n🔍 SUSPICIOUS COLUMNS ({len(report['suspicious_columns'])}):")
            for col_info in report['suspicious_columns'][:10]:  # Show first 10
                severity_icon = "🔴" if col_info['severity'] == 'critical' else "🟡"
                print(f"   {severity_icon} {col_info['table']}.{col_info['column']}: {col_info['issue']}")
        
        print("\n" + "=" * 80)
    
    def save_report(self, output_path: str):
        """Save report to JSON file."""
        if not self.report:
            self.generate_report()
        
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        
        print(f"\n✓ Report saved to: {output_path}")


if __name__ == "__main__":
    # This will be called from main script
    pass
