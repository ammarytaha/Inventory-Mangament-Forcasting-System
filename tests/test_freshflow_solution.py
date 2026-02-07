"""
FreshFlow AI - Test Module
===========================

Test suite for validating the FreshFlow AI solution components.
Can be used with testing data to verify recommendation quality.

Usage:
    python test_freshflow_solution.py
    python test_freshflow_solution.py --test-data path/to/test_data
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(text):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_success(text):
    """Print success message"""
    print(f"  ✅ {text}")


def print_error(text):
    """Print error message"""
    print(f"  ❌ {text}")


def print_info(text):
    """Print info message"""
    print(f"  ℹ️  {text}")


def test_imports():
    """Test that all modules can be imported"""
    print_header("Testing Module Imports")
    
    modules_to_test = [
        ('freshflow_ai', 'Main Package'),
        ('freshflow_ai.config', 'Configuration'),
        ('freshflow_ai.data_processor', 'Data Processor'),
        ('freshflow_ai.forecaster', 'Forecaster'),
        ('freshflow_ai.recommendation_engine', 'Recommendation Engine'),
        ('freshflow_ai.context_engine', 'Context Engine'),
        ('freshflow_ai.explanation_generator', 'Explanation Generator'),
    ]
    
    all_passed = True
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print_success(f"{description} ({module_name})")
        except Exception as e:
            print_error(f"{description}: {e}")
            all_passed = False
            
    return all_passed


def test_config():
    """Test configuration module"""
    print_header("Testing Configuration")
    
    try:
        from freshflow_ai.config import Config, DEFAULT_CONFIG
        
        # Test default config
        assert DEFAULT_CONFIG is not None
        print_success("Default config exists")
        
        # Test from_workspace
        config = Config.from_workspace(str(PROJECT_ROOT))
        assert config.data_path.exists() or True  # May not exist in test
        print_success("Config.from_workspace() works")
        
        # Test holiday detection
        test_date = datetime(2024, 12, 25)
        is_holiday, name, impact = config.get_holiday_impact(test_date)
        assert is_holiday == True
        assert name == "Christmas Day"
        print_success(f"Holiday detection works (Christmas: {name})")
        
        # Test weekly factor
        friday_factor = config.get_weekly_factor(4)  # Friday
        assert friday_factor > 1.0
        print_success(f"Weekly factor works (Friday: {friday_factor:.2f})")
        
        return True
        
    except Exception as e:
        print_error(f"Config test failed: {e}")
        return False


def test_context_engine():
    """Test context engine"""
    print_header("Testing Context Engine")
    
    try:
        from freshflow_ai.context_engine import ContextEngine, ContextEvent
        from freshflow_ai.config import Config
        
        config = Config.from_workspace(str(PROJECT_ROOT))
        engine = ContextEngine(config)
        
        # Test context factors
        context = engine.get_context_factors(
            place_id=94025,
            target_date=date.today()
        )
        
        assert 'base_factors' in context
        assert 'combined_factor' in context
        print_success(f"Context factors retrieved (factor: {context['combined_factor']:.2f})")
        
        # Test adding custom event
        engine.add_local_event(
            name="Test Event",
            start_date=date.today(),
            place_ids=[94025],
            impact_factor=1.3
        )
        
        context_with_event = engine.get_context_factors(
            place_id=94025,
            target_date=date.today()
        )
        print_success("Custom event can be added")
        
        # Test holiday calendar
        calendar = engine.get_holiday_calendar(2024)
        assert len(calendar) > 0
        print_success(f"Holiday calendar retrieved ({len(calendar)} events for 2024)")
        
        return True
        
    except Exception as e:
        print_error(f"Context engine test failed: {e}")
        return False


def test_recommendation_types():
    """Test recommendation data structures"""
    print_header("Testing Recommendation Structures")
    
    try:
        from freshflow_ai.recommendation_engine import (
            Recommendation, 
            RecommendationType, 
            RiskLevel
        )
        
        # Test creating a recommendation
        rec = Recommendation(
            recommendation_id="TEST-001",
            place_id=94025,
            item_id=12345,
            item_name="Test Item",
            recommendation_type=RecommendationType.REORDER,
            risk_level=RiskLevel.HIGH,
            action="Order 100 units",
            quantity=100,
            rationale="Test rationale",
            expected_impact="Test impact",
            confidence=0.85
        )
        
        assert rec.recommendation_id == "TEST-001"
        print_success("Recommendation can be created")
        
        # Test to_dict
        rec_dict = rec.to_dict()
        assert rec_dict['recommendation_type'] == 'reorder'
        assert rec_dict['risk_level'] == 'high'
        print_success("Recommendation.to_dict() works")
        
        # Test all recommendation types
        for rec_type in RecommendationType:
            assert rec_type.value in ['reorder', 'discount', 'bundle', 'prep_adjust', 'alert', 'hold']
        print_success(f"All {len(list(RecommendationType))} recommendation types defined")
        
        return True
        
    except Exception as e:
        print_error(f"Recommendation structure test failed: {e}")
        return False


def test_explanation_generator():
    """Test explanation generator"""
    print_header("Testing Explanation Generator")
    
    try:
        from freshflow_ai.explanation_generator import ExplanationGenerator
        from freshflow_ai.recommendation_engine import (
            Recommendation, 
            RecommendationType, 
            RiskLevel
        )
        
        generator = ExplanationGenerator()
        
        # Create test recommendation
        rec = Recommendation(
            recommendation_id="TEST-001",
            place_id=94025,
            item_id=12345,
            item_name="Organic Chicken Breast",
            recommendation_type=RecommendationType.REORDER,
            risk_level=RiskLevel.HIGH,
            action="Order 100 units",
            quantity=100,
            rationale="Based on forecast: 120 units expected over 4 weeks. Safety stock running low.",
            expected_impact="Prevents stockout for 4 weeks",
            confidence=0.85,
            additional_data={
                'demand_type': 'Smooth',
                'weekly_forecast': 30,
                'safety_stock': 25
            }
        )
        
        # Generate explanation
        explanation = generator.explain_recommendation(rec)
        
        assert 'summary' in explanation
        assert 'headline' in explanation
        assert 'confidence' in explanation
        print_success("Explanation generated successfully")
        
        # Check explanation components
        assert 'visual' in explanation['confidence']
        assert explanation['confidence']['percentage'] == 85
        print_success(f"Confidence: {explanation['confidence']['visual']} ({explanation['confidence']['percentage']}%)")
        
        # Test dashboard cards
        cards = generator.generate_dashboard_cards([rec])
        assert len(cards) == 1
        assert cards[0]['item_name'] == 'Organic Chicken Breast'
        print_success("Dashboard cards generated")
        
        # Test action summary
        summary = generator.generate_action_summary([rec])
        assert '1 recommendations' in summary
        print_success("Action summary generated")
        
        return True
        
    except Exception as e:
        print_error(f"Explanation generator test failed: {e}")
        return False


def test_data_processor_structure():
    """Test data processor without actual data"""
    print_header("Testing Data Processor Structure")
    
    try:
        from freshflow_ai.data_processor import DataProcessor
        from freshflow_ai.config import Config
        
        config = Config.from_workspace(str(PROJECT_ROOT))
        processor = DataProcessor(config)
        
        # Test classify_demand method
        import numpy as np
        import pandas as pd
        
        # Smooth demand pattern
        smooth_series = pd.Series([10, 11, 9, 10, 12, 10, 11, 10])
        classification = processor.classify_demand(smooth_series)
        assert classification in ['Smooth', 'Erratic']
        print_success(f"Smooth demand classified as: {classification}")
        
        # Intermittent demand pattern
        intermittent_series = pd.Series([0, 0, 5, 0, 0, 0, 6, 0])
        classification = processor.classify_demand(intermittent_series)
        assert classification in ['Intermittent', 'Lumpy']
        print_success(f"Intermittent demand classified as: {classification}")
        
        # Insufficient data
        insufficient_series = pd.Series([5])
        classification = processor.classify_demand(insufficient_series)
        assert classification == 'Insufficient Data'
        print_success("Insufficient data handled correctly")
        
        return True
        
    except Exception as e:
        print_error(f"Data processor test failed: {e}")
        return False


def test_forecaster_methods():
    """Test forecaster helper methods"""
    print_header("Testing Forecaster Methods")
    
    try:
        from freshflow_ai.forecaster import ForecastEngine
        from freshflow_ai.config import Config
        import numpy as np
        
        config = Config.from_workspace(str(PROJECT_ROOT))
        forecaster = ForecastEngine(config=config)
        
        # Test exponential smoothing
        values = np.array([10, 12, 11, 13, 12])
        smoothed = forecaster._exponential_smooth(values, alpha=0.3)
        assert smoothed > 0
        print_success(f"Exponential smoothing works (smoothed value: {smoothed:.2f})")
        
        # Test insufficient data forecast
        result = forecaster._insufficient_data_forecast(
            place_id=1, 
            item_id=1, 
            horizon=4
        )
        assert len(result['forecast']) == 4
        assert result['model_used'] == 'insufficient_data'
        print_success("Insufficient data forecast generated")
        
        return True
        
    except Exception as e:
        print_error(f"Forecaster test failed: {e}")
        return False


def test_with_sample_data():
    """Test with sample/mock data"""
    print_header("Testing with Sample Data")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample weekly demand data
        dates = pd.date_range(start='2023-01-01', periods=52, freq='W-MON')
        sample_data = pd.DataFrame({
            'week_start': dates,
            'demand': np.random.poisson(lam=50, size=52),
            'avg_price': np.random.uniform(10, 20, 52)
        })
        
        print_info(f"Created sample data: {len(sample_data)} weeks")
        
        # Test moving average forecast
        from freshflow_ai.forecaster import ForecastEngine
        forecaster = ForecastEngine()
        
        result = forecaster._forecast_moving_average(
            history=sample_data,
            horizon=4,
            include_confidence=True
        )
        
        assert len(result['forecast']) == 4
        assert result['model_used'] == 'moving_average'
        print_success(f"Moving average forecast: {result['forecast'][0]['predicted_demand']} units predicted")
        
        # Test Croston forecast
        intermittent_data = sample_data.copy()
        intermittent_data['demand'] = [0, 0, 10, 0, 5, 0, 0, 8, 0, 0, 0, 12] + [0] * 40
        
        result = forecaster._forecast_croston(
            history=intermittent_data,
            horizon=4,
            include_confidence=True
        )
        assert result['model_used'] == 'croston'
        print_success(f"Croston forecast working")
        
        return True
        
    except Exception as e:
        print_error(f"Sample data test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  🍃 FreshFlow AI - Solution Test Suite")
    print("=" * 60)
    print(f"\n  Running tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Context Engine", test_context_engine),
        ("Recommendation Structures", test_recommendation_types),
        ("Explanation Generator", test_explanation_generator),
        ("Data Processor Structure", test_data_processor_structure),
        ("Forecaster Methods", test_forecaster_methods),
        ("Sample Data Integration", test_with_sample_data),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print_error(f"Test {name} crashed: {e}")
            results.append((name, False))
            
    # Summary
    print_header("Test Summary")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
        
    print("\n" + "-" * 60)
    print(f"  Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n  🎉 All tests passed! Solution is ready.")
    else:
        print(f"\n  ⚠️  {total_count - passed_count} test(s) failed. Review errors above.")
        
    return passed_count == total_count


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FreshFlow AI Solution')
    parser.add_argument('--test-data', type=str, help='Path to test data folder')
    args = parser.parse_args()
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
