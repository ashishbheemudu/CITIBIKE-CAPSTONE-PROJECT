"""
Unit tests for prediction_service.py
Run with: python -m pytest test_prediction_service.py -v
"""
import pytest
import sys
import os
from datetime import datetime
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestPredictionService:
    """Tests for PredictionService class"""

    def test_import_prediction_service(self):
        """Test that prediction_service can be imported"""
        from prediction_service import PredictionService
        assert PredictionService is not None

    def test_initialization(self):
        """Test PredictionService initialization"""
        from prediction_service import PredictionService
        
        ps = PredictionService()
        assert ps.models == {}
        assert ps.scalers == {}
        assert ps.feature_names == []

    def test_lazy_load_models_loads_features(self):
        """Test that lazy load correctly loads feature names"""
        from prediction_service import PredictionService
        
        ps = PredictionService()
        ps._lazy_load_models()
        
        # Should have loaded 54 features
        assert len(ps.feature_names) == 54

    def test_create_features_batch_returns_correct_count(self):
        """Test that batch feature creation returns correct number of features"""
        from prediction_service import PredictionService
        
        ps = PredictionService()
        timestamps = [datetime(2024, 1, 1, i) for i in range(12)]
        
        features = ps._create_features_batch("Test Station", timestamps, None)
        
        assert len(features) == 12  # 12 timestamps
        assert len(features[0]) == 54  # 54 features each

    def test_create_features_single_returns_correct_count(self):
        """Test that single feature creation returns 54 features"""
        from prediction_service import PredictionService
        
        ps = PredictionService()
        ts = datetime(2024, 1, 1, 12)
        
        features = ps._create_features("Test Station", ts, None)
        
        assert len(features) == 54

    def test_feature_order_consistency(self):
        """Test that batch and single feature creation have same order"""
        from prediction_service import PredictionService
        
        ps = PredictionService()
        ts = datetime(2024, 1, 1, 12)
        
        single_features = ps._create_features("Test Station", ts, None)
        batch_features = ps._create_features_batch("Test Station", [ts], None)
        
        # First 10 features should match (weather and temporal)
        for i in range(10):
            assert abs(single_features[i] - batch_features[0][i]) < 0.01, f"Feature {i} mismatch"


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_station_cache(self):
        """Test with empty station cache"""
        from prediction_service import PredictionService
        
        ps = PredictionService()
        ts = datetime(2024, 1, 1, 12)
        
        # Empty dataframe
        empty_cache = pd.DataFrame()
        features = ps._create_features("Test", ts, empty_cache)
        
        assert len(features) == 54

    def test_holiday_detection(self):
        """Test holiday detection in features"""
        from prediction_service import PredictionService
        import holidays
        
        ps = PredictionService()
        
        # Christmas 2024
        christmas = datetime(2024, 12, 25, 12)
        features = ps._create_features("Test", christmas, None)
        
        # is_holiday should be 1 (index 3 based on feature order)
        assert features[3] == 1  # is_holiday

    def test_weekend_detection(self):
        """Test weekend detection"""
        from prediction_service import PredictionService
        
        ps = PredictionService()
        
        # Saturday
        saturday = datetime(2024, 1, 6, 12)
        features = ps._create_features("Test", saturday, None)
        
        # is_weekend should be 1 (index 5)
        assert features[5] == 1  # is_weekend

    def test_cyclical_features_range(self):
        """Test that cyclical features are in valid range [-1, 1]"""
        from prediction_service import PredictionService
        
        ps = PredictionService()
        ts = datetime(2024, 6, 15, 14)
        features = ps._create_features("Test", ts, None)
        
        # Cyclical features (hour_sin, hour_cos, etc. at indices 13-18)
        for i in range(13, 19):
            assert -1 <= features[i] <= 1, f"Cyclical feature {i} out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
