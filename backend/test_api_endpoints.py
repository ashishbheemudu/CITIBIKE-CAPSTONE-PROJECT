"""
Integration tests for API endpoints
Run with: python -m pytest test_api_endpoints.py -v
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_test_client():
    """Get test client with data loader initialized"""
    from main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints"""

    def test_root_endpoint(self):
        """Test root health check"""
        client = get_test_client()
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Backend is running"

    def test_system_overview_returns_kpis(self):
        """Test system overview returns expected structure"""
        client = get_test_client()
        response = client.get("/api/system-overview")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "kpis" in data
        assert "total_trips" in data["kpis"]
        assert "weekend_share" in data["kpis"]
        assert "peak_hour" in data["kpis"]


class TestStationEndpoints:
    """Tests for station-related endpoints"""

    def test_stations_list(self):
        """Test stations list endpoint"""
        client = get_test_client()
        response = client.get("/api/stations")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0

    def test_map_data_has_coordinates(self):
        """Test map data includes coordinates"""
        client = get_test_client()
        response = client.get("/api/map-data")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        if len(data) > 0:
            assert "lat" in data[0]
            assert "lon" in data[0]


class TestPredictionEndpoints:
    """Tests for prediction endpoints"""

    def test_predict_with_valid_input(self):
        """Test prediction with valid input"""
        client = get_test_client()
        response = client.post("/api/predict", json={
            "station_name": "Broadway & W 58 St",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-01T12:00:00"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) > 0

    def test_predict_with_invalid_station(self):
        """Test prediction with non-existent station"""
        client = get_test_client()
        response = client.post("/api/predict", json={
            "station_name": "NonExistent Station 12345",
            "start_date": "2024-01-01T00:00:00"
        })
        
        # Should still return 200 but with predictions using default data
        assert response.status_code in [200, 404, 422]


class TestAdvancedAnalyticsEndpoints:
    """Tests for advanced analytics endpoints"""

    def test_abm_endpoint(self):
        """Test ABM simulation endpoint"""
        client = get_test_client()
        response = client.get("/api/advanced-analytics/abm")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_equity_endpoint(self):
        """Test equity scores endpoint"""
        client = get_test_client()
        response = client.get("/api/advanced-analytics/equity")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestInputValidation:
    """Tests for input validation"""

    def test_invalid_date_format(self):
        """Test with invalid date format"""
        client = get_test_client()
        response = client.get("/api/system-overview?start_date=not-a-date")
        
        # Should handle gracefully
        assert response.status_code in [200, 422]

    def test_routes_with_valid_top_n(self):
        """Test routes endpoint with valid top_n"""
        client = get_test_client()
        response = client.get("/api/routes?top_n=5")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
