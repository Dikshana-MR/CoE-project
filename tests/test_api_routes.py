import pytest
import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_check(client):
    """Tests /api/health and /api/status endpoints."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "database_connected" in data


def test_api_stats(client):
    """Tests /api/stats endpoint."""
    res = client.get("/api/stats")
    assert res.status_code in [200, 503]
    if res.status_code == 200:
        data = res.json()
        assert "total_complaints" in data
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data


def test_api_list_complaints(client):
    """Tests /api/complaints endpoint."""
    res = client.get("/api/complaints")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_api_create_complaint_valid(client):
    """Tests POST /api/complaints with valid data."""
    payload = {
        "description": "Large road pothole near junction",
        "category": "Road",
        "latitude": 10.125,
        "longitude": 76.540,
    }
    res = client.post("/api/complaints", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert "complaint_id" in data
    assert "issue_id" in data


def test_api_create_complaint_invalid_description(client):
    """Tests POST /api/complaints with invalid description."""
    payload = {
        "description": "",
        "category": "Road",
        "latitude": 10.125,
        "longitude": 76.540,
    }
    res = client.post("/api/complaints", json=payload)
    assert res.status_code == 422


def test_api_create_complaint_missing_category(client):
    """Tests POST /api/complaints with missing category."""
    payload = {
        "description": "Streetlight broken",
        "category": "",
    }
    res = client.post("/api/complaints", json=payload)
    assert res.status_code == 422


def test_api_deduplicate_pair(client):
    """Tests POST /api/deduplicate pair check."""
    payload = {
        "complaint_a": {
            "complaint_id": "C_TEST1",
            "category": "Road",
            "description": "pothole on street",
            "latitude": 10.10,
            "longitude": 76.40,
        },
        "complaint_b": {
            "complaint_id": "C_TEST2",
            "category": "Road",
            "description": "pothole on street",
            "latitude": 10.10,
            "longitude": 76.40,
        },
        "threshold": 0.55,
    }
    res = client.post("/api/deduplicate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_duplicate"] is True
    assert "duplicate_score" in data


def test_api_list_issues(client):
    """Tests GET /api/issues endpoint."""
    res = client.get("/api/issues")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_api_resolve_issue(client):
    """Tests POST /api/issues/{issue_id}/resolve."""
    res = client.get("/api/issues")
    if res.status_code == 200 and len(res.json()) > 0:
        target_id = res.json()[0]["issue_id"]
        res_res = client.post(f"/api/issues/{target_id}/resolve")
        assert res_res.status_code == 200
        assert res_res.json()["status"] == "Resolved"
