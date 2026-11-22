import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_schema_validation():
    # Missing required fields should return 422 Unprocessable Entity
    response = client.post("/predict", json={"brand": "Toyota"})
    assert response.status_code == 422

def test_predict_valid_vehicle():
    payload = {
        "brand": "Lexus",
        "model": "RX 350 Base",
        "model_year": 2016,
        "milage": 67500,
        "fuel_type": "Gasoline",
        "engine": "295.0HP 3.5L V6 Cylinder Engine Gasoline Fuel",
        "transmission": "8-Speed A/T",
        "ext_col": "Black",
        "int_col": "Black",
        "accident": "None reported",
        "clean_title": "Yes",
        "asking_price": 26500,
    }
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Valuation & Duan smearing sanity checks
    assert data["expected_market_value_usd"] > 0
    assert data["median_listing_value_usd"] > 0
    assert data["expected_market_value_usd"] >= data["median_listing_value_usd"]

    # Conformalized 80% interval ordering check
    assert data["interval_80_low_usd"] < data["median_listing_value_usd"] < data["interval_80_high_usd"]

    # Deal grader & deterministic feature parsing checks
    assert data["deal_grade"] == "Fair Market Price"
    feats = data["engineered_features"]
    assert feats["vehicle_age"] == 8
    assert feats["horsepower"] == 295.0
    assert feats["displacement_l"] == 3.5
    assert feats["cylinders"] == 6.0
    assert feats["is_luxury"] is True
    assert feats["is_turbo"] is False