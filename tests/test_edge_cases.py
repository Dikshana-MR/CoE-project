import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path for direct module import in tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from deduplication import (
    calculate_pair_duplicate_score,
    haversine_distance,
    get_location_similarity,
    get_category_similarity,
    get_single_text_similarity,
)


def test_case_1_same_category_similar_desc_nearby_location():
    """
    CASE 1:
    Same category, similar description, nearby coordinates (within 50m).
    Expected: DUPLICATE (score >= 0.75)
    """
    cmp1 = {
        "complaint_id": "CMP-T1A",
        "category": "Road",
        "description": "Large pothole on main street near bus stop",
        "latitude": 10.125628,
        "longitude": 76.540440,
    }

    cmp2 = {
        "complaint_id": "CMP-T1B",
        "category": "Road",
        "description": "Deep pothole reported near bus stop on main street",
        "latitude": 10.125700,  # ~10 meters away
        "longitude": 76.540450,
    }

    res = calculate_pair_duplicate_score(cmp1, cmp2)

    assert res["category_match"] == 1
    assert res["location_similarity"] == 1.0  # <= 50m
    assert res["duplicate_score"] >= 0.75
    assert res["predicted_duplicate"] is True


def test_case_2_same_location_different_category():
    """
    CASE 2:
    Same location, different category.
    Expected: NOT DUPLICATE (score < 0.75)
    """
    cmp1 = {
        "complaint_id": "CMP-T2A",
        "category": "Road",
        "description": "Pothole on street",
        "latitude": 10.125628,
        "longitude": 76.540440,
    }

    cmp2 = {
        "complaint_id": "CMP-T2B",
        "category": "Lighting",  # Different category
        "description": "Streetlight pole broken",
        "latitude": 10.125628,  # Exact same location
        "longitude": 76.540440,
    }

    res = calculate_pair_duplicate_score(cmp1, cmp2)

    assert res["category_match"] == 0
    assert res["duplicate_score"] < 0.75
    assert res["predicted_duplicate"] is False


def test_case_3_same_description_faraway_location():
    """
    CASE 3:
    Same description, far-away location (> 250m).
    Expected: NOT DUPLICATE (score < 0.75)
    """
    cmp1 = {
        "complaint_id": "CMP-T3A",
        "category": "Waste",
        "description": "Waste pile near market plaza center",
        "latitude": 10.125628,
        "longitude": 76.540440,
    }

    cmp2 = {
        "complaint_id": "CMP-T3B",
        "category": "Waste",
        "description": "Waste pile near market plaza center",  # Identical description
        "latitude": 10.350000,  # ~25 km away
        "longitude": 76.800000,
    }

    res = calculate_pair_duplicate_score(cmp1, cmp2)

    assert res["location_similarity"] == 0.0  # > 250m
    # Max possible score: 0.50 (text) + 0.0 (loc) + 0.20 (cat) = 0.70 < 0.75
    assert res["duplicate_score"] < 0.75
    assert res["predicted_duplicate"] is False


def test_missing_lat_long_handling_does_not_crash():
    """
    Test ensuring missing latitude/longitude (None / NaN) does not crash the system
    and safely returns location similarity 0.0.
    """
    cmp1 = {
        "complaint_id": "CMP-M1",
        "category": "Lighting",
        "description": "Streetlight out near corner",
        "latitude": None,
        "longitude": np.nan,
    }

    cmp2 = {
        "complaint_id": "CMP-M2",
        "category": "Lighting",
        "description": "Dark street light pole",
        "latitude": 10.125628,
        "longitude": 76.540440,
    }

    dist = haversine_distance(cmp1["latitude"], cmp1["longitude"], cmp2["latitude"], cmp2["longitude"])
    loc_sim = get_location_similarity(dist)
    
    assert loc_sim == 0.0
    
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["location_similarity"] == 0.0
    assert "duplicate_score" in res
