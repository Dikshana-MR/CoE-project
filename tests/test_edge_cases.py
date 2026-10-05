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
    get_temporal_similarity,
    get_image_similarity,
    run_deduplication,
    DEFAULT_THRESHOLD,
)
from preprocessing import normalize_complaint_text


def test_1_exact_duplicate_complaint():
    """1. Exact duplicate complaint text and location."""
    cmp1 = {
        "complaint_id": "CMP-E1A",
        "category": "Road",
        "description": "Large pothole on main street near bus stop",
        "latitude": 10.125628,
        "longitude": 76.540440,
        "created_date": "2026-08-01",
    }
    cmp2 = {
        "complaint_id": "CMP-E1B",
        "category": "Road",
        "description": "Large pothole on main street near bus stop",
        "latitude": 10.125628,
        "longitude": 76.540440,
        "created_date": "2026-08-01",
    }
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["predicted_duplicate"] is True
    assert res["duplicate_score"] >= 0.85


def test_2_similar_wording():
    """2. Similar wording for same complaint."""
    cmp1 = {"complaint_id": "C2A", "category": "Lighting", "description": "street light not working", "latitude": 10.10, "longitude": 76.40, "created_date": "2026-08-05"}
    cmp2 = {"complaint_id": "C2B", "category": "Lighting", "description": "street lamp is not working", "latitude": 10.1001, "longitude": 76.4001, "created_date": "2026-08-05"}
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["predicted_duplicate"] is True


def test_3_different_wording_same_issue():
    """3. Different wording for same underlying issue."""
    cmp1 = {"complaint_id": "C3A", "category": "Road", "description": "road surface broken near junction", "latitude": 10.19475, "longitude": 76.40597, "created_date": "2026-08-10"}
    cmp2 = {"complaint_id": "C3B", "category": "Road", "description": "deep pothole causing traffic problem", "latitude": 10.19468, "longitude": 76.40640, "created_date": "2026-08-11"}
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["predicted_duplicate"] is True


def test_4_same_location_different_issue():
    """4. Same location but different issue / category."""
    cmp1 = {"complaint_id": "C4A", "category": "Road", "description": "pothole on main street", "latitude": 10.125628, "longitude": 76.540440}
    cmp2 = {"complaint_id": "C4B", "category": "Lighting", "description": "flickering lamp pole", "latitude": 10.125628, "longitude": 76.540440}
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["category_match"] == 0
    assert res["predicted_duplicate"] is False


def test_5_same_category_different_location():
    """5. Same category but far-away location (> 1km)."""
    cmp1 = {"complaint_id": "C5A", "category": "Waste", "description": "garbage accumulation near bus stop", "latitude": 10.125628, "longitude": 76.540440}
    cmp2 = {"complaint_id": "C5B", "category": "Waste", "description": "garbage accumulation near bus stop", "latitude": 10.450000, "longitude": 76.900000}
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["location_similarity"] == 0.0
    assert res["predicted_duplicate"] is False


def test_6_different_category_same_location():
    """6. Different category at exact same location."""
    cmp1 = {"complaint_id": "C6A", "category": "Waste", "description": "trash pile on road", "latitude": 10.20, "longitude": 76.30}
    cmp2 = {"complaint_id": "C6B", "category": "Lighting", "description": "broken light", "latitude": 10.20, "longitude": 76.30}
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["predicted_duplicate"] is False


def test_7_missing_latitude_handling():
    """7. Missing latitude handling without crash."""
    cmp1 = {"complaint_id": "C7A", "category": "Lighting", "description": "streetlight out near junction", "latitude": None, "longitude": 76.40}
    cmp2 = {"complaint_id": "C7B", "category": "Lighting", "description": "faulty lamp post", "latitude": 10.10, "longitude": 76.40}
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["location_similarity"] == 0.0
    assert "duplicate_score" in res


def test_8_missing_longitude_handling():
    """8. Missing longitude handling without crash."""
    cmp1 = {"complaint_id": "C8A", "category": "Road", "description": "pothole on road", "latitude": 10.10, "longitude": np.nan}
    cmp2 = {"complaint_id": "C8B", "category": "Road", "description": "pothole on road", "latitude": 10.10, "longitude": 76.40}
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["location_similarity"] == 0.0


def test_9_missing_complaint_text():
    """9. Missing complaint text handling."""
    cmp1 = {"complaint_id": "C9A", "category": "Waste", "description": "", "latitude": 10.10, "longitude": 76.40}
    cmp2 = {"complaint_id": "C9B", "category": "Waste", "description": "waste dumping", "latitude": 10.10, "longitude": 76.40}
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["text_similarity"] == 0.0


def test_10_duplicate_with_spelling_variation():
    """10. Duplicate complaint with spelling / synonym variations."""
    cmp1 = {"complaint_id": "C10A", "category": "Road", "description": "street light not workin", "latitude": 10.10, "longitude": 76.40}
    cmp2 = {"complaint_id": "C10B", "category": "Road", "description": "streetlight is not working", "latitude": 10.10, "longitude": 76.40}
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["predicted_duplicate"] is True


def test_11_duplicate_with_different_date():
    """11. Duplicate complaint reported 2 days apart."""
    cmp1 = {"complaint_id": "C11A", "category": "Road", "description": "deep pothole", "latitude": 10.10, "longitude": 76.40, "created_date": "2026-08-01"}
    cmp2 = {"complaint_id": "C11B", "category": "Road", "description": "deep pothole", "latitude": 10.10, "longitude": 76.40, "created_date": "2026-08-03"}
    res = calculate_pair_duplicate_score(cmp1, cmp2)
    assert res["temporal_similarity"] == 1.0
    assert res["predicted_duplicate"] is True


def test_12_unresolved_complaint_status():
    """12. Unresolved complaint status handling."""
    cmp_df = pd.DataFrame([
        {"complaint_id": "C12A", "category": "Waste", "description": "uncollected trash", "latitude": 10.10, "longitude": 76.40, "status": "OPEN"},
        {"complaint_id": "C12B", "category": "Waste", "description": "uncollected trash", "latitude": 10.10, "longitude": 76.40, "status": "IN PROGRESS"},
    ])
    cons_df, dup_pairs, _ = run_deduplication(cmp_df)
    assert len(dup_pairs) == 1
    assert "consolidated_issue_id" in cons_df.columns


def test_13_invalid_complaint_record():
    """13. Invalid complaint record normalization."""
    raw_text = "??? Street-Light  FLICKERING!!  "
    norm = normalize_complaint_text(raw_text)
    assert "streetlight" in norm
    assert "broken" in norm or "faulty" in norm or "flicker" in norm


def test_14_resolution_event_linkage():
    """14. Resolution event linkage data integrity."""
    from database import populate_db, get_issue_with_events, init_db, DB_PATH
    import os
    test_db = "outputs/test_temp.db"
    cmp_df = pd.DataFrame([{"complaint_id": "C14A", "category": "Lighting", "description": "light broken", "latitude": 10.1, "longitude": 76.4, "created_date": "2026-08-01", "status": "Open", "consolidated_issue_id": "ISS-0001"}])
    iss_df = pd.DataFrame([{"issue_id": "ISS-0001", "category": "Lighting", "latitude": 10.1, "longitude": 76.4, "issue_description": "light broken", "complaint_count": 1, "status": "Open", "assigned_to": "Lighting Team"}])
    evt_df = pd.DataFrame([{"event_id": "EVT-14", "issue_id": "ISSUE0001", "event_type": "Work Started", "event_date": "2026-08-02", "assigned_team": "Lighting Team", "notes": "Started repair"}])
    
    populate_db(cmp_df, iss_df, evt_df, test_db)
    issue_res = get_issue_with_events("ISS-0001", test_db)
    assert issue_res is not None
    assert len(issue_res["resolution_events"]) == 1
    assert issue_res["resolution_events"][0]["event_type"] == "Work Started"
    if os.path.exists(test_db):
        os.remove(test_db)


def test_15_large_dataset_scalability():
    """15. Scalable candidate generation complexity test."""
    from deduplication import generate_candidate_pairs
    n_records = 300
    df = pd.DataFrame({
        "complaint_id": [f"C{i:04d}" for i in range(n_records)],
        "category": ["Road"] * 100 + ["Lighting"] * 100 + ["Waste"] * 100,
        "description": ["sample pothole complaint"] * n_records,
        "latitude": np.random.uniform(10.0, 10.05, n_records),
        "longitude": np.random.uniform(76.0, 76.05, n_records),
    })
    cands = generate_candidate_pairs(df, max_distance_meters=350.0)
    total_pairs = n_records * (n_records - 1) // 2
    # Candidates should be significantly lower than total pairs due to category & spatial blocking
    assert len(cands) < total_pairs

