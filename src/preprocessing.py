import os
import pandas as pd
import numpy as np


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Safely normalizes column names to lowercase with single underscores."""
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    return cleaned


def load_raw_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw CSV files from the specified data directory."""
    complaints_path = os.path.join(data_dir, "complaints.csv")
    issues_path = os.path.join(data_dir, "issues.csv")
    events_path = os.path.join(data_dir, "resolution_events.csv")

    if not os.path.exists(complaints_path):
        raise FileNotFoundError(f"Complaints file not found at {complaints_path}")
    if not os.path.exists(issues_path):
        raise FileNotFoundError(f"Issues file not found at {issues_path}")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Resolution events file not found at {events_path}")

    cmp_raw = pd.read_csv(complaints_path)
    iss_raw = pd.read_csv(issues_path)
    evt_raw = pd.read_csv(events_path)

    return cmp_raw, iss_raw, evt_raw


def preprocess_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses and cleans the complaints DataFrame without deleting raw fields."""
    cleaned = normalize_column_names(df)

    # Standardize column mapping if names vary slightly
    col_mapping = {
        "complaint_id": "complaint_id",
        "description": "description",
        "category": "category",
        "latitude": "latitude",
        "longitude": "longitude",
        "image_path": "image_path",
        "created_date": "created_date",
        "status": "status",
        "issue_id": "issue_id",
    }
    cleaned = cleaned.rename(columns=col_mapping)

    # Missing value handling
    cleaned["description"] = cleaned["description"].fillna("No description").astype(str)
    cleaned["category"] = cleaned["category"].fillna("Unknown").astype(str).str.strip()
    
    # Safe float conversion for latitude / longitude
    cleaned["latitude"] = pd.to_numeric(cleaned["latitude"], errors="coerce")
    cleaned["longitude"] = pd.to_numeric(cleaned["longitude"], errors="coerce")

    # Missing image path
    cleaned["image_path"] = cleaned["image_path"].fillna("").astype(str)
    
    # Status default
    cleaned["status"] = cleaned["status"].fillna("Open").astype(str)

    # Date conversion
    if "created_date" in cleaned.columns:
        cleaned["created_date"] = pd.to_datetime(cleaned["created_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Preserve issue_id column (ground truth if present)
    if "issue_id" in cleaned.columns:
        cleaned["issue_id"] = cleaned["issue_id"].fillna("").astype(str)

    return cleaned


def preprocess_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses and cleans the issues DataFrame."""
    cleaned = normalize_column_names(df)
    
    cleaned["issue_description"] = cleaned.get("issue_description", cleaned.get("description", pd.Series([""] * len(cleaned)))).fillna("No description").astype(str)
    cleaned["category"] = cleaned["category"].fillna("Unknown").astype(str).str.strip()
    cleaned["latitude"] = pd.to_numeric(cleaned["latitude"], errors="coerce")
    cleaned["longitude"] = pd.to_numeric(cleaned["longitude"], errors="coerce")
    cleaned["status"] = cleaned["status"].fillna("Open").astype(str)
    cleaned["complaint_count"] = pd.to_numeric(cleaned.get("complaint_count", 1), errors="coerce").fillna(1).astype(int)

    return cleaned


def preprocess_resolution_events(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses and cleans the resolution events DataFrame."""
    cleaned = normalize_column_names(df)
    if "event_date" in cleaned.columns:
        cleaned["event_date"] = pd.to_datetime(cleaned["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return cleaned


def load_and_clean_all(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads and preprocesses all datasets."""
    cmp_raw, iss_raw, evt_raw = load_raw_data(data_dir)
    cmp_clean = preprocess_complaints(cmp_raw)
    iss_clean = preprocess_issues(iss_raw)
    evt_clean = preprocess_resolution_events(evt_raw)
    return cmp_clean, iss_clean, evt_clean


if __name__ == "__main__":
    cmp, iss, evt = load_and_clean_all()
    print(f"Loaded Clean Complaints: {len(cmp)}, Issues: {len(iss)}, Events: {len(evt)}")
