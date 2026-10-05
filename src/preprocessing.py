import os
import re
import pandas as pd
import numpy as np

# Controlled Synonym Dictionary for Municipality Complaint Terminology
CONTROLLED_SYNONYMS = {
    # Road terminology
    "potholes": "pothole",
    "roadway": "road",
    "street": "road",
    "avenue": "road",
    "lane": "road",
    "highway": "road",
    "asphalt": "road",
    "tarmac": "road",
    "crack": "damage",
    "cracked": "damage",
    "damaged": "damage",
    "broken": "damage",
    "breaking": "damage",
    "defective": "damage",
    "repair": "fix",
    "repairing": "fix",
    "hazard": "problem",
    "issue": "problem",
    # Lighting terminology
    "street light": "streetlight",
    "street lamp": "streetlight",
    "light pole": "streetlight",
    "lamp": "streetlight",
    "light": "streetlight",
    "lighting": "streetlight",
    "lantern": "streetlight",
    "bulb": "streetlight",
    "flickering": "faulty",
    "flicker": "faulty",
    "dark": "no_light",
    "outage": "no_light",
    # Waste terminology
    "garbage": "waste",
    "trash": "waste",
    "rubbish": "waste",
    "refuse": "waste",
    "bin": "waste_bin",
    "dumping": "waste_dump",
    "dumped": "waste_dump",
    "overflowing": "overflow",
    "overflowed": "overflow",
    "uncollected": "overflow",
    "accumulated": "overflow",
    "accumulation": "overflow",
    # Drain terminology
    "drainage": "drain",
    "gutter": "drain",
    "sewer": "drain",
    "waterlogging": "drain",
    "flooding": "drain",
    # Location anchors
    "bus stand": "bus_stop",
    "bus stop": "bus_stop",
    "market plaza": "market",
}


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


def normalize_complaint_text(text: str) -> str:
    """
    Normalizes complaint text with lowercase, punctuation removal,
    controlled synonym replacement, and lightweight stemming.
    """
    if not text or pd.isnull(text):
        return ""

    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)

    # Controlled phrase & term normalization
    for synonym, normalized in CONTROLLED_SYNONYMS.items():
        pattern = r"\b" + re.escape(synonym) + r"\b"
        t = re.sub(pattern, normalized, t)

    tokens = t.split()
    stems = []
    for tok in tokens:
        # Lightweight controlled suffix stemming
        if tok.endswith("ing") and len(tok) > 5:
            tok = tok[:-3]
        elif tok.endswith("ed") and len(tok) > 4:
            tok = tok[:-2]
        elif tok.endswith("es") and len(tok) > 4:
            tok = tok[:-2]
        elif tok.endswith("s") and len(tok) > 3 and not tok.endswith("ss"):
            tok = tok[:-1]
        stems.append(tok)

    return " ".join(stems)


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

    # Clean description & normalized text field
    cleaned["description"] = cleaned["description"].fillna("No description").astype(str).str.strip()
    cleaned["clean_description"] = cleaned["description"].apply(normalize_complaint_text)

    # Category normalization
    cleaned["category"] = cleaned["category"].fillna("Unknown").astype(str).str.strip()

    # Safe float conversion for latitude / longitude
    cleaned["latitude"] = pd.to_numeric(cleaned["latitude"], errors="coerce")
    cleaned["longitude"] = pd.to_numeric(cleaned["longitude"], errors="coerce")

    # Image path handling
    cleaned["image_path"] = cleaned["image_path"].fillna("").astype(str).str.strip()

    # Status handling (assign NEW if unassigned or default to Open/Resolved)
    cleaned["status"] = cleaned["status"].fillna("NEW").astype(str).str.strip()
    valid_statuses = {
        "NEW", "OPEN", "POSSIBLE_DUPLICATE", "CONFIRMED_DUPLICATE",
        "UNDERLYING_ISSUE_CREATED", "RESOLVED", "UNRESOLVED", "INVALID_DATA", "REVIEW_REQUIRED",
        "IN PROGRESS"
    }
    # Standardize string format
    cleaned["status_code"] = cleaned["status"].apply(
        lambda s: s.upper().replace(" ", "_") if s.upper().replace(" ", "_") in valid_statuses else s
    )

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

