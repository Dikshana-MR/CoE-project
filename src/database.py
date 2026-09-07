import os
import sqlite3
import pandas as pd


DB_PATH = "data/municipality.db"


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH):
    """Creates tables: complaints, issues, resolution_events in SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop existing tables to refresh clean schema
    cursor.execute("DROP TABLE IF EXISTS complaints;")
    cursor.execute("DROP TABLE IF EXISTS issues;")
    cursor.execute("DROP TABLE IF EXISTS resolution_events;")

    # Issues table
    cursor.execute(
        """
        CREATE TABLE issues (
            issue_id TEXT PRIMARY KEY,
            category TEXT,
            latitude REAL,
            longitude REAL,
            issue_description TEXT,
            complaint_count INTEGER DEFAULT 1,
            status TEXT DEFAULT 'Open',
            assigned_to TEXT
        );
        """
    )

    # Complaints table
    cursor.execute(
        """
        CREATE TABLE complaints (
            complaint_id TEXT PRIMARY KEY,
            description TEXT,
            category TEXT,
            latitude REAL,
            longitude REAL,
            image_path TEXT,
            created_date TEXT,
            status TEXT,
            issue_id TEXT,
            FOREIGN KEY (issue_id) REFERENCES issues (issue_id)
        );
        """
    )

    # Resolution events table
    cursor.execute(
        """
        CREATE TABLE resolution_events (
            event_id TEXT PRIMARY KEY,
            issue_id TEXT,
            event_type TEXT,
            event_date TEXT,
            assigned_team TEXT,
            notes TEXT,
            FOREIGN KEY (issue_id) REFERENCES issues (issue_id)
        );
        """
    )

    conn.commit()
    conn.close()


def populate_db(
    cmp_df: pd.DataFrame,
    iss_df: pd.DataFrame,
    evt_df: pd.DataFrame,
    db_path: str = DB_PATH,
):
    """
    Populates SQLite database with cleaned DataFrames.
    Ensures 1-to-many relationship: One Issue -> Many Complaints.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)

    # 1. Build issues table records from consolidated issue_ids
    # If cmp_df has 'consolidated_issue_id', update issue_id
    cmp_to_insert = cmp_df.copy()
    if "consolidated_issue_id" in cmp_to_insert.columns:
        cmp_to_insert["issue_id"] = cmp_to_insert["consolidated_issue_id"]
        cmp_to_insert = cmp_to_insert.drop(columns=["consolidated_issue_id"])

    # Aggregate underlying issues dynamically from complaints if needed
    issue_groups = cmp_to_insert.groupby("issue_id")
    issue_records = []

    for issue_id, group in issue_groups:
        first_row = group.iloc[0]
        cat = first_row["category"]
        lat = first_row["latitude"]
        lon = first_row["longitude"]
        desc = first_row["description"]
        cmp_count = len(group)

        # Determine status (if all complaints in group are resolved -> Resolved, else In Progress / Open)
        statuses = group["status"].astype(str).str.lower().tolist()
        if all(s == "resolved" for s in statuses):
            status = "Resolved"
        elif any(s in ["in progress", "assigned"] for s in statuses):
            status = "In Progress"
        else:
            status = "Open"

        # Check if this issue existed in iss_df for assigned_to
        assigned_team = "Municipal Response Team"
        if "issue_id" in iss_df.columns and "assigned_to" in iss_df.columns:
            match = iss_df[iss_df["issue_id"] == issue_id]
            if not match.empty and pd.notnull(match.iloc[0]["assigned_to"]):
                assigned_team = match.iloc[0]["assigned_to"]

        issue_records.append(
            {
                "issue_id": issue_id,
                "category": cat,
                "latitude": lat,
                "longitude": lon,
                "issue_description": desc,
                "complaint_count": cmp_count,
                "status": status,
                "assigned_to": assigned_team,
            }
        )

    constructed_issues_df = pd.DataFrame(issue_records)

    # Insert into database
    constructed_issues_df.to_sql("issues", conn, if_exists="append", index=False)
    
    # Standardize complaints columns for DB
    db_cmp_cols = ["complaint_id", "description", "category", "latitude", "longitude", "image_path", "created_date", "status", "issue_id"]
    for col in db_cmp_cols:
        if col not in cmp_to_insert.columns:
            cmp_to_insert[col] = None

    cmp_to_insert[db_cmp_cols].to_sql("complaints", conn, if_exists="append", index=False)

    # Standardize resolution_events columns for DB
    db_evt_cols = ["event_id", "issue_id", "event_type", "event_date", "assigned_team", "notes"]
    evt_to_insert = evt_df.copy()
    for col in db_evt_cols:
        if col not in evt_to_insert.columns:
            evt_to_insert[col] = None

    evt_to_insert[db_evt_cols].to_sql("resolution_events", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    print(f"Database populated at {db_path} with {len(cmp_to_insert)} complaints and {len(constructed_issues_df)} underlying issues.")


if __name__ == "__main__":
    from preprocessing import load_and_clean_all
    from deduplication import run_deduplication

    cmp_clean, iss_clean, evt_clean = load_and_clean_all()
    cons_df, _, _ = run_deduplication(cmp_clean)
    populate_db(cons_df, iss_clean, evt_clean)
