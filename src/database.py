import os
import sqlite3
import pandas as pd


DB_PATH = "data/municipality.db"


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(db_path, timeout=30.0)
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
    Ensures 1-to-many relationship: One Issue -> Many Complaints -> Resolution Events.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)

    cmp_to_insert = cmp_df.copy()
    if "consolidated_issue_id" in cmp_to_insert.columns:
        cmp_to_insert["issue_id"] = cmp_to_insert["consolidated_issue_id"]
        cmp_to_insert = cmp_to_insert.drop(columns=["consolidated_issue_id"])

    # Aggregate underlying issues dynamically from complaints
    issue_groups = cmp_to_insert.groupby("issue_id")
    issue_records = []

    for issue_id, group in issue_groups:
        first_row = group.iloc[0]
        cat = first_row["category"]
        lat = first_row["latitude"]
        lon = first_row["longitude"]
        desc = first_row["description"]
        cmp_count = len(group)

        statuses = group["status"].astype(str).str.lower().tolist()
        if all(s in ["resolved", "closed"] for s in statuses):
            status = "Resolved"
        elif any(s in ["in progress", "assigned"] for s in statuses):
            status = "In Progress"
        else:
            status = "Open"

        assigned_team = "Municipal Response Team"
        if cat.lower() == "road":
            assigned_team = "Road Maintenance Team"
        elif cat.lower() == "lighting":
            assigned_team = "Electrical & Lighting Division"
        elif cat.lower() == "waste":
            assigned_team = "Sanitation & Waste Management"

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
    constructed_issues_df.to_sql("issues", conn, if_exists="append", index=False)

    # Standardize complaints columns for DB
    db_cmp_cols = ["complaint_id", "description", "category", "latitude", "longitude", "image_path", "created_date", "status", "issue_id"]
    for col in db_cmp_cols:
        if col not in cmp_to_insert.columns:
            cmp_to_insert[col] = None

    cmp_to_insert[db_cmp_cols].to_sql("complaints", conn, if_exists="append", index=False)

    # Standardize resolution_events columns and map issue_id formatting
    evt_to_insert = evt_df.copy()
    if "issue_id" in evt_to_insert.columns:
        # Standardize ISSUE0001 / I0001 -> ISS-0001 format
        def reformat_issue_id(val):
            val_str = str(val or "").strip()
            num_part = "".join(filter(str.isdigit, val_str))
            if num_part:
                return f"ISS-{int(num_part):04d}"
            return val_str

        evt_to_insert["issue_id"] = evt_to_insert["issue_id"].apply(reformat_issue_id)

    db_evt_cols = ["event_id", "issue_id", "event_type", "event_date", "assigned_team", "notes"]
    for col in db_evt_cols:
        if col not in evt_to_insert.columns:
            evt_to_insert[col] = None

    evt_to_insert[db_evt_cols].to_sql("resolution_events", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    print(f"Database populated at {db_path} with {len(cmp_to_insert)} complaints and {len(constructed_issues_df)} underlying issues.")


def get_issue_with_events(issue_id: str, db_path: str = DB_PATH) -> dict:
    """Retrieves an issue record, its linked complaints, and resolution event history."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    issue_row = cursor.execute("SELECT * FROM issues WHERE issue_id = ?", (issue_id,)).fetchone()
    if not issue_row:
        conn.close()
        return None

    issue = dict(issue_row)
    cmp_rows = cursor.execute("SELECT * FROM complaints WHERE issue_id = ? ORDER BY complaint_id ASC", (issue_id,)).fetchall()
    evt_rows = cursor.execute("SELECT * FROM resolution_events WHERE issue_id = ? ORDER BY event_date ASC", (issue_id,)).fetchall()

    conn.close()

    issue["linked_complaints"] = [dict(r) for r in cmp_rows]
    issue["resolution_events"] = [dict(r) for r in evt_rows]
    return issue


def confirm_duplicate_linkage(complaint_id: str, target_issue_id: str, db_path: str = DB_PATH) -> bool:
    """Updates complaint issue_id and sets status to CONFIRMED_DUPLICATE."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE complaints SET issue_id = ?, status = 'CONFIRMED_DUPLICATE' WHERE complaint_id = ?",
        (target_issue_id, complaint_id),
    )
    # Recalculate complaint count
    cursor.execute(
        "UPDATE issues SET complaint_count = (SELECT COUNT(*) FROM complaints WHERE issue_id = ?) WHERE issue_id = ?",
        (target_issue_id, target_issue_id),
    )
    conn.commit()
    conn.close()
    return True


if __name__ == "__main__":
    from preprocessing import load_and_clean_all
    from deduplication import run_deduplication

    cmp_clean, iss_clean, evt_clean = load_and_clean_all()
    cons_df, _, _ = run_deduplication(cmp_clean)
    populate_db(cons_df, iss_clean, evt_clean)

