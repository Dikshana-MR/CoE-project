import os
import sqlite3
from typing import Optional
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Municipality Complaint Deduplication Tracker")

# Setup Jinja2 templates directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "municipality.db")


def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Please run data pipeline first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/api/stats")
def get_stats():
    """Calculates live stats from SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    total_complaints = cursor.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    total_issues = cursor.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    
    # Duplicate complaints = Complaints linked to issues that have > 1 complaint
    dup_complaints = cursor.execute(
        """
        SELECT COUNT(*) FROM complaints 
        WHERE issue_id IN (
            SELECT issue_id FROM complaints GROUP BY issue_id HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    open_issues = cursor.execute("SELECT COUNT(*) FROM issues WHERE LOWER(status) != 'resolved'").fetchone()[0]
    resolved_issues = cursor.execute("SELECT COUNT(*) FROM issues WHERE LOWER(status) = 'resolved'").fetchone()[0]

    consolidation_rate = (
        round((total_complaints - total_issues) / total_complaints, 4)
        if total_complaints > 0
        else 0.0
    )

    conn.close()

    return {
        "total_complaints": total_complaints,
        "total_underlying_issues": total_issues,
        "duplicate_complaints": dup_complaints,
        "open_issues": open_issues,
        "resolved_issues": resolved_issues,
        "duplicate_consolidation_rate": consolidation_rate,
        "duplicate_consolidation_rate_pct": f"{consolidation_rate * 100:.1f}%",
    }


@app.get("/", response_class=HTMLResponse)
def dashboard_view(request: Request):
    """Renders basic working dashboard."""
    stats = get_stats()
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"stats": stats})


@app.get("/issues", response_class=HTMLResponse)
def issues_view(request: Request, category: Optional[str] = None):
    """Renders Issue View listing all issues and linked complaints."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if category:
        issues_rows = cursor.execute(
            "SELECT * FROM issues WHERE LOWER(category) = LOWER(?) ORDER BY issue_id ASC", (category,)
        ).fetchall()
    else:
        issues_rows = cursor.execute("SELECT * FROM issues ORDER BY issue_id ASC").fetchall()

    issues = [dict(row) for row in issues_rows]

    # Fetch all complaints for linked display
    complaints_rows = cursor.execute("SELECT * FROM complaints ORDER BY complaint_id ASC").fetchall()
    conn.close()

    # Map issue_id -> list of complaints
    issue_cmp_map = {}
    for cmp_row in complaints_rows:
        cmp_dict = dict(cmp_row)
        iid = cmp_dict.get("issue_id", "")
        issue_cmp_map.setdefault(iid, []).append(cmp_dict)

    for issue in issues:
        iid = issue["issue_id"]
        issue["linked_complaints"] = issue_cmp_map.get(iid, [])

    return templates.TemplateResponse(request=request, name="issues.html", context={"issues": issues, "selected_category": category})


@app.get("/issues/{issue_id}", response_class=HTMLResponse)
def single_issue_view(request: Request, issue_id: str):
    """Renders details for a single issue and all its linked complaints."""
    conn = get_db_connection()
    cursor = conn.cursor()

    issue_row = cursor.execute("SELECT * FROM issues WHERE issue_id = ?", (issue_id,)).fetchone()
    if not issue_row:
        conn.close()
        return HTMLResponse("<h1>Issue Not Found</h1>", status_code=404)

    issue = dict(issue_row)
    complaints_rows = cursor.execute("SELECT * FROM complaints WHERE issue_id = ? ORDER BY complaint_id ASC", (issue_id,)).fetchall()
    conn.close()

    issue["linked_complaints"] = [dict(row) for row in complaints_rows]

    return templates.TemplateResponse(request=request, name="issue_detail.html", context={"issue": issue})


@app.get("/complaints", response_class=HTMLResponse)
def complaints_view(request: Request, q: Optional[str] = Query(None)):
    """Renders Basic Complaint View with search capability."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if q:
        search_pattern = f"%{q.strip()}%"
        rows = cursor.execute(
            """
            SELECT * FROM complaints 
            WHERE complaint_id LIKE ? OR description LIKE ? OR category LIKE ? OR issue_id LIKE ?
            ORDER BY complaint_id ASC
            """,
            (search_pattern, search_pattern, search_pattern, search_pattern),
        ).fetchall()
    else:
        rows = cursor.execute("SELECT * FROM complaints ORDER BY complaint_id ASC LIMIT 200").fetchall()

    conn.close()
    complaints = [dict(row) for row in rows]

    return templates.TemplateResponse(
        request=request, name="complaints.html", context={"complaints": complaints, "search_query": q or ""}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
