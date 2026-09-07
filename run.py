import sys
import os
import argparse
import subprocess

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from preprocessing import load_and_clean_all
from baseline import run_baseline
from deduplication import run_deduplication
from database import populate_db
from evaluation import evaluate_models, generate_error_analysis


def run_pipeline():
    print("=" * 60)
    print("MUNICIPALITY COMPLAINT DEDUPLICATION TRACKER PIPELINE")
    print("=" * 60)

    # 1. Preprocessing
    print("\n[Step 1/5] Loading and Preprocessing Data...")
    cmp_clean, iss_clean, evt_clean = load_and_clean_all("data")
    print(f"-> Preprocessed {len(cmp_clean)} complaints, {len(iss_clean)} issues, {len(evt_clean)} resolution events.")

    # 2. Baseline
    print("\n[Step 2/5] Running Baseline Exact Match Model...")
    baseline_res = run_baseline(cmp_clean, "outputs/baseline_results.csv")
    print(f"-> Baseline Duplicate Pairs: {baseline_res['duplicate_pairs']}")
    print(f"-> Baseline Unique Issues: {baseline_res['unique_issues']}")
    print(f"-> Baseline Results saved to outputs/baseline_results.csv")

    # 3. Prototype Deduplication & Consolidation
    print("\n[Step 3/5] Running Hybrid Prototype Deduplication Model...")
    cons_cmp_df, duplicate_pairs, score_records = run_deduplication(cmp_clean, threshold=0.75)
    print(f"-> Prototype Duplicate Pairs Found: {len(duplicate_pairs)}")
    print(f"-> Consolidated Unique Issues: {cons_cmp_df['consolidated_issue_id'].nunique()}")

    # 4. Database Population
    print("\n[Step 4/5] Populating SQLite Database (data/municipality.db)...")
    populate_db(cons_cmp_df, iss_clean, evt_clean, "data/municipality.db")

    # 5. Evaluation & Error Analysis
    print("\n[Step 5/5] Running Model Evaluation and Error Analysis...")
    eval_df = evaluate_models(cmp_clean, baseline_res, cons_cmp_df, duplicate_pairs, "outputs/evaluation_results.csv")
    error_df = generate_error_analysis(cmp_clean, score_records, "outputs/error_analysis.csv")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(eval_df.to_string(index=False))
    print("=" * 60)

    return cmp_clean, baseline_res, cons_cmp_df, eval_df, error_df


def run_tests():
    print("\nRunning pytest edge case test suite...")
    pytest_bin = os.path.join("venv", "bin", "pytest") if os.path.exists(os.path.join("venv", "bin", "pytest")) else "pytest"
    res = subprocess.run([pytest_bin, "-v", "tests/test_edge_cases.py"])
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Municipality Complaint Tracker CLI")
    parser.add_argument("--pipeline-only", action="store_true", help="Run data pipeline and evaluation without starting web server")
    parser.add_argument("--test-only", action="store_true", help="Run pytest test suite only")
    parser.add_argument("--server", action="store_true", help="Run web server only")
    parser.add_argument("--port", type=int, default=8000, help="Port to run FastAPI server on")

    args = parser.parse_args()

    if args.test_only:
        run_tests()
        return

    if not args.server:
        run_pipeline()
        run_tests()

    if not args.pipeline_only:
        print(f"\nStarting FastAPI Web Dashboard on http://127.0.0.1:{args.port} ...")
        import uvicorn
        uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()
