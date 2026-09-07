import os
import time
import pandas as pd


def run_baseline(cmp_df: pd.DataFrame, output_path: str = "outputs/baseline_results.csv") -> dict:
    """
    Runs baseline deduplication using exact match on:
    Same Category AND Exact Latitude AND Exact Longitude.
    """
    start_time = time.time()

    total_complaints = len(cmp_df)

    # Clean subset for grouping (ignoring missing lat/lon)
    valid_mask = cmp_df["latitude"].notnull() & cmp_df["longitude"].notnull()
    valid_df = cmp_df[valid_mask].copy()

    # Group by category, exact latitude, exact longitude
    grouped = valid_df.groupby(["category", "latitude", "longitude"])

    duplicate_pairs = 0
    unique_issues_count = 0
    duplicate_pair_list = []

    for _, group in grouped:
        unique_issues_count += 1
        n = len(group)
        if n > 1:
            # Add all unique pairs in this group
            complaint_ids = group["complaint_id"].tolist()
            duplicate_pairs += (n * (n - 1)) // 2
            for i in range(n):
                for j in range(i + 1, n):
                    duplicate_pair_list.append((complaint_ids[i], complaint_ids[j]))

    # Include complaints with missing location as individual issues
    missing_loc_count = total_complaints - len(valid_df)
    unique_issues_count += missing_loc_count

    processing_time = round(time.time() - start_time, 4)

    duplicate_rate = round(duplicate_pairs / total_complaints, 4) if total_complaints > 0 else 0.0

    results = {
        "metric": [
            "total_complaints",
            "duplicate_pairs",
            "unique_issues",
            "duplicate_rate",
            "processing_time_seconds",
        ],
        "value": [
            total_complaints,
            duplicate_pairs,
            unique_issues_count,
            duplicate_rate,
            processing_time,
        ],
    }

    results_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)

    return {
        "total_complaints": total_complaints,
        "duplicate_pairs": duplicate_pairs,
        "unique_issues": unique_issues_count,
        "duplicate_rate": duplicate_rate,
        "processing_time": processing_time,
        "duplicate_pair_list": duplicate_pair_list,
    }


if __name__ == "__main__":
    from preprocessing import load_and_clean_all
    cmp_clean, _, _ = load_and_clean_all()
    res = run_baseline(cmp_clean)
    print("Baseline Execution Completed:")
    print(res)
