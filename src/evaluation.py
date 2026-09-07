import os
import numpy as np
import pandas as pd


def evaluate_models(
    cmp_df: pd.DataFrame,
    baseline_res: dict,
    prototype_cons_df: pd.DataFrame,
    duplicate_pairs: list[dict],
    output_path: str = "outputs/evaluation_results.csv",
) -> pd.DataFrame:
    """
    Compares Baseline vs Prototype model results against ground truth Issue_ID.
    Calculates Precision, Recall, F1 score and saves to evaluation_results.csv.
    """
    total_complaints = len(cmp_df)
    
    # Extract upper triangular pairs index for Ground Truth comparison
    n = total_complaints
    iu = np.triu_indices(n, k=1)
    
    cids = cmp_df["complaint_id"].to_numpy()
    gt_issues = cmp_df["issue_id"].to_numpy()
    gt_dup_matrix = (gt_issues[iu[0]] == gt_issues[iu[1]]) & (gt_issues[iu[0]] != "")
    
    total_gt_pairs = int(np.sum(gt_dup_matrix))

    # --- BASELINE EVALUATION ---
    base_dup_pairs_count = baseline_res["duplicate_pairs"]
    base_unique_issues = baseline_res["unique_issues"]
    base_consolidation_rate = round(
        (total_complaints - base_unique_issues) / total_complaints, 4
    ) if total_complaints > 0 else 0.0

    # Baseline duplicate pairs matrix
    valid_mask = cmp_df["latitude"].notnull() & cmp_df["longitude"].notnull()
    valid_indices = set(cmp_df[valid_mask].index)
    cats = cmp_df["category"].astype(str).str.lower().to_numpy()
    lats = cmp_df["latitude"].to_numpy()
    lons = cmp_df["longitude"].to_numpy()

    base_pred_matrix = (
        (cats[iu[0]] == cats[iu[1]])
        & (lats[iu[0]] == lats[iu[1]])
        & (lons[iu[0]] == lons[iu[1]])
    )

    tp_base = int(np.sum(base_pred_matrix & gt_dup_matrix))
    fp_base = int(np.sum(base_pred_matrix & ~gt_dup_matrix))
    fn_base = int(np.sum(~base_pred_matrix & gt_dup_matrix))

    prec_base = round(tp_base / (tp_base + fp_base), 4) if (tp_base + fp_base) > 0 else 0.0
    rec_base = round(tp_base / (tp_base + fn_base), 4) if (tp_base + fn_base) > 0 else 0.0
    f1_base = round(2 * prec_base * rec_base / (prec_base + rec_base), 4) if (prec_base + rec_base) > 0 else 0.0

    # --- PROTOTYPE EVALUATION ---
    proto_dup_pairs_count = len(duplicate_pairs)
    proto_unique_issues = prototype_cons_df["consolidated_issue_id"].nunique()
    proto_consolidation_rate = round(
        (total_complaints - proto_unique_issues) / total_complaints, 4
    ) if total_complaints > 0 else 0.0

    # Map predicted pairs
    pred_pair_set = set((p["complaint_a"], p["complaint_b"]) for p in duplicate_pairs)
    pred_pair_set.update((p["complaint_b"], p["complaint_a"]) for p in duplicate_pairs)

    proto_pred_matrix = np.array(
        [(cids[i], cids[j]) in pred_pair_set for i, j in zip(iu[0], iu[1])]
    )

    tp_proto = int(np.sum(proto_pred_matrix & gt_dup_matrix))
    fp_proto = int(np.sum(proto_pred_matrix & ~gt_dup_matrix))
    fn_proto = int(np.sum(~proto_pred_matrix & gt_dup_matrix))

    prec_proto = round(tp_proto / (tp_proto + fp_proto), 4) if (tp_proto + fp_proto) > 0 else 0.0
    rec_proto = round(tp_proto / (tp_proto + fn_proto), 4) if (tp_proto + fn_proto) > 0 else 0.0
    f1_proto = round(2 * prec_proto * rec_proto / (prec_proto + rec_proto), 4) if (prec_proto + rec_proto) > 0 else 0.0

    eval_data = {
        "Metric": [
            "Total Complaints",
            "Duplicate Pairs Identified",
            "Unique Underlying Issues",
            "Duplicate Consolidation Rate",
            "Precision",
            "Recall",
            "F1 Score",
        ],
        "Baseline Model": [
            total_complaints,
            base_dup_pairs_count,
            base_unique_issues,
            base_consolidation_rate,
            prec_base,
            rec_base,
            f1_base,
        ],
        "Prototype Model": [
            total_complaints,
            proto_dup_pairs_count,
            proto_unique_issues,
            proto_consolidation_rate,
            prec_proto,
            rec_proto,
            f1_proto,
        ],
    }

    eval_df = pd.DataFrame(eval_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    eval_df.to_csv(output_path, index=False)

    print(f"Evaluation report exported to {output_path}")
    return eval_df


def generate_error_analysis(
    cmp_df: pd.DataFrame,
    score_records: list[dict],
    output_path: str = "outputs/error_analysis.csv",
) -> pd.DataFrame:
    """
    Generates detailed error analysis CSV identifying False Positives and False Negatives.
    """
    # Create lookup map for ground truth issue IDs
    gt_map = cmp_df.set_index("complaint_id")["issue_id"].to_dict()

    analysis_rows = []

    for rec in score_records:
        cid_a, cid_b = rec["complaint_a"], rec["complaint_b"]
        gt_a, gt_b = gt_map.get(cid_a, ""), gt_map.get(cid_b, "")
        
        expected_dup = (gt_a == gt_b) and (gt_a != "")
        pred_dup = rec["predicted_duplicate"]

        # Filter for errors or interesting edge pairs to keep CSV concise and informative
        if expected_dup != pred_dup or rec["duplicate_score"] >= 0.60:
            analysis_rows.append(
                {
                    "Complaint_A": cid_a,
                    "Complaint_B": cid_b,
                    "Expected_Result": "DUPLICATE" if expected_dup else "NOT DUPLICATE",
                    "Predicted_Result": "DUPLICATE" if pred_dup else "NOT DUPLICATE",
                    "Text_Similarity": rec["text_similarity"],
                    "Location_Distance_Meters": rec["location_distance"],
                    "Category_Match": rec["category_match"],
                    "Duplicate_Score": rec["duplicate_score"],
                    "Error_Category": (
                        "False Positive" if pred_dup and not expected_dup
                        else "False Negative" if not pred_dup and expected_dup
                        else "Correct Match"
                    ),
                }
            )

    error_df = pd.DataFrame(analysis_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    error_df.to_csv(output_path, index=False)

    print(f"Error analysis exported to {output_path} with {len(error_df)} analyzed pairs.")
    return error_df


if __name__ == "__main__":
    from preprocessing import load_and_clean_all
    from baseline import run_baseline
    from deduplication import run_deduplication

    cmp_clean, _, _ = load_and_clean_all()
    base_res = run_baseline(cmp_clean)
    cons_df, dup_pairs, score_records = run_deduplication(cmp_clean)

    eval_df = evaluate_models(cmp_clean, base_res, cons_df, dup_pairs)
    error_df = generate_error_analysis(cmp_clean, score_records)
    print(eval_df)
