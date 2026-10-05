import os
import numpy as np
import pandas as pd
from deduplication import UnionFind, DEFAULT_WEIGHTS, generate_candidate_pairs, haversine_distance, get_location_similarity, RAPIDFUZZ_AVAILABLE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from rapidfuzz import fuzz
except ImportError:
    pass


def run_threshold_analysis(
    cmp_df: pd.DataFrame,
    thresholds: list[float] = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    output_path: str = "outputs/threshold_analysis.csv",
) -> pd.DataFrame:
    """
    Runs systematic threshold analysis across specified threshold values.
    Calculates TP, FP, FN, Precision, Recall, F1 Score, Duplicate Pairs Count,
    Unique Issues, and Consolidation Rate for each threshold.
    Saves results to outputs/threshold_analysis.csv.
    """
    total_complaints = len(cmp_df)
    cids = cmp_df["complaint_id"].tolist()
    n = total_complaints
    iu0, iu1 = np.triu_indices(n, k=1)

    gt_issues = cmp_df["issue_id"].to_numpy()
    gt_mask = (gt_issues[iu0] == gt_issues[iu1]) & (gt_issues[iu0] != "")

    # Scalable Candidate Generation
    candidate_pairs = generate_candidate_pairs(cmp_df, max_distance_meters=350.0)

    # Pre-compute features for candidates
    norm_descs = cmp_df["clean_description"].tolist() if "clean_description" in cmp_df.columns else cmp_df["description"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(norm_descs)

    lats = cmp_df["latitude"].to_numpy()
    lons = cmp_df["longitude"].to_numpy()
    dates = pd.to_datetime(cmp_df["created_date"], errors="coerce").to_numpy()
    imgs = cmp_df["image_path"].fillna("").to_numpy()

    cand_indices = []
    cand_scores = []
    cand_gt = []

    pair_map = {(iu0[idx], iu1[idx]): idx for idx in range(len(iu0))}

    w = DEFAULT_WEIGHTS

    for i, j in candidate_pairs:
        idx = pair_map.get((i, j))
        if idx is None:
            continue

        norm1, norm2 = norm_descs[i], norm_descs[j]
        try:
            tfidf_sim = float(cosine_similarity(tfidf_matrix[i : i + 1], tfidf_matrix[j : j + 1])[0][0])
        except Exception:
            tfidf_sim = 0.0

        if RAPIDFUZZ_AVAILABLE:
            fz_sort = fuzz.token_sort_ratio(norm1, norm2) / 100.0
            fz_rat = fuzz.ratio(norm1, norm2) / 100.0
            fuzzy_sim = max(fz_sort, fz_rat)
            text_sim = 0.50 * tfidf_sim + 0.50 * fuzzy_sim
        else:
            text_sim = tfidf_sim

        dist = haversine_distance(lats[i], lons[i], lats[j], lons[j])
        loc_sim = get_location_similarity(dist)

        d1, d2 = dates[i], dates[j]
        if pd.notnull(d1) and pd.notnull(d2):
            diff_days = abs((d1 - d2) / np.timedelta64(1, "D"))
            temp_sim = 1.0 if diff_days <= 3 else (0.8 if diff_days <= 7 else (0.5 if diff_days <= 14 else 0.0))
        else:
            temp_sim = 0.0

        img1, img2 = imgs[i], imgs[j]
        img_sim = 1.0 if (img1 != "" and img1.lower() == img2.lower()) else 0.0

        cat_sim = 1.0

        score = (
            w.get("w_text", 0.35) * text_sim
            + w.get("w_location", 0.40) * loc_sim
            + w.get("w_category", 0.15) * cat_sim
            + w.get("w_temporal", 0.05) * temp_sim
            + w.get("w_image", 0.05) * img_sim
        )

        cand_indices.append((i, j))
        cand_scores.append(round(score, 4))
        cand_gt.append(gt_mask[idx])

    rows = []
    for t in thresholds:
        dsu = UnionFind(cids)
        tp, fp, fn = 0, 0, 0
        dup_count = 0

        for pair_idx, (i, j) in enumerate(cand_indices):
            s = cand_scores[pair_idx]
            is_gt = cand_gt[pair_idx]
            pred = s >= t

            if pred:
                dup_count += 1
                dsu.union(cids[i], cids[j])
                if is_gt:
                    tp += 1
                else:
                    fp += 1

        total_gt = int(np.sum(gt_mask))
        fn = total_gt - tp

        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        unique_issues = len(set(dsu.find(c) for c in cids))
        cons_rate = round((total_complaints - unique_issues) / total_complaints, 4) if total_complaints > 0 else 0.0

        rows.append(
            {
                "Threshold": round(t, 2),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "Duplicate_Pairs": dup_count,
                "Unique_Issues": unique_issues,
                "Consolidation_Rate": cons_rate,
            }
        )

    analysis_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    analysis_df.to_csv(output_path, index=False)
    print(f"Threshold analysis saved to {output_path}")
    return analysis_df


def evaluate_models(
    cmp_df: pd.DataFrame,
    baseline_res: dict,
    prototype_cons_df: pd.DataFrame,
    duplicate_pairs: list[dict],
    output_path: str = "outputs/evaluation_results.csv",
) -> pd.DataFrame:
    """
    Compares Baseline vs Original Prototype vs Improved Prototype model results against ground truth Issue_ID.
    Calculates Precision, Recall, F1 score and saves to evaluation_results.csv.
    """
    total_complaints = len(cmp_df)
    n = total_complaints
    iu = np.triu_indices(n, k=1)

    cids = cmp_df["complaint_id"].to_numpy()
    gt_issues = cmp_df["issue_id"].to_numpy()
    gt_dup_matrix = (gt_issues[iu[0]] == gt_issues[iu[1]]) & (gt_issues[iu[0]] != "")

    # --- BASELINE EVALUATION ---
    base_dup_pairs_count = baseline_res["duplicate_pairs"]
    base_unique_issues = baseline_res["unique_issues"]
    base_consolidation_rate = round(
        (total_complaints - base_unique_issues) / total_complaints, 4
    ) if total_complaints > 0 else 0.0

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

    # --- IMPROVED PROTOTYPE EVALUATION ---
    proto_dup_pairs_count = len(duplicate_pairs)
    proto_unique_issues = prototype_cons_df["consolidated_issue_id"].nunique()
    proto_consolidation_rate = round(
        (total_complaints - proto_unique_issues) / total_complaints, 4
    ) if total_complaints > 0 else 0.0

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
            "False Positives",
            "False Negatives",
        ],
        "Baseline Model": [
            total_complaints,
            base_dup_pairs_count,
            base_unique_issues,
            base_consolidation_rate,
            prec_base,
            rec_base,
            f1_base,
            fp_base,
            fn_base,
        ],
        "Improved Prototype Model": [
            total_complaints,
            proto_dup_pairs_count,
            proto_unique_issues,
            proto_consolidation_rate,
            prec_proto,
            rec_proto,
            f1_proto,
            fp_proto,
            fn_proto,
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
    gt_map = cmp_df.set_index("complaint_id")["issue_id"].to_dict()

    analysis_rows = []

    for rec in score_records:
        cid_a, cid_b = rec["complaint_a"], rec["complaint_b"]
        gt_a, gt_b = gt_map.get(cid_a, ""), gt_map.get(cid_b, "")

        expected_dup = (gt_a == gt_b) and (gt_a != "")
        pred_dup = rec["predicted_duplicate"]

        # Keep errors and high-score pairs for insightful review
        if expected_dup != pred_dup or rec["duplicate_score"] >= 0.50:
            analysis_rows.append(
                {
                    "Complaint_A": cid_a,
                    "Complaint_B": cid_b,
                    "Expected_Result": "DUPLICATE" if expected_dup else "NOT DUPLICATE",
                    "Predicted_Result": "DUPLICATE" if pred_dup else "NOT DUPLICATE",
                    "Text_Similarity": rec["text_similarity"],
                    "Location_Distance_Meters": rec["location_distance"],
                    "Category_Match": rec["category_match"],
                    "Temporal_Similarity": rec.get("temporal_similarity", 0.0),
                    "Image_Similarity": rec.get("image_similarity", 0.0),
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

    thresh_df = run_threshold_analysis(cmp_clean)
    eval_df = evaluate_models(cmp_clean, base_res, cons_df, dup_pairs)
    error_df = generate_error_analysis(cmp_clean, score_records)
    print(eval_df)

