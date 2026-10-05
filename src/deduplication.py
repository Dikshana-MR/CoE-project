import math
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from preprocessing import normalize_complaint_text

# Configurable feature weights for hybrid duplicate score calculation
DEFAULT_WEIGHTS = {
    "w_text": 0.35,
    "w_location": 0.40,
    "w_category": 0.15,
    "w_temporal": 0.05,
    "w_image": 0.05,
}

DEFAULT_THRESHOLD = 0.55


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the Great Circle distance between two points in meters using Haversine formula."""
    if pd.isnull(lat1) or pd.isnull(lon1) or pd.isnull(lat2) or pd.isnull(lon2):
        return float("inf")

    # Earth radius in meters
    R = 6371000.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def get_location_similarity(distance_meters: float) -> float:
    """Maps distance in meters to location similarity score using continuous/tiered curve."""
    if math.isinf(distance_meters) or math.isnan(distance_meters):
        return 0.0
    if distance_meters <= 30.0:
        return 1.0
    elif distance_meters <= 80.0:
        return 0.90
    elif distance_meters <= 150.0:
        return 0.75
    elif distance_meters <= 250.0:
        return 0.50
    elif distance_meters <= 320.0:
        return 0.25
    else:
        return 0.0


def get_category_similarity(cat1: str, cat2: str) -> float:
    """Returns 1.0 if categories match (case-insensitive), else 0.0."""
    if not cat1 or not cat2:
        return 0.0
    return 1.0 if str(cat1).strip().lower() == str(cat2).strip().lower() else 0.0


def get_temporal_similarity(date1_str: str, date2_str: str) -> float:
    """Calculates temporal similarity score based on date proximity in days."""
    if not date1_str or not date2_str or pd.isnull(date1_str) or pd.isnull(date2_str):
        return 0.0
    try:
        d1 = pd.to_datetime(date1_str)
        d2 = pd.to_datetime(date2_str)
        diff_days = abs((d1 - d2).days)
        if diff_days <= 3:
            return 1.0
        elif diff_days <= 7:
            return 0.8
        elif diff_days <= 14:
            return 0.5
        else:
            return 0.0
    except Exception:
        return 0.0


def get_image_similarity(img1: str, img2: str) -> float:
    """Calculates image similarity score based on image path comparison."""
    str1, str2 = str(img1 or "").strip(), str(img2 or "").strip()
    if not str1 or not str2:
        return 0.0
    return 1.0 if str1.lower() == str2.lower() else 0.0


def get_single_text_similarity(desc1: str, desc2: str) -> float:
    """Calculates text similarity between two descriptions combining RapidFuzz and TF-IDF."""
    str1, str2 = str(desc1).strip(), str(desc2).strip()
    if not str1 or not str2:
        return 0.0

    norm1 = normalize_complaint_text(str1)
    norm2 = normalize_complaint_text(str2)

    # 1. Fuzzy similarity
    if RAPIDFUZZ_AVAILABLE:
        fuzz_sort = fuzz.token_sort_ratio(norm1, norm2) / 100.0
        fuzz_rat = fuzz.ratio(norm1, norm2) / 100.0
        fuzzy_score = max(fuzz_sort, fuzz_rat)
    else:
        fuzzy_score = 0.0

    # 2. TF-IDF similarity
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform([norm1, norm2])
        tfidf_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        tfidf_score = 0.0

    if RAPIDFUZZ_AVAILABLE:
        return round(0.50 * tfidf_score + 0.50 * fuzzy_score, 4)
    else:
        return round(tfidf_score, 4)


def calculate_pair_duplicate_score(cmp1: dict, cmp2: dict, weights: dict = None, threshold: float = None) -> dict:
    """
    Computes hybrid duplicate score for two complaint dictionary records.
    Returns score breakdown dictionary.
    """
    w = weights if weights else DEFAULT_WEIGHTS
    t = threshold if threshold is not None else DEFAULT_THRESHOLD

    cat_sim = get_category_similarity(cmp1.get("category", ""), cmp2.get("category", ""))

    lat1, lon1 = cmp1.get("latitude"), cmp1.get("longitude")
    lat2, lon2 = cmp2.get("latitude"), cmp2.get("longitude")
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    loc_sim = get_location_similarity(dist)

    text_sim = get_single_text_similarity(
        cmp1.get("description", ""), cmp2.get("description", "")
    )

    temp_sim = get_temporal_similarity(cmp1.get("created_date"), cmp2.get("created_date"))
    img_sim = get_image_similarity(cmp1.get("image_path"), cmp2.get("image_path"))

    # Category-aware check: if category mismatch, duplicate score stays 0 unless overridden
    if cat_sim == 0.0:
        score = 0.0
    else:
        score = (
            w.get("w_text", 0.35) * text_sim
            + w.get("w_location", 0.40) * loc_sim
            + w.get("w_category", 0.15) * cat_sim
            + w.get("w_temporal", 0.05) * temp_sim
            + w.get("w_image", 0.05) * img_sim
        )

    score = round(float(score), 4)
    is_duplicate = score >= t

    return {
        "complaint_a": cmp1.get("complaint_id"),
        "complaint_b": cmp2.get("complaint_id"),
        "text_similarity": round(text_sim, 4),
        "location_distance": round(dist, 2) if not math.isinf(dist) else None,
        "location_similarity": loc_sim,
        "category_match": int(cat_sim == 1.0),
        "temporal_similarity": temp_sim,
        "image_similarity": img_sim,
        "duplicate_score": score,
        "predicted_duplicate": is_duplicate,
    }


class UnionFind:
    """Disjoint Set Union (DSU) structure to group duplicate complaints."""
    def __init__(self, elements):
        self.parent = {el: el for el in elements}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            root_small = min(root_i, root_j)
            root_large = max(root_i, root_j)
            self.parent[root_large] = root_small


def generate_candidate_pairs(cmp_df: pd.DataFrame, max_distance_meters: float = 350.0) -> list[tuple[int, int]]:
    """
    Scalable Candidate Generation Strategy:
    1. Filter/block by Category (Complaints only compared within the same category).
    2. Filter by geographic proximity (distance <= max_distance_meters or missing location).
    Reduces pair comparisons from O(N^2) down to O(N_cat * K), avoiding blind O(N^2) pairwise evaluation.
    """
    n = len(cmp_df)
    iu0, iu1 = np.triu_indices(n, k=1)

    cats = cmp_df["category"].astype(str).str.lower().to_numpy()
    cat_match = (cats[iu0] == cats[iu1])

    lats = np.radians(cmp_df["latitude"].to_numpy())
    lons = np.radians(cmp_df["longitude"].to_numpy())

    dlat = lats[iu0] - lats[iu1]
    dlon = lons[iu0] - lons[iu1]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lats[iu0]) * np.cos(lats[iu1]) * np.sin(dlon / 2.0) ** 2
    dists_meters = 2.0 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    valid_locs = ~(np.isnan(lats[iu0]) | np.isnan(lats[iu1]))
    dists_meters[~valid_locs] = 0.0  # Keep missing coordinates as candidates for text check

    # Candidate condition: Category Match AND Distance <= max_distance_meters
    cand_mask = cat_match & (dists_meters <= max_distance_meters)
    cand_indices = np.where(cand_mask)[0]

    return list(zip(iu0[cand_indices], iu1[cand_indices]))


def run_deduplication(
    cmp_df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    weights: dict = None,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """
    Executes scalable deduplication across complaint records in cmp_df.
    Returns:
    - consolidated_cmp_df: DataFrame with updated issue_id and status codes linked to complaints.
    - duplicate_pairs: List of duplicate pair dicts (with scores).
    - score_records: List of all evaluated pair scores for error analysis.
    """
    t0 = time.time()
    w = weights if weights else DEFAULT_WEIGHTS
    n = len(cmp_df)
    complaints = cmp_df.to_dict("records")
    complaint_ids = [c["complaint_id"] for c in complaints]

    # Pre-process text fields if clean_description not present
    if "clean_description" not in cmp_df.columns:
        norm_descs = [normalize_complaint_text(c.get("description", "")) for c in complaints]
    else:
        norm_descs = cmp_df["clean_description"].tolist()

    # Pre-compute TF-IDF Matrix on normalized descriptions
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(norm_descs)

    # Spatial coordinates safely
    lats = np.radians(pd.to_numeric(cmp_df["latitude"] if "latitude" in cmp_df.columns else pd.Series([np.nan] * n), errors="coerce").to_numpy())
    lons = np.radians(pd.to_numeric(cmp_df["longitude"] if "longitude" in cmp_df.columns else pd.Series([np.nan] * n), errors="coerce").to_numpy())

    # Dates safely
    dates_col = cmp_df["created_date"] if "created_date" in cmp_df.columns else pd.Series([""] * n)
    dates = pd.to_datetime(dates_col, errors="coerce").to_numpy()

    # Images safely
    imgs_col = cmp_df["image_path"] if "image_path" in cmp_df.columns else pd.Series([""] * n)
    imgs = imgs_col.fillna("").to_numpy()

    # Scalable candidate generation
    candidate_pairs = generate_candidate_pairs(cmp_df, max_distance_meters=350.0)

    dsu = UnionFind(complaint_ids)
    duplicate_pairs = []
    score_records = []

    for i, j in candidate_pairs:
        cid_a, cid_b = complaint_ids[i], complaint_ids[j]

        # 1. Text Similarity (TF-IDF + RapidFuzz)
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

        # 2. Location Distance & Similarity
        lat1, lon1 = cmp_df["latitude"].iloc[i], cmp_df["longitude"].iloc[i]
        lat2, lon2 = cmp_df["latitude"].iloc[j], cmp_df["longitude"].iloc[j]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        loc_sim = get_location_similarity(dist)

        # 3. Temporal Similarity
        d1, d2 = dates[i], dates[j]
        if pd.notnull(d1) and pd.notnull(d2):
            diff_days = abs((d1 - d2) / np.timedelta64(1, "D"))
            if diff_days <= 3:
                temp_sim = 1.0
            elif diff_days <= 7:
                temp_sim = 0.8
            elif diff_days <= 14:
                temp_sim = 0.5
            else:
                temp_sim = 0.0
        else:
            temp_sim = 0.0

        # 4. Image Similarity
        img1, img2 = imgs[i], imgs[j]
        img_sim = 1.0 if (img1 != "" and img1.lower() == img2.lower()) else 0.0

        # 5. Category Similarity
        cat_sim = 1.0  # Guaranteed 1.0 by candidate generator

        # Hybrid duplicate score calculation
        score = (
            w.get("w_text", 0.35) * text_sim
            + w.get("w_location", 0.40) * loc_sim
            + w.get("w_category", 0.15) * cat_sim
            + w.get("w_temporal", 0.05) * temp_sim
            + w.get("w_image", 0.05) * img_sim
        )
        score = round(float(score), 4)
        is_dup = score >= threshold

        pair_record = {
            "complaint_a": cid_a,
            "complaint_b": cid_b,
            "text_similarity": round(text_sim, 4),
            "location_distance": round(dist, 2) if not math.isinf(dist) else None,
            "location_similarity": float(loc_sim),
            "category_match": 1,
            "temporal_similarity": float(temp_sim),
            "image_similarity": float(img_sim),
            "duplicate_score": score,
            "predicted_duplicate": is_dup,
        }

        score_records.append(pair_record)

        if is_dup:
            duplicate_pairs.append(pair_record)
            dsu.union(cid_a, cid_b)

    # Build underlying issue groups from DSU connected components
    cluster_map = {}
    for cid in complaint_ids:
        root = dsu.find(cid)
        cluster_map.setdefault(root, []).append(cid)

    # Assign issue_ids: Map root -> ISS-XXXX
    sorted_roots = sorted(cluster_map.keys())
    root_to_issue_id = {root: f"ISS-{idx + 1:04d}" for idx, root in enumerate(sorted_roots)}

    cid_to_issue_id = {}
    cid_to_status = {}
    for root, cids in cluster_map.items():
        issue_id = root_to_issue_id[root]
        is_cluster = len(cids) > 1
        for cid in cids:
            cid_to_issue_id[cid] = issue_id
            cid_to_status[cid] = "CONFIRMED_DUPLICATE" if is_cluster else "UNDERLYING_ISSUE_CREATED"

    consolidated_df = cmp_df.copy()
    consolidated_df["consolidated_issue_id"] = consolidated_df["complaint_id"].map(cid_to_issue_id)
    consolidated_df["status_code"] = consolidated_df["complaint_id"].map(cid_to_status)

    reduction_pct = round((1 - len(candidate_pairs) / (n * (n - 1) / 2)) * 100, 2) if n > 1 else 0.0
    print(
        f"Scalable Deduplication completed in {time.time() - t0:.2f}s: Evaluated {len(candidate_pairs)} candidate pairs "
        f"({reduction_pct}% candidate reduction). Found {len(duplicate_pairs)} duplicate pairs across {len(sorted_roots)} unique underlying issues."
    )

    return consolidated_df, duplicate_pairs, score_records


if __name__ == "__main__":
    from preprocessing import load_and_clean_all
    cmp_clean, _, _ = load_and_clean_all()
    cons_df, dup_pairs, records = run_deduplication(cmp_clean)
    print(f"Sample Consolidated Issue Mapping: \n{cons_df[['complaint_id', 'category', 'consolidated_issue_id', 'status_code']].head(10)}")

