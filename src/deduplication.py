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
    """Maps distance in meters to location similarity score."""
    if math.isinf(distance_meters) or math.isnan(distance_meters):
        return 0.0
    if distance_meters <= 50.0:
        return 1.0
    elif distance_meters <= 100.0:
        return 0.8
    elif distance_meters <= 250.0:
        return 0.5
    else:
        return 0.0


def get_category_similarity(cat1: str, cat2: str) -> float:
    """Returns 1.0 if categories match (case-insensitive), else 0.0."""
    if not cat1 or not cat2:
        return 0.0
    return 1.0 if str(cat1).strip().lower() == str(cat2).strip().lower() else 0.0


def get_single_text_similarity(desc1: str, desc2: str) -> float:
    """Calculates text similarity between two descriptions using RapidFuzz or TF-IDF fallback."""
    str1, str2 = str(desc1).strip(), str(desc2).strip()
    if not str1 or not str2:
        return 0.0

    if RAPIDFUZZ_AVAILABLE:
        # RapidFuzz normalized ratio [0.0, 1.0]
        return fuzz.ratio(str1.lower(), str2.lower()) / 100.0
    else:
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf = vectorizer.fit_transform([str1, str2])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(sim)
        except Exception:
            return 0.0


def calculate_pair_duplicate_score(cmp1: dict, cmp2: dict) -> dict:
    """
    Computes duplicate score for two complaint dictionary records.
    Returns score breakdown dictionary.
    """
    cat_sim = get_category_similarity(cmp1.get("category", ""), cmp2.get("category", ""))
    
    lat1, lon1 = cmp1.get("latitude"), cmp1.get("longitude")
    lat2, lon2 = cmp2.get("latitude"), cmp2.get("longitude")
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    loc_sim = get_location_similarity(dist)

    text_sim = get_single_text_similarity(
        cmp1.get("description", ""), cmp2.get("description", "")
    )

    # Weighted formula: 50% text, 30% location, 20% category
    score = 0.50 * text_sim + 0.30 * loc_sim + 0.20 * cat_sim
    score = round(score, 4)
    is_duplicate = score >= 0.75

    return {
        "complaint_a": cmp1.get("complaint_id"),
        "complaint_b": cmp2.get("complaint_id"),
        "text_similarity": round(text_sim, 4),
        "location_distance": round(dist, 2) if not math.isinf(dist) else None,
        "location_similarity": loc_sim,
        "category_match": int(cat_sim == 1.0),
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
            self.parent[root_b if (root_b := min(root_i, root_j)) else root_i] = max(root_i, root_j)


def run_deduplication(cmp_df: pd.DataFrame, threshold: float = 0.75) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """
    Executes vectorized deduplication across all complaint records in cmp_df.
    Returns:
    - consolidated_cmp_df: DataFrame with updated issue_id linked to complaints.
    - duplicate_pairs: List of duplicate pair dicts (with scores).
    - score_records: List of all evaluated pair scores for error analysis.
    """
    t0 = time.time()
    n = len(cmp_df)
    complaints = cmp_df.to_dict("records")
    complaint_ids = [c["complaint_id"] for c in complaints]

    # Pre-compute TF-IDF Matrix for full corpus
    vectorizer = TfidfVectorizer(stop_words="english")
    descriptions = [c["description"] for c in complaints]
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    tfidf_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Pre-compute location distance matrix
    lats = np.radians(cmp_df["latitude"].to_numpy())
    lons = np.radians(cmp_df["longitude"].to_numpy())
    
    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2.0) ** 2
    dists_meters = 2.0 * 6371000.0 * np.arcsin(np.sqrt(a))
    
    # Handle NaN coordinates
    valid_locs = ~(np.isnan(lats)[:, None] | np.isnan(lats)[None, :])
    dists_meters[~valid_locs] = np.inf

    # Category matching matrix
    cats = cmp_df["category"].astype(str).str.lower().to_numpy()
    cat_match_matrix = (cats[:, None] == cats[None, :]).astype(float)

    # Location similarity matrix
    loc_sim_matrix = np.zeros_like(dists_meters)
    loc_sim_matrix[dists_meters <= 50.0] = 1.0
    loc_sim_matrix[(dists_meters > 50.0) & (dists_meters <= 100.0)] = 0.8
    loc_sim_matrix[(dists_meters > 100.0) & (dists_meters <= 250.0)] = 0.5
    loc_sim_matrix[dists_meters > 250.0] = 0.0

    # Calculate overall duplicate score matrix
    # Formula: 0.50 * text_similarity + 0.30 * location_similarity + 0.20 * category_similarity
    score_matrix = (
        0.50 * tfidf_sim_matrix
        + 0.30 * loc_sim_matrix
        + 0.20 * cat_match_matrix
    )

    dsu = UnionFind(complaint_ids)
    duplicate_pairs = []
    score_records = []

    iu_0, iu_1 = np.triu_indices(n, k=1)

    for idx in range(len(iu_0)):
        i, j = iu_0[idx], iu_1[idx]
        score = float(score_matrix[i, j])
        is_dup = score >= threshold

        cid_a, cid_b = complaint_ids[i], complaint_ids[j]

        pair_record = {
            "complaint_a": cid_a,
            "complaint_b": cid_b,
            "text_similarity": round(float(tfidf_sim_matrix[i, j]), 4),
            "location_distance": round(float(dists_meters[i, j]), 2) if not np.isinf(dists_meters[i, j]) else None,
            "location_similarity": float(loc_sim_matrix[i, j]),
            "category_match": int(cat_match_matrix[i, j]),
            "duplicate_score": round(score, 4),
            "predicted_duplicate": is_dup,
        }
        
        # Keep track of all evaluated pairs (or sample for error analysis)
        score_records.append(pair_record)

        if is_dup:
            duplicate_pairs.append(pair_record)
            dsu.union(cid_a, cid_b)

    # Build underlying issue groups from DSU connected components
    cluster_map = {}
    for cid in complaint_ids:
        root = dsu.find(cid)
        cluster_map.setdefault(root, []).append(cid)

    # Assign issue_ids
    # Map root -> ISS-XXXX
    sorted_roots = sorted(cluster_map.keys())
    root_to_issue_id = {root: f"ISS-{idx + 1:04d}" for idx, root in enumerate(sorted_roots)}

    cid_to_issue_id = {}
    for root, cids in cluster_map.items():
        issue_id = root_to_issue_id[root]
        for cid in cids:
            cid_to_issue_id[cid] = issue_id

    # Create consolidated complaints DataFrame without modifying original fields
    consolidated_df = cmp_df.copy()
    consolidated_df["consolidated_issue_id"] = consolidated_df["complaint_id"].map(cid_to_issue_id)

    print(f"Deduplication completed in {time.time() - t0:.2f}s: Found {len(duplicate_pairs)} duplicate pairs across {len(sorted_roots)} unique underlying issues.")

    return consolidated_df, duplicate_pairs, score_records


if __name__ == "__main__":
    from preprocessing import load_and_clean_all
    cmp_clean, _, _ = load_and_clean_all()
    cons_df, dup_pairs, records = run_deduplication(cmp_clean)
    print(f"Sample Consolidated Issue Mapping: \n{cons_df[['complaint_id', 'category', 'consolidated_issue_id']].head(10)}")
