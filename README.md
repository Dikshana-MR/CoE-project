# Municipality Complaint Deduplication & Underlying-Issue Tracker

The Municipality Complaint Deduplication and Underlying-Issue Tracker addresses municipal operational bottlenecks where citizens submit multiple duplicate complaints for the same real-world issue (e.g., potholes, streetlight outages, waste piles). 

The goal of this system is to automatically detect duplicate complaints, consolidate them under a single underlying issue (`Issue_ID`), and preserve all original citizen complaints for transparent auditability.

---

## 1. Problem
Municipal authorities face high volumes of redundant complaints. Manually filtering duplicate reports leads to inefficient resource allocation, delayed resolution, and confused tracking. This project automates duplicate identification while ensuring zero complaint data loss.

---

## 2. Dataset
The system operates on three core relational datasets:
- **`data/complaints.csv`**: Contains citizen reports including `Complaint_ID`, `Description`, `Category`, `Latitude`, `Longitude`, `Image_Path`, `Created_Date`, `Status`, and ground-truth `Issue_ID`.
- **`data/issues.csv`**: Contains consolidated municipal underlying issue records with `Issue_ID`, `Category`, `Latitude`, `Longitude`, `Issue_Description`, `Complaint_Count`, and `Status`.
- **`data/resolution_events.csv`**: Contains resolution lifecycle events linked to issue IDs.

---

## 3. Baseline
The baseline deduplication model uses strict exact matching:
```
Same Category AND Exact Latitude AND Exact Longitude
```
Because raw complaint coordinates feature subtle GPS precision variances, exact matching yields zero false positives but misses near-identical spatial duplicates (0 duplicate pairs found on this dataset).

---

## 4. Prototype Approach
The prototype implements an explainable hybrid scoring algorithm:
$$\text{Score} = 0.50 \times \text{TextSimilarity} + 0.30 \times \text{LocationSimilarity} + 0.20 \times \text{CategorySimilarity}$$

- **Text Similarity (50%)**: Calculated using TF-IDF vector cosine similarity or RapidFuzz string distance.
- **Location Similarity (30%)**: Calculated using Haversine geographic distance:
  - Distance $\le 50\text{m} \rightarrow 1.0$
  - Distance $\le 100\text{m} \rightarrow 0.8$
  - Distance $\le 250\text{m} \rightarrow 0.5$
  - Distance $> 250\text{m} \rightarrow 0.0$
- **Category Similarity (20%)**: $1.0$ if category matches, else $0.0$.
- **Decision Rule**: A pair is classified as **`DUPLICATE`** if $\text{Score} \ge 0.75$, otherwise **`NOT DUPLICATE`**.

---

## 5. Duplicate Consolidation
When complaints are identified as duplicates:
- Original complaint records are **never deleted or altered**.
- Each complaint retains its full details (`Complaint_ID`, description, category, coordinates, image path).
- Duplicate complaints are linked to a single underlying `Issue_ID` (e.g. `ISS-0001` $\rightarrow$ `CMP-001`, `CMP-005`, `CMP-012`).

---

## 6. Evaluation
The model evaluation compares the Baseline against the Prototype using ground-truth `Issue_ID` labels present in the dataset.

Generated Reports:
- `outputs/baseline_results.csv`
- `outputs/evaluation_results.csv`
- `outputs/error_analysis.csv`

Metrics calculated: Total complaints, Duplicate pairs, Unique issues, Duplicate consolidation rate, Precision, Recall, and F1 Score.

---

## 7. Edge Cases
The test suite explicitly tests three required edge cases:
1. **Case 1**: Same category, similar description, nearby coordinates ($\le 50\text{m}$) $\rightarrow$ **`DUPLICATE`** ($\text{Score} \ge 0.75$).
2. **Case 2**: Same location, different category $\rightarrow$ **`NOT DUPLICATE`** ($\text{Score} < 0.75$).
3. **Case 3**: Same description, far-away location ($> 250\text{m}$) $\rightarrow$ **`NOT DUPLICATE`** ($\text{Score} < 0.75$).
4. **Resilience**: Missing coordinates (`NaN`/`None`) return location similarity $0.0$ without system failure.

---

## 8. How to Run

### Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Unit & Edge Case Tests
```bash
pytest
```

### Execute Pipeline & Web Application
Run full pipeline (data preprocessing, baseline, deduplication, SQLite setup, evaluation, error analysis) and launch web dashboard:
```bash
python run.py
```

To run pipeline only without starting web server:
```bash
python run.py --pipeline-only
```

Access the Web Dashboard at: `http://127.0.0.1:8000`
