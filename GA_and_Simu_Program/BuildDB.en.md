# BuildDB.py Documentation

## 1. What This Does

`BuildDB.py` is a **post-processing tool** for the `GA.py` (V4) GA calibration pipeline. It:

1. Consolidates the `optimized_weights_detail.csv` files produced separately by `GA.py` under each gene's folder into **a single SQLite weight database (.db)**, so downstream code and SQL queries can use it directly instead of digging through CSVs folder by folder.
2. (Optional) While consolidating, attaches a **confidence score (`confidence_score`)** and a one-line **suggestion** to every SNP weight, based on metrics `GA.py` already computed — so downstream users know which weights can be trusted as-is, and which should be manually reviewed or reverted to the original PrediXcan weight.
3. (Optional) If you have the original PrediXcan model `.db` (e.g. `gtex_v7_Lung_imputed_europeans_tw_0_5_signif.db`), it can backfill `chr`, `pos`, `ref_allele`, `eff_allele` in the same run — information needed later, when predicting expression from genotype (VCF) data, to determine allele orientation and avoid sign errors in the weights.

The whole process **never reruns GA and doesn't need a GPU** — it purely reads existing output files to consolidate and compute statistics, so it runs fast.

---

## 2. Inputs and Outputs

### Inputs

| File | Description | Required? |
|---|---|---|
| `<base_dir>/<GENE_ID>/optimized_weights_detail.csv` | Per-SNP weight detail output by `GA.py` for each gene | Required |
| `<out_dir>/final_summary.csv` | Gene-level performance metrics aggregated across the whole batch by `GA.py` (`Correlation after GA`, `train_corr`, `weight_stability`, etc.) | Optional, but **strongly recommended** when computing the confidence score — otherwise gene-level info falls back to a neutral value of 0.5 |
| Original PrediXcan model `.db` (e.g. `gtex_v7_Lung_imputed_europeans_tw_0_5_signif.db`) | A standard PrediXcan model db; its `weights` table must include `rsid, varID, ref_allele, eff_allele` columns | Optional, but **strongly recommended** — `ref_allele`/`eff_allele` are needed later to determine allele orientation when doing genotype-based prediction |

### Outputs (SQLite `.db`, containing 1–2 tables)

| Table | Content | When produced |
|---|---|---|
| `weights` (or the name given by `--table_name`) | The consolidated SNP-level weight table, one row = one SNP for one gene | Always produced |
| `gene_summary` | A verbatim copy of `final_summary.csv` (gene-level metrics), for later `JOIN`s | Only produced if both `--summary_csv` and `--score` are given |

`weights` table columns (without `--score` or `--predixcan_db`):

```
Gene, rsid, EUR_AF, EAS_AF, AF_Diff, PrediXcan_W,
Allele_Factor, GA_Scaling_Mean, GA_Scaling_Std, GA_W
```

With `--predixcan_db` added, these additional columns appear:

```
chr, pos, ref_allele, eff_allele   # chr/pos parsed from varID + read directly from the source db
```

> `chr`/`pos` are parsed out of `varID` (format `<chr>_<pos>_<ref>_<alt>_<build>`, e.g. `1_1026707_C_A_b37`). If the original db lacks a `varID` column, only `ref_allele`/`eff_allele` are backfilled, `chr`/`pos` are left blank with a printed warning. Rows with no matching rsid are `NULL`, which does not interrupt the program — a "matched vs. unmatched" count is printed at runtime.

With `--score` added, these additional columns appear:

```
Correlation after GA, train_corr, train_corr_std,
weight_stability, weight_stability_penalty,   # from final_summary.csv
confidence_score, is_pruned_zero, suggestion  # newly computed columns
```

> By default the `source` column is dropped (since every row is fixed as `GA` and carries no information). Adjust with `--drop_cols`.

---

## 3. How the Confidence Score Is Computed

`confidence_score` (0–1, higher = more trustworthy) is a weighted combination of four signals that `GA.py` **already computed but scattered across different files** — no extra computation or GA rerun is needed:

| Signal | Source column | Logic |
|---|---|---|
| Stability of a SNP's scaling coefficient across folds | `GA_Scaling_Std / \|GA_Scaling_Mean\|` (coefficient of variation) | The more the selected scaling differs across LOOCV folds, the less stable → the lower the score |
| Population transferability | `AF_Diff` | The larger the EUR/EAS allele-frequency difference (the same basis `GA.py` uses to assign S/M/L tiers) → the lower the score |
| Whether it was pruned | `GA_W == 0` | GA has already judged this SNP's contribution negligible → directly flagged as `is_pruned_zero`, excluded from the continuous score, given its own suggestion |
| Overall gene model performance and stability | `final_summary.csv`'s `Correlation after GA` / `train_corr` / `weight_stability` | Even if a single SNP is stable, if the gene's overall calibration performs poorly or is unstable across folds, confidence is dragged down too |

Each of the four components is min-max normalized to [0,1], then summed with weights `--w_cv / --w_af / --w_gene / --w_zero` (set in-code, defaulting to 0.35 / 0.20 / 0.35 / 0.10).

### 3.1 Full Formula

```
confidence_score = w_cv × cv_score
                  + w_af × af_score
                  + w_gene × gene_component
                  − w_zero × zero_penalty

(result clipped to [0, 1]; defaults w_cv=0.35, w_af=0.20, w_gene=0.35, w_zero=0.10)
```

How each component is actually computed:

**① `cv_score` (stability of a SNP's scaling coefficient)**
```python
cv = GA_Scaling_Std / (|GA_Scaling_Mean| + eps)
cv_clipped = cv.clip(upper=99th percentile of cv)   # avoid extreme values crushing the overall score
cv_score = 1 − minmax_normalize(cv_clipped)
```

**② `af_score` (population transferability)**
```python
af = AF_Diff.clip(0, 1)
af_score = 1 − af
```

**③ `zero_penalty` (pruning flag — a direct penalty rather than a continuous score)**
```python
is_zero = (|GA_W| < 1e-12)
zero_penalty = 1.0 if is_zero else 0.0
```

**④ `gene_component` (overall gene performance and stability, itself a combination of two sub-components)**
```python
gene_component = 0.6 × gene_score + 0.4 × instab_score
```
- `gene_score` = `minmax_normalize(Correlation after GA)` (falls back to `train_corr` if that column doesn't exist)
- `instab_score` = `1 − minmax_normalize(weight_stability_penalty)` (falls back to `weight_stability` if that column doesn't exist)
- If `--summary_csv` is **not provided**: `gene_component` is directly set to the neutral value `0.5`, neither adding nor subtracting, to avoid misleading the ranking when data is missing.

### 3.2 Why This Weighting Ratio

The design logic: **the SNP's own statistical stability** (`cv_score` 0.35 + `af_score` 0.20 = 0.55) is weighted slightly more than **the overall performance of the gene it belongs to** (`gene_component` 0.35), because this score is primarily meant to answer "is this specific weight value trustworthy" rather than re-evaluating overall gene-model quality — that job is already done by `final_summary.csv`'s own `Improvement` / `Impact_Category`. The pruning flag (`w_zero` 0.10) is just a clear hint and doesn't need much weight.

This weighting is a **reasonable starting point given the existing pipeline columns, not a statistically validated optimum**. It's now exposed as CLI parameters `--w_cv` / `--w_af` / `--w_gene` / `--w_zero` (usage in 4.3, 4.5), e.g.:
- To care more about "did the gene overall transfer successfully" → `--w_gene 0.5 --w_cv 0.25 --w_af 0.15`
- To avoid penalizing already-pruned SNPs further (since `suggestion` already flags them) → `--w_zero 0`
- To ignore gene-level information entirely (equivalent to not providing `--summary_csv`, but more explicit) → `--w_gene 0`

> `w_zero` is applied as a **subtraction**, the other three are **additions**, so the four weights are not required to sum to 1; but if you want the score to stay readable in [0, 1], it's recommended that `w_cv + w_af + w_gene ≈ 1`.

**`suggestion` label reference:**

| Condition | Suggestion text |
|---|---|
| `GA_W == 0` (already pruned) | Pruned (weight=0) — excluded by GA, no action needed |
| `confidence_score >= 0.75` | High confidence — use GA weight as-is |
| `0.5 <= score < 0.75` | Moderate confidence — usable, worth spot-checking |
| `0.3 <= score < 0.5` | Low confidence — review manually before use |
| `score < 0.3` | Very low confidence — consider reverting to PrediXcan_W or excluding SNP |

> ⚠️ Scores are normalized **relative to the whole dataset being consolidated in this run**, so:
> - It's recommended to consolidate all genes you want to compare against each other **in one pass** (e.g. the entire `V4_LOOCV_Sign` folder) rather than splitting it into multiple batches, otherwise scores across batches are not comparable.
> - This is a practical starting point based on existing columns, not a formal statistical test; if using it for a formal report/publication, cross-validate against a familiar statistical method (e.g. Fisher's z confidence interval, bootstrap).

---

## 4. Usage

### Requirements

```bash
pip install pandas numpy --break-system-packages   # skip if already installed
```

### 4.1 Consolidate Only, No Scoring

```bash
python3 BuildDB.py \
    --base_dir "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db   "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db"
```

### 4.2 Consolidate + Add Confidence Score and Suggestion (recommended usage)

```bash
python3 BuildDB.py \
    --base_dir    "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db      "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db" \
    --summary_csv "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/final_summary.csv" \
    --score
```

### 4.2.1 Consolidate + Score + Backfill Allele Info (most complete, most recommended usage)

If you have the original PrediXcan model `.db` on hand, it's recommended to include it directly, so later when using this `weights.db` for genotype-based expression prediction, no extra processing is needed:

```bash
python3 BuildDB.py \
    --base_dir     "/mnt/data_40T/anywaycheng/UploadFile/GA/V4_10Fold_Sign" \
    --out_db       "/mnt/data_40T/anywaycheng/UploadFile/GA/GA_Weight_DataBase/V4_10Fold_Sign_weights.db" \
    --summary_csv  "/mnt/data_40T/anywaycheng/UploadFile/GA/V4_10Fold_Sign/final_summary.csv" \
    --predixcan_db "/path/to/gtex_v7_Lung_imputed_europeans_tw_0_5_signif.db" \
    --score
```

### 4.3 Full Argument Reference

| Argument | Required | Default | Description |
|---|---|---|---|
| `--base_dir` | Yes | — | Parent path containing one subfolder per gene (e.g. `V4_LOOCV_Sign`) |
| `--out_db` | Yes | — | Output SQLite `.db` path |
| `--filename` | No | `optimized_weights_detail.csv` | Filename to read within each gene's folder |
| `--summary_csv` | No | `None` | Path to `final_summary.csv`, providing gene-level metrics; strongly recommended together with `--score` |
| `--predixcan_db` | No | `None` | Path to the original PrediXcan model `.db`; if provided, backfills `chr`/`pos`/`ref_allele`/`eff_allele` columns |
| `--score` | No | off | Add this flag to compute `confidence_score` / `suggestion` |
| `--table_name` | No | `weights` | Name of the SNP weight table in the DB, useful for distinguishing multiple result sets (e.g. Sign / Unsign) stored in the same db |
| `--drop_cols` | No | `source` | Comma-separated columns to drop after consolidation; pass empty string `""` to keep everything |
| `--w_cv` | No | `0.35` | Weight of "SNP scaling-coefficient stability" in the confidence score (only takes effect with `--score`) |
| `--w_af` | No | `0.20` | Weight of "population transferability AF_Diff" in the confidence score |
| `--w_gene` | No | `0.35` | Weight of "overall gene performance and stability" in the confidence score; only effective together with `--summary_csv` |
| `--w_zero` | No | `0.10` | Pruning penalty (amount subtracted from the score when `GA_W == 0`) |

### 4.4 Common Scenarios

**Consolidating both LOOCV Sign and Unsign result sets into the same db (distinguished by table name):**

```bash
python3 BuildDB.py \
    --base_dir "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db   "/mnt/data_80T/anywaycheng/GA/5%_Result/all_weights.db" \
    --summary_csv "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/final_summary.csv" \
    --score --table_name weights_sign

python3 BuildDB.py \
    --base_dir "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Unsign" \
    --out_db   "/mnt/data_80T/anywaycheng/GA/5%_Result/all_weights.db" \
    --summary_csv "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Unsign/final_summary.csv" \
    --score --table_name weights_unsign
```

**Keep the `source` column instead of dropping it:**

```bash
python3 BuildDB.py --base_dir ... --out_db ... --drop_cols ""
```

### 4.5 Customizing Confidence-Score Weights

```bash
# Weight overall gene performance more, lower weight on SNP self-stability
python3 BuildDB.py \
    --base_dir    "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db      "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db" \
    --summary_csv "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/final_summary.csv" \
    --score --w_cv 0.25 --w_af 0.15 --w_gene 0.5 --w_zero 0.10

# Don't further penalize already-pruned SNPs
python3 BuildDB.py --base_dir ... --out_db ... --summary_csv ... --score --w_zero 0
```

At runtime the terminal prints `[INFO] Scoring weights: w_cv=..., w_af=..., w_gene=..., w_zero=...` so you can confirm which weight set this run used.

---

## 5. Checking Results After Running

```bash
sqlite3 "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db"
```

```sql
.tables                                                  -- see which tables exist
.schema weights                                          -- see column structure

SELECT COUNT(DISTINCT Gene) FROM weights;                -- how many genes were consolidated

SELECT Gene, rsid, GA_W, confidence_score, suggestion
FROM weights
WHERE confidence_score < 0.3;                             -- find low-confidence weights

SELECT suggestion, COUNT(*)
FROM weights
GROUP BY suggestion;                                       -- count distribution per suggestion category

SELECT w.Gene, w.rsid, w.confidence_score, g."Correlation after GA"
FROM weights w
JOIN gene_summary g ON w.Gene = g.gene_id
ORDER BY w.confidence_score ASC
LIMIT 20;                                                   -- cross-table query: 20 least-trustworthy SNPs with their gene performance

SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN eff_allele IS NULL THEN 1 ELSE 0 END) AS missing_allele_info
FROM weights;                                               -- check coverage of --predixcan_db backfill (if that option was used)
```

Or read directly with Python:

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("weights.db")
df = pd.read_sql("SELECT * FROM weights", conn)
```
