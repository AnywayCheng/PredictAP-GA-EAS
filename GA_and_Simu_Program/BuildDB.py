#!/usr/bin/env python3
"""
BuildDB.py
-------------------
Consolidate per-gene `optimized_weights_detail.csv` files (produced by GA.py)
into a single SQLite database, and optionally attach a per-SNP confidence score
and a human-readable suggestion.

Typical usage
-------------
# Step 1: just build the raw consolidated DB (no scoring yet)
python3 BuildDB.py \
    --base_dir "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db   "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db"

# Step 2: build DB AND compute confidence score / suggestion in the same run,
# using the run-level final_summary.csv for gene-level context.
python3 BuildDB.py \
    --base_dir    "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db      "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db" \
    --summary_csv "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/final_summary.csv" \
    --score

# Step 3 (recommended if you have the original PrediXcan model db): backfill
# chr/pos/ref_allele/eff_allele at the same time, needed later for prediction.
python3 BuildDB.py \
    --base_dir     "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db       "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db" \
    --summary_csv  "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/final_summary.csv" \
    --predixcan_db "/path/to/gtex_v7_Lung_imputed_europeans_tw_0_5_signif.db" \
    --score
"""

import argparse
import glob
import os
import sqlite3
import sys

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Step 1: collect every gene's optimized_weights_detail.csv
# ──────────────────────────────────────────────
def collect_weight_files(base_dir: str, filename: str = "optimized_weights_detail.csv") -> pd.DataFrame:
    """Walk every gene subfolder under base_dir and concatenate their weight CSVs."""
    pattern = os.path.join(base_dir, "*", filename)
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"[ERROR] No '{filename}' found under {base_dir}. Check --base_dir.")

    print(f"[INFO] Found {len(files)} gene weight files under {base_dir}")

    frames = []
    bad = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            # In case the folder name and the Gene column ever disagree, trust the folder name
            gene_from_dir = os.path.basename(os.path.dirname(fp))
            if "Gene" not in df.columns:
                df["Gene"] = gene_from_dir
            frames.append(df)
        except Exception as e:
            bad.append((fp, str(e)))

    if bad:
        print(f"[WARN] {len(bad)} files failed to read, skipped:")
        for fp, err in bad[:10]:
            print(f"   - {fp}: {err}")

    all_df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Total rows (SNP x Gene): {len(all_df):,}")
    return all_df


# ──────────────────────────────────────────────
# Step 1.5 (optional): backfill chr/pos/ref_allele/eff_allele from the
# original PrediXcan model db (e.g. gtex_v7_Lung_imputed_europeans_tw_0_5_signif.db)
# ──────────────────────────────────────────────
def load_predixcan_allele_info(predixcan_db: str) -> pd.DataFrame:
    """
    Read the original PrediXcan model db's `weights` table and derive a
    rsid -> (chr, pos, ref_allele, eff_allele) lookup table.

    The standard PrediXcan model db `weights` table has columns:
        gene, rsid, varID, ref_allele, eff_allele, weight
    where varID is formatted as "<chr>_<pos>_<ref>_<alt>_<build>"
    (e.g. "1_1026707_C_A_b37"). chr/pos are parsed out of varID here.

    ref_allele/eff_allele are SNP-level (not gene-level) properties, so this
    returns one row per distinct rsid -- if the same rsid somehow has
    conflicting allele info across genes in the source db (rare), the first
    occurrence wins and a warning is printed.
    """
    conn = sqlite3.connect(predixcan_db)
    try:
        cols_info = pd.read_sql("PRAGMA table_info(weights);", conn)
        available = set(cols_info["name"])
    finally:
        pass

    required = {"rsid", "ref_allele", "eff_allele"}
    if not required.issubset(available):
        sys.exit(f"[ERROR] --predixcan_db's 'weights' table is missing columns {required - available}. "
                  f"Found columns: {sorted(available)}")

    select_cols = ["rsid", "ref_allele", "eff_allele"] + (["varID"] if "varID" in available else [])
    allele_df = pd.read_sql(f"SELECT DISTINCT {', '.join(select_cols)} FROM weights;", conn)

    if "varID" in allele_df.columns:
        # varID format: chr_pos_ref_alt_build -> split from the left, keep first two parts
        parts = allele_df["varID"].str.split("_", n=2, expand=True)
        allele_df["chr"] = parts[0]
        allele_df["pos"] = pd.to_numeric(parts[1], errors="coerce")
        allele_df = allele_df.drop(columns=["varID"])
    else:
        print("[WARN] --predixcan_db has no varID column; chr/pos cannot be derived, "
              "only ref_allele/eff_allele will be backfilled.")
        allele_df["chr"] = None
        allele_df["pos"] = None

    n_before = len(allele_df)
    dup_rsids = allele_df["rsid"][allele_df["rsid"].duplicated()].nunique()
    if dup_rsids:
        print(f"[WARN] {dup_rsids:,} rsid(s) have multiple distinct allele records in "
              f"--predixcan_db (unexpected); keeping the first occurrence for each.")
    allele_df = allele_df.drop_duplicates(subset="rsid", keep="first")

    print(f"[INFO] Loaded allele info for {len(allele_df):,} distinct SNPs from {predixcan_db} "
          f"({n_before - len(allele_df):,} duplicate rsid rows collapsed)")
    return allele_df[["rsid", "chr", "pos", "ref_allele", "eff_allele"]]



def compute_confidence_score(
    weights_df: pd.DataFrame,
    summary_df: pd.DataFrame | None,
    w_cv: float = 0.35,
    w_af: float = 0.20,
    w_gene: float = 0.35,
    w_zero: float = 0.10,
) -> pd.DataFrame:
    """
    Combine SNP-level and gene-level signals already produced by GA.py into a single
    0~1 confidence score, plus a categorical suggestion.

    Signals used (all already exist in your pipeline output, nothing new to compute
    on GPU / rerun GA for):

      SNP-level (from optimized_weights_detail.csv):
        - GA_Scaling_Std / |GA_Scaling_Mean|  -> coefficient of variation (CV) across
          folds. High CV = GA picked very different scaling factors on different LOOCV
          folds -> unstable -> low confidence.
        - AF_Diff -> EUR/EAS allele-frequency divergence. Larger divergence = the SNP
          is exactly the kind GA.py's own tier system already treats as less
          transferable -> low confidence.
        - GA_W == 0 (pruned) -> GA effectively decided this SNP contributes nothing;
          flagged separately rather than folded into the continuous score.

      Gene-level (from final_summary.csv, merged in by Gene):
        - Correlation after GA / train_corr -> how well the whole gene model performs.
          A perfectly stable SNP inside a gene whose overall model is unreliable
          should still be flagged.
        - train_corr_std -> fold-to-fold instability of the gene's training correlation.
        - weight_stability / weight_stability_penalty -> GA.py's own instability
          warning metric (see 'stability_warn_threshold' in GA.py).

    All components are min-max normalized to [0,1] (higher = more confident) and
    combined with weights w_cv, w_af, w_gene, w_zero (sums to 1 by default, but not
    enforced -- caller is responsible for choosing a sensible combination since
    w_zero is subtracted rather than added).
    """
    df = weights_df.copy()

    # ---- SNP-level component 1: fold-to-fold scaling stability (CV) ----
    eps = 1e-8
    cv = df["GA_Scaling_Std"].abs() / (df["GA_Scaling_Mean"].abs() + eps)
    cv_clipped = cv.clip(upper=cv.quantile(0.99))  # avoid a few extreme outliers dominating
    cv_score = 1 - (cv_clipped - cv_clipped.min()) / (cv_clipped.max() - cv_clipped.min() + eps)

    # ---- SNP-level component 2: AF transferability ----
    af = df["AF_Diff"].clip(lower=0, upper=1)
    af_score = 1 - af  # AF_Diff already in [0,1]-ish range, lower diff = more confident

    # ---- SNP-level component 3: pruned-to-zero flag (handled outside the blended score) ----
    is_zero = df["GA_W"].abs() < 1e-12

    # ---- Gene-level component (optional, needs final_summary.csv) ----
    if summary_df is not None:
        gcols = [c for c in ["gene_id", "Correlation after GA", "train_corr",
                              "train_corr_std", "weight_stability",
                              "weight_stability_penalty"] if c in summary_df.columns]
        gdf = summary_df[gcols].rename(columns={"gene_id": "Gene"})
        df = df.merge(gdf, on="Gene", how="left")

        gene_perf = df.get("Correlation after GA", df.get("train_corr"))
        if gene_perf is not None:
            gene_perf = gene_perf.fillna(gene_perf.median())
            gene_score = (gene_perf - gene_perf.min()) / (gene_perf.max() - gene_perf.min() + eps)
        else:
            gene_score = pd.Series(0.5, index=df.index)

        instability_col = df.get("weight_stability_penalty", df.get("weight_stability"))
        if instability_col is not None:
            instability_col = instability_col.fillna(instability_col.median())
            instab_score = 1 - (instability_col - instability_col.min()) / (
                instability_col.max() - instability_col.min() + eps
            )
        else:
            instab_score = pd.Series(0.5, index=df.index)

        gene_component = 0.6 * gene_score + 0.4 * instab_score
    else:
        gene_component = pd.Series(0.5, index=df.index)  # neutral if no gene-level info supplied

    # ---- Combine ----
    zero_penalty = is_zero.astype(float)  # 1 if pruned to zero
    confidence = (
        w_cv * cv_score
        + w_af * af_score
        + w_gene * gene_component
        - w_zero * zero_penalty
    ).clip(lower=0, upper=1)

    df["confidence_score"] = confidence.round(4)
    df["is_pruned_zero"] = is_zero

    # ---- Suggestion label ----
    def label(row):
        if row["is_pruned_zero"]:
            return "Pruned (weight=0) - excluded by GA, no action needed"
        s = row["confidence_score"]
        if s >= 0.75:
            return "High confidence - use GA weight as-is"
        elif s >= 0.5:
            return "Moderate confidence - usable, worth spot-checking"
        elif s >= 0.3:
            return "Low confidence - review manually before use"
        else:
            return "Very low confidence - consider reverting to PrediXcan_W or excluding SNP"

    df["suggestion"] = df.apply(label, axis=1)
    return df


# ──────────────────────────────────────────────
# Step 3: write to SQLite
# ──────────────────────────────────────────────
def write_db(df: pd.DataFrame, out_db: str, table_name: str = "weights"):
    if os.path.exists(out_db):
        print(f"[WARN] {out_db} already exists, table '{table_name}' will be replaced.")
    conn = sqlite3.connect(out_db)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        # Only index columns that actually exist in this particular table
        gene_col = "Gene" if "Gene" in df.columns else ("gene_id" if "gene_id" in df.columns else None)
        if gene_col:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_gene ON {table_name}({gene_col});")
        if "rsid" in df.columns:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_rsid ON {table_name}(rsid);")
        conn.commit()
    finally:
        conn.close()
    print(f"[INFO] Wrote {len(df):,} rows to {out_db} (table: {table_name})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True, help="Folder containing one subfolder per gene (e.g. V4_LOOCV_Sign)")
    ap.add_argument("--out_db", required=True, help="Output SQLite .db path")
    ap.add_argument("--filename", default="optimized_weights_detail.csv", help="Per-gene weight filename")
    ap.add_argument("--summary_csv", default=None, help="Path to final_summary.csv for gene-level context (optional but recommended for --score)")
    ap.add_argument(
        "--predixcan_db",
        default=None,
        help="Path to the original PrediXcan model .db (e.g. gtex_v7_Lung_imputed_europeans_tw_0_5_signif.db). "
             "If given, backfills chr/pos/ref_allele/eff_allele into the consolidated weights table by rsid -- "
             "needed later for correctly orienting genotype dosage during prediction.",
    )
    ap.add_argument("--score", action="store_true", help="Also compute confidence_score + suggestion columns")
    ap.add_argument("--table_name", default="weights", help="Table name to write into")
    ap.add_argument(
        "--drop_cols",
        default="source",
        help="Comma-separated list of columns to drop after merging (default: 'source'). "
             "Pass an empty string '' to keep all columns.",
    )

    # ---- Confidence score weights (only used when --score is set) ----
    score_group = ap.add_argument_group(
        "confidence score weights",
        "Only used when --score is set. Defaults reproduce the original fixed weighting "
        "(w_cv=0.35, w_af=0.20, w_gene=0.35, w_zero=0.10). w_zero is SUBTRACTED (penalty), "
        "the other three are ADDED, so they don't strictly need to sum to 1 -- but keeping "
        "w_cv + w_af + w_gene ~= 1 makes the resulting score easiest to interpret.",
    )
    score_group.add_argument("--w_cv", type=float, default=0.35,
                              help="Weight for SNP-level fold-to-fold scaling stability (CV). Default: 0.35")
    score_group.add_argument("--w_af", type=float, default=0.20,
                              help="Weight for EUR/EAS allele-frequency transferability (AF_Diff). Default: 0.20")
    score_group.add_argument("--w_gene", type=float, default=0.35,
                              help="Weight for gene-level performance/stability (needs --summary_csv). Default: 0.35")
    score_group.add_argument("--w_zero", type=float, default=0.10,
                              help="Penalty subtracted when GA_W was pruned to zero. Default: 0.10")

    args = ap.parse_args()

    weights_df = collect_weight_files(args.base_dir, args.filename)

    if args.predixcan_db:
        if os.path.exists(args.predixcan_db):
            allele_df = load_predixcan_allele_info(args.predixcan_db)
            n_before_cols = set(weights_df.columns)
            weights_df = weights_df.merge(allele_df, on="rsid", how="left")
            n_matched = weights_df["eff_allele"].notna().sum()
            n_missing = weights_df["eff_allele"].isna().sum()
            print(f"[INFO] Backfilled chr/pos/ref_allele/eff_allele: "
                  f"{n_matched:,} rows matched, {n_missing:,} rows have no match in --predixcan_db "
                  f"(SNP not present in the original model, or rsid mismatch).")
        else:
            print(f"[WARN] --predixcan_db given but not found: {args.predixcan_db}")

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    existing_drop_cols = [c for c in drop_cols if c in weights_df.columns]
    if existing_drop_cols:
        weights_df = weights_df.drop(columns=existing_drop_cols)
        print(f"[INFO] Dropped columns: {existing_drop_cols}")

    summary_df = None
    if args.summary_csv:
        if os.path.exists(args.summary_csv):
            summary_df = pd.read_csv(args.summary_csv)
            print(f"[INFO] Loaded gene-level summary: {args.summary_csv} ({len(summary_df)} genes)")
        else:
            print(f"[WARN] --summary_csv given but not found: {args.summary_csv}")

    if args.score:
        if summary_df is None:
            print("[WARN] --score without --summary_csv: gene-level component will be neutral (0.5) for every SNP.")
        print(f"[INFO] Scoring weights: w_cv={args.w_cv}, w_af={args.w_af}, w_gene={args.w_gene}, w_zero={args.w_zero}")
        weights_df = compute_confidence_score(
            weights_df, summary_df,
            w_cv=args.w_cv, w_af=args.w_af, w_gene=args.w_gene, w_zero=args.w_zero,
        )
        # Also persist the gene-level summary as its own table for reference/joins
        if summary_df is not None:
            write_db(summary_df, args.out_db, table_name="gene_summary")

    write_db(weights_df, args.out_db, table_name=args.table_name)


if __name__ == "__main__":
    main()