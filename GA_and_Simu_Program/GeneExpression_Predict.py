#!/usr/bin/env python3
"""
GeneExpression_Predict.py
----------
Compute PrediXcan-style predicted gene expression from a GA-corrected weights.db
and a plink2-exported dosage matrix.

This is normally invoked by GeneExpression_Predict.sh, which handles VCF -> dosage extraction
via plink2 first. You can also run it directly if you already have a dosage
.raw file (see --dosage_raw below).

Formula (same as PrediXcan):
    PredictedExpression(gene, sample) = sum_i  weight(gene, SNP_i) x dosage(SNP_i, sample)

Outputs (mirrors PrediXcan's official Predict.py output layout):
    <out_prefix>_predicted_expression.csv
        Wide format, one row per sample:  FID, IID, <gene_1>, <gene_2>, ...
        (same layout as PrediXcan's `*_predicted_expression.txt`)

    <out_prefix>_summary.csv
        One row per gene, QC/coverage metrics analogous to PrediXcan's
        `*_summary.txt` (n_snps_in_model, n_snps_used, var_g, ...), plus
        GA-specific quality fields (avg_confidence_score, corr_after_GA).

Typical usage
-------------
python3 GeneExpression_Predict.py \
    --dosage_raw dosage_matrix.raw \
    --weights_db V4_3Fold_Sign_weights.db \
    --out_prefix predicted
"""

import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy import sparse


# ──────────────────────────────────────────────
# Load model weights + gene-level context from weights.db
# ──────────────────────────────────────────────
def load_weights(db_path: str, table_name: str, weight_col: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    cols_info = pd.read_sql(f"PRAGMA table_info({table_name});", conn)
    available = set(cols_info["name"])

    select_cols = ["Gene", "rsid", weight_col]
    has_allele_info = {"eff_allele", "ref_allele"}.issubset(available)
    if has_allele_info:
        select_cols += ["ref_allele", "eff_allele"]
    if "confidence_score" in available:
        select_cols += ["confidence_score"]

    # Keep only SNPs with a non-zero final weight -- these are the SNPs that
    # actually enter the prediction formula (analogous to PrediXcan's
    # "n.snps.in.model", which already excludes elastic-net-zeroed predictors).
    query = f"SELECT {', '.join(select_cols)} FROM {table_name} WHERE {weight_col} != 0;"
    df = pd.read_sql(query, conn)
    df = df.dropna(subset=["Gene", "rsid", weight_col])

    if not has_allele_info:
        print("[WARN] weights.db has no ref_allele/eff_allele columns. Dosage will be used "
              "as-is from plink2's counted allele; some SNPs may end up with a flipped sign. "
              "Rebuild weights.db with --predixcan_db (BuildDB.py) to fix this.", file=sys.stderr)

    print(f"[INFO] Loaded {len(df):,} non-zero (gene, SNP) weight entries "
          f"across {df['Gene'].nunique():,} genes from {db_path} (table: {table_name})")
    return df, has_allele_info


def load_gene_summary(db_path: str, table_name: str = "gene_summary") -> pd.DataFrame | None:
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"].tolist()
    if table_name not in tables:
        print(f"[WARN] No '{table_name}' table found in weights.db; "
              f"summary.csv will not include gene-level GA performance columns.")
        return None
    gdf = pd.read_sql(f"SELECT * FROM {table_name};", conn)
    keep = [c for c in ["gene_id", "train_corr", "Correlation after GA", "n_snps"] if c in gdf.columns]
    return gdf[keep].rename(columns={"gene_id": "Gene", "Correlation after GA": "corr_after_GA"})


# ──────────────────────────────────────────────
# Load plink2 --export A dosage matrix
# ──────────────────────────────────────────────
def load_dosage_raw(raw_path: str):
    raw = pd.read_csv(raw_path, sep=r"\s+")
    meta_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    snp_cols = [c for c in raw.columns if c not in meta_cols]

    counted_allele, rename_map = {}, {}
    for c in snp_cols:
        rsid, _, allele = c.rpartition("_")
        counted_allele[rsid] = allele
        rename_map[c] = rsid

    fid_iid = raw[["FID", "IID"]].copy()
    dosage = raw.set_index("IID")[snp_cols].rename(columns=rename_map)
    print(f"[INFO] Loaded dosage matrix: {dosage.shape[0]:,} samples x {dosage.shape[1]:,} SNPs "
          f"from {raw_path}")
    return dosage, counted_allele, fid_iid


# ──────────────────────────────────────────────
# Align alleles, build sparse weight matrix + dense dosage matrix, multiply
# ──────────────────────────────────────────────
def predict(weights_df, has_allele_info, dosage, counted_allele, assume_aligned):
    available_snps = set(dosage.columns)
    w = weights_df.copy()
    w["in_vcf"] = w["rsid"].isin(available_snps)

    w_used = w[w["in_vcf"]].copy()
    w_used["counted_allele"] = w_used["rsid"].map(counted_allele)

    if has_allele_info and not assume_aligned:
        unresolved = w_used["eff_allele"].isna()
        needs_flip = (w_used["counted_allele"] != w_used["eff_allele"]) & ~unresolved
        if unresolved.any():
            print(f"[WARN] {unresolved.sum():,} SNP-gene rows have no eff_allele on record; "
                  f"treated as already aligned.")
    else:
        needs_flip = pd.Series(False, index=w_used.index)

    w_used["flip"] = needs_flip.values
    n_flip = int(w_used["flip"].sum())
    if n_flip:
        print(f"[INFO] Flipping dosage (2 - dosage) for {n_flip:,} SNP-gene rows "
              f"where plink2's counted allele != model's effect allele.")

    # ---- sparse gene x snp weight matrix ----
    genes = w_used["Gene"].astype("category")
    snps = w_used["rsid"].astype("category")
    weight_col = weights_df.columns[2]  # e.g. GA_W
    W = sparse.csr_matrix(
        (w_used[weight_col].values, (genes.cat.codes, snps.cat.codes)),
        shape=(len(genes.cat.categories), len(snps.cat.categories)),
    )

    # ---- dense snp x sample dosage matrix, oriented to effect allele ----
    snp_order = snps.cat.categories.tolist()
    D_raw = dosage[snp_order].to_numpy(dtype=np.float32).T
    flip_per_snp = (
        w_used.drop_duplicates("rsid").set_index("rsid").reindex(snp_order)["flip"].fillna(False).to_numpy()
    )
    D = np.where(flip_per_snp[:, None], 2.0 - D_raw, D_raw)
    if np.isnan(D).any():
        col_means = np.nanmean(D, axis=1, keepdims=True)
        nan_mask = np.isnan(D)
        D[nan_mask] = np.take(col_means, np.where(nan_mask)[0])
        print(f"[INFO] Mean-imputed {nan_mask.sum():,} missing genotype calls "
              f"({nan_mask.sum() / D.size:.3%} of used SNP x sample entries).")

    print(f"[INFO] Computing PredictedExpression = W @ D  "
          f"({W.shape[0]:,} genes x {W.shape[1]:,} SNPs) x ({D.shape[0]:,} SNPs x {D.shape[1]:,} samples)")
    pred = W @ D  # (n_genes_used, n_samples)

    pred_df = pd.DataFrame(pred, index=genes.cat.categories.tolist(), columns=dosage.index.tolist())
    return pred_df, w  # w retains in_vcf/flip info per row for the summary step


# ──────────────────────────────────────────────
# Build PrediXcan-style summary table
# ──────────────────────────────────────────────
def build_summary(weights_df_all_genes, w_annotated, pred_df, gene_summary_df):
    all_genes = weights_df_all_genes["Gene"].unique().tolist()

    n_in_model = weights_df_all_genes.groupby("Gene").size().rename("n_snps_in_model")
    n_used = w_annotated[w_annotated["in_vcf"]].groupby("Gene").size().rename("n_snps_used")

    summary = pd.DataFrame(index=all_genes)
    summary.index.name = "gene"
    summary = summary.join(n_in_model).join(n_used)
    summary["n_snps_in_model"] = summary["n_snps_in_model"].fillna(0).astype(int)
    summary["n_snps_used"] = summary["n_snps_used"].fillna(0).astype(int)
    summary["pct_snps_used"] = np.where(
        summary["n_snps_in_model"] > 0,
        (summary["n_snps_used"] / summary["n_snps_in_model"]).round(4),
        np.nan,
    )

    # var_g: variance of predicted expression across samples in THIS cohort --
    # PrediXcan's standard QC signal; genes with var_g == 0 carry no information here.
    pred_stats = pd.DataFrame({
        "pred_mean": pred_df.mean(axis=1),
        "pred_sd":   pred_df.std(axis=1),
    })
    pred_stats["var_g"] = pred_stats["pred_sd"] ** 2
    summary = summary.join(pred_stats)

    avg_conf = None
    if "confidence_score" in w_annotated.columns:
        avg_conf = (
            w_annotated[w_annotated["in_vcf"]].groupby("Gene")["confidence_score"].mean().rename("avg_confidence_score")
        )
        summary = summary.join(avg_conf)

    if gene_summary_df is not None:
        gdf = gene_summary_df.set_index("Gene")
        summary = summary.join(gdf, how="left")

    def status(row):
        if row["n_snps_in_model"] == 0:
            return "NO_MODEL_SNPS"          # gene was fully pruned by GA, no usable weight
        if row["n_snps_used"] == 0:
            return "NO_SNPS_IN_VCF"          # none of the model's SNPs were found in the input VCF
        if pd.notna(row.get("var_g")) and row["var_g"] == 0:
            return "ZERO_VARIANCE"           # predicted, but identical across all samples (uninformative)
        if row["pct_snps_used"] < 0.5:
            return "LOW_SNP_COVERAGE"
        return "OK"

    summary["status"] = summary.apply(status, axis=1)
    summary = summary.reset_index().sort_values(["status", "gene"])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dosage_raw", required=True, help="plink2 --export A output (.raw)")
    ap.add_argument("--weights_db", required=True, help="weights.db built by BuildDB.py")
    ap.add_argument("--table_name", default="weights", help="SNP weight table name in weights.db")
    ap.add_argument("--gene_summary_table", default="gene_summary", help="Gene-level summary table name")
    ap.add_argument("--weight_col", default="GA_W", help="Which weight column to predict with (GA_W or PrediXcan_W)")
    ap.add_argument("--out_prefix", required=True, help="Output prefix, e.g. 'predicted' -> "
                                                          "predicted_predicted_expression.csv / predicted_summary.csv")
    ap.add_argument("--assume_aligned", action="store_true",
                     help="Skip effect-allele flip check; trust plink2's counted allele as-is")
    args = ap.parse_args()

    weights_df, has_allele_info = load_weights(args.weights_db, args.table_name, args.weight_col)
    gene_summary_df = load_gene_summary(args.weights_db, args.gene_summary_table)
    dosage, counted_allele, fid_iid = load_dosage_raw(args.dosage_raw)

    pred_df, w_annotated = predict(weights_df, has_allele_info, dosage, counted_allele, args.assume_aligned)

    # ---- Write wide-format predicted expression (PrediXcan layout) ----
    wide = pred_df.T  # samples x genes
    wide.index.name = "IID"
    wide = wide.reset_index().merge(fid_iid.drop_duplicates("IID"), on="IID", how="left")
    front = ["FID", "IID"]
    wide = wide[front + [c for c in wide.columns if c not in front]]
    expr_path = f"{args.out_prefix}_predicted_expression.csv"
    wide.to_csv(expr_path, index=False)
    print(f"[INFO] Wrote predicted expression: {wide.shape[0]:,} samples x "
          f"{wide.shape[1] - 2:,} genes -> {expr_path}")

    # ---- Write summary (PrediXcan-style + GA-specific QC) ----
    summary = build_summary(weights_df, w_annotated, pred_df, gene_summary_df)
    summary_path = f"{args.out_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    n_ok = (summary["status"] == "OK").sum()
    print(f"[INFO] Wrote summary: {len(summary):,} genes ({n_ok:,} status=OK) -> {summary_path}")


if __name__ == "__main__":
    main()
