#!/usr/bin/env bash
# GeneExpression_Predict.sh
# ------------
# One-command PrediXcan-style gene expression prediction from a genotype VCF,
# using GA-corrected weights (V4_3Fold_Sign_weights.db).
#
# Usage:
#   ./GeneExpression_Predict.sh genotype.vcf [output_dir]
#
# The user only needs to supply the VCF; everything else (weights db path,
# plink2 extraction, allele orientation, matrix multiply) is handled here.
# Edit the "CONFIG" block below once for your environment, then just run
# this script with a new VCF each time.

set -euo pipefail

# ── CONFIG: edit these once for your environment ────────────────────────
WEIGHTS_DB="/mnt/data_40T/anywaycheng/UploadFile/GA/GA_Weight_DataBase/V4_3Fold_Sign_weights.db"
WEIGHT_COL="GA_W"                 # or PrediXcan_W to predict with uncorrected weights for comparison
PREDICT_PY="$(dirname "$0")/GeneExpression_Predict.py"
PLINK2="plink2"                   # path to plink2 binary, or just "plink2" if on $PATH
THREADS=8
MEMORY_MB=16000
# ──────────────────────────────────────────────────────────────────────

VCF="/mnt/data_40T/anywaycheng/UploadFile/GA/Data_For_GA/Data_For_GA/WholeGenome_RSID.vcf.gz"
OUT_DIR="/mnt/data_40T/anywaycheng/UploadFile/GA/PredictTesting"
TMP_DIR="${OUT_DIR}/tmp"

mkdir -p "$OUT_DIR" "$TMP_DIR"

echo "=================================================="
echo " GA-corrected PrediXcan prediction pipeline"
echo "   VCF        : $VCF"
echo "   Weights DB : $WEIGHTS_DB"
echo "   Weight col : $WEIGHT_COL"
echo "   Output dir : $OUT_DIR"
echo "=================================================="

# ── Step 1: get the list of SNPs actually needed (huge speed win -- we only
#            ever touch the SNPs the model uses, not the whole VCF) ────────
echo "[1/3] Extracting required SNP list from weights.db ..."
sqlite3 "$WEIGHTS_DB" "SELECT DISTINCT rsid FROM weights WHERE ${WEIGHT_COL} != 0;" \
    > "${TMP_DIR}/used_snps.txt"
N_SNPS=$(wc -l < "${TMP_DIR}/used_snps.txt")
echo "      -> ${N_SNPS} distinct SNPs needed"

# ── Step 2: plink2 extracts dosage only for those SNPs ─────────────────────
echo "[2/3] Running plink2 to extract dosage (only for the ${N_SNPS} needed SNPs) ..."
"$PLINK2" \
    --vcf "$VCF" \
    --extract "${TMP_DIR}/used_snps.txt" \
    --export A \
    --threads "$THREADS" \
    --memory "$MEMORY_MB" \
    --out "${TMP_DIR}/dosage_matrix" \
    > "${TMP_DIR}/plink2.log" 2>&1
echo "      -> dosage matrix: ${TMP_DIR}/dosage_matrix.raw (log: ${TMP_DIR}/plink2.log)"

# ── Step 3: matrix-multiply weights x dosage, write outputs ────────────────
echo "[3/3] Computing predicted expression ..."
python3 "$PREDICT_PY" \
    --dosage_raw "${TMP_DIR}/dosage_matrix.raw" \
    --weights_db "$WEIGHTS_DB" \
    --weight_col "$WEIGHT_COL" \
    --out_prefix "${OUT_DIR}/predicted"

echo "=================================================="
echo " Done. Outputs:"
echo "   ${OUT_DIR}/predicted_expression.csv"
echo "   ${OUT_DIR}/predicted_summary.csv"
echo "=================================================="
