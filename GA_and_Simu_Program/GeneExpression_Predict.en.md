# GeneExpression_Predict.md — GA-Calibrated Gene Expression Prediction Pipeline

## 1. What This Does

This is the **downstream application** of the `GA.py` calibration pipeline: it takes a user-supplied genotype file `genotype.vcf`, applies the GA-calibrated weights (`GA_W`) from `V4_3Fold_Sign_weights.db`, and computes predicted expression for each sample and gene using the **PrediXcan prediction formula**.

```
PredictedExpression(gene, sample) = Σ_i  weight(gene, SNP_i) × dosage(SNP_i, sample)
```

`dosage` is the count of the effect allele (0/1/2) that sample carries at that SNP. This is exactly the same formula the original PrediXcan uses; the only difference is that `weight` here is the GA-calibrated version better suited for transfer to the EAS population, rather than the weights originally trained for the EUR population.

## 2. Role of the Three Files

| File | Role |
|---|---|
| **`GeneExpression_Predict.sh`** | The only entry point the user needs to run. Given a `genotype.vcf`, it automatically completes all three steps: "extract the needed SNPs → compute dosage with plink2 → call GeneExpression_Predict.py to compute predictions". |
| **`GeneExpression_Predict.py`** | The actual computation core. Reads the dosage matrix produced by plink2 plus `weights.db`, performs allele-orientation correction and sparse matrix multiplication, and outputs predicted expression and a summary. Not normally called directly — it's invoked with the right arguments automatically by `GeneExpression_Predict.sh`. |
| **`GeneExpression_Predict.md`** | This document. |

## 3. Usage

### Requirements

- **plink2** (used to quickly extract only the needed SNP dosage from a VCF; install it yourself and make sure it's on `$PATH`, or edit the `PLINK2` variable in `GeneExpression_Predict.sh` to point to the executable)
- **sqlite3** CLI
- Python packages: `pip install pandas numpy scipy --break-system-packages`

### Running It

```bash
chmod +x GeneExpression_Predict.sh
./GeneExpression_Predict.sh genotype.vcf
```

Specify an output folder (defaults to `./predixcan_ga_output` if omitted):

```bash
./GeneExpression_Predict.sh genotype.vcf /path/to/my_output_dir
```

**The user only needs to supply this one `genotype.vcf`** — everything else (the `weights.db` path, plink2 parameters, weight column) is already set in the `CONFIG` block at the top of `GeneExpression_Predict.sh`, defaulting to `V4_3Fold_Sign_weights.db`. To switch to a different `.db` (e.g. `V4_LOOCV_Sign_weights.db`) or to compare "before vs. after GA calibration", just change these lines:

```bash
WEIGHTS_DB="/mnt/data_40T/anywaycheng/UploadFile/GA/GA_Weight_DataBase/V4_3Fold_Sign_weights.db"
WEIGHT_COL="GA_W"        # change to PrediXcan_W to get the uncorrected weights as a control group
```

### What Happens During Execution

1. **[1/3]** Pulls the list of rsids used by all non-zero weights from `weights.db` (only a few thousand to tens of thousands — not the whole genome)
2. **[2/3]** `plink2 --extract` computes the dosage matrix from `genotype.vcf` for only that list — this is the key to controlling runtime, since plink2 never touches variants unrelated to the model
3. **[3/3]** `GeneExpression_Predict.py` performs allele-orientation correction (comparing `weights.db`'s `eff_allele` against the allele plink2 counted, using `2 − dosage` when the orientation differs), then computes predicted values for all genes and all samples with a single sparse matrix multiplication `W @ D`

## 4. Output File Formats

### 4.1 `predicted_predicted_expression.csv` (corresponds to PrediXcan's `*_predicted_expression.txt`)

**Wide format**, matching the official PrediXcan `Predict.py` output layout: one row per sample, one column per gene.

| FID | IID | ENSG00000000971 | ENSG00000001167 | ... |
|---|---|---|---|---|
| 1 | S0 | 0.1074 | -0.1747 | ... |
| 1 | S1 | 0.0480 | 0.0341 | ... |

- `FID`/`IID`: sample IDs from the VCF/plink2
- Every following column is a gene ID; the value is that sample's predicted expression (raw scale, computed directly from the summation formula; if you need standardized z-scores, apply `(x - mean) / std` to each column yourself)
- **Only genes with at least 1 SNP found in this VCF appear here**; genes with zero SNP hits do not appear in this file at all (check `predicted_summary.csv` — its `status` will be `NO_SNPS_IN_VCF`)

### 4.2 `predicted_summary.csv` (corresponds to PrediXcan's `*_summary.txt`)

**One row per gene**, covering every gene in `weights.db` (not just the ones predicted successfully). Column design follows the QC logic of PrediXcan's official summary output, with additional GA-pipeline-specific quality fields:

| Column | Description | Source |
|---|---|---|
| `gene` | Gene ID | — |
| `n_snps_in_model` | Number of non-zero-weight SNPs after GA calibration (i.e. the set of SNPs GA judged to genuinely contribute) | `weights.db` |
| `n_snps_used` | Number of SNPs actually found in `genotype.vcf` and successfully orientation-aligned in this prediction run | computed this run |
| `pct_snps_used` | `n_snps_used / n_snps_in_model` — represents this gene's model coverage on this VCF | computed this run |
| `pred_mean` / `pred_sd` | Mean and standard deviation of predicted expression across this batch of samples | computed this run |
| `var_g` | Variance of predicted expression (`pred_sd²`); a standard PrediXcan QC metric — `var_g = 0` means this gene has zero discriminative power in this batch of samples | computed this run |
| `avg_confidence_score` | Average `confidence_score` (computed by `BuildDB.py`) across the SNPs actually used | `weights.db` (GA-pipeline-specific) |
| `train_corr` / `corr_after_GA` | This gene's overall performance during GA training/validation (from `final_summary.csv`); can be used as a reference indicator of whether the gene's model itself is trustworthy | `weights.db`'s `gene_summary` table (GA-pipeline-specific) |
| `n_snps` | Total number of original SNPs used when training this gene with GA (including ones later pruned) | `gene_summary` |
| `status` | This gene's quality flag for this prediction run — see the table below | computed this run |

**`status` reference:**

| status | Meaning |
|---|---|
| `OK` | Normal — `n_snps_used` coverage ≥ 50% |
| `LOW_SNP_COVERAGE` | Some SNPs found, but coverage < 50%; predicted value is less trustworthy, use with caution |
| `NO_SNPS_IN_VCF` | None of this gene's model SNPs in `weights.db` appear anywhere in this VCF (common causes: VCF variant IDs aren't rsids, or this VCF doesn't cover those positions); this gene **will not** appear in `predicted_predicted_expression.csv` |
| `NO_MODEL_SNPS` | This gene was entirely pruned during the GA stage (`n_snps_in_model = 0`) — it had no usable weights to begin with, unrelated to this VCF |
| `ZERO_VARIANCE` | A predicted value was computed, but every sample in this batch has the same value (`var_g = 0`); this gene has no discriminative power in this batch of samples — by PrediXcan convention such genes are usually filtered out in downstream analysis |

> It's recommended that downstream analysis first filter to genes with `status == 'OK'` (or relax to also include `LOW_SNP_COVERAGE` depending on the situation), to avoid treating predictions with extremely low coverage or no discriminative power as trustworthy results.

## 5. Performance and Runtime

- **Only the needed SNPs are processed**: Steps 1/2 let plink2 skip every variant unrelated to the model entirely, shrinking the data volume from "the whole genome" down to "the few thousand to tens of thousands of SNPs the model actually uses".
- **A single sparse matrix multiplication**: rather than running a regression per gene, `GeneExpression_Predict.py` does one `(gene × SNP) sparse matrix @ (SNP × sample) matrix` multiplication — for close to 5000 genes and thousands of samples this is a sub-second computation.
- The overall bottleneck is usually the plink2 VCF-reading step, which scales with VCF file size / sample count, not gene count (since the SNP list is already fixed).

## 6. Common Scenarios

**Outputting both "before vs. after GA calibration" predictions for comparison:**

```bash
# After calibration (default)
./GeneExpression_Predict.sh genotype.vcf output_ga

# Use the uncorrected PrediXcan_W as a control group
sed 's/WEIGHT_COL="GA_W"/WEIGHT_COL="PrediXcan_W"/' GeneExpression_Predict.sh > GeneExpression_Predict_predixcan.sh
chmod +x GeneExpression_Predict_predixcan.sh
./GeneExpression_Predict_predixcan.sh genotype.vcf output_predixcan
```

**VCF variant IDs aren't rsids (`--extract` finds nothing):**

```bash
bcftools annotate -c ID -a dbSNP.vcf.gz genotype.vcf.gz -Oz -o genotype.rsid.vcf.gz
gunzip -k genotype.rsid.vcf.gz   # if you need an uncompressed .vcf
./GeneExpression_Predict.sh genotype.rsid.vcf
```

**Just want to manually rerun Step 3 (already have dosage_matrix.raw, don't want to rerun plink2):**

```bash
python3 GeneExpression_Predict.py \
    --dosage_raw predixcan_ga_output/tmp/dosage_matrix.raw \
    --weights_db "/mnt/data_40T/anywaycheng/UploadFile/GA/GA_Weight_DataBase/V4_3Fold_Sign_weights.db" \
    --out_prefix predixcan_ga_output/predicted
```
