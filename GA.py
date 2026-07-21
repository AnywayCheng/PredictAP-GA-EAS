import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import json
import logging
import seaborn as sns
import sqlite3 as sql
import time
from datetime import timedelta

import pandas as pd
import numpy as np
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from numba import cuda
import cupy as cp
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RepeatedKFold
import gc
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_args():
    """
    Parse command-line arguments for the GPU-accelerated GA weight optimizer.

    Args:
        None (reads from sys.argv via argparse).

    Returns:
        argparse.Namespace: parsed CLI arguments covering GA hyperparameters,
        SNP tier thresholds, sign-flip settings, TAS filter settings, adaptive
        scaling settings, GPU batching, test mode, and cross-validation config.

    Notes:
        - s_tier/m_tier/l_tier bounds define the multiplicative scaling range
          allowed for each SNP tier during GA search (tiers are assigned based
          on EUR-EAS allele-frequency difference; see build_ga_params/
          process_single_gene_gpu for how *_af_diff_max is converted into an
          allele-similarity threshold).
        - allow_sign_flip enables negative scaling factors for M/L-tier SNPs so
          the GA can flip the direction of a SNP's contribution when the
          EUR-trained sign appears unreliable in EAS.
        - tas_decay_scale / tas_keep_frac_ml / tas_min_snps_to_filter control the
          Transferability-Adjusted Score (TAS) pre-filter that drops the least
          transferable M/L-tier SNPs before GA optimization.
        - adaptive_scale (and related min_population/min_iter1-3/adaptive_ref_snps)
          let population size and per-phase iteration counts shrink for genes with
          few SNPs, saving GPU time without sacrificing quality on larger genes.
    """
    parser = argparse.ArgumentParser(description="Batch GA Optimization (GPU) - V3")

    # === General GA Parameters ===
    parser.add_argument("--config", type=str, default="config.json", help="Path to JSON config")
    parser.add_argument("--GA_Phase1_iterations", type=int, default=500)
    parser.add_argument("--GA_Phase2_iterations", type=int, default=750)
    parser.add_argument("--GA_Phase3_iterations", type=int, default=250)
    parser.add_argument("--GA_population_sizes", type=int, default=500)
    parser.add_argument("--Numbers_of_GA_runs", type=int, default=50)
    parser.add_argument("--Initial_mutation_rate", type=float, default=0.5)

    # === GA Weighting Parameters ===
    parser.add_argument("--s_tier_af_diff_max", type=float, default=0.05,
                        help="Max AF diff for S-tier SNPs")
    parser.add_argument("--s_tier_low_bound", type=float, default=0.9)
    parser.add_argument("--s_tier_high_bound", type=float, default=1.1)
    parser.add_argument("--m_tier_af_diff_max", type=float, default=0.25,
                        help="Max AF diff for M-tier SNPs")
    parser.add_argument("--m_tier_low_bound", type=float, default=0.5)
    parser.add_argument("--m_tier_high_bound", type=float, default=3.0)
    parser.add_argument("--l_tier_low_bound", type=float, default=0.1)
    parser.add_argument("--l_tier_high_bound", type=float, default=5.0)
    parser.add_argument("--pruning_threshold", type=float, default=1e-8,
                        help="Threshold for hard pruning weights after GA")

    # === V3 addition: sign-flip and fitness hyperparameters ===
    parser.add_argument("--allow_sign_flip", action="store_true",
                        help="Allow negative scaling for L/M-tier SNPs (can flip prediction direction); S-tier stays same-sign")
    parser.add_argument("--sign_flip_threshold", type=float, default=0.01,
                        help="baseline corr lower than 0.01 than try signed flipped")

    # === TAS filter parameters ===
    parser.add_argument("--tas_decay_scale", type=float, default=0.20,
                        help="Decay scale for TAS score transferability")
    parser.add_argument("--tas_keep_frac_ml", type=float, default=0.8,
                        help="Fraction of M/L-tier SNPs to keep")
    parser.add_argument("--tas_min_snps_to_filter", type=int, default=5,
                        help="If a gene has fewer than this many SNPs, it will not be filtered")
    parser.add_argument("--adaptive_scale", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable adaptive pop & iter scaling (--adaptive_scale / --no-adaptive_scale)")
    parser.add_argument("--adaptive_ref_snps", type=int, default=20,
                        help="Reference SNP count for adaptive scaling (full config at this count)")
    parser.add_argument("--min_population", type=int, default=300,
                        help="Minimum population size after adaptive scaling")
    parser.add_argument("--min_iter1", type=int, default=200,
                        help="Minimum Phase 1 iterations after adaptive scaling")
    parser.add_argument("--min_iter2", type=int, default=300,
                        help="Minimum Phase 2 iterations after adaptive scaling")
    parser.add_argument("--min_iter3", type=int, default=100,
                        help="Minimum Phase 3 iterations after adaptive scaling")

    # === GPU Batch Control ===
    parser.add_argument("--batch_limit", type=int, default=2048,
                        help="Max parallel GA tasks per GPU batch")
    parser.add_argument("--plot_thread", type=int, default=4)

    # === Test Mode Arguments ===
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--test_count", type=int, default=50)
    parser.add_argument("--test_seed", type=int, default=42)

    # === Cross-Validation Parameters ===
    parser.add_argument("--cv_method", type=str, default="loocv",
                        choices=["loocv", "repeated_kfold"],
                        help="Cross-validation strategy")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--cv_seed", type=int, default=42)

    return parser.parse_args()


# ──────────────────────────────────────────────
# Data loading and preprocessing
# ──────────────────────────────────────────────

def reading_file_path(config):
    """
    Load all raw input datasets referenced by the JSON config.

    Args:
        config (dict): parsed config.json contents. Must contain the keys
            'genotype_path', 'predict_expr_path', 'real_expr_path',
            'allele_freq_path', 'annotation_path', 'db_path', 'out_dir'.

    Returns:
        tuple:
            genotype (dask.dataframe.DataFrame): genotype matrix loaded lazily from parquet.
            predict_expr (pd.DataFrame): raw PrediXcan-predicted expression table (samples x genes).
            real_expr (pd.DataFrame): observed/real expression table (genes x samples).
            allele_freq (pd.DataFrame): EUR/EAS allele frequency table keyed by RSID.
            annotation (pd.DataFrame): gene/SNP annotation table.
            db_path (str): path to the PrediXcan SQLite weights database.
            out_dir (str): output directory for results.

    Notes:
        Uses config.get() so a missing key yields None rather than raising, then
        explicitly validates that all required paths are present before loading.
    """
    genotype_path    = config.get('genotype_path')
    predict_expr_path = config.get('predict_expr_path')
    real_expr_path   = config.get('real_expr_path')
    allele_freq_path = config.get('allele_freq_path')
    annotation_path  = config.get('annotation_path')
    db_path          = config.get('db_path')
    out_dir          = config.get('out_dir')

    if not all([genotype_path, predict_expr_path, real_expr_path,
                allele_freq_path, annotation_path, db_path, out_dir]):
        raise ValueError("Config file is missing one or more required paths.")

    genotype = dd.read_parquet(genotype_path)

    if predict_expr_path.endswith('.txt') or predict_expr_path.endswith('.tsv'):
        predict_expr = pd.read_csv(predict_expr_path, sep='\t')
    else:
        predict_expr = pd.read_csv(predict_expr_path)

    real_expr   = pd.read_csv(real_expr_path)
    allele_freq = pd.read_csv(allele_freq_path, sep='\t')
    annotation  = pd.read_csv(annotation_path, sep=',')

    return genotype, predict_expr, real_expr, allele_freq, annotation, db_path, out_dir


def PrediXcanResultDealing(expr_df):
    """
    Reshape a raw PrediXcan prediction table into gene-indexed, sample-column format.

    Args:
        expr_df (pd.DataFrame): raw PrediXcan output with columns ['FID', 'IID', gene1, gene2, ...],
            one row per sample.

    Returns:
        pd.DataFrame: transposed table indexed by 'EnsemblID' (gene, versionless) with one column
        per sample (IID).

    Notes:
        Drops the version suffix from Ensembl gene IDs (e.g. 'ENSG00000123.4' -> 'ENSG00000123')
        so gene IDs can be matched against the real-expression and weight tables.
    """
    df_temp = expr_df.set_index('IID').drop(columns=['FID'])
    df_transposed = df_temp.T
    df_transposed.index = df_transposed.index.str.split('.').str[0]
    df_transposed.index.name = 'EnsemblID'
    return df_transposed


def get_db_info(db_path):
    """
    Read PrediXcan weight and gene metadata tables from the model's SQLite database.

    Args:
        db_path (str): filesystem path to the PrediXcan .db SQLite file.

    Returns:
        tuple:
            weight_df (pd.DataFrame): columns include ['gene', 'rsid', 'weight', ...] — one row
                per SNP-gene weight entry.
            gene_info_df (pd.DataFrame): gene-level metadata from the 'extra' table.
    """
    with sql.connect(db_path) as con:
        weight_df   = pd.read_sql_query("SELECT * FROM weights;", con)
        gene_info_df = pd.read_sql_query("SELECT * FROM extra;", con)
    return weight_df, gene_info_df


def db_allele_freq_filtered(weight_df, allele_freq,
                             eur_maf_thresh=0.05,
                             delta_af_low_maf_thresh=0.10,
                             delta_af_mid_maf_thresh=0.20,
                             mid_maf_boundary=0.20):
    """
    Merge PrediXcan weights with EUR/EAS allele frequencies and drop SNPs unlikely to transfer.

    Args:
        weight_df (pd.DataFrame): PrediXcan weights with columns ['gene', 'rsid', 'weight'].
        allele_freq (pd.DataFrame): allele frequency table with columns ['RSID', 'EUR_AF', 'EAS_AF'].
        eur_maf_thresh (float): minimum EUR allele frequency required to trust the trained weight (Rule 3).
        delta_af_low_maf_thresh (float): max allowed EAS-EUR AF gap when EUR AF is in
            [eur_maf_thresh, mid_maf_boundary) (Rule 4a).
        delta_af_mid_maf_thresh (float): max allowed EAS-EUR AF gap when EUR AF >= mid_maf_boundary (Rule 4b).
        mid_maf_boundary (float): EUR AF cutoff separating the low-MAF and mid/high-MAF delta-AF rules.

    Returns:
        pd.DataFrame: subset of the merged weight/AF table (columns ['gene','rsid','weight',
        'EUR_AF','EAS_AF']) with unreliable SNPs excluded, index reset.

    Notes:
        Exclusion rules (union of all four flags a SNP is dropped):
          Rule 1: EAS_AF < 0.01 (near-monomorphic in EAS, weight uninformative there).
          Rule 2: |weight| < 1e-4 (negligible contribution to predicted expression).
          Rule 3: EUR_AF < eur_maf_thresh (weight was heavily shrunk during EUR training, untrustworthy).
          Rule 4a/4b: EUR_AF < EAS_AF with a large gap, split by EUR MAF band, indicating
              LD-structure divergence between populations that breaks transferability.
    """
    db_weight_freq_table = (
        pd.merge(weight_df, allele_freq, left_on="rsid", right_on="RSID", how="inner")
          .loc[:, ["gene", "rsid", "weight", "EUR_AF", "EAS_AF"]]
    )

    n_total = len(db_weight_freq_table)

    rule1 = db_weight_freq_table['EAS_AF'] < 0.01
    rule2 = db_weight_freq_table['weight'].abs() < 1e-4
    rule3 = db_weight_freq_table['EUR_AF'] < eur_maf_thresh

    eur_lt_eas = db_weight_freq_table['EUR_AF'] < db_weight_freq_table['EAS_AF']
    delta_af   = db_weight_freq_table['EAS_AF'] - db_weight_freq_table['EUR_AF']
    eur_af     = db_weight_freq_table['EUR_AF']

    rule4a = (eur_lt_eas &
              eur_af.between(eur_maf_thresh, mid_maf_boundary, inclusive='left') &
              (delta_af > delta_af_low_maf_thresh))

    rule4b = (eur_lt_eas &
              (eur_af >= mid_maf_boundary) &
              (delta_af > delta_af_mid_maf_thresh))

    exclude_mask = rule1 | rule2 | rule3 | rule4a | rule4b

    logger.info(f"  SNP filtering summary (total: {n_total}):")
    logger.info(f"    Rule 1 (EAS AF < 0.01)                    : {rule1.sum():>6}")
    logger.info(f"    Rule 2 (|weight| < 1e-4)                  : {rule2.sum():>6}")
    logger.info(f"    Rule 3 (EUR MAF < {eur_maf_thresh})              : {rule3.sum():>6}")
    logger.info(f"    Rule 4a (low EUR MAF, ΔAF > {delta_af_low_maf_thresh})     : {rule4a.sum():>6}")
    logger.info(f"    Rule 4b (mid EUR MAF, ΔAF > {delta_af_mid_maf_thresh})     : {rule4b.sum():>6}")
    logger.info(f"    Total excluded (union)                     : {exclude_mask.sum():>6}")
    logger.info(f"    Retained                                   : {(~exclude_mask).sum():>6}")

    return db_weight_freq_table[~exclude_mask].reset_index(drop=True)


def data_intersection(predict_expr, db_weight_table, real_expr):
    """
    Restrict predicted expression, weights, and real expression to their common genes/samples.

    Args:
        predict_expr (pd.DataFrame): shape (G_genes, M_samples), gene-indexed PrediXcan predictions
            (output of PrediXcanResultDealing).
        db_weight_table (pd.DataFrame): PrediXcan weight table with a 'gene' column.
        real_expr (pd.DataFrame): shape (G_genes, N_samples) with an 'EnsemblID' column to be set as index.

    Returns:
        tuple:
            predict_expr_filtered (pd.DataFrame): shape (G_common, S_common), restricted to common genes/samples.
            db_weight_table_filtered (pd.DataFrame): weight rows restricted to common genes.
            real_expr_filtered (pd.DataFrame): shape (G_common, S_common), restricted to common genes/samples.

    Notes:
        Gene IDs in db_weight_table are stripped of Ensembl version suffixes before intersecting,
        matching the versionless IDs produced by PrediXcanResultDealing.
    """
    predict_expr     = predict_expr.copy()
    db_weight_table  = db_weight_table.copy()
    real_expr        = real_expr.copy()
    real_expr        = real_expr.set_index('EnsemblID')

    db_weight_table['gene'] = db_weight_table['gene'].astype(str).str.replace(r"\.\d+$", "", regex=True)

    common_genes = sorted(set(predict_expr.index) &
                          set(real_expr.index) &
                          set(db_weight_table['gene']))

    real_samples    = list(real_expr.columns)
    predict_samples = predict_expr.columns.tolist()
    common_samples  = sorted(set(real_samples) & set(predict_samples))

    predict_expr_filtered    = predict_expr.loc[common_genes, common_samples]
    real_expr_filtered       = real_expr.loc[common_genes, common_samples]
    db_weight_table_filtered = db_weight_table[db_weight_table['gene'].isin(common_genes)]

    return predict_expr_filtered, db_weight_table_filtered, real_expr_filtered


# ──────────────────────────────────────────────
# Population initialization (supports sign-flip)
# ──────────────────────────────────────────────

def Weight_Generate_Population_Cupy(rsid, af_vec, num_batches, target_size=None, **kwargs):
    """
    Generate a random initial GA population of per-SNP weight-scaling factors on GPU.

    Args:
        rsid (array-like): length n_snps, SNP identifiers (used only for length/order).
        af_vec (cupy.ndarray): shape (n_snps,), per-SNP allele-similarity score in [0, 1]
            (1 - |EUR_AF - EAS_AF|, clipped); higher means more transferable.
        num_batches (int): number of independent GA batches (e.g. folds x runs) to generate in parallel.
        target_size (int, optional): population size to generate; overrides kwargs['population_size'] if given.
        **kwargs: GA hyperparameters, notably:
            population_size (int): default population size if target_size is None.
            allow_sign_flip (bool): if True, M/L-tier scaling values can be negative.
            s_tier_threshold/s_tier_low/s_tier_high: S-tier af_vec cutoff and uniform sampling range.
            m_tier_threshold/m_tier_low/m_tier_high: M-tier af_vec cutoff and bimodal sampling range.
            l_tier_low/l_tier_high: L-tier bimodal sampling range.

    Returns:
        cupy.ndarray: shape (num_batches, population_size, n_snps + 1), float32. The last column
        (index n_snps) is the fitness slot, left at 0 here (fitness is filled in later by
        Weight_Fit_Value_Calculate_Cupy).

    Notes:
        SNPs are stratified into three transferability tiers based on af_vec:
          S-tier (af_vec >= s_thresh): AF difference is small, so direction is reliable;
              scaling is always sampled positive from a narrow range [s_low, s_high].
          M-tier (m_thresh <= af_vec < s_thresh): sampled bimodally from
              [m_low, s_low] union [s_high, m_high] (50/50) to avoid the S-tier center band;
              if allow_sign_flip, each draw additionally gets a random +/- sign.
          L-tier (af_vec < m_thresh): sampled bimodally from [l_low, m_low] union [m_high, l_high],
              same sign-flip behavior as M-tier.
        After sampling, values are clipped to their tier's valid range (symmetric range when
        sign-flip is enabled, [0, l_high] otherwise) to guard against numerical drift.
    """
    if target_size is not None:
        population_size = int(target_size)
    else:
        population_size = int(kwargs.get('population_size', 500))

    allow_sign_flip = kwargs.get('allow_sign_flip', False)

    s_thresh = kwargs.get('s_tier_threshold', 0.95)
    s_low    = kwargs.get('s_tier_low', 0.8)
    s_high   = kwargs.get('s_tier_high', 1.2)

    m_thresh = kwargs.get('m_tier_threshold', 0.75)
    m_low    = kwargs.get('m_tier_low', 0.3)
    m_high   = kwargs.get('m_tier_high', 3.0)

    l_low  = kwargs.get('l_tier_low', 0.0)
    l_high = kwargs.get('l_tier_high', 7.0)

    n_snps = len(rsid)
    population    = cp.zeros((num_batches, population_size, n_snps + 1), dtype=cp.float32)
    weight_scaling = population[:, :, :n_snps]

    is_s_tier = af_vec >= s_thresh
    is_m_tier = (af_vec >= m_thresh) & (~is_s_tier)
    is_l_tier = af_vec < m_thresh

    # S-tier: always positive (small AF diff, direction is reliable)
    if is_s_tier.any():
        weight_scaling[:, :, is_s_tier] = cp.random.uniform(
            s_low, s_high,
            size=(num_batches, population_size, int(is_s_tier.sum()))
        )

    # M-tier: bimodal [m_low, s_low] + [s_high, m_high], 50/50, avoiding the S-tier center band
    if is_m_tier.any():
        n_m   = int(is_m_tier.sum())
        shape = (num_batches, population_size, n_m)
        seg1  = cp.random.uniform(m_low,  s_low,  size=shape)
        seg2  = cp.random.uniform(s_high, m_high, size=shape)
        sel   = cp.random.uniform(0.0, 1.0, size=shape) < 0.5
        vals  = cp.where(sel, seg1, seg2)
        if allow_sign_flip:
            signs = cp.where(cp.random.uniform(0.0, 1.0, size=shape) < 0.5, 1.0, -1.0)
            vals  = vals * signs
        weight_scaling[:, :, is_m_tier] = vals

    # L-tier: bimodal [l_low, m_low] + [m_high, l_high], 50/50, avoiding the M-tier center band
    if is_l_tier.any():
        n_l   = int(is_l_tier.sum())
        shape = (num_batches, population_size, n_l)
        seg1  = cp.random.uniform(l_low,  m_low,  size=shape)
        seg2  = cp.random.uniform(m_high, l_high, size=shape)
        sel   = cp.random.uniform(0.0, 1.0, size=shape) < 0.5
        vals  = cp.where(sel, seg1, seg2)
        if allow_sign_flip:
            signs = cp.where(cp.random.uniform(0.0, 1.0, size=shape) < 0.5, 1.0, -1.0)
            vals  = vals * signs
        weight_scaling[:, :, is_l_tier] = vals

    # Clip: M-tier magnitude capped at m_high, L-tier magnitude capped at l_high
    if allow_sign_flip:
        if is_s_tier.any():
            weight_scaling[:, :, is_s_tier] = cp.clip(
                weight_scaling[:, :, is_s_tier], s_low, s_high
            )
        if is_m_tier.any():
            weight_scaling[:, :, is_m_tier] = cp.clip(
                weight_scaling[:, :, is_m_tier], -m_high, m_high
            )
        if is_l_tier.any():
            weight_scaling[:, :, is_l_tier] = cp.clip(
                weight_scaling[:, :, is_l_tier], -l_high, l_high
            )
    else:
        cp.clip(weight_scaling, 0.0, l_high, out=weight_scaling)

    return population


# ──────────────────────────────────────────────
# Fitness calculation (Fisher z-transform loss)
# ──────────────────────────────────────────────

def Weight_Fit_Value_Calculate_Cupy(backend, population, rsid_count, mask=None, y_target=None, **kwargs):
    """
    Compute the GA fitness (loss) for every individual in the population on GPU.

    Args:
        backend (tuple): (G, w, y_backend, a) where
            G (cupy.ndarray): shape (n_snps, n_samples), genotype dosage matrix.
            w (cupy.ndarray): shape (n_snps,), original PrediXcan weights.
            y_backend (cupy.ndarray or None): shape (1, n_samples) or similar, fallback target
                expression if y_target is not supplied.
            a (cupy.ndarray): shape (n_snps,), allele-similarity factor (a_vec).
        population (cupy.ndarray): shape (n_batches, pop_size, n_snps + 1), individuals whose
            last column will be overwritten with fitness.
        rsid_count (int): number of SNPs (n_snps), used to slice out the scaling-factor columns.
        mask (cupy.ndarray, optional): shape (n_batches, 1, n_samples), 1 = training sample,
            0 = held-out sample for this batch/fold. If None, all samples are used.
        y_target (cupy.ndarray, optional): shape (n_batches, 1, n_samples), per-batch normalized
            target expression (e.g. fold-specific z-scored y). Falls back to y_backend if None.
        **kwargs: unused extra GA hyperparameters (absorbed for call-signature compatibility).

    Returns:
        cupy.ndarray: the same `population` array, in-place modified so that
        population[:, :, -1] holds each individual's fitness (lower is better).

    Notes:
        Prediction: pred = (P * w * a) @ G, where P is the per-individual scaling matrix.
        Correlation is computed with masked mean/std so held-out samples are excluded from
        training-fitness statistics when `mask` is given.
        Loss = -arctanh(corr) (Fisher z-transform): continuous and monotonic with no cliffs or
        truncation near corr = +/-1; corr -> +1 gives loss -> -inf (best), corr = 0 gives loss = 0,
        corr -> -1 gives loss -> +inf (worst). This keeps the fitness landscape smooth even for
        near-perfect or near-anti correlated candidates, unlike using 1 - corr or MSE-of-corr.
        An entropy-evenness penalty (beta = 1e-3) is added to discourage the population from
        collapsing onto a few dominant SNPs, preserving diversity for exploration.
        NaN fitness values (e.g. from degenerate/constant predictions) are replaced with 1e6 so
        they are effectively unselectable.
    """
    G, w, y_backend, a = backend
    y_true = y_target if y_target is not None else y_backend

    P = population[:, :, :rsid_count]

    w_eff         = w * a
    scaled_weights = P * w_eff
    pred           = cp.matmul(scaled_weights, G)

    if mask is not None:
        n_train    = mask.sum(axis=2, keepdims=True)
        p_mean     = (pred * mask).sum(axis=2, keepdims=True) / n_train
        p_centered = (pred - p_mean) * mask
        p_std      = cp.sqrt((p_centered ** 2).sum(axis=2, keepdims=True) / n_train + 1e-12)

        y_mean     = (y_true * mask).sum(axis=2, keepdims=True) / n_train
        y_centered = (y_true - y_mean) * mask
        y_std      = cp.sqrt((y_centered ** 2).sum(axis=2, keepdims=True) / n_train + 1e-12)

        covariance  = (p_centered * y_centered).sum(axis=2)
        std_product = (p_std * y_std * n_train).squeeze(axis=2)
        mae         = (cp.abs(pred - y_true) * mask).sum(axis=2) / n_train.squeeze(axis=2)
    else:
        p_mean     = pred.mean(axis=2, keepdims=True)
        p_std      = pred.std(axis=2, keepdims=True) + 1e-12
        y_mean     = y_true.mean(axis=2, keepdims=True)
        y_std      = y_true.std(axis=2, keepdims=True) + 1e-12
        p_centered = pred - p_mean
        y_centered = y_true - y_mean
        covariance  = cp.sum(p_centered * y_centered, axis=2)
        std_product = (p_std * y_std * pred.shape[2]).squeeze(axis=2)
        mae         = cp.mean(cp.abs(pred - y_true), axis=2)

    correlation = covariance / (std_product + 1e-12)

    # Fisher z-transform loss: -arctanh(corr), continuous and monotonic, no cliffs/truncation.
    corr_clipped = cp.clip(correlation, -0.9999, 0.9999)
    loss_corr    = -cp.arctanh(corr_clipped)

    # Composite loss (MAE term currently disabled; corr-only loss).
    loss  = loss_corr

    # Entropy-evenness penalty to maintain population diversity.
    beta = 1e-3
    P_abs          = cp.abs(P) + 1e-12
    p_sum          = P_abs.sum(axis=2, keepdims=True)
    p_prob         = P_abs / p_sum
    entropy        = -(p_prob * cp.log(p_prob)).sum(axis=2)
    max_entropy    = cp.log(cp.maximum(cp.array(rsid_count), 2))
    evenness_penalty = beta * (1.0 - (entropy / (max_entropy + 1e-12)))

    fit = loss + evenness_penalty
    population[:, :, -1] = cp.where(cp.isnan(fit), 1e6, fit)

    return population


# ──────────────────────────────────────────────
# Selection
# ──────────────────────────────────────────────

def Weight_Selection_Cupy(population):
    """
    Select parent individuals via fitness-weighted stochastic sampling (per batch).

    Args:
        population (cupy.ndarray): shape (n_batches, pop_size, n_snps + 1), sorted or unsorted
            individuals; last column is fitness (lower is better).

    Returns:
        cupy.ndarray: shape (n_batches, n_parents, n_snps + 1), the selected parent individuals,
        where n_parents = ceil-to-even(pop_size * 0.5).

    Notes:
        Uses an inverse-square fitness transform (adj_fitness) so lower-fitness (better)
        individuals get exponentially higher selection weight, combined with Gumbel-style noise
        (-log(uniform) / adj_fitness) to convert weighted sampling into an argsort — this avoids
        explicit cumulative-probability sampling loops and is fully vectorized on GPU.
    """
    n_batches, pop_size, _ = population.shape

    n_parents = int(pop_size * 0.5)
    if n_parents % 2 != 0:
        n_parents += 1

    fit_values = population[:, :, -1]

    # Inverse square fitness for minimization (lower fit_value = higher probability);
    # epsilon avoids division by zero.
    fit_min     = fit_values.min(axis=1, keepdims=True)
    adj_fitness = (1.0 / (fit_values - fit_min + 1.0)) ** 2
    random_noise = cp.random.rand(n_batches, pop_size)
    scores       = -cp.log(random_noise + 1e-12) / (adj_fitness + 1e-12)
    all_indices  = cp.argsort(scores, axis=1)

    selected_indices  = all_indices[:, :n_parents]
    indices_expanded  = selected_indices[:, :, cp.newaxis]
    new_population    = cp.take_along_axis(population, indices_expanded, axis=1)

    return new_population


# ──────────────────────────────────────────────
# Crossover kernel (alpha-blending; supports negative values)
# ──────────────────────────────────────────────

@cuda.jit
def crossover_kernel(parents, perm_indices, alpha, children_out):
    """
    CUDA kernel: alpha-blend pairs of parents into two children per pair, per SNP, per batch.

    Args:
        parents (device array): shape (n_batches, n_parents, n_snps), parent scaling factors
            (fitness column already stripped).
        perm_indices (device array): shape (n_batches, n_parents), int32 indices giving each
            parent's crossover partner (a random permutation within the batch).
        alpha (device array): shape (n_batches, n_parents), float32 blending coefficients in [0, 1]
            drawn from a Beta(2, 2) distribution.
        children_out (device array): shape (n_batches, 2 * n_parents, n_snps), output buffer;
            written in-place by this kernel.

    Returns:
        None. Writes results into `children_out`.

    Notes:
        Thread/block indexing: threadIdx.x/blockIdx.x -> snp_idx, threadIdx.y/blockIdx.y ->
        parent_idx, blockIdx.z -> batch_idx (grid dims chosen so blockDim.z is always 1, one
        batch per z-block). Each thread computes one (batch, parent, snp) cell.
        For parent pair (g1 at parent_idx, g2 at its permuted partner) with blend factor a:
            child1 = a*g1 + (1-a)*g2   (written to children_out[:, parent_idx, :])
            child2 = a*g2 + (1-a)*g1   (written to children_out[:, parent_idx + n_parents, :])
        This linear blend naturally supports negative scaling values (sign-flip mode) since it is
        a simple affine combination, not clamped here.
    """
    snp_idx    = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    parent_idx = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    batch_idx  = cuda.blockIdx.z

    n_batches = parents.shape[0]
    n_parents = parents.shape[1]
    n_snps    = parents.shape[2]

    if batch_idx < n_batches and parent_idx < n_parents and snp_idx < n_snps:
        g1     = parents[batch_idx, parent_idx, snp_idx]
        p2_idx = perm_indices[batch_idx, parent_idx]
        g2     = parents[batch_idx, p2_idx, snp_idx]
        a      = alpha[batch_idx, parent_idx]

        c1_val = a * g1 + (1.0 - a) * g2
        c2_val = a * g2 + (1.0 - a) * g1

        # First half stores Child 1, second half stores Child 2.
        children_out[batch_idx, parent_idx,            snp_idx] = c1_val
        children_out[batch_idx, parent_idx + n_parents, snp_idx] = c2_val


def Weight_CrossOver_Cupy(parents):
    """
    Launch the crossover kernel to breed children from a batch of parent populations.

    Args:
        parents (cupy.ndarray): shape (n_batches, n_parents, n_snps + 1), parent individuals
            including the trailing fitness column (stripped internally before crossover).

    Returns:
        cupy.ndarray: shape (n_batches, 2 * n_parents, n_snps + 1), float32, children with a
        zeroed fitness column (to be filled by Weight_Fit_Value_Calculate_Cupy).

    Notes:
        Random pairing is generated by argsorting per-batch random keys (a cheap random
        permutation trick), and blend factors alpha are drawn from Beta(2, 2), which concentrates
        mass away from the extremes (0/1) for a more balanced blend than Uniform(0, 1).
        Grid/block layout: 32 threads along the SNP axis, 4 along the parent axis, one block per
        batch along z, matching crossover_kernel's indexing scheme.
    """
    p_genes = parents[:, :, :-1]
    n_batches, n_parents, n_snps = p_genes.shape

    rand_keys    = cp.random.rand(n_batches, n_parents, dtype=cp.float32)
    perm_indices = cp.argsort(rand_keys, axis=1).astype(cp.int32)
    alpha        = cp.random.beta(2, 2, size=(n_batches, n_parents)).astype(cp.float32)

    children = cp.zeros((n_batches, n_parents * 2, n_snps + 1), dtype=cp.float32)

    threads_per_block = (32, 4, 1)
    blocks_x = (n_snps    + 31) // 32
    blocks_y = (n_parents + 3)  // 4
    blocks_z = n_batches
    grid_dims = (blocks_x, blocks_y, blocks_z)

    crossover_kernel[grid_dims, threads_per_block](p_genes, perm_indices, alpha, children)

    return children


# ──────────────────────────────────────────────
# Mutation kernel (supports sign-flip)
# ──────────────────────────────────────────────

@cuda.jit
def mutation_kernel(population, target_idx, a_vec, rand_triggers, rand_vals,
                    mutation_rate, s_thresh, s_low, s_high,
                    m_thresh, m_low, m_high, l_low, l_high,
                    allow_sign_flip_int):
    """
    CUDA kernel: stochastically re-sample scaling values for targeted SNPs, tier-aware.

    Args:
        population (device array): shape (n_batches, n_pop, n_snps + 1), individuals to mutate
            in-place (only entries in `target_idx` are eligible).
        target_idx (device array): shape (n_targets,), int, indices (into the SNP axis) of the
            SNPs that this GA phase is allowed to mutate.
        a_vec (device array): shape (n_snps,), allele-similarity factor used to pick the tier.
        rand_triggers (device array): shape (n_batches, n_pop, n_targets), float32 in [0,1);
            mutation fires for a given cell when rand_triggers < mutation_rate.
        rand_vals (device array): shape (n_batches, n_pop, n_targets), float32 in [0,1); source of
            randomness for the new scaling value when mutation fires.
        mutation_rate (float): probability threshold that a given (batch, individual, target SNP) mutates.
        s_thresh, s_low, s_high (float): S-tier af_vec cutoff and uniform sampling bounds.
        m_thresh, m_low, m_high (float): M-tier af_vec cutoff and bimodal sampling bounds.
        l_low, l_high (float): L-tier bimodal sampling bounds.
        allow_sign_flip_int (int): 0 = disabled, 1 = enabled (CUDA kernels cannot take Python bool).

    Returns:
        None. Mutates `population` in-place.

    Notes:
        Thread/block indexing: threadIdx.x/blockIdx.x -> target_ptr (index within target_idx),
        threadIdx.y/blockIdx.y -> individual_idx, blockIdx.z -> batch_idx.
        S-tier always resamples to a positive uniform value in [s_low, s_high] regardless of
        sign-flip (S-tier direction is considered reliable).
        M/L-tier: when allow_sign_flip_int == 1, the [0,1) draw `r` is split into 4 quadrants of
        25% each, giving +seg1, +seg2, -seg1, -seg2 (mirrors the bimodal segments used in
        Weight_Generate_Population_Cupy but with an explicit sign choice); when disabled, only 2
        segments (50/50) are used and the value stays positive.
        Values are clamped to the tier's valid range after sampling (symmetric range under
        sign-flip, one-sided otherwise).
    """
    target_ptr     = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    individual_idx = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    batch_idx      = cuda.blockIdx.z

    n_batches = population.shape[0]
    n_pop     = population.shape[1]
    n_targets = target_idx.shape[0]

    if batch_idx < n_batches and individual_idx < n_pop and target_ptr < n_targets:
        if rand_triggers[batch_idx, individual_idx, target_ptr] < mutation_rate:
            real_snp_idx = target_idx[target_ptr]
            af = a_vec[real_snp_idx]

            r = rand_vals[batch_idx, individual_idx, target_ptr]

            if af >= s_thresh:
                # S-tier: uniform [s_low, s_high]
                val = s_low + r * (s_high - s_low)
                val = max(s_low, min(s_high, val))

            elif af >= m_thresh:
                # M-tier bimodal: [m_low, s_low] + [s_high, m_high]
                # sign_flip: 4 quadrants (+/-seg1, +/-seg2), each 25%
                # no sign_flip: 2 segments, each 50%
                if allow_sign_flip_int == 1:
                    if r < 0.25:
                        val =  (m_low  + (r * 4.0) * (s_low  - m_low))
                    elif r < 0.5:
                        val =  (s_high + ((r - 0.25) * 4.0) * (m_high - s_high))
                    elif r < 0.75:
                        val = -(m_low  + ((r - 0.5)  * 4.0) * (s_low  - m_low))
                    else:
                        val = -(s_high + ((r - 0.75) * 4.0) * (m_high - s_high))
                    val = max(-m_high, min(m_high, val))
                else:
                    if r < 0.5:
                        val = m_low  + (r * 2.0) * (s_low  - m_low)
                    else:
                        val = s_high + ((r - 0.5) * 2.0) * (m_high - s_high)
                    val = max(m_low, min(m_high, val))

            else:
                # L-tier bimodal: [l_low, m_low] + [m_high, l_high]
                if allow_sign_flip_int == 1:
                    if r < 0.25:
                        val =  (l_low  + (r * 4.0) * (m_low  - l_low))
                    elif r < 0.5:
                        val =  (m_high + ((r - 0.25) * 4.0) * (l_high - m_high))
                    elif r < 0.75:
                        val = -(l_low  + ((r - 0.5)  * 4.0) * (m_low  - l_low))
                    else:
                        val = -(m_high + ((r - 0.75) * 4.0) * (l_high - m_high))
                    val = max(-l_high, min(l_high, val))
                else:
                    if r < 0.5:
                        val = l_low  + (r * 2.0) * (m_low  - l_low)
                    else:
                        val = m_high + ((r - 0.5) * 2.0) * (l_high - m_high)
                    val = max(l_low, min(l_high, val))

            population[batch_idx, individual_idx, real_snp_idx] = val


def Weight_Mutation_Cupy(individuals, a_vec, target_idx, mutation_rate=0.5, **kwargs):
    """
    Prepare random buffers and launch the mutation kernel on a batch of individuals.

    Args:
        individuals (cupy.ndarray): shape (n_batches, pop_size, n_snps + 1), individuals to mutate
            in-place; last column (fitness) is reset to 0 after mutation.
        a_vec (cupy.ndarray): shape (n_snps,), allele-similarity factor for tier assignment.
        target_idx (cupy.ndarray): shape (n_targets,), SNP indices eligible for mutation this phase.
        mutation_rate (float): per-cell mutation probability.
        **kwargs: GA hyperparameters (tier thresholds/bounds, allow_sign_flip); see mutation_kernel.

    Returns:
        cupy.ndarray: the same `individuals` array (mutated in-place), with fitness column zeroed
        so it will be recomputed by Weight_Fit_Value_Calculate_Cupy.

    Notes:
        No-op (returns input unchanged) when there are no target SNPs or an empty population.
        Grid/block layout mirrors crossover_kernel: 32 threads along the target-SNP axis, 4 along
        the individual axis, one z-block per batch.
    """
    if target_idx.size == 0 or individuals.shape[1] == 0:
        return individuals

    s_thresh = kwargs.get('s_tier_threshold', 0.95)
    s_low    = kwargs.get('s_tier_low', 0.8)
    s_high   = kwargs.get('s_tier_high', 1.2)
    m_thresh = kwargs.get('m_tier_threshold', 0.75)
    m_low    = kwargs.get('m_tier_low', 0.3)
    m_high   = kwargs.get('m_tier_high', 3.0)
    l_low    = kwargs.get('l_tier_low', 0.0)
    l_high   = kwargs.get('l_tier_high', 7.0)

    # Sign-flip flag passed as int (CUDA kernels do not support Python bool).
    allow_sign_flip_int = 1 if kwargs.get('allow_sign_flip', False) else 0

    n_batches, pop_size, _ = individuals.shape
    num_targets = target_idx.size

    rand_triggers = cp.random.random((n_batches, pop_size, num_targets), dtype=cp.float32)
    rand_vals     = cp.random.random((n_batches, pop_size, num_targets), dtype=cp.float32)

    threads_per_block = (32, 4, 1)
    blocks_x  = (num_targets + 31) // 32
    blocks_y  = (pop_size    + 3)  // 4
    blocks_z  = n_batches
    grid_dims = (blocks_x, blocks_y, blocks_z)

    mutation_kernel[grid_dims, threads_per_block](
        individuals, target_idx, a_vec, rand_triggers, rand_vals,
        mutation_rate, s_thresh, s_low, s_high,
        m_thresh, m_low, m_high, l_low, l_high,
        allow_sign_flip_int
    )

    individuals[:, :, -1] = 0.0
    return individuals


# ──────────────────────────────────────────────
# Main GA flow (phase inheritance)
# ──────────────────────────────────────────────
def _is_phase_bad(corr_val, n_samples, bad_thresh=0.05):
    """
    Decide whether a GA phase's result is bad enough that the next phase should widen its search.

    Args:
        corr_val (float): training correlation achieved by the current phase's best solution.
        n_samples (int): number of training samples used to compute corr_val (for CI width).
        bad_thresh (float): correlation threshold below which a phase is considered unreliable.

    Returns:
        bool: True if the phase should be treated as "bad" (triggering the next phase to expand
        its SNP search scope instead of just fine-tuning), False otherwise.

    Notes:
        A negative correlation is always bad. Otherwise, when there are enough samples
        (n_samples > 3), a Fisher-z 95% confidence interval is built around corr_val
        (se = 1/sqrt(n-3)); if the CI's upper bound still falls below bad_thresh, the phase is
        judged bad even though the point estimate might be marginally above threshold — this
        guards against declaring a noisy small-sample correlation "good" prematurely. With too
        few samples, it falls back to a simple point-estimate comparison.
    """
    if corr_val < 0.0:
        return True  # Clearly negative correlation, immediately bad.

    if n_samples > 3:
        se = 1.0 / np.sqrt(n_samples - 3)
        z  = np.arctanh(np.clip(corr_val, -0.9999, 0.9999))
        ci_upper = np.tanh(z + 1.96 * se)  # 95% CI upper bound
        return ci_upper < bad_thresh
    else:
        return corr_val < bad_thresh


def Weight_GA_Cupy(backend, rsid_all, num_batches, **kwargs):
    """
    Run the full 3-phase genetic algorithm to optimize per-SNP weight-scaling factors.

    Args:
        backend (tuple): (G, w, y, a_vec) — see Weight_Fit_Value_Calculate_Cupy for shapes/meaning.
        rsid_all (list): length n_snps, SNP identifiers (order matches G/w/a_vec columns/rows).
        num_batches (int): number of independent GA batches to run in parallel (e.g. fold x run combinations).
        **kwargs: GA hyperparameters, notably:
            s_tier_threshold/m_tier_threshold (float): af_vec cutoffs separating S/M/L tiers.
            iter_num_s1/s2/s3 (int): max iterations for Phase 1/2/3.
            population_size (int): individuals per batch.
            initial_mutation_rate (float): starting mutation rate (annealed during each phase).
            mask (cupy.ndarray, optional, popped from kwargs): shape (n_batches, 1, n_samples), train/held-out mask.
            y_target (cupy.ndarray, optional, popped from kwargs): shape (n_batches, 1, n_samples), per-batch target.
            allow_sign_flip (bool): whether M/L-tier scaling may be negative.
            sign_flip_threshold (float): reused here as the "phase bad" correlation threshold (bad_thresh).
            gene_id (str, optional): used only for log messages.

    Returns:
        tuple:
            final_best_sols (cupy.ndarray): shape (n_batches, n_snps + 1), best individual per
                batch after Phase 3 (including its fitness value in the last column).
            total_history (cupy.ndarray): shape (n_batches, total_iters), concatenated best-fitness
                trace across Phase 1 + Phase 2 + Phase 3 (each phase's history is truncated to
                its actual/early-stopped iteration count before concatenation).
            phase_lens (tuple): (len(history_s1_cols), len(history_s2_cols), len(history_s3_cols)),
                number of iterations actually run in each phase — used to draw phase-boundary
                markers on the loss-curve plot.

    Notes:
        SNPs are partitioned once by af_vec into idx_phase1 (L-tier, af < m_thresh),
        idx_phase2 (M-tier), idx_phase3 (S-tier, af >= s_thresh). The GA optimizes tiers from
        least to most transferable (L -> M -> S), on the theory that large-AF-difference SNPs
        need the most correction and should be resolved first, while high-similarity S-tier SNPs
        need only minor S-tier fine-tuning at the end.
        Phase 1: search idx_phase1 (L-tier) from a fresh random population.
        Phase 2: warm-start from Phase 1's best solution; if Phase 1's training correlation is
            judged bad by _is_phase_bad, Phase 2 widens its target set to L+M tiers (self-healing
            search) instead of locking in the (unreliable) Phase 1 L-tier solution.
        Phase 3: warm-start from Phase 2's best solution; if Phase 2 is still bad, Phase 3 expands
            to re-search all tiers (L+M+S) as a last resort; otherwise it only fine-tunes S-tier
            while locking the converged L+M-tier values.
        run_stage_batch (nested) implements one phase's generation loop: selection -> crossover ->
        mutation -> fitness, then rebuilds the next generation from elites (top 10%), best children
        (a ratio growing from 0.4 to 0.7 across the phase's iterations, i.e. current_child_ratio),
        and random newcomers (to preserve diversity) with non-target SNPs held fixed at the current
        best solution's values. `fixed_snps` (if given) keeps specific SNP columns pinned to their
        Phase-start best value across the whole phase, on top of the diversity-injection fixing.
        Early stopping combines two criteria:
          Primary: stagnation of the per-iteration mean best-fitness (mirrors the plotted loss
            curve) for `patience` = max(50, 10% of current_iters) iterations without improving by
            at least min_delta.
          Secondary (only active when `mask` is given, i.e. under CV): stagnation of a held-out
            metric — Fisher-z loss on the held-out samples when n_held > 1 (K-Fold), or MAE when
            n_held == 1 (LOOCV, where correlation is undefined for a single point). This secondary
            check only actually gates stopping in the K-Fold case (n_held > 1 on average); LOOCV's
            noisy single-sample MAE is tracked but not used to force an early stop.
        `_compute_train_corr` (nested) recomputes the current phase-best solution's mean training
        correlation across batches, used as the input to _is_phase_bad for the phase 1->2 and
        2->3 escalation decisions described above.
    """
    G, w, y, a_vec = backend
    num_snps = len(rsid_all)

    s_thresh  = kwargs.get('s_tier_threshold', 0.95)
    m_thresh  = kwargs.get('m_tier_threshold', 0.75)
    iter_s1   = kwargs.get('iter_num_s1', 750)
    iter_s2   = kwargs.get('iter_num_s2', 500)
    iter_s3   = kwargs.get('iter_num_s3', 250)
    population_size = kwargs.get('population_size', 500)
    init_mutate     = kwargs.get('initial_mutation_rate', 0.5)
    mask      = kwargs.pop('mask', None)
    y_target  = kwargs.pop('y_target', None)
    allow_sign_flip = kwargs.get('allow_sign_flip', False)

     # Phase 1: high AF-difference SNPs (L-tier, af < m_thresh)
    idx_phase1 = cp.where(a_vec < m_thresh)[0]
    # Phase 2: medium AF-difference SNPs (M-tier, m_thresh <= af < s_thresh)
    idx_phase2 = cp.where((a_vec >= m_thresh) & (a_vec < s_thresh))[0]
    # Phase 3: low AF-difference SNPs (S-tier, af >= s_thresh)
    idx_phase3 = cp.where(a_vec >= s_thresh)[0]

    loss_history = []

    def _compute_train_corr(pop_final):
        """
        Compute the mean training correlation of the current phase's best solution across batches.

        Args:
            pop_final (cupy.ndarray): shape (n_batches, pop_size, n_snps + 1), a phase's final
                population (only the top individual per batch, index 0, is used).

        Returns:
            float: mean (over batches) Pearson correlation between prediction and target,
            computed on the training (masked) portion of the samples if `mask` is set, else on
            all samples.

        Notes:
            Mirrors the correlation computation inside Weight_Fit_Value_Calculate_Cupy but only
            for the single best individual per batch, used to feed _is_phase_bad.
        """
        best_P = pop_final[:, 0:1, :num_snps]
        w_eff  = w * a_vec
        pred   = cp.matmul(best_P * w_eff, G)
        y_eval = y_target if y_target is not None else y

        if mask is not None:
            n_tr = mask.sum(axis=2, keepdims=True)
            pm   = (pred   * mask).sum(axis=2, keepdims=True) / n_tr
            pc   = (pred   - pm)  * mask
            ym   = (y_eval * mask).sum(axis=2, keepdims=True) / n_tr
            yc   = (y_eval - ym)  * mask
            cov  = (pc * yc).sum(axis=2)
            sp   = (cp.sqrt((pc ** 2).sum(axis=2, keepdims=True) / n_tr + 1e-12) *
                    cp.sqrt((yc ** 2).sum(axis=2, keepdims=True) / n_tr + 1e-12) *
                    n_tr).squeeze(axis=2)
        else:
            pm   = pred.mean(axis=2, keepdims=True)
            pc   = pred - pm
            ym   = y_eval.mean(axis=2, keepdims=True)
            yc   = y_eval - ym
            cov  = cp.sum(pc * yc, axis=2)
            sp   = (pred.shape[2] *
                    (cp.sqrt((pc ** 2).mean(axis=2, keepdims=True)) + 1e-12) *
                    (cp.sqrt((yc ** 2).mean(axis=2, keepdims=True)) + 1e-12)).squeeze(axis=2)

        return float((cov / (sp + 1e-12)).mean())

    def run_stage_batch(initial_pop, target_snps, current_iters, phase_idx, fixed_snps=None):
        """
        Run one GA phase's generation loop (selection -> crossover -> mutation -> fitness).

        Args:
            initial_pop (cupy.ndarray): shape (n_batches, population_size, n_snps + 1), starting
                population for this phase (already fitness-evaluated).
            target_snps (cupy.ndarray): SNP indices this phase is allowed to mutate/optimize.
            current_iters (int): maximum number of generations to run before forced stop.
            phase_idx (int): 1, 2, or 3 — informational only (used for readability; not branched on).
            fixed_snps (cupy.ndarray, optional): SNP indices whose values must stay pinned to the
                phase-start best solution throughout this phase (typically the converged tier(s)
                from a previous phase).

        Returns:
            tuple:
                pop (cupy.ndarray): shape (n_batches, population_size, n_snps + 1), final
                    fitness-sorted population (index 0 per batch is the best individual).
                stage_history (cupy.ndarray): shape (n_batches, actual_iters), best fitness per
                    batch at each executed generation (trimmed to the actual number of iterations
                    run before early stopping, if any).

        Notes:
            Per-generation schedule:
              current_child_ratio ramps 0.4 -> 0.7 over the phase (progress = it / current_iters),
                controlling how much of the next generation comes from crossover children vs.
                random newcomers.
              m_rate (mutation rate) anneals linearly from init_mutate down to 0.2 over the phase.
            Next-generation composition: n_elites (top 10% of current pop) + n_best_child (top
              children after crossover+mutation+fitness) + n_random (freshly sampled individuals,
              used to inject diversity and fight premature convergence); non-target SNP columns of
              the random newcomers are overwritten with the current best solution's values so they
              don't regress the phase's already-converged/inherited SNPs.
            fixed_snps handling: children's fixed-SNP columns are pinned to fixed_vals (captured
              once at phase start) immediately after crossover, before mutation and fitness scoring.
            See Weight_GA_Cupy's docstring for the two-tier early-stopping criteria (primary =
              mean-fitness stagnation; secondary = held-out metric stagnation, active for K-Fold).
            GPU memory pool is proactively trimmed every 200 iterations to limit fragmentation
              during long-running phases.
        """
        pop = initial_pop

        ranks          = cp.argsort(pop[:, :, -1], axis=1)
        ranks_expanded = cp.broadcast_to(ranks[:, :, None], pop.shape)
        pop            = cp.take_along_axis(pop, ranks_expanded, axis=1)

        stage_history = cp.zeros((num_batches, current_iters), dtype=cp.float32)
        n_elites      = max(1, int(population_size * 0.1))

        # Early-stop state: stop when most batches fail to improve beyond min_delta.
        patience      = max(50, int(current_iters * 0.1))
        min_delta     = 1e-3
        best_mean_fitness = float('inf')   # scalar mean across batches (mirrors plotted loss curve)
        mean_stale_count  = 0
        best_held_out = cp.full(num_batches, cp.inf, dtype=cp.float32)
        stale_held    = cp.zeros(num_batches, dtype=cp.int32)
        actual_iters  = current_iters

        # Cache the current best values for SNPs that should remain fixed during this phase.
        if fixed_snps is not None and fixed_snps.size > 0:
            fixed_vals = pop[:, 0:1, fixed_snps]
        else:
            fixed_vals = None

        # Generation loop with phase-aware mutation scheduling.
        for it in range(current_iters):
            progress           = it / current_iters
            current_child_ratio = 0.4 + 0.3 * progress
            m_rate = init_mutate + (0.2 - init_mutate) * progress

            # Core GA step: selection -> crossover -> mutation -> fitness evaluation.
            parents  = Weight_Selection_Cupy(pop)
            children = Weight_CrossOver_Cupy(parents)

            if fixed_snps is not None and fixed_snps.size > 0:
                children[:, :, fixed_snps] = fixed_vals

            children = Weight_Mutation_Cupy(children, a_vec, target_snps, m_rate, **kwargs)
            children = Weight_Fit_Value_Calculate_Cupy(
                backend, children, num_snps, mask=mask, y_target=y_target, **kwargs
            )

            child_ranks    = cp.argsort(children[:, :, -1], axis=1)
            child_ranks_exp = cp.broadcast_to(child_ranks[:, :, None], children.shape)
            children       = cp.take_along_axis(children, child_ranks_exp, axis=1)

            # Build the next generation from elites, best children, and optional random newcomers.
            n_best_child = int(population_size * current_child_ratio)
            n_random     = population_size - n_elites - n_best_child

            elites    = pop[:, :n_elites, :]
            best_kids = children[:, :n_best_child, :]
            parts     = [elites, best_kids]

            # Inject random individuals to preserve diversity while keeping non-target SNPs fixed.
            if n_random > 0:
                new_blood = Weight_Generate_Population_Cupy(
                    rsid_all, a_vec, num_batches, target_size=n_random, **kwargs
                )
                all_indices  = cp.arange(num_snps)
                target_mask  = cp.isin(all_indices, target_snps)
                fixed_indices = all_indices[~target_mask]

                # Keep non-target SNPs inherited from the current best solution.
                if fixed_indices.size > 0:
                    best_genes              = pop[:, 0:1, fixed_indices]
                    new_blood[:, :, fixed_indices] = best_genes

                new_blood = Weight_Fit_Value_Calculate_Cupy(
                    backend, new_blood, num_snps, mask=mask, y_target=y_target, **kwargs
                )
                parts.append(new_blood)

            pop        = cp.concatenate(parts, axis=1)
            ranks      = cp.argsort(pop[:, :, -1], axis=1)
            ranks_exp  = cp.broadcast_to(ranks[:, :, None], pop.shape)
            pop        = cp.take_along_axis(pop, ranks_exp, axis=1)

            # Track the current best fitness for convergence and plotting.
            stage_history[:, it] = pop[:, 0, -1]

            # Primary early-stop: stagnation of mean best fitness (mirrors the plotted loss curve).
            current_mean = float(cp.mean(pop[:, 0, -1]))
            if current_mean < best_mean_fitness - min_delta:
                best_mean_fitness = current_mean
                mean_stale_count  = 0
            else:
                mean_stale_count += 1

            stop_condition = mean_stale_count >= patience

            # Secondary early-stop criterion (CV mode).
            # kfold (n_held > 1): use held-out Fisher-z loss — same metric as train fitness.
            # LOOCV (n_held == 1): correlation undefined for 1 sample, fall back to MAE.
            if mask is not None:
                best_P        = pop[:, 0:1, :num_snps]
                w_eff         = w * a_vec
                pred_best     = cp.matmul(best_P * w_eff, G)
                held_out_mask = 1.0 - mask
                n_held        = held_out_mask.sum(axis=2).squeeze(axis=1)   # (batch,)
                n_held_kk     = n_held[:, None, None]                        # (batch,1,1)

                # MAE fallback (always valid, used when n_held == 1)
                held_mae = (cp.abs(pred_best - y_target) * held_out_mask) \
                               .sum(axis=2).squeeze(axis=1) / (n_held + 1e-12)

                # Held-out Fisher-z (consistent with train fitness, valid when n_held > 1)
                p_mean_h = (pred_best * held_out_mask).sum(axis=2, keepdims=True) / (n_held_kk + 1e-12)
                y_mean_h = (y_target  * held_out_mask).sum(axis=2, keepdims=True) / (n_held_kk + 1e-12)
                p_c_h    = (pred_best - p_mean_h) * held_out_mask
                y_c_h    = (y_target  - y_mean_h) * held_out_mask
                cov_h    = (p_c_h * y_c_h).sum(axis=2).squeeze(axis=1)
                p_std_h  = cp.sqrt((p_c_h**2).sum(axis=2).squeeze(axis=1) / (n_held + 1e-12) + 1e-12)
                y_std_h  = cp.sqrt((y_c_h**2).sum(axis=2).squeeze(axis=1) / (n_held + 1e-12) + 1e-12)
                corr_h   = cov_h / (p_std_h * y_std_h * n_held + 1e-12)
                held_fisher = -cp.arctanh(cp.clip(corr_h, -0.9999, 0.9999))

                # Pick metric per batch: Fisher-z for kfold, MAE for LOOCV
                held_loss     = cp.where(n_held > 1, held_fisher, held_mae)
                improved_held = (best_held_out - held_loss) > min_delta
                best_held_out = cp.where(improved_held, held_loss, best_held_out)
                stale_held    = cp.where(improved_held, 0, stale_held + 1)
                # Secondary stop is only used for kfold (n_held > 1);
                # LOOCV's n_held=1 MAE is too noisy to gate stopping.
                if float(n_held.mean()) > 1:
                    stop_condition = (
                        mean_stale_count >= patience or
                        float(cp.mean(stale_held  >= int(patience * 0.5))) >= 0.8
                    )

            if stop_condition:
                actual_iters = it + 1
                break

            # Periodically release cached GPU memory blocks to control fragmentation.
            if it % 200 == 0 and it > 0:
                cp.get_default_memory_pool().free_all_blocks()

        # Trim history to the actual number of iterations executed (after early stop).
        stage_history = stage_history[:, :actual_iters]
        return pop, stage_history

    bad_thresh = kwargs.get('sign_flip_threshold', 0.05)
    n_samples  = int(backend[0].shape[1])

    # -- Phase 1: optimize L-tier --
    if idx_phase1.size > 0:
        init_pop_s1 = Weight_Generate_Population_Cupy(rsid_all, a_vec, num_batches, population_size, **kwargs)
        init_pop_s1 = Weight_Fit_Value_Calculate_Cupy(
            backend, init_pop_s1, num_snps, mask=mask, y_target=y_target, **kwargs
        )
        pop_s1_final, history_s1 = run_stage_batch(
            init_pop_s1, idx_phase1, iter_s1, phase_idx=1, fixed_snps=None
        )
        loss_history.append(history_s1)
        phase1_bad = _is_phase_bad(
            _compute_train_corr(pop_s1_final),
            n_samples=n_samples,
            bad_thresh=bad_thresh
        )
        if phase1_bad:
            _gid = kwargs.get('gene_id', '?')
            logger.info(f"  [Phase] {_gid}: Phase1 poor -> Phase2 expanded to L+M tier")
    else:
        # L-tier is empty: skip Phase 1 and initialize randomly
        pop_s1_final = Weight_Generate_Population_Cupy(rsid_all, a_vec, num_batches, population_size, **kwargs)
        pop_s1_final = Weight_Fit_Value_Calculate_Cupy(
            backend, pop_s1_final, num_snps, mask=mask, y_target=y_target, **kwargs
        )
        history_s1   = cp.zeros((num_batches, 1), dtype=cp.float32)
        loss_history.append(history_s1)
        phase1_bad = False

    # -- Phase 2: optimize M-tier (or L+M-tier if Phase 1 is negatively correlated) --
    best_sol_s1 = pop_s1_final[:, 0, :]
    init_pop_s2 = Weight_Generate_Population_Cupy(rsid_all, a_vec, num_batches, population_size, **kwargs)
    if idx_phase1.size > 0:
        init_pop_s2[:, :, idx_phase1] = best_sol_s1[:, None, idx_phase1]  # Warm-start from Phase 1
    init_pop_s2 = Weight_Fit_Value_Calculate_Cupy(
        backend, init_pop_s2, num_snps, mask=mask, y_target=y_target, **kwargs
    )
    if phase1_bad:
        # Phase 1 direction is wrong: mutate both L+M tiers in Phase 2; fixed locking from Phase 1 loses meaning
        idx_p2_target = cp.concatenate([idx_phase1, idx_phase2])
        fixed_p2      = None
        iter_p2       = iter_s1 + iter_s2
    else:
        idx_p2_target = idx_phase2
        fixed_p2      = idx_phase1 if idx_phase1.size > 0 else None  # Normal mode: lock converged L-tier SNPs
        iter_p2       = iter_s2
    if idx_p2_target.size > 0:
        pop_s2_final, history_s2 = run_stage_batch(
            init_pop_s2, idx_p2_target, iter_p2, phase_idx=2, fixed_snps=fixed_p2
        )
        loss_history.append(history_s2)
        phase2_bad = _is_phase_bad(
            _compute_train_corr(pop_s2_final),
            n_samples=n_samples,
            bad_thresh=bad_thresh
        )
        if phase2_bad:
            _gid = kwargs.get('gene_id', '?')
            logger.info(f"  [Phase] {_gid}: Phase2 still poor -> Phase3 re-searches all tiers")
    else:
        # M-tier is empty: skip Phase 2
        pop_s2_final = init_pop_s2
        history_s2   = cp.zeros((num_batches, 1), dtype=cp.float32)
        loss_history.append(history_s2)
        phase2_bad = False

    # -- Phase 3: S-tier fine-tuning; expand to all tiers if Phase 2 is still negative --
    best_sol_s2 = pop_s2_final[:, 0, :]
    init_pop_s3 = Weight_Generate_Population_Cupy(rsid_all, a_vec, num_batches, population_size, **kwargs)
    if idx_phase1.size > 0:
        init_pop_s3[:, :, idx_phase1] = best_sol_s2[:, None, idx_phase1]
    if idx_phase2.size > 0:
        init_pop_s3[:, :, idx_phase2] = best_sol_s2[:, None, idx_phase2]
    init_pop_s3 = Weight_Fit_Value_Calculate_Cupy(
        backend, init_pop_s3, num_snps, mask=mask, y_target=y_target, **kwargs
    )
    if phase2_bad:
        # Phase 2 fix still failed: re-search all SNPs in Phase 3 (last resort)
        idx_p3_target = cp.concatenate([idx_phase1, idx_phase2, idx_phase3])
        fixed_p3      = None
        iter_p3       = iter_s1 + iter_s2 + iter_s3
    else:
        # Normal mode: fine-tune S-tier and lock converged L+M tiers
        idx_p3_target = idx_phase3
        fixed_p3      = cp.concatenate([idx_phase1, idx_phase2]) if (idx_phase1.size + idx_phase2.size) > 0 else None
        iter_p3       = iter_s3
    if idx_p3_target.size > 0:
        pop_s3_final, history_s3 = run_stage_batch(
            init_pop_s3, idx_p3_target, iter_p3, phase_idx=3, fixed_snps=fixed_p3
        )
        loss_history.append(history_s3)
    else:
        # S-tier is empty: skip Phase 3
        pop_s3_final = init_pop_s3
        history_s3   = cp.zeros((num_batches, 1), dtype=cp.float32)
        loss_history.append(history_s3)

    final_best_sols = pop_s3_final[:, 0, :]
    total_history   = cp.concatenate(loss_history, axis=1)
    phase_lens = (int(history_s1.shape[1]), int(history_s2.shape[1]), int(history_s3.shape[1]))
    return final_best_sols, total_history, phase_lens


# ──────────────────────────────────────────────
# LOOCV (passes **kwargs through to GA)
# ──────────────────────────────────────────────

def run_loocv_batch_ga_cupy(G_all_gpu, y_all_gpu, w_gpu, a_vec_gpu, rsid_all, **ga_params):
    """
    Run leave-one-out cross-validation of the GA weight optimizer for one gene.

    Args:
        G_all_gpu (cupy.ndarray): shape (n_snps, n_samples), genotype dosage matrix.
        y_all_gpu (cupy.ndarray): shape (n_samples,), observed (real) expression values.
        w_gpu (cupy.ndarray): shape (n_snps,), original PrediXcan weights.
        a_vec_gpu (cupy.ndarray): shape (n_snps,), allele-similarity factor.
        rsid_all (list): length n_snps, SNP identifiers.
        **ga_params: GA hyperparameters, notably num_runs (independent GA restarts per fold) and
            batch_limit (max parallel GA batches sent to the GPU at once).

    Returns:
        tuple:
            performance_df (pd.DataFrame): one row per held-out sample with columns
                ['sample', 'actual', 'predict', 'train_mu', 'train_std'] (actual/predict are
                z-scored using that fold's train-only mean/std).
            fold_weights_df (pd.DataFrame): shape (n_samples, n_snps), run-averaged GA scaling
                factors per fold (row = held-out sample), columns = rsid_all.
            best_preds_norm (np.ndarray): shape (n_samples,), normalized held-out predictions
                (same values as performance_df['predict'], returned separately for convenience).
            loss_df (pd.DataFrame): single column 'Average_Loss', the run/fold-averaged GA loss
                curve; loss_df.attrs['phase_boundaries'] holds the mean (phase1_end, phase2_end)
                iteration indices for plotting phase-transition markers.
            overall_weight_stability (float): mean per-SNP std of scaling factors across the
                num_runs repeats, averaged across folds and SNPs — lower means more reproducible GA
                solutions.
            train_corr (float): in-sample (overfit-diagnostic) correlation using the run-and-fold
                averaged weights applied to all samples.
            fold_diagnostics_df (pd.DataFrame): per-fold training correlation and sample counts.

    Notes:
        For each held-out sample i, a train-only y-normalization (mean/std excluding sample i) is
        precomputed (y_lookup) so the GA fitness function only ever sees train-consistent targets,
        preventing test-set leakage into normalization.
        Total GA tasks = n_samples * num_runs (num_runs independent restarts per fold to average
        out GA stochasticity); tasks are grouped into GPU batches of at most batch_limit and run
        sequentially through Weight_GA_Cupy, with GPU memory freed between chunks.
        Per-fold GA scaling solutions are reshaped to (n_samples, num_runs, n_snps) and averaged
        over the run axis to get one solution per fold; the std over runs gives a stability metric.
        Held-out predictions are normalized using that fold's train-only prediction mean/std
        (computed from the training genotypes only), keeping the CV evaluation leakage-free.
    """
    n_snps, n_samples = G_all_gpu.shape
    num_runs    = ga_params.get('num_runs', 5)
    batch_limit = ga_params.get('batch_limit', 2048)
    if batch_limit <= 0:
        batch_limit = 2048

    # Create train-only normalization targets for y per held-out fold
    y_lookup   = cp.zeros((n_samples, n_samples), dtype=cp.float32)
    train_mus  = cp.zeros(n_samples, dtype=cp.float32)
    train_stds = cp.zeros(n_samples, dtype=cp.float32)
    # Fold-wise prediction and normalization using train-only genotype statistics.
    for i in range(n_samples):
        train_mask = cp.ones(n_samples, dtype=bool)
        train_mask[i] = False
        y_train    = y_all_gpu[train_mask]
        t_mu       = y_train.mean()
        t_std      = y_train.std() + 1e-6
        train_mus[i]   = t_mu
        train_stds[i]  = t_std
        y_lookup[i, :] = (y_all_gpu - t_mu) / t_std

    # --- Build all fold x run masks/y_target once, then send to GPU in batch_limit chunks ---
    total_tasks  = n_samples * num_runs
    mask_all     = cp.ones((total_tasks, 1, n_samples), dtype=cp.float32)
    y_target_all = cp.zeros((total_tasks, 1, n_samples), dtype=cp.float32)
    for i in range(n_samples):
        s, e = i * num_runs, (i + 1) * num_runs
        # hold-out sample i
        mask_all[s:e, 0, i]     = 0.0
        # y for fold i (broadcast)
        y_target_all[s:e, 0, :] = y_lookup[i:i + 1, :]

    all_final_solutions = []
    all_history_batch   = []
    all_phase_lens      = []
    offset = 0
    while offset < total_tasks:
        bs         = min(batch_limit, total_tasks - offset)
        mask_chunk = mask_all[offset:offset + bs]
        y_chunk    = y_target_all[offset:offset + bs]

        backend = (G_all_gpu, w_gpu, None, a_vec_gpu)
        solutions, history, phase_lens = Weight_GA_Cupy(
            backend, rsid_all,
            num_batches=bs,
            mask=mask_chunk,
            y_target=y_chunk,
            **ga_params
        )

        all_final_solutions.append(solutions[:, :-1])
        all_history_batch.append(history.get())
        all_phase_lens.append(phase_lens)
        offset += bs

        del mask_chunk, y_chunk, solutions, history
        cp.get_default_memory_pool().free_all_blocks()

    del mask_all, y_target_all
    cp.get_default_memory_pool().free_all_blocks()

    # --- Results Processing ---
    weights_flat       = cp.concatenate(all_final_solutions, axis=0)
    weights_structured = weights_flat.reshape(n_samples, num_runs, n_snps)
    avg_weights_per_fold = weights_structured.mean(axis=1)
    std_weights_per_fold  = weights_structured.std(axis=1)
    avg_std_across_folds  = std_weights_per_fold.mean(axis=0)
    overall_weight_stability = avg_std_across_folds.mean()

    # Average loss history across all runs.
    # Histories can have different lengths due to early stopping in each batch.
    if all_history_batch:
        max_iters   = max(h.shape[1] for h in all_history_batch)
        total_rows  = sum(h.shape[0] for h in all_history_batch)
        history_padded = np.full((total_rows, max_iters), np.nan, dtype=np.float32)
        cursor = 0
        for h in all_history_batch:
            rows, cols = h.shape
            history_padded[cursor:cursor + rows, :cols] = h
            cursor += rows
        avg_history = np.nanmean(history_padded, axis=0).tolist()
    else:
        avg_history = []

    # --- LOOCV (Validation) ---
    all_actuals_gpu  = cp.zeros(n_samples, dtype=cp.float32)
    all_preds_gpu    = cp.zeros(n_samples, dtype=cp.float32)
    fold_train_corrs = []
    wa_gpu           = w_gpu * a_vec_gpu

    for i in range(n_samples):
        train_mask = cp.ones(n_samples, dtype=bool)
        train_mask[i] = False

        G_fold_gpu    = G_all_gpu
        w_eff         = avg_weights_per_fold[i] * wa_gpu
        pred_train    = cp.dot(w_eff, G_fold_gpu[:, train_mask])
        pred_held_out = cp.dot(w_eff, G_fold_gpu[:, i])
        tp_mu         = pred_train.mean()
        tp_std        = pred_train.std() + 1e-6
        # Per-fold train corr (in-sample for this fold's weights)
        y_tr    = y_all_gpu[train_mask]
        y_tr_z  = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-6)
        p_tr_z  = (pred_train - pred_train.mean()) / (pred_train.std() + 1e-6)
        fold_tc = float(cp.corrcoef(p_tr_z, y_tr_z)[0, 1])
        fold_train_corrs.append(fold_tc if not np.isnan(fold_tc) else 0.0)
        all_preds_gpu[i]   = (pred_held_out - tp_mu) / tp_std
        all_actuals_gpu[i] = (y_all_gpu[i] - train_mus[i]) / train_stds[i]
        del G_fold_gpu

    # --- Overfit Diagnostic (In-sample) ---
    G_full    = G_all_gpu
    avg_w_eff = avg_weights_per_fold.mean(axis=0) * wa_gpu
    pred_full = cp.dot(avg_w_eff, G_full)
    y_z       = (y_all_gpu - y_all_gpu.mean()) / (y_all_gpu.std() + 1e-6)
    pred_z    = (pred_full - pred_full.mean()) / (pred_full.std() + 1e-6)
    if cp.std(pred_z) < 1e-9 or cp.std(y_z) < 1e-9:
        train_corr = 0.0
    else:
        train_corr = float(cp.corrcoef(pred_z, y_z)[0, 1])
        if np.isnan(train_corr):
            train_corr = 0.0

    # --- Final DataFrames Preparation ---
    all_weights   = avg_weights_per_fold.get()
    all_actuals   = all_actuals_gpu.get()
    all_preds     = all_preds_gpu.get()
    all_train_mu  = train_mus.get()
    all_train_std = train_stds.get()

    # 1. Performance DataFrame
    all_results = []
    for i in range(n_samples):
        all_results.append({
            'sample':    f"Sample_{i+1}",
            'actual':    float(all_actuals[i]),
            'predict':   float(all_preds[i]),
            'train_mu':  float(all_train_mu[i]),
            'train_std': float(all_train_std[i])
        })
    performance_df  = pd.DataFrame(all_results)
    # 2. Weights DataFrame
    fold_weights_df = pd.DataFrame(all_weights, columns=rsid_all)
    # 3. Loss DataFrame
    fold_losses     = {"Average_Loss": avg_history}
    loss_df = pd.DataFrame(fold_losses)
    if all_phase_lens:
        p1 = int(np.mean([p[0] for p in all_phase_lens]))
        p2 = int(np.mean([p[1] for p in all_phase_lens]))
        loss_df.attrs['phase_boundaries'] = (p1, p1 + p2)
    # 4. Per-fold diagnostics (train corr per fold, for reliability check)
    fold_diagnostics_df = pd.DataFrame({
        'fold_id':    [f"Sample_{i+1}" for i in range(n_samples)],
        'train_corr': fold_train_corrs,
        'n_train':    n_samples - 1,
        'n_test':     1,
    })
    # Keep compatibility with downstream plotting/output.
    best_preds_norm = all_preds_gpu.copy()

    del weights_flat, weights_structured
    cp.get_default_memory_pool().free_all_blocks()

    return (
        performance_df,
        fold_weights_df,
        best_preds_norm.get(),
        loss_df,
        float(overall_weight_stability),
        float(train_corr),
        fold_diagnostics_df,
    )


# ──────────────────────────────────────────────
# K-Fold (passes **kwargs through to GA)
# ──────────────────────────────────────────────

def run_kfold_batch_ga_cupy(G_all_gpu, y_all_gpu, w_gpu, a_vec_gpu, rsid_all, **ga_params):
    """
    Run (repeated) K-Fold cross-validation of the GA weight optimizer for one gene.

    Args:
        G_all_gpu (cupy.ndarray): shape (n_snps, n_samples), genotype dosage matrix.
        y_all_gpu (cupy.ndarray): shape (n_samples,), observed expression values.
        w_gpu (cupy.ndarray): shape (n_snps,), original PrediXcan weights.
        a_vec_gpu (cupy.ndarray): shape (n_snps,), allele-similarity factor.
        rsid_all (list): length n_snps, SNP identifiers.
        **ga_params: GA hyperparameters plus n_splits (folds), n_repeats (>1 uses
            RepeatedKFold), cv_seed, num_runs, batch_limit.

    Returns:
        tuple: same structure as run_loocv_batch_ga_cupy —
            (performance_df, fold_weights_df, best_preds_norm, loss_df,
             overall_weight_stability, train_corr, fold_diagnostics_df), except:
              performance_df has one row per SAMPLE (predictions averaged across all folds/repeats
                that held it out); fold_weights_df has one row per FOLD (K or K*n_repeats rows,
                not n_samples rows); fold_diagnostics_df is indexed by integer fold_id and includes
                n_train/n_test counts per fold.

    Notes:
        Uses sklearn's KFold (n_repeats == 1) or RepeatedKFold (n_repeats > 1) to build the fold
        list; each fold's train-only y statistics are precomputed for leakage-free normalization,
        mirroring the LOOCV function's y_lookup approach.
        Total GA tasks = n_folds * num_runs, batched through Weight_GA_Cupy identically to LOOCV.
        Since a sample can appear in multiple held-out test sets under RepeatedKFold, predictions
        and actuals are accumulated per sample and averaged over however many times (count) that
        sample was held out, giving one final prediction per sample.
        The in-sample overfit diagnostic (train_corr) is computed identically to the LOOCV version,
        using the fold-and-run-averaged weights applied to the full sample set.
    """
    n_snps, n_samples = G_all_gpu.shape
    num_runs    = ga_params.get('num_runs', 5)
    batch_limit = ga_params.get('batch_limit', 2048)
    n_splits    = ga_params.get('n_splits', 5)
    n_repeats   = ga_params.get('n_repeats', 1)
    cv_seed     = ga_params.get('cv_seed', 42)

    # --- Build fold list ---
    # Each element: (train_indices_np, test_indices_np)
    if n_repeats > 1:
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)

    fold_list = list(splitter.split(np.arange(n_samples)))
    n_folds   = len(fold_list)

    # --- Precompute per-fold y normalization (train-only statistics) ---
    fold_train_mus  = cp.zeros(n_folds, dtype=cp.float32)
    fold_train_stds = cp.zeros(n_folds, dtype=cp.float32)
    y_lookup_fold   = cp.zeros((n_folds, n_samples), dtype=cp.float32)

    for f, (train_idx, _) in enumerate(fold_list):
        train_idx_gpu  = cp.array(train_idx)
        y_train        = y_all_gpu[train_idx_gpu]
        t_mu           = y_train.mean()
        t_std          = y_train.std() + 1e-6
        fold_train_mus[f]  = t_mu
        fold_train_stds[f] = t_std
        y_lookup_fold[f]   = (y_all_gpu - t_mu) / t_std

    # --- Build all fold x run masks/y_target once, then send to GPU in batch_limit chunks ---
    total_tasks  = n_folds * num_runs
    mask_all     = cp.ones((total_tasks, 1, n_samples), dtype=cp.float32)
    y_target_all = cp.zeros((total_tasks, 1, n_samples), dtype=cp.float32)
    for f, (_, test_idx) in enumerate(fold_list):
        s, e = f * num_runs, (f + 1) * num_runs
        mask_all[s:e, 0, test_idx]  = 0.0
        y_target_all[s:e, 0, :]     = y_lookup_fold[f:f + 1, :]

    all_final_solutions = []
    all_history_batch   = []
    all_phase_lens      = []
    offset = 0
    while offset < total_tasks:
        bs         = min(batch_limit, total_tasks - offset)
        mask_chunk = mask_all[offset:offset + bs]
        y_chunk    = y_target_all[offset:offset + bs]

        backend = (G_all_gpu, w_gpu, None, a_vec_gpu)
        solutions, history, phase_lens = Weight_GA_Cupy(
            backend, rsid_all,
            num_batches=bs,
            mask=mask_chunk,
            y_target=y_chunk,
            **ga_params
        )

        all_final_solutions.append(solutions[:, :-1])
        all_history_batch.append(history.get())
        all_phase_lens.append(phase_lens)
        offset += bs

        del mask_chunk, y_chunk, solutions, history
        cp.get_default_memory_pool().free_all_blocks()

    del mask_all, y_target_all
    cp.get_default_memory_pool().free_all_blocks()

    # --- Results Processing ---
    # Concatenate: (n_folds * num_runs, n_snps)
    weights_flat       = cp.concatenate(all_final_solutions, axis=0)
    # Reshape: (n_folds, num_runs, n_snps)
    weights_structured = weights_flat.reshape(n_folds, num_runs, n_snps)
    # Average across runs within each fold: (n_folds, n_snps)
    avg_weights_per_fold     = weights_structured.mean(axis=1)
    # Weight stability (same as LOOCV)
    std_weights_per_fold      = weights_structured.std(axis=1)
    avg_std_across_folds      = std_weights_per_fold.mean(axis=0)
    overall_weight_stability  = float(avg_std_across_folds.mean())

    # History padding (same as LOOCV)
    if all_history_batch:
        max_iters   = max(h.shape[1] for h in all_history_batch)
        total_rows  = sum(h.shape[0] for h in all_history_batch)
        history_padded = np.full((total_rows, max_iters), np.nan, dtype=np.float32)
        cursor = 0
        for h in all_history_batch:
            rows, cols = h.shape
            history_padded[cursor:cursor + rows, :cols] = h
            cursor += rows
        avg_history = np.nanmean(history_padded, axis=0).tolist()
    else:
        avg_history = []

    # --- K-Fold Validation ---
    # Map each sample to its fold(s).
    # In Repeated K-Fold a sample appears in exactly n_repeats test sets.
    # We collect all predictions and average them.
    pred_accumulator        = cp.zeros(n_samples, dtype=cp.float32)
    actual_accumulator      = cp.zeros(n_samples, dtype=cp.float32)
    count_accumulator       = cp.zeros(n_samples, dtype=cp.float32)
    fold_diagnostics_records = []
    wa_gpu                  = w_gpu * a_vec_gpu

    for f, (train_idx, test_idx) in enumerate(fold_list):
        w_eff         = avg_weights_per_fold[f] * wa_gpu
        train_idx_gpu = cp.array(train_idx)
        pred_train    = cp.dot(w_eff, G_all_gpu[:, train_idx_gpu])
        tp_mu         = pred_train.mean()
        tp_std        = pred_train.std() + 1e-6
        # Per-fold train corr
        y_tr    = y_all_gpu[train_idx_gpu]
        y_tr_z  = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-6)
        p_tr_z  = (pred_train - pred_train.mean()) / (pred_train.std() + 1e-6)
        fold_tc = float(cp.corrcoef(p_tr_z, y_tr_z)[0, 1])
        fold_diagnostics_records.append({
            'fold_id':    f,
            'train_corr': fold_tc if not np.isnan(fold_tc) else 0.0,
            'n_train':    len(train_idx),
            'n_test':     len(test_idx),
        })
        for i in test_idx:
            pred_held_out          = cp.dot(w_eff, G_all_gpu[:, i])
            pred_accumulator[i]   += (pred_held_out - tp_mu) / tp_std
            actual_accumulator[i] += (y_all_gpu[i] - fold_train_mus[f]) / fold_train_stds[f]
            count_accumulator[i]  += 1.0

    # Average over repeats (for plain K-Fold count == 1, no change)
    all_preds_gpu   = pred_accumulator   / (count_accumulator + 1e-12)
    all_actuals_gpu = actual_accumulator / (count_accumulator + 1e-12)

    # --- In-sample overfit diagnostic (identical to LOOCV) ---
    G_full    = G_all_gpu
    avg_w_eff = avg_weights_per_fold.mean(axis=0) * wa_gpu
    pred_full = cp.dot(avg_w_eff, G_full)
    y_z       = (y_all_gpu - y_all_gpu.mean()) / (y_all_gpu.std() + 1e-6)
    pred_z    = (pred_full - pred_full.mean()) / (pred_full.std() + 1e-6)
    if cp.std(pred_z) < 1e-9 or cp.std(y_z) < 1e-9:
        train_corr = 0.0
    else:
        train_corr = float(cp.corrcoef(pred_z, y_z)[0, 1])
        if np.isnan(train_corr):
            train_corr = 0.0

    all_weights  = avg_weights_per_fold.get()
    all_actuals  = all_actuals_gpu.get()
    all_preds    = all_preds_gpu.get()

    # Performance: one row per SAMPLE (same shape as LOOCV output)
    all_results = []
    for i in range(n_samples):
        all_results.append({
            'sample':  f"Sample_{i+1}",
            'actual':  float(all_actuals[i]),
            'predict': float(all_preds[i]),
        })
    performance_df  = pd.DataFrame(all_results)
    # Weights: one row per FOLD (K rows, not n_samples rows)
    fold_weights_df     = pd.DataFrame(all_weights, columns=rsid_all)
    fold_diagnostics_df = pd.DataFrame(fold_diagnostics_records)
    fold_losses         = {"Average_Loss": avg_history}
    loss_df = pd.DataFrame(fold_losses)
    if all_phase_lens:
        p1 = int(np.mean([p[0] for p in all_phase_lens]))
        p2 = int(np.mean([p[1] for p in all_phase_lens]))
        loss_df.attrs['phase_boundaries'] = (p1, p1 + p2)
    best_preds_norm = all_preds_gpu.copy()

    del weights_flat, weights_structured
    cp.get_default_memory_pool().free_all_blocks()

    return (
        performance_df,
        fold_weights_df,
        best_preds_norm.get(),
        loss_df,
        overall_weight_stability,
        train_corr,
        fold_diagnostics_df,
    )


# ──────────────────────────────────────────────
# TAS (Transferability-Adjusted Score) SNP filter
# ──────────────────────────────────────────────

def apply_tas_filter_cpu(G_cpu, w_cpu, a_vec_cpu, rsid_list, af_diff_arr,
                         current_gene_weight_df,
                         s_tier_threshold=0.05,
                         decay_scale=0.20,
                         keep_frac_ml=0.75,
                         min_snps_to_filter=15):
    """
    Pre-filter M/L-tier SNPs by transferability before GPU transfer, keeping all S-tier SNPs.

    Args:
        G_cpu (np.ndarray): shape (n_snps, n_samples), genotype dosage matrix.
        w_cpu (np.ndarray): shape (n_snps,), PrediXcan weights.
        a_vec_cpu (np.ndarray): shape (n_snps,), allele-similarity factor.
        rsid_list (list): length n_snps, SNP identifiers.
        af_diff_arr (np.ndarray): shape (n_snps,), |EUR_AF - EAS_AF| per SNP.
        current_gene_weight_df (pd.DataFrame): per-SNP weight/AF metadata for the gene, same row
            order as rsid_list.
        s_tier_threshold (float): af_diff cutoff below which a SNP is S-tier and always kept.
        decay_scale (float): exponential decay scale controlling how fast transferability drops
            as af_diff exceeds s_tier_threshold.
        keep_frac_ml (float): fraction of M/L-tier SNPs (by TAS score) to retain.
        min_snps_to_filter (int): if the gene has fewer total SNPs than this, filtering is skipped
            entirely and all SNPs are returned unmodified.

    Returns:
        tuple:
            G_cpu, w_cpu, a_vec_cpu, rsid_list, af_diff_arr, current_gene_weight_df — all filtered
                to the kept SNPs (or unmodified if min_snps_to_filter guard triggered).
            keep_idx (np.ndarray): indices (into the original arrays) of the SNPs that were kept.

    Notes:
        TAS (Transferability-Adjusted Score) = w_norm * transferability, where:
          w_norm = percentile rank of |PrediXcan weight| among M/L-tier SNPs (rankdata / n) —
              rewards SNPs that PrediXcan considered more important.
          transferability = exp(-(af_diff - s_tier_threshold) / decay_scale) — an exponential
              decay that penalizes SNPs whose EUR-EAS allele-frequency gap exceeds the S-tier
              cutoff, decaying faster for smaller decay_scale.
        Only the top ceil(len(ml_idx) * keep_frac_ml) M/L-tier SNPs by TAS score are kept; S-tier
        SNPs (af_diff <= s_tier_threshold) always pass through untouched, since their direction and
        magnitude are already considered reliable.
    """
    n_total_snps = len(rsid_list)

    # Early exit: If the total number of SNPs is less than the threshold, return everything unmodified.
    if n_total_snps < min_snps_to_filter:
        keep_idx = np.arange(n_total_snps)
        return (
            G_cpu,
            w_cpu,
            a_vec_cpu,
            rsid_list,
            af_diff_arr,
            current_gene_weight_df.copy(),
            keep_idx,
        )

    abs_w  = np.abs(w_cpu)
    s_idx  = np.where(af_diff_arr <= s_tier_threshold)[0]
    ml_idx = np.where(af_diff_arr >  s_tier_threshold)[0]

    keep_ml = np.array([], dtype=int)
    if len(ml_idx) > 0:
        abs_w_ml        = abs_w[ml_idx]
        af_diff_ml      = af_diff_arr[ml_idx]
        w_norm_ml       = rankdata(abs_w_ml) / len(abs_w_ml)
        excess          = af_diff_ml - s_tier_threshold
        transferability = np.exp(-excess / decay_scale)
        tas_ml          = w_norm_ml * transferability
        top_k           = int(np.ceil(len(ml_idx) * keep_frac_ml))
        if top_k > 0:
            keep_sub = np.argsort(tas_ml)[-top_k:]
            keep_ml  = ml_idx[keep_sub]

    keep_idx = np.sort(np.concatenate([s_idx, keep_ml]))

    return (
        G_cpu[keep_idx, :],
        w_cpu[keep_idx],
        a_vec_cpu[keep_idx],
        [rsid_list[i] for i in keep_idx],
        af_diff_arr[keep_idx],
        current_gene_weight_df.iloc[keep_idx].reset_index(drop=True),
        keep_idx,
    )


# ──────────────────────────────────────────────
# Single-gene GPU processing
# ──────────────────────────────────────────────

def process_single_gene_gpu(gene_id, subset_genotype, current_gene_weight,
                             current_real_expr, raw_predixcan_val, out_dir, ga_params):
    """
    End-to-end GA optimization pipeline for a single gene: filter, scale, optimize, evaluate, save.

    Args:
        gene_id (str): Ensembl gene ID (versionless).
        subset_genotype (pd.DataFrame): genotype rows for this gene's candidate SNPs, columns
            include 'ID', 'CHROM', 'POS', 'ALT', 'REF' plus one column per sample.
        current_gene_weight (pd.DataFrame): PrediXcan weight rows for this gene, with columns
            including 'rsid', 'weight', 'EUR_AF', 'EAS_AF'.
        current_real_expr (pd.DataFrame): single-row (this gene) observed expression, columns = samples.
        raw_predixcan_val (pd.Series): raw PrediXcan-predicted expression for this gene, indexed by sample.
        out_dir (str): base output directory; a `gene_id` subdirectory is created for this gene's outputs.
        ga_params (dict): GA hyperparameters (see build_ga_params / get_args), plus adaptive-scaling
            and TAS-filter knobs.

    Returns:
        dict: on success, the result of _build_gene_result(...) — a flat dict of scalar metrics
        (correlations, MAE, timings, stability, run counts) plus 'raw_predixcan'/'raw_actuals'/
        'raw_predictions' lists and a 'plot_payload' dict for save_plots_async. On failure or if
        fewer than 5 valid SNPs remain, returns a dict with 'gene_id', 'status' (error message),
        and 'time'.

    Notes:
        Pipeline stages, in order:
          1. Seed cp.random per-gene (hash of gene_id) so runs are reproducible but decorrelated
             across genes.
          2. Restrict PrediXcan weight rows to SNPs actually present in genotype data; skip the
             gene entirely if fewer than 5 SNPs remain.
          3. Compute af_diff = |EUR_AF - EAS_AF| and a_vec = clip(1 - af_diff, 0, 1) (allele
             similarity factor used throughout the GA/tiering).
          4. Compute the pre-GA baseline correlation (predixcan_corr_pre) between z-scored raw
             PrediXcan predictions and observed expression, used to drive several adaptive
             decisions below.
          5. Adaptive run-count scaling: more independent GA runs (effective_num_runs) are used
             when baseline correlation is negative (neg_corr_runs_scale) or weak
             (< 0.15, low_corr_runs_scale), and again scaled up for large genes
             (n_total_snps >= snp_count_runs_threshold, snp_count_runs_scale) since larger search
             spaces benefit from more restarts to escape local optima.
          6. Sign-flip guard: if allow_sign_flip is requested but baseline corr already clears
             sign_flip_threshold, sign-flip is disabled for this gene (already-good baseline
             direction shouldn't be risked).
          7. TAS filter (apply_tas_filter_cpu) drops low-transferability M/L-tier SNPs before any
             GPU transfer, shrinking the search space.
          8. Adaptive population/iteration scaling: if adaptive_scale is on, population_size and
             each phase's iteration count are scaled down proportionally to
             min(1, n_total_snps / adaptive_ref_snps), floored at min_population/min_iter1-3 —
             genes with few SNPs need less search budget.
          9. Transfer arrays to GPU and run LOOCV or K-Fold GA optimization
             (run_loocv_batch_ga_cupy / run_kfold_batch_ga_cupy) depending on ga_params['cv_method'].
          10. Aggregate per-fold GA scaling solutions into final weights: if sign-flip was enabled,
              use majority-vote sign alignment (_aggregate_weights_with_vote) since fold solutions
              may disagree on sign; otherwise use IQR-robust averaging + pruning
              (_aggregate_and_prune_weights).
          11. Recompute a leakage-free PrediXcan baseline z-score for fair comparison against the
              GA-optimized prediction, using the same CV scheme (per-fold train-only mean/std) as
              the GA evaluation.
          12. Package everything into plot_payload and return via _build_gene_result.
        On any exception, GPU memory is freed and a 'Failed: ...' status dict is returned instead
        of raising, so the caller's gene loop can continue processing other genes.
    """
    gene_seed = int(abs(hash(gene_id)) % (2**32 - 1))
    cp.random.seed(gene_seed)

    start_gene_time = time.time()

    try:
        gene_specific_dir = os.path.join(out_dir, gene_id)
        os.makedirs(gene_specific_dir, exist_ok=True)

        sample_cols = [c for c in subset_genotype.columns
                       if c not in ['ID', 'CHROM', 'POS', 'ALT', 'REF']]

        available_rsids = subset_genotype['ID'].values
        current_gene_weight = current_gene_weight[
            current_gene_weight['rsid'].isin(available_rsids)
        ].copy()

        rsid_list = current_gene_weight['rsid'].tolist()
        if len(rsid_list) < 5:
            logger.warning(f"  [{gene_id}] Skipped: Only {len(rsid_list)} valid SNPs found in Genotype (requires >= 5).")
            return {
                'gene_id': gene_id,
                'status': f'Skipped: only {len(rsid_list)} SNPs',
                'time': time.time() - start_gene_time
            }

        G_cpu = np.ascontiguousarray(
            subset_genotype.set_index('ID').loc[rsid_list][sample_cols].values,
            dtype=np.float32
        )
        y_cpu = np.ascontiguousarray(
            current_real_expr[sample_cols].iloc[0].values,
            dtype=np.float32
        )
        w_cpu = current_gene_weight['weight'].values.astype(np.float32)

        af_diff   = np.abs(current_gene_weight['EUR_AF'].values -
                           current_gene_weight['EAS_AF'].values)
        a_vec_cpu = np.clip(1.0 - af_diff, 0.0, 1.0).astype(np.float32)

        threshold       = ga_params.get('s_tier_threshold', 0.95)
        n_total_snps    = len(rsid_list)
        n_below_threshold = np.sum(a_vec_cpu >= threshold)
        n_samples       = G_cpu.shape[1]

        predixcan_aligned = raw_predixcan_val.reindex(sample_cols)
        if predixcan_aligned.isna().any():
            missing = predixcan_aligned[predixcan_aligned.isna()].index.tolist()
            raise ValueError(f"Missing PrediXcan values for samples: {missing[:5]}")

        predixcan_raw_pre = predixcan_aligned.values.astype(np.float32).ravel()
        predixcan_z_pre   = (predixcan_raw_pre - predixcan_raw_pre.mean()) / \
                             (predixcan_raw_pre.std() + 1e-6)
        y_z_pre           = (y_cpu - y_cpu.mean()) / (y_cpu.std() + 1e-6)

        if np.std(predixcan_z_pre) < 1e-9 or np.std(y_z_pre) < 1e-9:
            predixcan_corr_pre = 0.0
        else:
            predixcan_corr_pre = float(np.corrcoef(predixcan_z_pre, y_z_pre)[0, 1])
            if np.isnan(predixcan_corr_pre):
                predixcan_corr_pre = 0.0

        # Adaptive runs
        base_runs             = int(ga_params.get('num_runs', 20))
        neg_corr_scale        = float(ga_params.get('neg_corr_runs_scale', 2.0))
        low_corr_scale        = float(ga_params.get('low_corr_runs_scale', 1.5))

        if predixcan_corr_pre < 0.0:
            effective_num_runs = int(round(base_runs * neg_corr_scale))
        elif predixcan_corr_pre < 0.15:
            effective_num_runs = int(round(base_runs * low_corr_scale))
        else:
            effective_num_runs = base_runs

        # Adaptive run count by SNP size: genes with more SNPs get more runs
        snp_runs_scale     = float(ga_params.get('snp_count_runs_scale', 1.5))
        snp_runs_threshold = int(ga_params.get('snp_count_runs_threshold', 50))
        if n_total_snps >= snp_runs_threshold:
            effective_num_runs = int(round(effective_num_runs * snp_runs_scale))

        ga_params_gene   = {**ga_params, 'num_runs': max(1, effective_num_runs), 'gene_id': gene_id}

        # Sign-flip guard: disable sign flip when baseline corr does not pass the threshold
        sign_flip_threshold = float(ga_params.get('sign_flip_threshold', 0.01))
        if ga_params.get('allow_sign_flip', False) and predixcan_corr_pre >= sign_flip_threshold:
            ga_params_gene['allow_sign_flip'] = False

        actual_sign_flip = ga_params_gene.get('allow_sign_flip', False)
        logger.info(
            f"  [{gene_id}] baseline_corr={predixcan_corr_pre:.3f} -> "
            f"runs={effective_num_runs}, "
            f"sign_flip={'Yes' if actual_sign_flip else 'No'}"
            f"{'(overridden)' if ga_params.get('allow_sign_flip') and not actual_sign_flip else ''}"
        )

        # === TAS SNP filter (CPU-side, before GPU transfer) ===
        n_snps_before = len(rsid_list)
        G_cpu, w_cpu, a_vec_cpu, rsid_list, af_diff, current_gene_weight, _ = \
            apply_tas_filter_cpu(
                G_cpu, w_cpu, a_vec_cpu, rsid_list, af_diff,
                current_gene_weight,
                s_tier_threshold = ga_params.get('s_tier_af_diff_max', 0.05),
                decay_scale      = ga_params.get('tas_decay_scale', 0.20),
                keep_frac_ml     = ga_params.get('tas_keep_frac_ml', 0.75),
                min_snps_to_filter = ga_params.get('tas_min_snps_to_filter', 10)
            )
        n_total_snps = len(rsid_list)
        logger.info(f"  [{gene_id}] TAS filter: {n_snps_before} → {n_total_snps} SNPs")

        # === Adaptive population & iterations (scaled by post-TAS-filter SNP count) ===
        if ga_params.get('adaptive_scale', True):
            ref_snps    = ga_params.get('adaptive_ref_snps', 20)
            scale       = min(1.0, n_total_snps / ref_snps)

            base_pop    = ga_params.get('population_size', 500)
            base_iter1  = ga_params.get('iter_num_s1', 500)
            base_iter2  = ga_params.get('iter_num_s2', 750)
            base_iter3  = ga_params.get('iter_num_s3', 250)

            adapt_pop   = max(ga_params.get('min_population', 300), int(base_pop   * scale))
            adapt_iter1 = max(ga_params.get('min_iter1', 200),      int(base_iter1 * scale))
            adapt_iter2 = max(ga_params.get('min_iter2', 300),      int(base_iter2 * scale))
            adapt_iter3 = max(ga_params.get('min_iter3', 100),      int(base_iter3 * scale))

            ga_params_gene.update({
                'population_size': adapt_pop,
                'iter_num_s1':     adapt_iter1,
                'iter_num_s2':     adapt_iter2,
                'iter_num_s3':     adapt_iter3,
            })
            logger.info(
                f"  [{gene_id}] Adaptive scale={scale:.2f}: "
                f"pop={adapt_pop}, iter=({adapt_iter1}/{adapt_iter2}/{adapt_iter3})"
            )

        transfer_start = time.time()
        G_gpu     = cp.array(G_cpu)
        y_gpu     = cp.array(y_cpu)
        w_gpu     = cp.array(w_cpu)
        a_vec_gpu = cp.array(a_vec_cpu)
        transfer_time = time.time() - transfer_start

        compute_start = time.time()
        cv_method = ga_params.get('cv_method', 'loocv')

        if cv_method == 'loocv':
            performance_df, weight_history_df, ga_best_dist, loss_df, weight_stability, train_corr, fold_diagnostics_df = \
                run_loocv_batch_ga_cupy(G_gpu, y_gpu, w_gpu, a_vec_gpu, rsid_list, **ga_params_gene)
        else:
            if cv_method == 'kfold':
                ga_params_gene['n_repeats'] = 1
            performance_df, weight_history_df, ga_best_dist, loss_df, weight_stability, train_corr, fold_diagnostics_df = \
                run_kfold_batch_ga_cupy(G_gpu, y_gpu, w_gpu, a_vec_gpu, rsid_list, **ga_params_gene)

        compute_time = time.time() - compute_start

        weight_history_df.index.name = 'Fold_ID'
        weight_filename = f"{cv_method}_weights.csv"
        weight_history_df.to_csv(os.path.join(gene_specific_dir, weight_filename), index=True)
        performance_df.to_csv(os.path.join(gene_specific_dir, "performance.csv"), index=False)
        fold_diagnostics_df.to_csv(os.path.join(gene_specific_dir, "fold_diagnostics.csv"), index=False)
        train_corr_std = float(fold_diagnostics_df['train_corr'].std())

        prune_thresh = ga_params.get('pruning_threshold', 1e-8)
        if ga_params.get('allow_sign_flip', False):
            # Sign-flip enabled: vote-based aggregation is needed since fold solutions may disagree
            # on sign; without sign-flip all scalings are positive, so simple averaging suffices.
            avg_weights, inconsistency_rate, bad_snps, inconsistent_mask = \
                _aggregate_weights_with_vote(
                    weight_history_df.values,   # (n_folds, n_snps)
                    rsid_list,
                    inconsistent_thresh=0.5
                )
            if inconsistency_rate > 0.3:
                logger.warning(
                    f"  [{gene_id}] {inconsistency_rate:.1%} of SNPs had unstable sign votes and were zeroed"
                )
            std_weights      = weight_history_df.std(axis=0).values
            ga_final_weights = w_cpu * a_vec_cpu * avg_weights
            ga_final_weights[np.abs(ga_final_weights) < prune_thresh] = 0.0
        else:
            avg_weights, std_weights, ga_final_weights = _aggregate_and_prune_weights(
                weight_history_df, w_cpu, a_vec_cpu, prune_thresh
            )

        stability_penalty = float(std_weights.mean())

        stability_warn_thresh = ga_params.get('stability_warn_threshold', 0.5)

        if stability_penalty > stability_warn_thresh:
            logger.warning(
                f"  [{gene_id}] High weight instability: std={stability_penalty:.3f} "
                f"(threshold={stability_warn_thresh}). "
                f"GA results vary too much across folds; predictions may be unreliable."
            )

        detailed_weight_df = pd.DataFrame({
            'Gene':             gene_id,
            'rsid':             rsid_list,
            'EUR_AF':           current_gene_weight['EUR_AF'].values,
            'EAS_AF':           current_gene_weight['EAS_AF'].values,
            'AF_Diff':          af_diff,
            'PrediXcan_W':      w_cpu,
            'Allele_Factor':    a_vec_cpu,
            'GA_Scaling_Mean':  avg_weights,
            'GA_Scaling_Std':   std_weights,
            'GA_W':             ga_final_weights,
            'source':           'GA'
        })
        detailed_weight_df.to_csv(
            os.path.join(gene_specific_dir, "optimized_weights_detail.csv"), index=False
        )
        loss_df.to_csv(os.path.join(gene_specific_dir, "loss_history.csv"), index=True)

        real_z  = performance_df['actual'].to_numpy(dtype=np.float32).ravel()
        ga_z    = performance_df['predict'].to_numpy(dtype=np.float32).ravel()

        predixcan_raw = predixcan_aligned.values.astype(np.float32).ravel()
        predixcan_z   = np.empty_like(predixcan_raw)
        n_eval_samples = predixcan_raw.shape[0]

        if cv_method == 'loocv':
            for i in range(n_eval_samples):
                train_mask = np.ones(n_eval_samples, dtype=bool)
                train_mask[i] = False
                px_mu  = predixcan_raw[train_mask].mean()
                px_std = predixcan_raw[train_mask].std() + 1e-6
                predixcan_z[i] = (predixcan_raw[i] - px_mu) / px_std
        else:
            n_splits  = ga_params.get('n_splits', 5)
            n_repeats = ga_params.get('n_repeats', 1)
            cv_seed   = ga_params.get('cv_seed', 42)
            if n_repeats > 1:
                splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed)
            else:
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
            pred_acc  = np.zeros(n_eval_samples)
            count_acc = np.zeros(n_eval_samples)
            for train_idx, test_idx in splitter.split(np.arange(n_eval_samples)):
                px_mu  = predixcan_raw[train_idx].mean()
                px_std = predixcan_raw[train_idx].std() + 1e-6
                pred_acc[test_idx]  += (predixcan_raw[test_idx] - px_mu) / px_std
                count_acc[test_idx] += 1.0
            predixcan_z = pred_acc / (count_acc + 1e-12)

        mae_ga        = np.mean(np.abs(real_z - ga_z))
        mae_predixcan = np.mean(np.abs(real_z - predixcan_z))

        if np.std(ga_z) < 1e-9 or np.std(real_z) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(real_z, ga_z)[0, 1]
            if np.isnan(corr):
                corr = 0.0

        if np.std(predixcan_z) < 1e-9 or np.std(real_z) < 1e-9:
            raw_corr = 0.0
        else:
            raw_corr = np.corrcoef(real_z, predixcan_z)[0, 1]
            if np.isnan(raw_corr):
                raw_corr = 0.0

        cv_label_map = {
            'loocv':          'LOOCV',
            'kfold':          'KFold',
            'repeated_kfold': 'RepKFold'
        }

        phase_boundaries = loss_df.attrs.get('phase_boundaries', None)

        plot_payload = {
            'gene_id':           gene_id,
            'gene_specific_dir': gene_specific_dir,
            'loss_df':           loss_df.copy(),
            'phase_boundaries':  phase_boundaries,
            'real_dist':         real_z,
            'predixcan_dist':    predixcan_z,
            'ga_best_dist':      ga_z,
            'mae_predixcan':     mae_predixcan,
            'mae_ga':            mae_ga,
            'n_total_snps':      n_total_snps,
            'n_below_threshold': n_below_threshold,
            'threshold_val':     1.0 - threshold,
            'Correlation before GA': float(raw_corr),
            'Correlation after GA':  float(corr),
            'cv_tag':            cv_label_map.get(cv_method, 'CV'),
        }

        del G_gpu, y_gpu, w_gpu, a_vec_gpu
        cp.get_default_memory_pool().free_all_blocks()

        total_time = time.time() - start_gene_time

        return _build_gene_result(
            gene_id, raw_corr, corr, mae_predixcan, mae_ga,
            total_time, weight_stability, train_corr, train_corr_std, predixcan_corr_pre,
            effective_num_runs,
            transfer_time, compute_time,
            n_total_snps, n_samples, predixcan_z, real_z, ga_z, plot_payload,
            stability_penalty=stability_penalty
        )

    except Exception as e:
        cp.get_default_memory_pool().free_all_blocks()
        logger.error(f"Gene {gene_id} failed: {traceback.format_exc()}")
        return {
            'gene_id': gene_id,
            'status':  f'Failed: {str(e)}',
            'time':    time.time() - start_gene_time
        }


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def save_plots_async(payload):
    """
    Render and save the four diagnostic plots for one gene (run off the main thread).

    Args:
        payload (dict): plot_payload produced by process_single_gene_gpu, containing
            'gene_id', 'gene_specific_dir', 'loss_df', 'phase_boundaries', 'real_dist',
            'predixcan_dist', 'ga_best_dist', 'mae_predixcan', 'mae_ga', 'n_total_snps',
            'n_below_threshold', 'threshold_val', 'Correlation before GA', 'Correlation after GA'.

    Returns:
        None. Writes PNG files into payload['gene_specific_dir']:
            loss_curve.png, distribution_comparison.png, correlation_scatter.png,
            sample_wise_scatter.png.

    Notes:
        All distributions received here (real_dist/predixcan_dist/ga_best_dist) are already
        Z-score normalized, so all four plots share a common, comparable scale.
        loss_curve.png overlays vertical lines at phase_boundaries (Phase1->2 and Phase2->3
        iteration indices) when available, to visually separate the 3 GA phases.
        Exceptions are caught and logged rather than raised, since this function runs inside a
        ThreadPoolExecutor worker and must not crash the pool.
    """
    try:
        gene_id  = payload['gene_id']
        out_dir  = payload['gene_specific_dir']
        loss_df  = payload['loss_df']
        # All inputs received here are already Z-score values
        real_z   = payload['real_dist']
        predixcan_z = payload['predixcan_dist']
        ga_z     = payload['ga_best_dist']
        n_total  = payload.get('n_total_snps', 0)
        n_below  = payload.get('n_below_threshold', 0)
        thresh   = payload.get('threshold_val', 0.05)
        mae_predixcan = payload.get('mae_predixcan', 0.0)
        mae_ga   = payload.get('mae_ga', 0.0)
        corr_before = payload.get('Correlation before GA', 0.0)
        corr_after  = payload.get('Correlation after GA', 0.0)

        # 1. Loss Curve
        plt.figure(figsize=(10, 6))
        loss_data = loss_df.mean(axis=1) if loss_df.shape[1] > 1 else loss_df.iloc[:, 0]
        plt.plot(loss_data, color='red', linewidth=2, label='Avg Loss')
        phase_boundaries = payload.get('phase_boundaries', None)
        if phase_boundaries:
            plt.axvline(x=phase_boundaries[0], color='steelblue', linestyle='--',
                        linewidth=1.5, alpha=0.85, label=f'Phase 1→2 (iter {phase_boundaries[0]})')
            plt.axvline(x=phase_boundaries[1], color='forestgreen', linestyle='--',
                        linewidth=1.5, alpha=0.85, label=f'Phase 2→3 (iter {phase_boundaries[1]})')
        plt.legend(fontsize=9, loc='upper right')
        plt.title(f"GA Convergence: {gene_id}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(out_dir, "loss_curve.png"))
        plt.close()

        # 2. Distribution (Z-score Comparison)
        plt.figure(figsize=(10, 6))
        if len(real_z) > 1 and np.std(real_z) > 0:
            sns.kdeplot(real_z, label='Real (Z-score)', color='black', linewidth=2)
        if len(predixcan_z) > 1 and np.std(predixcan_z) > 0:
            sns.kdeplot(predixcan_z, label='PrediXcan (Z-score)', color='gray', linestyle=':')
        if len(ga_z) > 1 and np.std(ga_z) > 0:
            sns.kdeplot(ga_z, label='GA Optimized (Z-score)', color='red', linewidth=1.5)
        plt.title(f"Distribution: {gene_id} (Standardized)")
        plt.xlabel("Z-Score")
        plt.legend()
        plt.grid(alpha=0.3)
        stats_text = (f"Total SNPs: {n_total}\nAF_Diff < {thresh:.2f}: {n_below}")
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.savefig(os.path.join(out_dir, "distribution_comparison.png"))
        plt.close()

        # 3. Correlation Scatter Plot
        plt.figure(figsize=(7, 7))
        # Plot PrediXcan (Before) in blue
        sns.scatterplot(x=real_z, y=predixcan_z, color='blue', alpha=0.3,
                        label=f'PrediXcan (Corr: {corr_before:.3f})', s=40)
        # Plot GA (After) in red (slightly larger and overlaid on top)
        sns.scatterplot(x=real_z, y=ga_z, color='red', alpha=0.3,
                        label=f'GA (Corr: {corr_after:.3f})', s=40)
        # Draw reference line
        min_val = min(min(real_z), min(ga_z), min(predixcan_z))
        max_val = max(max(real_z), max(ga_z), max(predixcan_z))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        plt.title(f"Correlation Improvement: {gene_id}")
        plt.xlabel("Real Expression (Z-score)")
        plt.ylabel("Predicted Expression (Z-score)")
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(os.path.join(out_dir, "correlation_scatter.png"))
        plt.close()

        # 4. Sample-wise Line Plot
        plt.figure(figsize=(12, 6))
        sample_indices = np.arange(len(real_z))
        # (1) Real Data: black solid line
        plt.plot(sample_indices, real_z, color='black', label='Real (Z-score)', linewidth=2, alpha=0.8)
        plt.plot(sample_indices, ga_z, color='red',
                 label=f'GA (MAE: {mae_ga:.3f})', linewidth=2, alpha=0.8)
        plt.plot(sample_indices, predixcan_z, color='blue',
                 label=f'PrediXcan (MAE: {mae_predixcan:.3f})',
                 linewidth=1.5, linestyle='--', alpha=0.5)
        plt.title(f"Sample-wise Expression Trend: {gene_id}")
        plt.xlabel("Sample Index")
        plt.ylabel("Expression Value (Z-score)")
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sample_wise_scatter.png"))
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting for {payload.get('gene_id', 'unknown')}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# process_single_gene_gpu helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _build_gene_result(
    gene_id, raw_corr, corr, mae_predixcan, mae_ga,
    total_time, weight_stability, train_corr, train_corr_std, predixcan_corr_pre,
    effective_num_runs,
    transfer_time, compute_time,
    n_total_snps, n_samples, predixcan_z, real_z, ga_z, plot_payload,
    stability_penalty=0.0
):
    """
    Package one gene's GA results into the flat dictionary consumed by main()'s aggregation loop.

    Args:
        gene_id (str): gene identifier.
        raw_corr (float): PrediXcan-vs-real correlation before GA optimization.
        corr (float): GA-optimized-vs-real correlation after optimization.
        mae_predixcan (float): mean absolute error of PrediXcan predictions vs real (z-scored).
        mae_ga (float): mean absolute error of GA predictions vs real (z-scored).
        total_time (float): wall-clock seconds spent processing this gene.
        weight_stability (float): overall weight stability metric from the CV routine.
        train_corr (float): in-sample (overfit-diagnostic) correlation.
        train_corr_std (float): std of per-fold training correlations.
        predixcan_corr_pre (float): baseline PrediXcan correlation used for adaptive-run decisions.
        effective_num_runs (int): actual number of GA runs used per fold for this gene.
        transfer_time (float): seconds spent transferring arrays to GPU.
        compute_time (float): seconds spent in GA/CV computation.
        n_total_snps (int): number of SNPs used after TAS filtering.
        n_samples (int): number of samples for this gene.
        predixcan_z, real_z, ga_z (array-like): per-sample z-scored values for downstream global
            aggregation (converted to plain Python lists in the output).
        plot_payload (dict): payload to be consumed later by save_plots_async.
        stability_penalty (float): mean std of GA scaling factors, used as a per-gene reliability flag.

    Returns:
        dict: gene-level result record with 'status': 'Success' and all metrics above, plus
        'raw_predixcan'/'raw_actuals'/'raw_predictions' (as lists) and 'plot_payload'.
    """
    return {
        'gene_id': gene_id,
        'status': 'Success',
        'Correlation before GA': float(raw_corr),
        'Correlation after GA': float(corr),
        'MAE_PrediXcan_Real': float(mae_predixcan),
        'MAE_GA_Real': float(mae_ga),
        'time': float(total_time),
        'weight_stability': float(weight_stability),
        'weight_stability_penalty': float(stability_penalty),
        'train_corr': float(train_corr),
        'train_corr_std': float(train_corr_std),
        'baseline_corr': float(predixcan_corr_pre),
        'effective_runs': int(effective_num_runs),
        'transfer_time': float(transfer_time),
        'compute_time': float(compute_time),
        'n_snps': int(n_total_snps),
        'n_samples': int(n_samples),

        # Converted to plain Python lists so downstream code can use .extend() in main().
        'raw_predixcan': predixcan_z.tolist() if hasattr(predixcan_z, 'tolist') else list(predixcan_z),
        'raw_actuals': real_z.tolist() if hasattr(real_z, 'tolist') else list(real_z),
        'raw_predictions': ga_z.tolist() if hasattr(ga_z, 'tolist') else list(ga_z),

        'plot_payload': plot_payload
    }

def _aggregate_and_prune_weights(weight_history_df, w_cpu, a_vec_cpu, prune_thresh):
    """
    Aggregate per-fold GA scaling factors via IQR-robust averaging, then prune tiny final weights.

    Args:
        weight_history_df (pd.DataFrame): shape (n_folds, n_snps), GA scaling factors per fold.
        w_cpu (np.ndarray): shape (n_snps,), original PrediXcan weights.
        a_vec_cpu (np.ndarray): shape (n_snps,), allele-similarity factor.
        prune_thresh (float): final weights with |value| below this threshold are hard-set to 0.

    Returns:
        tuple:
            avg_weights (np.ndarray): shape (n_snps,), IQR-robust mean scaling factor per SNP.
            std_weights (np.ndarray): shape (n_snps,), plain std of scaling factor per SNP across folds.
            ga_final_weights (np.ndarray): shape (n_snps,), final weight = w_cpu * a_vec_cpu * avg_weights,
                with small-magnitude entries pruned to exactly 0.

    Notes:
        "IQR-robust" here means: for each SNP, fold values outside [Q1-1.5*IQR, Q3+1.5*IQR] are
        excluded from the mean (but not from the reported std), reducing the influence of a single
        outlier fold on the final aggregated scaling factor. Used only when sign-flip is disabled,
        where all scaling values share the same sign and a simple robust mean is appropriate.
    """
    weights_array = weight_history_df.values
    Q1            = np.percentile(weights_array, 25, axis=0)
    Q3            = np.percentile(weights_array, 75, axis=0)
    IQR           = Q3 - Q1
    lower         = Q1 - 1.5 * IQR
    upper         = Q3 + 1.5 * IQR
    mask_inlier   = (weights_array >= lower) & (weights_array <= upper)
    inlier_sum    = (weights_array * mask_inlier).sum(axis=0)
    inlier_count  = mask_inlier.sum(axis=0) + 1e-9
    avg_weights   = inlier_sum / inlier_count
    std_weights   = weight_history_df.std(axis=0).values.copy()

    ga_final_weights = w_cpu * a_vec_cpu * avg_weights

    if prune_thresh > 0:
        ga_final_weights[np.abs(ga_final_weights) < prune_thresh] = 0.0

    return avg_weights, std_weights, ga_final_weights

def _aggregate_weights_with_vote(weights_per_fold, rsid_list, inconsistent_thresh=0.5):
    """
    Aggregate per-fold GA scaling factors under sign-flip mode using majority-vote sign alignment.

    Args:
        weights_per_fold (np.ndarray): shape (n_folds, n_snps), GA scaling factors per fold
            (may be positive or negative per SNP under sign-flip mode).
        rsid_list (list): length n_snps, SNP identifiers.
        inconsistent_thresh (float): minimum |mean sign| (vote_ratio) required to trust a SNP's
            majority-vote direction; SNPs below this are considered truly unreliable and zeroed.

    Returns:
        tuple:
            avg_weights (np.ndarray): shape (n_snps,), sign-aligned mean scaling factor per SNP
                (0 for SNPs whose direction vote was too inconsistent).
            inconsistency_rate (float): fraction of SNPs zeroed out due to unstable sign voting.
            inconsistent_rsids (list): rsids of the zeroed-out SNPs.
            inconsistent_mask (np.ndarray): shape (n_snps,), boolean mask of zeroed-out SNPs.

    Notes:
        Because sign-flip mode allows each fold's GA run to independently choose a SNP's direction,
        naively averaging raw scaling values could cancel out a genuinely useful but sign-flipping
        SNP. Instead: (1) compute each fold's sign per SNP, (2) take the majority sign
        (mean_sign, ties default to +1), (3) flip any fold's value that disagrees with the majority
        sign before averaging, so magnitudes combine constructively regardless of fold-level sign
        noise. SNPs whose votes are close to a 50/50 split (|mean_sign| < inconsistent_thresh) are
        deemed genuinely unreliable and zeroed rather than trusted to an arbitrary majority.
    """
    signs     = np.sign(weights_per_fold)   # (n_folds, n_snps); each value is +1 or -1
    mean_sign = signs.mean(axis=0)          # (n_snps,); positive -> majority positive, negative -> majority negative

    majority_sign = np.sign(mean_sign)      # (n_snps,)
    majority_sign[majority_sign == 0] = 1.0 # exact tie defaults to positive

    # Align each fold's scaling to the majority-vote direction before averaging;
    # flip any fold whose sign disagrees with the majority.
    aligned_weights = weights_per_fold * majority_sign[np.newaxis, :]  # (n_folds, n_snps)
    avg_weights     = aligned_weights.mean(axis=0)                     # (n_snps,)

    # SNPs with a low vote ratio (even the majority direction is unstable) are treated as
    # genuinely unreliable and zeroed out.
    vote_ratio         = np.abs(mean_sign)                  # 0~1, higher = more consistent
    inconsistent_mask  = vote_ratio < inconsistent_thresh
    avg_weights[inconsistent_mask] = 0.0

    inconsistency_rate = inconsistent_mask.mean()
    inconsistent_rsids = [rsid_list[i] for i in np.where(inconsistent_mask)[0]]

    return avg_weights, inconsistency_rate, inconsistent_rsids, inconsistent_mask


# ──────────────────────────────────────────────
# Task Generator
# ──────────────────────────────────────────────

def task_generator(common_genes, gene_to_rsids, genotype_all_pd,
                   db_weight_f, real_expr_f, predict_expr_f, out_dir, ga_params):
    """
    Lazily yield per-gene input bundles ready for process_single_gene_gpu.

    Args:
        common_genes (list): gene IDs to process, in iteration order.
        gene_to_rsids (dict): gene_id -> list of rsids targeting that gene (from the weight table).
        genotype_all_pd (pd.DataFrame): genotype matrix indexed by rsid, already loaded into RAM.
        db_weight_f (pd.DataFrame): filtered PrediXcan weight table (post allele-freq filtering).
        real_expr_f (pd.DataFrame): filtered real expression table, indexed by gene.
        predict_expr_f (pd.DataFrame): filtered PrediXcan predicted expression table, indexed by gene.
        out_dir (str): base output directory (forwarded unchanged to each task).
        ga_params (dict): GA hyperparameters (forwarded unchanged to each task).

    Yields:
        tuple: (gene_id, subset_genotype, subset_weight, subset_real_expr, raw_predixcan_val,
        out_dir, ga_params) — positional arguments matching process_single_gene_gpu's signature.

    Notes:
        Genes with no valid (genotype-present) target rsids are silently skipped. Any exception
        while preparing a single gene's data (e.g. KeyError from a missing index entry) is logged
        and that gene is skipped rather than aborting the whole generator, so one bad gene doesn't
        stop the batch run.
    """
    existing_rsids = set(genotype_all_pd.index)

    for gene_id in common_genes:
        try:
            all_target_rsids = gene_to_rsids.get(gene_id, [])
            valid_rsids      = [rs for rs in all_target_rsids if rs in existing_rsids]
            if not valid_rsids:
                continue
            try:
                subset_genotype = genotype_all_pd.loc[valid_rsids].reset_index()
            except KeyError:
                continue

            subset_weight     = db_weight_f[db_weight_f['gene'] == gene_id].copy()
            subset_real_expr  = real_expr_f.loc[[gene_id]].copy()
            raw_predixcan_val = predict_expr_f.loc[gene_id].copy()

            yield (gene_id, subset_genotype, subset_weight,
                   subset_real_expr, raw_predixcan_val, out_dir, ga_params)
        except Exception as e:
            logger.error(f"Error preparing data for gene {gene_id}: {str(e)}")
            continue


# ─────────────────────────────────────────────────────────────────────────────
# main helper functions
# ─────────────────────────────────────────────────────────────────────────────

def select_test_genes(common_genes, real_expr_f, test_count, seed, num_bins=None):
    """
    Stratified test-gene sampling by expression std bins with proportional allocation.

    Args:
        common_genes (list): full pool of candidate gene IDs.
        real_expr_f (pd.DataFrame): real expression table indexed by gene, columns = samples.
        test_count (int): desired number of genes to sample for test mode.
        seed (int): random seed for reproducible sampling.
        num_bins (int, optional): number of std-based strata; defaults to 10 if test_count >= 100 else 5.

    Returns:
        tuple:
            selected_genes (list): sampled gene IDs.
            gene_bin_map (dict): gene_id -> bin index, for logging which expression-variability
                stratum each sampled gene came from.

    Notes:
        Genes are binned by their across-sample expression std (pd.cut into num_bins equal-width
        bins), and sample counts are allocated to bins proportionally to bin size, with the
        Largest Remainder Method distributing the rounding remainder to the bins with the largest
        fractional allocation. This ensures genes with a wide range of expression variability are
        all represented in test mode, rather than random sampling skewing toward the most common
        std range. Falls back to plain random sampling if the stratification step raises.
    """
    import random
    random.seed(seed)
    count    = min(test_count, len(common_genes))
    num_bins = num_bins or (10 if count >= 100 else 5)

    logger.info("=" * 70)
    logger.info(f" TEST MODE ENABLED (Seed: {seed})")
    logger.info(f" Target Test Count: {count}")
    logger.info(f" Stratification Strategy: {num_bins} Bins based on Expression Std")

    test_gene_stats = real_expr_f.loc[common_genes].std(axis=1)
    try:
        test_gene_stats_df = pd.DataFrame({'std': test_gene_stats})
        test_gene_stats_df['bin'] = pd.cut(test_gene_stats_df['std'], bins=num_bins, labels=False)
        ret_bins = pd.cut(test_gene_stats_df['std'], bins=num_bins, retbins=True)[1]

        logger.info(f" Stratification Ranges (Std):")
        for i in range(len(ret_bins) - 1):
            logger.info(f"   Bin {i}: {ret_bins[i]:.4f} - {ret_bins[i+1]:.4f}")

        bin_sizes   = test_gene_stats_df.groupby('bin').size()
        total_g     = bin_sizes.sum()
        raw_alloc   = {b: count * (sz / total_g) for b, sz in bin_sizes.items()}
        floor_alloc = {b: int(v) for b, v in raw_alloc.items()}
        remainder   = count - sum(floor_alloc.values())
        fractional  = sorted(raw_alloc.keys(), key=lambda b: raw_alloc[b] - floor_alloc[b], reverse=True)
        for b in fractional[:remainder]:
            floor_alloc[b] += 1

        selected_genes = []
        gene_bin_map   = {}
        for bin_idx in range(num_bins):
            genes_in_bin = test_gene_stats_df[test_gene_stats_df['bin'] == bin_idx].index.tolist()
            n_sample     = min(floor_alloc.get(bin_idx, 0), len(genes_in_bin))
            sampled      = random.sample(genes_in_bin, n_sample)
            selected_genes.extend(sampled)
            for g in sampled:
                gene_bin_map[g] = bin_idx
            logger.info(f"   - Bin {bin_idx} (Std range): Sampled {n_sample} genes")

    except Exception as e:
        logger.warning(f" Stratified sampling failed ({e}), falling back to random sampling.")
        selected_genes = random.sample(common_genes, count)
        gene_bin_map   = {}

    logger.info(f" Final Selected Genes: {len(selected_genes)}")
    logger.info("=" * 70)
    return selected_genes, gene_bin_map


def build_ga_params(args):
    """
    Translate parsed CLI args into the flat GA-hyperparameter dict passed through the pipeline.

    Args:
        args (argparse.Namespace): parsed arguments from get_args().

    Returns:
        dict: GA hyperparameters keyed as expected by Weight_Generate_Population_Cupy,
        Weight_Mutation_Cupy, Weight_GA_Cupy, run_loocv_batch_ga_cupy, run_kfold_batch_ga_cupy,
        apply_tas_filter_cpu, and process_single_gene_gpu.

    Notes:
        s_tier_threshold/m_tier_threshold are derived as 1 - af_diff_max, converting a "maximum
        allowed AF difference" CLI input into the "minimum allele-similarity" internal threshold
        used for tier classification (a_vec = 1 - af_diff).
        n_repeats is forced to 1 unless cv_method == 'repeated_kfold', so plain 'loocv'/'kfold'
        runs never accidentally repeat folds.
    """
    return {
        'num_runs':              args.Numbers_of_GA_runs,
        'iter_num_s1':           args.GA_Phase1_iterations,
        'iter_num_s2':           args.GA_Phase2_iterations,
        'iter_num_s3':           args.GA_Phase3_iterations,
        'population_size':       args.GA_population_sizes,
        'initial_mutation_rate': args.Initial_mutation_rate,
        's_tier_threshold':      1.0 - args.s_tier_af_diff_max,
        's_tier_low':            args.s_tier_low_bound,
        's_tier_high':           args.s_tier_high_bound,
        'm_tier_threshold':      1.0 - args.m_tier_af_diff_max,
        'm_tier_low':            args.m_tier_low_bound,
        'm_tier_high':           args.m_tier_high_bound,
        'l_tier_low':            args.l_tier_low_bound,
        'l_tier_high':           args.l_tier_high_bound,
        'cv_method':             args.cv_method,
        'n_splits':              args.n_splits,
        'n_repeats':              args.n_repeats if args.cv_method == 'repeated_kfold' else 1,
        'cv_seed':               args.cv_seed,
        'batch_limit':           args.batch_limit,
        'tas_decay_scale':       args.tas_decay_scale,
        'tas_keep_frac_ml':      args.tas_keep_frac_ml,
        'tas_min_snps_to_filter': args.tas_min_snps_to_filter,
        'adaptive_scale':        args.adaptive_scale,
        'adaptive_ref_snps':     args.adaptive_ref_snps,
        'min_population':        args.min_population,
        'min_iter1':             args.min_iter1,
        'min_iter2':             args.min_iter2,
        'min_iter3':             args.min_iter3,
        'pruning_threshold':     args.pruning_threshold,
        'allow_sign_flip':       args.allow_sign_flip,
        'sign_flip_threshold':   args.sign_flip_threshold,
    }


def _plot_overfitting_diagnostic(ax, df, cv_tag, annotate=False):
    """
    Draw a 4-quadrant overfitting/instability diagnostic scatter plot on a given Axes.

    Args:
        ax (matplotlib.axes.Axes): target axes to draw on.
        df (pd.DataFrame): must contain columns 'train_corr', 'train_corr_std',
            'Correlation after GA', 'gene_id' (one row per gene).
        cv_tag (str): cross-validation label used in the x-axis caption (e.g. 'LOOCV', 'KFold').
        annotate (bool): if True, label the 5 worst-performing genes (by Correlation after GA)
            in each non-Normal quadrant with their gene_id.

    Returns:
        None. Draws directly onto `ax`; renders a placeholder message if required columns/data
        are missing.

    Notes:
        X = train_gap = train_corr - Correlation after GA (higher = more overfit: good on training
            data but not on held-out data).
        Y = train_corr_std (per-fold instability of the training correlation across folds).
        Color = Correlation after GA (actual validation performance).
        Quadrant thresholds: gap_thresh is fixed at 0.15; std_thresh is adaptive
            (75th percentile of std, floored at 0.02) to avoid every gene being flagged "unstable"
            when most std values are near 0 (common under LOOCV, which has only 1 held-out sample
            per fold and thus low fold-to-fold variance in some metrics).
        Four categories: Normal (low gap, low std), Unstable (low gap, high std), Sys.Overfit
            (high gap, low std — consistently overfits every fold the same way), Overfit+Unstable
            (high gap, high std — both overfits and is inconsistent about it, the worst case).
    """
    need_cols = {'train_corr', 'train_corr_std', 'Correlation after GA', 'gene_id'}
    if not need_cols.issubset(df.columns):
        ax.text(0.5, 0.5, 'train_corr_std not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='gray')
        ax.set_title("Overfitting Diagnostic", fontsize=13)
        return

    valid = df[list(need_cols)].dropna()
    if valid.empty:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='gray')
        return

    gap        = (valid['train_corr'] - valid['Correlation after GA']).values
    std        = valid['train_corr_std'].values
    corr_after = valid['Correlation after GA'].values
    gene_ids   = valid['gene_id'].values

    # Adaptive std threshold: 75th percentile (avoids flagging everyone "high std" when
    # LOOCV std is almost always 0).
    gap_thresh = 0.15
    std_thresh = max(float(np.percentile(std, 75)), 0.02)

    cat = np.array(['Normal'] * len(gap), dtype=object)
    cat[(gap <= gap_thresh) & (std > std_thresh)]  = 'Unstable'
    cat[(gap > gap_thresh)  & (std <= std_thresh)] = 'Sys.Overfit'
    cat[(gap > gap_thresh)  & (std > std_thresh)]  = 'Overfit+Unstable'

    palette = {
        'Normal':           '#4CAF50',
        'Unstable':         '#FF9800',
        'Sys.Overfit':      '#F44336',
        'Overfit+Unstable': '#9C27B0',
    }
    markers = {
        'Normal': 'o', 'Unstable': 's', 'Sys.Overfit': '^', 'Overfit+Unstable': 'D'
    }

    sc = None
    for label, color in palette.items():
        mask = cat == label
        if not mask.any():
            continue
        sc = ax.scatter(
            gap[mask], std[mask],
            c=corr_after[mask], cmap='RdYlGn', vmin=-0.3, vmax=0.8,
            alpha=0.85, s=55 if annotate else 40,
            edgecolors=color, linewidth=1.2,
            marker=markers[label],
            label=f'{label} (n={mask.sum()})',
            zorder=3
        )
        if annotate:
            # Annotate the 5 worst (by corr_after) genes in each non-Normal quadrant.
            if label != 'Normal':
                worst_idx = np.where(mask)[0][np.argsort(corr_after[mask])[:5]]
                for idx in worst_idx:
                    ax.annotate(
                        gene_ids[idx],
                        (gap[idx], std[idx]),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=6, color=color, alpha=0.9
                    )

    # Quadrant divider lines
    ax.axvline(gap_thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1.2)
    ax.axhline(std_thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1.2)

    # Quadrant labels (top area)
    xmax = max(gap.max() * 1.05, gap_thresh * 1.3)
    ymax = max(std.max() * 1.05, std_thresh * 1.5)
    ax.text(gap_thresh * 0.5, ymax * 0.97, 'Normal / Unstable',
            ha='center', va='top', fontsize=7, color='gray', style='italic')
    ax.text(gap_thresh * 1.6, ymax * 0.97, 'Sys.Overfit / O+U',
            ha='center', va='top', fontsize=7, color='gray', style='italic')

    if sc is not None:
        plt.colorbar(sc, ax=ax, label=f'Corr after GA', fraction=0.046, pad=0.04)

    ax.set_xlabel(f'Train Gap  (train_corr − {cv_tag} corr)', fontsize=10)
    ax.set_ylabel('train_corr_std  (per-fold instability)', fontsize=10)
    ax.set_title('Overfitting Diagnostic\n(color = val corr, shape = category)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.7)
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.set_xlim(left=min(gap.min() - 0.02, -0.05))
    ax.set_ylim(bottom=-0.002)


def generate_summary_plots(summary_df, out_dir, cv_method, signed_improvement=False):
    """
    Generate cross-gene summary plots and enrich summary_df with improvement/impact categories.

    Args:
        summary_df (pd.DataFrame): one row per processed gene, with columns including
            'Correlation before GA', 'Correlation after GA', 'MAE_PrediXcan_Real', 'MAE_GA_Real',
            'train_corr', 'train_corr_std', 'gene_id'.
        out_dir (str): directory to save PNG/CSV outputs into.
        cv_method (str): 'loocv' | 'kfold' | 'repeated_kfold', used for plot labeling.
        signed_improvement (bool): if True (V3 mode), Impact_Category bins the signed Improvement
            (After - Before), distinguishing worsened genes from improved ones; if False, bins the
            unsigned Improvement_Magnitude instead.

    Returns:
        pd.DataFrame: summary_df with added columns 'Is_Improved', 'Improvement',
        'Improvement_Magnitude', 'Impact_Category'.

    Notes:
        Writes these files into out_dir:
          gene_wise_mae_comparison.png — histogram of MAE_PrediXcan_Real - MAE_GA_Real, binned,
            showing how many genes improved/were neutral/worsened by MAE.
          impact_category_summary.csv — per-Impact_Category gene counts, average improvement, and
            percentage of total.
          overall_improvement_summary.png — 2x2 grid: (0,0) train_corr vs before-GA corr scatter,
            (0,1) after-GA vs before-GA corr scatter (both color-coded improved/unchanged/worsened
            using a +/-0.1 margin), (1,0) binned bar chart of train_corr vs before-GA corr
            distribution with Fisher-exact + multiple-testing-corrected significance stars,
            (1,1) same binned bar chart for after-GA vs before-GA.
          overfitting_diagnostic.png — the annotated 4-quadrant plot from _plot_overfitting_diagnostic.
        The nested add_significance_asterisks() helper runs a per-bin 2x2 Fisher exact test
        (gene-in-bin vs gene-not-in-bin, comparing the two methods being contrasted) and corrects
        the resulting p-values across bins using either Bonferroni or Benjamini-Hochberg FDR
        (method='fdr_bh' by default) before drawing significance brackets (*/**/***) over bars
        whose corrected p-value is < 0.05/0.01/0.001 respectively.
    """
    # -- 1. MAE improvement distribution --
    if 'MAE_PrediXcan_Real' in summary_df.columns and 'MAE_GA_Real' in summary_df.columns:
        plt.figure(figsize=(15, 7))
        plot_df = summary_df.copy()
        plot_df['MAE_Diff'] = plot_df['MAE_PrediXcan_Real'] - plot_df['MAE_GA_Real']

        bins   = [-np.inf, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, np.inf]
        labels = [
            '< -0.5', '-0.5~-0.4', '-0.4~-0.3', '-0.3~-0.2', '-0.2~-0.1',
            '-0.1~0.1 (Neutral)', '0.1~0.2', '0.2~0.3', '0.3~0.4', '0.4~0.5', '> 0.5'
        ]
        plot_df['MAE_Diff_Bin'] = pd.cut(plot_df['MAE_Diff'], bins=bins, labels=labels, right=False)
        dist_counts = plot_df['MAE_Diff_Bin'].value_counts().reindex(labels, fill_value=0).reset_index()
        dist_counts.columns = ['MAE_Difference_Range', 'Gene_Count']

        palette_colors = [
            'maroon', 'darkred', 'firebrick', 'indianred', 'lightcoral',
            'lightgray',
            'lightgreen', 'mediumseagreen', 'seagreen', 'darkgreen', 'darkolivegreen'
        ]
        ax = sns.barplot(data=dist_counts, x='MAE_Difference_Range', y='Gene_Count',
                         palette=palette_colors, edgecolor='black', alpha=0.9)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=11, fontweight='bold',
                            xytext=(0, 3), textcoords='offset points')

        n_improved  = sum(plot_df['MAE_Diff'] >= 0.1)
        n_neutral   = sum((plot_df['MAE_Diff'] >= -0.1) & (plot_df['MAE_Diff'] < 0.1))
        n_worsened  = sum(plot_df['MAE_Diff'] < -0.1)
        avg_pre_val = plot_df['MAE_PrediXcan_Real'].mean()
        avg_ga_val  = plot_df['MAE_GA_Real'].mean()

        plt.title(
            f"Distribution of MAE Improvement\nAvg MAE: {avg_pre_val:.3f} -> {avg_ga_val:.3f}",
            fontsize=15, fontweight='bold'
        )
        plt.xlabel("MAE Difference Range (Positive = GA has lower MAE)", fontsize=13)
        plt.ylabel("Number of Genes", fontsize=13)
        plt.xticks(rotation=20)
        stats_text = (
            f"Improved (Diff >= 0.1): {n_improved} genes\n"
            f"Neutral (-0.1~0.1): {n_neutral} genes\n"
            f"Worsened (Diff < -0.1): {n_worsened} genes"
        )
        plt.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                 verticalalignment='top', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        max_count = dist_counts['Gene_Count'].max()
        ax.set_ylim(0, max_count * 1.15)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "gene_wise_mae_comparison.png"), dpi=300)
        plt.close()
        logger.info("Saved Gene-wise MAE distribution bar chart")

    # -- 2. Correlation overview (2x2) --
    if 'Correlation after GA' not in summary_df.columns or 'Correlation before GA' not in summary_df.columns:
        return summary_df

    summary_df = summary_df.copy()
    summary_df['Is_Improved'] = (
        summary_df['Correlation after GA'] > summary_df['Correlation before GA']
    )
    summary_df['Improvement'] = (
        summary_df['Correlation after GA'] - summary_df['Correlation before GA']
    )
    summary_df['Improvement_Magnitude'] = summary_df['Improvement'].abs()

    if signed_improvement:
        imp_bins   = [-np.inf, -0.1, 0.1, 0.3, 0.6, np.inf]
        imp_labels = [
            'Worsened (< -0.1)', 'No Impact (-0.1~0.1)',
            'Slight Improvement (0.1~0.3)', 'Moderate Improvement (0.3~0.6)',
            'Significant Improvement (> 0.6)'
        ]
        summary_df['Impact_Category'] = pd.cut(
            summary_df['Improvement'], bins=imp_bins, labels=imp_labels, right=False
        )
    else:
        imp_bins   = [0.0, 0.1, 0.3, 0.6, 2.0]
        imp_labels = [
            'No Impact (0-0.1)', 'Slight Impact (0.1-0.3)',
            'Moderate Impact (0.3-0.6)', 'Significant Impact (>0.6)'
        ]
        summary_df['Impact_Category'] = pd.cut(
            summary_df['Improvement_Magnitude'], bins=imp_bins, labels=imp_labels, right=False
        )

    impact_summary_df = summary_df.groupby('Impact_Category', observed=False).agg(
        Gene_Count=('gene_id', 'count'),
        Avg_Improvement=('Improvement', 'mean'),
        Improved_Count=('Is_Improved', 'sum')
    ).reset_index()
    total_genes = len(summary_df)
    impact_summary_df['Percentage'] = (impact_summary_df['Gene_Count'] / total_genes * 100).round(2)
    impact_csv_path = os.path.join(out_dir, "impact_category_summary.csv")
    impact_summary_df.to_csv(impact_csv_path, index=False)
    logger.info(f"  Impact Category Summary saved to {impact_csv_path}")

    corr_bins   = [-np.inf, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, np.inf]
    corr_labels = ['<-0.6', '-0.6~-0.3', '-0.3~-0.1', '-0.1~0.1', '0.1~0.3', '0.3~0.6', '>0.6']

    def make_bar_data(col_a, col_b, label_a, label_b):
        """
        Build a long-format bar-chart DataFrame comparing two correlation columns' binned distributions.

        Args:
            col_a, col_b (str): column names in the enclosing summary_df to compare (e.g.
                'Correlation before GA' vs 'train_corr').
            label_a, label_b (str): display labels for each method/column in the output 'Method' field.

        Returns:
            pd.DataFrame: columns ['Correlation_Range', 'Count', 'Method'], concatenated for both
            columns, using the shared corr_bins/corr_labels binning.
        """
        ca = pd.cut(summary_df[col_a], bins=corr_bins, labels=corr_labels, right=False) \
               .value_counts().reindex(corr_labels, fill_value=0).reset_index()
        ca.columns = ['Correlation_Range', 'Count']
        ca['Method'] = label_a
        cb = pd.cut(summary_df[col_b], bins=corr_bins, labels=corr_labels, right=False) \
               .value_counts().reindex(corr_labels, fill_value=0).reset_index()
        cb.columns = ['Correlation_Range', 'Count']
        cb['Method'] = label_b
        return pd.concat([ca, cb], ignore_index=True)

    def make_scatter_lims(*cols):
        """
        Compute shared symmetric axis limits covering all given summary_df columns' value ranges.

        Args:
            *cols (str): column names in the enclosing summary_df to include in the range.

        Returns:
            list[float, float]: [min - 0.05, max + 0.05] across all provided columns' non-NaN
            values, or [-1.0, 1.0] if no valid values are found.
        """
        vals = []
        for col in cols:
            if col in summary_df.columns:
                arr = summary_df[col].dropna().values
                if arr.size > 0:
                    vals.append(arr)
        if not vals:
            return [-1.0, 1.0]
        merged = np.concatenate(vals)
        return [np.min(merged) - 0.05, np.max(merged) + 0.05]

    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()

    # Scatter: Before vs Train (in-sample) — top-left
    ax = axes[0]

    if 'train_corr' in summary_df.columns:
        # Margin defining "unchanged" status (+/- 0.1)
        margin = 0.1

        tc_diff = summary_df['train_corr'] - summary_df['Correlation before GA']

        t_improved = summary_df[tc_diff > margin]
        t_worsened = summary_df[tc_diff < -margin]
        t_unchanged = summary_df[tc_diff.abs() <= margin]

        if not t_worsened.empty:
            ax.scatter(t_worsened['Correlation before GA'], t_worsened['train_corr'],
                       alpha=0.6, color='red', edgecolors='darkred', linewidth=0.5,
                       label=f'Worsened ({len(t_worsened)})', s=35)

        if not t_unchanged.empty:
            ax.scatter(t_unchanged['Correlation before GA'], t_unchanged['train_corr'],
                       alpha=0.6, color='gray', edgecolors='dimgray', linewidth=0.5,
                       label=f'Unchanged ({len(t_unchanged)})', s=35)

        if not t_improved.empty:
            ax.scatter(t_improved['Correlation before GA'], t_improved['train_corr'],
                       alpha=0.6, color='green', edgecolors='darkgreen', linewidth=0.5,
                       label=f'Improved ({len(t_improved)})', s=35)

        tc_lims = make_scatter_lims('Correlation before GA', 'train_corr')

        ax.plot(tc_lims, tc_lims, 'k--', alpha=0.3, linewidth=2, label='y=x', zorder=0)

        ax.plot(tc_lims, [x + margin for x in tc_lims], 'k:', alpha=0.2, zorder=0)
        ax.plot(tc_lims, [x - margin for x in tc_lims], 'k:', alpha=0.2, zorder=0)

        ax.set_xlim(tc_lims); ax.set_ylim(tc_lims)
        ax.set_title("Correlation: Before vs After GA (Train)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Correlation Before GA", fontsize=11)
        ax.set_ylabel("Correlation After GA", fontsize=11)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'train_corr not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title("Correlation: Before vs After GA (Train)", fontsize=13)

    # Scatter: Before vs After (validation) — top-right
    ax = axes[1]

    margin = 0.1

    diff = summary_df['Correlation after GA'] - summary_df['Correlation before GA']

    improved = summary_df[diff > margin]
    worsened = summary_df[diff < -margin]
    unchanged = summary_df[diff.abs() <= margin]

    if not worsened.empty:
        ax.scatter(worsened['Correlation before GA'], worsened['Correlation after GA'],
                   alpha=0.6, color='red', edgecolors='darkred', linewidth=0.5,
                   label=f'Worsened ({len(worsened)})', s=35)

    if not unchanged.empty:
        ax.scatter(unchanged['Correlation before GA'], unchanged['Correlation after GA'],
                   alpha=0.6, color='gray', edgecolors='dimgray', linewidth=0.5,
                   label=f'Unchanged ({len(unchanged)})', s=35)

    if not improved.empty:
        ax.scatter(improved['Correlation before GA'], improved['Correlation after GA'],
                   alpha=0.6, color='green', edgecolors='darkgreen', linewidth=0.5,
                   label=f'Improved ({len(improved)})', s=35)

    lims = make_scatter_lims('Correlation before GA', 'Correlation after GA')

    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=2, label='y=x', zorder=0)

    ax.plot(lims, [x + margin for x in lims], 'k:', alpha=0.2, zorder=0)
    ax.plot(lims, [x - margin for x in lims], 'k:', alpha=0.2, zorder=0)

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title("Correlation: Before vs After GA (Valid)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Correlation Before GA", fontsize=11)
    ax.set_ylabel("Correlation After GA", fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)

    def add_significance_asterisks(ax, bar_data, method='fdr_bh'):
        """
        Annotate a grouped bar plot with significance stars, correcting for multiple comparisons.

        Args:
            ax (matplotlib.axes.Axes): axes containing the grouped bar plot (bars grouped in pairs
                per correlation-range bin, ordered by bin then by method, matching make_bar_data's output).
            bar_data (pd.DataFrame): long-format data used to draw the bars, columns
                ['Correlation_Range', 'Count', 'Method'] as produced by make_bar_data.
            method (str): multiple-testing correction method passed to
                statsmodels.stats.multitest.multipletests ('fdr_bh' = Benjamini-Hochberg, more
                lenient; 'bonferroni' = stricter).

        Returns:
            None. Draws bracket+asterisk annotations directly onto `ax` for bins whose corrected
            p-value is significant, and expands the y-axis if a bracket would otherwise be clipped.

        Notes:
            For each correlation-range bin, builds a 2x2 contingency table
            [[count_method1, total_n - count_method1], [count_method2, total_n - count_method2]]
            and runs Fisher's exact test; the resulting per-bin p-values are corrected together
            (across all bins) via `multipletests` before any star is drawn. Significance thresholds:
            p < 0.001 -> '***', p < 0.01 -> '**', p < 0.05 -> '*'; p >= 0.05 draws nothing.
        """
        ranges   = bar_data['Correlation_Range'].unique()
        methods  = bar_data['Method'].unique()
        total_n  = bar_data['Count'].sum() // 2
        patches  = ax.patches
        n_ranges = len(ranges)

        if len(methods) < 2 or total_n == 0:
            return

        m1, m2 = methods[0], methods[1]
        offset  = bar_data['Count'].max() * 0.05
        y_max_ax = ax.get_ylim()[1]
        y_min    = ax.get_ylim()[0]

        # Step 1: collect all p-values first
        pvals = []
        valid_indices = []

        for i, r in enumerate(ranges):
            c1 = bar_data[(bar_data['Correlation_Range'] == r) & (bar_data['Method'] == m1)]['Count'].values
            c2 = bar_data[(bar_data['Correlation_Range'] == r) & (bar_data['Method'] == m2)]['Count'].values
            c1 = int(c1[0]) if len(c1) > 0 else 0
            c2 = int(c2[0]) if len(c2) > 0 else 0

            if c1 == 0 and c2 == 0:
                pvals.append(1.0)        # No data, assign maximal p-value
                valid_indices.append(i)
                continue

            table = [[c1, total_n - c1], [c2, total_n - c2]]
            _, p  = fisher_exact(table)
            pvals.append(p)
            valid_indices.append(i)

        # Step 2: correct for multiple comparisons in one pass
        if len(pvals) > 0:
            _, pvals_corrected, _, _ = multipletests(pvals, method=method)
        else:
            return

        # Step 3: use corrected p-values to decide which bins to annotate
        for idx, (i, p_corr) in enumerate(zip(valid_indices, pvals_corrected)):
            if p_corr >= 0.05:
                continue

            if p_corr < 0.001:   text = '***'
            elif p_corr < 0.01:  text = '**'
            else:                 text = '*'

            p1 = patches[i]
            p2 = patches[i + n_ranges]
            x1 = p1.get_x() + p1.get_width() / 2
            x2 = p2.get_x() + p2.get_width() / 2
            bar_y_max = max(p1.get_height(), p2.get_height())
            h_base = bar_y_max + (offset * 0.8)
            h_top  = h_base + (offset * 0.2)

            ax.plot([x1, x1, x2, x2], [h_base, h_top, h_top, h_base],
                    lw=1.2, c='black', alpha=0.7)
            ax.text((x1 + x2) / 2, h_top, text,
                    ha='center', va='bottom', color='black',
                    fontsize=12, fontweight='bold')

            if h_top + offset > y_max_ax:
                y_max_ax = h_top + offset * 1.5
                ax.set_ylim(y_min, y_max_ax)

    # Bar: Before vs Train
    ax = axes[2]
    if 'train_corr' in summary_df.columns:
        bar_data_train = make_bar_data(
            'Correlation before GA', 'train_corr',
            'PrediXcan (Before)', 'Train Corr (In-sample)'
        )
        sns.barplot(data=bar_data_train, x='Correlation_Range', y='Count',
                    hue='Method', palette=['blue', 'red'], ax=ax, alpha=0.85)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, fontsize=9)

        add_significance_asterisks(ax, bar_data_train)

        ax.set_title("Gene Distribution: Corr Before and After GA (Train)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Correlation Range", fontsize=11)
        ax.set_ylabel("Number of Genes", fontsize=11)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'train_corr not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title("Gene Distribution: Corr Before and After GA (Train)", fontsize=13)

    # Bar: Before vs After
    ax = axes[3]
    bar_data_after = make_bar_data(
        'Correlation before GA', 'Correlation after GA',
        'PrediXcan (Before)', 'GA Optimized (After)'
    )
    sns.barplot(data=bar_data_after, x='Correlation_Range', y='Count',
                hue='Method', palette=['blue', 'red'], ax=ax, alpha=0.85)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3, fontsize=9)

    add_significance_asterisks(ax, bar_data_after)

    ax.set_title("Gene Distribution: Corr Before and After GA (Valid)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Correlation Range", fontsize=11)
    ax.set_ylabel("Number of Genes", fontsize=11)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overall_improvement_summary.png"), dpi=300)
    plt.close()
    logger.info("Saved improvement plots (2x3 layout)")

    # Standalone large figure: overfitting_diagnostic.png (with gene_id annotations)
    _cv_tag_diag = {'loocv': 'LOOCV', 'kfold': 'KFold', 'repeated_kfold': 'RepKFold'}.get(cv_method, 'CV')
    fig_diag, ax_diag = plt.subplots(figsize=(14, 9))
    _plot_overfitting_diagnostic(ax_diag, summary_df, _cv_tag_diag, annotate=True)
    fig_diag.tight_layout()
    fig_diag.savefig(os.path.join(out_dir, "overfitting_diagnostic.png"), dpi=300)
    plt.close(fig_diag)
    logger.info("Saved overfitting diagnostic plot")

    return summary_df


def write_summary_output(summary_df, out_dir, overall_stats, processed_count,
                         failed_genes, total_time, extra_metrics=None):
    """
    Log and persist the final run-level summary (metrics text file + final_summary.csv).

    Args:
        summary_df (pd.DataFrame): per-gene results, enriched by generate_summary_plots.
        out_dir (str): directory to write output files into.
        overall_stats (dict): accumulator dict with keys 'total_times', 'all_actuals',
            'all_predictions_ga', 'weight_stabilities' (lists collected across all genes in main()).
        processed_count (int): total number of genes attempted.
        failed_genes (list): gene IDs that failed processing.
        total_time (float): total wall-clock seconds for the whole run.
        extra_metrics (dict, optional): additional key: value pairs appended to the metrics text file.

    Returns:
        None. Logs a summary block and writes:
          overall_performance_metrics.txt — headline run statistics.
          final_summary.csv — per-gene summary table sorted by Improvement (descending).

    Notes:
        "Unreliable genes" are defined as those with Correlation after GA < 0.1 (n_unreliable /
        n_near_zero), a looser and more informative threshold than requiring statistical
        significance alone; negative-correlation genes are reported as a sub-count of these.
        Global Corr aggregates all genes' actual/predicted values into one long vector before
        computing a single pooled correlation — a different (and generally higher, due to
        between-gene variance) statistic than the mean of per-gene correlations.
        Avg Weight Std (weight_stabilities average) summarizes how reproducible GA solutions were
        across runs/folds; lower is better (more consistent optimization).
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXECUTION SUMMARY")
    logger.info(f"  Total Time      : {str(timedelta(seconds=int(total_time)))}")
    logger.info(f"  Genes Processed : {processed_count}")
    logger.info(f"  Successful      : {processed_count - len(failed_genes)}")
    logger.info(f"  Failed          : {len(failed_genes)}")

    if failed_genes:
        logger.warning(f"  Failed Genes List: {failed_genes[:10]} ...")

    if not summary_df.empty and 'Correlation after GA' in summary_df.columns:
        corr_after = summary_df['Correlation after GA'].dropna()
        n_total_genes = len(corr_after)

        n_negative   = int((corr_after < 0.0).sum())
        n_near_zero  = int((corr_after < 0.1).sum())
        n_unreliable = n_near_zero   # Primary indicator: corr < 0.1

        if 'r_pval_after' in summary_df.columns:
            n_insig = int((summary_df['r_pval_after'].dropna() >= 0.05).sum())
            logger.info(
                f"  Unreliable Genes: {n_unreliable} / {n_total_genes} "
                f"({n_unreliable/n_total_genes:.1%}) — Corr after GA < 0.1"
            )
            logger.info(
                f"    ├ Negative Corr : {n_negative} genes")
            logger.info(
                f"    └ p >= 0.05     : {n_insig} genes (not statistically significant)"
            )
        else:
            logger.info(
                f"  Unreliable Genes: {n_unreliable} / {n_total_genes} "
                f"({n_unreliable/n_total_genes:.1%}) — Corr after GA < 0.1"
            )
            logger.info(f"    ├ Negative Corr : {n_negative} genes")

    if len(overall_stats['total_times']) > 0:
        avg_time_per_gene = np.mean(overall_stats['total_times'])
        actuals     = np.array(overall_stats['all_actuals'],      dtype=np.float64)
        predictions = np.array(overall_stats['all_predictions_ga'], dtype=np.float64)

        valid_mask  = np.isfinite(actuals) & np.isfinite(predictions)
        n_valid     = valid_mask.sum()

        if n_valid > 2 and np.std(actuals[valid_mask]) > 1e-9 and np.std(predictions[valid_mask]) > 1e-9:
            global_corr = np.corrcoef(actuals[valid_mask], predictions[valid_mask])[0, 1]
        else:
            global_corr = float('nan')

        logger.info(
            f"  Global Corr     : {global_corr:.4f} "
            f"(Aggregated across {processed_count} genes, {n_valid} valid samples)"
        )
        avg_stability = np.mean(overall_stats['weight_stabilities'])
        logger.info(f"  Avg Time/Gene   : {avg_time_per_gene:.2f}s")
        logger.info(f"  Global Corr     : {global_corr:.4f} (Aggregated across {processed_count} genes)")
        logger.info(f"  Avg Weight Std  : {avg_stability:.4f} (Stability Score)")

        with open(os.path.join(out_dir, "overall_performance_metrics.txt"), "w") as f:
            f.write(f"Total Genes: {processed_count}\n")
            f.write(f"Avg Time Per Gene: {avg_time_per_gene:.4f}s\n")
            f.write(f"Unreliable Genes (Corr<0.1): {n_unreliable} / {n_total_genes}\n")
            f.write(f"Negative Corr Genes: {n_negative}\n")
            f.write(f"Global Correlation: {global_corr:.4f}\n")
            f.write(f"Avg Weight Stability (Std): {avg_stability:.4f}\n")
            f.write("Note: Lower Weight Std means the GA converges to similar weights across runs.\n")
            if extra_metrics:
                for k, v in extra_metrics.items():
                    f.write(f"{k}: {v}\n")

    if 'Improvement' in summary_df.columns:
        desired_columns = [
            'gene_id', 'weight_stability', 'weight_stability_penalty', 'n_snps',
            'Correlation before GA', 'Correlation after GA',
            'train_corr', 'train_corr_std', 'baseline_corr', 'effective_runs',
            'MAE_PrediXcan_Real', 'MAE_GA_Real',
            'Improvement', 'Impact_Category',  'compute_time'
        ]
        final_cols = [c for c in desired_columns if c in summary_df.columns]
        summary_df_clean = summary_df[final_cols].sort_values(by='Improvement', ascending=False)
        summary_path = os.path.join(out_dir, "final_summary.csv")
        summary_df_clean.to_csv(summary_path, index=False)
        logger.info(f"  Summary saved to {summary_path}")

    logger.info("=" * 70)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    """
    Entry point: load config/data, run the GA optimization pipeline over all genes, write outputs.

    Args:
        None (reads CLI args via get_args() and the JSON config file it points to).

    Returns:
        None. Side effects: writes per-gene result directories, summary CSVs/plots, and an
        overall_performance_metrics.txt into the configured output directory.

    Notes:
        High-level flow:
          1. Parse CLI args and load config.json (paths to genotype/expression/weight/annotation data).
          2. reading_file_path -> load raw genotype (dask, lazy), predicted/real expression, allele
             frequencies, annotation, DB path.
          3. PrediXcanResultDealing reshapes raw PrediXcan output; get_db_info loads the weight DB;
             db_allele_freq_filtered drops untransferable SNPs; data_intersection restricts
             everything to common genes/samples.
          4. Optional test_mode: select_test_genes stratified-samples a subset of genes for a
             quick run instead of the full gene list.
          5. build_ga_params converts CLI args into the GA hyperparameter dict used throughout.
          6. Genotype for all needed rsids is loaded fully into RAM (not kept lazy) since it is
             reused across every gene's task.
          7. task_generator lazily yields one (gene_id, subset_genotype, ...) tuple per gene;
             process_single_gene_gpu is called synchronously per gene in the main loop (GPU work is
             inherently serialized per gene, but plotting is offloaded to a ThreadPoolExecutor so
             it doesn't block the next gene's GPU computation).
          8. Successful results are accumulated into `results`/`overall_stats`; failed genes are
             logged and tracked separately; GPU memory pools are freed after every gene and `gc.collect()`
             is run every 100 genes to bound peak memory.
          9. After the loop (or on KeyboardInterrupt), generate_summary_plots and
             write_summary_output produce the final cross-gene summary artifacts.
        KeyboardInterrupt during the main loop is caught so that partial results are still saved
        and plots are still flushed via the `finally` block.
    """
    # Parse arguments
    args = get_args()
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return

    # --- Load Data ---
    try:
        with open(args.config, 'r') as f:
            config_paths = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON config file: {e}")
        return

    # --- Logging Configuration Summary ---
    logger.info("")
    # --- Log GPU Config ---
    logger.info("=" * 70)
    logger.info(" " * 20 + "V3 CONFIGURATION SUMMARY")
    logger.info("=" * 70)
    logger.info("GA Parameters:")
    logger.info(f"  Phase 1 Iterations    : {args.GA_Phase1_iterations}")
    logger.info(f"  Phase 2 Iterations    : {args.GA_Phase2_iterations}")
    logger.info(f"  Phase 3 Iterations    : {args.GA_Phase3_iterations}")
    logger.info(f"  Population Size       : {args.GA_population_sizes}")
    logger.info(f"  Runs per Fold         : {args.Numbers_of_GA_runs}")
    logger.info(f"  CV Method             : {args.cv_method}")
    logger.info("V3 New Parameters:")
    logger.info(f"  Sign Flip Mode        : {'Enabled (negative scaling allowed for L/M-tier)' if args.allow_sign_flip else 'Disabled'}")
    logger.info(f"  Sign Flip Threshold   : {args.sign_flip_threshold}  (enabled only when baseline corr is below this)")
    logger.info("")
    logger.info("Data Paths:")
    for key, value in config_paths.items():
        short_path = value if len(value) < 50 else "..." + value[-47:]
        logger.info(f"  {key:20s}: {short_path}")
    logger.info("=" * 70)

    # Ensure all GPU memory is released
    try:
        genotype, predict_expr_raw, real_expr_raw, allele_freq, annotation, db_path, out_dir = \
            reading_file_path(config_paths)
        logger.info("  Raw data loaded successfully")
    except Exception as e:
        logger.error(f"  Failed to load raw data: {e}")
        logger.error(traceback.format_exc())
        return

    predict_expr = PrediXcanResultDealing(predict_expr_raw)
    real_expr    = real_expr_raw

    weight_df, gene_info_df   = get_db_info(db_path)
    db_weight_filtered        = db_allele_freq_filtered(weight_df, allele_freq)
    # Data Intersection
    predict_expr_f, db_weight_f, real_expr_f = data_intersection(
        predict_expr, db_weight_filtered, real_expr
    )
    common_genes = predict_expr_f.index.tolist()

    # === Test Mode ===
    if args.test_mode:
        common_genes, gene_bin_map = select_test_genes(
            common_genes, real_expr_f, args.test_count, args.test_seed
        )
    else:
        gene_bin_map = {}

    n_genes = len(common_genes)
    logger.info(f"  Genes to process: {n_genes}")

    logger.info(f"  CUDA Device: {cp.cuda.runtime.getDevice()}")

    # --- Prepare Parameters ---
    start_time = time.time()

    ga_params = build_ga_params(args)

    # --- Load Genotype to RAM ---
    logger.info("Loading genotype data to RAM...")
    all_needed_rsids = db_weight_f['rsid'].unique().tolist()
    genotype_all_pd  = genotype[genotype['ID'].isin(all_needed_rsids)].compute()
    genotype_all_pd  = genotype_all_pd.set_index('ID')
    logger.info(f"  Genotype indexed: {len(genotype_all_pd):,} SNPs loaded")

    # --- Gene to SNP Mapping ---
    gene_to_rsids = db_weight_f.groupby('gene')['rsid'].apply(list).to_dict()

    # --- Task Generator ---
    gen = task_generator(
        common_genes, gene_to_rsids, genotype_all_pd,
        db_weight_f, real_expr_f, predict_expr_f, out_dir, ga_params
    )

    # --- Execution Loop ---
    num_plot_threads = args.plot_thread
    plot_executor    = ThreadPoolExecutor(max_workers=num_plot_threads)

    results          = []
    performance_stats = {'compute_times': [], 'total_times': [], 'improvements': []}
    failed_genes     = []
    processed_count  = 0
    total_expected   = len(common_genes)

    overall_stats = {
        'total_times':          [],
        'all_predixcan':        [],
        'all_actuals':          [],
        'all_predictions_ga':   [],
        'weight_stabilities':   []
    }

    logger.info(f"Starting processing of {total_expected} genes...")

    try:
        for task_args in gen:
            processed_count += 1
            res    = process_single_gene_gpu(*task_args)
            status = res.get('status', 'Unknown')

            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass


            if 'Success' in status:
                # Collect performance stats
                if 'compute_time' in res: performance_stats['compute_times'].append(res['compute_time'])
                if 'time' in res:         performance_stats['total_times'].append(res['time'])
                # Overall stats for final summary
                overall_stats['total_times'].append(res['time'])
                overall_stats['all_predixcan'].extend(res['raw_predixcan'])
                overall_stats['all_actuals'].extend(res['raw_actuals'])
                overall_stats['all_predictions_ga'].extend(res['raw_predictions'])
                overall_stats['weight_stabilities'].append(res['weight_stability'])

                # Plotting
                if 'plot_payload' in res:
                    plot_executor.submit(save_plots_async, res.pop('plot_payload'))

                res.pop('raw_predixcan', None)
                res.pop('raw_actuals', None)
                res.pop('raw_predictions', None)

            elif 'Failed' in status:
                failed_genes.append(res['gene_id'])

            results.append(res)

            # Logging
            elapsed   = time.time() - start_time
            avg_time  = elapsed / processed_count
            eta_str   = str(timedelta(seconds=int((total_expected - processed_count) * avg_time)))
            before_str = f"{res.get('Correlation before GA', 0.0):.3f}" if 'Correlation before GA' in res else "N/A"
            after_str  = f"{res.get('Correlation after GA', 0.0):.3f}" if 'Correlation after GA' in res else "N/A"
            train_str  = f"{res.get('train_corr', 0.0):.3f}" if 'train_corr' in res else "N/A"

            bin_str = ""
            if args.test_mode:
                b_idx   = gene_bin_map.get(res['gene_id'], "?")
                bin_str = f"Bin: {b_idx} | "

            logger.info(
                f"[{processed_count}/{total_expected}] "
                f"{res['gene_id']:15s} | "
                f"Before: {before_str} | "
                f"After: {after_str} | "
                f"Train: {train_str} | "
                f"{bin_str}"
                f"Time: {res.get('time',0):5.1f}s | "
                f"ETA: {eta_str}"
            )

            # Memory Management
            if processed_count % 100 == 0:
                gc.collect()

    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user.")
    finally:
        logger.info("Waiting for remaining plots...")
        plot_executor.shutdown(wait=True)

    # --- Final Output ---
    total_time = time.time() - start_time
    summary_df = pd.DataFrame(results)
    summary_df = generate_summary_plots(summary_df, out_dir, args.cv_method,
                                        signed_improvement=True)
    write_summary_output(
        summary_df, out_dir, overall_stats, processed_count, failed_genes, total_time,
        extra_metrics={'Sign Flip Mode': 'Enabled' if args.allow_sign_flip else 'Disabled'}
    )

    # Ensure all GPU memory is released
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


if __name__ == "__main__":
    main()