import os
import time
import logging
import numpy as np
import pandas as pd
from numba import cuda
import cupy as cp
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from GA import (
    reading_file_path,
    get_db_info,
    db_allele_freq_filtered,
    Weight_GA_Cupy,
    apply_tas_filter_cpu,
    logger
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [SIMULATION] %(message)s')

def simulation_task_generator(n_genes, n_samples, n_snps_min, n_snps_max, out_dir, ga_params,
                              db_weight_freq_table,
                              sim_ratios={'identical': 0.5, 'small': 0.3, 'large': 0.2},
                              seed_base=42, l_effect_ratio=0.3, m_effect_ratio=0.7,
                              out_range=False, outlier_scaling=2.0):
    """Generate synthetic per-gene GA-optimization tasks by sampling real EUR/EAS allele
    frequencies and weights from a reference DB table, split into four AF-difference tiers.

    Args:
        n_genes (int): Number of synthetic genes to yield.
        n_samples (int): Number of synthetic individuals to simulate genotypes for.
        n_snps_min (int): Lower bound of the per-gene SNP count range.
        n_snps_max (int): Upper bound of the per-gene SNP count range.
        out_dir (str): Output directory to attach to each yielded task dict (passed through unused here).
        ga_params (dict): GA / tier configuration, including s/m/l tier low-high bounds and thresholds.
        db_weight_freq_table (pd.DataFrame): Reference table with columns 'EUR_AF', 'EAS_AF', 'weight',
            used as the sampling pool for all four AF-difference tiers.
        sim_ratios (dict): Probabilities for assigning each SNP to 'identical'/'small'/'medium'/'large'
            tiers, plus optional 'sign_flip_prob' applied to M/L-tier effect SNPs.
        seed_base (int): Base seed for the local numpy Generator (rng), ensures reproducibility.
        l_effect_ratio (float): Fraction of L-tier SNPs treated as true effect SNPs (rest are noise).
        m_effect_ratio (float): Fraction of M-tier SNPs treated as true effect SNPs (rest are noise).
        out_range (bool): If True, widen the L-tier effect P-space range by outlier_scaling to create outliers.
        outlier_scaling (float): Multiplicative factor applied to L-tier bounds when out_range is True.

    Returns:
        Generator[dict]: Yields one dict per gene with keys: 'gene_id', 'G' (genotype matrix,
            n_samples x n_snps), 'y_true_eas' (z-scored EAS expression), 'w_start' (EUR/biased
            weights), 'w_true' (EAS/true weights), 'a_vec' (AF-similarity attenuation vector),
            'y_eur' (z-scored EUR-predicted expression), 'out_dir', 'ga_params', 'eur_af', 'eas_af',
            'snp_categories' (0=identical,1=small,2=medium,3=large), 'is_effect' (bool mask).

    Notes:
        - AF-difference tiers: 'identical' (diff=0, EAS_AF copied from EUR_AF), 'small'
          (diff <= s_af_diff_max, derived from s_tier_threshold), 'medium'
          (s_af_diff_max < diff <= m_af_diff_max), 'large' (diff > m_af_diff_max).
        - a_vec = clip(1 - af_diff, 0, 1) models the intuition that larger AF differences between
          populations should attenuate the transferable weight signal.
        - For S-tier, EAS weight = EUR weight * multiplier, where multiplier is drawn either from
          an amplification range [1.0, max_amp] or a shrink range [min_shrink, 1.0], each scaled by
          how close af_diff is to the 0.05 tier boundary.
        - For M/L-tier, SNPs are split into 'effect' (weight follows a bimodal P-space distribution
          scaled by a_vec) and 'noise' (EAS weight forced to ~0, i.e. GA should learn to shrink it).
          The bimodal P ranges are aligned with the GA's own search-space initialization tiers so
          that ground truth is representative of what the GA is expected to discover.
        - sign_flip_prob randomly negates effect M/L-tier EAS weights to simulate directionality
          reversal across populations.
        - Genotype dosages are sampled from Hardy-Weinberg proportions using EAS allele frequency:
          p(0)=(1-af)^2, p(1)=2*af*(1-af), p(2)=af^2.
        - Both y_true_eas and y_pred_eur are z-scored (mean 0, std 1) after computing G @ weights.
    """
    rng = np.random.default_rng(seed_base)

    s_low = ga_params.get('s_tier_low', 0.9)
    s_high = ga_params.get('s_tier_high', 1.1)

    m_low = ga_params.get('m_tier_low', 0.4)
    m_high = ga_params.get('m_tier_high', 3.0)

    l_low = ga_params.get('l_tier_low', 0.0)
    l_high = ga_params.get('l_tier_high', 5.0)

    # Pre-process the DB table into pools grouped by AF difference
    db_valid = db_weight_freq_table.dropna(subset=['EUR_AF', 'EAS_AF', 'weight']).copy()
    db_valid = db_valid[db_valid['weight'].abs() >= 1e-4]
    db_valid['af_diff'] = np.abs(db_valid['EUR_AF'] - db_valid['EAS_AF'])

    s_thresh = ga_params.get('s_tier_threshold', 0.95)
    m_thresh = ga_params.get('m_tier_threshold', 0.75)

    # Convert tier "similarity threshold" into an AF-difference cutoff
    s_af_diff_max = 1.0 - s_thresh   # 0.05
    m_af_diff_max = 1.0 - m_thresh   # 0.25

    pool_small  = db_valid[db_valid['af_diff'] <= s_af_diff_max]
    pool_medium = db_valid[(db_valid['af_diff'] > s_af_diff_max) &
                       (db_valid['af_diff'] <= m_af_diff_max)]
    pool_large  = db_valid[db_valid['af_diff'] > m_af_diff_max]

    ratio_identical  = sim_ratios.get('identical',  0.4)
    ratio_small      = sim_ratios.get('small',      0.3)
    ratio_medium     = sim_ratios.get('medium',     0.2)
    ratio_large      = sim_ratios.get('large',      0.1)
    sign_flip_prob   = sim_ratios.get('sign_flip_prob', 0.0)  # applied to M/L-tier effect SNPs

    for i in range(n_genes):
        gene_id = f"SIM_{i+1:03d}"

        # Draw total SNP count for this gene from a Beta distribution rescaled to [n_snps_min, n_snps_max]
        mean_snps = 25
        mu = (mean_snps - n_snps_min) / (n_snps_max - n_snps_min)

        # concentration controls spread (higher = tighter around mean)
        concentration = 5.0

        alpha = mu * concentration
        beta_param = (1 - mu) * concentration

        sample = rng.beta(alpha, beta_param)
        n_snps = int(round(n_snps_min + sample * (n_snps_max - n_snps_min)))

        # Assign SNPs to tier categories based on configured ratios
        categories = rng.choice(
            [0, 1, 2, 3],
            size=n_snps,
            p=[ratio_identical, ratio_small, ratio_medium, ratio_large]
        )

        mask_identical = (categories == 0)
        mask_small     = (categories == 1)
        mask_medium    = (categories == 2)
        mask_large     = (categories == 3)

        n_identical  = np.sum(mask_identical)
        n_small      = np.sum(mask_small)
        n_medium     = np.sum(mask_medium)
        n_large      = np.sum(mask_large)

        sim_eur_af = np.zeros(n_snps)
        sim_eas_af = np.zeros(n_snps)
        eur_w = np.zeros(n_snps, dtype=np.float32)
        eas_w = np.zeros(n_snps, dtype=np.float32)
        is_effect = np.ones(n_snps, dtype=bool)

        # --- Tier 1: Identical AF (diff = 0.0) ---
        if n_identical > 0:
            sampled = db_valid.sample(n=n_identical, replace=True, random_state=rng.bit_generator)
            sim_eur_af[mask_identical] = sampled['EUR_AF'].values
            sim_eas_af[mask_identical] = sampled['EUR_AF'].values
            eur_w[mask_identical] = sampled['weight'].values
            eas_w[mask_identical] = sampled['weight'].values

        # --- Tier 2: Small AF diff (0.0 - 0.05) ---
        if n_small > 0:
            if len(pool_small) > 0:
                sampled = pool_small.sample(n=n_small, replace=True, random_state=rng.bit_generator)
            else:
                # Fallback to entire DB if pool is empty
                sampled = db_valid.sample(n=n_small, replace=True, random_state=rng.bit_generator)

            sim_eur_af[mask_small] = sampled['EUR_AF'].values
            sim_eas_af[mask_small] = sampled['EAS_AF'].values
            eur_w[mask_small] = sampled['weight'].values

            # Map af_diff (0.0 ~ 0.05) to a scale_factor (0.0 ~ 1.0)
            af_diff_small = np.abs(sim_eur_af[mask_small] - sim_eas_af[mask_small])
            scale_factor = np.clip(af_diff_small / 0.05, 0.0, 1.0)

            # Max bounds for amplification and shrinkage, scaled by how large af_diff is
            max_amp = 1.0 + (s_high - 1.0) * scale_factor
            min_shrink = 1.0 - (1.0 - s_low) * scale_factor

            # 50% chance to amplify, 50% chance to shrink
            is_amp = rng.random(size=n_small) < 0.5
            multipliers = np.ones(n_small, dtype=np.float32)

            # Amplify (1.0 to max_amp)
            if np.any(is_amp):
                multipliers[is_amp] = rng.uniform(1.0, max_amp[is_amp])

            # Shrink (min_shrink to 1.0)
            if np.any(~is_amp):
                multipliers[~is_amp] = rng.uniform(min_shrink[~is_amp], 1.0)

            eas_w[mask_small] = eur_w[mask_small] * multipliers
        # --- Tier 3: Medium AF diff (0.05 ~ 0.25) ---
        if n_medium > 0:
            if len(pool_medium) > 0:
                sampled = pool_medium.sample(n=n_medium, replace=True, random_state=rng.bit_generator)
            else:
                sampled = db_valid.sample(n=n_medium, replace=True, random_state=rng.bit_generator)

            sim_eur_af[mask_medium] = sampled['EUR_AF'].values
            sim_eas_af[mask_medium] = sampled['EAS_AF'].values
            eur_w[mask_medium]      = sampled['weight'].values

            # M-tier: split into effect/noise (m_effect_ratio), then apply bimodal P-space to effect SNPs
            af_diff_med = np.abs(sim_eur_af[mask_medium] - sim_eas_af[mask_medium])
            a_vec_med   = np.clip(1.0 - af_diff_med, 0.0, 1.0)

            effect_mask_med = rng.random(size=n_medium) < m_effect_ratio
            is_effect[mask_medium] = effect_mask_med

            # Noise SNPs: eas_w forced to ~0
            eas_w[mask_medium] = 0.0

            # Effect SNPs: bimodal P in [m_low, s_low] ∪ [s_high, m_high]
            n_effect_med = int(effect_mask_med.sum())
            if n_effect_med > 0:
                effect_idx_in_med = np.where(effect_mask_med)[0]
                effect_global_med = np.where(mask_medium)[0][effect_idx_in_med]

                is_amp = rng.random(size=n_effect_med) < 0.5
                P_med  = np.empty(n_effect_med, dtype=np.float32)
                P_med[is_amp]  = rng.uniform(s_high, m_high, size=int(is_amp.sum()))    # [1.2, 3.0]
                P_med[~is_amp] = rng.uniform(m_low,  s_low,  size=int((~is_amp).sum())) # [0.3, 0.8]

                a_vec_effect_med = a_vec_med[effect_idx_in_med]
                eas_w[effect_global_med] = eur_w[effect_global_med] * (P_med * a_vec_effect_med)

                # Randomly negate effect M-tier SNPs
                if sign_flip_prob > 0.0:
                    flip_mask = rng.random(size=n_effect_med) < sign_flip_prob
                    eas_w[effect_global_med] = np.where(flip_mask,
                                                         -eas_w[effect_global_med],
                                                         eas_w[effect_global_med])
        # --- Tier 4: Large AF diff (diff > 0.25) ---
        if n_large > 0:
            if len(pool_large) > 0:
                sampled = pool_large.sample(n=n_large, replace=True, random_state=rng.bit_generator)
            else:
                sampled = db_valid.sample(n=n_large, replace=True, random_state=rng.bit_generator)

            sim_eur_af[mask_large] = sampled['EUR_AF'].values
            sim_eas_af[mask_large] = sampled['EAS_AF'].values
            eur_w[mask_large]      = sampled['weight'].values

            # L-tier: P-space bimodal aligned with GA init: [l_low, m_low] ∪ [m_high, l_high]
            af_diff_large = np.abs(sim_eur_af[mask_large] - sim_eas_af[mask_large])
            a_vec_large   = np.clip(1.0 - af_diff_large, 0.0, 1.0)

            effect_mask_large = rng.random(size=n_large) < l_effect_ratio
            is_effect[mask_large] = effect_mask_large

            # Noise SNPs: eas_w ≈ 0 (GA should shrink weight to near zero)
            eas_w[mask_large] = 0.0

            # Effect SNPs: bimodal P in [l_low, m_low] ∪ [m_high, l_high]
            n_effect_large = int(effect_mask_large.sum())
            if n_effect_large > 0:
                effect_idx_in_large = np.where(effect_mask_large)[0]  # relative to mask_large
                effect_global_idx   = np.where(mask_large)[0][effect_idx_in_large]

                is_amp_l = rng.random(size=n_effect_large) < 0.5
                P_large  = np.empty(n_effect_large, dtype=np.float32)
                if out_range:
                    # out_range mode: widen upper bound / shrink lower bound to create outliers
                    P_large[is_amp_l]  = rng.uniform(m_high,          l_high * outlier_scaling, size=int(is_amp_l.sum()))
                    P_large[~is_amp_l] = rng.uniform(l_low / outlier_scaling, m_low,             size=int((~is_amp_l).sum()))
                else:
                    P_large[is_amp_l]  = rng.uniform(m_high, l_high, size=int(is_amp_l.sum()))    # [3.0, 7.0]
                    P_large[~is_amp_l] = rng.uniform(l_low,  m_low,  size=int((~is_amp_l).sum())) # [1e-4, 0.3]

                a_vec_effect = a_vec_large[effect_idx_in_large]
                eas_w[effect_global_idx] = eur_w[effect_global_idx] * (P_large * a_vec_effect)

                # Randomly negate effect L-tier SNPs
                if sign_flip_prob > 0.0:
                    flip_l = rng.random(size=n_effect_large) < sign_flip_prob
                    eas_w[effect_global_idx] = np.where(flip_l,
                                                         -eas_w[effect_global_idx],
                                                         eas_w[effect_global_idx])

        # Recompute a_vec across all SNPs for GA output
        af_diff_all = np.abs(sim_eur_af - sim_eas_af)
        a_vec = np.clip(1.0 - af_diff_all, 0.0, 1.0)

        # Genotype matrix: sample dosages 0/1/2 per SNP from Hardy-Weinberg proportions
        G_matrix = np.zeros((n_samples, n_snps), dtype=np.float32)
        for k, af in enumerate(sim_eas_af):
            p = [(1-af)**2, 2*af*(1-af), af**2]
            G_matrix[:, k] = rng.choice([0, 1, 2], size=n_samples, p=p)

        # EAS Expression = Genotype * True_Weights, then z-scored
        y_true_eas = G_matrix @ eas_w
        y_true_eas = (y_true_eas - y_true_eas.mean()) / (y_true_eas.std() + 1e-12)

        # EUR-predicted expression using the biased weight, also z-scored
        y_pred_eur = G_matrix @ eur_w
        y_pred_eur = (y_pred_eur - y_pred_eur.mean()) / (y_pred_eur.std() + 1e-12)

        yield {
            'gene_id': gene_id,
            'G': G_matrix,
            'y_true_eas': y_true_eas,
            'w_start': eur_w,
            'w_true': eas_w,
            'a_vec': a_vec,
            'y_eur': y_pred_eur,
            'out_dir': out_dir,
            'ga_params': ga_params,
            'eur_af': sim_eur_af,
            'eas_af': sim_eas_af,
            'snp_categories': categories,
            'is_effect': is_effect
        }

def run_kfold_simulation_batched(G_gpu, y_gpu, w_gpu, a_vec_gpu, rsid_list, **ga_params):
    """
    Batched Repeated K-Fold CV — same architecture as run_loocv_batch_ga_cupy in V3.py.

    G is shared across all folds (no fold-specific normalization); fold membership is
    expressed via a binary mask passed to Weight_GA_Cupy, matching real-data LOOCV behaviour.
    All n_folds × num_runs tasks are pre-built and dispatched in batch_limit-sized chunks,
    reducing Weight_GA_Cupy call count from n_folds to ceil(total_tasks / batch_limit).

    Args:
        G_gpu (cupy.ndarray): Genotype matrix on GPU, shape (n_snps, n_samples).
        y_gpu (cupy.ndarray): Target expression vector on GPU, shape (n_samples,).
        w_gpu (cupy.ndarray): Starting (biased/EUR) weight vector on GPU, shape (n_snps,).
        a_vec_gpu (cupy.ndarray): AF-similarity attenuation vector on GPU, shape (n_snps,).
        rsid_list (list[str]): SNP identifiers, length n_snps, forwarded to Weight_GA_Cupy.
        **ga_params: GA/CV hyperparameters. Recognized keys: 'num_runs' (GA repeats per fold),
            'n_splits' (K for K-Fold), 'n_repeats' (RepeatedKFold repeats, 1 = plain K-Fold),
            'cv_seed' (fold split RNG seed), 'batch_limit' (max tasks per Weight_GA_Cupy call).
            All other keys are forwarded to Weight_GA_Cupy as GA config.

    Returns:
        tuple:
            performance_df (pd.DataFrame): columns 'predict'/'actual', per-sample out-of-fold
                z-scored prediction and actual value, averaged across repeats.
            weight_hist_df (pd.DataFrame): one row per fold (RepeatedKFold fold), columns = rsid_list,
                the run-averaged GA scaling solution for that fold.
            weight_stability (float): mean per-SNP std of the GA scaling solution across folds —
                low value means the GA consistently converges to the same weights regardless of
                which samples are held out.
            train_corr (float): Pearson correlation between actual and predicted expression
                computed on the *full* dataset using the fold-averaged weights (overfitting diagnostic;
                not a true held-out metric).

    Notes:
        - fold membership is encoded as a (task, 1, n_samples) binary mask (0 = held out) rather than
          slicing G, so the same GPU G is reused across all folds/runs.
        - y is normalized per-fold using train-only mean/std (t_mu, t_std) before being broadcast to
          all num_runs GA runs of that fold, avoiding leakage of test-set statistics into training.
        - Test-set predictions are re-standardized using train-only prediction mean/std (tp_mu, tp_std)
          for a fair, leakage-free comparison against the train-normalized actual values.
        - Accumulators average predictions/actuals per sample across all repeats a sample appeared
          in as a test case (RepeatedKFold visits each sample n_repeats times).
    """
    from sklearn.model_selection import KFold, RepeatedKFold

    n_snps, n_samples = G_gpu.shape
    num_runs    = ga_params.get('num_runs', 5)
    n_splits    = ga_params.get('n_splits', 10)
    n_repeats   = ga_params.get('n_repeats', 5)
    cv_seed     = ga_params.get('cv_seed', 42)
    batch_limit = ga_params.get('batch_limit', 2048)
    if batch_limit <= 0:
        batch_limit = 2048

    _SKIP = {'n_splits', 'n_repeats', 'cv_seed'}
    ga_kw = {k: v for k, v in ga_params.items() if k not in _SKIP}

    # --- Build fold list ---
    if n_repeats > 1:
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=cv_seed)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)

    fold_list   = list(splitter.split(np.arange(n_samples)))
    n_folds     = len(fold_list)
    total_tasks = n_folds * num_runs

    # --- Pre-build mask and y_target for all folds × runs ---
    mask_all     = cp.ones((total_tasks, 1, n_samples), dtype=cp.float32)
    y_target_all = cp.zeros((total_tasks, 1, n_samples), dtype=cp.float32)
    fold_meta    = []   # (train_idx, test_idx, t_mu, t_std) per fold

    for f, (train_idx, test_idx) in enumerate(fold_list):
        y_train = y_gpu[cp.array(train_idx)]
        t_mu    = float(y_train.mean())
        t_std   = float(y_train.std()) + 1e-6
        y_norm  = (y_gpu - t_mu) / t_std          # normalize with train-only y stats

        s, e = f * num_runs, (f + 1) * num_runs
        mask_all[s:e, 0, test_idx] = 0.0          # mask out test samples
        y_target_all[s:e, 0, :]    = y_norm        # broadcast to all runs of this fold

        fold_meta.append((train_idx, test_idx, t_mu, t_std))

    # --- Dispatch in batch_limit chunks (identical to LOOCV) ---
    all_solutions = []
    offset  = 0
    backend = (G_gpu, w_gpu, None, a_vec_gpu)

    while offset < total_tasks:
        bs           = min(batch_limit, total_tasks - offset)
        mask_chunk   = mask_all[offset:offset + bs]
        y_chunk      = y_target_all[offset:offset + bs]

        solutions, _, _ = Weight_GA_Cupy(
            backend, rsid_list,
            num_batches=bs,
            mask=mask_chunk,
            y_target=y_chunk,
            **ga_kw
        )
        all_solutions.append(solutions[:, :-1])
        offset += bs

        del mask_chunk, y_chunk, solutions
        cp.get_default_memory_pool().free_all_blocks()

    del mask_all, y_target_all
    cp.get_default_memory_pool().free_all_blocks()

    all_solutions = cp.concatenate(all_solutions, axis=0)  # (total_tasks, n_snps)

    # --- Accumulate out-of-fold predictions ---
    pred_accumulator   = cp.zeros(n_samples, dtype=cp.float32)
    actual_accumulator = cp.zeros(n_samples, dtype=cp.float32)
    count_accumulator  = cp.zeros(n_samples, dtype=cp.float32)
    weight_history     = []

    for f, (train_idx, test_idx, t_mu, t_std) in enumerate(fold_meta):
        s, e = f * num_runs, (f + 1) * num_runs
        best_individual = all_solutions[s:e].mean(axis=0)   # average across runs
        weight_history.append(best_individual)

        final_w    = w_gpu * a_vec_gpu * best_individual
        pred_full  = cp.matmul(final_w, G_gpu)

        train_idx_gpu = cp.array(train_idx)
        test_idx_gpu  = cp.array(test_idx)

        pred_train = pred_full[train_idx_gpu]
        tp_mu  = pred_train.mean()
        tp_std = pred_train.std() + 1e-12

        pred_z = (pred_full[test_idx_gpu] - tp_mu) / tp_std

        pred_accumulator[test_idx_gpu]   += pred_z
        actual_accumulator[test_idx_gpu] += (y_gpu[test_idx_gpu] - t_mu) / t_std
        count_accumulator[test_idx_gpu]  += 1.0

    # --- Average over repeats ---
    predict_y = pred_accumulator   / (count_accumulator + 1e-12)
    actual_y  = actual_accumulator / (count_accumulator + 1e-12)

    # --- Weight stability ---
    weight_stack     = cp.stack(weight_history)
    weight_stability = float(weight_stack.std(axis=0).mean())

    # --- Train correlation (overfitting diagnostic) ---
    avg_w_all = weight_stack.mean(axis=0)
    wa_full   = w_gpu * a_vec_gpu * avg_w_all
    pred_full = cp.matmul(wa_full, G_gpu)
    y_z = (y_gpu    - y_gpu.mean())    / (y_gpu.std()    + 1e-6)
    p_z = (pred_full - pred_full.mean()) / (pred_full.std() + 1e-6)
    if cp.std(p_z) < 1e-9 or cp.std(y_z) < 1e-9:
        train_corr = 0.0
    else:
        train_corr = float(cp.corrcoef(y_z, p_z)[0, 1])
        if np.isnan(train_corr):
            train_corr = 0.0

    performance_df = pd.DataFrame({
        'predict': cp.asnumpy(predict_y),
        'actual':  cp.asnumpy(actual_y)
    })
    weight_hist_df = pd.DataFrame(cp.asnumpy(weight_stack), columns=rsid_list)

    return performance_df, weight_hist_df, weight_stability, train_corr

def process_simulation_gene(task):
    """Run the full GA optimization + evaluation pipeline for a single simulated gene task.

    Args:
        task (dict): One task dict as yielded by simulation_task_generator, containing 'gene_id',
            'G' (genotype matrix), 'y_true_eas', 'w_start', 'w_true', 'a_vec', 'ga_params',
            'is_effect', 'eur_af', 'eas_af', 'y_eur'.

    Returns:
        dict: On success, contains per-gene summary metrics ('N_SNPs', 'N_SNPs_After_Filter',
            'Expr_MAE_EUR', 'Expr_MAE', 'Weight_MAE', 'Weight_Corr', 'Weight_Stability',
            'Corr_Before', 'Corr_After', 'Train_Corr', 'Improvement', 'Time', 'Status': 'Success')
            plus per-SNP/per-sample arrays ('w_final', 'is_effect', 'tas_kept_mask', 'y_ga_pred_z',
            'y_actual_z', 'avg_scaling') used for downstream record logging. On failure, returns
            {'Status': 'Failed'}.

    Notes:
        - Sign-flip guard: if 'allow_sign_flip' is enabled but the gene's baseline (pre-GA)
          correlation is already >= sign_flip_threshold, sign-flip search is disabled for this
          gene — matches V3.py behavior, since sign-flipping is only useful for genes where the
          naive EUR-weight prediction is uncorrelated or anti-correlated with the true EAS signal.
        - TAS (Tier-Adaptive-Scaling / trans-ancestry SNP) filter is applied unconditionally before
          the GA runs, dropping/down-weighting SNPs whose AF divergence makes them unreliable;
          apply_tas_filter_cpu returns keep_idx mapping filtered SNPs back to original indices.
        - adaptive_scale only rescales the GA's population size and per-phase iteration counts
          (proportionally to how many SNPs survived the TAS filter, relative to adaptive_ref_snps)
          — it does not affect which SNPs are filtered.
        - After GA, the run-averaged GA scaling factor is expanded from the filtered (kept) SNP set
          back to the full original SNP set; SNPs dropped by the TAS filter get scaling=0, so
          their final weight w_final=0 (i.e. TAS treats them as pure noise).
        - Hard pruning: any SNP whose effective weight |w_start * a_vec * avg_scaling| falls below
          pruning_threshold has its scaling forced to exactly 0.0, to avoid near-zero numerical noise.
        - Correlation metrics: Corr_Before is EUR-predicted vs. true EAS expression (pre-GA
          baseline); Corr_After is GA out-of-fold prediction vs. actual (post-GA validation);
          Train_Corr is the in-sample correlation used only as an overfitting diagnostic.
    """
    gene_id = task['gene_id']
    G = task['G']
    y_true_eas = task['y_true_eas']
    w_start = task['w_start']
    w_true = task['w_true']
    a_vec = task['a_vec']
    ga_params = task['ga_params']
    is_effect = task['is_effect']
    n_snps_original = len(w_start)

    rsid_dummy = [f"rs{i}" for i in range(n_snps_original)]
    y_gpu = cp.array(y_true_eas)

    y_eur = task['y_eur']
    if np.std(y_eur) < 1e-9 or np.std(y_true_eas) < 1e-9:
        baseline_corr = 0.0
    else:
        baseline_corr = float(np.corrcoef(y_eur, y_true_eas)[0, 1])
        if np.isnan(baseline_corr):
            baseline_corr = 0.0

    # Sign-flip guard: keeps behavior consistent with V3.py
    ga_params_gene = dict(ga_params)
    sign_flip_threshold = float(ga_params.get('sign_flip_threshold', 0.01))
    if ga_params.get('allow_sign_flip', False) and baseline_corr >= sign_flip_threshold:
        ga_params_gene['allow_sign_flip'] = False
        logger.info(
            f"  [{gene_id}] baseline_corr={baseline_corr:.3f} >= {sign_flip_threshold} "
            f"→ sign_flip disabled"
        )

    # TAS filter: applied unconditionally, matching V3.py behavior
    af_diff_arr = np.abs(task['eur_af'] - task['eas_af'])
    dummy_df = pd.DataFrame({'weight': w_start})
    G_filt, w_filt, a_filt, rsid_filt, _, _, keep_idx = apply_tas_filter_cpu(
        G.T, w_start, a_vec, rsid_dummy, af_diff_arr, dummy_df,
        s_tier_threshold   = ga_params.get('s_tier_af_diff_max', 0.05),
        decay_scale        = ga_params.get('tas_decay_scale', 0.20),
        keep_frac_ml       = ga_params.get('tas_keep_frac_ml', 0.8),
        min_snps_to_filter = ga_params.get('tas_min_snps_to_filter', 5),
    )
    G_gpu     = cp.array(G_filt)
    w_gpu     = cp.array(w_filt)
    a_vec_gpu = cp.array(a_filt)
    rsid_dummy = rsid_filt

    # adaptive_scale only controls pop/iteration scaling, not the TAS filter itself
    adaptive_scale_used = ga_params.get('adaptive_scale', False)
    n_snps_after = len(rsid_dummy)

    start_time = time.time()

    try:

        ga_params_gene = dict(ga_params)

        if adaptive_scale_used:
            ref_snps = max(1, int(ga_params.get('adaptive_ref_snps', 20)))
            scale = min(1.0, n_snps_after / ref_snps)

            base_pop   = int(ga_params.get('population_size', 500))
            base_iter1 = int(ga_params.get('iter_num_s1', 500))
            base_iter2 = int(ga_params.get('iter_num_s2', 750))
            base_iter3 = int(ga_params.get('iter_num_s3', 250))

            adapt_pop   = max(int(ga_params.get('min_population', 300)), int(base_pop * scale))
            adapt_iter1 = max(int(ga_params.get('min_iter1', 200)), int(base_iter1 * scale))
            adapt_iter2 = max(int(ga_params.get('min_iter2', 300)), int(base_iter2 * scale))
            adapt_iter3 = max(int(ga_params.get('min_iter3', 100)), int(base_iter3 * scale))

            ga_params_gene.update({
                'population_size': adapt_pop,
                'iter_num_s1': adapt_iter1,
                'iter_num_s2': adapt_iter2,
                'iter_num_s3': adapt_iter3,
            })

        perf_df, weight_hist_df, weight_stability, train_corr = run_kfold_simulation_batched(
            G_gpu, y_gpu, w_gpu, a_vec_gpu, rsid_dummy, **ga_params_gene
        )

        # Average GA scaling across kept SNPs, then expand back to original size
        # Dropped SNPs get scaling=0 → w_final=0 (correct: TAS treated them as noise)
        avg_scaling_kept = weight_hist_df.mean(axis=0).values.copy()

        # Hard pruning: effective_w = |w * a_vec * avg_scaling| < pruning_threshold → forced to 0
        prune_thresh = ga_params.get('pruning_threshold', 1e-8)
        if prune_thresh > 0:
            effective_w = np.abs(w_filt * a_filt * avg_scaling_kept)
            avg_scaling_kept[effective_w < prune_thresh] = 0.0

        avg_scaling = np.zeros(n_snps_original, dtype=np.float32)
        avg_scaling[keep_idx] = avg_scaling_kept

        w_final = w_start * a_vec * avg_scaling   # dropped SNPs → 0

        y_ga_pred_z = perf_df['predict'].values   # CV out-of-fold predictions (z-scored)
        y_actual_z  = perf_df['actual'].values    # corresponding actuals  (z-scored)

        y_eur_z = task['y_eur']

        expr_mae_eur = np.mean(np.abs(task['y_true_eas'] - y_eur_z))
        expr_mae_ga  = np.mean(np.abs(y_actual_z - y_ga_pred_z))

        weight_mae  = np.mean(np.abs(w_true - w_final))
        weight_corr = np.corrcoef(w_true, w_final)[0, 1]

        corr_eur = np.corrcoef(task['y_true_eas'], y_eur_z)[0, 1]
        corr_ga  = np.corrcoef(y_actual_z, y_ga_pred_z)[0, 1]

        # Effect / null SNP counts
        n_effect     = int(np.sum(is_effect))
        n_null  = int(np.sum(~is_effect))

        cp.get_default_memory_pool().free_all_blocks()

        total_time = time.time() - start_time

        tas_kept_mask = np.zeros(n_snps_original, dtype=bool)
        tas_kept_mask[keep_idx] = True

        del G_gpu, w_gpu, a_vec_gpu, y_gpu
        cp.get_default_memory_pool().free_all_blocks()

        return {
            'gene_id':            gene_id,
            'N_SNPs':             n_snps_original,
            'N_SNPs_After_Filter': n_snps_after,
            'Adaptive_Scale':     adaptive_scale_used,
            'N_SNPs_Effect':      n_effect,
            'N_SNPs_Null':   n_null,
            'Expr_MAE_EUR':       expr_mae_eur,
            'Expr_MAE':           expr_mae_ga,
            'Weight_MAE':         weight_mae,
            'Weight_Corr':        weight_corr,
            'Weight_Stability':   weight_stability,
            'Corr_Before':        corr_eur,
            'Corr_After':         corr_ga,
            'Train_Corr':         train_corr,
            'Improvement':        np.abs(corr_ga) - np.abs(corr_eur),
            'Time':               total_time,
            'Status':             'Success',
            # Arrays for per-SNP record logging
            'w_final':            w_final,
            'is_effect':          is_effect,
            'tas_kept_mask':      tas_kept_mask,
            'y_ga_pred_z':        y_ga_pred_z,
            'y_actual_z':         y_actual_z,
            'avg_scaling':        avg_scaling,
        }

    except Exception as e:
        logger.error(f"Gene {gene_id} failed: {e}")
        cp.get_default_memory_pool().free_all_blocks()
        return {'Status': 'Failed'}


def main():
    """Parse CLI arguments, run the full simulation pipeline, and write reports/plots to disk.

    Args:
        None (reads sys.argv via argparse).

    Returns:
        None. Writes 'Final_Simulation_Report.csv', 'Simulated_SNP_Details.csv',
        'Simulated_Expression_GroundTruth.csv', 'Tier_Summary.csv', and
        'Simulation_Summary_Detail.png' into the resolved output subdirectory. Returns early
        (no output) if the config file is missing/invalid, the DB path is unreadable, or the
        tier ratios do not sum to 1.0.

    Notes:
        - Builds a descriptive sub_dir_name from the tier ratios, sign-flip probability, sample
          count, and adaptive_scale flag so that different simulation configs don't overwrite
          each other's outputs.
        - Loads the reference DB (gene/SNP weights + allele frequency table) via V4.reading_file_path
          / get_db_info / db_allele_freq_filtered, then feeds it into simulation_task_generator to
          produce one task per synthetic gene, each processed by process_simulation_gene.
        - Accumulates per-SNP and per-sample flat records (all_snp_records, all_expr_records)
          across all genes for CSV export, plus a Tier_Summary.csv computed by grouping SNP-level
          relative/absolute weight errors and noise-suppression ratios by AF tier
          (Identical/Small/Medium/Large).
        - Produces a 3x2 grid figure: gene-level weight relative/absolute error histograms,
          SNP-level weight relative/absolute error histograms, and correlation-distribution bar
          charts (Before-vs-Train, Before-vs-After) annotated with FDR-corrected significance
          markers from a per-bin Fisher's exact test (see nested add_significance_asterisks).
    """
    parser = argparse.ArgumentParser(description="GA Simulation Validation")

    parser.add_argument("--out_dir", type=str, default="./Simulation")
    parser.add_argument("--n_genes", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=200)

    # Supports a SNP-count range
    parser.add_argument("--n_snps_min", type=int, default=10, help="Minimum SNPs per gene")
    parser.add_argument("--n_snps_max", type=int, default=80, help="Maximum SNPs per gene")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--l_effect_ratio",   type=float, default=0.3, help="Proportion of effect SNPs in L-tier")
    parser.add_argument("--m_effect_ratio", type=float, default=0.7, help="Proportion of effect SNPs in M-tier")

    parser.add_argument("--ratio_identical",  type=float, default=0.4)
    parser.add_argument("--ratio_small",      type=float, default=0.3)
    parser.add_argument("--ratio_medium",     type=float, default=0.2)
    parser.add_argument("--ratio_large",      type=float, default=0.1)
    parser.add_argument("--sign_flip_prob",   type=float, default=0.0,
                        help="Probability that an effect M/L-tier SNP has its EAS weight sign flipped (0.0 = disabled)")
    parser.add_argument("--allow_sign_flip", action="store_true",
                        help="Allow negative scaling for L/M-tier SNPs in GA (enables direction flip search)")
    parser.add_argument("--sign_flip_threshold", type=float, default=0.1,
                        help="Enable GA sign-flip mode only when baseline correlation is below this threshold")
    parser.add_argument("--out_range",        action=argparse.BooleanOptionalAction, default=False,
                        help="Extend L-tier effect P range by outlier_scaling (--out_range / --no-out_range)")
    parser.add_argument("--outlier_scaling",  type=float, default=2.0,
                        help="Scaling factor for out_range mode: upper=[m_high, l_high*k], lower=[l_low/k, m_low]")

    parser.add_argument("--config", type=str, default="config.json", help="Path to JSON config")
    parser.add_argument("--GA_Phase1_iterations", type=int, default=200)
    parser.add_argument("--GA_Phase2_iterations", type=int, default=200)
    parser.add_argument("--GA_Phase3_iterations", type=int, default=100)

    parser.add_argument("--Initial_mutation_rate", type=float, default=0.5)
    parser.add_argument("--runs", type=int, default=5, dest="Numbers_of_GA_runs")
    parser.add_argument("--pop_size", type=int, default=200, dest="GA_population_sizes")

    parser.add_argument("--s_tier_af_diff", type=float, default=0.05)
    parser.add_argument("--s_tier_low_bound", type=float, default=0.9)
    parser.add_argument("--s_tier_high_bound", type=float, default=1.1)
    parser.add_argument("--m_tier_af_diff", type=float, default=0.25)
    parser.add_argument("--m_tier_low_bound", type=float, default=0.4)
    parser.add_argument("--m_tier_high_bound", type=float, default=3.0)
    parser.add_argument("--l_tier_low_bound", type=float, default=0.0)
    parser.add_argument("--l_tier_high_bound", type=float, default=5.0)

    parser.add_argument("--batch_limit", type=int, default=2048)

    parser.add_argument("--n_splits",   type=int, default=5,  help="K for K-Fold / Repeated K-Fold")
    parser.add_argument("--n_repeats",  type=int, default=1,  help="Repeats for Repeated K-Fold (1 = plain K-Fold)")
    parser.add_argument("--cv_seed",    type=int, default=42, help="Random seed for fold splitting")

    parser.add_argument("--adaptive_scale", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply TAS filter before GA (--adaptive_scale / --no-adaptive_scale)")
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
    parser.add_argument("--tas_decay_scale",        type=float, default=0.20)
    parser.add_argument("--tas_keep_frac_ml",       type=float, default=0.8)
    parser.add_argument("--tas_min_snps_to_filter", type=int,   default=5)

    args = parser.parse_args()

    sub_dir_name = (
        f"id{args.ratio_identical}_"
        f"sm{args.ratio_small}_"
        f"med{args.ratio_medium}_"
        f"lg{args.ratio_large}_"
        f"sfp{args.sign_flip_prob}_"
        f"S{args.n_samples}_"
        f"as{'1' if args.adaptive_scale else '0'}"
        + (f"_or{args.outlier_scaling}" if args.out_range else "")
    )
    out_dir = os.path.join(args.out_dir, sub_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return
    try:
        with open(args.config, 'r') as f:
            config_paths = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON config file: {e}")
        return

    total_ratio = (args.ratio_identical + args.ratio_small +
                   args.ratio_medium + args.ratio_large)
    if not np.isclose(total_ratio, 1.0):
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio:.4f})")
        return

    try:
        genotype, predict_expr_raw, real_expr_raw, allele_freq, annotation, db_path, _ = reading_file_path(config_paths)
        logger.info("  Raw data loaded successfully")
    except Exception as e:
        logger.error(f"  Failed to load raw data: {e}")
        logger.error(traceback.format_exc())
        return

    weight_df, gene_info_df = get_db_info(db_path)
    db_weight_filtered = db_allele_freq_filtered(weight_df, allele_freq)

    sim_ratios = {
        'identical':      args.ratio_identical,
        'small':          args.ratio_small,
        'medium':         args.ratio_medium,
        'large':          args.ratio_large,
        'sign_flip_prob': args.sign_flip_prob,
    }

    ga_params = {
        'num_runs': args.Numbers_of_GA_runs,
        'iter_num_s1': args.GA_Phase1_iterations,
        'iter_num_s2': args.GA_Phase2_iterations,
        'iter_num_s3': args.GA_Phase3_iterations,
        'population_size': args.GA_population_sizes,
        'initial_mutation_rate': args.Initial_mutation_rate,
        's_tier_threshold': 1.0 - args.s_tier_af_diff,
        's_tier_low': args.s_tier_low_bound,
        's_tier_high': args.s_tier_high_bound,
        'm_tier_threshold': 1.0 - args.m_tier_af_diff,
        'm_tier_low': args.m_tier_low_bound,
        'm_tier_high': args.m_tier_high_bound,
        'l_tier_low': args.l_tier_low_bound,
        'l_tier_high': args.l_tier_high_bound,
        'pruning_threshold': 1e-8,
        'batch_limit':       args.batch_limit,
        'allow_sign_flip': args.allow_sign_flip,
        'sign_flip_threshold': args.sign_flip_threshold,
        'seed':     args.seed,
        'n_splits':  args.n_splits,
        'n_repeats': args.n_repeats,
        'cv_seed':   args.cv_seed,
        'adaptive_scale':          args.adaptive_scale,
        'adaptive_ref_snps':       args.adaptive_ref_snps,
        'min_population':          args.min_population,
        'min_iter1':               args.min_iter1,
        'min_iter2':               args.min_iter2,
        'min_iter3':               args.min_iter3,
        's_tier_af_diff_max':      args.s_tier_af_diff,
        'tas_decay_scale':         args.tas_decay_scale,
        'tas_keep_frac_ml':        args.tas_keep_frac_ml,
        'tas_min_snps_to_filter':  args.tas_min_snps_to_filter,
    }


    logger.info("Starting Simulation...")

    gen = simulation_task_generator(
        args.n_genes, args.n_samples, args.n_snps_min, args.n_snps_max, out_dir, ga_params,
        db_weight_filtered,
        sim_ratios=sim_ratios, seed_base=args.seed,
        l_effect_ratio=args.l_effect_ratio, m_effect_ratio=args.m_effect_ratio,
        out_range=args.out_range, outlier_scaling=args.outlier_scaling,
    )

    all_snp_records = []
    all_expr_records = []
    results = []

    for task in gen:
        res = process_simulation_gene(task)
        if res['Status'] == 'Success':
            results.append(res)
            logger.info(
                f"Processed {res['gene_id']:8s} | "
                f"SNPs: {res['N_SNPs']:3d} (E:{res['N_SNPs_Effect']}/N:{res['N_SNPs_Null']}) | "
                f"Corr: {res['Corr_Before']:.3f}->{res['Corr_After']:.3f} "
                f"(Train:{res['Train_Corr']:.3f}) | "
                f"W_Stab: {res['Weight_Stability']:.4f} | "
                f"Time: {res['Time']:.1f}s"
            )

            n_snps_current = len(task['w_start'])
            gene_id = task['gene_id']

            w_final_array    = res['w_final']
            is_effect_array  = res['is_effect']
            avg_scaling_arr  = res['avg_scaling']
            tas_kept_arr     = res['tas_kept_mask']

            for j in range(n_snps_current):
                all_snp_records.append({
                    'Gene_ID':           gene_id,
                    'SNP_Idx':           j,
                    'RSID':              f"rs{j}",
                    'Is_Effect':         bool(is_effect_array[j]),
                    'TAS_Kept':          bool(tas_kept_arr[j]),
                    'EUR_Weight_Biased': float(task['w_start'][j]),
                    'EAS_Weight_True':   float(task['w_true'][j]),
                    'GA_Weight_Final':   float(w_final_array[j]),
                    'EUR_AF':            float(task['eur_af'][j]),
                    'EAS_AF':            float(task['eas_af'][j]),
                    'AF_Diff_Cat':       int(task['snp_categories'][j]),
                    'A_Vec':             float(task['a_vec'][j]),
                    'GA_Scaling':        float(avg_scaling_arr[j]),
                    'Effective_W':       float(task['a_vec'][j] * avg_scaling_arr[j]),
                })

            n_samples_current = len(task['y_true_eas'])
            y_ga_pred_arr  = res['y_ga_pred_z']   # per-sample CV prediction
            y_actual_z_arr = res['y_actual_z']    # per-sample z-scored actual
            for s in range(n_samples_current):
                all_expr_records.append({
                    'Gene_ID':    gene_id,
                    'Sample_ID':  f"Sample_{s}",
                    'Y_True_EAS': float(task['y_true_eas'][s]),
                    'Y_Pred_EUR': float(task['y_eur'][s]),
                    'Y_Pred_GA':  float(y_ga_pred_arr[s]),    # GA CV prediction
                    'Y_Actual_Z': float(y_actual_z_arr[s]),   # fold-normalised actual
                })

    df = pd.DataFrame(results)

    if not df.empty:
        logger.info("Saving Ground Truth Data...")

        # Drop array columns before writing summary CSV
        csv_cols = [c for c in df.columns if c not in ('w_final', 'is_effect', 'tas_kept_mask', 'y_ga_pred_z', 'y_actual_z', 'avg_scaling')]
        df[csv_cols].to_csv(os.path.join(out_dir, "Final_Simulation_Report.csv"), index=False)

        pd.DataFrame(all_snp_records).to_csv(os.path.join(out_dir, "Simulated_SNP_Details.csv"), index=False)
        pd.DataFrame(all_expr_records).to_csv(os.path.join(out_dir, "Simulated_Expression_GroundTruth.csv"), index=False)

        # --- Per-AF-tier grouped statistics ---
        tier_label_map = {0: 'Identical', 1: 'Small', 2: 'Medium', 3: 'Large'}
        snp_df_tier = pd.DataFrame(all_snp_records).copy()
        snp_df_tier['AF_Tier'] = snp_df_tier['AF_Diff_Cat'].map(tier_label_map)
        snp_df_tier['Rel_Error'] = (
            np.abs(snp_df_tier['GA_Weight_Final'] - snp_df_tier['EAS_Weight_True'])
            / (np.abs(snp_df_tier['EAS_Weight_True']) + 0.01)
        )
        snp_df_tier['Abs_Error'] = np.abs(snp_df_tier['GA_Weight_Final'] - snp_df_tier['EAS_Weight_True'])

        # Effect-only: weight recovery metrics per tier
        effect_tier = snp_df_tier[snp_df_tier['Is_Effect'] == True]
        tier_effect_stats = effect_tier.groupby('AF_Tier').agg(
            N_Effect_SNPs    = ('RSID', 'count'),
            Avg_Rel_Error    = ('Rel_Error', 'mean'),
            Avg_Abs_Error    = ('Abs_Error', 'mean'),
            Avg_GA_Scaling   = ('GA_Scaling', 'mean'),
            Avg_Effective_W  = ('Effective_W', 'mean'),
        ).reset_index()

        # Null: noise suppression per tier
        null_tier = snp_df_tier[snp_df_tier['Is_Effect'] == False].copy()
        if not null_tier.empty:
            null_tier['Suppression_Ratio'] = (
                np.abs(null_tier['GA_Weight_Final'])
                / (np.abs(null_tier['EUR_Weight_Biased']) + 1e-20)
            )
            tier_noise_stats = null_tier.groupby('AF_Tier').agg(
                N_Null_SNPs    = ('RSID', 'count'),
                Avg_Suppression_Ratio = ('Suppression_Ratio', 'mean'),
            ).reset_index()
            tier_summary = tier_effect_stats.merge(tier_noise_stats, on='AF_Tier', how='outer')
        else:
            tier_summary = tier_effect_stats

        # Enforce display order
        tier_order = ['Identical', 'Small', 'Medium', 'Large']
        tier_summary['AF_Tier'] = pd.Categorical(tier_summary['AF_Tier'], categories=tier_order, ordered=True)
        tier_summary = tier_summary.sort_values('AF_Tier').reset_index(drop=True)

        tier_summary.to_csv(os.path.join(out_dir, "Tier_Summary.csv"), index=False)
        logger.info("Saved Tier_Summary.csv")
        logger.info(f"\n{tier_summary.to_string(index=False)}")

        try:
            sns.set_style("whitegrid")
            fig = plt.figure(figsize=(28, 33))
            gs  = fig.add_gridspec(3, 2, hspace=0.48, wspace=0.3)

            # Row 0 — Gene-level weight error
            ax_rel = fig.add_subplot(gs[0, 0])
            ax_abs = fig.add_subplot(gs[0, 1])
            # Row 1 — SNP-level weight error
            ax_snp_rel = fig.add_subplot(gs[1, 0])
            ax_snp_abs = fig.add_subplot(gs[1, 1])
            # Row 2 — Correlation charts
            ax_corr_train = fig.add_subplot(gs[2, 0])
            ax_corr_val   = fig.add_subplot(gs[2, 1])

            snp_df = pd.DataFrame(all_snp_records)
            snp_df['Rel_Error'] = (
                np.abs(snp_df['GA_Weight_Final'] - snp_df['EAS_Weight_True'])
                / (np.abs(snp_df['EAS_Weight_True']) + 0.01)
            )
            snp_df['Abs_Error'] = np.abs(snp_df['GA_Weight_Final'] - snp_df['EAS_Weight_True'])
            effect_snp = snp_df[snp_df['Is_Effect'] == True]

            # Top-left: Weight Relative Fidelity
            gene_rel_mae = effect_snp.groupby('Gene_ID')['Rel_Error'].mean()
            rel_bins   = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf]
            rel_labels = ['<10%\n(Excellent)', '10-20%', '20-50%', '50-100%', '1-2x', '2-5x', '>5x\n(Failed)']
            binned_rel = pd.cut(gene_rel_mae, bins=rel_bins, labels=rel_labels, right=False)
            dist_rel   = binned_rel.value_counts().reindex(rel_labels, fill_value=0).reset_index()
            dist_rel.columns = ['Range', 'Count']

            sns.barplot(data=dist_rel, x='Range', y='Count', palette='RdYlGn_r',
                        ax=ax_rel, edgecolor='black', alpha=0.8)
            ax_rel.set_title(f"Weight Relative Fidelity (Relative Error %)\nAvg Rel-Error: {gene_rel_mae.mean():.2%}",
                             fontsize=15, fontweight='bold')
            ax_rel.set_xlabel("Error Magnitude Relative to Ground Truth", fontsize=13)
            ax_rel.set_ylabel("Number of Genes", fontsize=13)
            for p in ax_rel.patches:
                if p.get_height() > 0:
                    ax_rel.annotate(f'{int(p.get_height())}',
                                    (p.get_x() + p.get_width() / 2., p.get_height()),
                                    ha='center', va='bottom', fontweight='bold',
                                    xytext=(0, 5), textcoords='offset points')

            # Top-right: Weight Absolute MAE
            gene_abs_mae = effect_snp.groupby('Gene_ID')['Abs_Error'].mean()
            abs_bins   = [0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0, np.inf]
            abs_labels = ['<1e-5', '1e-5~1e-4', '1e-4~5e-4', '5e-4~1e-3', '1e-3~5e-3', '5e-3~1e-2',
                          '1e-2~5e-2', '5e-2~0.1', '0.1~0.5', '0.5~1.0', '>1.0']
            binned_abs = pd.cut(gene_abs_mae, bins=abs_bins, labels=abs_labels, right=False)
            dist_abs   = binned_abs.value_counts().reindex(abs_labels, fill_value=0).reset_index()
            dist_abs.columns = ['Range', 'Count']

            sns.barplot(data=dist_abs, x='Range', y='Count', palette='viridis',
                        ax=ax_abs, edgecolor='black', alpha=0.8)
            ax_abs.set_title(f"Weight Absolute MAE (Actual Values)\nAvg Abs-MAE: {gene_abs_mae.mean():.4f}",
                             fontsize=15, fontweight='bold')
            ax_abs.set_xlabel("Absolute Error Range (Mean per Gene)", fontsize=13)
            ax_abs.set_ylabel("Number of Genes", fontsize=13)
            ax_abs.tick_params(axis='x', rotation=30)
            for p in ax_abs.patches:
                if p.get_height() > 0:
                    ax_abs.annotate(f'{int(p.get_height())}',
                                    (p.get_x() + p.get_width() / 2., p.get_height()),
                                    ha='center', va='bottom', fontweight='bold',
                                    xytext=(0, 5), textcoords='offset points')

            # Middle-left: SNP-level Relative Error distribution
            binned_rel_snp = pd.cut(effect_snp['Rel_Error'], bins=rel_bins, labels=rel_labels, right=False)
            dist_rel_snp   = binned_rel_snp.value_counts().reindex(rel_labels, fill_value=0).reset_index()
            dist_rel_snp.columns = ['Range', 'Count']

            sns.barplot(data=dist_rel_snp, x='Range', y='Count', palette='RdYlGn_r',
                        ax=ax_snp_rel, edgecolor='black', alpha=0.8)
            ax_snp_rel.set_title(
                f"SNP-level Relative Error Distribution\n"
                f"Median: {effect_snp['Rel_Error'].median():.2%}  Mean: {effect_snp['Rel_Error'].mean():.2%}",
                fontsize=15, fontweight='bold')
            ax_snp_rel.set_xlabel("Error Magnitude Relative to Ground Truth", fontsize=13)
            ax_snp_rel.set_ylabel("Number of Effect SNPs", fontsize=13)
            for p in ax_snp_rel.patches:
                if p.get_height() > 0:
                    ax_snp_rel.annotate(f'{int(p.get_height())}',
                                        (p.get_x() + p.get_width() / 2., p.get_height()),
                                        ha='center', va='bottom', fontweight='bold',
                                        xytext=(0, 5), textcoords='offset points')

            # Middle-right: SNP-level Absolute Error distribution
            binned_abs_snp = pd.cut(effect_snp['Abs_Error'], bins=abs_bins, labels=abs_labels, right=False)
            dist_abs_snp   = binned_abs_snp.value_counts().reindex(abs_labels, fill_value=0).reset_index()
            dist_abs_snp.columns = ['Range', 'Count']

            sns.barplot(data=dist_abs_snp, x='Range', y='Count', palette='viridis',
                        ax=ax_snp_abs, edgecolor='black', alpha=0.8)
            ax_snp_abs.set_title(
                f"SNP-level Absolute Error Distribution\n"
                f"Median: {effect_snp['Abs_Error'].median():.4f}  Mean: {effect_snp['Abs_Error'].mean():.4f}",
                fontsize=15, fontweight='bold')
            ax_snp_abs.set_xlabel("Absolute Error Range (Per SNP)", fontsize=13)
            ax_snp_abs.set_ylabel("Number of Effect SNPs", fontsize=13)
            ax_snp_abs.tick_params(axis='x', rotation=30)
            for p in ax_snp_abs.patches:
                if p.get_height() > 0:
                    ax_snp_abs.annotate(f'{int(p.get_height())}',
                                        (p.get_x() + p.get_width() / 2., p.get_height()),
                                        ha='center', va='bottom', fontweight='bold',
                                        xytext=(0, 5), textcoords='offset points')

            # Bottom: Correlation charts (Train | Validation side by side)
            corr_bins   = [-np.inf, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, np.inf]
            corr_labels = ['<-0.6', '-0.6~-0.3', '-0.3~-0.1', '-0.1~0.1', '0.1~0.3', '0.3~0.6', '>0.6']

            def make_bar_data(col_a, col_b, label_a, label_b):
                """Bin two correlation columns of df into corr_bins and stack them into one long
                DataFrame for a grouped seaborn barplot.

                Args:
                    col_a (str): Name of the first correlation column in df (e.g. 'Corr_Before').
                    col_b (str): Name of the second correlation column in df (e.g. 'Train_Corr').
                    label_a (str): Display label for col_a's method in the 'Method' column.
                    label_b (str): Display label for col_b's method in the 'Method' column.

                Returns:
                    pd.DataFrame: Concatenation of both binned value-count tables, with columns
                        'Correlation_Range', 'Count', 'Method' — ready for sns.barplot(hue='Method').
                """
                ca = pd.cut(df[col_a], bins=corr_bins, labels=corr_labels, right=False) \
                       .value_counts().reindex(corr_labels, fill_value=0).reset_index()
                ca.columns = ['Correlation_Range', 'Count']
                ca['Method'] = label_a
                cb = pd.cut(df[col_b], bins=corr_bins, labels=corr_labels, right=False) \
                       .value_counts().reindex(corr_labels, fill_value=0).reset_index()
                cb.columns = ['Correlation_Range', 'Count']
                cb['Method'] = label_b
                return pd.concat([ca, cb], ignore_index=True)

            def add_significance_asterisks(ax, bar_data, method='fdr_bh'):
                """Annotate a grouped correlation-range barplot with significance markers, using
                multiple-comparison-corrected Fisher's exact tests per bin.

                Args:
                    ax (matplotlib.axes.Axes): Axes containing the two-method grouped barplot to annotate.
                    bar_data (pd.DataFrame): Output of make_bar_data, columns 'Correlation_Range',
                        'Count', 'Method' with exactly two distinct Method values.
                    method (str): Multiple-testing correction method passed to
                        statsmodels.multipletests ('bonferroni' or 'fdr_bh'; fdr_bh is more lenient).

                Returns:
                    None. Mutates ax in place by drawing bracket lines and asterisk text above
                    bars whose corrected p-value < 0.05.

                Notes:
                    - For each correlation-range bin, builds a 2x2 contingency table
                      [[count_method1, total_n - count_method1], [count_method2, total_n - count_method2]]
                      and runs Fisher's exact test to compare the proportion of genes falling in
                      that bin between the two methods.
                    - total_n is bar_data['Count'].sum() // 2 (each method contributes half of the
                      total row count since every gene appears once per method).
                    - Raw p-values across all bins are corrected together via statsmodels
                      multipletests before thresholding, to control the false discovery/family-wise
                      error rate rather than testing each bin independently.
                    - Significance thresholds: p_corr < 0.001 → '***', < 0.01 → '**', < 0.05 → '*';
                      p_corr >= 0.05 bins are left unannotated.
                    - If annotation height would exceed the current y-axis limit, the axis is
                      rescaled (y_max_ax) so brackets remain visible.
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

                # Step 1: collect all raw p-values first
                pvals = []
                valid_indices = []

                for i, r in enumerate(ranges):
                    c1 = bar_data[(bar_data['Correlation_Range'] == r) & (bar_data['Method'] == m1)]['Count'].values
                    c2 = bar_data[(bar_data['Correlation_Range'] == r) & (bar_data['Method'] == m2)]['Count'].values
                    c1 = int(c1[0]) if len(c1) > 0 else 0
                    c2 = int(c2[0]) if len(c2) > 0 else 0

                    if c1 == 0 and c2 == 0:
                        pvals.append(1.0)        # no data for this bin, use max p-value
                        valid_indices.append(i)
                        continue

                    table = [[c1, total_n - c1], [c2, total_n - c2]]
                    _, p  = fisher_exact(table)
                    pvals.append(p)
                    valid_indices.append(i)

                # Step 2: correct all p-values together
                if len(pvals) > 0:
                    _, pvals_corrected, _, _ = multipletests(pvals, method=method)
                else:
                    return

                # Step 3: use corrected p-values to decide annotation
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

            # --- Bottom-left: Before GA vs Train Corr ---
            bar_data_train = make_bar_data('Corr_Before', 'Train_Corr',
                                           'PrediXcan (Before)', 'Train Corr (In-sample)')
            sns.barplot(data=bar_data_train, x='Correlation_Range', y='Count', hue='Method',
                        palette=['royalblue', 'darkorange'], ax=ax_corr_train,
                        alpha=0.85, edgecolor='black')
            for container in ax_corr_train.containers:
                ax_corr_train.bar_label(container, fmt='%d', padding=3, fontsize=10, fontweight='bold')
            add_significance_asterisks(ax_corr_train, bar_data_train)
            avg_corr_pre   = df['Corr_Before'].mean()
            avg_corr_train = df['Train_Corr'].mean()
            ax_corr_train.set_title(
                f"Gene Distribution: Before GA vs Train Corr\nAvg Corr: {avg_corr_pre:.3f} → {avg_corr_train:.3f}",
                fontsize=13, fontweight='bold', pad=12)
            ax_corr_train.set_xlabel("Correlation Range (Higher is Better)", fontsize=11)
            ax_corr_train.set_ylabel("Number of Genes", fontsize=11)
            ax_corr_train.legend(loc='upper left', fontsize=10, framealpha=0.9)
            ax_corr_train.tick_params(axis='x', rotation=20)
            ax_corr_train.grid(True, linestyle='--', alpha=0.3, axis='y')

            # --- Bottom-right: Before GA vs After GA (Validation) ---
            bar_data_val = make_bar_data('Corr_Before', 'Corr_After',
                                         'PrediXcan (Before)', 'GA Optimized (After)')
            sns.barplot(data=bar_data_val, x='Correlation_Range', y='Count', hue='Method',
                        palette=['royalblue', 'crimson'], ax=ax_corr_val,
                        alpha=0.85, edgecolor='black')
            for container in ax_corr_val.containers:
                ax_corr_val.bar_label(container, fmt='%d', padding=3, fontsize=10, fontweight='bold')
            add_significance_asterisks(ax_corr_val, bar_data_val)
            avg_corr_ga  = df['Corr_After'].mean()
            n_improved   = int(((df['Corr_After'] - df['Corr_Before']) > 0.05).sum())
            n_worsened   = int(((df['Corr_Before'] - df['Corr_After']) > 0.05).sum())
            ax_corr_val.set_title(
                f"Gene Distribution: Before vs After GA\nAvg Corr: {avg_corr_pre:.3f} → {avg_corr_ga:.3f}",
                fontsize=13, fontweight='bold', pad=12)
            ax_corr_val.set_xlabel("Correlation Range (Higher is Better)", fontsize=11)
            ax_corr_val.set_ylabel("Number of Genes", fontsize=11)
            ax_corr_val.legend(loc='upper left', fontsize=10, framealpha=0.9)
            ax_corr_val.tick_params(axis='x', rotation=20)
            ax_corr_val.grid(True, linestyle='--', alpha=0.3, axis='y')
            ax_corr_val.text(0.98, 0.95,
                             f"Significantly Improved (>0.05): {n_improved} genes\n"
                             f"Significantly Worsened: {n_worsened} genes",
                             transform=ax_corr_val.transAxes, va='top', ha='right',
                             fontsize=10, fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

            plot_path = os.path.join(out_dir, "Simulation_Summary_Detail.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Summary Detail plot saved to {plot_path}")

        except Exception as e:
            logger.warning(f"Failed to generate visualization plots: {e}")

        logger.info(f"Simulation Complete. All results safely stored in {out_dir}")

if __name__ == "__main__":
    main()