#!/bin/bash
# =============================================================================
# GA_pipeline.sh
# 正式執行 GA.py 的批次腳本。
#
# 預設策略：平常只需要跑 LOOCV（Sign + Unsign 兩種），這是最終要用的結果。
# 10-Fold / 3-Fold 只有在「要比較不同 CV 方法的穩定度/差異」時才需要開啟，
# 平常請保持關閉，以節省 GPU 時間。
#
# 開關方式：修改最上面的 RUN_* 變數（true/false）即可，不需要更動下面的邏輯。
# =============================================================================
set -euo pipefail

# ── 是否執行各個 job（依你的目的切換 true/false）──────────────────────────
RUN_LOOCV_SIGN=true      # 一般情況下只需要開這兩個
RUN_LOOCV_UNSIGN=true
RUN_10FOLD_SIGN=false    # 只有要比較 CV 方法差異時才開
RUN_10FOLD_UNSIGN=false
RUN_3FOLD_SIGN=false     # 只有要比較 CV 方法差異時才開
RUN_3FOLD_UNSIGN=false

# ── 是否等前一個 job 跑完才跑下一個（同一張 GPU 建議 true，避免搶記憶體）──
SEQUENTIAL=true

# ── 路徑設定 ────────────────────────────────────────────────────────────
SCRIPT="/mnt/data_80T/anywaycheng/GA/GA.py"
CONFIG="/mnt/data_80T/anywaycheng/GA/config.json"
OUT_BASE="/mnt/data_80T/anywaycheng/GA/5%_Result"

# ── 共用參數（所有 job 皆相同，對應 get_args 預設之外的正式執行設定）────
COMMON_PARAMS=(
  --config                 "${CONFIG}"
  --GA_Phase1_iterations   500
  --GA_Phase2_iterations   750
  --GA_Phase3_iterations   250
  --GA_population_sizes    1500
  --Numbers_of_GA_runs     50
  --Initial_mutation_rate  0.5
  --pruning_threshold      1e-8
  --s_tier_af_diff_max     0.05
  --s_tier_low_bound       0.9
  --s_tier_high_bound      1.1
  --m_tier_af_diff_max     0.25
  --m_tier_low_bound       0.5
  --l_tier_low_bound       0.0
  --l_tier_high_bound      0.5
  --adaptive_scale
  --adaptive_ref_snps      20
  --tas_keep_frac_ml       0.8
  --tas_min_snps_to_filter 5
  --cv_seed                42
  --plot_thread            8
)

# m_tier_high_bound 大多數 job 用 3.0，只有 3Fold_Sign 用 2.0（沿用歷史設定，
# 若之後要統一請直接改這裡）。
M_TIER_HIGH_DEFAULT=3.0
M_TIER_HIGH_3FOLD_SIGN=2.0

# batch_limit 大多數 job 用 6400，只有 3Fold_Unsign 用 12800（沿用歷史設定）。
BATCH_LIMIT_DEFAULT=6400
BATCH_LIMIT_3FOLD_UNSIGN=12800

# ── Runner function ─────────────────────────────────────────────────────
# 用法: run_job <label> <out_subdir> <extra_params...>
run_job() {
  local label="$1"; local out_subdir="$2"; shift 2
  local extra=("$@")

  local out_dir="${OUT_BASE}/${out_subdir}"
  local log="${out_dir}/ga_run_optimized.log"
  mkdir -p "${out_dir}"

  echo ""
  echo "-------------------------------------"
  echo "[$(date '+%H:%M:%S')] START  ${label}"
  echo "  -> log: ${log}"
  echo "-------------------------------------"

  nohup python "${SCRIPT}" \
    "${COMMON_PARAMS[@]}" \
    "${extra[@]}" \
    > "${log}" 2>&1 &

  local pid=$!
  echo "  PID: ${pid}"

  if [ "${SEQUENTIAL}" = true ]; then
    wait "${pid}"
    local rc=$?
    if [ $rc -eq 0 ]; then
      echo "[$(date '+%H:%M:%S')] DONE   ${label}"
    else
      echo "[$(date '+%H:%M:%S')] FAILED ${label}  (log: ${log})"
      FAILED+=("${label}")
    fi
  else
    # 平行模式：不等待，PID 記錄下來供之後自行 `wait` 或 `ps` 查詢。
    PIDS+=("${pid}:${label}")
  fi
}

FAILED=()
PIDS=()

# ── LOOCV（一般情況下只需要這兩個）──────────────────────────────────────
if [ "${RUN_LOOCV_SIGN}" = true ]; then
  run_job "V4 LOOCV (Sign)" "V4_LOOCV_Sign" \
    --m_tier_high_bound "${M_TIER_HIGH_DEFAULT}" \
    --allow_sign_flip --sign_flip_threshold 0.1 \
    --cv_method loocv \
    --batch_limit "${BATCH_LIMIT_DEFAULT}"
fi

if [ "${RUN_LOOCV_UNSIGN}" = true ]; then
  run_job "V4 LOOCV (Unsign)" "V4_LOOCV_Unsign" \
    --m_tier_high_bound "${M_TIER_HIGH_DEFAULT}" \
    --cv_method loocv \
    --batch_limit "${BATCH_LIMIT_DEFAULT}"
fi

# ── 10-Fold（僅比較用）───────────────────────────────────────────────────
if [ "${RUN_10FOLD_SIGN}" = true ]; then
  run_job "V4 10Fold (Sign)" "V4_10Fold_Sign" \
    --m_tier_high_bound "${M_TIER_HIGH_DEFAULT}" \
    --allow_sign_flip --sign_flip_threshold 0.1 \
    --cv_method repeated_kfold --n_splits 10 --n_repeats 5 \
    --batch_limit "${BATCH_LIMIT_DEFAULT}"
fi

if [ "${RUN_10FOLD_UNSIGN}" = true ]; then
  run_job "V4 10Fold (Unsign)" "V4_10Fold_Unsign" \
    --m_tier_high_bound "${M_TIER_HIGH_DEFAULT}" \
    --cv_method repeated_kfold --n_splits 10 --n_repeats 5 \
    --batch_limit "${BATCH_LIMIT_DEFAULT}"
fi

# ── 3-Fold（僅比較用）────────────────────────────────────────────────────
if [ "${RUN_3FOLD_SIGN}" = true ]; then
  run_job "V4 3Fold (Sign)" "V4_3Fold_Sign" \
    --m_tier_high_bound "${M_TIER_HIGH_3FOLD_SIGN}" \
    --allow_sign_flip --sign_flip_threshold 0.1 \
    --cv_method repeated_kfold --n_splits 3 --n_repeats 5 \
    --batch_limit "${BATCH_LIMIT_DEFAULT}"
fi

if [ "${RUN_3FOLD_UNSIGN}" = true ]; then
  run_job "V4 3Fold (Unsign)" "V4_3Fold_Unsign" \
    --m_tier_high_bound "${M_TIER_HIGH_DEFAULT}" \
    --cv_method repeated_kfold --n_splits 3 --n_repeats 5 \
    --batch_limit "${BATCH_LIMIT_3FOLD_UNSIGN}"
fi

# ── 平行模式下，最後統一等待並回報結果 ──────────────────────────────────
if [ "${SEQUENTIAL}" != true ] && [ ${#PIDS[@]} -gt 0 ]; then
  echo ""
  echo "-------------------------------------"
  echo "Parallel jobs launched, waiting for all to finish..."
  echo "-------------------------------------"
  for entry in "${PIDS[@]}"; do
    pid="${entry%%:*}"
    label="${entry#*:}"
    if wait "${pid}"; then
      echo "[$(date '+%H:%M:%S')] DONE   ${label}"
    else
      echo "[$(date '+%H:%M:%S')] FAILED ${label}"
      FAILED+=("${label}")
    fi
  done
fi

# ── Final summary ────────────────────────────────────────────────────────
echo ""
echo "======================================"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "  All selected jobs completed successfully."
else
  echo "  Failed: ${FAILED[*]}"
  exit 1
fi
echo "======================================"
