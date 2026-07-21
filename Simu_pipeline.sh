#!/bin/bash
# =============================================================================
# run_simulation_grid.sh
# Grid: AF-tier ratios × sample sizes × modes (flip/adaptive/out_range)
# =============================================================================

set -euo pipefail

CONFIG="/mnt/data_80T/anywaycheng/GA/Testing/simu_config.json"
SIM_SCRIPT="/mnt/data_80T/anywaycheng/GA/Testing/GA_Simulation.py"
BASE_OUT="/mnt/data_80T/anywaycheng/GA/Testing/Simulation"
LOG_DIR="${BASE_OUT}/logs"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Fixed GA / CV parameters  (sign_flip flags 移到 mode 層級控制)
# ---------------------------------------------------------------------------
FIXED_PARAMS=(
    --config                "${CONFIG}"
    --n_splits              10
    --n_repeats             5
    --cv_seed               42
    --GA_Phase1_iterations  500
    --GA_Phase2_iterations  750
    --GA_Phase3_iterations  250
    --pop_size              1500
    --runs                  50
    --Initial_mutation_rate 0.5
    --s_tier_af_diff        0.05
    --m_tier_af_diff        0.25
    --adaptive_scale \
    --batch_limit           12800
    --n_genes               100
    --n_snps_min            5
    --n_snps_max            100
    --l_effect_ratio        0.3
    --m_effect_ratio        0.7
    --seed                  42
)

# ---------------------------------------------------------------------------
# Mode configurations:
#   "sign_flip_prob  dir_name  out_range(0/1)  outlier_scaling  use_sign_flip(0/1)"
# ---------------------------------------------------------------------------
MODE_CONFIGS=(
    "0.05  SimuFlip_NoFlipGA      0  2.0  0"
    "0.05  SimuFlip_FlipGA    0  2.0  1"
    #"0.05    SimuFlipOutRange_NoFlipGA    1  2.0 0"
    #"0.05    SimuFlipOutRange_FlipGA    1  2.0 1"
)

# ---------------------------------------------------------------------------
# Ratio combinations: "id sm med lg  label"
# ---------------------------------------------------------------------------
RATIO_CONFIGS=(
    "0.0 0.20 0.60 0.20  medium_heavy"
    "0.0 0.33 0.34 0.33  fully_balanced"
    #"0.25 0.25 0.25 0.25  fully_balanced"
    #"0.60 0.20 0.10 0.10  identical_heavy"
    "0.0 0.60 0.20 0.20  small_heavy"
    "0.0 0.20 0.20 0.60  large_heavy"
)

#SAMPLE_SIZES=(50 100 150 200 250)
SAMPLE_SIZES=(100 200)

# ---------------------------------------------------------------------------
TOTAL=$(( ${#MODE_CONFIGS[@]} * ${#RATIO_CONFIGS[@]} * ${#SAMPLE_SIZES[@]} ))
RUN_IDX=0

echo "================================================================"
echo " GA Simulation Grid"
echo " Modes         : ${#MODE_CONFIGS[@]}"
echo " Ratio configs : ${#RATIO_CONFIGS[@]}"
echo " Sample sizes  : ${SAMPLE_SIZES[*]}"
echo " Total runs    : ${TOTAL}"
echo "================================================================"

for MODE_ENTRY in "${MODE_CONFIGS[@]}"; do
    read -r SFP DIR_NAME OUT_RANGE OUTLIER_SCALE USE_SIGN_FLIP <<< "${MODE_ENTRY}"
    MODE_OUT="${BASE_OUT}/${DIR_NAME}"
    mkdir -p "${MODE_OUT}"

    # Build out_range flags
    if [ "${OUT_RANGE}" -eq 1 ]; then
        OR_FLAGS=(--out_range --outlier_scaling "${OUTLIER_SCALE}")
    else
        OR_FLAGS=(--no-out_range)
    fi

    # Build sign_flip flags
    if [ "${USE_SIGN_FLIP}" -eq 1 ]; then
        SF_FLAGS=(--allow_sign_flip --sign_flip_threshold 0.1)
    else
        SF_FLAGS=()
    fi

    echo ""
    echo "════ MODE: ${DIR_NAME}  (sfp=${SFP}, out_range=${OUT_RANGE}, sign_flip=${USE_SIGN_FLIP}) ════"

    for RATIO_ENTRY in "${RATIO_CONFIGS[@]}"; do
        read -r ID SM MED LG LABEL <<< "${RATIO_ENTRY}"

        for N_SAMPLES in "${SAMPLE_SIZES[@]}"; do
            RUN_IDX=$(( RUN_IDX + 1 ))
            RUN_TAG="run${RUN_IDX}_${DIR_NAME}_${LABEL}_S${N_SAMPLES}"
            LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"

            echo "---[ ${RUN_IDX}/${TOTAL} ] ${RUN_TAG}"

            python "${SIM_SCRIPT}"         \
                "${FIXED_PARAMS[@]}"       \
                "${OR_FLAGS[@]}"           \
                "${SF_FLAGS[@]}"           \
                --out_dir          "${MODE_OUT}"  \
                --n_samples        "${N_SAMPLES}" \
                --sign_flip_prob   "${SFP}"       \
                --ratio_identical  "${ID}"        \
                --ratio_small      "${SM}"        \
                --ratio_medium     "${MED}"       \
                --ratio_large      "${LG}"        \
                2>&1 | tee "${LOG_FILE}"
        done
    done
done

echo ""
echo "================================================================"
echo " All ${TOTAL} runs complete."
echo " Results: ${BASE_OUT}/"
echo " Logs   : ${LOG_DIR}/"
echo "================================================================"