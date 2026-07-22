# Simu_Result_Analysis.ipynb — 技術文件

## 概述

本 notebook 針對遺傳演算法 (GA) 優化 PrediXcan 基因表現預測模型的模擬結果進行完整分析。主要比較 GA 優化前後的 LOOCV（Leave-One-Out Cross-Validation）相關係數變化，以及 MAE 改善程度，並評估模擬參數（SNP tier 比例、樣本數、Flip/Unflip 條件）對結果的影響。

---

## 資料來源

### 目錄結構

```
D:\GA - 複製\Simulation\Simulation\
├── SimuFlip_FlipGA\       # GA 執行時啟用 sign flip
│   └── id{id}_sm{sm}_med{med}_lg{lg}_sfp{sfp}_S{n}\
│       ├── Final_Simulation_Report.csv
│       ├── Simulated_SNP_Details.csv
│       ├── Simulated_Expression_GroundTruth.csv
│       ├── Tier_Summary.csv
│       └── Simulation_Summary_Detail.png
└── SimuFlip_NoFlipGA\     # GA 執行時停用 sign flip
    └── id{id}_sm{sm}_med{med}_lg{lg}_sfp{sfp}_S{n}\
        └── （同上）
```

### 資料夾命名規則

| 參數     | 說明                           | 範例           |
| -------- | ------------------------------ | -------------- |
| `id`     | indel 比例                     | `id0.0`        |
| `sm`     | small-effect SNP 比例          | `sm0.2`        |
| `med`    | medium-effect SNP 比例         | `med0.6`       |
| `lg`     | large-effect SNP 比例          | `lg0.2`        |
| `sfp`    | sign flip probability          | `sfp0.05`      |
| `S`      | 樣本數 (50 / 100 / 150 / 200 / 250) | `S50`     |

### 輸入檔案

| 檔案                              | 內容                                      |
| --------------------------------- | ----------------------------------------- |
| `Final_Simulation_Report.csv`     | 基因層級統計（每條記錄代表一個基因）      |
| `Simulated_SNP_Details.csv`       | SNP 層級詳細資料                          |
| `ga_run_optimized.log`            | GA 執行 log（含 LOOCV 相關係數逐基因記錄）|
| `final_summary.csv`               | GA 最終摘要（含 MAE、weight stability）   |

---

## 實驗條件

### Category 分類邏輯

```python
def get_category(id_r, sm_r, med_r, lg_r):
    dominant_val = max(sm_r, med_r, lg_r)
    if dominant_val < 0.5:
        return 'average'         # 三者均衡
    dominant = max({'small': sm_r, 'medium': med_r, 'large': lg_r}.items())
    return f"{dominant[0]}_heavy"
```

| Category       | 說明                         |
| -------------- | ---------------------------- |
| `average`      | 三種 tier 比例相近（< 0.5）  |
| `small_heavy`  | small-effect SNP 佔多數      |
| `medium_heavy` | medium-effect SNP 佔多數     |
| `large_heavy`  | large-effect SNP 佔多數      |

### 交叉條件

- **Flip 條件**：`Flip` / `Unflip`（各 20 個資料夾）
- **樣本數**：50, 100, 150, 200, 250
- **Category**：average, small_heavy, medium_heavy, large_heavy
- **總計**：40 個條件，約 101,614 SNP 行

---

## Notebook 結構

### Cell 1 — 載入 Final_Simulation_Report

彙整所有條件的 `Final_Simulation_Report.csv`，輸出到：
```
D:\GA - 複製\All_Simulation_Results.csv
```

**衍生欄位：** `Category`, `Flip`, `N_samples`, `sm_ratio`, `med_ratio`, `lg_ratio`, `folder`

---

### Cell 2 — 載入 Simulated_SNP_Details

彙整所有條件的 `Simulated_SNP_Details.csv`，輸出到：
```
D:\GA - 複製\All_SNP_Level_Results.csv
```

---

### Cell 3 — 設定 LOG_PATH

指定要分析的 GA log 路徑（目前預設為 LOOCV_Unsign）：

```python
LOG_PATH = Path(r"D:\GA - 複製\LOOCV_Unsign\ga_run_optimized.log")
```

> **切換方法：** 修改 `LOG_PATH` 後執行 `Kernel → Restart & Run All`

---

### Cell 4 — 匯入套件

```python
import re, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
```

---

## 分析章節

### § 1 — Log Parsing（GA Log 解析）

使用正規表達式逐行解析 `ga_run_optimized.log`，擷取以下欄位：

| 欄位              | 說明                                      |
| ----------------- | ----------------------------------------- |
| `baseline_corr`   | PrediXcan baseline 相關係數               |
| `before`          | GA 執行前 LOOCV 相關係數                  |
| `after`           | GA 執行後 LOOCV 相關係數                  |
| `train`           | GA in-sample train 相關係數               |
| `sign_flip`       | 是否發生 sign flip                        |
| `snp_orig`        | TAS filter 前 SNP 數                      |
| `snp_tas`         | TAS filter 後 SNP 數                      |
| `adaptive_scale`  | 自適應縮放係數                            |
| `time_s`          | 執行時間（秒）                            |

**Outcome 分類規則：**

| Outcome      | 條件                          |
| ------------ | ----------------------------- |
| `Skipped`    | 跳過處理                      |
| `Neg_Strong` | `after ≤ -0.3`                |
| `Neg_Weak`   | `-0.3 < after < 0`            |
| `Improved`   | `delta ≥ 0.1`                 |
| `Worsened`   | `delta ≤ -0.1`                |
| `Neutral`    | 其他（-0.1 < delta < 0.1）    |

**實際結果（LOOCV_Unsign）：**

| Outcome    | Count | Pct (%) |
| ---------- | ----- | ------- |
| Neutral    | 2201  | 46.0    |
| Neg_Weak   | 1422  | 29.7    |
| Improved   | 708   | 14.8    |
| Worsened   | 298   | 6.2     |
| Neg_Strong | 153   | 3.2     |

---

### § 2 — 整體分布概覽

輸出 6 個直方圖（`fig_01_distributions.png`）：

- `baseline_corr`：PrediXcan 基礎相關係數
- `before`：GA 前 LOOCV
- `after`：GA 後 LOOCV
- `train`：Train 相關係數（in-sample）
- `delta`：After − Before（GA 貢獻量）
- `train_gap`：Train − After（過擬合程度）

**關鍵統計（4,782 個有效基因）：**

| 指標           | Mean   | Median | Std   |
| -------------- | ------ | ------ | ----- |
| baseline\_corr | 0.103  | 0.091  | 0.204 |
| after          | 0.094  | 0.090  | 0.215 |
| delta          | -0.008 | -0.002 | 0.141 |
| train\_gap     | 0.096  | 0.074  | 0.089 |
| snp\_orig      | 15.1   | 13     | 9.1   |
| snp\_tas       | 13.3   | 11     | 7.9   |

---

### § 1.5 — 合併 final_summary.csv

從 `final_summary.csv` 合併下列欄位至主 DataFrame（`df_merged`）：

| 欄位                      | 說明                              |
| ------------------------- | --------------------------------- |
| `MAE_PrediXcan_Real`      | PrediXcan 的 MAE（真實資料）      |
| `MAE_GA_Real`             | GA 優化後的 MAE（真實資料）       |
| `Improvement`             | MAE 改善量（PrediXcan − GA）      |
| `weight_stability`        | GA SNP weight 穩定性指標          |
| `train_corr_std`          | train correlation 標準差          |
| `Impact_Category`         | 以 MAE 改善量為基礎的影響分類     |

**衍生欄位：**
```python
df_merged['mae_delta'] = df_merged['MAE_PrediXcan_Real'] - df_merged['MAE_GA_Real']
```

**Impact_Category 分布：**

| 類別                            | 基因數 | 比例 (%) |
| ------------------------------- | ------ | -------- |
| No Impact (-0.1~0.1)            | 3023   | 63.2     |
| Worsened (< -0.1)               | 925    | 19.3     |
| Slight Improvement (0.1~0.3)    | 761    | 15.9     |
| Moderate Improvement (0.3~0.6)  | 72     | 1.5      |
| Significant Improvement (> 0.6) | 1      | 0.0      |

---

### § 2b — SNP 數分布

輸出 `fig_02_snp_dist.png`：TAS filter 前後的 SNP 數直方圖。

---

### § 3 — 描述統計表

輸出兩張描述統計表：
1. 相關係數與 SNP 指標
2. MAE 與穩定性指標

---

### § 3b — Impact Category 分布圖（MAE-based）

輸出 `fig_R2_impact_category.png`：長條圖顯示各 Impact Category 的基因數與百分比。

---

### § 4 — Before vs After Scatter

輸出 `fig_03_scatter.png`：Before（PrediXcan）vs After（GA LOOCV）散點圖，以 Outcome 著色。

---

### § 4b — MAE 比較與 Corr/MAE 一致性（R4 & R5）

**R4**（`fig_R4_mae_comparison.png`）：
- Violin plot：各 Impact Category 下 PrediXcan 與 GA 的 MAE 分布對比
- Box plot：各 Impact Category 下的 MAE 改善量（Improvement）

**R5**（`fig_R5_concordance.png`）：
- Scatter：Corr Δ vs MAE Improvement，以 Impact Category 著色
- Bar：四象限分析（Both Improved / Corr↑ MAE Worse / Corr↓ MAE Better / Both Worsened）

> **結果：** Corr 與 MAE 方向一致率 **99.6%**，兩項指標高度同向。

---

### § 5 — 過擬合診斷

輸出 `fig_04_overfit.png`：
- Train vs LOOCV 散點圖（以 `train_gap` 著色）
- 各 Outcome 類別的 `train_gap` 密度分布

---

### § 6 — 區間別分析（After Corr Bins）

將 After 相關係數切為 8 個區間（`<-0.5`, `-0.5~-0.3`, ..., `>0.5`），逐區間統計：

- baseline_mean, before_mean, after_mean, train_mean
- train_gap_mean（過擬合程度，高 after 基因更小）
- snp_tas_mean / snp_tas_med
- delta_mean
- pct_sign_flip

輸出 `fig_05_violin_bins.png`（各區間 baseline / train / after violin）與 `fig_06_snp_bins.png`（各區間 SNP 數 boxplot）。

---

### § 7 — 問題基因深挖（After ≤ -0.3）

鎖定 155 個 After ≤ -0.3 的問題基因，分析：
- Before 相關係數分布（131/155 在執行前 Train < 0）
- Train 相關係數分布
- Before × Train 交叉表（heatmap）

輸出 `fig_07_neg_deep.png` 與 `fig_08_crosstab.png`。

---

### § 8 — Pre-filter 模擬

模擬若預先跳過 `|baseline_corr| < threshold` 的基因，對不同閾值（0.02 ～ 0.20）計算：

| 指標                    | 說明                          |
| ----------------------- | ----------------------------- |
| `n_skipped`             | 被跳過的基因數                |
| `lost_improved`         | 原本可改善卻被跳過的基因數    |
| `avoided_neg`           | 避免了 LOOCV < -0.1 的基因數  |
| `avoided_worse`         | 避免了 Δ < -0.05 的基因數     |
| `net_benefit`           | avoided_neg + avoided_worse − lost_improved |

輸出 `fig_09_prefilter_sim.png`（各閾值下 cost vs benefit 折線圖）。

---

### § 9 — SNP 數對結果的影響

將 `snp_tas` 切為 7 個區間（1-5, 6-10, ..., >50），逐區間統計：

- after_mean / after_median
- train_mean, gap_mean
- pct_neg（LOOCV < 0 的比例）
- pct_improved（Improved outcome 的比例）

**趨勢：** SNP 數越多，after_mean 越高、pct_neg 越低、pct_improved 越高（>50 SNP 例外，樣本數僅 11）。

---

## 輸出檔案彙整

| 檔案                         | 對應章節 | 說明                              |
| ---------------------------- | -------- | --------------------------------- |
| `fig_01_distributions.png`   | § 2      | 6 項指標整體分布直方圖            |
| `fig_02_snp_dist.png`        | § 2b     | SNP 數分布（TAS 前後）            |
| `fig_R2_impact_category.png` | § 3b     | Impact Category 長條圖            |
| `fig_03_scatter.png`         | § 4      | Before vs After scatter           |
| `fig_R4_mae_comparison.png`  | § 4b     | MAE 比較 violin + improvement box |
| `fig_R5_concordance.png`     | § 4b     | Corr × MAE 一致性分析             |
| `fig_04_overfit.png`         | § 5      | Train vs LOOCV 過擬合診斷         |
| `fig_05_violin_bins.png`     | § 6      | 各 After bin 的三指標 violin      |
| `fig_06_snp_bins.png`        | § 6      | 各 After bin 的 SNP 數 boxplot    |
| `fig_07_neg_deep.png`        | § 7      | 問題基因（After ≤ -0.3）深挖      |
| `fig_08_crosstab.png`        | § 7      | Before × Train 交叉表 heatmap     |
| `fig_09_prefilter_sim.png`   | § 8      | Pre-filter 閾值模擬折線圖         |

---

## 關鍵指標定義

| 指標              | 定義                                      |
| ----------------- | ----------------------------------------- |
| `delta`           | `after − before`（GA LOOCV 提升量）       |
| `train_gap`       | `train − after`（過擬合程度，越小越好）   |
| `mae_delta`       | `MAE_PrediXcan − MAE_GA`（正值 = 改善）   |
| `Improvement`     | 來自 final_summary.csv，同上              |
| `abs_baseline`    | `|baseline_corr|`（用於 pre-filter 模擬） |

---

## 執行環境

- Python 3.11+
- 套件：`pandas`, `numpy`, `matplotlib`, `seaborn`
- 建議在 JupyterLab 或 VS Code Jupyter extension 中執行
- 切換分析對象：修改 Cell 3 的 `LOG_PATH`，執行 `Kernel → Restart & Run All`
