# GeneExpression_Predict.md — GA 校正基因表現量預測流程

## 1. 這是做什麼的

這是 `GA.py` GA 校正流程的**下游應用**：拿一份使用者的基因型檔案 `genotype.vcf`，套用 `V4_3Fold_Sign_weights.db` 裡 GA 校正過的權重（`GA_W`），依照 **PrediXcan 的預測公式**算出每個樣本、每個基因的預測表現量。

```
PredictedExpression(gene, sample) = Σ_i  weight(gene, SNP_i) × dosage(SNP_i, sample)
```

`dosage` 是該樣本在該 SNP 上 effect allele 的個數（0/1/2）。這跟原版 PrediXcan 用的公式完全一樣，差別只在於這裡的 `weight` 是 GA 校正過、更適合轉移到 EAS 族群的版本，而不是原始針對 EUR 族群訓練的權重。

## 2. 三個檔案的分工

| 檔案 | 角色 |
|---|---|
| **`GeneExpression_Predict.sh`** | 使用者唯一需要執行的入口。輸入一份 `genotype.vcf`，自動完成「萃取需要的 SNP → 用 plink2 算 dosage → 呼叫 GeneExpression_Predict.py 算預測值」全部三步驟。 |
| **`GeneExpression_Predict.py`** | 實際運算核心。讀 plink2 產生的 dosage 矩陣 + `weights.db`，做等位基因方向校正與稀疏矩陣相乘，輸出預測表現量與 summary。一般不需要直接呼叫，由 `GeneExpression_Predict.sh` 自動帶參數呼叫。 |
| **`GeneExpression_Predict.md`** | 這份文件。 |

## 3. 使用方式

### 環境需求

- **plink2**（用來從 VCF 快速萃取只需要的 SNP dosage，需自行安裝並可從 `$PATH` 執行，或修改 `GeneExpression_Predict.sh` 裡的 `PLINK2` 變數指到執行檔路徑）
- **sqlite3** CLI
- Python 套件：`pip install pandas numpy scipy --break-system-packages`

### 執行

```bash
chmod +x GeneExpression_Predict.sh
./GeneExpression_Predict.sh genotype.vcf
```

指定輸出資料夾（不指定則預設 `./predixcan_ga_output`）：

```bash
./GeneExpression_Predict.sh genotype.vcf /path/to/my_output_dir
```

**使用者只需要提供這一份 `genotype.vcf`**，其餘（`weights.db` 路徑、plink2 參數、權重欄位）都已經寫在 `GeneExpression_Predict.sh` 最上面的 `CONFIG` 區塊裡，預設指向 `V4_3Fold_Sign_weights.db`。如果要換成別的 `.db`（例如 `V4_LOOCV_Sign_weights.db`）或想比較「GA 校正前 vs 校正後」，改這幾行就好：

```bash
WEIGHTS_DB="/mnt/data_40T/anywaycheng/UploadFile/GA/GA_Weight_DataBase/V4_3Fold_Sign_weights.db"
WEIGHT_COL="GA_W"        # 改成 PrediXcan_W 可以拿校正前的權重做對照組
```

### 執行過程做了什麼

1. **[1/3]** 從 `weights.db` 撈出所有非零權重用到的 rsid 清單（只有幾千~幾萬個，不是全基因組）
2. **[2/3]** `plink2 --extract` 只針對這份清單從 `genotype.vcf` 算出 dosage 矩陣——這是控制運算時間的關鍵，plink2 完全不會碰到跟模型無關的變異
3. **[3/3]** `GeneExpression_Predict.py` 做等位基因方向校正（比對 `weights.db` 的 `eff_allele` 跟 plink2 計數的等位基因，方向不同就用 `2 − dosage`）,接著用一次稀疏矩陣相乘 `W @ D` 算出所有基因、所有樣本的預測值

## 4. 輸出檔案格式

### 4.1 `predicted_predicted_expression.csv`（對應 PrediXcan `*_predicted_expression.txt`）

**寬格式**，跟 PrediXcan 官方 `Predict.py` 輸出版面一致：每列一個樣本，每欄一個基因。

| FID | IID | ENSG00000000971 | ENSG00000001167 | ... |
|---|---|---|---|---|
| 1 | S0 | 0.1074 | -0.1747 | ... |
| 1 | S1 | 0.0480 | 0.0341 | ... |

- `FID`/`IID`：來自 VCF/plink2 的樣本 ID
- 之後每欄是一個基因 ID，數值是該樣本的預測表現量（原始尺度，總和公式算出來的值；如果需要標準化 z-score，可以自行對每欄做 `(x - mean) / std`）
- **只會出現「至少有 1 個 SNP 在這份 VCF 中被找到」的基因**；完全沒有任何 SNP 命中的基因不會出現在這個檔案裡（去 `predicted_summary.csv` 查它的 `status` 會是 `NO_SNPS_IN_VCF`）

### 4.2 `predicted_summary.csv`（對應 PrediXcan `*_summary.txt`）

**每列一個基因**，涵蓋 `weights.db` 裡所有基因（不只是預測成功的），欄位設計參考 PrediXcan 官方 summary 輸出的 QC 邏輯，並額外加上 GA pipeline 特有的品質資訊：

| 欄位 | 說明 | 來源 |
|---|---|---|
| `gene` | 基因 ID | — |
| `n_snps_in_model` | GA 校正後、非零權重的 SNP 數（即 GA 判定真正有貢獻的 SNP 集合） | `weights.db` |
| `n_snps_used` | 這次預測中，實際在 `genotype.vcf` 找到並成功對齊方向的 SNP 數 | 本次執行計算 |
| `pct_snps_used` | `n_snps_used / n_snps_in_model`，代表這個基因的模型在這份 VCF 上的覆蓋率 | 本次執行計算 |
| `pred_mean` / `pred_sd` | 這批樣本間預測表現量的平均與標準差 | 本次執行計算 |
| `var_g` | 預測表現量的變異數（`pred_sd²`）；PrediXcan 標準 QC 指標，`var_g = 0` 代表這個基因在這批樣本裡完全沒有鑑別度 | 本次執行計算 |
| `avg_confidence_score` | 實際用到的 SNP 們，在 `BuildDB.py` 算出的 `confidence_score` 平均值 | `weights.db`（GA pipeline 特有） |
| `train_corr` / `corr_after_GA` | 該基因在 GA 訓練/驗證階段的整體表現（來自 `final_summary.csv`），可視為這個基因模型本身可不可信的參考指標 | `weights.db` 的 `gene_summary` 表（GA pipeline 特有） |
| `n_snps` | 該基因在 GA 訓練時使用的原始 SNP 總數（含後來被剪枝的） | `gene_summary` |
| `status` | 這個基因這次預測的品質旗標，見下表 | 本次執行計算 |

**`status` 對照：**

| status | 意義 |
|---|---|
| `OK` | 正常，`n_snps_used` 覆蓋率 ≥ 50% |
| `LOW_SNP_COVERAGE` | 有找到部分 SNP，但覆蓋率 < 50%，預測值可信度較低，建議謹慎使用 |
| `NO_SNPS_IN_VCF` | `weights.db` 裡這個基因的模型 SNP，完全沒有一個出現在這份 VCF（常見原因：VCF 變異 ID 不是 rsid、或這份 VCF 沒涵蓋到那些位點），這個基因**不會**出現在 `predicted_predicted_expression.csv` 裡 |
| `NO_MODEL_SNPS` | 這個基因在 GA 階段就被全部剪枝（`n_snps_in_model = 0`），本來就沒有可用權重，跟這次的 VCF 無關 |
| `ZERO_VARIANCE` | 有算出預測值，但在這批樣本裡所有人數值都一樣（`var_g = 0`），這個基因在這批樣本裡沒有鑑別度，PrediXcan 慣例上這類基因通常會被下游分析濾掉 |

> 建議下游分析先篩 `status == 'OK'`（或視情況放寬到含 `LOW_SNP_COVERAGE`）的基因再使用，避免把覆蓋率極低或無鑑別度的預測值當成可信結果。

## 5. 效能與運算時間

- **只處理需要的 SNP**：Step 1/2 讓 plink2 完全跳過跟模型無關的變異，資料量從「全基因組」壓到「模型實際用到的幾千~幾萬個 SNP」。
- **單次稀疏矩陣相乘**：不是逐基因跑迴歸，`GeneExpression_Predict.py` 是一次 `(基因 × SNP) 稀疏矩陣 @ (SNP × 樣本) 矩陣`，對近 5000 個基因、上千樣本來說是秒級運算。
- 整體瓶頸通常在 plink2 讀取 VCF 那一步，跟 VCF 檔案大小 / 樣本數成正比，跟基因數量無關（因為 SNP 清單已經固定）。

## 6. 常見情境

**同時輸出「校正前 vs 校正後」兩份預測值做比較：**

```bash
# 校正後（預設）
./GeneExpression_Predict.sh genotype.vcf output_ga

# 改用未校正的 PrediXcan_W 做對照組
sed 's/WEIGHT_COL="GA_W"/WEIGHT_COL="PrediXcan_W"/' GeneExpression_Predict.sh > GeneExpression_Predict_predixcan.sh
chmod +x GeneExpression_Predict_predixcan.sh
./GeneExpression_Predict_predixcan.sh genotype.vcf output_predixcan
```

**VCF 變異 ID 不是 rsid（`--extract` 會抓不到東西）：**

```bash
bcftools annotate -c ID -a dbSNP.vcf.gz genotype.vcf.gz -Oz -o genotype.rsid.vcf.gz
gunzip -k genotype.rsid.vcf.gz   # 若需要未壓縮的 .vcf
./GeneExpression_Predict.sh genotype.rsid.vcf
```

**只想手動重跑 Step 3（已經有 dosage_matrix.raw，不想重跑 plink2）：**

```bash
python3 GeneExpression_Predict.py \
    --dosage_raw predixcan_ga_output/tmp/dosage_matrix.raw \
    --weights_db "/mnt/data_40T/anywaycheng/UploadFile/GA/GA_Weight_DataBase/V4_3Fold_Sign_weights.db" \
    --out_prefix predixcan_ga_output/predicted
```
