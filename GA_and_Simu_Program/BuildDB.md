# BuildDB.py 說明文件

## 1. 這是做什麼的

`BuildDB.py` 是 `GA.py` GA 校正流程的**後處理工具**，用途是：

1. 把 `GA.py` 每個基因資料夾下各自產生的 `optimized_weights_detail.csv` 全部整合成**一個 SQLite 權重資料庫（.db）**，方便之後用 SQL 查詢、下游程式讀取，不用再一個一個資料夾去翻 CSV。
2. （可選）在整合的同時，根據 `GA.py` 本身已經算出來的指標，幫每一個 SNP 權重加上一個 **可信度分數 (`confidence_score`)** 與一句**建議 (`suggestion`)**，讓使用端知道哪些權重可以直接信任、哪些該人工複查或考慮改用原始 PrediXcan 權重。
3. （可選）如果有原始 PrediXcan model `.db`（例如 `gtex_v7_Lung_imputed_europeans_tw_0_5_signif.db`），可以在整合的**同一次執行**順便把 `chr`、`pos`、`ref_allele`、`eff_allele` 補進來——這些是之後拿基因型 (VCF) 做表現量預測時，判斷等位基因方向、避免權重正負號算反所必要的資訊。

整個過程**不會重跑 GA、不需要 GPU**，純粹是讀取既有輸出檔案做整理與統計，執行很快。

---

## 2. 輸入與輸出

### 輸入

| 檔案 | 說明 | 必要性 |
|---|---|---|
| `<base_dir>/<GENE_ID>/optimized_weights_detail.csv` | `GA.py` 每個基因輸出的 SNP 層級權重明細 | 必要 |
| `<out_dir>/final_summary.csv` | `GA.py` 整批彙總的基因層級表現指標（`Correlation after GA`、`train_corr`、`weight_stability` 等） | 選用，但**強烈建議**在算可信度分數時提供，否則基因層級的資訊會用中性值 0.5 代替 |
| 原始 PrediXcan model `.db`（例如 `gtex_v7_Lung_imputed_europeans_tw_0_5_signif.db`） | 標準 PrediXcan model db，`weights` 表需含 `rsid, varID, ref_allele, eff_allele` 欄位 | 選用，但**強烈建議**提供，之後做基因型預測時需要 `ref_allele`/`eff_allele` 判斷等位基因方向 |

### 輸出（SQLite `.db`，內含 1～2 張表）

| 表名 | 內容 | 何時產生 |
|---|---|---|
| `weights`（或 `--table_name` 指定的名稱） | 整合後的 SNP 層級權重表，每列 = 一個基因的一個 SNP | 一定會產生 |
| `gene_summary` | 原封不動存一份 `final_summary.csv`（基因層級指標），方便之後 `JOIN` | 只有同時給 `--summary_csv` 且加 `--score` 時才會產生 |

`weights` 表欄位（未加 `--score`、未加 `--predixcan_db`）：

```
Gene, rsid, EUR_AF, EAS_AF, AF_Diff, PrediXcan_W,
Allele_Factor, GA_Scaling_Mean, GA_Scaling_Std, GA_W
```

加了 `--predixcan_db` 之後，還會多出：

```
chr, pos, ref_allele, eff_allele   # 從原始 PrediXcan db 的 varID 拆解 + 直接讀取
```

> `chr`/`pos` 是從 `varID`（格式 `<chr>_<pos>_<ref>_<alt>_<build>`，例如 `1_1026707_C_A_b37`）拆解出來的。若原始 db 沒有 `varID` 欄位，只會補 `ref_allele`/`eff_allele`，`chr`/`pos` 留空並印警告。找不到對應 rsid 的列會是 `NULL`，不會中斷程式，並在執行時印出「幾筆有 match、幾筆沒有」的統計。

加了 `--score` 之後，還會多出：

```
Correlation after GA, train_corr, train_corr_std,
weight_stability, weight_stability_penalty,   # 來自 final_summary.csv
confidence_score, is_pruned_zero, suggestion  # 新算出來的欄位
```

> 預設會把 `source` 欄位丟掉（因為每列都固定是 `GA`，沒有資訊量）。可用 `--drop_cols` 調整。

---

## 3. 可信度分數是怎麼算的

`confidence_score`（0～1，越高越可信）是把下面四個「`GA.py` 早就算好、只是散落在不同檔案」的訊號加權組合而成，**不需要額外運算或重跑 GA**：

| 訊號 | 來源欄位 | 邏輯 |
|---|---|---|
| SNP 在各 fold 間的縮放係數穩定度 | `GA_Scaling_Std / |GA_Scaling_Mean|`（變異係數） | LOOCV 不同 fold 選到的縮放差異越大 → 越不穩定 → 分數越低 |
| 族群可轉移性 | `AF_Diff` | EUR/EAS 等位基因頻率差異越大（GA.py 自己拿來分 S/M/L tier 的依據）→ 分數越低 |
| 是否被剪枝 | `GA_W == 0` | GA 已判定此 SNP 貢獻可忽略 → 直接標記為 `is_pruned_zero`，不併入連續分數，另外給建議 |
| 整體基因模型表現與穩定度 | `final_summary.csv` 的 `Correlation after GA` / `train_corr` / `weight_stability` | 就算單一 SNP 很穩定，若該基因整體校正效果差、fold 間不穩，也連帶降低信心 |

四個分量各自做 min-max 正規化到 [0,1] 後，依權重 `--w_cv / --w_af / --w_gene / --w_zero`（程式內設定，預設 0.35 / 0.20 / 0.35 / 0.10）加總。

### 3.1 完整公式

```
confidence_score = w_cv × cv_score
                  + w_af × af_score
                  + w_gene × gene_component
                  − w_zero × zero_penalty

（結果 clip 到 [0, 1] 之間；預設 w_cv=0.35, w_af=0.20, w_gene=0.35, w_zero=0.10）
```

各分量的實際計算方式：

**① `cv_score`（SNP 縮放係數穩定度）**
```python
cv = GA_Scaling_Std / (|GA_Scaling_Mean| + eps)
cv_clipped = cv.clip(upper=cv 的 99% 分位數)   # 避免極端值把整體分數壓垮
cv_score = 1 − minmax_normalize(cv_clipped)
```

**② `af_score`（族群可轉移性）**
```python
af = AF_Diff.clip(0, 1)
af_score = 1 − af
```

**③ `zero_penalty`（剪枝旗標，直接扣分而非連續分數）**
```python
is_zero = (|GA_W| < 1e-12)
zero_penalty = 1.0 if is_zero else 0.0
```

**④ `gene_component`（基因整體表現與穩定度，內部又是兩個子分量的組合）**
```python
gene_component = 0.6 × gene_score + 0.4 × instab_score
```
- `gene_score` = `minmax_normalize(Correlation after GA)`（若欄位不存在則退而用 `train_corr`）
- `instab_score` = `1 − minmax_normalize(weight_stability_penalty)`（若欄位不存在則退而用 `weight_stability`）
- 若**沒有提供** `--summary_csv`：整個 `gene_component` 直接給中性值 `0.5`，不加分也不扣分，避免在缺資料時誤導排序。

### 3.2 為什麼是這個權重比例

設計邏輯是：**SNP 自身的統計穩定度**（`cv_score` 0.35 + `af_score` 0.20 = 0.55）比重略高於**它所屬基因的整體表現**（`gene_component` 0.35），因為這個分數主要是要回答「這個特定權重數值靠不靠得住」，而不是重複評估基因整體模型好壞——那件事 `final_summary.csv` 本身的 `Improvement` / `Impact_Category` 已經在做了。剪枝旗標（`w_zero` 0.10）只是一個明確的提示，不需要占太大比重。

這組權重是依現有 pipeline 欄位設計的**合理起點，不是統計上驗證過的最佳解**。現在已開放成 CLI 參數 `--w_cv` / `--w_af` / `--w_gene` / `--w_zero`（用法見 4.3、4.5），例如：
- 更在意「基因整體轉移成不成功」→ `--w_gene 0.5 --w_cv 0.25 --w_af 0.15`
- 不想讓已剪枝的 SNP 被額外扣分（反正 `suggestion` 已經標出來了）→ `--w_zero 0`
- 完全不看基因層級（等同不給 `--summary_csv`，但語意更明確）→ `--w_gene 0`

> `w_zero` 是用**扣分**的方式套用，其餘三個是**加分**，所以四個權重不強制要求總和為 1；但若想維持分數 0~1 好解讀，建議讓 `w_cv + w_af + w_gene ≈ 1`。

**`suggestion` 標籤對照：**

| 條件 | 建議文字 |
|---|---|
| `GA_W == 0`（已被剪枝） | Pruned (weight=0) — excluded by GA, no action needed |
| `confidence_score >= 0.75` | High confidence — use GA weight as-is |
| `0.5 <= score < 0.75` | Moderate confidence — usable, worth spot-checking |
| `0.3 <= score < 0.5` | Low confidence — review manually before use |
| `score < 0.3` | Very low confidence — consider reverting to PrediXcan_W or excluding SNP |

> ⚠️ 分數是**相對於這次整合進來的整個資料集**做正規化的，所以：
> - 建議把要互相比較的所有基因**一次整合完再算分**（例如整個 `V4_LOOCV_Sign` 資料夾），不要切成好幾批分開算，否則分數不能互相比較。
> - 這是根據既有欄位設計的實用起點，不是正式統計檢定；若要用於正式報告/發表，建議再跟熟悉的統計方法（如 Fisher's z 信賴區間、bootstrap）交叉驗證。

---

## 4. 使用方式

### 環境需求

```bash
pip install pandas numpy --break-system-packages   # 若已安裝可跳過
```

### 4.1 只整合、不評分

```bash
python3 BuildDB.py \
    --base_dir "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db   "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db"
```

### 4.2 整合 + 加上可信度分數與建議（建議用法）

```bash
python3 BuildDB.py \
    --base_dir    "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db      "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db" \
    --summary_csv "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/final_summary.csv" \
    --score
```

### 4.2.1 整合 + 評分 + 補齊等位基因資訊（最完整、最建議的用法）

如果手邊有原始 PrediXcan model `.db`，建議直接一起帶入，之後要用這個 `weights.db` 對基因型資料做表現量預測時就不用再另外處理：

```bash
python3 BuildDB.py \
    --base_dir     "/mnt/data_40T/anywaycheng/UploadFile/GA/V4_10Fold_Sign" \
    --out_db       "/mnt/data_40T/anywaycheng/UploadFile/GA/GA_Weight_DataBase/V4_10Fold_Sign_weights.db" \
    --summary_csv  "/mnt/data_40T/anywaycheng/UploadFile/GA/V4_10Fold_Sign/final_summary.csv" \
    --predixcan_db "/path/to/gtex_v7_Lung_imputed_europeans_tw_0_5_signif.db" \
    --score
```

### 4.3 全部參數說明

| 參數 | 必要 | 預設值 | 說明 |
|---|---|---|---|
| `--base_dir` | 是 | — | 每個基因一個子資料夾的上層路徑（例如 `V4_LOOCV_Sign`） |
| `--out_db` | 是 | — | 輸出的 SQLite `.db` 路徑 |
| `--filename` | 否 | `optimized_weights_detail.csv` | 每個基因資料夾內要讀取的檔名 |
| `--summary_csv` | 否 | `None` | `final_summary.csv` 路徑，提供基因層級指標；加 `--score` 時強烈建議一併提供 |
| `--predixcan_db` | 否 | `None` | 原始 PrediXcan model `.db` 路徑；提供的話會補齊 `chr`/`pos`/`ref_allele`/`eff_allele` 欄位 |
| `--score` | 否 | 關閉 | 加這個 flag 才會計算 `confidence_score` / `suggestion` |
| `--table_name` | 否 | `weights` | SNP 權重表在 DB 裡的表名，方便同一個 db 存多個結果（如 Sign / Unsign）時區分 |
| `--drop_cols` | 否 | `source` | 整合後要丟棄的欄位，逗號分隔；傳空字串 `""` 表示全部保留 |
| `--w_cv` | 否 | `0.35` | 可信度分數中「SNP 縮放係數穩定度」的權重（只有 `--score` 時生效） |
| `--w_af` | 否 | `0.20` | 可信度分數中「族群可轉移性 AF_Diff」的權重 |
| `--w_gene` | 否 | `0.35` | 可信度分數中「基因整體表現與穩定度」的權重，需搭配 `--summary_csv` 才有效果 |
| `--w_zero` | 否 | `0.10` | 剪枝扣分（`GA_W == 0` 時從分數中扣掉的量） |

### 4.4 常見情境

**同時整合 LOOCV Sign 和 Unsign 兩批結果到同一個 db（用不同表名區分）：**

```bash
python3 BuildDB.py \
    --base_dir "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db   "/mnt/data_80T/anywaycheng/GA/5%_Result/all_weights.db" \
    --summary_csv "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/final_summary.csv" \
    --score --table_name weights_sign

python3 BuildDB.py \
    --base_dir "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Unsign" \
    --out_db   "/mnt/data_80T/anywaycheng/GA/5%_Result/all_weights.db" \
    --summary_csv "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Unsign/final_summary.csv" \
    --score --table_name weights_unsign
```

**保留 `source` 欄位，不要丟：**

```bash
python3 BuildDB.py --base_dir ... --out_db ... --drop_cols ""
```

### 4.5 自訂可信度分數權重

```bash
# 更看重基因整體表現，SNP 自身穩定度權重降低
python3 BuildDB.py \
    --base_dir    "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign" \
    --out_db      "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db" \
    --summary_csv "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/final_summary.csv" \
    --score --w_cv 0.25 --w_af 0.15 --w_gene 0.5 --w_zero 0.10

# 已剪枝的 SNP 不額外扣分
python3 BuildDB.py --base_dir ... --out_db ... --summary_csv ... --score --w_zero 0
```

執行時終端機會印出 `[INFO] Scoring weights: w_cv=..., w_af=..., w_gene=..., w_zero=...`，方便確認這次跑的是哪組權重。

---

## 5. 執行後怎麼檢查結果

```bash
sqlite3 "/mnt/data_80T/anywaycheng/GA/5%_Result/V4_LOOCV_Sign/weights.db"
```

```sql
.tables                                                  -- 看有哪些表
.schema weights                                          -- 看欄位結構

SELECT COUNT(DISTINCT Gene) FROM weights;                -- 整合了幾個基因

SELECT Gene, rsid, GA_W, confidence_score, suggestion
FROM weights
WHERE confidence_score < 0.3;                             -- 找出低可信度的權重

SELECT suggestion, COUNT(*)
FROM weights
GROUP BY suggestion;                                       -- 各建議類別的數量分布

SELECT w.Gene, w.rsid, w.confidence_score, g."Correlation after GA"
FROM weights w
JOIN gene_summary g ON w.Gene = g.gene_id
ORDER BY w.confidence_score ASC
LIMIT 20;                                                   -- 跨表查詢：最不可信的 20 個 SNP 連同其基因表現

SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN eff_allele IS NULL THEN 1 ELSE 0 END) AS missing_allele_info
FROM weights;                                               -- 檢查 --predixcan_db 補齊的覆蓋率（若有使用該參數）
```

也可以用 Python 直接讀：

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("weights.db")
df = pd.read_sql("SELECT * FROM weights", conn)
```
