# Data / 資料說明

## English

The training and testing input data files used by this project — genotype matrices, real/predicted expression tables, allele frequency tables, and annotation files — are too large to be stored on GitHub.

These files are stored on the internal server at:

```
/mnt/data_40T/anywaycheng/UploadFile/GA/
```

To run any of the scripts in this repository (`GA.py`, `GA_Simulation.py`, `BuildDB.py`, `GeneExpression_Predict.py`), point the relevant `config.json` fields (`genotype_path`, `predict_expr_path`, `real_expr_path`, `allele_freq_path`, `annotation_path`, `db_path`) or CLI arguments to the corresponding files under the path above.

Access to this path requires access to the internal server; contact the repository owner if you need the data.

---

## 中文

本專案使用的訓練與測試輸入資料檔案——包含基因型矩陣、真實/預測表現量表、等位基因頻率表、註解檔案——檔案過大，無法存放於 GitHub。

這些檔案存放於內部伺服器上的路徑：

```
/mnt/data_40T/anywaycheng/UploadFile/GA/
```

要執行本 repo 內的任何腳本（`GA.py`、`GA_Simulation.py`、`BuildDB.py`、`GeneExpression_Predict.py`），請將對應 `config.json` 欄位（`genotype_path`、`predict_expr_path`、`real_expr_path`、`allele_freq_path`、`annotation_path`、`db_path`）或 CLI 參數指向上述路徑下的對應檔案。

存取此路徑需要內部伺服器的權限；若需要這些資料，請聯絡 repo 擁有者。
