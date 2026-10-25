# Lab 2：Large Language Model Compression

學號：**315551049**

| 目錄 | 內容 | 狀態 |
|---|---|---|
| `Part1/StudentID_Part1.ipynb` | Linear quantization Q1–Q10 | 完成，7 個測試全部 `* Test passed.`（含 MobileNetV2 的直方圖與量化圖） |
| `Part2/` | Qwen3-4B 四個壓縮 case（GPTQ + RHT + 混合精度、白化 SVD、OBS 結構化剪枝） | 完成，四個 case 全數達標（正式結果見下） |
| `Part3/` | Calibration-free per-tile MX 格式選擇（AdaptQuant-style） | 完成，PPL 14.26（門檻 ≤ 15） |
| `tools/` | Part 1 填答 / 測試腳本、`make_submission.sh` 打包 | |
| `lab2_template/` | 助教提供的原始檔案（未修改） | |

## 正式結果

執行環境：Python 3.11.17、`torch 2.5.1+cu124`、`transformers 4.57.1`、`datasets 3.6.0`；
模型為作業指定的 **`Qwen/Qwen3-4B`**（36 層、hidden 2560、intermediate 9728），
評估皆使用提供的 `run_eval.py` / `evaluate.py`（WikiText-2 test、seqlen 2048），未修改 PPL 計算。
原始模型 FP16 PPL = **13.6382**。

| Part | Case | 要求 | 實測 | 結果 |
|---|---|---|---|---|
| 2 | Case 1 | CR > 75%、PPL < 15 | **CR 75.50%、PPL 13.87** | 達標 |
| 2 | Case 2 | CR ≥ 81%、PPL < 20 | **CR 81.10%、PPL 14.82** | 達標 |
| 2 | Case 3 | SVD CR ≥ 20%、整體 CR 50%、PPL < 50 | **SVD 20.22%、CR 50.02%、PPL 20.29** | 達標 |
| 2 | Case 4 | Pruning ≥ 20%、CR ≥ 85%、PPL < 50 | **Pruning 20.32%、CR 85.10%、PPL 16.57** | 達標 |
| 3 | – | PPL ≤ 15（AdaptQuant 13.98） | **4.5 bits、PPL 14.257** | 滿分區間 |

硬體與時間（單張 RTX 4090 24 GB）：Part 2 四個 case 合計約 34 分鐘
（Case 1 約 9 分、Case 2 約 5 分、Case 3 約 8 分、Case 4 約 9 分，含模型載入與 WikiText-2 評估）；
Part 3 量化 7.4 秒、評估約 1 分鐘。

## 重現步驟

```bash
git clone -b claude/gallant-meitner-e1gegm https://github.com/daniel1688tw/LAB2.git && cd LAB2
```

### Part 1
在 Kaggle 開啟 `Part1/StudentID_Part1.ipynb`，加入 MobileNetV2 模型
（`/kaggle/input/model/pytorch/default/1/mobilenetv2_0.963.pth`），Accelerator 選 GPU T4，Run All。
所有 `* Test passed.` 都應出現。若要在其他環境執行，把該路徑改成模型實際位置即可。

### Part 2（GPU，RTX 4090 約 34 分鐘 / T4 約 1.5–2 小時）
```bash
cd Part2
bash run.sh                 # 安裝 requirements、依序跑四個 case、輸出 CR 與 PPL
python run_eval.py          # 原始模型 PPL（報告用，13.6382）
bash ablation.sh            # 報告用消融（見 Part2/README.md）
python summarize_logs.py    # 把 logs/caseN.log 轉成報告表格
```
可只跑部分 case：`LAB2_CASES="1 3" bash run.sh`；或覆寫設定：
`LAB2_CASE2_CFG='{"nsamples":256}' LAB2_CASES=2 bash run.sh`（詳見 `Part2/README.md`）。

### Part 3（GPU，約 2 分鐘）
```bash
cd Part3
bash run.sh                 # 4.5 bits、MXFP4/6/8、128x128 tile、MXFP8 activation
bash run.sh --fp16          # 原始模型 PPL
bash sweep.sh               # 報告用消融（結果在 logs/results.jsonl）
```

### 報告與打包
* 報告：`Part2/StudentID_report.md`、`Part3/StudentID_report.md`（已填入正式數值），
  PDF 為 `Part2/315551049_report.pdf`、`Part3/315551049_report.pdf`。
* 打包：`bash tools/make_submission.sh 315551049` → `dist/315551049.zip`。

## 不需要 GPU 的測試
```bash
python tools/test_part1.py Part1/StudentID_Part1.ipynb          # Part 1 所有測試
python Part2/tests/test_units.py                                 # GPTQ / RHT / OBS / SVD / allocator
python Part2/tests/test_realdim.py                               # Qwen3-4B 層維度的形狀測試（4 核 CPU 約 45 分鐘）
python Part3/tests/test_mx.py                                    # MX 格式（E4M3 與 torch.float8 逐位元一致）
python Part2/tests/make_tiny_model.py /tmp/tb && source Part2/tests/env.sh /tmp/tb   # 離線小模型 + 假 WikiText
```
完整的端到端指令見 `Part2/README.md` 第 6 節。
