# Lab 2：Large Language Model Compression

| 目錄 | 內容 | 狀態 |
|---|---|---|
| `Part1/StudentID_Part1.ipynb` | Linear quantization Q1–Q10 | 完成，所有測試在 CPU 上通過（MobileNetV2 的兩個 cell 需在 Kaggle 執行） |
| `Part2/` | Qwen3-4B 四個壓縮 case（GPTQ + RHT + 混合精度、白化 SVD、OBS 結構化剪枝） | 程式完成並在小模型上端到端驗證；需在 GPU 上跑正式結果 |
| `Part3/` | Calibration-free per-tile MX 格式選擇（AdaptQuant-style） | 程式完成並在小模型上驗證；需在 GPU 上跑正式結果 |
| `tools/` | Part 1 填答 / 測試腳本、`make_submission.sh` 打包 | |
| `lab2_template/` | 助教提供的原始檔案（未修改） | |

> 開發環境沒有 GPU，也無法連到 Hugging Face，所以**所有 Qwen3-4B 的數值都還沒有跑**。程式在
> (1) 元件單元測試、(2) 自行訓練的 6 層小型 Qwen3（同樣的 `Qwen3ForCausalLM` 架構）+ 未修改的 `run_eval.py`、
> (3) 與 Qwen3-4B 完全相同層維度的 3 層隨機模型 上驗證過，並在評分環境的版本（Python 3.11、torch 2.5.1、
> transformers 4.57.1）與新版（torch 2.14、transformers 5.20）都能執行。

## 正式實驗步驟（GPU）

需要一張 ≥ 16 GB 的 GPU（Kaggle / Colab T4 即可）與網路（下載 Qwen3-4B 約 8 GB、WikiText-2）。

```bash
git clone -b claude/gallant-meitner-e1gegm https://github.com/daniel1688tw/LAB2.git && cd LAB2
nvidia-smi                                    # 確認有 GPU
```

### Part 1
在 Kaggle 開啟 `Part1/StudentID_Part1.ipynb`，加入 MobileNetV2 模型（`/kaggle/input/model/pytorch/default/1/mobilenetv2_0.963.pth`），
Accelerator 選 GPU T4，Run All。所有 `* Test passed.` 都應出現。

### Part 2（約 1.5–2 小時）
```bash
cd Part2
bash run.sh                 # 評分環境 (torch 2.5.1) 直接執行；輸出四個 case 的 Compression Ratio / PPL
# Kaggle/Colab 的 torch 版本不同、不想降版時：
pip install transformers==4.57.1 accelerate==1.10.1 datasets==3.6.0 && LAB2_SKIP_INSTALL=1 bash run.sh
python run_eval.py          # 原始模型 PPL（報告用）
bash ablation.sh            # 報告用消融（約 3–4 小時，可分次跑）
python summarize_logs.py    # 把 logs/caseN.log 轉成報告表格
```
若某個 case 沒達標，依 `Part2/README.md` 第 5 節調整（例如 `LAB2_CASE2_CFG='{"nsamples":256}' LAB2_CASES=2 bash run.sh`），
最後把確定的設定寫回 `caseN.py` 的 `DEFAULTS`。

### Part 3（約 10 分鐘 / 每個設定）
```bash
cd Part3
bash run.sh                 # 預設 4.5 bits、MXFP4/6/8、128x128 tile、MXFP8 activation
bash run.sh --budget 4.75   # 若 PPL > 15 可提高預算
bash sweep.sh               # baseline 與消融（結果在 logs/results.jsonl）
```

### 報告與打包
* 報告草稿：`Part2/StudentID_report.md`、`Part3/StudentID_report.md`（方法、分析已寫好，`【待填】` 處填入實驗數值），
  轉成 PDF 存成 `Part2/StudentID_report.pdf`、`Part3/StudentID_report.pdf`。
* 打包：`bash tools/make_submission.sh <學號>` → `dist/<學號>.zip`（自動改名 `<學號>_Part1.ipynb`、`<學號>_run.sh`、`<學號>_report.pdf`）。

## 不需要 GPU 的測試
```bash
python tools/test_part1.py Part1/StudentID_Part1.ipynb          # Part 1 所有測試
python Part2/tests/test_units.py                                 # GPTQ / RHT / OBS / SVD / allocator
python Part2/tests/test_realdim.py                               # Qwen3-4B 層維度的形狀測試（CPU 約 10–20 分鐘）
python Part3/tests/test_mx.py                                    # MX 格式（E4M3 與 torch.float8 逐位元一致）
python Part2/tests/make_tiny_model.py /tmp/tb && source Part2/tests/env.sh /tmp/tb   # 離線小模型 + 假 WikiText
```
完整的端到端指令見 `Part2/README.md` 第 6 節。
