# Part 3：Calibration-free PTQ（AdaptQuant-style per-tile MX format selection）

目標：在**不使用任何校正資料**的前提下量化 `Qwen/Qwen3-4B`，以提供的 `evaluate.py`（WikiText-2 test，seqlen 2048）
評估 PPL；參考值 AdaptQuant PPL 13.98（≤ 15 即滿分）。T4 不支援 MX 低精度 kernel，所以以 fake quantization 模擬數值。

## 1. 重建環境與執行（TA 只需要這一節）

```bash
# 評分環境：Python 3.11 + torch==2.5.1+cu121，單張 GPU（T4 16 GB 即可）
cd Part3
pip install -r requirements.txt     # run.sh 也會自動執行
bash run.sh                         # 量化 + 評估，輸出最終 PPL
```

`run.sh` 會：安裝 `requirements.txt` → 下載 Qwen3-4B → 以預設設定量化（約 1–2 分鐘）→ 用 `evaluate.eval_ppl_wikitext2`
計算 PPL（T4 約 3–5 分鐘），最後只輸出：

```text
Average bits per weight: 4.500
PPL: xx.xx
```

完整紀錄：`logs/adaptquant.log`（每種格式佔比、每層型 / 每個 decoder layer 的平均 bits），結果也會附加到 `logs/results.jsonl`。
等價的單一指令：

```bash
python run_ptq.py --budget 4.5 --formats mxfp4,mxfp6,mxfp8 --tile 128x128 --colw analytic --act mxfp8
python run_ptq.py --fp16                 # 原始模型 PPL
python run_ptq.py --uniform mxfp4        # 每個 tile 都用 MXFP4 的 baseline
bash sweep.sh                            # 報告用的全部實驗（baseline、bit 預算、消融）
```

不需要任何額外設定；WikiText-2 / 模型由 Hugging Face 自動下載（`evaluate.py` 為助教提供的原檔，未修改）。

## 2. 方法

> AdaptQuant 原文（OpenReview）在開發環境無法存取，以下依論文標題「Hardware-Aligned Per-Tile MX Format Selection
> for PTQ on NVIDIA Blackwell」、OCP MX 規格與相關論文（MXSens、MixFP4、M2XFP、Block-rotation for MXFP4）重建，
> 差異列在第 4 節。

1. **格式**（Blackwell tensor core 原生支援的 block-scaled 格式，皆為每 32 個 K 方向元素共用一個 E8M0 scale）：
   MXFP4（E2M1，4.25 bits）、MXFP6（E2M3，6.25 bits）、MXFP8（E4M3，8.25 bits）；可選 NVFP4（E2M1 + E4M3 scale/16，4.5 bits）。
   數值模擬見 `mxformats.py`（E4M3/E5M2 與 `torch.float8_*` 逐位元一致，已測試）。
2. **Hardware-aligned tile**：權重切成 128×128 的 tile（N 方向對齊 128-row MMA tile、K 方向為 4 個 MX block），
   每個 tile 選一種格式，使同一個 tile 的 MMA 用同一種 `kind::f8f6f4 / mxf4` 指令。格式 id 每 tile 2 bits，計入平均 bits。
3. **Calibration-free 誤差度量**：tile 誤差 `Σ c_j (W_ij − Q_ij)²`，`c_j ≈ E[x_j²]` 只從權重推得：
   q/k/v、gate/up 的輸入是 RMSNorm 輸出 → `c = γ²`；o_proj 的輸入是 attention 加權後的 V → `c = diag(W_v diag(γ²) W_vᵀ)`
   （依 GQA 展開）；down_proj 的輸入是 `silu(g)·u`，假設 `g,u` 為常態 → `c = E[silu(g)²]·E[u²]`（Gauss–Hermite 積分）。
   每層再除以 `Σ c_j W_ij²` 變成相對誤差，使不同層可比較。
4. **E8M0 scale search**：OCP 規則 `e = ⌊log2 amax⌋ − emax` 會把 MXFP4 block 最大值截到 6；每個 block 在 `e` 與 `e+1`
   之間選加權誤差較小者（只看權重本身）。
5. **全域預算分配**：所有層的所有 tile 一起做 Lagrangian relaxation：最小化總誤差，限制平均 bits（含 scale 與 tile id）≤ 4.5。
6. **Activation**：每個 Linear 的輸入動態量化成 MXFP8（per-32 block，OCP scale），對應 Blackwell 的 MXFP8 × MXFP4/6/8 GEMM；
   `--act none` 為 weight-only。

量化對象：36 層 × 7 個 Linear（與 Part 2 相同），embedding / lm_head 保持 FP16。

## 3. 為何是 calibration-free

所有決策（每個 block 的 scale、每個 tile 的格式、bit 分配）都只用到權重本身與 RMSNorm 的 γ（也是權重），
沒有任何輸入資料、沒有 forward pass、沒有針對目標模型的校正流程；activation 的 scale 是推論時從當下輸入動態計算
（MX 格式的標準行為），不是事先校正的。

## 4. 與 AdaptQuant 的可能差異 / 已知限制

* 原文的 tile 大小、候選格式集合與選擇準則未知；本實作把它們都做成參數（`--tile`、`--formats`、`--colw`、`--metric`），
  `sweep.sh` 提供比較。
* 只模擬數值（fake quant），沒有實作 Blackwell kernel；PPL 與實際硬體輸出可能有極小差異（累加順序 / 精度）。
* 預設預算 4.5 bits/weight（與 NVFP4 同樣大小）；若 PPL 高於 15，可用 `bash run.sh --budget 4.75` 或 `--budget 5.0`。

## 5. 檔案

| 檔案 | 內容 |
|---|---|
| `run.sh` | 主程式入口（安裝 + 量化 + 評估） |
| `run_ptq.py` | CLI：載入模型 → `adaptquant()` → `eval_ppl_wikitext2` |
| `adaptquant.py` | data-free 輸入統計、tile 誤差、全域分配、寫回權重、activation hook |
| `mxformats.py` | E2M1/E2M3/E3M2/E4M3/E5M2 minifloat、MX block 量化、NVFP4 |
| `evaluate.py` | 助教提供的評估程式（未修改） |
| `sweep.sh` | 報告用實驗 |
| `tests/test_mx.py` | 格式與分配的單元測試（CPU） |
| `StudentID_report.md` | 報告草稿（輸出成 `StudentID_report.pdf`） |
