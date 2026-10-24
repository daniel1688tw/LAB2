# Part 2：Qwen3-4B Linear 層壓縮

目標模型：`Qwen/Qwen3-4B`（36 層 decoder，每層 7 個目標 Linear：`q/k/v/o_proj`、`gate/up/down_proj`，
共 3.63 B 參數；embedding / tied `lm_head` 不是目標層，完全不動）。

| Case | 要求 | 預設目標 (`run.sh`) | 方法 |
|---|---|---|---|
| 1 | CR > 75%，PPL < 15 | rate 0.755、wbits 4 | RHT + GPTQ + 依敏感度的 {4,3}-bit 混合精度 |
| 2 | CR ≥ 81%，PPL < 20 | rate 0.811、wbits 3 | RHT + GPTQ + 依敏感度的 {4,3,2}-bit 混合精度 |
| 3 | SVD CR ≥ 20%，整體 CR = 50%，PPL < 50 | rate 0.5、wbits 8 | 白化 SVD（SVD-LLM）+ 全域 rank 分配 + 8-bit 因子 |
| 4 | Pruning ≥ 20%，CR ≥ 85%，PPL < 50 | rate 0.851、wbits 3 | OBS 結構化剪枝 MLP channel + RHT + GPTQ 混合精度 |

## 1. 執行

```bash
cd Part2
bash run.sh            # = bash StudentID_run.sh（繳交前把 StudentID 改成學號）
```

`run.sh` 會自動：(1) `pip install -r requirements.txt`；(2) 每個 case 都從原始 Qwen3-4B 重新載入並壓縮；
(3) 用提供的 `run_eval.py` 評估；(4) 只輸出：

```text
Case 1
Compression Ratio: xx.x%
PPL: xx.xx
...
```

完整過程（每層的 bit 配置、GPTQ loss、rank、剪枝計畫、各層型的儲存量）寫在 `logs/caseN.log`。

* 環境：Python 3.11、`torch==2.5.1+cu121`（評分環境），單張 T4 16 GB 即可（峰值約 11–12 GB）。
* 時間（T4 估計）：Case 1 ≈ 25 分（含 ~11 分敏感度分析）、Case 2 ≈ 15 分（共用 Case 1 的敏感度快取 `cache/`）、
  Case 3 ≈ 15 分、Case 4 ≈ 30 分，整體約 1.5–2 小時（第一次還要下載 8 GB 模型與 WikiText-2）。
* 只跑部分 case：`LAB2_CASES="1 3" bash run.sh`；不重裝套件：`LAB2_SKIP_INSTALL=1 bash run.sh`。
* 單獨跑一個 case：`python run_eval.py --case 2 --rate 0.811 --wbits 3`；原始模型：`python run_eval.py`。

### 調整超參數（不改程式碼）

每個 `caseN.py` 開頭有 `DEFAULTS`，可用環境變數以 JSON 覆寫任一 key，例如：

```bash
LAB2_CASE2_CFG='{"nsamples": 256}' LAB2_CASES=2 bash run.sh
LAB2_CASE2_CFG='{"specs": ["w4g128a","w4cha","w3g64a","w3g128a","w3g256a","w3cha"]}' LAB2_CASES=2 bash run.sh   # 不用 2-bit
LAB2_CASE4_CFG='{"prune_ratio": 0.20, "skip_first": 3}' LAB2_CASES=4 bash run.sh
LAB2_CASE3_CFG='{"rank_alloc": "uniform"}' LAB2_CASES=3 bash run.sh
```

spec 字串格式：`w<bits>g<group>a|s` 或 `w<bits>ch a|s`（`ch` = 每個 output row 一組，`a` = asymmetric、`s` = symmetric）。
若最終想交出的設定和預設不同，請把它寫回 `caseN.py` 的 `DEFAULTS`（評分時只會執行 `bash run.sh`）。

`bash ablation.sh` 會跑報告用的消融實驗（RTN vs GPTQ、有無 RHT、有無混合精度、SVD rank 分配、剪枝比例），結果累積在
`logs/ablation.txt`；`python summarize_logs.py` 把 `logs/caseN.log` 整理成 Markdown 表格。

## 2. 對提供檔案的修改

* `evaluate.py`：只實作原本留空的 `compression_rate()`（呼叫 `common.compressed_linear_bits`）。
  `eval_ppl_wikitext2`、`load_model`、資料讀取完全沒改。
* `run_eval.py`：在 `load_model` 與 `eval_ppl_wikitext2` 之間加入 5 行：`common.record_base(model)`，
  再 `importlib.import_module(f"case{N}").compress(model, tok, rate, wbits, seed)`。其餘（含輸出格式）未改。

## 3. Compression Ratio 的計算（含所有額外參數）

`CR = 1 − (壓縮後目標 Linear 層總 bits) / (原始目標 Linear 層 numel × 16)`，每層的儲存量（`common.set_linear_bits`）：

* **量化權重**：`out × in × b`（整數碼）+ 每個 group 的 FP16 scale（16 bits）+ asymmetric 的整數 zero point（b bits）。
  GPTQ 使用 *static groups*，act-order 不需要額外的 g_idx。
* **RHT**：每層 input/output 兩個隨機 Hadamard 變換由 seed 產生，保守地計入 `in + out` 個 sign bits 再加 2×64-bit seed。
* **SVD**：兩個因子 `A[out×k]`、`B[k×in]` 的整數碼 + 每 row 一個 FP16 scale（或 FP16 因子）。
* **結構化剪枝**：矩陣真的變小（`gate/up` 少 row、`down` 少 column），不需要 mask/index。
* 沒被壓縮的層以目前形狀 × 16 bits 計。

評估時使用 fake quantization：反量化後的權重以 FP16 寫回 `nn.Linear`（數值與 packed 格式推論相同），
所以 PPL 反映的就是上述壓縮表示的品質。

## 4. 方法細節

### 共用：依層序 (sequential) 的校正流程
校正資料：WikiText-2 **train** split 的 128 段 × 2048 tokens（測試集從不用於壓縮）。
`common.capture_block_inputs` 取得第 0 層輸入後逐層前進；每層內依 GPTQ 的 true-sequential 順序
`(q,k,v) → o → (gate,up) → down` 收集 Hessian `H = 2/n Σ xxᵀ`、壓縮、再用「已壓縮」的層產生下一組輸入，
讓後面的層補償前面的誤差。這些都是一次性的閉式計算，沒有任何梯度訓練 / fine-tuning。

### GPTQ（`quant.gptq_quantize`）
參考 Frantar et al. 2022：Cholesky 形式的 OBQ 逐 column 量化並把誤差補償到尚未量化的 column；
act-order（依 diag(H) 排序）+ static groups；group 的 clipping 範圍以 MSE grid search（`mse_grid=20`）決定；
dampening 1%。

### 隨機 Hadamard 變換 / Incoherence processing（`quant.RHT`，QuIP / QuIP# / QuaRot）
量化 `W' = R_out W R_inᵀ`，`R = H_blk · diag(±1) · P`（P 隨機排列，H_blk 為 2 的冪次 block 的 Hadamard，
2560/9728 用 512、4096 用 4096），Hessian 同步旋轉 `H' = R_in H R_inᵀ`。正交變換不改變層輸出，
但把少數 outlier 權重攤平成近似高斯，讓 3-bit、甚至 per-channel 量化的誤差大幅下降，
也讓 group size 可以加大（scale overhead 從 0.16 bit 降到 ~0.01 bit），這是 Case 2/4 能在 ~3 bits 下達標的關鍵。

### 依敏感度的混合精度（`alloc.py`）
1. 對每個目標層、每個參考 bit 寬（`w4g128a / w3g128a / w2g128a`）：只把「這一層」做 RTN 量化，
   從該層跑到 logits，量測 `KL(p_fp ‖ p_q)`（8 段 × 512 tokens）。
2. 每個候選 spec 以 Hessian 對角線 proxy `Σ_j H_jj ‖ΔW_:,j‖²` 換算同 bit 寬下不同 group size 的成本。
3. 在 bits 預算下最小化 Σ cost：Lagrangian relaxation（二分搜尋 λ）+ 貪婪補滿剩餘預算。
敏感度結果快取在 `cache/`，Case 1/2 共用。

### Case 3：白化 SVD（`svd.py`，SVD-LLM）
`H = S Sᵀ`（eigendecomposition），對 `W S` 做 SVD；截斷誤差 `‖(W−W_k)X‖² = Σ_{i>k} σ_i²`，所以在白化空間截斷對輸出是最佳的。
Pass 1 在原始模型上取得每層白化奇異值，依「丟掉的能量比例」做**全域 Lagrangian rank 分配**
（每層可選擇保持 dense 或 rank `k ≤ mn/(m+n)`），使參數量 ≤ 79.8%（SVD CR ≥ 20.2%）。
Pass 2 依層序用已壓縮模型的輸入重新白化並分解。因子 `A = U_k√Σ_k`、`B = √Σ_k V_kᵀ S⁻¹` 量化成 8-bit（每 row 對稱）；
8-bit 全量化會到 60%，所以從前面的層開始把因子留在 FP16，直到整體 CR 剛好 ≥ 50%。

### Case 4：OBS 結構化剪枝（`prune.py`，SlimGPT 類）
以 `down_proj` 輸入的 Hessian 計算 channel 分數 `‖W_:,j‖² / [H⁻¹]_jj`，每次移除 32 個最不重要的 channel，
並以 OBS 閉式公式更新剩餘權重（least-squares 重建 MLP 輸出），同時刪掉 `gate/up` 對應的 row。
前 2 層與最後 1 層不剪，其餘層平均分配，channel 數對齊 128。預設移除 21% 的目標參數，再對剪枝後的模型
重新做敏感度分析 + RHT + GPTQ 混合精度，使整體 CR ≥ 85.1%。

## 5. 若 PPL 沒達標的建議調整順序

* Case 1（PPL ≥ 15）：`nsamples` 256；`specs` 只留 `w4*`/`w3g128a`；確認 `rht: true`。
* Case 2（PPL ≥ 20）：`nsamples` 256、`sens_nsamples` 16；去掉 2-bit 候選（見上）；或加入 `w3g64a` 以外的 4-bit 候選。
* Case 3（PPL ≥ 50）：`rank_alloc: "uniform"` 比較；`type_weights` 例如 `{"down_proj": 2, "v_proj": 2}`。
* Case 4（PPL ≥ 50）：`prune_ratio` 0.20（下限）、`skip_first` 3–4；或提高 `rate` 以外的 nsamples。

## 6. 測試（不需 GPU）

```bash
python tests/test_units.py                              # 元件單元測試（RHT、GPTQ、OBS、SVD、allocator）
python tests/make_tiny_model.py /tmp/tb                 # 建立小型 Qwen3 + 假的 WikiText 快取（離線）
source tests/env.sh /tmp/tb
LAB2_MODEL=$TB/tiny-qwen3 LAB2_RUN_EVAL=tests/run_eval_cpu.py LAB2_SKIP_INSTALL=1 \
LAB2_CASE1_CFG='{"nsamples":16,"seqlen":512,"sens_nsamples":4,"sens_seqlen":256}' ... bash run.sh
```
小模型（6 層、hidden 256、基準 PPL 45.83）上四個 case 都能完整跑完並達到目標壓縮率：Case 1 75.5% / 45.89、Case 2 81.1% / 45.89、Case 3 50.0% / 45.87、Case 4 85.1% / 46.05（小模型對量化極不敏感，數值只代表流程正確，不代表 Qwen3-4B 的結果）。

## 7. 參考
GPTQ (Frantar et al., ICLR 2023)；QuIP (Chee et al., NeurIPS 2023)、QuIP# (Tseng et al., ICML 2024)、QuaRot (Ashkboos et al., 2024)；
SVD-LLM (Wang et al., ICLR 2025)；SlimGPT (Ling et al., NeurIPS 2024)、Optimal Brain Surgeon (Hassibi & Stork, 1993)；
HAWQ-V2 (Dong et al., 2020)、MXSens (2026) 的敏感度分配想法；An Empirical Study of Qwen3 Quantization (Zheng et al., 2025)。
