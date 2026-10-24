# Lab 2 Part 2 報告：Qwen3-4B Linear 層壓縮

> 草稿：方法與分析已寫好；`【待填】` 處請以 `bash run.sh` / `bash ablation.sh` 在 GPU 上的實際結果填入
> （`python summarize_logs.py` 可把 `logs/caseN.log` 轉成表格）。完成後輸出為 `StudentID_report.pdf`。

## 0. 共同設定

* 模型 `Qwen/Qwen3-4B`，FP16，壓縮對象為 36 層 × 7 個 Linear（3.63 B 參數，原始 6.92 GiB）。
* 評估：提供的 `run_eval.py`（WikiText-2 test，seqlen 2048）。原始模型 PPL：【待填，文獻值約 13.7】。
* 校正資料：WikiText-2 train 128 × 2048 tokens（敏感度分析 8 × 512 tokens），不使用 test split，沒有任何 fine-tuning。
* 壓縮率計算包含所有額外參數：FP16 scale、整數 zero point、RHT sign/seed、SVD 因子的 scale。

## 1. Case 1（CR > 75%，PPL < 15）

**壓縮策略**
* 方法：隨機 Hadamard 變換（RHT，incoherence processing）+ GPTQ（act-order、static groups、MSE clipping）+
  依 KL 敏感度的逐層混合精度。
* 壓縮的層：全部 252 個目標 Linear。
* 候選格式：`w4g128a / w4g256a / w4ch / w3g128a / w3g256a / w3ch`；預算 = 原始大小的 24.5%（rate 0.755）。
* 配置結果（各層型的 spec 數量、平均 bits）：【待填，summarize_logs.py】

**結果**：Compression Ratio【待填】、PPL【待填】

**分析**
* 75% 意味著含 overhead 平均 < 4 bits/weight。均勻的 4-bit g128 asymmetric 本身就要 4.156 bits（CR 74.0%），
  因此一定要有一部分層降到 3 bits 或使用更大的 group；RHT 讓 per-channel / g256 的 4-bit 幾乎不損失，
  所以 allocator 可以把省下的 scale overhead 留給敏感層。
* 被降成 3-bit 的主要是【待填：依 log，通常是中間層的 gate/up】；`v_proj`、`down_proj` 及前幾層維持 4-bit。
* 與消融比較：RTN w4g128【待填】、GPTQ w4g128【待填】、GPTQ+RHT w4g128【待填】。

## 2. Case 2（CR ≥ 81%，PPL < 20）

**壓縮策略**
* 與 Case 1 相同的管線，候選格式加入 2-bit 與更多 group size：
  `w4g128a / w4ch / w3g64a / w3g128a / w3g256a / w3ch / w2g64a / w2g128a`，預算 18.9%（rate 0.811），
  平均約 3.03 bits/weight（含 overhead）。
* 配置結果：【待填】

**結果**：Compression Ratio【待填】、PPL【待填】

**分析**
* 81% 的預算低於 3-bit g128 + FP16 scale/zero（3.148 bits → 80.3%），所以不能用「標準」的 3-bit 設定。
  做法是 (1) RHT 後使用大 group / per-channel 3-bit（overhead ≈ 0.01 bit）；(2) 用敏感度把預算挪給敏感層：
  少數層 4-bit、魯棒的層 2-bit（例如【待填】）。
* 文獻上 Qwen3-4B 的 AWQ w3g128 PPL 高達 91、GPTQ w2g128 為 195（Zheng et al. 2025），顯示 Qwen3 對低 bit 非常敏感；
  RHT + GPTQ 的消融：GPTQ w3g128（無 RHT）【待填】 vs. GPTQ+RHT w3g128【待填】 vs. GPTQ+RHT w3ch【待填】。
* 混合精度 vs. 均勻精度、有無 2-bit 候選：【待填，ablation c2_mixed_*】

## 3. Case 3（SVD CR ≥ 20%，整體 CR = 50%，PPL < 50）

**壓縮策略**
* 方法：truncation-aware 白化 SVD（SVD-LLM）：`H = SSᵀ`，對 `WS` 做 SVD 並截斷，輸出誤差 = 丟掉的 σ²。
* Rank 分配：全域 Lagrangian，最小化各層「丟掉的白化能量比例」總和，限制低秩參數 ≤ 79.8%；
  每層可保持 dense 或 rank ≤ mn/(m+n)。結果：【待填：多少層被分解、各層型 rank】。
* 低秩因子 8-bit（per-row 對稱）量化；8-bit 全量化會到 60%，所以前面的層因子保留 FP16，使整體剛好 ≥ 50%。

**結果**：SVD Compression Ratio【待填，log 中 "SVD compression rate"】、Overall Compression Ratio【待填】、PPL【待填】

**分析**
* SVD 對 LLM 權重的效率遠低於量化：同樣 20% 的參數縮減，PPL 增加【待填】，而 8-bit 量化把大小減半幾乎無損
  （消融 c3_svd_only_fp16 vs. Case 3）。
* 全域分配 vs. 均勻分配：【待填 c3_uniform】。全域分配傾向壓縮奇異值衰減快的層（通常 `q_proj`、`k_proj`），保留 `down_proj`、`v_proj`。
* 白化的必要性：普通 SVD 截斷的是權重能量而非輸出能量，在 outlier 通道明顯的 LLM 上誤差大很多（單元測試中已驗證白化截斷的輸出誤差較小）。

## 4. Case 4（Pruning ≥ 20%，整體 CR ≥ 85%，PPL < 50）

**壓縮策略**
* 結構化剪枝 MLP intermediate channel：以 `down_proj` 輸入 Hessian 的 OBS 分數 `‖W_:,j‖²/[H⁻¹]_jj` 選擇 channel，
  批次移除並以閉式公式補償剩餘權重（SlimGPT 類），同時刪除 `gate/up` 的對應 row。前 2 層、最後 1 層不剪，channel 數對齊 128。
* 剪掉 21% 的目標參數（實際：【待填 pruning rate】），矩陣真的變小，不需要 mask。
* 剪枝後重新量測敏感度，再以 RHT + GPTQ 混合精度量化到 rate 0.851（剩餘權重平均約 3 bits）。

**結果**：Pruning Rate【待填】、Compression Ratio【待填】、PPL【待填】

**分析**
* 為何選結構化剪枝：非結構化剪枝 20% 需要 mask/index（bitmask 就要 1 bit/weight），在 2.4 bits/weight 的總預算下不划算；
  結構化剪枝直接縮小矩陣，剪 20% 等於把量化預算放寬到 2.4/0.8 = 3.0 bits。
* 只剪枝（8-bit 量化）的 PPL：【待填 c4_prune_only】→ 剪枝本身的損失；再量化後：【待填】。
* 剪枝比例 0.21 vs 0.25：【待填 c4_prune025】（剪多一點可讓量化 bits 變多，兩者的取捨）。

## 5. 總結

* **CR 與 PPL 的取捨**：【待填：四個 case 的 (CR, PPL) 圖或表】。在 ~4 bits 以上幾乎無損，3 bits 開始明顯上升，
  2 bits 以下急劇惡化；低秩分解與結構化剪枝每減少 1% 參數造成的 PPL 損失遠大於量化。
* **敏感層**：依敏感度分析【待填：KL 最大的層型與層號】；一般觀察是 `v_proj`、`down_proj`、前幾層與最後一層最敏感，
  `q_proj/k_proj` 與中間層的 `gate/up` 最魯棒。
* **各策略比較**：量化（GPTQ+RHT）在相同大小下品質最好；SVD 只適合低壓縮率；結構化剪枝適合與量化合用以換取 bit 預算。
* **限制**：fake quantization（沒有實作 packed kernel）；敏感度的加總假設忽略層間交互；校正資料與評估同分佈（WikiText），
  在其他資料上的泛化需另外驗證；混合 bit 寬在實際部署需要多種 kernel。
* **最有效的策略**：在本作業限制下，RHT + GPTQ + 依敏感度的混合精度最有效，它能把 overhead 降到最低並把 bits 放在最敏感的層。
