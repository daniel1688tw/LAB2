# Lab 2 Part 2 報告：Qwen3-4B Linear 層壓縮

**學號：315551049**

## 0. 共同設定

* 模型：`Qwen/Qwen3-4B`（36 層 decoder，hidden 2560、intermediate 9728、32/8 heads），FP16。
* 壓縮對象：每層 7 個 Linear（`q/k/v/o_proj`、`gate/up/down_proj`），共 252 層、3.63 B 參數
  （原始目標層儲存 6,930 MiB）。embedding 與 tied `lm_head` 不是目標層，完全不動。
* 評估：作業提供的 `run_eval.py`（WikiText-2 test、seqlen 2048）。原始模型 PPL = **13.6382**
  （Part 3 的 `evaluate.py` 得到相同值）。
* 校正資料：WikiText-2 **train** split 的 128 段 × 2048 tokens；敏感度分析 8 段 × 512 tokens。
  測試集從未用於壓縮，也沒有任何梯度訓練 / fine-tuning。
* 壓縮率計算包含所有額外參數：FP16 scale、整數 zero point、RHT 的 sign / seed、SVD 因子的 scale；
  結構化剪枝後的矩陣真的變小，不需要 mask。評估時以 fake quantization 將反量化權重寫回 FP16。

## 1. Case 1（CR > 75%，PPL < 15）

### 壓縮策略

* 方法：隨機 Hadamard 變換（RHT，incoherence processing）+ GPTQ + 依 KL 敏感度的逐層混合精度。
* 所有 252 個目標 Linear 都壓縮。候選格式：`w4g128a`、`w4g256a`、`w4cha`、
  `w3g128a`、`w3g256a`、`w3cha`（a = asymmetric）。
* 預算：原始大小的 24.5%（1,697.8 MiB），由 allocator 在候選格式間分配。
* GPTQ 設定：act-order、static groups、MSE clipping（grid 20）、dampening 1%、blocksize 128；
  校正 128 × 2048 tokens。

### 結果

* **Compression Ratio：75.50%**
* **PPL：13.87**（原始 13.6382，+0.23）
* 配置：`w4g128a` 128 層、`w4g256a` 71 層、`w3g128a` 21 層、`w3g256a` 12 層、
  `w3cha` 11 層、`w4cha` 9 層。
* 各層型平均 bits（相對原始權重）：`q_proj` 3.80、`k_proj` 4.10、`v_proj` 4.16、
  `o_proj` 3.69、`gate_proj` 3.82、`up_proj` 4.05、`down_proj` 3.99。

### 分析

* 均勻的 4-bit g128 asymmetric 本身需要 4.156 bits/weight，壓縮率只有 74.02%，**達不到 75%**；
  因此一定要有部分層降到 3 bits，或使用更大的 group 來降低 scale overhead。
* 敏感度分析把 `v_proj` 全部留在 4-bit（平均 4.16 bits），`k_proj` 也幾乎都是 4-bit；
  被降到 3-bit 的主要是 `q_proj`、`o_proj` 與 MLP 的部分層（合計 44 層）。
  這與「value 投影直接決定 attention 輸出、最敏感」的預期一致。
* RHT 把 outlier 攤平後，`w4g256a` 與 `w4cha` 的品質接近 `w4g128a`，allocator 因此能把
  scale / zero-point overhead 從 0.16 bit 降到 0.01–0.08 bit，把省下的預算留給敏感層。
* 消融（第 5.1 節）：均勻 `w4g128a` 用 RTN 時 PPL 15.68，換成 GPTQ 降到 13.84；
  混合精度反而少用 0.24 bit 的預算（75.50% vs 74.02%），就維持 13.87。
  值得注意的是，在此模型上 4-bit 均勻量化加 RHT（14.09）略差於不加 RHT（13.84），
  但 RHT 讓 3-bit 與大 group 的誤差大幅下降，混合精度整體仍能在 3.94 bits 平均下達到 13.87。

## 2. Case 2（CR ≥ 81%，PPL < 20）

### 壓縮策略

* 與 Case 1 相同的 RHT + GPTQ 管線，候選格式加入 2-bit 與更多 group size：
  `w4g128a`、`w4cha`、`w3g64a`、`w3g128a`、`w3g256a`、`w3cha`、`w2g64a`、`w2g128a`。
* 預算：原始大小的 18.9%（1,309.8 MiB），平均 3.02 bits/weight（含所有 overhead）。
* 敏感度快取與 Case 1 共用（同樣的校正設定與 RHT），因此 Case 2 只需做 GPTQ 與評估。

### 結果

* **Compression Ratio：81.10%**
* **PPL：14.82**（原始 13.6382，+1.18）
* 配置：`w3cha` 117 層、`w4g128a` 46 層、`w3g256a` 31 層、`w2g128a` 26 層、
  `w4cha` 18 層、`w3g128a` 11 層、`w3g64a` 3 層。
* 各層型平均 bits：`q_proj` 2.94、`k_proj` 3.71、`v_proj` 4.08、`o_proj` 2.75、
  `gate_proj` 2.95、`up_proj` 3.03、`down_proj` 3.06。

### 分析

* 標準的 3-bit g128 asymmetric 需要 3.148 bits（CR 80.32%），**低於 81% 的要求**。
  本 case 的關鍵是 (1) 使用 per-channel 3-bit（`w3cha`，overhead 僅約 0.01 bit）壓縮較魯棒的層；
  (2) 用敏感度把 2-bit 分配給 `q_proj`（6 層）、`o_proj`（12 層）與少數 MLP 層（合計 26 層），
  把省下的 bits 留給 `v_proj` 與 `k_proj`。
* `v_proj` 仍然全部保持 4-bit（平均 4.08），`k_proj` 平均 3.71；被壓到 2-bit 的主要是
  `o_proj` 與 `q_proj`，它們的誤差會被後面的 residual / MLP 部分吸收。
* 消融：均勻 3-bit `w3g128a` 不加 RHT 的 PPL 14.66（CR 80.32%，未達標）；
  RHT 與混合精度讓 CR 超過 81% 的同時 PPL 只增加 1.18。
  這說明在 3-bit 附近，**overhead 與 layer-wise 分配比單純的 bit 數更重要**。

## 3. Case 3（SVD CR ≥ 20%，整體 CR = 50%，PPL < 50）

### 壓縮策略

* 方法：truncation-aware 白化 SVD（SVD-LLM）。對 Linear 的輸入二階動差 `H = S Sᵀ` 做特徵分解，
  對 `W S` 做 SVD；截斷誤差 `‖(W − Wₖ) X‖² = Σ(i>k) σᵢ²`，因此在白化空間截斷對輸出是最佳的。
* Rank 分配：Pass 1 在原始模型上取得每層的白化奇異值，預設讓**每一層都保留 79.8% 的參數**
  （rank 依層的形狀換算），使 SVD compression rate = 20.22%。這是先用消融比較後選定的：
  全域能量分配（Lagrangian）會讓後段層的 rank 過低，PPL 30.74；均勻分配則是 20.29。
* 低秩因子 `A = Uₖ√Σₖ`、`B = √Σₖ Vₖᵀ S⁻¹` 以 8-bit（per-row 對稱）量化；
  8-bit 全量化會讓整體 CR 到 60%，因此從前面的層開始把因子留在 FP16，直到整體 CR 剛好 ≥ 50%。

### 結果

* **SVD Compression Ratio：20.22%**（252/252 層都被分解，參數減少 20.22%）
* **Overall Compression Ratio：50.02%**
* **PPL：20.29**（原始 13.6382，+6.65）
* 每層保留相同的參數比例（79.8%），換算成 rank：`q_proj` 1257、`k_proj`/`v_proj` 583、
  `o_proj` 1257、`gate_proj`/`up_proj`/`down_proj` 1617。沒有 dense 層。
* 因子精度：前 11 層（layer 0–10）共 66 個因子矩陣用 FP16，其餘 186 個用 8-bit。
* 各層型平均 bits：`q_proj` 7.99、`k_proj` 8.33、`v_proj` 8.15、`o_proj` 7.98、
  `gate_proj` 7.98、`up_proj` 7.98、`down_proj` 7.98。

### 分析

* SVD 對 LLM 的效率遠低於量化：只減少 20.22% 的參數，PPL 就從 13.64 升到 20.29（+6.65）；
  而 8-bit 量化把大小減半幾乎無損。低秩分解丟掉的是整個方向的資訊，
  量化只是把每個權重變粗，後者對這個模型友善得多。
* 均勻分配的關鍵是**不要讓任何一層被過度壓縮**：每層保留 79.8% 參數，
  rank 隨層的形狀自動調整（寬的 MLP 得到 1617、窄的 k/v 得到 583）。
  誤差不會集中在少數後段層，所以比全域能量分配好 10.45 PPL。
* 全域能量分配（`rank_alloc="global"`）最小化「丟掉的白化能量總和」，
  但這個目標偏好絕對能量大的前段層，容易讓後段層的 rank 過低；
  誤差會逐層放大。消融編號 `c3_uniform` 與正式結果一致（20.29），
  全域分配則是 30.74，因此預設改成 uniform。
* 「整體 CR = 50%」不是只靠 SVD：SVD 只提供 20.22%，其餘 29.8% 來自因子量化。
  8-bit 全量化會到 60%，本實作把前 11 層的 66 個因子留在 FP16，讓整體剛好落在 50.02%。
* 白化的必要性：普通 SVD 截斷的是權重能量而非輸出能量，在 outlier 通道明顯的 LLM 上誤差會大很多；
  單元測試中已驗證白化截斷的輸出誤差較小（`tests/test_units.py` 的 `svd ok`）。
* 在全域分配下做的 SVD-only 消融（因子全 FP16、CR 20.21%）PPL 是 30.59，
  和同分配 + 8-bit 因子的 30.74 幾乎相同；這說明 Case 3 的誤差幾乎全部來自低秩截斷，
  8-bit 因子量化不是瓶頸（也解釋了為什麼 uniform 的 per-type bits 都接近 8）。
* 與量化相比，SVD 的優點是不需要校正資料以外的假設，但在此壓縮率下 PPL 代價明顯偏高；
  它比較適合當成「低壓縮率、與量化互補」的工具，而不是單獨達到 50%。

## 4. Case 4（Pruning ≥ 20%，CR ≥ 85%，PPL < 50）

### 壓縮策略

* 結構化剪枝 MLP intermediate channel：以 `down_proj` 輸入的 Hessian 計算 OBS 分數
  `‖W[:,j]‖² / [H⁻¹]ⱼⱼ`，每次移除 32 個 channel，並以閉式公式更新剩餘權重
  （least-squares 重建 MLP 輸出，SlimGPT 類）；同時刪除 `gate_proj` / `up_proj` 對應的 row。
* 前 2 層與最後 1 層不剪（最敏感），其餘 32 層平均分配，channel 數對齊 128。
* 移除 20.32% 的目標參數：每個被剪的層從 9,728 個 channel 保留 6,784–6,912 個
  （每層移除 2,816–2,944 個）；矩陣真的變小，不需要 mask / index。
* 剪枝後重新量測敏感度，再以 RHT + GPTQ 混合精度量化，使整體 CR ≥ 85.1%。

### 結果

* **Pruning Rate：20.32%**（參數移除比例）
* **Compression Ratio：85.10%**
* **PPL：16.57**（原始 13.6382，+2.93）
* 量化配置：`w3cha` 116 層、`w4g128a` 35 層、`w3g256a` 34 層、`w2g128a` 32 層、
  `w3g128a` 16 層、`w4cha` 14 層、`w3g64a` 5 層。
* 各層型平均 bits（相對原始權重）：`q_proj` 2.75、`k_proj` 3.38、`v_proj` 4.07、
  `o_proj` 2.68、`gate_proj` 2.16、`up_proj` 2.20、`down_proj` 2.23。
  若以剪枝後剩餘的 MLP 權重計算，實際約 2.7 bits/weight。

### 分析

* 為何選結構化剪枝：非結構化剪枝 20% 需要 mask / index（光是 bitmask 就 1 bit/weight），
  在 2.4 bits/weight 的總預算下不划算；結構化剪枝直接把矩陣縮小 20%，
  等於把量化預算放寬到 `2.4 / 0.8 ≈ 3.0` bits。
* MLP 是最適合剪枝的部位：`gate/up/down` 平均只需 2.16–2.23 bits（相對原始），
  而 `v_proj` 仍保持 4-bit（4.07）、`k_proj` 3.38，顯示 attention 的 value 路徑不能承受太多移除。
* 剪枝與量化都付出代價，但量化比較划算：只剪枝 20.3% 再加均勻 8-bit（CR 60.12%）
  的 PPL 是 15.14（+1.50），而同樣的 8-bit 用在未剪枝模型幾乎無損（約 13.7）。
  不過要從 85% 再往上，低 bit 量化會進入 2-bit 的災難區間；
  結構化剪枝先把 20% 的參數移除，讓剩下的權重可以用約 3 bits 表示，
  才能在 PPL 16.57 下達到 CR 85.10%。
* 消融（第 5.1 節）：把 prune ratio 提高到 25%（最終 CR 同樣是 85.10%）時，
  PPL 反而從 16.57 升到 16.69。多剪的 4.7% 雖然讓剩下的權重有更多 bit 可用，
  但額外的重建誤差更大，因此本 case 選擇 20.32%（剛好超過 20%），把預算留給量化精度。

## 5. 總結

### 5.1 消融實驗（`bash ablation.sh`，同一個 `run_eval.py`）

| 實驗 | 設定 | CR | PPL |
|---|---|---:|---:|
| 原始模型 | – | 0% | 13.638 |
| Case 1：RTN，`w4g128a`，無 RHT | 均勻 4-bit | 74.02% | 15.677 |
| Case 1：GPTQ，`w4g128a`，無 RHT | 均勻 4-bit | 74.02% | 13.843 |
| Case 1：GPTQ，`w4g128a`，RHT | 均勻 4-bit | 74.02% | 14.087 |
| Case 2：GPTQ，`w3g128a`，無 RHT | 均勻 3-bit | 80.32% | 14.665 |
| Case 2：GPTQ，`w3g128a`，RHT | 均勻 3-bit | 80.32% | 14.537 |
| Case 2：GPTQ，`w3cha`，RHT | per-channel 3-bit | 81.21% | 15.394 |
| Case 2：混合精度，無 RHT | rate 0.811 | 81.10% | 15.047 |
| Case 2：混合精度，無 2-bit 候選 | rate 0.811 | 81.10% | 14.661 |
| Case 3：global 能量 rank 分配 | SVD 20% | 50.02% | 30.740 |
| Case 3：SVD-only，因子 FP16（global 分配） | SVD 20% | 20.21% | 30.593 |
| Case 4：剪枝 25% + 混合量化 | rate 0.851 | 85.10% | 16.691 |
| Case 4：剪枝 20.3% + 均勻 8-bit | rate 0 | 60.12% | 15.144 |

> 觀察：
> (1) RTN → GPTQ 是最大的單一改善（4-bit 均勻：15.68 → 13.84）；
> (2) RHT 在 3-bit 有幫助（14.66 → 14.54）、在混合精度也有幫助（15.05 → 14.82），
>     但在 4-bit 均勻量化反而略差（13.84 → 14.09）；
> (3) 在這個預算下，2-bit 候選反而讓 allocator 做了較差的取捨：拿掉 2-bit 後 PPL 從 14.82 降到 14.66；
> (4) SVD 的 rank 分配策略影響極大：uniform 20.29 vs global 30.74；
> (5) 剪枝超過 20% 的邊際效益是負的（25% 的 16.69 比 20.3% 的 16.57 差）。

### 5.2 結論

* **CR 與 PPL 的取捨**：Case 1 的平均 3.94 bits/weight 幾乎無損（13.87，只比原始高 0.23）；
  降到 3.02 bits（Case 2）時 PPL 升到 14.82，再往下就需要 2-bit 與更細緻的混合精度分配。
  低秩分解的損失遠大於量化：SVD 的 20.22% 參數減少量已讓 PPL 增加 6.65；
  結構化剪枝 20.3% 加 8-bit（CR 60.12%）只增加 1.50。
* **敏感層**：`v_proj` 在四個 case 中都被分配最高精度（Case 1/2/4 都是 4-bit），`k_proj` 次之；
  `q_proj`、`o_proj` 與 MLP 的 `gate/up/down` 最魯棒，常被降到 2–3 bits。
  前幾層與最後幾層在敏感度分析中 KL 較高，因此 Case 4 直接跳過前 2 層與最後 1 層。
* **策略差異**：量化（GPTQ + RHT + 混合精度）在相同大小下品質最好；
  SVD 只適合低壓縮率，且需要混合 FP16 / 8-bit 才能剛好達到 50%；
  結構化剪枝適合與量化合用，用「移除整個 channel」換取 quantization budget。
* **Rank 分配**：SVD 的「每層保留相同參數比例」比「全域最小化丟棄能量」好 10.45 PPL。
  後者只看絕對能量，會把 rank 集中在少數大層；前者讓每層的截斷誤差平均，
  避免後段層的誤差被逐層放大。這是本次實驗中最大的單一改進（30.74 → 20.29）。
* **最有效的策略**：在本作業限制下，RHT + GPTQ + 依敏感度的混合精度最有效。
  它把 scale overhead 壓到最低，並把 bits 精確地放在最敏感的層；
  Case 1/2 的 PPL 分別只比原始高 0.23 / 1.18，卻減少了 75.5% / 81.1% 的目標層儲存。
* **限制**：使用 fake quantization，沒有實作 packed integer kernel；敏感度以單層 KL 近似，
  忽略層間交互；校正與評估同為 WikiText 分佈，其他資料上的泛化需另外驗證；
  混合 bit 寬在實際部署需要多種 kernel。SVD 與剪枝的實作也依賴校正資料的 Hessian，
  與 Part 3 的 calibration-free 設定不同。

### 5.3 學習心得

* 量化誤差不是均勻分佈的：少數 outlier 權重 / 通道主導誤差，RHT 與混合精度比單純降低 bit 更有效。
* 壓縮率的分母是「原始 16-bit 參數」，所以 scale / zero point / metadata 都必須計入；
  在 75–85% 的區間，這些 overhead 足以決定是否達標（例如 3-bit g128 的 80.32% vs 要求 81%）。
* 剪枝改變的是參數數量，量化改變的是每個參數的位元數；兩者結合可以在 PPL 與 CR 之間取得比單一方法更好的平衡。
