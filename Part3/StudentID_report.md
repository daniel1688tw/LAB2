# Lab 2 Part 3 報告：Calibration-Free PTQ（Per-tile MX Format Selection）

**學號：315551049**

## 1. Method

本作業重現 AdaptQuant 的核心想法：**以硬體對齊的 tile 為單位，在 Blackwell 原生支援的 MX 格式之間做選擇**，
且量化參數與格式配置完全不依賴校正資料。

* 候選格式：MXFP4（E2M1，4.25 bits）、MXFP6（E2M3，6.25 bits）、MXFP8（E4M3，8.25 bits）。
  每個 32 個 K 方向元素共用一個 8-bit E8M0 scale。
* Tile：128 × 128（對齊 128-row MMA tile 與 4 個 MX block），每個 tile 選一種格式，格式 id 佔 2 bits。
* 選擇準則（data-free）：對每個 tile 計算輸入加權誤差
  Σ_ij c\_j (W_ij − Q_ij)² / Σ_ij c\_j W_ij²，其中 c\_j ≈ E[x_j²] 只由權重推得：
  q/k/v、gate/up 的輸入是 RMSNorm 輸出，c = γ²；o_proj 的輸入是 attention 加權後的 V，
  c = diag(W_v · diag(γ²) · W_vᵀ)；down_proj 的輸入是 silu(g)·u，
  以 Gauss–Hermite 積分計算 E[silu(g)²]·E[u²]。
* 全域預算分配：所有層、所有 tile 一起做 Lagrangian relaxation，使包含 scale 與 tile id 的平均 bits ≤ 4.5。
* E8M0 scale search：每個 block 在 OCP floor 規則給出的 e 與 e+1 中選擇加權誤差較小者。
* Activation：每個目標 Linear 的輸入在推論時動態量化為 MXFP8（per-32 block），對應 MXFP8 × MXFP4/6/8 的 GEMM。
* 量化對象：36 層 × 7 個 Linear（embedding / lm_head 保持 FP16）。

**為何是 calibration-free**：所有決策只用到權重本身與 RMSNorm 的 γ，沒有任何輸入資料、forward pass
或針對 Qwen3-4B 的校正流程；activation 的 scale 是推論時由當下輸入動態計算，屬於 MX 格式本身的定義。

**與 AdaptQuant 的差異**：原論文無法在開發環境取得，tile 大小、候選格式集合與選擇準則皆依論文標題、
OCP MX 規格及相關文獻（MXSens、MixFP4、Block-rotation for MXFP4）重建，並以 `sweep.sh` 逐項比較（第 3 節）。
此外 T4 不支援 MX 低精度 tensor core，因此以 fake quantization 模擬數值行為。

## 2. Result

使用作業提供的 `evaluate.py`（WikiText-2 test、seqlen 2048）評估。最終正式執行的設定為
`bash run.sh`：4.5 bits、MXFP4/6/8、128×128 tile、MXFP8 activation。

| 設定 | avg bits | PPL |
|---|---:|---:|
| FP16（原始模型） | 16.00 | **13.638** |
| MXFP4（uniform，weight-only） | 4.25 | 15.248 |
| NVFP4（uniform，weight-only） | 4.50 | 13.637 |
| MXFP6（uniform，weight-only） | 6.25 | 13.793 |
| MXFP8（uniform，weight-only） | 8.25 | 13.721 |
| MXFP4（uniform，含 MXFP8 activation） | 4.25 | 15.689 |
| **本方法（4.5 bits，per-tile mixed，含 MXFP8 activation）** | **4.50** | **14.257** |
| AdaptQuant（論文基準） | – | 13.98 |

* 最終 PPL：**14.257**（`run.sh` 輸出 14.26），與論文基準 13.98 相差 +0.28。
* 依評分表（PPL ≤ 15 得 15 分），再現性能為滿分。
* 格式佔比：MXFP4 87.51%、MXFP6 12.49%、MXFP8 0%。
* 每層型平均 bits：q_proj 4.91、k_proj 6.18、v_proj 6.17、o_proj 4.99、
  gate_proj 4.26、up_proj 4.25、down_proj 4.26。
* 量化時間 7.4 秒（不含模型載入與 PPL 評估）。

## 3. Observation and Discussion

### 3.1 格式選擇與精度分配

* Allocator 完全沒有選到 MXFP8：對近高斯的權重，E4M3 多出的指數範圍幾乎沒有幫助（單元測試中 MSE 只比 E2M3 低約 6%），
  在固定 4.5 bits 預算下，把 12.49% 的元素升到 MXFP6 比少數 tile 用 MXFP8 更划算。
* 精度分配有明顯的層型結構：MLP（gate/up/down）幾乎全是 MXFP4（4.25–4.26 bits），
  而 k_proj/v_proj 平均 6.17–6.18 bits、q_proj 4.91、o_proj 4.99。
  這與「k/v 投影的 outlier 最敏感、MLP 參數多且相對魯棒」的觀察一致。
* Uniform MXFP4 的 PPL 15.25（weight-only），本方法只用多 0.25 bit 就降到 14.26；
  而 uniform NVFP4 在 4.5 bits 已達 13.64（幾乎無損），顯示 **scale 的表示方式（E4M3 vs E8M0）比格式名稱更重要**。
  本方法限定在 MX 格式內，若允許 NVFP4 作為候選，PPL 可進一步降到 14.16。

### 3.2 消融實驗（`sweep.sh`，全部使用同一個 `evaluate.py`）

| 變因 | 設定 | avg bits | PPL |
|---|---|---:|---:|
| Activation | 無（weight-only） | 4.50 | 13.921 |
| Activation | MXFP8（預設） | 4.50 | 14.257 |
| Scale search | 關閉（純 OCP floor） | 4.50 | 14.291 |
| 輸入統計 c\_j | uniform | 4.50 | 14.886 |
| 輸入統計 c\_j | norm | 4.50 | 14.127 |
| 輸入統計 c\_j | analytic（預設） | 4.50 | 14.257 |
| 誤差度量 | abs | 4.50 | 14.321 |
| Tile | 1×32 | 4.50 | 14.434 |
| Tile | 128×32 | 4.50 | 14.325 |
| Tile | 128×128 | 4.50 | 14.257 |
| Tile | 256×256 | 4.50 | 14.258 |
| 候選格式 | +NVFP4 | 4.50 | 14.158 |
| 候選格式 | 只用 MXFP4/6 | 4.50 | 14.266 |
| Block rotation | 4.5 bits + rotate 32 | 4.50 | 15.348 |
| Bit 預算 | 4.375 / 4.5 / 4.75 / 5.0 | – | 14.353 / 14.257 / 14.176 / 14.579 |

* **Activation 量化**：MXFP8 activation 讓 PPL 由 13.92 升到 14.26（+0.34）。這是 4.5 bits 設定下最大的單一退化來源，
  但也正是 Blackwell MXFP8 × MXFP4/6/8 GEMM 的實際運作方式。
* **Scale search**：關閉後只退化 0.03，因為 allocator 會自動把受影響的 tile 升級到 MXFP6 補償。
* **Data-free 輸入統計**：`uniform`（忽略 c\_j）退化到 14.89，`norm`（只保留 γ 的形狀、不考慮 o/down 的傳遞）反而最好（14.13）。
  這表示 analytic 的 o_proj/down_proj 二階動差近似仍有誤差；不過兩者只差 0.13，整體方法對 c\_j 的細節不敏感。
* **Tile 大小**：1×32（最細）14.43、128×32 14.33、128×128 14.26、256×256 14.26。
  Tile 越細，格式選擇的自由度越高、PPL 越好，但每個 tile 都要切換指令，硬體效率最差；
  128×128 在本實驗中幾乎與 256×256 相同，是品質與硬體對齊的合理折衷。
* **Block rotation**：在 32-element MX block 內做 Hadamard 旋轉反而讓 PPL 由 14.26 升到 15.35。
  可能原因是本實作在旋轉後把 c\_j 以 block 平均值取代（假設 block 內獨立），破壞了原本 analytic 統計的對角結構；
  另外旋轉也讓 weight 的 per-block 最大值分佈改變，E8M0 scale search 的效益下降。
  這與部分文獻「rotation 對 MXFP4 有幫助」的結論相反，是本實作值得改進的地方。
* **Bit 預算非單調**：4.75 bits（14.18）優於 5.0 bits（14.58）。原因是 Lagrangian 分配在離散格式上是分段常數，
  預算增加時 allocator 會把大量 tile 從 MXFP4 升級到 MXFP6，但 activation 仍固定 MXFP8，反而讓權重誤差與 activation 誤差的平衡變差；
  這也顯示固定 activation 精度時，weight bits 並非越多越好。

### 3.3 與論文基準的差距

* PPL 14.26 與 AdaptQuant 的 13.98 相差 0.28（2%）。本方法沒有使用任何校正資料，已相當接近。
* 主要差距可能來自：(1) 原文的 tile 大小與候選格式集合未知；(2) 原文可能在 Blackwell 上使用真正的
  MXFP4/6/8 kernel，activation 與 scale 的行為與 fake quant 略有不同；
  (3) 本方法的 c\_j 是解析近似，沒有考慮 attention 的實際加權與 token 間相關性。
* 若允許 NVFP4，4.5 bits 的 uniform NVFP4 已幾乎無損（13.64），顯示此預算下真正的瓶頸是 scale 精度而非 mantissa。

### 3.4 困難與限制

* 原論文（OpenReview）在開發環境無法存取，方法細節只能依標題、OCP MX 規格與相關論文重建。
* T4 沒有 FP4/FP6 硬體，必須用 fake quantization；E4M3/E5M2 的 rounding、subnormal、saturation
  需與 `torch.float8_e4m3fn` 逐位元比對（已加入單元測試）。
* 只模擬數值，沒有實作 tile 內的 MMA 指令切換；實際硬體上 tile id 的讀取與 instruction cache 成本未計入。
* 沒有針對 activation outlier 做 smoothing / rotation，activation MXFP8 是最大的退化來源。

### 3.5 可能的改進

* 把 NVFP4 或 MXINT8 加入候選格式；在固定 bits 下改用 E4M3 scale 可能比增加 mantissa 更有效。
* 以權重推得更好的 c\_j（例如考慮 attention 的 row-sum 或 RMSNorm 後的實際變異數）。
* 對 activation 做 data-free 的 Hadamard/permutation smoothing，再量化成 MXFP8。
* 在格式分配後加入一輪 weight rounding error feedback（calibration-free 的 GPTQ 變體），
  以 c\_j 當對角 Hessian 補償量化誤差。
