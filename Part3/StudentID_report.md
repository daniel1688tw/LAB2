# Lab 2 Part 3 報告：Calibration-free PTQ（Per-tile MX format selection）

> 草稿：`【待填】` 請以 `bash run.sh` 與 `bash sweep.sh`（結果在 `logs/results.jsonl`）的實際數值填入，完成後輸出 PDF。

## 1. Method

本作業重現 AdaptQuant 的核心想法：**以硬體對齊的 tile 為單位，在 Blackwell 原生支援的 MX 格式之間做選擇**，
且所有決策都不需要校正資料。

* 候選格式：MXFP4（E2M1，4.25 bits）、MXFP6（E2M3，6.25 bits）、MXFP8（E4M3，8.25 bits），每 32 個 K 方向元素共用 E8M0 scale。
* Tile：128 × 128（對齊 MMA tile 與 4 個 MX block），每 tile 一種格式（2-bit 格式 id）。
* 選擇準則（data-free）：`Σ c_j (W−Q)²_ij / Σ c_j W²_ij`，`c_j ≈ E[x_j²]` 由權重推得（RMSNorm γ²、
  `diag(W_v γ² W_vᵀ)`、`E[silu(g)²]E[u²]`）；全模型所有 tile 一起以 Lagrangian 分配在 4.5 bits/weight 的預算內。
* E8M0 scale：每個 block 在 OCP 規則的 `e` 與 `e+1` 中選誤差小者（避免 MXFP4 的 block 最大值被截斷）。
* Activation：動態 MXFP8（per-32 block）。
* 為何 calibration-free：所有參數只依賴權重本身（含 RMSNorm 權重），沒有任何 forward pass 或資料；
  activation 的 scale 是推論時動態計算，屬於 MX 格式本身的定義。
* 與 AdaptQuant 的差異：原文無法取得，tile 大小、候選格式、選擇準則為依標題與 MX 規格的重建（見 README 第 4 節），
  並用 `sweep.sh` 比較各選項。

## 2. Result

| 設定 | avg bits | PPL |
|---|---:|---:|
| FP16 | 16 | 【待填】 |
| MXFP4（uniform, weight-only） | 4.25 | 【待填】 |
| NVFP4（uniform, weight-only） | 4.50 | 【待填】 |
| MXFP6（uniform, weight-only） | 6.25 | 【待填】 |
| **本方法（4.5 bits, W-mixed / A-MXFP8）** | 4.50 | **【待填】** |
| AdaptQuant（論文） | – | 13.98 |

最終 PPL：【待填】

## 3. Observation and Discussion

* **格式比較**：MXFP6（E2M3）與 MXFP8（E4M3）都有 3 個 mantissa bits，在共享 scale 的 block 內，E4M3 多出的指數範圍
  對近高斯的權重幾乎沒有幫助（單元測試中 MSE 只差約 6%），所以 allocator 主要在 MXFP4 ↔ MXFP6 之間取捨，
  MXFP8 只留給含極端 outlier 的 tile。格式佔比：【待填，log 的 format share】。
* **MXFP4 的主要誤差來源**：power-of-two scale 讓 block 的有效範圍浪費最多一半；OCP 的 floor 規則又會把最大值截到 6。
  scale search（e / e+1）在不增加任何儲存的前提下降低誤差：PPL【待填 no-scale-search vs 預設】。
* **哪裡需要較高精度**：每層型 / 每層的平均 bits（log）：【待填】。預期 `down_proj`（輸入為 SwiGLU 乘積，有大 outlier）、
  `v_proj/o_proj` 與前幾層拿到較多 MXFP6/8。
* **Data-free 權重 (c_j) 的影響**：analytic vs norm vs uniform：【待填】。
* **Tile 大小**：1×32（block 級，硬體最不友善）vs 128×32 vs 128×128 vs 256×256：【待填】；越細的 tile 品質越好但
  kernel 需要在 tile 間切換指令種類。
* **Activation 量化**：MXFP8 activation 的影響：【待填 act none vs mxfp8】。
* **困難**：原論文無法取得；T4 無 FP4/FP6 硬體，只能用 fake quant；E4M3 等 minifloat 的 rounding / subnormal / saturation
  需要與 `torch.float8_e4m3fn` 逐位元比對才確定正確。
* **PPL 退化的可能原因**：MXFP4 tile 上的 rounding 噪音累積；power-of-two scale 的量化步階過粗；
  data-free 的 `c_j` 只是近似（忽略 attention 的平均效果與 token 間相關性）。
* **可能的改進**：在 MX block 內加入 Hadamard 旋轉（block rotation）讓 MXFP4 更接近高斯；
  在格式集合中加入 NVFP4 或 MXINT8；用 Hessian-free 的 GPTQ 變體（如以 c_j 為對角 Hessian 的誤差回饋）。
