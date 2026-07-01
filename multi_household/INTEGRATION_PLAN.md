# 整合計畫 — 我們的系統為主幹,吸收管少棋的協調引擎

> **定調(2026-06-30):我們的 `multi_household/` 是主幹**;管少棋的 `coordinator/`
> 是「拿來跑、比效果」的最佳化協調方法。**不需要跟他對介面** —— 他的引擎已在
> `coordinator/`,自包含可跑。目標是**一份 REFIT 資料上,把兩套協調 + oracle 放同一張表**。

## 兩個層次(先做 Level 1,Level 2 選做)

### 🟢 Level 1 — 結果層整合(低工、高價值,先做)
不改他的 code,把兩套都在 REFIT 上跑,**結果放同一張比較表**。

- **我們的**(主幹):rule-based 廣播協調 + LLM 顧問 + 閉環 → peak/P95(對齊窗)
- **他的**(比較):shadow-price 協調 + oracle → PAR/削峰(他的社區)
- 論文敘事:**我們 = 人在迴路主系統**;**他的 = 進階最佳化協調 baseline**;oracle = 上界。

**產出:控制器階梯表**
`No-DR ｜ Rule-based(我們,可部署)｜ Shadow-price(他,近最優)｜ Oracle(上界)`

### 🔵 Level 2 — 程式層 fuse(高工、選做,之後)
把他的協調器接進我們的 pipeline(輸出翻成我們的 Recommendation → 餵 LLM 顧問 + 閉環)。多週工程,等 Level 1 數字漂亮再決定要不要。

## 步驟(Level 1)

| # | 做什麼 | 狀態 |
|---|---|---|
| 1 | 他的引擎跑我們 REFIT(3 戶 phase4b 已驗證 −32% PAR、72% oracle) | ✅ 部分完成 |
| 2 | **訓練 17 戶 phase2 模型** → 跑他的 phase4d 社區評估(規模化效果) | ⏳ 進行中 |
| 3 | **算我們系統的 PAR**(用同一份資料),跟他的 + oracle 併一張表 | ☐ |
| 4 | 統一敘事:同一份 REFIT、標清楚兩種社區建構的差別 | ☐ |
| 5 | (選)Level 2 深度 fuse:他的協調 → 我們的 LLM 顧問 | ☐ 之後 |

## 關鍵誠實點

- **兩套「社區建構」不同**:我們用真實對齊窗、他用合成社區 → **同一張表要標清楚,或統一到我們的對齊窗**(建議後者,較嚴謹)。
- **兩套 forecaster**:主幹用我們的;他的協調暫用他的 baseload LSTM(Level 2 再統一)。
- **我們的 rule 協調不是「爛」,是 baseline** —— reviewer 要看「rule vs 最佳化 vs oracle」。

## 一句話

我們主系統(人在迴路 + LLM + 閉環)為核心 novelty;他的 shadow-price + oracle 補上「最佳化協調 + 最優上界」的硬對照。兩個放一份 REFIT 比一比 = Q1 骨架。
