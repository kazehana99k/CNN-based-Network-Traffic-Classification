# 実験ノート / Experiment Notebook

このドキュメントは、卒業研究の各改善実験の動機・実装・結果を詳しく書いたもの。
README の補足として、興味のある実験だけ拾い読みする想定。

---

## 目次

1. [オリジナル卒業研究 (2022)](#1-オリジナル卒業研究-2022)
2. [原作で残された問題](#2-原作で残された問題)
3. [改善実験](#3-改善実験)
   - 3.1 再現とバグ修正
   - 3.2 2D vs 1D CNN
   - 3.3 Dilated 1D CNN
   - 3.4 ATTACK クラス専門対策 (失敗の連続)
   - 3.5 GPU 移行 (TF → PyTorch)
   - 3.6 Batch sweep
4. [教訓](#4-教訓)

---

## 1. オリジナル卒業研究 (2022)

### タスク

12 クラスのネットワークトラフィック分類:

| クラス | 例 |
|------|---|
| WWW | http |
| MAIL | imap, pop, smtp |
| FTP-CONTROL / FTP-PASV / FTP-DATA | ftp |
| ATTACK | Internet worms, virus attacks |
| P2P | KazaA, BitTorrent, Nutella |
| DATABASE | postgres, sqlnet, oracle |
| MULTIMEDIA | Windows media player |
| SERVICES | dns, ntp, ldap |
| INTERACTIVE | ssh, telnet |
| GAMES | half-life |

### 手法

- データ: Moore Dataset 10 ファイル (entry01 ~ entry10)
- 各サンプル 248 次元 → 8 次元ゼロパディング → **256 次元** → **16×16** に reshape
- モデル (TF/Keras):
  ```
  Conv2D(8, 3×3, ReLU) → MaxPool(2×2)
  Conv2D(16, 3×3, ReLU) → MaxPool(2×2)
  Flatten → Dense(256) → Dense(128) → Dense(12, Softmax)
  ```
- 比較: CNN, BP NN, KNN, Naive Bayes, SVM, Decision Tree

### 論文の結果

| Algorithm | Accuracy | Time(s) |
|-----------|---------:|--------:|
| **CNN** | **99.58%** | 214.95 |
| BP | 99.52% | 37.7 |
| KNN | 99.01% | 313 |
| Decision Tree | 99.37% | 213 |
| SVM | 97.97% | 3159 |
| Naive Bayes | 53.87% | < 2 |

ATTACK クラスの認識率は 70% 前後で頭打ち。論文自身も「攻撃パケットは他種ポートに偽装するため現特徴量では判別困難」と将来課題に残した。

---

## 2. 原作で残された問題

4 年越しに自分のコードを読み返して気付いた点。後続の各改善実験はそれぞれこの問題に対応する。

### 2.1 コードバグ — クラスラベル破壊
データ行全体に `i.replace('N','0')` を適用していたため、ラベル `FTP-CONTROL` と `INTERACTIVE` の中の `N` まで `0` に置換され、再現時にこれら 2 クラスのサンプル数が 0 になった。
→ ラベル列とフィーチャー列を分離してから置換 (`data_preprocess.py:parse_arff_line`)

### 2.2 2D Reshape の擬似近傍問題
248 次元の統計量を 16×16 に強制 reshape していたが、`feature[0]` と `feature[16]` が 2D 近傍として扱われるのは意味的に偶然でしかない。ポート/パケット数/サイズ/時間/フラグといった本来異質な特徴を、2D Conv が無関係な位置で混ぜていた。

### 2.3 ATTACK クラスの上限問題
全アルゴリズム共通で ATTACK の認識率が ~70% で頭打ち。論文では具体的な解決策が提示されなかった。

### 2.4 バッチサイズの未検証
batch=128 が最良という主張に対し、論文の検証は 64/256/1024 の 3 点のみで、系統的なスイープなし。

### 2.5 GPU 未活用
TensorFlow-CPU 2.8.0 を「公平性のため」と称して CPU で実行。

### 2.6 不均衡データ処理の粗雑さ
ガウス白色雑音による mean-filling は MULTIMEDIA/P2P/DATABASE のみ。ATTACK には何も拡張せず、不均衡のまま投入。

---

## 3. 改善実験

### 3.1 再現とバグ修正

- Python 3.12 + TensorFlow 2.21 で再構築 (TF 2.8 は Windows + Python 3.12 と非互換)
- ラベルバグ修正
- Moore データ 10 ファイル (約 94 MB 圧縮) を Cambridge 公式から取得
- entry12 (~12 MB の追加サンプル) も外部テストセットとして取得

**再現結果** (25 epochs, batch=128):

| Algorithm | Accuracy | Time(s) |
|-----------|---------:|--------:|
| CNN (2D) | 98.34% | 37.38 |
| BP | 95.82% | 22.79 |
| KNN | 98.39% | 6.34 |
| **Decision Tree** | **99.28%** | 35.68 |
| SVM (subset=20k) | 90.29% | 26.67 |
| Naive Bayes | 69.49% | 0.46 |

論文の数値と多少ズレるが、CNN/決定木/KNN が高精度・Naive Bayes が低精度という大局は再現できた。

### 3.2 2D CNN vs 1D CNN

**仮説**: 1D Conv のほうが線形シーケンス的特徴に適しているはず。

```python
# 1D 版アーキテクチャ
Conv1D(8,  k=5) → MaxPool1D(p=4)
Conv1D(16, k=5) → MaxPool1D(p=4)
Flatten → Dense(256) → Dense(128) → Dense(12)
```

パラメータ予算を 2D 版に揃えて比較 (25 epochs):

| Model | Internal Acc | External (entry12) Acc | Time |
|-------|-------------:|----------------------:|-----:|
| 2D CNN (元論文) | 98.86% | 96.65% | 36.3s |
| **1D CNN** | **99.11%** | **97.10%** | 35.7s |

クラス別で見ると、FTP-PASV (+3.04%)、INTERACTIVE_ (+4.0%) など小サンプルクラスで 1D が顕著に優位。**仮説は支持された** — 2D は意味的に無関係な近傍を畳み込んでいた。

### 3.3 Dilated 1D CNN

**動機**: 通常の Conv1D(k=5) では受容野が 5 セルしかない。Dilated Conv で複数の意味グループ (ポート + パケット数 + サイズ...) を同時に見たい。

```python
# 設計
Conv1D(16, k=3, dilation=1)   # 受容野 3
Conv1D(32, k=3, dilation=2)   # 受容野 7
Conv1D(32, k=3, dilation=4)   # 受容野 15
```

**最初の失敗**: 最終層を GlobalAveragePooling にしたところ、256 位置を平均することで信号が薄まり、Accuracy **82.26%** に墜落。

**修正版**: GAP → MaxPool1d 段階的縮小に置換 → **99.09%** (internal) / **97.42%** (external) で全変種中トップに。

**学び**: アーキテクチャ図に書かれない「最後の 1 行」が結果を決める。

### 3.4 ATTACK クラス専門対策 — 失敗の連続

問題 2.3 への対応として 6 つのアプローチを試した:

| # | 方法 | 内容 | ATTACK F1 |
|---|------|------|----------:|
| 1 | Weighted CE (ATTACK×5) | クラス重み付けクロスエントロピー | 49.69% |
| 1' | Weighted CE (ATTACK×20) | 重みを増やす | 42.41% |
| 1'' | Focal Loss (γ=2) | 難しいサンプルへ自動的に重み付け | 54.03% |
| 2 | Two-stage classifier | binary (is_attack) + 11-class | 43.72% |
| 6 | Two-stage + ポート列ゼロ化 | ポート特徴を消す | 59.20% |
| - | **Plain CE baseline** | 何もしない | **75.89%** ⭐ |

**結論**: 全部 baseline に負けた。

特に注目: ポート関連の派生特徴 (well-known port indicator, log scale) を追加したら ATTACK Precision が 62% → 46% に**崩壊**。攻撃パケットは「ポートを偽装する」からこそ ATTACK なのであり、ポートを目立たせることはモデルを欺かれやすくする。

### 3.5 GPU 移行 (TF → PyTorch)

TF 2.11+ は Windows native での GPU サポートを停止しており、RTX 5090 (Blackwell sm_120) を活かす選択肢:

- WSL2 + Linux TF (要セットアップ)
- TensorFlow-DirectML (TF 2.10 限定)
- **PyTorch 2.7 + CUDA 12.8** ← 採用

`models_torch.py` で同じアーキテクチャを再実装。さらに最適化:

1. **データ全体を GPU メモリに常駐** (130k × 256 floats ≈ 130 MB)
2. **GPU 上で permutation インデックス生成** (CPU-GPU 転送を排除)
3. **`set_to_none=True` for `zero_grad()`**

結果: 学習時間は元の **約 1/12** に短縮。ただし batch 設定によっては GPU の旨味が出ない (次節参照)。

### 3.6 Batch Sweep — 最大の発見

**仮説**: 大バッチ (1024+) は GPU をフル活用できて速い、その上で精度も維持できるはず。

**実測** (Dilated CNN1D, 25 epochs):

| Batch | Overall | ATTACK-P | ATTACK-R | **ATTACK-F1** | Time |
|------:|--------:|---------:|---------:|--------------:|-----:|
| 128 | 98.83% | 98.32% | 67.28% | **79.89%** ⭐ | 46.3s |
| 256 | 98.54% | 98.21% | 63.36% | 77.03% | 22.7s |
| 512 | 95.22% | 74.48% | 65.90% | 69.93% | 11.7s |
| 1024 | 90.91% | 53.48% | 65.44% | 58.86% | 5.8s |
| 2048 | 87.89% | 50.00% | 6.91% | 12.15% | 3.6s |

完全に単調な曲線。**仮説は否定された** — 大バッチで精度を維持できるどころか、ATTACK の認識率が壊滅的に崩壊。

**理由**: 大バッチは「シャープな極小値」を見つける性質があり、これは多数派クラスでは汎化するが、少数派クラス (ATTACK は train の 1.3%) のロス信号がバッチ平均に埋もれて学習されなくなる。

**結論**: Section 3.4 の loss-function 系列の試行は全部不要だった。バッチサイズだけ正しく選べば良かった。

---

## 4. 教訓 / Lessons Learned

4 年前の自分への手紙として:

### 4.1 表形式データに 2D CNN を無理に使わない
セルとセルの空間的関係が意味を持つのは画像だけ。統計特徴量をパディングして正方形にしても、それは画像ではない。1D Conv なら少なくとも順序情報を尊重する。

### 4.2 小モデル + 小データでは GPU が必ずしも速くない
RTX 5090 + batch=128 ですら、batch=2048 と比べて GPU 利用率は低い。Kernel-launch オーバーヘッドが計算時間を支配するため。「GPU を使う = 速い」は単純すぎる。

### 4.3 バッチサイズは不均衡データの最強チューニングノブ
Focal Loss、SMOTE、Two-Stage、Class Weight — 全部試したが、batch を 2048 → 128 にする 1 行の変更が一番効いた。

### 4.4 「特徴量を強化する」が逆効果になることがある
ATTACK 検知でポートの log 変換や well-known port 指示子を追加したら、ATTACK Precision が 62% → 46% に崩壊。「強化」が悪化させるドメインがある。

### 4.5 Baseline を超えてから「改善」と言う
Section 3.4 の loss-function トリックは個別に見れば筋が通っていたが、結局 plain baseline に勝てなかった。新規手法を提案する前に baseline の上限を本当に知っているか確認する必要がある。

### 4.6 Pooling の細部が結果を決めることがある
Dilated CNN の初版は GlobalAveragePooling で 82% に沈んだが、MaxPool に変えるだけで 99% に復活。アーキテクチャ図に書かれない「最後の 1 行」が往々にして決定的。

---

## References

[1] Moore, A. W., & Papagiannaki, K. (2005). Toward the accurate identification of network applications. *PAM*.
[2] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT press.
[3] Lin, M., Chen, Q., & Yan, S. (2013). Network in network. *arXiv:1312.4400*.
[4] Yu, F., & Koltun, V. (2015). Multi-scale context aggregation by dilated convolutions. *arXiv:1511.07122*.
[5] Lin, T. Y. et al. (2017). Focal loss for dense object detection. *ICCV*.

---

*Last updated: 2026-05*
