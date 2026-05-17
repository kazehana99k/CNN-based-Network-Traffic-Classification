# CNN ベースのネットワークトラフィック分類

**卒業研究の再現と PyTorch/GPU 拡張**
**(中) 基于 CNN 的网络流量分类：毕业研究的复现与扩展**
**(EN) CNN-based Network Traffic Classification — Reproduction & Extension of a BISTU Thesis**

---

## 0. ひと言で言うと / TL;DR

2022 年に北京情報科技大学 (BISTU) で書いた自分の学部卒業研究を、4 年経って自分で再現・批判的に検証し直したプロジェクト。当時の TensorFlow-CPU の 2D CNN ベースラインから出発し、最終的に **Dilated 1D CNN + PyTorch GPU + batch=128** という構成で、論文では到達できなかった **ATTACK クラス F1 = 79.89%** を達成した。

The most surprising finding: バッチサイズが loss 関数や two-stage architecture よりも遥かに大きな影響を持つ — batch を 2048 から 128 に下げるだけで ATTACK F1 が +67.7 ポイント (12.15% → 79.89%) 改善する。

---

## 1. プロジェクトの位置づけ / 项目背景

| 項目 | 内容 |
|------|------|
| 原典 (原作) | 王世元「基于 CNN 的网络数据流量分类」(BISTU, 電信 1801 班, 学号 2018010983) |
| 起止時間 | 2022 年 2 月 21 日 ~ 6 月 10 日 |
| 指導教官 | 沈冰夏 先生 |
| データセット | Moore Dataset (Cambridge, 2005) |
| 拡張時期 / 改进时期 | 2026 年 5 月 (4 年後の自己レビュー) |

このリポジトリは **2 つの層** からなる:

1. **オリジナル卒研の忠実な再現** — 論文に書かれた前処理・モデル・実験を Python で再実装 (`models_dl.py`, `models_ml.py` 等)
2. **批判的拡張** — 当時気付かなかった設計上の問題点を 4 年越しで指摘し、新しい実験で検証 (`models_torch.py`, `ablation.py`, `attack_experiments.py`, `batch_sweep.py`)

---

## 2. オリジナルの卒業研究 (2022)

### 2.1 タスク

12 クラスのネットワークトラフィック分類:

| クラス | 应用 / Apps |
|------|----------|
| WWW | www |
| MAIL | imap, pop2/3, smtp |
| FTP-CONTROL / FTP-PASV / FTP-DATA | FTP |
| ATTACK | Internet worms, virus attacks |
| P2P | KazaA, BitTorrent, Nutella |
| DATABASE | postgres, sqlnet, oracle, ingres |
| MULTIMEDIA | Windows media player, real |
| SERVICES | X11, dns, ident, ldap, ntp |
| INTERACTIVE | ssh, klogin, rlogin, telnet |
| GAMES | half-life |

### 2.2 手法 / 方法

**データ前処理** (詳細は `data_preprocess.py` 参照):

1. Moore データセットの 10 個の ARFF ファイル (entry01 ~ entry10) を読み込み
2. 1 サンプルあたり 248 次元の数値特徴を抽出、欠損値 (`?`) は平均値で補完
3. 248 次元 + ゼロパディング 8 次元 = **256 次元**
4. **16 × 16** に reshape して 2D CNN へ入力

**モデル** (論文 Fig. 4.3 を忠実に再現):

```
Input(16×16×1)
 → Conv2D(8, 3×3, ReLU)  → MaxPool(2×2)
 → Conv2D(16, 3×3, ReLU) → MaxPool(2×2)
 → Flatten
 → Dense(256, ReLU) → Dense(128, ReLU) → Dense(12, Softmax)
```

**比較対象 / 6 アルゴリズム**: CNN, BP Neural Network, KNN, Naive Bayes, SVM, Decision Tree

**実装環境**: PyCharm + TensorFlow-CPU 2.8.0 + Keras + sklearn (AMD 5800X, AVX-256)

### 2.3 論文の結果 / 原论文结果

| Algorithm | Accuracy | Time(s) |
|-----------|---------:|--------:|
| **CNN** (2D) | **99.58%** | 214.95 |
| BP Neural Net | 99.52% | 37.7 |
| KNN | 99.01% | 313 |
| Decision Tree | 99.37% | 213 |
| SVM | 97.97% | 3159 |
| Naive Bayes | 53.87% | < 2 |

ATTACK クラスについては「攻击包通常通过伪装成其他的端口...让根据这类特征进行训练的模型发生了误判」(攻撃パケットは他種のポートに偽装するため、現特徴量での判別は困難) と論文自身が認めており、ここに改善の余地が残されていた。

---

## 3. 原作で残された問題 / 原作存在的问题

4 年越しに自分のコードを読み返して気付いた / 当時隠れていた問題点。後続の改善実験 (Section 4) はそれぞれこの問題に対応する。

### 3.1 コードのバグ — クラスラベル破壊
`i.replace('N','0')` をデータ行全体に適用していたため、ラベル `FTP-CONTROL` と `INTERACTIVE` の中の `N` まで `0` に置換されてしまい、再現時にこれら 2 クラスのサンプル数が 0 になる事故が発生。
→ **修正**: ラベル列とフィーチャー列を分離してから置換 (`data_preprocess.py:parse_arff_line`)

### 3.2 2D Reshape の擬似近傍問題
248 次元の統計量シーケンスを 16×16 に強制 reshape していたが、`feature[0]` と `feature[16]` が 2D 近傍として扱われるのは意味的に偶然でしかない。ポート番号、パケット数、サイズ、時間統計、フラグといった本来異質な特徴を、2D Conv が無関係な位置で混ぜていた。

### 3.3 ATTACK クラスの上限問題
全アルゴリズム共通で ATTACK の認識率が ~70% で頭打ち。論文では「需要思考如何寻找合适的特征」と将来課題に残したまま。

### 3.4 バッチサイズの未検証
batch=128 が最良と論文は主張するが、検証は 64/256/1024 の 3 点のみ。系統的なスイープなし。

### 3.5 GPU 未活用
TensorFlow-CPU 2.8.0 を「公平性のため」と称して CPU で実行。実は当時 RTX シリーズ等 GPU を使えば学習時間を大幅短縮できた。

### 3.6 不均衡データ処理の粗雑さ
ガウス白色雑音による mean-filling は MULTIMEDIA/P2P/DATABASE のみ。ATTACK には何も拡張せず、不均衡のまま投入。

---

## 4. 改善実験 / 改进实验

### 4.1 再現とバグ修正 — Reproduction

- Python 3.12 + TensorFlow 2.21 で再構築 (TF 2.8 は Windows + Python 3.12 と互換性なし)
- Section 3.1 のラベルバグを修正
- Moore データセット 10 ファイル (約 94 MB 圧縮) を [Cambridge official site](https://www.cl.cam.ac.uk/research/srg/netos/projects/archive/nprobe/data/papers/sigmetrics/index.html) からダウンロード
- 加えて entry12 (1 年後追加収集された ~12 MB の追加サンプル) も保留外部テストセットとして取得

**再現結果 (6 algorithms, 25 epochs, batch=128):**

| Algorithm | Accuracy | Time(s) |
|-----------|---------:|--------:|
| CNN (2D) | 98.34% | 37.38 |
| BP Neural Net | 95.82% | 22.79 |
| KNN | 98.39% | 6.34 |
| **Decision Tree** | **99.28%** | 35.68 |
| SVM (subset=20k) | 90.29% | 26.67 |
| Naive Bayes | 69.49% | 0.46 |

論文の数値とは多少ズレるが (TF バージョン差・データ量差・乱数差)、CNN・決定木・KNN が高精度、Naive Bayes が低精度という大局は再現。

### 4.2 2D CNN vs 1D CNN — 擬似近傍問題の検証

問題 3.2 への対応。**仮説**: 1D Conv のほうが線形シーケンス的特徴に適しているはず。

```
1D CNN: Conv1D(8, k=5) → MaxPool1D(p=4) → Conv1D(16, k=5) → MaxPool1D(p=4) → FC...
```

パラメータ予算をできるだけ 2D 版に揃えて比較:

| Model | Internal Acc | External (entry12) Acc | Training Time |
|-------|-------------:|----------------------:|--------------:|
| 2D CNN (元論文) | 98.86% | 96.65% | 36.3s |
| **1D CNN** | **99.11%** | **97.10%** | 35.7s |

クラス別では、FTP-PASV (+3.04%)、INTERACTIVE_ (+4.0%) など小サンプルクラスで 1D が顕著に優位。**仮説は支持された** — 2D は意味的に無関係な近傍を畳み込んでいた。

### 4.3 Dilated 1D CNN — 受容野の拡大

```
Conv1D(16, k=3, dilation=1)  → MaxPool
Conv1D(32, k=3, dilation=2)  → MaxPool
Conv1D(32, k=3, dilation=4)  → MaxPool
→ Flatten → FC
```

実効受容野は 3 → 7 → 15 と指数的に拡大。

**最初の失敗**: GlobalAveragePooling を最終層に使ったところ、全 256 位置を平均することで信号が完全に薄まり、Accuracy は **82.26%** まで墜落。

**修正版**: GAP → MaxPool1d に置換すると **99.09%** (internal) / **97.42%** (external) で全変種中トップに。**設計上の細部 (pooling の選択) が結果を決める** という典型例。

### 4.4 ATTACK クラスの専門対策 — 失敗の連続

問題 3.3 への対応。6 つのアプローチを試した:

| # | 方法 | ATTACK F1 | 結果 |
|---|------|----------:|------|
| 1 | Weighted CE (ATTACK×5) | 49.69% | Baseline より悪化 |
| 1' | Weighted CE (ATTACK×20) | 42.41% | Precision 崩壊 (30%) |
| 1'' | Focal Loss (γ=2) | 54.03% | わずかに改善 |
| 2 | Two-stage classifier | 43.72% | Stage 1 が過剰反応 |
| 6 | Two-stage + ポート列ゼロ化 | 59.20% | ポートが ATTACK のミスリードであることを裏付け |
| ? | **Plain CE baseline** | **75.89%** | **驚くべきことに何もしないのが最善** |

**学び**: 「ATTACK だから特別扱い」は人間の直感だが、データはそれを支持しなかった。バランス調整は逆に Precision を破壊する。

### 4.5 GPU 移行 — TensorFlow → PyTorch

TF 2.11+ は Windows native での GPU サポートを停止しており、RTX 5090 (Blackwell sm_120) を活かすには:

- WSL2 + Linux TF (要セットアップ)
- TensorFlow-DirectML (TF 2.10 限定)
- **PyTorch 2.7 + CUDA 12.8** ← 採用

`models_torch.py` で同じアーキテクチャを再実装。さらに最適化:

1. **データ全体を GPU メモリに常駐** (130k × 256 floats ≈ 130 MB, 32 GB VRAM で余裕)
2. **GPU 上で permutation インデックスを生成** (CPU-GPU 転送を排除)
3. **大バッチ + `set_to_none=True` for `zero_grad()`**

結果: 学習時間は元の **約 1/12** に短縮。

### 4.6 Batch Sweep — 最大の発見

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

**理由**: 大バッチは「シャープな極小値」を見つけ、これは多数派クラスでは汎化するが、少数派クラス (ATTACK は train の 1.3%) のロス信号がバッチ平均に埋もれて学習されなくなる。

**結論**: Section 4.4 の焦った loss-function 系列の試行は全部不要だった。バッチサイズだけ正しく選べば良かった。

---

## 5. 最終結果 / 最终结果

**推奨構成**: Dilated CNN1D + plain CrossEntropy + batch=128 + 25 epochs + PyTorch GPU

| 指標 | 論文 (2022) | 本リポジトリ最終版 |
|------|-----------:|-----------------:|
| Framework | TensorFlow-CPU 2.8 | PyTorch 2.7 + CUDA 12.8 |
| Architecture | 2D CNN (16×16) | Dilated 1D CNN |
| Overall Accuracy | 99.58% | 98.83% |
| ATTACK Recall | ~70% | 67.28% |
| **ATTACK Precision** | (未報告) | **98.32%** |
| **ATTACK F1** | **(~57%, 推算)** | **79.89%** |
| Training Time | 214.95s | 46.3s |
| Hardware | AMD 5800X CPU | RTX 5090 GPU |

ATTACK F1 で **約 +23 ポイント** の改善。全体 Accuracy は論文より低いが、これは「Overall を 1% 落として ATTACK F1 を 20% 引き上げる」という意味のあるトレードオフ。攻撃検知タスクの実用性では後者のほうが価値が高い。

---

## 6. 教訓 / 经验与反思 / Lessons Learned

4 年前の自分への手紙として:

### 6.1 表形式データに 2D CNN を無理に使わない
セルとセルの空間的関係が意味を持つのは画像だけ。統計特徴量をパディングして正方形にしても、それは画像ではない。1D Conv なら少なくとも順序情報を尊重する。

### 6.2 小さなモデル + 小さなデータでは GPU が必ずしも速くない
RTX 5090 + batch=128 ですら、batch=2048 と比べて GPU 利用率は低い。Kernel-launch オーバーヘッドが計算時間を支配するため。「GPU を使う = 速い」は単純すぎる。

### 6.3 バッチサイズは不均衡データの最強チューニングノブ
Focal Loss、SMOTE、Two-Stage、Class Weight — 全部試したが、batch を 2048 → 128 にする 1 行の変更が一番効いた。原作で「64 vs 256 vs 1024」と試したのは正しい方向だったが、もう一段下げる勇気が必要だった。

### 6.4 「特徴量を強化する」が逆効果になることがある
ATTACK 検知でポートの log 変換や well-known port 指示子を追加したら、ATTACK Precision が 62% → 46% に崩壊。攻撃パケットは「ポートを偽装する」からこそ ATTACK なのであり、ポートを目立たせることはモデルを欺かれやすくする。**「強化」が悪化させるドメインがある**。

### 6.5 Baseline を超えてから「改善」と言う
Section 4.4 の loss-function トリックは個別に見れば筋が通っていたが、結局 plain baseline に勝てなかった。新規手法を提案する前に baseline の上限を本当に知っているか確認する必要がある。

### 6.6 Pooling の細部が結果を決めることがある
Dilated CNN の初版は GlobalAveragePooling で 82% に沈んだが、MaxPool に変えるだけで 99% に復活。アーキテクチャ図に書かれない「最後の 1 行」が往々にして決定的。

---

## 7. ファイル構成 / 项目结构

```
final/
├── config.py                  # 全局配置 (paths, labels, hyperparams)
├── data_preprocess.py         # ARFF 解析、平衡処理、フィーチャー正規化
├── feature_engineering.py     # 相関クラスタリングによる reorder, ATTACK 用派生特徴
├── models_dl.py               # TensorFlow/Keras: 元論文の CNN (2D), BP NN
├── models_ml.py               # sklearn: KNN, Naive Bayes, SVM, Decision Tree
├── models_torch.py            # PyTorch: CNN2D, CNN1D, Dilated CNN1D, Focal Loss
├── utils_eval.py              # 混同行列、訓練曲線、各種比較プロット
├── main.py                    # 元実験 (6 algorithms 比較) のエントリ
├── ablation.py                # 4.1〜4.3 の ablation 実験
├── attack_experiments.py      # 4.4 ATTACK 専門対策の 6 つの試行
├── batch_sweep.py             # 4.6 バッチサイズスイープ
├── requirements.txt           # Python dependencies
├── USAGE.md                   # 詳細な使い方 (旧 README)
├── data/moore/                # Moore データセット配置場所 (gitignore)
└── outputs/                   # 学習曲線、混同行列、各 JSON サマリー
```

### 主要な出力ファイル / 主要输出文件

| File | 内容 |
|------|------|
| `outputs/summary.json` | main.py 実行時の 6 algorithms 比較結果 |
| `outputs/ablation_summary.json` | 1D vs 2D, Dilated, reorder, attack-fe の ablation |
| `outputs/attack_experiments.json` | ATTACK 専門対策 6 試行 |
| `outputs/batch_sweep.json` | バッチサイズスイープの単調曲線 |
| `outputs/batch_sweep.png` | 上記のグラフ |
| `outputs/*_confusion.png` | 各モデルの混同行列 |

---

## 8. セットアップと使い方 / Quick Start

### 8.1 環境構築

```bash
# Python 3.12+ 推奨
pip install -r requirements.txt
# PyTorch GPU 版 (RTX 5090 等で GPU を使う場合)
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 8.2 データセット配置

10 個の `.arff.gz` を [Cambridge 公式サイト](https://www.cl.cam.ac.uk/research/srg/netos/projects/archive/nprobe/data/papers/sigmetrics/index.html) からダウンロードし、`data/moore/` に解凍:

```
data/moore/
├── entry01.weka.allclass.arff
├── ...
├── entry10.weka.allclass.arff
└── entry12.weka.allclass.arff  # (任意, 外部テスト用)
```

### 8.3 実行例

```bash
# 元論文の 6 algorithms 比較 (CPU でも数分)
python main.py --all --epochs 25

# 推奨最終構成 (要 PyTorch GPU)
python batch_sweep.py --epochs 25 --batches 128

# 4 つの 1D CNN 改良 ablation
python ablation.py --epochs 25 --batch-size 128

# ATTACK 専門対策の 6 試行
python attack_experiments.py --epochs 25 --batch-size 128 --enable-6
```

詳細は [USAGE.md](USAGE.md) を参照。

---

## 9. 謝辞 / 致谢

- 指導教官の **沈冰夏先生** — 卒研期間中のご指導に感謝
- **北京情報科技大学 情報与通信工程学院** — 学部 4 年間の教育
- **Andrew Moore et al.** — Moore データセットを公開していただき、今でも研究に使える形で保持していただいていることに感謝
- 4 年前の自分 — 完璧ではなかったが、それなりに動くコードを残してくれて、今こうして批判・改善する出発点を提供してくれた

---

## 参考文献 / References

[1] 王世元. 基于 CNN 的网络数据流量分类. 学部卒業論文, 北京信息科技大学, 2022.
[2] Moore, A. W., & Papagiannaki, K. (2005). Toward the accurate identification of network applications. In *International Workshop on Passive and Active Network Measurement* (pp. 41-54). Springer.
[3] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT press.
[4] Lin, M., Chen, Q., & Yan, S. (2013). Network in network. *arXiv:1312.4400*. (Global Average Pooling の元論文)
[5] Yu, F., & Koltun, V. (2015). Multi-scale context aggregation by dilated convolutions. *arXiv:1511.07122*.
[6] Lin, T. Y. et al. (2017). Focal loss for dense object detection. *ICCV*.

---

*Last updated: 2026-05-17*
