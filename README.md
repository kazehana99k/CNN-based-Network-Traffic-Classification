<p align="right">
  <b>日本語</b> | <a href="README.en.md">English</a>
</p>

# CNN ベースのネットワークトラフィック分類

> ネットワーク通信パケットがどのアプリ由来か (Web、メール、P2P、攻撃...) を当てる分類器。

---

## 🎯 これは何？

インターネット上を流れるパケットは、見た目だけでは「これは YouTube」「これはマルウェア」と区別できない。本プロジェクトは **248 種の統計特徴量** (パケット数、サイズ、タイミング、TCP フラグ...) を入力に、CNN で **12 クラスのアプリ種別** を当てる。

**2 つの層** からなる:

1. 🎓 2022 年に書いた学部卒業研究の再現 (TensorFlow ベース、6 アルゴリズム比較)
2. 🔬 4 年後の自己レビューによる拡張 (PyTorch GPU + 1D CNN + Dilated Conv + 系統的 ablation)

**主な成果**: ATTACK クラス F1 を **~57% → 79.89%** まで改善。手法ではなく「正しいバッチサイズ」が鍵という発見。

---

## 📊 一目で分かる結果

| Algorithm | Overall Acc | ATTACK F1 | Notes |
|-----------|------------:|----------:|-------|
| 元論文 2D CNN (2022) | 99.58% | ~57% | TF-CPU baseline |
| **Dilated 1D CNN (本リポ)** | **98.83%** | **79.89%** ⭐ | **PyTorch GPU + batch 128** |

詳細グラフ: [`outputs/batch_sweep.png`](outputs/batch_sweep.png), [`outputs/per_class_accuracy.png`](outputs/per_class_accuracy.png)

---

## 🚀 クイックスタート

```bash
pip install -r requirements.txt

# Moore データセットを data/moore/ に配置 (詳細は USAGE.md)
# Cambridge 公式: https://www.cl.cam.ac.uk/research/srg/netos/projects/archive/nprobe/data/papers/sigmetrics/

python main.py --all                              # 元論文の 6 algorithms 再現
python batch_sweep.py --epochs 25 --batches 128   # 推奨最終構成 (GPU 推奨)
```

詳細な実行手順: [USAGE.md](USAGE.md)
実験ノート (各改善実験の動機・実装・結果): [EXPERIMENTS.md](EXPERIMENTS.md)

---

## 🗂️ プロジェクト構成

```
final/
├── 📄 README.md / README.en.md / USAGE.md / EXPERIMENTS.md
├── ⚙️  config.py                  # 全局設定
├── 🔧 data_preprocess.py          # Moore データの読込・平衡処理
├── 🔧 feature_engineering.py      # 特徴量並べ替え, ATTACK 用派生特徴
├── 🧠 models_dl.py                # TF/Keras: 元論文の CNN, BP
├── 🧠 models_ml.py                # sklearn: KNN, NB, SVM, DT
├── 🧠 models_torch.py             # PyTorch GPU: CNN/CNN1D/Dilated + Focal Loss
├── 📈 utils_eval.py               # 混同行列, 訓練曲線, 比較プロット
├── 🚀 main.py                     # 6 algorithms 比較のエントリ
├── 🚀 ablation.py                 # 1D vs 2D, Dilated, reorder の ablation
├── 🚀 attack_experiments.py       # ATTACK 専門 6 試行
├── 🚀 batch_sweep.py              # バッチサイズ系統スイープ
└── 📁 data/moore/  outputs/       # データ・結果出力 (gitignore)
```

---

## 💡 主な発見 (3 行で)

1. **2D reshape は擬似近傍を作る** — 248 個の統計特徴量を 16×16 に並べても画像にはならない。1D Conv の方が筋が良い。
2. **小モデル + 不均衡データではバッチサイズが最強のチューニング** — Focal Loss / Two-Stage / SMOTE / class weight 全部試したが、batch を 2048 → 128 にする 1 行が一番効いた。
3. **「特徴量強化」が逆効果なドメインがある** — ATTACK 検知でポート特徴を強調したら Precision が崩壊した。攻撃側がポートを偽装するからこそ ATTACK なのだから。

詳細は [EXPERIMENTS.md](EXPERIMENTS.md) を参照。

---

## 📚 背景

これは **王世元** の学部卒業研究 (2022 年) を、4 年後に自分で再現・批判的に検証し直したプロジェクトです。

データセット: **Moore Dataset** (A. Moore et al., University of Cambridge, 2005)

---

## 📜 ライセンス・引用

```bibtex
@misc{wang2022cnn,
  author = {Wang, Shiyuan},
  title  = {CNN-based Network Traffic Classification},
  year   = {2022},
}

@inproceedings{moore2005identification,
  author    = {Moore, Andrew W. and Papagiannaki, Konstantina},
  title     = {Toward the accurate identification of network applications},
  booktitle = {PAM},
  year      = {2005},
}
```

---

*Last updated: 2026-05*
