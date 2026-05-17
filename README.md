<p align="right">
  <b>日本語</b> | <a href="README.en.md">English</a>
</p>

# CNN ベースのネットワークトラフィック分類

ネットワークを流れているパケットが Web 通信なのか、メールなのか、それとも攻撃トラフィックなのかを、CNN を使って分類するプロジェクトです。

---

## このプロジェクトについて

ネット上のパケットは、見ただけでは「これは YouTube」「これはマルウェア」とは判断できません。本プロジェクトでは、パケット数・サイズ・通信時間・TCP フラグなど 248 個の統計量を CNN に入力して、12 種類のアプリケーションのうちどれに該当するかを当てます。

このリポジトリには、内容が 2 つあります。

1. **2022 年に書いた学部卒業研究の再現** — TensorFlow を使って、CNN を含む 6 つのアルゴリズムを比較しました。
2. **4 年後に自分で書き直した拡張版** — PyTorch と GPU を使って、1D CNN や Dilated Conv などを試し、バッチサイズの影響を系統的に調べました。

一番の成果は、ATTACK クラスの F1 スコアを約 57% から 79.89% まで上げられたことです。「難しい手法をたくさん試す」よりも「バッチサイズを正しく選ぶ」方が結果に効くというのが、今回の一番大きな発見でした。

---

## 結果サマリー

| Algorithm | Overall Acc | ATTACK F1 | Notes |
|-----------|------------:|----------:|-------|
| 元論文 2D CNN (2022) | 99.58% | ~57% | TF-CPU baseline |
| **Dilated 1D CNN (本リポジトリ)** | **98.83%** | **79.89%** | **PyTorch GPU + batch 128** |

詳細グラフ: [`outputs/batch_sweep.png`](outputs/batch_sweep.png), [`outputs/per_class_accuracy.png`](outputs/per_class_accuracy.png)

---

## 動かし方

```bash
pip install -r requirements.txt

# Moore データセットを data/moore/ に置いてください (詳しくは USAGE.md)
# Cambridge 公式: https://www.cl.cam.ac.uk/research/srg/netos/projects/archive/nprobe/data/papers/sigmetrics/

python main.py --all                              # 元論文の 6 アルゴリズム比較を再現
python batch_sweep.py --epochs 25 --batches 128   # おすすめの最終構成 (GPU 推奨)
```

詳しい実行手順は [USAGE.md](USAGE.md) を、各改善実験の動機・実装・結果は [EXPERIMENTS.md](EXPERIMENTS.md) を見てください。

---

## ファイル構成

```
final/
├── README.md / README.en.md / USAGE.md / EXPERIMENTS.md
├── config.py                  -- 全体の設定
├── data_preprocess.py         -- Moore データの読み込みとクラスバランス調整
├── feature_engineering.py     -- 特徴量の並べ替え、ATTACK 向けの派生特徴
├── models_dl.py               -- TF/Keras: 元論文の CNN, BP
├── models_ml.py               -- sklearn: KNN, NB, SVM, DT
├── models_torch.py            -- PyTorch GPU: CNN/CNN1D/Dilated + Focal Loss
├── utils_eval.py              -- 混同行列・学習曲線・比較プロット
├── main.py                    -- 6 アルゴリズム比較のエントリーポイント
├── ablation.py                -- 1D vs 2D、Dilated、reorder の ablation
├── attack_experiments.py      -- ATTACK クラス専用の 6 つの試み
├── batch_sweep.py             -- バッチサイズの系統的スイープ
└── data/moore/  outputs/      -- データと出力 (gitignore 対象)
```

---

## 主な発見 (3 行で)

1. **2D reshape は意味のない近傍を作ってしまう** — 248 個の統計量を 16×16 に並べても、それは画像ではありません。1D Conv の方が自然です。
2. **小さなモデル + 不均衡データではバッチサイズが一番効く** — Focal Loss、Two-Stage、SMOTE、class weight などを色々と試しましたが、batch を 2048 から 128 に下げる 1 行の変更が一番効きました。
3. **特徴量を強調すると逆効果になることがある** — ATTACK 検知のためにポート関連の特徴を強調したら、Precision が下がってしまいました。攻撃側はポートを偽装することが多いので、ポートを重視すると逆に騙されやすくなります。

詳しい話は [EXPERIMENTS.md](EXPERIMENTS.md) を見てください。

---

## 背景

これは王世元の学部卒業研究 (2022 年) を、4 年後に自分で再現して書き直したプロジェクトです。

データセット: Moore Dataset (A. Moore et al., University of Cambridge, 2005)

---

## 引用

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
