<p align="right">
  <b>日本語</b> | <a href="README.en.md">English</a>
</p>

# CNN ベースのネットワークトラフィック分類

パケットの統計的特徴 (パケット数・サイズ・タイミング・TCP フラグ等) から、それが Web 通信なのか、メールなのか、あるいは攻撃トラフィックなのかを CNN で識別するプロジェクトです。

---

## プロジェクト概要

通信内容そのものを覗き見しなくても、「通信パターン」だけで、それが何のアプリケーションによるものかをある程度推定できます。本リポジトリでは、248 種類の統計量を入力として CNN を構築し、12 種類のアプリケーション識別に取り組みました。

内容は大きく 2 つの段階に分かれています。

1. **2022 年の学部卒業研究の再現**
   TensorFlow を用いた CNN・BP・KNN・Naive Bayes・SVM・Decision Tree の 6 アルゴリズム比較を、当時のコードを再構築しながら再現しました。前処理コードに紛れていたラベル置換の不具合 — 文字列の一括置換が `FTP-CONTROL` と `INTERACTIVE` の中の `N` まで書き換え、これら 2 クラスが完全に欠落する問題 — も発見・修正しています。

2. **4 年後の再検証と拡張**
   PyTorch + GPU 環境への移行を起点に、「表形式データには 2D より 1D CNN のほうが構造的に適している」という仮説の検証、Dilated Conv の導入、そして不均衡クラスへの対策 (Focal Loss・Two-Stage 分類器・派生特徴量設計・バッチサイズスイープ) の系統的な実験を行いました。

最終的な成果としては、原論文 Section 5.2 の改良版 (filters 16/32 + batch 64) を本パイプライン上で公平に再評価したうえで、ATTACK クラスの F1 を 81.32% から **82.64%** へ、ATTACK Precision を 95.64% から **99.35%** へ引き上げました。検証を通じて最も興味深かった知見は、Focal Loss や二段階分類といった「いかにも効きそうな」手法群がベースラインに対して有意な改善を示さなかった一方で、バッチサイズを 2048 から 128 へ下げるという地味な調整こそが、不均衡データに対して支配的な効果を持っていたという点でした。複雑な手法に手を伸ばす前に、まずベースラインの本当の上限を測るべきだという教訓を、4 年越しに自分のコードから学び直すことになりました。

---

## 結果サマリー

公平な比較のため、原論文の構成と本リポジトリの提案手法を、同一の PyTorch + GPU パイプラインで再評価しました。

| 構成 | Batch | Overall | ATTACK Precision | ATTACK F1 | 時間 |
|------|------:|--------:|----------------:|----------:|-----:|
| 原論文 2D CNN (filters 8/16) | 128 | 98.53% | 88.44% | 78.46% | 38.4s |
| 原論文 Section 5.2 改良版 (filters 16/32) | 64 | 99.08% | 95.64% | 81.32% | 76.6s |
| 本リポジトリ: Dilated 1D CNN | 128 | 98.30% | 91.54% | 79.22% | 43.7s |
| **本リポジトリ: Dilated 1D CNN** | **64** | **99.12%** | **99.35%** | **82.64%** | 89.0s |

Dilated 1D CNN (batch 64) は、原論文の最良構成と比べて、Overall +0.04 / ATTACK Precision +3.71 / ATTACK F1 +1.32 ポイントの改善となりました。とりわけ ATTACK Precision 99.35% は、誤検知が極めて少ないという点で運用上の意味が大きいと考えています。

バッチサイズと ATTACK F1 の関係 (本検証で最も興味深かった単調曲線):

![batch sweep](docs/batch_sweep.png)

クラス別の精度比較:

![per-class accuracy](docs/per_class_accuracy.png)

他のグラフ (アルゴリズム比較、混同行列、学習曲線): [docs/](docs/)

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
