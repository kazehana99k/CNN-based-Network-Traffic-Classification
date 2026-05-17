<p align="right">
  <b>日本語</b> | <a href="USAGE.en.md">English</a>
</p>

# 使い方

このページでは、環境の準備からデータセットの取得、各スクリプトの動かし方までを順番に説明します。プロジェクト全体の話は [README.md](README.md) を見てください。

---

## 1. 環境を整える

Python 3.10 以上を推奨します。

```bash
pip install -r requirements.txt
```

PyTorch 版のスクリプト (`batch_sweep.py`、`attack_experiments.py` など) を GPU で動かしたい場合は、別途インストールしてください。CUDA 12.8 + RTX 5090 で動作確認しています。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

GPU が使えるかは次のように確認できます。

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. データセットの準備

Moore データセット (A. Moore et al., Cambridge 大学, 2005 年) の 10 個の ARFF ファイルが必要です。

ダウンロード元: <https://www.cl.cam.ac.uk/research/srg/netos/projects/archive/nprobe/data/papers/sigmetrics/index.html>

`entry01.weka.allclass.arff.gz` から `entry10.weka.allclass.arff.gz` までを取ってきて、解凍したものを `data/moore/` に置いてください。

```
data/moore/
├── entry01.weka.allclass.arff
├── entry02.weka.allclass.arff
├── ...
└── entry10.weka.allclass.arff
```

`entry12.weka.allclass.arff` も同じディレクトリに置いておくと、`--external-test` オプションを使ったときに、訓練に使っていない別のサンプルでモデルを評価できます (任意)。

---

## 3. 各スクリプトの動かし方

### 3.1 main.py — 6 アルゴリズムの比較 (元論文の再現)

```bash
# CNN だけを動かす
python main.py --cnn

# CNN と BP 神経網
python main.py --cnn --bp

# 6 つすべて (CNN + BP + KNN + Naive Bayes + SVM + Decision Tree)
python main.py --all

# epochs と batch_size を指定する
python main.py --cnn --epochs 25 --batch-size 128

# SVM は大きいデータだと時間がかかるので、部分サンプルで動かす
python main.py --svm --svm-subset 20000

# entry12 を外部テスト用に使う
python main.py --cnn --external-test entry12.weka.allclass.arff
```

### 3.2 batch_sweep.py — バッチサイズの系統的な検証 (推奨の最終構成)

```bash
python batch_sweep.py --epochs 25 --batches 128 256 512 1024 2048
```

Dilated 1D CNN を、指定された各バッチサイズで学習させて、ATTACK F1 や全体の精度がどう変わるかを比較します。出力は `outputs/batch_sweep.json` と `outputs/batch_sweep.png` です。

### 3.3 ablation.py — 1D CNN の各改善の効果を切り分ける

```bash
python ablation.py --epochs 25 --batch-size 128
```

baseline / k9 / reorder / attack-fe / dilated / all の 6 つの構成を比較します。

### 3.4 attack_experiments.py — ATTACK クラスを上げるための試み

```bash
python attack_experiments.py --epochs 25 --batch-size 128 --enable-6
```

Weighted CE、Focal Loss、Two-Stage 分類器、ポート特徴を消す方法など、ATTACK F1 を上げるための 6 つの方法を試します。

### 3.5 共通オプション

| オプション | 説明 |
|----------|------|
| `--no-balance` | クラスバランス調整を無効にする |
| `--no-normalize` | 特徴量の正規化を無効にする |
| `--reorder` | 相関クラスタリングで特徴量を並べ替える |
| `--attack-features` | ATTACK 向けの 6 つの派生特徴を追加する |
| `--cnn1d-kernel N` | 1D CNN のカーネルサイズを変える (デフォルトは 5) |
| `--external-test FILE` | 指定したファイルを外部テストセットとして使う |

---

## 4. 出力ファイルの説明

スクリプトを動かすと `outputs/` の下に次のようなファイルが作られます。

| ファイル | 内容 |
|---------|------|
| `summary.json` | main.py の結果 (各アルゴリズムの精度と所要時間) |
| `ablation_summary.json` | ablation.py の結果 |
| `attack_experiments.json` | attack_experiments.py の結果 |
| `batch_sweep.json` | batch_sweep.py の結果 |
| `*_confusion.png` | 各モデルの混同行列 |
| `*_history.png` | CNN/BP の学習曲線 |
| `overall_comparison.png` | 全アルゴリズムの精度・時間の比較バーチャート |
| `per_class_accuracy.png` | クラスごとの精度比較 |
| `batch_sweep.png` | バッチサイズと精度の関係を表す折れ線グラフ |

README に載せている図は `docs/` にもコピーしてあります。

---

## 5. ソースコードと論文の対応

| 論文の章節 | 実装場所 |
| -------- | -------- |
| 2.1 CNN | `models_dl.py:build_cnn_model` |
| 2.2.1 BP 神経網 | `models_dl.py:build_bp_model` |
| 2.2.2 KNN | `models_ml.py:train_knn` |
| 2.2.3 Naive Bayes | `models_ml.py:train_naive_bayes` |
| 2.2.4 Decision Tree | `models_ml.py:train_decision_tree` |
| 2.2.5 SVM | `models_ml.py:train_svm` |
| 3.1.2 クラスバランス調整 (表 3.2) | `data_preprocess.py:balance_dataset` |
| 3.1.3 平均値で埋める処理 | `data_preprocess.py:balance_dataset` |
| 4.2 データ前処理 | `data_preprocess.py:parse_arff_line` |
| 4.3 CNN の構築 | `models_dl.py:build_cnn_model` |
| 4.5 評価指標 | `utils_eval.py:plot_confusion_matrix` |
| 5.1 学習曲線 | `utils_eval.py:plot_training_history` |
| 5.3 6 アルゴリズム比較 | `utils_eval.py:plot_overall_comparison` |

---

## 6. うまく動かないとき

1. **ARFF ファイルが見つからないとエラーが出る** — `data/moore/` の中に 10 個のファイルが揃っているかを確認してください。パスを変えたい場合は `config.py:DATA_DIR` をいじります。
2. **SVM がいつまで経っても終わらない** — `--svm-subset 20000` のようにサンプル数を絞ってください。
3. **TensorFlow のインストールでエラーになる** — Python 3.12 以上を使っている場合は、TF 2.16 以上が必要です (`requirements.txt` の指定どおり `tensorflow-cpu>=2.16` でインストールしてください)。
4. **メモリ不足で落ちる** — `--batch-size` を小さくするか、`config.py:MOORE_FILES` で読み込むファイル数を減らしてください。
5. **GPU が認識されない (PyTorch)** — `python -c "import torch; print(torch.cuda.is_available())"` で確認します。`False` の場合は CUDA とドライバのバージョンが合っていない可能性があるので、`pip install torch --index-url ...` で再インストールしてください。

---

## 7. このリポジトリと元論文の関係

これは王世元の学部卒業研究 (2022 年) を、4 年後に自分で見直して書き直したものです。当時の方針・特徴量の選び方・モデル構成は基本的にそのまま使いつつ、コードのバグ修正、1D CNN や Dilated Conv への発展、PyTorch + GPU 化、バッチサイズの系統的な検証などを足しました。

それぞれの改善の動機・実装・結果については [EXPERIMENTS.md](EXPERIMENTS.md) を見てください。
