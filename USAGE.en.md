<p align="right">
  <a href="USAGE.md">日本語</a> | <b>English</b>
</p>

# Usage Guide

This page walks through environment setup, dataset preparation, and how to run each script. For the high-level overview, see [README.en.md](README.en.md).

---

## 1. Environment Setup

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

If you want to run the PyTorch scripts (`batch_sweep.py`, `attack_experiments.py`, etc.) on GPU, install PyTorch separately. Tested on CUDA 12.8 + RTX 5090.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Check whether GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. Dataset Preparation

You need the 10 ARFF files of the Moore Dataset (A. Moore et al., University of Cambridge, 2005).

Download source: <https://www.cl.cam.ac.uk/research/srg/netos/projects/archive/nprobe/data/papers/sigmetrics/index.html>

Grab `entry01.weka.allclass.arff.gz` through `entry10.weka.allclass.arff.gz`, decompress them, and place under `data/moore/`:

```
data/moore/
├── entry01.weka.allclass.arff
├── entry02.weka.allclass.arff
├── ...
└── entry10.weka.allclass.arff
```

Optionally place `entry12.weka.allclass.arff` in the same directory to use it as a held-out external test set via the `--external-test` flag.

---

## 3. How to Run Each Script

### 3.1 main.py — 6-algorithm comparison (reproduce the original paper)

```bash
# CNN only
python main.py --cnn

# CNN and BP neural net
python main.py --cnn --bp

# All six (CNN + BP + KNN + Naive Bayes + SVM + Decision Tree)
python main.py --all

# Custom epochs and batch size
python main.py --cnn --epochs 25 --batch-size 128

# SVM is slow on large data; train on a subset
python main.py --svm --svm-subset 20000

# Use entry12 as an external test set
python main.py --cnn --external-test entry12.weka.allclass.arff
```

### 3.2 batch_sweep.py — systematic batch-size study (recommended final config)

```bash
python batch_sweep.py --epochs 25 --batches 128 256 512 1024 2048
```

Trains the Dilated 1D CNN for each batch size and compares ATTACK F1 and overall accuracy. Outputs `outputs/batch_sweep.json` and `outputs/batch_sweep.png`.

### 3.3 ablation.py — isolate the contribution of each 1D CNN improvement

```bash
python ablation.py --epochs 25 --batch-size 128
```

Compares six configurations: baseline / k9 / reorder / attack-fe / dilated / all.

### 3.4 attack_experiments.py — attempts to boost the ATTACK class

```bash
python attack_experiments.py --epochs 25 --batch-size 128 --enable-6
```

Tries six approaches to improve ATTACK F1: Weighted CE, Focal Loss, Two-Stage classifier, port-zeroed two-stage, etc.

### 3.5 Shared options

| Option | Description |
|--------|-------------|
| `--no-balance` | Disable class balancing |
| `--no-normalize` | Disable feature normalization |
| `--reorder` | Reorder features via correlation clustering |
| `--attack-features` | Append 6 ATTACK-targeted derived features |
| `--cnn1d-kernel N` | Change the 1D CNN kernel size (default 5) |
| `--external-test FILE` | Use the specified file as a held-out test set |

---

## 4. Output Files

Running scripts generates files under `outputs/`:

| File | Contents |
|------|----------|
| `summary.json` | main.py results (accuracy and time per algorithm) |
| `ablation_summary.json` | ablation.py results |
| `attack_experiments.json` | attack_experiments.py results |
| `batch_sweep.json` | batch_sweep.py results |
| `*_confusion.png` | Confusion matrices per model |
| `*_history.png` | Training curves for CNN/BP |
| `overall_comparison.png` | Bar chart comparing all algorithms |
| `per_class_accuracy.png` | Per-class accuracy comparison |
| `batch_sweep.png` | Batch size vs accuracy curve |

The figures shown in the README are also copied under `docs/`.

---

## 5. Source Code ↔ Paper Mapping

| Paper Section | Implementation |
| ------------- | -------------- |
| 2.1 CNN | `models_dl.py:build_cnn_model` |
| 2.2.1 BP NN | `models_dl.py:build_bp_model` |
| 2.2.2 KNN | `models_ml.py:train_knn` |
| 2.2.3 Naive Bayes | `models_ml.py:train_naive_bayes` |
| 2.2.4 Decision Tree | `models_ml.py:train_decision_tree` |
| 2.2.5 SVM | `models_ml.py:train_svm` |
| 3.1.2 Class balancing (Table 3.2) | `data_preprocess.py:balance_dataset` |
| 3.1.3 Mean-filling algorithm | `data_preprocess.py:balance_dataset` |
| 4.2 Data preprocessing | `data_preprocess.py:parse_arff_line` |
| 4.3 CNN architecture | `models_dl.py:build_cnn_model` |
| 4.5 Evaluation metrics | `utils_eval.py:plot_confusion_matrix` |
| 5.1 Training curves | `utils_eval.py:plot_training_history` |
| 5.3 6-algorithm comparison | `utils_eval.py:plot_overall_comparison` |

---

## 6. Troubleshooting

1. **ARFF files not found** — make sure all 10 files are under `data/moore/`. To change the path, edit `config.py:DATA_DIR`.
2. **SVM hangs forever** — limit the training set size with `--svm-subset 20000` or smaller.
3. **TensorFlow install fails** — on Python 3.12+, TF 2.16 or newer is required (`tensorflow-cpu>=2.16` as specified in `requirements.txt`).
4. **Out of memory** — reduce `--batch-size`, or edit `config.py:MOORE_FILES` to load fewer ARFF files.
5. **GPU not detected (PyTorch)** — verify with `python -c "import torch; print(torch.cuda.is_available())"`. If `False`, your CUDA driver may be incompatible — reinstall PyTorch with the matching CUDA version (`pip install torch --index-url ...`).

---

## 7. Relation to the Original Thesis

This repository is a 4-year-later self-review and rewrite of my own undergraduate thesis (2022). The original methodology — feature selection, model architecture, evaluation procedure — is largely preserved, while the rewrite adds: bug fixes in data preprocessing, evolution from 2D to 1D and Dilated CNN, migration to PyTorch + GPU, and a systematic study of the batch-size effect.

For motivation, implementation, and results of each improvement, see [EXPERIMENTS.md](EXPERIMENTS.md).
