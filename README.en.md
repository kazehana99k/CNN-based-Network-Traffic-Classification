<p align="right">
  <a href="README.md">日本語</a> | <b>English</b>
</p>

# CNN-based Network Traffic Classification

Classifying network packets — Web, mail, attack traffic, and 9 other categories — using a CNN trained on packet-level statistical features (count, size, timing, TCP flags).

---

## Overview

Even without inspecting payload content, the *traffic pattern alone* (packet counts, size distribution, timing, TCP flags) carries enough signal to guess what application produced it. This project builds a CNN that takes **248 statistical features** as input and predicts which of **12 application classes** each flow belongs to.

The repository is organized in two stages:

1. **Reproduction of my 2022 undergraduate thesis**
   A six-algorithm comparison (CNN, BP NN, KNN, Naive Bayes, SVM, Decision Tree) implemented in TensorFlow. During reproduction I also caught and fixed a label-preprocessing bug — a blanket string replacement was clobbering the letter `N` inside the class labels `FTP-CONTROL` and `INTERACTIVE`, causing those two classes to vanish entirely.

2. **A 4-year-later self-review and extension**
   Migration to PyTorch + GPU, validation of the hypothesis that 1D CNNs are structurally better suited to tabular features than 2D, addition of Dilated Conv, and a systematic study of imbalanced-class countermeasures (Focal Loss, two-stage classifiers, derived features, and a batch-size sweep).

The main result: after fairly re-evaluating the paper's Section 5.2 improved variant (filters 16/32 + batch 64) under the same pipeline, the proposed method lifts ATTACK F1 from 81.32% to **82.64%** and ATTACK Precision from 95.64% to **99.35%**. The most interesting finding along the way was this: none of the "obvious" tricks — Focal Loss, two-stage classification, oversampling — beat the plain baseline. What did help was dropping the batch size from 2048 to 128, a mundane hyperparameter change that dominated every fancy technique I tried. The lesson, 4 years after writing the original code: measure your baseline's true ceiling before reaching for complex methods.

---

## Results at a glance

For a fair comparison, both the paper's configurations and the proposed method are re-evaluated under the same PyTorch + GPU pipeline.

| Configuration | Batch | Overall | ATTACK Precision | ATTACK F1 | Time |
|---------------|------:|--------:|-----------------:|----------:|-----:|
| Paper original 2D CNN (filters 8/16) | 128 | 98.53% | 88.44% | 78.46% | 38.4s |
| Paper Section 5.2 improved 2D CNN (filters 16/32) | 64 | 99.08% | 95.64% | 81.32% | 76.6s |
| This repo: Dilated 1D CNN | 128 | 98.30% | 91.54% | 79.22% | 43.7s |
| **This repo: Dilated 1D CNN** | **64** | **99.12%** | **99.35%** | **82.64%** | 89.0s |

The proposed Dilated 1D CNN (batch 64) improves on the paper's best configuration by +0.04 in Overall, +3.71 in ATTACK Precision, and +1.32 in ATTACK F1. The 99.35% ATTACK Precision is particularly meaningful in practice — false positives on attack detection are very rare.

Batch size vs ATTACK F1 (the most interesting monotonic curve in this study):

![batch sweep](docs/batch_sweep.png)

Per-class accuracy comparison:

![per-class accuracy](docs/per_class_accuracy.png)

Other figures (algorithm comparison, confusion matrices, training curves): [docs/](docs/)

---

## Quick Start

```bash
pip install -r requirements.txt

# Place the Moore dataset under data/moore/  (see USAGE.en.md for details)
# Cambridge official: https://www.cl.cam.ac.uk/research/srg/netos/projects/archive/nprobe/data/papers/sigmetrics/

python main.py --all                              # Reproduce the original 6-algorithm comparison
python batch_sweep.py --epochs 25 --batches 128   # Recommended final config (GPU advised)
```

Full instructions: [USAGE.en.md](USAGE.en.md). Experiment notes (motivation, implementation, results for each improvement): [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Project Layout

```
final/
├── README.md / README.en.md / USAGE.en.md / EXPERIMENTS.md
├── config.py                  -- Global config
├── data_preprocess.py         -- Moore ARFF parsing + class balancing
├── feature_engineering.py     -- Feature reordering, ATTACK-aware derived features
├── models_dl.py               -- TF/Keras: original 2D CNN, BP
├── models_ml.py               -- sklearn: KNN, NB, SVM, DT
├── models_torch.py            -- PyTorch GPU: CNN/CNN1D/Dilated + Focal Loss
├── utils_eval.py              -- Confusion matrix, training curves, comparison plots
├── main.py                    -- Entry point for the 6-algorithm comparison
├── ablation.py                -- 1D vs 2D, Dilated, reorder ablation
├── attack_experiments.py      -- 6 attempts at boosting ATTACK class
├── batch_sweep.py             -- Systematic batch-size sweep
└── data/moore/  outputs/      -- Dataset & results (gitignored)
```

---

## Key Findings (3 lines)

1. **2D reshape creates fake spatial neighbors** — packing 248 statistical features into 16×16 does not make them an image. 1D Conv respects sequence locality and works better.
2. **For small models on imbalanced data, batch size is the strongest knob** — I tried Focal Loss, Two-Stage, SMOTE, and class weighting. The single line that helped most was dropping batch from 2048 to 128.
3. **"Strengthening features" can backfire** — emphasizing port features tanked ATTACK precision. Attackers spoof ports *on purpose*, so making the model focus on them only made it easier to fool.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full story.

---

## Background

This is a 4-year-later self-review and extension of my own undergraduate thesis (2022) by **Wang Shiyuan**.

Dataset: Moore Dataset (A. Moore et al., University of Cambridge, 2005).

---

## License / Citation

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
