<p align="right">
  <a href="README.md">日本語</a> | <b>English</b>
</p>

# CNN-based Network Traffic Classification

> A classifier that tells what kind of app each network packet belongs to (Web, mail, P2P, attack...).

---

## 🎯 What is this?

Network packets flowing on the Internet cannot be told apart by appearance alone — you can't tell "this is YouTube" or "this is malware" just by looking. This project takes **248 statistical features** (packet counts, sizes, timing, TCP flags...) as input and uses a CNN to predict **which of 12 application classes** each packet belongs to.

The repo has **two layers**:

1. 🎓 Reproduction of my 2022 undergraduate thesis (TensorFlow-based, 6-algorithm comparison)
2. 🔬 A 4-year-later self-review extension (PyTorch GPU + 1D CNN + Dilated Conv + systematic ablation)

**Main result**: ATTACK class F1 improved from **~57% → 79.89%**. The key turned out to be picking the right batch size, not exotic loss functions.

---

## 📊 Results at a glance

| Algorithm | Overall Acc | ATTACK F1 | Notes |
|-----------|------------:|----------:|-------|
| Original 2D CNN (2022 thesis) | 99.58% | ~57% | TF-CPU baseline |
| **Dilated 1D CNN (this repo)** | **98.83%** | **79.89%** ⭐ | **PyTorch GPU + batch 128** |

Detailed plots: [`outputs/batch_sweep.png`](outputs/batch_sweep.png), [`outputs/per_class_accuracy.png`](outputs/per_class_accuracy.png)

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Place the Moore dataset under data/moore/  (see USAGE.md for details)
# Cambridge official: https://www.cl.cam.ac.uk/research/srg/netos/projects/archive/nprobe/data/papers/sigmetrics/

python main.py --all                              # Reproduce the original 6-algorithm comparison
python batch_sweep.py --epochs 25 --batches 128   # Recommended final config (GPU advised)
```

Full instructions: [USAGE.md](USAGE.md)
Experiment notes (motivation, implementation, results for each improvement): [EXPERIMENTS.md](EXPERIMENTS.md)

---

## 🗂️ Project Layout

```
final/
├── 📄 README.md / README.en.md / USAGE.md / EXPERIMENTS.md
├── ⚙️  config.py                  # Global config
├── 🔧 data_preprocess.py          # Moore ARFF parsing + balancing
├── 🔧 feature_engineering.py      # Feature reordering, ATTACK-aware derived features
├── 🧠 models_dl.py                # TF/Keras: original 2D CNN, BP
├── 🧠 models_ml.py                # sklearn: KNN, NB, SVM, DT
├── 🧠 models_torch.py             # PyTorch GPU: CNN/CNN1D/Dilated + Focal Loss
├── 📈 utils_eval.py               # Confusion matrix, training curves, comparison plots
├── 🚀 main.py                     # Entry point for 6-algorithm comparison
├── 🚀 ablation.py                 # 1D vs 2D, Dilated, reorder ablation
├── 🚀 attack_experiments.py       # 6 attempts at boosting ATTACK class
├── 🚀 batch_sweep.py              # Systematic batch-size sweep
└── 📁 data/moore/  outputs/       # Dataset & results (gitignored)
```

---

## 💡 Key Findings (3 lines)

1. **2D reshape creates fake spatial neighbors** — packing 248 statistical features into 16×16 doesn't make them an image. 1D Conv respects sequence locality and works better.
2. **For small models + imbalanced data, batch size is the strongest knob** — I tried Focal Loss / Two-Stage / SMOTE / class weighting. The single line that helped most was dropping batch from 2048 to 128.
3. **"Strengthening features" can backfire** — emphasizing port features tanked ATTACK precision. Attackers spoof ports *on purpose*; making the model focus on them made it easier to fool.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full story.

---

## 📚 Background

This is a 4-year-later self-review and extension of my own undergraduate thesis (2022) by **Wang Shiyuan**.

Dataset: **Moore Dataset** (A. Moore et al., University of Cambridge, 2005).

---

## 📜 License / Citation

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
