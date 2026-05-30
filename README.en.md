<p align="right">
  <a href="README.md">日本語</a> | <b>English</b>
</p>

# CNN-based Network Traffic Classification

A CNN that tells whether a network flow is Web browsing, email, file transfer, a cyber attack, or one of nine other categories — using only surface-level statistics, without inspecting payload content.

![class-wise accuracy](docs/per_class_accuracy.png)

## Why I built this

From home routers to enterprise firewalls, networking equipment has relied on **port numbers** to classify traffic for decades. SMTP is port 25, HTTPS is 443, and so on. But this approach has been quietly breaking down for years.

- P2P apps, VoIP and games dynamically switch ports, so fixed rules can't track them.
- Attackers deliberately use legitimate ports (80, 443) to sneak past the same rules.
- Now that HTTPS is everywhere, Deep Packet Inspection (DPI) only works in a few narrow cases.

In short, port-and-rule-based classification has hit the limits of both effectiveness and security. This project explores an alternative: using only 248 surface-level statistics — "packets per second," "average packet size," "TCP flag frequency," and similar — to let a machine learning model figure out what kind of traffic each flow is. Because the model never looks at payload, it keeps working even when the traffic is encrypted, and respects user privacy by design.

## How this repository is organized

The repo has two stages.

**Stage 1 — Reproducing my 2022 undergraduate thesis**
I rebuilt the original TensorFlow implementation of a six-algorithm comparison (CNN, BP NN, KNN, Naive Bayes, SVM, Decision Tree) while rereading my own code from four years ago. Along the way I caught a preprocessing bug in the original: a blanket string-replacement was clobbering the letter `N` inside the class labels `FTP-CONTROL` and `INTERACTIVE`, silently erasing every sample of those two classes from the training set.

**Stage 2 — Building an improved version**
Starting from a migration to PyTorch + GPU, I tested the hypothesis that "treating tabular features as a 16×16 pseudo-image with a 2D CNN is structurally unnatural" and worked through more than a dozen configurations: 1D CNN, Dilated convolutions, Focal Loss, two-stage classifiers, derived port-aware features, and a systematic batch-size sweep.

## Results

For a fair comparison, both the paper's configurations and the proposed method are re-evaluated under the same PyTorch + GPU pipeline.

| Configuration | Batch | Overall | ATTACK Precision | ATTACK F1 | Time |
|---------------|------:|--------:|-----------------:|----------:|-----:|
| Paper original 2D CNN (filters 8/16) | 128 | 98.53% | 88.44% | 78.46% | 38.4s |
| Paper Section 5.2 improved (filters 16/32) | 64 | 99.08% | 95.64% | 81.32% | 76.6s |
| **Proposed: Dilated 1D CNN** | 64 | **99.12%** | **99.35%** | **82.64%** | 89.0s |

The proposed method beats the paper's best configuration on every metric. The 99.35% ATTACK Precision is particularly meaningful: it means the model almost never mistakes legitimate traffic for an attack — and in real security products, that "low false-positive rate" matters more than F1 itself.

![batch size vs attack F1](docs/batch_sweep.png)

The chart above shows how attack-detection F1 changes with batch size. The fact that it improves monotonically as batch shrinks was the most surprising finding of this study. None of the clever techniques I tried — Focal Loss, two-stage classification, oversampling — could budge the F1 score; but dropping the batch size from 2048 to 128, a single line of code, did. A textbook lesson, learned from my own four-year-old code.

## Quick start

```bash
pip install -r requirements.txt
# Place the Moore dataset under data/moore/  (see USAGE.en.md)

python main.py --all                            # Reproduce the six-algorithm comparison
python batch_sweep.py --epochs 25 --batches 128 # Recommended final config (GPU advised)
```

Full instructions: [USAGE.en.md](USAGE.en.md). Motivation, implementation, and per-experiment results: [EXPERIMENTS.md](EXPERIMENTS.md).

## Dataset

The experiments use the Moore Dataset, made publicly available by the University of Cambridge (A. Moore et al., 2005). It consists of 10 ARFF files containing roughly 250,000 labeled traffic flows. Download instructions are in [USAGE.en.md](USAGE.en.md).

## Citation

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
