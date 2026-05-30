<p align="right">
  <a href="README.md">日本語</a> | <b>English</b>
</p>

# CNN-based Network Traffic Classification

This project builds a CNN that automatically classifies network traffic into 12 categories — Web browsing, email, file transfer, cyber attacks, and so on. Crucially, the model never looks at the payload content (such as the encrypted body of an HTTPS request). It works only from surface-level statistics: "how many packets per second," "average packet size," "how often a specific TCP flag was set," and so on.

## Why I built this

From home routers to enterprise firewalls, networking equipment has classified traffic by **port number** for decades. SMTP is port 25, HTTPS is 443, and so on. But this approach has quietly stopped working in the modern Internet:

- P2P apps, VoIP, and online games dynamically switch ports, so fixed rules can't track them.
- Attackers deliberately use legitimate ports (80, 443) to slip past the same rules.
- Now that HTTPS is everywhere, Deep Packet Inspection (DPI) only works in a narrow set of cases.

In short, port-and-rule traffic classification has hit the limits of both effectiveness and security. This project explores the alternative: letting a machine-learning model recognize traffic *by its behavior* — its statistical fingerprint — without ever inspecting payload. Because no payload is inspected, the system continues to work even when traffic is encrypted, and respects user privacy by design.

## The starting point: my own undergraduate thesis from 2022

This repository builds on my own undergraduate thesis, written at **Beijing Information Science and Technology University in 2022** ("CNN-based Network Traffic Classification" by Wang Shiyuan). The original work was:

- **Dataset**: The publicly available **Moore Dataset** from the University of Cambridge — roughly 250,000 labeled flows across 12 classes.
- **Inputs**: 248 statistical features extracted from each network flow.
- **Methods compared**: a **2D CNN** (with the 248 features reshaped to a 16×16 grid), plus BP neural network, KNN, Naive Bayes, SVM, and Decision Tree — six approaches in total.
- **Main finding**: the CNN achieved the highest overall accuracy (99.58%) and a reasonable trade-off between accuracy and runtime.
- **Unresolved difficulty**: the **ATTACK class accuracy plateaued at ~70%**, with the thesis explicitly noting that "attack traffic disguises itself via legitimate ports, so a fundamental redesign of the feature space would be required to overcome this."

Four years later, when I reread my own code, two questions came to mind:

1. Were the original conclusions actually correct? Was there a structural problem in the design that I had missed at the time?
2. With today's frameworks (PyTorch + GPU) and techniques that have matured since (1D CNNs for tabular data, Dilated convolutions, Focal Loss, etc.), could I push past the ATTACK ceiling that the 2022 me had given up on?

This repository is the record of that re-examination and extension.

## What I did

The work breaks into four stages.

### 1. Getting my four-year-old code to run on a modern stack

I ported the original TensorFlow + Keras + Python 3.9 implementation to Python 3.12 / TF 2.21 and re-ran the six-algorithm comparison. In the process, I discovered a preprocessing bug that my younger self had introduced: a blanket string-replacement was clobbering the letter `N` inside the class names `FTP-CONTROL` and `INTERACTIVE`, silently erasing every sample of those two classes from the training set. The project literally began by hunting down a quiet bug that I had written four years earlier.

### 2. Questioning the "table as pseudo-image" design

The original paper reshapes 248 statistical features into a 16×16 pseudo-image and feeds it to a 2D CNN. The problem is that those 248 columns are semantically independent statistics, so the "up/down/left/right" neighbors a 2D convolution sees are essentially coincidence. Switching to a 1D CNN — which respects the linear feature order — improved accuracy, with the largest gains (+3 to +4 points) on the rarer classes such as FTP-PASV and INTERACTIVE_.

### 3. Stretching the receptive field with Dilated 1D CNN

A vanilla Conv1D kernel only sees ~5 cells. Replacing it with a stack of dilated convolutions extended the effective receptive field to ~15 cells. The first attempt — using Global Average Pooling at the head — backfired, washing out the signal and dropping accuracy to 82%. Replacing GAP with Max Pooling restored 99%-level accuracy, and this final configuration ended up beating every variant from the original paper.

### 4. A systematic study of the ATTACK class

The ATTACK class represents only ~1% of the data and was the unresolved difficulty in the original thesis. I tried over a dozen approaches: Focal Loss, two-stage classifiers, SMOTE-style synthetic samples, port-aware derived features, and a systematic batch-size sweep. What ended up winning was the single-line change of dropping the mini-batch size from 2048 to 128. All the clever techniques underperformed plain Cross-Entropy at the right batch size.

## Results

For a fair comparison, both the paper's configurations and the proposed method are re-evaluated under the same PyTorch + GPU pipeline.

| Configuration | Batch | Overall | ATTACK Precision | ATTACK F1 | Time |
|---------------|------:|--------:|-----------------:|----------:|-----:|
| Paper original 2D CNN (filters 8/16) | 128 | 98.53% | 88.44% | 78.46% | 38.4s |
| Paper Section 5.2 improved (filters 16/32) | 64 | 99.08% | 95.64% | 81.32% | 76.6s |
| **Proposed: Dilated 1D CNN** | 64 | **99.12%** | **99.35%** | **82.64%** | 89.0s |

The proposed method beats the paper's best configuration on every metric.

![ATTACK class metric comparison](docs/attack_comparison.png)

The ATTACK Precision of 99.35% in particular is meaningful in practice: it means the model almost never mistakes legitimate traffic for an attack, and in real security products that low false-positive rate often matters more than F1 itself.

![batch size vs attack F1](docs/batch_sweep.png)

The chart above shows how attack-detection F1 changes with batch size. The monotonic improvement as batch shrinks was the most surprising finding in this study. None of the "obvious" tricks — Focal Loss, two-stage classification, oversampling — could budge the F1 score. But dropping the batch size from 2048 to 128, a single line of code, did. A textbook lesson, learned the hard way from my own four-year-old code.

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
