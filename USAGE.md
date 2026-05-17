# 基于 CNN 的网络数据流量分类（论文复现）

本项目复现 王世元 的本科毕业论文《基于 CNN 的网络数据流量分类》（北京信息科技大学 2022）。论文使用 **Moore 数据集**，对比 **CNN、BP 神经网络、KNN、朴素贝叶斯、SVM、决策树** 共 6 种算法的网络流量分类效果。

---

## 目录结构

```
final/
├── config.py            # 全局配置（路径、超参、12 个分类标签）
├── data_preprocess.py   # Moore 数据集解析、均值填充、类别平衡
├── models_dl.py         # CNN、BP 神经网络（TensorFlow/Keras）
├── models_ml.py         # KNN、朴素贝叶斯、SVM、决策树（sklearn）
├── utils_eval.py        # 混淆矩阵、训练曲线、各分类准确率柱状图
├── main.py              # 主入口，整合 6 种算法的训练与对比
├── requirements.txt     # Python 依赖
├── data/moore/          # 放置 Moore 数据集（entry01~entry10.weka.allclass.arff）
└── outputs/             # 训练产生的图表与 summary.json
```

---

## 1. 环境准备

建议使用 Python 3.9（论文中使用 PyCharm + anaconda3，TensorFlow-CPU 2.8.0）。

```bash
# 创建虚拟环境（可选）
python -m venv venv
venv\Scripts\activate           # Windows
# 或 source venv/bin/activate   # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

---

## 2. 数据集下载

Moore 数据集即剑桥大学 Andrew Moore 等人公开的网络流量数据，共 10 个子集（`entry01.weka.allclass.arff` ~ `entry10.weka.allclass.arff`）。

**下载地址（任选其一）：**

- 官方页面（剑桥大学）：<https://www.cl.cam.ac.uk/research/srg/netos/projects/archive/nprobe/data/papers/sigmetrics/index.html>
- 学术镜像（Kaggle / GitHub 上有多份转储，搜索关键词 `Moore dataset network traffic arff` 即可）

将解压后的 10 个 `.arff` 文件放入 `data/moore/`：

```
data/moore/
├── entry01.weka.allclass.arff
├── entry02.weka.allclass.arff
├── ...
└── entry10.weka.allclass.arff
```

---

## 3. 运行

```bash
# 只运行 CNN（默认）
python main.py --cnn

# 同时运行 CNN 和 BP
python main.py --cnn --bp

# 运行全部 6 种算法（与论文实验一致）
python main.py --all

# 自定义 epochs / batch_size
python main.py --cnn --epochs 25 --batch-size 128

# SVM 在大数据上极慢，可以指定子集大小（论文中 SVM 跑了 3159 秒）
python main.py --svm --svm-subset 20000
```

所有产出（混淆矩阵、训练曲线、综合对比图、`summary.json`）会保存到 `outputs/`。

---

## 4. 与论文对应关系

| 论文章节 | 实现位置 |
| -------- | -------- |
| 2.1 卷积神经网络 | `models_dl.py:build_cnn_model` |
| 2.2.1 BP 神经网络 | `models_dl.py:build_bp_model` |
| 2.2.2 KNN | `models_ml.py:train_knn` |
| 2.2.3 朴素贝叶斯 | `models_ml.py:train_naive_bayes` |
| 2.2.4 决策树 | `models_ml.py:train_decision_tree` |
| 2.2.5 SVM | `models_ml.py:train_svm` |
| 3.1.2 类别平衡 (表 3.2) | `data_preprocess.py:balance_dataset` |
| 3.1.3 均值填充算法 | `data_preprocess.py:balance_dataset`（高斯白噪声） |
| 4.2 数据预处理 (图 4.2) | `data_preprocess.py:parse_arff_line` |
| 4.3 CNN 搭建 (图 4.3) | `models_dl.py:build_cnn_model` |
| 4.5 评价指标 (图 4.9) | `utils_eval.py:plot_confusion_matrix` |
| 5.1 训练曲线 | `utils_eval.py:plot_training_history` |
| 5.3 6 种算法对比 (图 5.10/5.11) | `utils_eval.py:plot_overall_comparison` |

---

## 5. CNN 网络结构（与论文图 4.3 一致）

```
Input(16×16×1)
 → Conv2D(8, 3×3, ReLU, same) → MaxPool(2×2)
 → Conv2D(16, 3×3, ReLU, same) → MaxPool(2×2)
 → Flatten
 → Dense(256, ReLU)
 → Dense(128, ReLU)
 → Dense(12, Softmax)
```

数据预处理时把每个样本的 248 个特征 + 8 个 0 填充凑成 256 维，再 reshape 为 16×16。

---

## 6. 论文实验结果（供参考）

| 算法 | 准确率 | 用时 (s) |
| ---- | ------ | -------- |
| CNN | 99.58 % | 214.95 |
| BP  | 99.52 % | 37.7 |
| KNN | 99.01 % | 313 |
| SVM | 97.97 % | 3159 |
| 决策树 | 99.37 % | 213 |
| 朴素贝叶斯 | 53.87 % | 1.55 |

实际结果会受到数据集子集、随机种子、CPU 性能等影响。

---

## 7. 常见问题

1. **找不到 arff 文件**：检查 `data/moore/` 下是否放好 10 个文件；或修改 `config.py:DATA_DIR`。
2. **SVM 太慢**：用 `--svm-subset 20000` 或更小的值。
3. **TensorFlow 版本兼容**：本项目用 `tensorflow-cpu==2.8.0`；如装 GPU 版本，将 `requirements.txt` 中相应行改成 `tensorflow==2.8.0` 即可。
4. **报内存错误**：调小 `--batch-size`，或在 `data_preprocess.py:prepare_data` 中减少加载的 arff 文件数量。

---

## 致谢

本复现项目基于 王世元 同学的本科毕业论文进行重构。原作者使用的方案、特征选取、模型结构等创意均归原作者所有。
