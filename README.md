<p align="right">
  <b>日本語</b> | <a href="README.en.md">English</a>
</p>

# CNN ベースのネットワークトラフィック分類

通信の中身を一切覗かずに、表面的な統計量だけを使って、ネットワーク上を流れている通信が「Web 閲覧」「メール」「ファイル転送」「サイバー攻撃」など 12 種類のうちどれなのかを CNN で識別するプロジェクトです。

![class-wise accuracy](docs/per_class_accuracy.png)

## なぜこのプロジェクトを作ったのか

家庭用ルーターから企業のファイアウォールまで、ネットワーク機器は長らく「ポート番号」を頼りに通信を分類してきました。SMTP は 25 番、HTTPS は 443 番、というふうにです。しかしこのやり方は、現代のインターネットではほとんど通用しません。

- P2P アプリや VoIP、ゲームなどは動的にポートを切り替えるので、固定ルールでは追跡できない
- 攻撃者は正規のポート (80 や 443) をわざと使うことで、簡単にすり抜けられる
- HTTPS が一般化した結果、パケットの中身を見る Deep Packet Inspection (DPI) も限定的な状況でしか使えない

つまり、ポートと固定ルールに基づく従来のトラフィック分類は、有効性の面でも安全性の面でも限界に来ています。本プロジェクトでは、通信の中身を一切見ずに、「1 秒間のパケット数」「平均パケットサイズ」「TCP フラグの出現頻度」など 248 種類の表面的な統計量だけを材料にして、機械学習が通信種別をどこまで識別できるかを検証しました。中身を見ないので、通信が暗号化されていても、ユーザのプライバシーに踏み込めない状況でも動作します。

## このリポジトリの構成

リポジトリは 2 つの段階から成っています。

**第 1 段階: 2022 年の学部卒業研究の再現**
当時 TensorFlow で書いた、CNN を含む 6 種類のアルゴリズム比較 (CNN, BP NN, KNN, Naive Bayes, SVM, Decision Tree) を、4 年後の自分が読み直しながら再構築しました。再現作業中、前処理コードに紛れていたバグ — クラス名 `FTP-CONTROL` と `INTERACTIVE` 内の `N` まで誤って書き換えていたラベル置換の不具合 — も発見・修正しています。

**第 2 段階: 改善版の構築**
PyTorch + GPU 環境に移行したうえで、「表形式の統計量を 16×16 の疑似画像として扱う 2D CNN は構造的に不自然なのではないか」という仮説の検証から始め、1D CNN、Dilated 畳み込み、Focal Loss、二段階分類器、派生特徴量、バッチサイズの系統的検証など、十数通りの構成を比較しました。

## 結果

公平な比較のため、原論文の構成と提案手法を同じ PyTorch + GPU パイプラインで再評価しました。

| 構成 | Batch | Overall | ATTACK Precision | ATTACK F1 | 時間 |
|------|------:|--------:|----------------:|----------:|-----:|
| 原論文 2D CNN (filters 8/16) | 128 | 98.53% | 88.44% | 78.46% | 38.4s |
| 原論文 Section 5.2 改良版 (filters 16/32) | 64 | 99.08% | 95.64% | 81.32% | 76.6s |
| **提案: Dilated 1D CNN** | 64 | **99.12%** | **99.35%** | **82.64%** | 89.0s |

提案手法は原論文の最良構成を、Overall、Precision、F1 のすべてで上回りました。特に攻撃検知の Precision 99.35% は、「正常な通信を攻撃と誤判定する確率がほぼゼロ」を意味しており、セキュリティ製品としての実用性に直結する数字だと考えています。

![batch size vs attack F1](docs/batch_sweep.png)

上の図は、バッチサイズと攻撃検知 F1 の関係を表したものです。バッチを小さくするほど性能が単調に向上する — というのは、本検証で最も意外だった発見でした。Focal Loss や二段階分類のような「いかにも効きそうな」手法をいくつ試しても上がらなかった F1 が、「バッチサイズを下げる」たった 1 行の変更で改善するという、教科書的な教訓を 4 年越しに自分のコードから学び直すことになりました。

## 動かし方

```bash
pip install -r requirements.txt
# Moore データセットを data/moore/ に配置 (詳細は USAGE.md)

python main.py --all                            # 6 アルゴリズム比較を再現
python batch_sweep.py --epochs 25 --batches 128 # 推奨最終構成 (GPU 推奨)
```

詳しい実行手順は [USAGE.md](USAGE.md) を、各改善実験の動機・実装・結果は [EXPERIMENTS.md](EXPERIMENTS.md) を参照してください。

## データセット

実験には Cambridge 大学が公開している Moore Dataset (A. Moore et al., 2005) を使用しています。10 個の ARFF ファイルからなる、約 25 万件のラベル付き通信フローです。ダウンロード手順は [USAGE.md](USAGE.md) に書きました。

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
