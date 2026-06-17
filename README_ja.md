# nanochat  
  
![nanochat logo](dev/nanochat.png)  
![scaling laws](dev/scaling_laws_jan26.png)  
  
nanochat は、LLM（大規模言語モデル）を学習するための最もシンプルな実験用フレームワークです。単一の GPU ノードで動作するよう設計されており、コードは最小限かつ改変しやすく、トークナイズ・事前学習・ファインチューニング・評価・推論・チャット UI といった LLM の主要なすべてのステージをカバーしています。たとえば、2019 年に約 $43,000 かかった GPT-2 相当の LLM を、わずか $48（8×H100 GPU ノードで約 2 時間）で自分自身で学習し、おなじみの ChatGPT 風 Web UI で会話することができます。スポットインスタンスを使えば、総コストは約 $15 に近づきます。より一般的には、nanochat は `--depth`（GPT トランスフォーマーモデルのレイヤー数）という単一の複雑さダイヤルを設定するだけで、コンピュート最適なモデルのミニシリーズ全体を学習できるよう、すぐに使える設定になっています（GPT-2 相当の能力はおよそ depth 26 に相当します）。その他のすべてのハイパーパラメータ（トランスフォーマーの幅、ヘッド数、学習率の調整、学習期間、重み減衰など）は最適な方法で自動的に計算されます。  
  
リポジトリに関する質問は、[DeepWiki](https://deepwiki.com/karpathy/nanochat)（Devin/Cognition 製）を使ってリポジトリについて質問するか、[Discussions タブ](https://github.com/karpathy/nanochat/discussions)を利用するか、Discord の [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) チャンネルにお越しください。  
  
## GPT2 リーダーボードまでの時間  
  
現在、開発の主な焦点は最も多くの計算量を必要とする事前学習ステージのチューニングです。modded-nanogpt リポジトリに触発され、進歩とコミュニティの協力を促進するために、nanochat は「GPT-2 スピードラン」のリーダーボードを維持しています。これは DCLM CORE スコアで測定した GPT-2 相当の能力に達するまでの実時間です。[runs/speedrun.sh](runs/speedrun.sh) スクリプトは常に GPT-2 相当モデルを学習して会話するための参照方法を反映しています。現在のリーダーボードは以下の通りです：  
  
| # | 時間 | val_bpb | CORE | 説明 | 日付 | コミット | 貢献者 |  
|---|------|---------|------|------|------|--------|--------|  
| 0 | 168 時間 | - | 0.2565 | OpenAI オリジナル GPT-2 チェックポイント | 2019 | - | OpenAI |  
| 1 | 3.04 | 0.74833 | 0.2585 | d24 ベースライン、わずかに過学習 | 2026年1月29日 | 348fbb3 | @karpathy |  
| 2 | 2.91 | 0.74504 | 0.2578 | d26 わずかに未学習 **+fp8** | 2026年2月2日 | a67eba3 | @karpathy |  
| 3 | 2.76 | 0.74645 | 0.2602 | 総バッチサイズを 1M トークンに増加 | 2026年2月5日 | 2c062aa | @karpathy |  
| 4 | 2.02 | 0.71854 | 0.2571 | データセットを NVIDIA ClimbMix に変更 | 2026年3月4日 | 324e69c | @ddudek @karpathy |  
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [ラウンド 1](https://x.com/karpathy/status/2031135152349524125) | 2026年3月9日 | 6ed7d1d | @karpathy |  
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch ラウンド 2 | 2026年3月14日 | a825e63 | @karpathy |  
  
主要な指標は「GPT-2 までの時間」です。8×H100 GPU ノードで GPT-2（1.6B）の CORE スコア（0.256525）を上回るために必要な実時間です。2019 年の GPT-2 学習コストは約 $43,000 でしたが、7 年間にわたるスタック全体の多くの進歩により、現在ははるかに速く、$100 以下で実現できます（現在の約 $3/GPU/時間で、8×H100 ノードは約 $24/時間、2 時間で約 $48）。  
  
リーダーボードの解釈と貢献方法については [dev/LEADERBOARD.md](dev/LEADERBOARD.md) を参照してください。  
  
## はじめに

### セットアップ

nanochat は依存関係管理に `uv` を使用しています。インストール方法は以下の通りです。

```bash
uv sync --extra gpu    # CUDA 環境向け (A100/H100 など)
uv sync --extra cpu    # CPU または Apple Silicon (MPS) 向け
source .venv/bin/activate
```

開発用ツール（pytest、matplotlib、ipykernel、transformers など）もインストールする場合は次を実行してください。

```bash
uv sync --extra gpu --group dev
```

### GPT-2 を再現して対話する

nanochat の最も楽しい使い方は、自分自身で GPT-2 クラスのモデルを学習し、そのモデルと会話することです。

そのための一連のパイプラインは、単一のスクリプト `runs/speedrun.sh` にまとめられています。このスクリプトは 8×H100 GPU ノードでの実行を想定しています。

お好みのクラウドプロバイダー（例: Lambda）で新しい 8×H100 GPU ノードを起動し、以下を実行してください。

```bash
bash runs/speedrun.sh
```

実行には約 3 時間かかるため、`screen` や `tmux` セッション内で実行することをおすすめします。

学習が完了したら、ChatGPT ライクな Web UI からモデルと対話できます。まず仮想環境を有効化し、Web サーバーを起動します。

```bash
source .venv/bin/activate
python -m scripts.chat_web
```

その後、表示された URL にアクセスしてください。

例えば Lambda 上で実行している場合は、ノードのパブリック IP アドレスとポート番号を使用します。

```text
http://209.20.xxx.xxx:8000/
```

ブラウザで開けば、ChatGPT と同じようにモデルと会話できます。

試しに以下のようなことを尋ねてみてください。

* 小説や詩を書かせる
* 「私は誰？」と質問して幻覚（ハルシネーション）を観察する
* 空が青い理由を聞く
* 空が緑だと主張させてみる

この Speedrun モデルは約 4e19 FLOPs 規模の能力を持っており、感覚的には「幼稚園児と会話している」ようなレベルです。

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

補足事項:

* Ampere 世代の 8×A100 GPU ノードでも問題なく動作しますが、実行速度はやや遅くなります。
* `torchrun` を使用しなければ単一 GPU でも動作し、結果もほぼ同等になります（自動的に勾配累積へ切り替わります）。ただし学習時間は約 8 倍になります。
* GPU メモリが 80GB 未満の場合は、一部のハイパーパラメータを調整する必要があります。VRAM が不足する場合はスクリプト内の `--device-batch-size` を小さくしてください。

例:

```text
32 (デフォルト) → 16 → 8 → 4 → 2 → 1
```

* コードは比較的標準的な PyTorch 実装であるため、CUDA 以外にも XPU や MPS など PyTorch が対応する環境で動作するはずです。ただし、すべての実行経路を十分に検証しているわけではないため、一部に未検証の箇所が存在する可能性があります。

## 研究用途

nanochat の改善に取り組みたい研究者向けには、以下のスクリプトが特に重要です。

* `runs/scaling_laws.sh`
* `runs/miniseries.sh`

関連ドキュメントとして「Jan 7 miniseries v1」も参照してください。

短時間（約 5 分）の事前学習実験を行う場合、作者は GPT-1 相当の 12 層モデルをよく利用しています。

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

この設定では:

* Weights & Biases の実行名は `"d12"`
* CORE メトリクスは最終ステップのみ評価
* 学習途中のサンプル生成を無効化
* 中間チェックポイントの保存を無効化

コードを変更した後に d12 や d16 を再実行し、改善したかどうかを確認する実験サイクルに適しています。

評価時には、以下の指標を重点的に確認します。

1. `val_bpb`

   * 語彙サイズに依存しない bits-per-byte 単位の検証損失
   * `step`、`total_training_time`、`total_training_flops` に対する推移を確認

2. `core_metric`

   * DCLM CORE スコア

3. システム指標

   * VRAM 使用率
   * `train/mfu`（Model FLOPS Utilization）
   * `train/tok_per_sec`（学習スループット）

nanochat の最も重要な設計思想は、「Transformer の深さ（depth）」という単一の複雑さパラメータを中心に構成されている点です。

この整数値ひとつから以下が自動的に決定されます。

* モデル幅
* Attention Head 数
* 学習率
* 学習ステップ数
* Weight Decay
* その他のハイパーパラメータ

その結果、ユーザーは単に

> 「もっと小さいモデルが欲しい」
>
> 「もっと大きいモデルが欲しい」

という意図で `--depth` を変更するだけで済みます。

深さを変化させることで、計算効率的に最適化された nanochat モデル群（Miniseries）を得ることができます。

現時点では GPT-2 クラスの能力は、おおよそ d24〜d26 の範囲に位置しています。

ただし、リポジトリへの改善提案は、特定の深さだけでなくすべての depth 設定で有効であることが求められます。

## CPU / MPS での実行

`runs/runcpu.sh` には、CPU や Apple Silicon 上で実行するためのシンプルな例が含まれています。

実行時間を現実的な数十分程度に収めるため、学習するモデルは大幅に縮小されています。

そのため、この方法では高い性能は期待できません。

## 精度（dtype）

nanochat は `torch.amp.autocast` を使用しません。

代わりに、`nanochat/common.py` で定義されるグローバル変数 `COMPUTE_DTYPE` を利用して精度を明示的に管理します。

デフォルト設定はハードウェアに応じて自動選択されます。

| ハードウェア                      | デフォルト dtype | 理由                                                           |
| --------------------------- | ----------- | ------------------------------------------------------------ |
| CUDA SM 80+ (A100, H100 など) | `bfloat16`  | bf16 Tensor Core をネイティブサポート                                  |
| CUDA SM < 80 (V100, T4 など)  | `float32`   | bf16 非対応。`NANOCHAT_DTYPE=float16` で fp16 利用可能（GradScaler 使用） |
| CPU / MPS                   | `float32`   | 低精度 Tensor Core 非対応                                          |

環境変数 `NANOCHAT_DTYPE` を利用して上書きできます。

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"
```

```bash
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train
```

内部的な仕組み:

* モデル重みは最適化精度を維持するため fp32 で保存
* カスタム `Linear` レイヤーが Forward 時に `COMPUTE_DTYPE` へ変換
* Embedding はメモリ節約のため最初から `COMPUTE_DTYPE` で保持

これにより、autocast と同様の Mixed Precision の利点を得ながら、どの演算をどの精度で実行するかを明示的に制御できます。

### float16 に関する注意

`float16` 学習時は、勾配のアンダーフローを防ぐために `base_train.py` 内で自動的に `GradScaler` が有効になります。

* SFT（教師ありファインチューニング）: 対応済み
* RL（強化学習）: 現時点では未対応

なお、推論時の fp16 はすべての環境で問題なく利用できます。


## ガイド

nanochat に関する各種ガイドを公開しています。新しいものから順に掲載しています。

* [2026年2月1日: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)

  * 100ドル未満で GPT-2 を超えるまでの nanochat 開発の歩みを解説しています。

* [2026年1月7日: Miniseries v1](https://github.com/karpathy/nanochat/discussions/420)

  * nanochat 初の「Miniseries」モデル群について解説しています。

* nanochat に新しい能力を追加したい場合は、
  [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164)
  を参照してください。

* nanochat の個性や人格をカスタマイズしたい場合は、
  [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139)
  を参照してください。合成データの生成と SFT（教師ありファインチューニング）への組み込みによって、モデルの人格を調整する方法が説明されています。

* [2025年10月13日: nanochat 初公開記事](https://github.com/karpathy/nanochat/discussions/1)

  * nanochat の最初の紹介記事です。現在では一部情報が古くなっており、当時のモデル性能も現在の master ブランチより低いものとなっています。

## ディレクトリ構成

```text
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # 人格付与用の合成データ生成サンプル
│   ├── generate_logo.html          # ロゴ生成用 HTML
│   ├── nanochat.png                # nanochat ロゴ画像
│   └── repackage_data_reference.py # 事前学習データシャード生成
├── nanochat
│   ├── __init__.py                 # 空ファイル
│   ├── checkpoint_manager.py       # モデルチェックポイントの保存・読み込み
│   ├── common.py                   # 共通ユーティリティ
│   ├── core_eval.py                # CORE スコア評価（DCLM 論文準拠）
│   ├── dataloader.py               # 分散トークナイズデータローダー
│   ├── dataset.py                  # 事前学習データの取得・読み込み
│   ├── engine.py                   # KV キャッシュ対応高速推論エンジン
│   ├── execution.py                # Python コード実行ツール機能
│   ├── gpt.py                      # GPT Transformer 本体
│   ├── logo.svg                    # ロゴ SVG
│   ├── loss_eval.py                # Bits Per Byte 評価
│   ├── optim.py                    # AdamW + Muon オプティマイザ
│   ├── report.py                   # nanochat レポート生成補助
│   ├── tokenizer.py                # GPT-4 スタイル BPE トークナイザ
│   └── ui.html                     # Web UI (HTML/CSS/JavaScript)
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries 学習スクリプト
│   ├── runcpu.sh                   # CPU / MPS 実行例
│   ├── scaling_laws.sh             # スケーリング則実験
│   └── speedrun.sh                 # 約100ドルで GPT-2 クラスを学習
├── scripts
│   ├── base_eval.py                # 基盤モデル評価
│   ├── base_train.py               # 基盤モデル学習
│   ├── chat_cli.py                 # CLI チャット
│   ├── chat_eval.py                # チャットモデル評価
│   ├── chat_rl.py                  # 強化学習 (RL)
│   ├── chat_sft.py                 # SFT 学習
│   ├── chat_web.py                 # Web UI チャット
│   ├── tok_eval.py                 # トークナイザ評価
│   └── tok_train.py                # トークナイザ学習
├── tasks
│   ├── arc.py                      # 理科系選択問題
│   ├── common.py                   # TaskMixture / TaskSequence
│   ├── customjson.py               # 任意 JSONL 会話データから Task 作成
│   ├── gsm8k.py                    # 小学校レベル算数問題
│   ├── humaneval.py                # 簡易 Python コーディング課題
│   ├── mmlu.py                     # 幅広い分野の選択問題
│   ├── smoltalk.py                 # Hugging Face SmolTalk データセット
│   └── spellingbee.py              # スペル・文字数カウント学習タスク
├── tests
│   └── test_engine.py
└── uv.lock
```

## コントリビューション

nanochat の目標は、

**「1000ドル未満の予算でエンドツーエンドに扱える小規模 LLM の性能を向上させること」**

です。

ここでいうアクセシビリティとは、単なる計算コストの低さだけではありません。

認知的な複雑さの低さも重要です。

nanochat は、無数の設定項目を持つ巨大な LLM フレームワークではありません。

以下のようなものは意図的に排除しています。

* 巨大な設定オブジェクト
* 複雑なモデルファクトリ
* if-then-else が入り乱れたコード

代わりに nanochat は、

* シンプル
* 一貫性がある
* 読みやすい
* 改造しやすい
* フォークしやすい

という特徴を持つ「強力なベースライン実装」を目指しています。

リポジトリを最初から最後まで実行するだけで、実際に会話可能な ChatGPT スタイルのモデルを構築できます。

現在もっとも興味深い研究テーマは、

**GPT-2 レベルに到達するまでの時間をさらに短縮すること**

です。

現在は約 3 時間で GPT-2（CORE スコア 0.256525）を超えられますが、事前学習工程の改善によってさらに短縮できる可能性があります。

### AI 利用ポリシー

現在のポリシーは **Disclosure（開示）** です。

Pull Request を提出する際は、

* LLM が大きく関与した部分
* 自分で書いていないコード
* 自分で十分理解できていないコード

について明示してください。

## 謝辞

* nanochat の名称は、以前のプロジェクトである nanoGPT に由来しています。nanoGPT は事前学習のみを対象としていました。

* nanochat は modded-nanoGPT にも大きく影響を受けています。明確な指標やランキングによって学習をゲーム化したプロジェクトであり、事前学習部分のアイデアや実装を一部参考にしています。

* Hugging Face の FineWeb および SmolTalk データセットに感謝します。

* 本プロジェクトの開発に使用した計算資源を提供してくれた Lambda に感謝します。

* 「Chief LLM Whisperer」こと Alec Radford 氏の助言と指導に感謝します。

* Issue、Pull Request、Discussions の管理を支援してくれたリポジトリ管理者 Sofie (@svlandeg) に感謝します。

## 引用

研究で nanochat を利用した場合は、以下の形式で引用してください。

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## ライセンス

MIT License
