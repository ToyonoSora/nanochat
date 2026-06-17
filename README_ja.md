# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat は、LLM（大規模言語モデル）を学習するための最もシンプルな実験用フレームワークです。単一の GPU ノードで動作するよう設計されており、コードは最小限かつ改変しやすく、トークナイズ・事前学習・ファインチューニング・評価・推論・チャット UI といった LLM の主要なすべてのステージをカバーしています。たとえば、2019 年に約 $43,000 かかった GPT-2 相当の LLM を、わずか $48（8×H100 GPU ノードで約 2 時間）で自分自身で学習し、おなじみの ChatGPT 風 Web UI で会話することができます。スポットインスタンスを使えば、総コストは約 $15 に近づきます。より一般的には、nanochat は `--depth`（GPT トランスフォーマーモデルのレイヤー数）という単一の複雑さダイヤルを設定するだけで、コンピュート最適なモデルのミニシリーズ全体を学習できるよう、すぐに使える設定になっています（GPT-2 相当の能力はおよそ depth 26 に相当します）。その他のすべてのハイパーパラメータは自動的に決定されます。

リポジトリに関する質問は、[DeepWiki](https://deepwiki.com/karpathy/nanochat)（Devin/Cognition 製）を使ってリポジトリについて質問するか、[Discussions タブ](https://github.com/karpathy/nanochat/discussions)を利用するか、Discord の [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) チャンネルにお越しください。

## Time-to-GPT-2 リーダーボード

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

nanochat は依存関係の管理に [uv](https://docs.astral.sh/uv/) を使用しています。インストール方法：

```bash
uv sync --extra gpu    # CUDA（A100/H100 等）の場合
uv sync --extra cpu    # CPU のみ / MPS の場合
source .venv/bin/activate
```

開発用（pytest、matplotlib、ipykernel、transformers 等を追加）：

```bash
uv sync --extra gpu --group dev
```

### GPT-2 を再現して会話する

最も楽しい体験は、自分の GPT-2 を学習して会話することです。そのためのパイプライン全体が [runs/speedrun.sh](runs/speedrun.sh) という単一ファイルに含まれており、8×H100 GPU ノードで実行するよう設計されています。お好みのプロバイダー（例：[Lambda](https://lambda.ai/service/gpu-cloud)）から新しい 8×H100 GPU ボックスを起動し、学習スクリプトを開始してください：

```bash
bash runs/speedrun.sh
```

約 3 時間かかるため、screen セッション内で実行することをお勧めします。完了したら、ChatGPT 風 Web UI で会話できます。ローカルの uv 仮想環境がアクティブになっていることを確認し（`source .venv/bin/activate` を実行）、サーバーを起動してください：

```bash
python -m scripts.chat_web
```

表示された URL にアクセスしてください。例えば Lambda では、ノードのパブリック IP にポートを付けてアクセスします（例：[http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/)）。あとは ChatGPT と同じように LLM と会話してください！物語や詩を書かせたり、「あなたは誰ですか？」と聞いてハルシネーションを見たり、空が青い理由や緑の理由を聞いてみてください。スピードランは 4e19 FLOPs の能力モデルなので、幼稚園児と話しているような感覚です :)

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

補足事項：

- Ampere 8×A100 GPU ノードでも問題なく動作しますが、少し遅くなります。
- `torchrun` を省略することで単一 GPU でも同様に動作し、ほぼ同一の結果が得られます（コードは自動的に勾配累積に切り替わります）が、8 倍の時間がかかります。
- GPU の VRAM が 80GB 未満の場合、一部のハイパーパラメータを調整しないと OOM（メモリ不足）になります。スクリプト内の `--device-batch-size` を探して、32（デフォルト）から 16、8、4、2、または 1 に減らしてください。
- コードの大部分は標準的な PyTorch なので、xpu、mps 等でも動作するはずですが、すべてのコードパスを個人的に検証しているわけではないため、問題が生じる可能性があります。

## 研究

研究者の方で nanochat の改善に協力したい場合は、[runs/scaling_laws.sh](runs/scaling_laws.sh) と [runs/miniseries.sh](runs/miniseries.sh) の 2 つのスクリプトが参考になります。関連ドキュメントは [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) を参照してください。素早い実験（約 5 分の事前学習）には、12 層モデル（GPT-1 サイズ）の学習が好みです：

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

これは wandb（実行名 "d12"）を使用し、最終ステップのみ CORE メトリクスを実行し、中間チェックポイントのサンプリングと保存を行いません。コードを変更して d12（または d16 等）を再実行し、改善されたかどうかを確認するというイテレーションループが好みです。実行が有効かどうかを確認するために、wandb のプロットで以下を監視します：

1. `val_bpb`（語彙サイズに依存しない単位のビット/バイトで表した検証損失）を `step`、`total_training_time`、`total_training_flops` の関数として
2. `core_metric`（DCLM CORE スコア）
3. VRAM 使用率、`train/mfu`（モデル FLOPS 使用率）、`train/tok_per_sec`（学習スループット）

例は[こちら](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)を参照してください。

重要な点として、nanochat はトランスフォーマーの深さという単一の複雑さダイヤルを中心に書かれ設定されています。この単一の整数が他のすべてのハイパーパラメータ（トランスフォーマーの幅、ヘッド数、学習率の調整、学習期間、重み減衰など）を自動的に決定し、学習済みモデルがコンピュート最適になるようにします。ユーザーはこれらについて考えたり設定したりする必要はなく、`--depth` を使って小さいモデルか大きいモデルかを指定するだけで、すべてが「うまく動作」します。深さをスイープすることで、さまざまなサイズのコンピュート最適モデルの nanochat ミニシリーズが得られます。

## CPU / MPS での実行

[runs/runcpu.sh](runs/runcpu.sh) スクリプトは、CPU または Apple Silicon での実行の非常にシンプルな例を示しています。学習する LLM を大幅に縮小して、数十分という合理的な時間内に収めています。この方法では強力な結果は得られません。

## 精度 / dtype

nanochat は `torch.amp.autocast` を使用しません。代わりに、精度は `nanochat/common.py` で定義された単一のグローバル変数 `COMPUTE_DTYPE` を通じて明示的に管理されます。デフォルトではハードウェアに基づいて自動検出されます：

| ハードウェア | デフォルト dtype | 理由 |
|------------|--------------|------|
| CUDA SM 80+（A100、H100 等） | `bfloat16` | ネイティブ bf16 テンソルコア |
| CUDA SM < 80（V100、T4 等） | `float32` | bf16 なし；fp16 は `NANOCHAT_DTYPE=float16` で利用可能（GradScaler を使用） |
| CPU / MPS | `float32` | 低精度テンソルコアなし |

`NANOCHAT_DTYPE` 環境変数でデフォルトを上書きできます：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # fp32 を強制
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # bf16 を強制
```

仕組み：モデルの重みは fp32 で保存されます（オプティマイザの精度のため）が、カスタム `Linear` レイヤーがフォワードパス中にそれらを `COMPUTE_DTYPE` にキャストします。埋め込みはメモリ節約のために直接 `COMPUTE_DTYPE` で保存されます。これにより、autocast と同じ混合精度の利点が得られますが、どの精度で何が実行されるかを完全に明示的に制御できます。

注意：`float16` 学習では、勾配のアンダーフローを防ぐために `base_train.py` で `GradScaler` が自動的に有効になります。SFT もこれをサポートしていますが、RL は現在サポートしていません。fp16 での推論はどこでも問題なく動作します。

## ガイド

参考になる情報を含むガイドを公開しています（新しい順）：

- [2026年2月1日: GPT-2 を $100 以下で超える：nanochat の旅](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420)：最初の nanochat モデルミニシリーズを文書化
- nanochat に新しい能力を追加するには、[ガイド：strawberry の r を数える（および一般的な能力の追加方法）](https://github.com/karpathy/nanochat/discussions/164)を参照
- nanochat をカスタマイズするには、Discussions の[ガイド：nanochat にアイデンティティを注入する](https://github.com/karpathy/nanochat/discussions/139)を参照。合成データ生成と SFT ステージへのデータ混合を通じて nanochat の個性を調整する方法を説明しています。
- [2025年10月13日：nanochat オリジナル投稿](https://github.com/karpathy/nanochat/discussions/1)：nanochat を紹介していますが、現在は一部の情報が古くなっており、モデルも現在の master より古い（結果も悪い）です。

## ファイル構成

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # アイデンティティ用の合成データ例
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # 事前学習データシャード生成
├── nanochat
│   ├── __init__.py                 # 空
│   ├── checkpoint_manager.py       # モデルチェックポイントの保存/読み込み
│   ├── common.py                   # 各種小ユーティリティ
│   ├── core_eval.py                # ベースモデルの CORE スコア評価（DCLM 論文）
│   ├── dataloader.py               # トークナイズ分散データローダー
│   ├── dataset.py                  # 事前学習データのダウンロード/読み込みユーティリティ
│   ├── engine.py                   # KV キャッシュを使った効率的なモデル推論
│   ├── execution.py                # LLM がツールとして Python コードを実行できるようにする
│   ├── gpt.py                      # GPT nn.Module トランスフォーマー
│   ├── logo.svg
│   ├── loss_eval.py                # ビット/バイト評価（損失の代わり）
│   ├── optim.py                    # AdamW + Muon オプティマイザ（1GPU および分散）
│   ├── report.py                   # nanochat レポート作成ユーティリティ
│   ├── tokenizer.py                # GPT-4 スタイルの BPE トークナイザーラッパー
│   └── ui.html                     # nanochat フロントエンドの HTML/CSS/JS
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # ミニシリーズ学習スクリプト
│   ├── runcpu.sh                   # CPU/MPS での実行例
│   ├── scaling_laws.sh             # スケーリング則実験
│   └── speedrun.sh                 # ~$100 の nanochat d20 を学習
├── scripts
│   ├── base_eval.py                # ベースモデル：CORE スコア、ビット/バイト、サンプル
│   ├── base_train.py               # ベースモデル：学習
│   ├── chat_cli.py                 # チャットモデル：CLI で会話
│   ├── chat_eval.py                # チャットモデル：タスク評価
│   ├── chat_rl.py                  # チャットモデル：強化学習
│   ├── chat_sft.py                 # チャットモデル：SFT 学習
│   ├── chat_web.py                 # チャットモデル：WebUI で会話
│   ├── tok_eval.py                 # トークナイザー：圧縮率評価
│   └── tok_train.py                # トークナイザー：学習
├── tasks
│   ├── arc.py                      # 多肢選択式理科問題
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 任意の jsonl 会話からタスクを作成
│   ├── gsm8k.py                    # 8K 小学校算数問題
│   ├── humaneval.py                # 名称は誤解を招くが、シンプルな Python コーディングタスク
│   ├── mmlu.py                     # 幅広いトピックの多肢選択問題
│   ├── smoltalk.py                 # HF の SmolTalk 複合データセット
│   └── spellingbee.py              # モデルにスペル/文字数を教えるタスク
├── tests
│   └── test_engine.py
└── uv.lock
```

## コントリビューション

nanochat の目標は、$1,000 以下の予算でエンドツーエンドで扱えるマイクロモデルの最先端を改善することです。アクセシビリティとは全体的なコストだけでなく、認知的な複雑さについても言えます。nanochat は網羅的に設定可能な LLM「フレームワーク」ではありません。巨大な設定オブジェクト、モデルファクトリー、if-then-else の怪物はコードベースにありません。最初から最後まで実行して会話できる ChatGPT モデルを生成するよう設計された、単一の、一貫した、最小限の、読みやすい、改変しやすい、最大限フォーク可能な「強力なベースライン」コードベースです。現在、個人的に最も興味深いのは GPT-2 までのレイテンシを短縮すること（つまり CORE スコアを 0.256525 以上にすること）です。現在は約 3 時間かかりますが、事前学習ステージを改善することでさらに短縮できます。

現在の AI ポリシー：開示。PR を提出する際は、LLM が実質的に貢献した部分、自分で書いていない部分、または完全に理解していない部分を申告してください。

## 謝辞

- nanochat という名前は、事前学習のみをカバーしていた以前のプロジェクト [nanoGPT](https://github.com/karpathy/nanoGPT) に由来しています。
- nanochat は [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) にも触発されています。このリポジトリは nanoGPT リポジトリを明確なメトリクスとリーダーボードでゲーム化し、事前学習のアイデアと実装の多くを借用しています。
- [HuggingFace](https://huggingface.co/) に fineweb と smoltalk を提供していただいたことに感謝します。
- このプロジェクトの開発に使用した計算リソースを提供してくれた [Lambda](https://lambda.ai/service/gpu-cloud) に感謝します。
- アドバイスと指導をいただいた LLM の達人 🧙‍♂️ Alec Radford に感謝します。
- nanochat の issue、プルリクエスト、ディスカッションの管理を手伝ってくれたリポジトリ管理者 Sofie [@svlandeg](https://github.com/svlandeg) に感謝します。

## 引用

研究で nanochat が役立った場合は、以下のように引用してください：

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

MIT
