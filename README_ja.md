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
