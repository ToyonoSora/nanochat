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
  
## 始める

### セットアップ

NanoChatは依存管理に[uv](https://docs.astral.sh/uv/)を使用しています。インストール方法:

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

開発用(pytest、matplotlib、ipykernel、transformerなどを追加)。

```bash
uv sync --extra gpu --group dev
```

### 繁殖してGPT-2と話す

一番楽しいのは、自分でGPT-2をトレーニングして会話することです。そのためのパイプライン全体は、8XH100 GPUノード上で動作するよう設計された単一のファイル[runs/speedrun.sh](runs/speedrun.sh), に含まれています。お気に入りのプロバイダー(e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)),から新しい8XH100 GPUボックスを起動し、トレーニングスクリプトを起動します:

```bash
bash runs/speedrun.sh
```

You may wish to do so in a screen session as this will take ~3 hours to run. Once it's done, you can talk to it via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

A few more notes:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device-batch-size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges.

## Research

If you are a researcher and wish to help improve nanochat, two scripts of interest are [runs/scaling_laws.sh](runs/scaling_laws.sh) and [runs/miniseries.sh](runs/miniseries.sh). See [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) for related documentation. For quick experimentation (~5 min pretraining runs) my favorite scale is to train a 12-layer model (GPT-1 sized), e.g. like this:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This uses wandb (run name "d12"), only runs the CORE metric on last step, and it doesn't sample and save intermediate checkpoints. I like to change something in the code, re-run a d12 (or a d16 etc) and see if it helped, in an iteration loop. To see if a run helps, I like to monitor the wandb plots for:

1. `val_bpb` (validation loss in vocab-size-invariant units of bits per byte) as a function of `step`, `total_training_time` and `total_training_flops`.
2. `core_metric` (the DCLM CORE score)
3. VRAM utilization, `train/mfu` (Model FLOPS utilization), `train/tok_per_sec` (training throughput)

See an example [here](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044).

The important thing to note is that nanochat is written and configured around one single dial of complexity - the depth of the transformer. This single integer automatically determines all other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) so that the trained model comes out compute optimal. The idea is that the user doesn't have to think about or set any of this, they are simply asking for a smaller or bigger model using `--depth`, and everything "just works". By sweeping out the depth, you achieve the nanochat miniseries of compute optimal models at various sizes. GPT-2 capability model (which is of most interest at the moment) happens to be somewhere around d24-d26 range with the current code. But any candidate changes to the repo have to be principled enough that they work for all settings of depth.

## Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows a very simple example of running on CPU or Apple Silicon. It dramatically shrinks the LLM that is being trained to make things fit into a reasonable time interval of a few ten minutes of training. You will not get strong results in this way.

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`). By default this is auto-detected based on your hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | No reduced-precision tensor cores |

You can override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

How it works: model weights are stored in fp32 (for optimizer precision), but our custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE` to save memory. This gives us the same mixed-precision benefit as autocast but with full explicit control over what runs in which precision.

Note: `float16` training automatically enables a `GradScaler` in `base_train.py` to prevent gradient underflow. SFT supports this too but RL currently does not. Inference in fp16 works fine everywhere.

## Guides

I've published a number of guides that might contain helpful information, most recent to least recent:

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) documents the first nanochat miniseries of models.
- To add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).
- To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into the SFT stage.
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) introducing nanochat, though now it contains some deprecated information and the model is a lot older (with worse results) than current master.

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretraining data shard generation
├── nanochat
│   ├── __init__.py                 # empty
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Train the ~$100 nanochat d20
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model: talk to over CLI
│   ├── chat_eval.py                # Chat model: eval tasks
│   ├── chat_rl.py                  # Chat model: reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model: talk to over WebUI
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
└── uv.lock
```

## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part personally is speeding up the latency to GPT-2 (i.e. getting a CORE score above 0.256525). Currently this takes ~3 hours, but by improving the pretraining stage we can improve this further.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
