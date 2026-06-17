## dev/
開発・研究ドキュメントとユーティリティ。コードではなく、実験記録・リーダーボード・データ準備スクリプト・分析ノートブックが置かれています。 
| ファイル | 役割 | 
|---------|------|
| LOG.md | 実験用ログ(成功と失敗を記録する) | 
| LEADERBOARD.md | "Time-to-GPT-2" リーダーボードの記録と参加方法 | 
| gen_synthetic_data.py | SFT用の合成会話データ生成 | 
|repackage_data_reference.py | 事前学習データの再パッケージ | 
| scaling_analysis.ipynb / estimate_gpt3_core.ipynb | スケーリング則の分析ノートブック | 

## nanochat/
このプロジェクトのコアライブラリ。LLMのライフサイクル全体を支える実装が集まっている。

| ファイル	| 役割 |
|---------|------|
| gpt.py | GPTモデル本体の定義する |
| dataloader.py	| 分散データローダー |
| tokenizer.py | トークナイザー |
| optim.py | MuonAdamWオプティマイザー |
| fp8.py | FP8精度トレーニング (Float8Linear) |
| engine.py | 推論エンジン・KVキャッシュ |
| ui.html | WebチャットUI |
| checkpoint_manager.py | チェックポイント管理 |
| report.py | 学習レポート生成 |

## runs/
学習実行のエントリーポイントとなるシェルスクリプト群。データ取得からトークナイザー訓練、事前学習、SFT、評価まで一連のパイプラインをまとめて実行します。

ファイル	役割
speedrun.sh	8xH100で GPT-2超えを目指すメインの実行スクリプト
miniseries.sh	複数の--depthでモデルシリーズをスイープ
scaling_laws.sh	スケーリング則の調査用スイープ
runcpu.sh	CPU環境での実行用

tasks/
評価・SFT用のタスク（データセット）定義。Task基底クラスを継承した各ベンチマークの実装が入っています。

ファイル	役割
common.py	Task / TaskMixture / TaskSequence 基底クラス
mmlu.py	MMLUベンチマーク
arc.py	ARC (AI2 Reasoning Challenge)
gsm8k.py	GSM8K（数学推論）
humaneval.py	HumanEval（コード生成）
smoltalk.py	SmolTalk会話データ（SFT用）
spellingbee.py	スペリングビータスク
customjson.py	カスタムJSONL形式のタスク
TaskMixtureはSFT時に複数データセットをシャッフル混合し、TaskSequenceはカリキュラム学習のために順番に並べる用途で使われます。 common.py:1-6 common.py:54-58

tests/
自動テスト。pytestで実行するユニットテストが入っています。

ファイル	役割
test_engine.py	Engine・KVCacheの動作テスト（サンプリングの多様性、シード再現性、温度0の決定性など）
test_attention_fallback.py	アテンション実装のフォールバック動作テスト
