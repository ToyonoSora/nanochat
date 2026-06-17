## dev/
開発・研究ドキュメントとユーティリティ。コードではなく、実験記録・リーダーボード・データ準備スクリプト・分析ノートブックが置かれています。 
| ファイル | 役割 | 
|---------|------|
| gen_synthetic_data.py | SFT用の合成会話データ生成 | 
|repackage_data_reference.py | 事前学習データの再パッケージ | 
| scaling_analysis.ipynb / estimate_gpt3_core.ipynb | スケーリング則の分析ノートブック | 

## nanochat/
このプロジェクトのコアライブラリ。LLMのライフサイクル全体を支える実装が集まっている。

| ファイル	| 役割 |
|---------|------|
| checkpoint_manager.py | チェックポイント管理 |
| common.py | 共通ユーティリティ:ログ出力を色分けして見やすくする |
| core_eval.py | ARC/MMLUなど22種のベンチマークのアンサンブルスコア計算 |
| dataloader.py	| 分散データローダー |
| dataset.py | 事前学習データセット管理 |
| engine.py | 推論エンジン・KVキャッシュ |
| execution.py | LLMが生成したPythonコードを安全に実行する |
| flash_attention.py | Flash Attention 3（FA3）とPyTorchのSDPAを自動切り替えする |
| fp8.py | FP8精度トレーニング (Float8Linear) |
| gpt.py | GPTモデル本体の定義する |
| loss_eval.py | ベースモデルの評価指標を計算する |
| optim.py | MuonAdamWオプティマイザー |
| report.py | 学習レポート生成 |
| tokenizer.py | トークナイザー |
| ui.html | WebチャットUI |


## runs/
学習実行のエントリーポイントとなるシェルスクリプト群。データ取得からトークナイザー訓練、事前学習、SFT、評価まで一連のパイプラインをまとめて実行するファイル。

|ファイル|役割|
|-------|----|
|speedrun.sh|8xH100で GPT-2超えを目指すメインの実行スクリプト|
|miniseries.sh|複数の--depthでモデルシリーズをスイープ|
|scaling_laws.sh|スケーリング則の調査用スイープ|
|runcpu.sh|CPU環境での実行用|

## scripts/
各フェーズの実行スクリプト群。nanochat/のコアライブラリを呼び出す形で、AIの学習・評価・推論の各ステップを実行するファイル。ファイル名のプレフィックスで役割が分類されている。

tok_ — トークナイザー関連
|ファイル|役割|
|-------|----|
|tok_train.py|GPT-4スタイルのBPEトークナイザーを学習する|
|tok_eval.py|トークナイザーの圧縮率を評価する|

base_ — ベースモデル（事前学習）関連
|ファイル|役割|
|-------|----|
|base_train.py|GPTモデルの事前学習（単体GPU・分散対応|
|base_eval.py|ベースモデルの評価（CORE / BPB / サンプル生成の3モード|

chat_ — チャットモデル（ファインチューニング・推論）関連
|ファイル|役割|
|-------|----|
|chat_sft.py|Supervised Fine-Tuning（SFT）の実行|
|chat_rl.py|GSM8KによるGRPO/REINFORCEスタイルの強化学習|
|chat_eval.py|チャットモデルのベンチマーク評価（ARC-Easyなど)|
|chat_cli.py|ターミナル上でモデルとチャットするCLIインターフェース|
|chat_web.py|FastAPIベースのWebチャットサーバー（複数GPU対応)|

## tasks/
評価・SFT用のタスク（データセット）定義。Task基底クラスを継承した各ベンチマークの実装が入っている。

|ファイル|役割|
|-------|----|
|arc.py|ARC (AI2 Reasoning Challenge)|
|common.py|Task / TaskMixture / TaskSequence 基底クラス|
|customjson.py|カスタムJSONL形式のタスク|
|gsm8k.py|GSM8K（数学推論）|
|humaneval.py|HumanEval（コード生成）|
|mmlu.py|MMLUベンチマーク|
|smoltalk.py|SmolTalk会話データ（SFT用）|
|spellingbee.py|英単語のスペリング（つづり）の学習|

TaskMixtureはSFT時に複数データセットをシャッフル混合し、TaskSequenceはカリキュラム学習のために順番に並べる用途で使われる。

## tests/
自動テスト。pytestで実行するユニットテストが入っている。

|ファイル|役割|
|-------|----|
|test_engine.py|Engine・KVCacheの動作テスト（サンプリングの多様性、シード再現性、温度0の決定性など）|
|test_attention_fallback.py|アテンション実装のフォールバック動作テスト|
