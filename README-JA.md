# グラフ検索拡張生成に基づく学際的知識発見研究

日本語版ドキュメントです。内容は `README.md` と同じ方針で、現在の研究実装版に合わせて整理しています。

## プロジェクト概要
本プロジェクトは、学際的知識発見を目的とした GraphRAG 研究プロトタイプです。論文「基于图检索增强生成的跨学科知识发现研究」に沿って実装されており、科学文献を対象に、以下の一連の処理を統合しています。

- 大規模言語モデルによる知識抽出
- 知識グラフ構築
- グラフ検索拡張 QA
- 評価と分析

非構造化な学術文献から潜在的な学際的関連を発見し、研究支援に活用することを目的としています。

現在の実装は、汎用 GraphRAG フレームワークの紹介ではなく、次の研究課題に焦点を当てています。

- 学際的な科学文献に対する構造化知識グラフの構築
- グラフに基づく複雑な質問の検索、分解、推論
- AIGC / 大規模言語モデルの教育応用文献を中心とした実験
- 知識グラフ構築評価と QA 評価の二系統の実験基盤
- Web プロトタイプによるアップロード、構築、可視化、QA

## 研究上の位置づけ
本プロジェクトは、一般的なチャットボットではなく、「学際的な科学知識の発見」を主題としています。中核となる研究課題は次のとおりです。

1. 文献メタデータに基づいて学際的な科学知識グラフを自動構築するにはどうすればよいか。
2. GraphRAG を用いて、学際的知識発見の効率、網羅性、説明可能性をどう高めるか。
3. 知識グラフ構築品質と QA 品質をどう評価し、異なるモデルをどう比較するか。

現在の主要サンプルは「大規模言語モデルの教育分野への応用研究」であり、リポジトリ内には `AIGC-EDU` や `AIGC-EDU-test` などのデータセットが含まれています。

## 現在のシステム機能

### 1. 文献アップロードとデータセット管理
- Web UI から `.txt`、`.md`、`.json`、`.pdf`、`.docx`、`.doc` をアップロード可能
- `data/uploaded/<dataset_name>/corpus.json` を自動生成
- データセット一覧、削除、再構築、カスタム schema のアップロードに対応
- `demo` データセットを同梱

### 2. 知識グラフ構築
- 主な入口は `main.py` と `backend.py`
- 中核モジュールは `models/constructor/kt_gen.py`
- schema に基づくエンティティ・関係・属性抽出
- 文書横断リンク、コミュニティ検出、chunk 監査、グラフ出力
- 出力先は `output/graphs/`、`output/chunks/`、`output/logs/`

### 3. グラフ検索拡張 QA
- `agent` と `noagent` の 2 モードをサポート
- `agent` モードでは質問分解、サブ質問処理、反復検索、推論を実行
- 検索はグラフ、FAISS、chunk 証拠を組み合わせて実施
- グラフ可視化、推論過程表示、検索結果表示に対応

### 4. 評価と実験
- `eval/kg_eval/`：知識グラフ構築品質評価
- `eval/rag_eval/`：QA 品質評価
- `eval/utils/sample_kg_eval_stratified.py`：層化ランダムサンプリング
- gold 生成、候補モデル比較、文書横断レビュー用テンプレート出力、Markdown レポート生成

## 技術ルートと実装の対応

| 研究テーマ | 実装 |
| --- | --- |
| 科学文献知識抽出と意味統合 | `models/constructor/kt_gen.py`、`utils/document_parser.py`、`schemas/` |
| GraphRAG による学際的 QA | `models/retriever/agentic_decomposer.py`、`models/retriever/enhanced_kt_retriever.py`、`backend.py` |
| 評価体系とモデル比較 | `eval/kg_eval/`、`eval/rag_eval/`、`test_kg_eval.py` |

## プロジェクト構成

```text
youtu-graphrag/
├─ backend.py                  # FastAPI バックエンド
├─ main.py                     # CLI エントリポイント
├─ config/
│  ├─ base_config.yaml         # メイン設定
│  └─ config_loader.py         # 設定読み込みとパス正規化
├─ frontend/
│  ├─ index_new.html           # フロントエンド画面
│  ├─ script.js                # UI ロジック
│  └─ style.css                # スタイル
├─ models/
│  ├─ constructor/
│  │  └─ kt_gen.py             # 知識グラフ構築コア
│  └─ retriever/
│     ├─ agentic_decomposer.py # 質問分解
│     ├─ enhanced_kt_retriever.py
│     └─ faiss_filter.py       # FAISS 検索
├─ utils/
│  ├─ document_parser.py       # 文書解析
│  ├─ call_llm_api.py          # LLM 呼び出し
│  ├─ dataset_audit.py         # データセット監査
│  ├─ tree_comm.py             # コミュニティ関連
│  └─ paths.py                 # リポジトリルート基準のパス解決
├─ data/
│  ├─ demo/
│  └─ uploaded/
├─ schemas/
├─ output/
│  ├─ graphs/
│  ├─ chunks/
│  └─ logs/
├─ eval/
│  ├─ kg_eval/
│  ├─ rag_eval/
│  ├─ utils/
│  └─ results/
├─ test_kg_eval.py
└─ test_sample_kg_eval_stratified.py
```

## 動作環境
- Python 3.10 以上
- 仮想環境の利用を推奨
- CPU でも動作可能
- 文書解析互換性向上のため Java ランタイムを推奨

主な依存関係は次のファイルを参照してください。

- `requirements.txt`
- `requirements-server.txt`
- `requirements-optional.txt`

## LLM 環境変数
タスクごとにモデル設定を分離できます。

### 共通デフォルト設定
```env
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_key
```

### 構築と QA を分ける場合
```env
KG_LLM_MODEL=qwen3-max
KG_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
KG_LLM_API_KEY=your_key

RAG_LLM_MODEL=deepseek-chat
RAG_LLM_BASE_URL=https://api.deepseek.com
RAG_LLM_API_KEY=your_key
```

Azure OpenAI を使う場合は次も設定できます。

```env
OPENAI_PROVIDER=azure
API_VERSION=2025-01-01-preview
```

評価モジュールは別の環境ファイルを使います。

- `eval/.env`
- `eval/rag_eval/.env`

## クイックスタート

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

中国語テキストを扱う場合は spaCy の中国語モデルを推奨します。

```bash
python -m spacy download zh_core_web_lg
```

### 2. 環境変数の設定
ルートディレクトリの `.env.example` を参考にしてください。

### 3. Web プロトタイプの起動
```bash
python backend.py
```

起動後に以下へアクセスします。

```text
http://localhost:8000
```

### 4. CLI で構築 / 検索を実行
```bash
python main.py --config config/base_config.yaml --datasets demo
```

特定の処理だけ実行したい場合は `--override` を使います。

```bash
python main.py --datasets demo --override "{\"triggers\": {\"constructor_trigger\": true, \"retrieve_trigger\": false}}"
```

## Web 利用フロー
現在のフロントエンドでは次の流れをサポートしています。

1. 文書をアップロードしてデータセットを作成
2. 必要に応じてカスタム schema をアップロード
3. 知識グラフを構築
4. グラフを可視化
5. データセットを選択して研究 QA を実行
6. 既存データセットを再構築または削除

主要 API は `backend.py` にあります。

- `GET /api/datasets`
- `POST /api/upload`
- `POST /api/construct-graph`
- `POST /api/ask-question`
- `GET /api/graph/{dataset_name}`
- `GET /api/dataset-audit/{dataset_name}`

## 設定
メイン設定ファイルは `config/base_config.yaml` です。現在の研究設定をよく表している主な項目は以下です。

- `active_dataset: demo`
- `construction.mode: agent`
- `nlp.spacy_model: zh_core_web_lg`
- `datasets.demo` は `data/demo/` を参照
- 出力は `output/` 以下に統一

特に注目すべき設定群：

- `construction.*`：構築、chunk 分割、文書横断リンク、並列数
- `retrieval.*`：検索パラメータ、リコール経路、キャッシュ
- `triggers.mode`：`agent` / `noagent`
- `datasets.*`：コーパス、QA セット、schema、グラフ出力位置

## 評価フロー

### 1. 知識グラフ構築評価
設定ファイル：

- `eval/kg_eval/config.yaml`

代表的なコマンド：

```bash
python -m eval.kg_eval.run generate_gold
python -m eval.kg_eval.run run
python -m eval.kg_eval.run cross_doc_review
```

このモジュールでは次を行います。

- gold アノテーション草稿生成
- 候補抽出結果と gold の比較
- グラフ構造と文書横断リンクの品質分析
- 評価レポート生成

### 2. QA 評価
設定ファイル：

- `eval/rag_eval/config.yaml`

代表的なコマンド：

```bash
python -m eval.rag_eval.run
python -m eval.rag_eval.run --dataset AIGC-EDU-test --qa-mode agent
```

このモジュールでは次を行います。

- 質問セットの読み込み
- 現在の GraphRAG パイプラインで回答生成
- 精度、完全性、論理性、説明可能性、学際性などの観点で採点
- 構造化結果と要約レポート生成

## テスト
現在含まれている基本テスト：

```bash
python test_kg_eval.py
python test_sample_kg_eval_stratified.py
```

バックエンドと画面の基本確認だけなら次で十分です。

```bash
python backend.py
```

## 論文に近い理解のしかた
このプロジェクトを一文で表すなら、次のように言えます。

> 学際的な科学文献知識発見のための GraphRAG 実験プラットフォームであり、「学術テキストをグラフ化する」「グラフに基づいて QA する」「構築結果と QA 結果を評価する」という 3 つの問題を扱う。

元の汎用 GraphRAG 紹介と比べて、現在のリポジトリは次をより重視しています。

- 学術文献と学際的知識発見
- 教育分野における大規模言語モデル研究サンプル
- 評価の再現性
- Web プロトタイプと実験ツールの統合

## 関連ドキュメント
- `README.md`：現在の主 README
- `README-CN.md`：中国語版
- `FULLGUIDE-CN.md`：中国語の詳細ガイド
- `FULLGUIDE.md`：英語の詳細ガイド
- `AGENTS.md`：開発エージェント向け作業指針

## 補足
- この README-JA は、現在の研究実装版に合わせて書き直したものです。
- 今後公開や論文化を進める場合は、次の追記を推奨します。
  - データソースの説明
  - モデル選定理由
  - 再現実験表
  - 代表的な QA 事例
  - 論文との対応関係
