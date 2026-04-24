# プロジェクト概要
社内ドキュメントを対象としたRAG（検索拡張生成）システム。
ユーザーの質問に対して、関連ドキュメントを検索して回答する。

# 技術スタック
- Python 3.11
- FAISS（ベクトル検索）
- Anthropic API（claude-3-5-sonnet、回答生成）
- sentence-transformers（埋め込みベクトル化）
- LangChainは使わない

# コーディングルール
- 変数名・関数名は英語、コメントは日本語OK
- 型ヒント（Type Hints）を必ず書く
- 関数は1つの責務のみ持つ

# ディレクトリ構成
- src/ingestion.py  - ドキュメント読み込み
- src/retriever.py  - 検索ロジック
- src/generator.py  - 回答生成
- data/             - 対象ドキュメント置き場