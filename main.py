"""
仮想RAGシステム - ToBe与信照会システム要件定義書 自動生成
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_ASIS_DIR = "data/asis"
DATA_INPUT_DIR = "data/input"
DATA_OUTPUT_DIR = "data/output"

# RAG検索クエリ（出力セクションごとの検索キーワード）
RETRIEVAL_QUERIES = [
    "与信照会システムの機能要件 照会処理 申請",
    "非機能要件 可用性 性能 レスポンスタイム セキュリティ",
    "業務フロー 与信照会 審査 承認 通知",
    "システムスコープ 対象業務 システム概要",
    "外部インターフェース 外部連携 API",
    "データ管理 データベース マスタ",
    "運用保守 バックアップ 監視",
    "セキュリティ 認証 アクセス制御 暗号化",
]


def main() -> None:
    print("=" * 55)
    print("  仮想RAGシステム - ToBe要件定義書 自動生成")
    print("=" * 55)
    print()

    # API key check
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[ERROR] ANTHROPIC_API_KEY が設定されていません。")
        print("        .env ファイルに ANTHROPIC_API_KEY=sk-... を記載してください。")
        sys.exit(1)

    # Ensure output directory exists
    Path(DATA_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── 1. Load documents ─────────────────────────────────────────
    print("STEP 1: ドキュメントを読み込み中...")
    from src.document_loader import DocumentLoader

    loader = DocumentLoader()

    asis_docs = loader.load_directory(DATA_ASIS_DIR)
    if asis_docs:
        print(f"  AsIs要件定義書: {len(asis_docs)} 件")
    else:
        print(f"  [警告] {DATA_ASIS_DIR}/ に .md ファイルが見つかりません。")
        print("         AsISなしでも生成は続行します。")

    input_docs = loader.load_directory(DATA_INPUT_DIR)
    if not input_docs:
        print(f"\n[ERROR] {DATA_INPUT_DIR}/ に .md ファイルが見つかりません。")
        print("        ToBe RFP と議事録を配置してください。")
        sys.exit(1)

    rfp_docs = [
        d for d in input_docs
        if "rfp" in d.metadata["filename"].lower()
    ]
    minutes_docs = [d for d in input_docs if d not in rfp_docs]
    print(f"  RFP: {len(rfp_docs)} 件  /  議事録: {len(minutes_docs)} 件")

    # ── 2. Build vector store ─────────────────────────────────────
    print("\nSTEP 2: ナレッジベースをインデックス中...")
    from src.embeddings import EmbeddingModel
    from src.vector_store import VectorStore
    from src.text_splitter import TextSplitter

    embedding_model = EmbeddingModel()
    vector_store = VectorStore(dimension=embedding_model.dimension)

    if asis_docs:
        splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(asis_docs)
        print(f"  チャンク数: {len(chunks)}")
        embeddings = embedding_model.embed_texts([c.content for c in chunks])
        vector_store.add_documents(chunks, embeddings)

    # ── 3. Retrieve relevant AsIs context ─────────────────────────
    print("\nSTEP 3: 関連情報を検索中...")
    from src.retriever import Retriever

    retriever = Retriever(embedding_model, vector_store)

    if asis_docs:
        asis_context = retriever.retrieve_for_sections(
            RETRIEVAL_QUERIES, k_per_query=3
        )
        print(f"  取得チャンク数: {asis_context.count('【出典:')}")
    else:
        asis_context = "（AsIs要件定義書は提供されていません）"

    rfp_content = (
        "\n\n".join(d.content for d in rfp_docs)
        if rfp_docs
        else "（RFPは提供されていません）"
    )
    minutes_content = (
        "\n\n".join(d.content for d in minutes_docs)
        if minutes_docs
        else "（議事録は提供されていません）"
    )

    # ── 4. Generate requirements ──────────────────────────────────
    print("\nSTEP 4: ToBe要件定義書を生成中...")
    print("  (Claude API を呼び出しています。しばらくお待ちください)")
    from src.generator import RequirementsGenerator

    generator = RequirementsGenerator(api_key=api_key)
    result = generator.generate(asis_context, rfp_content, minutes_content)

    # ── 5. Save output ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(DATA_OUTPUT_DIR) / f"tobe_requirements_{timestamp}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\n{'=' * 55}")
    print(f"  生成完了!")
    print(f"  出力ファイル: {output_path}")
    print(f"{'=' * 55}\n")
    print("--- 冒頭プレビュー (500文字) ---")
    print(result[:500])
    print("...\n")


if __name__ == "__main__":
    main()
