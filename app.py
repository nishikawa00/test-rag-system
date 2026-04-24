"""
仮想RAGシステム - Streamlit UI
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.document_loader import Document
from src.text_splitter import TextSplitter
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import RequirementsGenerator

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


@st.cache_resource(show_spinner="埋め込みモデルを読み込み中…")
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


def read_uploaded_files(uploaded_files) -> list[Document]:
    docs = []
    for f in uploaded_files:
        f.seek(0)
        content = f.read().decode("utf-8").strip()
        if content:
            docs.append(Document(
                content=content,
                metadata={"source": f.name, "filename": f.name},
            ))
    return docs


def run_pipeline(
    asis_docs: list[Document],
    rfp_docs: list[Document],
    minutes_docs: list[Document],
    api_key: str,
) -> str:
    bar = st.progress(0, text="STEP 1: ドキュメントを確認中…")

    # ── Vector store ──────────────────────────────────────────────
    bar.progress(15, text="STEP 2: ナレッジベースをインデックス中…")
    embedding_model = get_embedding_model()
    vector_store = VectorStore(dimension=embedding_model.dimension)

    if asis_docs:
        splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(asis_docs)
        embeddings = embedding_model.embed_texts([c.content for c in chunks])
        vector_store.add_documents(chunks, embeddings)

    # ── Retrieval ─────────────────────────────────────────────────
    bar.progress(40, text="STEP 3: 関連情報を検索中…")
    retriever = Retriever(embedding_model, vector_store)

    asis_context = (
        retriever.retrieve_for_sections(RETRIEVAL_QUERIES, k_per_query=3)
        if asis_docs
        else "（AsIs要件定義書は提供されていません）"
    )
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

    # ── Generation ────────────────────────────────────────────────
    bar.progress(60, text="STEP 4: ToBe要件定義書を生成中（Claude API 呼び出し中）…")
    generator = RequirementsGenerator(api_key=api_key)
    result = generator.generate(asis_context, rfp_content, minutes_content)

    bar.progress(100, text="生成完了！")
    return result


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="仮想RAGシステム",
    page_icon="📋",
    layout="wide",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 設定")
    api_key = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        placeholder="sk-ant-...",
        help="https://console.anthropic.com/ で取得したAPIキーを入力してください。",
    )
    st.divider()
    st.markdown("**LLM モデル**")
    st.code("claude-sonnet-4-6", language=None)
    st.markdown("**埋め込みモデル**")
    st.code("paraphrase-multilingual-mpnet-base-v2", language=None)
    st.divider()
    st.caption("RFP ファイルはファイル名に `rfp`（大文字小文字不問）を含めてください。")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("📋 仮想RAGシステム")
st.caption(
    "AsIs要件定義書をナレッジベースとして活用し、"
    "ToBe RFP・議事録から要件定義書を自動生成します。"
)
st.divider()

# ─── File upload ──────────────────────────────────────────────────────────────
col_asis, col_input = st.columns(2)

with col_asis:
    st.subheader("📂 RAGナレッジ")
    st.caption("AsIs要件定義書（複数可）")
    asis_files = st.file_uploader(
        "AsIs要件定義書 (.md)",
        type=["md"],
        accept_multiple_files=True,
        key="asis_upload",
        label_visibility="collapsed",
    )
    if asis_files:
        for f in asis_files:
            st.success(f"✓ {f.name}")
    else:
        st.info("省略可（AsIsなしでも生成できます）")

with col_input:
    st.subheader("📥 インプット")
    st.caption("ToBe RFP・顧客ヒアリング議事録（複数可）")
    input_files = st.file_uploader(
        "ToBe RFP・議事録 (.md)",
        type=["md"],
        accept_multiple_files=True,
        key="input_upload",
        label_visibility="collapsed",
    )
    if input_files:
        rfp_count = sum(1 for f in input_files if "rfp" in f.name.lower())
        min_count = len(input_files) - rfp_count
        st.success(f"✓ RFP: {rfp_count} 件　/　議事録: {min_count} 件")

st.divider()

# ─── Generate button ──────────────────────────────────────────────────────────
can_generate = bool(api_key) and bool(input_files)

btn_col, warn_col = st.columns([1, 3])
with btn_col:
    generate = st.button(
        "🚀 要件定義書を生成",
        type="primary",
        disabled=not can_generate,
        use_container_width=True,
    )
with warn_col:
    if not api_key:
        st.warning("サイドバーに Anthropic API Key を入力してください。")
    elif not input_files:
        st.warning("ToBe RFP または議事録をアップロードしてください。")

# ─── Pipeline execution ───────────────────────────────────────────────────────
if generate:
    asis_docs = read_uploaded_files(asis_files or [])
    all_input = read_uploaded_files(input_files or [])
    rfp_docs = [d for d in all_input if "rfp" in d.metadata["filename"].lower()]
    minutes_docs = [d for d in all_input if d not in rfp_docs]

    try:
        result = run_pipeline(asis_docs, rfp_docs, minutes_docs, api_key)
        st.session_state["result"] = result
        st.session_state["ts"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.success("要件定義書が生成されました！")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# ─── Result display ───────────────────────────────────────────────────────────
if "result" in st.session_state:
    st.divider()
    st.subheader("📄 生成された ToBe 要件定義書")

    dl_col, _ = st.columns([1, 4])
    with dl_col:
        st.download_button(
            label="⬇️ Markdownをダウンロード",
            data=st.session_state["result"].encode("utf-8"),
            file_name=f"tobe_requirements_{st.session_state['ts']}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )

    tab_preview, tab_raw = st.tabs(["📖 プレビュー", "📝 生テキスト（コピー用）"])
    with tab_preview:
        st.markdown(st.session_state["result"])
    with tab_raw:
        st.code(st.session_state["result"], language="markdown")
