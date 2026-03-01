import streamlit as st
from pathlib import Path


VECTOR_SIZE = 384
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
DB_PATH = ".qdrant_db"

# configure session state
if "pipeline_ingest" not in st.session_state:
    st.session_state.pipeline_ingest = None
if "pipeline_retrieval" not in st.session_state:
    st.session_state.pipeline_retrieval = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# landing
st.title("Local RAG System!")
with st.sidebar:
    col_name = st.text_input("Qdrant collection", "rag_docs")
    embed_model = st.text_input("FastEmbed dense", "BAAI/bge-small-en-v1.5")
    llm_model = st.text_input("Ollama model", "qwen3:8b")
    top_k = st.slider("Top-K", 3, 10, 5)


# chached loading pipeline
@st.cache_resource
def init_pipelines(col_name=col_name, llm_model=llm_model):
    from src.parsing.docling import DoclingParser
    from src.chunking.langchain import LangChainChunker
    from src.embedding.fastembed import FastEmbedEmbedder
    from src.database.qdrant import QdrantVectorStore
    from src.llm.ollama import OllamaProvider
    from src.pipeline.ingestion import IngestionPipeline
    from src.pipeline.retrieval import RetrievalPipeline

    parser = DoclingParser(extract_images=False, extract_tables=True)
    chunker = LangChainChunker(chunk_size=1000, chunk_overlap=200)
    embedder = FastEmbedEmbedder()
    vectorstore = QdrantVectorStore(col_name, vector_size=384, path="./qdrant_db")

    ingest = IngestionPipeline(parser, chunker, embedder, vectorstore)
    llm = OllamaProvider(llm_model)
    retrieval = RetrievalPipeline(embedder, vectorstore, llm, top_k)
    return ingest, retrieval


if not st.session_state.initialized:
    st.session_state.pipeline_ingest, st.session_state.pipeline_retrieval = (
        init_pipelines()
    )
    st.session_state.initialized = True
    st.success("Pipelines ready!")


# sidebar with ability to input sources
st.sidebar.header("Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Files (.pdf)", accept_multiple_files=True, type="pdf"
)
if st.sidebar.button("Index") and uploaded_files:
    with st.spinner("Preprocessing..."):
        for file in uploaded_files:
            path = Path(f"/tmp/{file.name}")
            path.write_bytes(file.read())
            chunks = st.session_state.pipeline_ingest.process_file(path)
            st.sidebar.success(f"Add {chunks} chunks from {file.name}")
    st.rerun()

# chat
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Input your answer here!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown("prompt")
    with st.chat_message("assistant"):
        with st.spinner("RAG preprocessing..."):
            result = st.session_state.pipeline_retrieval.query(prompt)
            response = result.answer
            st.markdown(response)
            sources = "\n".join(
                [f"- {doc.metadata.get('source', 'N/A')}" for doc in result.source[:3]]
            )
            with st.expander("Sources"):
                for i, doc in enumerate(result.source_chunks[:3], 1):
                    source = doc.metadata.get("source", "N/A")
                    st.markdown(f"**{i}. {source}**\n\n{doc.page_content[:400]}...")

    st.session_state.messages.append({"role": "assistant", "content": response})
