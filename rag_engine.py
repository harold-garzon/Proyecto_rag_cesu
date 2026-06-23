# -*- coding: utf-8 -*-
"""
rag_engine.py — Motor RAG reutilizable para la app Flask "Calidoso".
Carga embeddings locales + ChromaDB persistente + Groq LLM UNA SOLA VEZ
y expone responder(pregunta) para usarlo desde Flask.
"""

import os
import sys
import unittest.mock as mock

# Parche VertexAI (igual que en flujo_rag_proyecto.py)
sys.modules['langchain_community.chat_models.vertexai'] = mock.MagicMock()
sys.modules['langchain_community.llms.vertexai'] = mock.MagicMock()

from dotenv import load_dotenv

# ── Configuración ──────────────────────────────────────────────────────────
BASE_DIR = os.environ.get(
    "RAG_BASE_DIR",
    r"C:\Users\Usuario\Desktop\MAESTRÍA\4_SEMESTRE\PROYECTO_RAG"
)
PDF_DIR         = os.path.join(BASE_DIR, "pdf")
PERSIST_DIR     = os.path.join(BASE_DIR, "chroma_db_groq")
COLLECTION_NAME = "mis_programas"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL      = "llama-3.1-8b-instant"

load_dotenv(os.path.join(BASE_DIR, ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY no encontrada en .env")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Mismo prompt que en tu script
PROMPT_TEMPLATE = '''Eres un asesor académico experto en reglamentos y procedimientos de alta calidad de programas e institución de IES colombianas.
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado.
Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:'''

# ── Imports pesados ────────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Singletons (se llenan en init())
_embeddings = None
_vector_store = None
_llm = None
_prompt = None


def _construir_base(embeddings):
    """Solo si NO existe chroma_db_groq: reconstruye desde los PDFs."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    documents_merged = []
    for pdf_file in pdf_files:
        pages = PyPDFLoader(os.path.join(PDF_DIR, pdf_file)).load()
        full_text = "\n".join(p.page_content for p in pages)
        documents_merged.append(
            Document(page_content=full_text, metadata={"source": pdf_file})
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents_merged)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )


def init():
    """Inicializa embeddings, base vectorial, LLM y prompt una sola vez."""
    global _embeddings, _vector_store, _llm, _prompt
    if _vector_store is not None:
        return

    print("[RAG] Cargando modelo de embeddings...")
    _embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Reutiliza la base ya persistida; la reconstruye solo si falta o está vacía.
    necesita_construir = not os.path.exists(PERSIST_DIR)
    if not necesita_construir:
        print("[RAG] Cargando ChromaDB existente...")
        _vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=_embeddings,
            collection_name=COLLECTION_NAME,
        )
        if _vector_store._collection.count() == 0:
            necesita_construir = True
    if necesita_construir:
        print("[RAG] No hay base previa; construyendo desde los PDFs...")
        _vector_store = _construir_base(_embeddings)

    _llm = ChatGroq(model=GROQ_MODEL, temperature=0.0, api_key=GROQ_API_KEY)
    _prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    print(f"[RAG] Listo. Fragmentos indexados: {_vector_store._collection.count()}")


def responder(pregunta: str, k: int = 5) -> dict:
    """Recupera contexto del RAG y genera la respuesta con Groq."""
    if _vector_store is None:
        init()

    retriever = _vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    docs = retriever.invoke(pregunta)

    contexto = "\n\n---\n\n".join(
        f"[Fragmento {i+1} — Pág. {d.metadata.get('page', '?')}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    prompt = _prompt.invoke({"context": contexto, "question": pregunta})
    texto = _llm.invoke(prompt).content

    fuentes = sorted({os.path.basename(d.metadata.get("source", "?")) for d in docs})
    return {"respuesta": texto, "fragmentos": docs, "fuentes": fuentes}