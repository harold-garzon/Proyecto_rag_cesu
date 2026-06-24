# -*- coding: utf-8 -*-
"""
RAG Retrieval Augmented Generation (Groq) — adaptado para ejecución local
Pipeline: PDF → Chunking → Embeddings (local) → ChromaDB → Groq LLM → RAGAs
"""

import os
import sys
import unittest.mock as mock

# ── Directorio base del proyecto ──────────────────────────────────────────────
BASE_DIR = r"C:\Users\Usuario\Desktop\MAESTRÍA\4_SEMESTRE\PROYECTO_RAG"
os.chdir(BASE_DIR)

# ── Parche para bug de VertexAI en ragas ─────────────────────────────────────
sys.modules['langchain_community.chat_models.vertexai'] = mock.MagicMock()
sys.modules['langchain_community.llms.vertexai'] = mock.MagicMock()
print("[OK] Parche VertexAI aplicado")

# ── Cargar variables de entorno ───────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY no encontrada en .env")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
print(f"[OK] API Key cargada (...{GROQ_API_KEY[-6:]})")

# ── Verificar paquetes instalados ─────────────────────────────────────────────
import importlib.metadata

paquetes = [
    "langchain-core",
    "langchain-community",
    "langchain-huggingface",
    "langchain-groq",
    "langchain-chroma",
    "ragas",
    "sentence-transformers",
    "chromadb",
]

for paquete in paquetes:
    try:
        version = importlib.metadata.version(paquete)
        print(f"  {paquete:<30} {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {paquete:<30} ❌ no instalado")


# =============================================================================
# PASO 1 — Carga de Documentos PDF
# =============================================================================

from langchain_community.document_loaders import PyPDFLoader

pdf_dir = os.path.join(BASE_DIR, "pdf")
pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])

print(f"PDFs disponibles en \"{pdf_dir}\" ({len(pdf_files)} archivos):")
for i, f in enumerate(pdf_files, 1):
    size_kb = os.path.getsize(os.path.join(pdf_dir, f)) // 1024
    print(f"  [{i}] {f}  ({size_kb} KB)")

from langchain_core.documents import Document

pages_by_file = {}
for pdf_file in pdf_files:
    path = os.path.join(pdf_dir, pdf_file)
    loader = PyPDFLoader(path)
    pages = loader.load()
    pages_by_file[pdf_file] = pages
    print(f"  {pdf_file}: {len(pages)} páginas cargadas")

# Combinar todas las páginas de cada PDF en un único Document por archivo.
# Esto permite que los chunks crucen límites de página, evitando que un
# artículo quede partido entre el final de una página y el inicio de la siguiente.
documents_merged = []
for pdf_file, pages in pages_by_file.items():
    full_text = "\n".join(p.page_content for p in pages)
    documents_merged.append(
        Document(page_content=full_text, metadata={"source": pdf_file})
    )

print(f"\nTotal de páginas cargadas: {sum(len(p) for p in pages_by_file.values())}")
print(f"Documentos combinados:     {len(documents_merged)}")

doc_ejemplo = list(pages_by_file.values())[0][0]
print("\nMetadatos del primer documento:")
for k, v in doc_ejemplo.metadata.items():
    print(f"  {k}: {v}")
print(f"\nPrimeros 500 caracteres:")
print("-" * 60)
print(doc_ejemplo.page_content[:500])


# =============================================================================
# PASO 2 — División en Fragmentos (Chunking)
# =============================================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", " "]
)

chunks = text_splitter.split_documents(documents_merged)

total_paginas = sum(len(p) for p in pages_by_file.values())
print(f"\nDocumentos originales (páginas): {total_paginas}")
print(f"Fragmentos generados:            {len(chunks)}")
print(f"Factor de expansión:             {len(chunks)/total_paginas:.1f}x")
print(f"\nEjemplo — Fragmento #20:")
print(f"  Fuente: {chunks[20].metadata.get('source', '?')}")
print(f"  Página: {chunks[20].metadata.get('page', '?')}")
print(f"  Longitud: {len(chunks[20].page_content)} caracteres")
print(f"  Contenido: {chunks[20].page_content}")


# =============================================================================
# PASO 3 — Embeddings 
# =============================================================================


from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print(f"\n[OK] Embeddings locales cargados: {EMBEDDING_MODEL}")

reglamento = [
    "Loa programas académicos sona quellos que cumplen condiciones de calidad y pertinencia.",
    "Las acreditaciones de alta calidad se otorgan a programas que demuestran excelencia en su gestión y resultados.",
]

corpus_embeddings = embeddings.embed_documents(reglamento)
query = "FACTORES DE CALIDAD DE LOS PROGRAMAS ACADÉMICOS"
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], corpus_embeddings)
indices_ordenados = np.argsort(scores[0])[::-1]

print(f"\nConsulta: \"{query}\"\n")
for idx in indices_ordenados:
    barra = "█" * int(scores[0][idx] * 20)
    print(f"  {scores[0][idx]:.4f} {barra}")
    print(f"  {reglamento[idx]}\n")


embeddings_model = embeddings

sample_text = chunks[20].page_content
sample_vector = embeddings_model.embed_query(sample_text)

print("\nEmbedding del fragmento #20:")
print(f"  Dimensiones del vector: {len(sample_vector)}")
print(f"  Rango de valores:       [{min(sample_vector):.4f},  {max(sample_vector):.4f}]")
print(f"  Primeros 8 valores:     {[round(v, 4) for v in sample_vector[:8]]}")


# =============================================================================
# PASO 4 — Almacenamiento en ChromaDB
# =============================================================================

import shutil
from langchain_chroma import Chroma

PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_groq")

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"\nBase de datos anterior eliminada: {PERSIST_DIR}")

print(f"Indexando {len(chunks)} fragmentos en ChromaDB...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=PERSIST_DIR,
    collection_name="mis_programas",
    collection_metadata={"hnsw:space": "cosine"}
)

total = vector_store._collection.count()
print(f"\n[OK] Base vectorial creada!")
print(f"  Ubicación:             {PERSIST_DIR}")
print(f"  Fragmentos indexados:  {total}")
print(f"  Modelo embeddings:     {EMBEDDING_MODEL} (local, CPU)")
print(f"  Dimensión de vectores: 384")

coleccion = vector_store._collection
print(f"\nColección ChromaDB:")
print(f"  Nombre:              {coleccion.name}")
print(f"  Total documentos:    {coleccion.count()}")
print(f"  Metadata:            {coleccion.metadata}")


# =============================================================================
# PASO 5 — Recuperación Vectorial (Retrieval)
# =============================================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

pregunta = "cuales son los objetivos de la acreditación en alta calidad"

print(f"\nConsulta: \"{pregunta}\"")
print(f"Recuperando los 10 fragmentos más relevantes...\n")

documentos_recuperados = retriever.invoke(pregunta)

print(f"Fragmentos recuperados: {len(documentos_recuperados)}")
print("=" * 60)
for i, doc in enumerate(documentos_recuperados, 1):
    fuente = os.path.basename(doc.metadata.get("source", "desconocido"))
    pagina = doc.metadata.get("page", "?")
    print(f"\n[{i}] {fuente} — Pág. {pagina} ({len(doc.page_content)} chars)")
    print(f"    {doc.page_content[:250]}...")


# =============================================================================
# PASO 6 — Construcción del Prompt Aumentado
# =============================================================================

from langchain_core.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = '''Eres un asesor académico experto en reglamentos y procedimientos de alta calidad de programas e institución de IES colombianas.
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado.
Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en la base de conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:'''

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

contexto = "\n\n---\n\n".join(
    f"[Fragmento {i+1} — Pág. {doc.metadata.get('page', '?')}]\n{doc.page_content}"
    for i, doc in enumerate(documentos_recuperados)
)

prompt_aumentado = prompt_template.invoke({
    "context": contexto,
    "question": pregunta
})

print("\nPrompt aumentado construido correctamente.")
print(f"  Fragmentos en el contexto:       {len(documentos_recuperados)}")
print(f"  Caracteres totales del contexto: {len(contexto)}")
print(f"  Tokens aproximados del contexto: ~{len(contexto)//4}")


# =============================================================================
# PASO 7 — Generación con el LLM (Groq)
# =============================================================================

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

GROQ_MODEL = "llama-3.1-8b-instant"

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.0,
    api_key=GROQ_API_KEY
)

print(f"\n[OK] LLM configurado: {GROQ_MODEL}")
print(f"Enviando prompt a Groq...")

respuesta = llm.invoke(prompt_aumentado)
texto_respuesta = respuesta.content

print("=" * 60)
print("RESPUESTA DEL LLM (RAG):")
print("=" * 60)
print(texto_respuesta)

# Comparación RAG vs sin contexto
pregunta_test = "¿Qué estudiantes están matriculados en el curso?"

docs_test = retriever.invoke(pregunta_test)
contexto_test = "\n\n---\n\n".join(
    f"[Fragmento {i+1}]\n{d.page_content}" for i, d in enumerate(docs_test)
)
prompt_con_rag = prompt_template.invoke({"context": contexto_test, "question": pregunta_test})
texto_con_rag = llm.invoke(prompt_con_rag).content
texto_sin_rag = llm.invoke([HumanMessage(content=pregunta_test)]).content

print(f"\nPregunta: \"{pregunta_test}\"")
print("\n[CON RAG — basado en el documento]")
print(texto_con_rag)
print("\n[SIN RAG — solo conocimiento del LLM]")
print(texto_sin_rag)


# =============================================================================
# Pipeline RAG reutilizable
# =============================================================================

def rag_pipeline(pregunta: str, k: int = 5, verbose: bool = False) -> dict:
    retriever_k = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    docs = retriever_k.invoke(pregunta)

    contexto = "\n\n---\n\n".join(
        f"[Fragmento {i+1} — Pág. {d.metadata.get('page','?')}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )

    prompt = prompt_template.invoke({"context": contexto, "question": pregunta})
    texto = llm.invoke(prompt).content

    if verbose:
        print(f"Fragmentos recuperados: {len(docs)}")
        for i, d in enumerate(docs, 1):
            fuente = os.path.basename(d.metadata.get("source", "?"))
            pagina = d.metadata.get("page", "?")
            print(f"  [{i}] {fuente} Pág.{pagina}: {d.page_content[:80]}...")

    return {
        "pregunta": pregunta,
        "fragmentos": docs,
        "contexto": contexto,
        "respuesta": texto,
        "tokens_contexto_aprox": len(contexto) // 4
    }

print("\nFunción rag_pipeline() lista.")

import time
preguntas_prueba = [
    "¿que es un programa académico acreditable?",
    "¿Cuales son las politicas que las ies deben demostrar en el proceso de autoevaluación?",
    "¿cuantos factores de calidad hay?",
    "¿cuales son los actores que evalua?"
]

print("\nPRUEBA DEL PIPELINE RAG CON MÚLTIPLES PREGUNTAS")
print("=" * 65)

for pregunta in preguntas_prueba:
    print(f"\n? {pregunta}")
    resultado = rag_pipeline(pregunta, 10)
    print(f"-> {resultado['respuesta']}")
    print(f"   (~{resultado['tokens_contexto_aprox']} tokens | {len(resultado['fragmentos'])} fragmentos)")
    print("-" * 65)
    time.sleep(10)


# =============================================================================
# PASO 8 — Evaluación con RAGAs
# =============================================================================

from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import pandas as pd

print("\n[OK] RAGas importado correctamente")

muestras_evaluacion = [
    {
        "user_input": "Qué es un vehículo de servicio diplomático o consular?",
        "reference":  "Vehículo automotor destinado al servicio de funcionarios diplomáticos o consulares"
    },
    {
        "user_input": "¿en los pasos de nivel que deben colocar las entidades ferroviarias?",
        "reference":  "señales, barreras y luces en los pasos de nivel"
    },
    {
        "user_input": "Que es la acreditación en alta calidad?",
        "reference":  "La acreditación de alta calidad es un reconocimiento otorgado a programas académicos que cumplen con estándares de excelencia en su gestión y resultados, demostrando calidad y pertinencia en la educación ofrecida."
    },
]

registros = []
print("Ejecutando pipeline RAG para cada muestra...\n")

for muestra in muestras_evaluacion:
    pregunta = muestra["user_input"]
    resultado = rag_pipeline(pregunta, k=3)
    registros.append({
        "user_input":         pregunta,
        "retrieved_contexts": [doc.page_content for doc in resultado["fragmentos"]],
        "response":           resultado["respuesta"],
        "reference":          muestra["reference"]
    })
    print(f"[OK] {pregunta[:55]}")
    print(f"     -> {resultado['respuesta'][:90]}...\n")

print(f"Dataset de evaluación listo: {len(registros)} muestras")

llm_juez        = LangchainLLMWrapper(llm)
embeddings_juez = LangchainEmbeddingsWrapper(embeddings_model)

dataset = EvaluationDataset.from_list(registros)

print("Evaluando con RAGAs (Groq como juez)...\n")

resultados = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
    llm=llm_juez,
    embeddings=embeddings_juez
)

df = resultados.to_pandas()

print("=" * 65)
print("RESULTADOS DE EVALUACIÓN RAGAs")
print("=" * 65)

cols_score = ["faithfulness", "answer_relevancy", "context_precision"]
cols_mostrar = ["user_input"] + [c for c in cols_score if c in df.columns]
print(df[cols_mostrar].to_string(index=False))

print("\nPromedios globales:")
for col in cols_score:
    if col in df.columns:
        barra = "█" * int(df[col].mean() * 20)
        print(f"  {col:<25} {df[col].mean():.4f}  {barra}")
