# Calidoso — Asistente RAG sobre Acreditación en Alta Calidad (CNA)

## 1. Información del estudiante

**Nombre:** Harold Duvan Garzón González  
**Fecha:** 22/06/2026

---

## 2. Documento seleccionado y justificación

**Documento:** Acuerdo 01 de 2025 del Consejo Nacional de Educación Superior (CESU)

**Justificación:** Este acuerdo establece los lineamientos del Consejo Nacional de Acreditación (CNA) para los procesos de acreditación en alta calidad de programas académicos, instituciones de educación superior (IES) y unidades académicas en Colombia. Su elección se justifica porque concentra en un solo instrumento normativo todos los criterios, factores, características y procedimientos que rigen la acreditación, tanto por primera vez como para renovación, convirtiéndolo en la fuente primaria de consulta para cualquier actor involucrado en este proceso.

---

## 3. Persona usuaria objetivo y caso de uso

### Público objetivo

Esta aplicación está dirigida a actores institucionales vinculados a procesos de **solicitud por primera vez o renovación de la acreditación en alta calidad**, ya sea de programas académicos o de instituciones de educación superior como un todo.

Los perfiles de usuario típicos incluyen: coordinadores de autoevaluación, directivos académicos, líderes de unidades de aseguramiento de la calidad y equipos de acreditación institucional.

### Caso de uso ilustrativo

Una IES que inicia por primera vez su proceso de acreditación institucional necesita comprender cuáles son los criterios y condiciones que debe cumplir antes de radicar su solicitud ante el CNA. El sistema permite resolver preguntas como:

- ¿Cuál es el porcentaje mínimo de programas acreditados para que una institución sea acreditable?
- ¿Cuáles son los factores de calidad que se evalúan a nivel institucional?
- ¿Qué etapas comprende el proceso, desde la autoevaluación hasta la respuesta del Ministerio de Educación Nacional (MEN)?

El asistente responde con base **exclusiva** en el texto del Acuerdo 01 de 2025, citando las fuentes documentales en cada respuesta, lo que garantiza trazabilidad normativa y evita interpretaciones no fundamentadas.

---

## 4. Flujo RAG del sistema

El sistema implementa un pipeline de **Retrieval-Augmented Generation (RAG)** que combina recuperación semántica de documentos con generación de lenguaje mediante un LLM. El flujo completo se ejecuta en `flujo_rag_groq.py` y el motor reutilizable en `rag_engine.py`.

### Diagrama del pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        INDEXACIÓN (offline)                     │
│                                                                 │
│   PDF  ──►  Carga de páginas  ──►  Chunking  ──►  Embeddings   │
│             (PyPDFLoader)          (1500 chars     (HuggingFace │
│                                    overlap 100)     local CPU)  │
│                                         │                       │
│                                         ▼                       │
│                                    ChromaDB                     │
│                                 (persistida en disco)           │
└─────────────────────────────────────────────────────────────────┘
                                         │
                                         │  (ya indexado)
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONSULTA (online / por pregunta)           │
│                                                                 │
│  Pregunta  ──►  Embedding  ──►  Retrieval  ──►  Top-K chunks   │
│  del usuario    de la query     (similitud       (k=5 por       │
│                                  coseno)          defecto)      │
│                                         │                       │
│                                         ▼                       │
│                               Prompt aumentado                  │
│                          (contexto + pregunta)                  │
│                                         │                       │
│                                         ▼                       │
│                              Groq LLM (llama-3.1-8b-instant)   │
│                                         │                       │
│                                         ▼                       │
│                          Respuesta + fuentes citadas            │
└─────────────────────────────────────────────────────────────────┘
```

### Descripción de cada paso

**Paso 1 — Carga del documento PDF**

`PyPDFLoader` (LangChain Community) lee el archivo `acuerdo_01_2025_cesu.pdf` página a página. Todas las páginas de un mismo archivo se concatenan en un único `Document` para evitar que artículos queden partidos entre el final de una página y el inicio de la siguiente.

**Paso 2 — División en fragmentos (chunking)**

`RecursiveCharacterTextSplitter` divide el texto en fragmentos de **1 500 caracteres** con un solapamiento de **100 caracteres** entre fragmentos consecutivos. Los separadores se aplican en orden de preferencia: párrafo (`\n\n`), salto de línea (`\n`) y espacio. Esto preserva la coherencia semántica dentro de cada chunk.

**Paso 3 — Generación de embeddings**

El modelo `paraphrase-multilingual-MiniLM-L12-v2` de HuggingFace convierte cada fragmento en un vector de **384 dimensiones**. El modelo se ejecuta localmente en CPU, por lo que no requiere API externa ni incurre en costos adicionales. Los vectores se normalizan para que la similitud coseno sea el criterio de comparación.

**Paso 4 — Almacenamiento en ChromaDB**

Los vectores y sus metadatos (fuente, número de página) se persisten en la carpeta `chroma_db_groq/` usando ChromaDB con espacio métrico coseno (`hnsw:space: cosine`). La base se crea una sola vez; en ejecuciones posteriores se reutiliza directamente.

**Paso 5 — Recuperación vectorial (retrieval)**

Ante cada pregunta del usuario, el sistema genera su embedding con el mismo modelo y busca en ChromaDB los **k = 5 fragmentos** más similares mediante búsqueda por similitud coseno. Los fragmentos recuperados forman el contexto factual de la respuesta.

**Paso 6 — Construcción del prompt aumentado**

Los fragmentos recuperados se concatenan como contexto dentro de una plantilla de prompt que instruye al LLM a responder **únicamente** con la información del documento. Si la respuesta no está en el contexto, el sistema lo indica explícitamente en lugar de generar contenido inventado.

```
Eres un asesor académico experto en reglamentos y procedimientos
de alta calidad de programas e institución de IES colombianas.
Responde la pregunta usando ÚNICAMENTE la información del contexto
proporcionado. Si la respuesta no está en el contexto, indica
exactamente: "No encontré información sobre esto en la base de
conocimientos."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}
```

**Paso 7 — Generación con Groq LLM**

El prompt aumentado se envía al modelo `llama-3.1-8b-instant` a través de la API de Groq con `temperature=0.0` para obtener respuestas deterministas y sin variación creativa. La respuesta generada, junto con las fuentes citadas, se retorna al usuario.

**Paso 8 — Capa social y enrutamiento en la app web**

`app.py` aplica una capa de filtrado previa al RAG: detecta saludos, despedidas y preguntas sobre identidad del bot mediante expresiones regulares, y responde con mensajes predefinidos. Cualquier consulta que no coincida con estos patrones sociales se enruta directamente al motor RAG (`rag_engine.responder()`).

---

## 5. Preguntas y respuestas generadas por el sistema

### Pregunta 1 — ¿Qué es un programa académico acreditable?

**Respuesta del sistema:**

Según el Acuerdo 01 de 2025, un programa académico acreditable es aquel que cuenta con registro calificado vigente para ser ofrecido y desarrollado, y que tiene al menos cuatro (4) cohortes de graduados y ocho (8) años de funcionamiento, periodo durante el cual la institución debe haber reportado estudiantes matriculados en cualquier periodo académico de cada año.

Son acreditables los programas técnicos profesionales, tecnológicos, profesionales universitarios, de especialidad médico-quirúrgica, de maestría y de doctorado.

---

### Pregunta 2 — ¿Cuántos factores de calidad tienen los programas académicos?

**Respuesta del sistema:**

Los programas académicos cuentan con **12 factores de calidad**:

1. Objetivos y propósitos del programa
2. Comunidad académica del programa
3. Desarrollo profesional y académico de la comunidad académica
4. Gestión de la calidad del programa
5. Planificación y organización del programa
6. Recursos humanos del programa
7. Recursos físicos y tecnológicos del programa
8. Recursos financieros del programa
9. Bienestar de la comunidad académica del programa
10. Recursos físicos, tecnológicos, medios educativos y ambientes de aprendizaje
11. Organización, administración y financiación del programa académico
12. Aseguramiento de la alta calidad del programa

---

### Pregunta 3 — ¿Cuáles son los factores de calidad de las instituciones?

**Respuesta del sistema:**

Conforme al artículo 26 del Acuerdo 01 de 2025, los factores de calidad institucional son:

1. Identidad institucional
2. Gobierno institucional, organización y transparencia
3. Desarrollo ambientalmente sostenible y gestión institucional
4. Sistema interno de aseguramiento de la calidad
5. Estructura y procesos académicos
6. Aportes de la investigación, la innovación, el desarrollo tecnológico, la creación e investigación-creación artística y cultural al entorno
7. Extensión, proyección e impacto social
8. Visibilidad regional, nacional e internacional
9. Bienestar institucional
10. Comunidad de profesores
11. Comunidad de estudiantes
12. Comunidad de egresados

Cada factor se evalúa a través de características que expresan referentes universales de alta calidad y las particularidades propias de cada institución.

---

### Pregunta 4 — ¿Qué es una institución multicampus?

**Respuesta del sistema:**

Una institución multicampus es aquella que ofrece programas académicos en diferentes campus del territorio colombiano y funciona de manera integrada. Para ser reconocida como tal, debe cumplir cuatro condiciones simultáneas:

1. Operar bajo un único nombre o razón social
2. Compartir un único propósito institucional expresado en una misión común
3. Contar con un único sistema de gobierno institucional
4. Regirse por un mismo conjunto de políticas y normas internas (estatutos, reglamentos y sistema interno de aseguramiento de la calidad)

---

### Pregunta 5 — ¿Qué es una institución de educación superior acreditable?

**Respuesta del sistema:**

Una institución de educación superior es acreditable cuando, manteniendo su carácter académico o tipología por más de cinco (5) años, tiene acreditados al menos el **treinta por ciento (30 %)** de sus programas académicos acreditables.

---

## 5. Instrucciones para ejecutar el sistema

### Requisitos previos

- Python 3.10 o superior
- Cuenta activa en [Groq Cloud](https://console.groq.com) para obtener una API key gratuita
- Conexión a internet (solo en la primera ejecución, para descargar el modelo de embeddings)

### Paso 1 — Clonar o descargar el proyecto

Ubique la carpeta del proyecto en su equipo. La estructura esperada es:

```
PROYECTO_RAG/
├── app.py
├── rag_engine.py
├── flujo_rag_groq.py
├── requirements.txt
├── .env
├── pdf/
│   └── acuerdo_01_2025_cesu.pdf
├── chroma_db_groq/        ← se crea automáticamente
├── templates/
└── static/
```

### Paso 2 — Crear y activar el entorno virtual

```bash
# Crear el entorno
python -m venv .venv

# Activar en Windows
.venv\Scripts\activate

# Activar en Linux / macOS
source .venv/bin/activate
```

### Paso 3 — Instalar dependencias

```bash
pip install -r requirements.txt
```

> La instalación puede tardar varios minutos la primera vez, ya que incluye paquetes como `chromadb`, `sentence-transformers`, `langchain` y `scipy`.

### Paso 4 — Configurar la API key de Groq

Edite el archivo `.env` en la raíz del proyecto y reemplace el valor con su propia clave:

```
GROQ_API_KEY=su_clave_aqui
```

Puede obtener una API key gratuita en [console.groq.com](https://console.groq.com) → *API Keys* → *Create API Key*.

### Paso 5 — (Opcional) Reconstruir la base vectorial

Este paso solo es necesario si la carpeta `chroma_db_groq/` no existe o si desea reprocesar el PDF desde cero. Al ejecutar `app.py`, el motor RAG reconstruye la base automáticamente si no la encuentra; sin embargo, si desea ejecutar el pipeline completo con trazas de consola:

```bash
python flujo_rag_groq.py
```

Este script realiza todo el flujo: carga del PDF, chunking, generación de embeddings, indexación en ChromaDB, recuperación y generación de respuestas de prueba con Groq.

### Paso 6 — Iniciar la aplicación web

```bash
python app.py
```

Al arrancar, el sistema carga el modelo de embeddings y conecta con ChromaDB. Una vez listo, verá en consola:

```
[RAG] Cargando modelo de embeddings...
[RAG] Cargando ChromaDB existente...
[RAG] Listo. Fragmentos indexados: <N>
 * Running on http://127.0.0.1:5000
```

Abra su navegador en `http://localhost:5000` para interactuar con **Calidoso**.

### Resumen del orden de ejecución

| Paso | Comando | Cuándo ejecutarlo |
|------|---------|-------------------|
| Instalar dependencias | `pip install -r requirements.txt` | Una sola vez |
| Pipeline completo (opcional) | `python flujo_rag_groq.py` | Si `chroma_db_groq/` no existe |
| Aplicación web | `python app.py` | Cada vez que use el sistema |

### Tecnologías utilizadas

| Componente | Tecnología |
|------------|------------|
| Framework web | Flask 3.0 |
| LLM | Groq — `llama-3.1-8b-instant` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (local, CPU) |
| Base vectorial | ChromaDB (persistida en disco) |
| Orquestación RAG | LangChain |
| Carga de PDF | PyPDF (LangChain Community) |
