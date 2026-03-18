# A6: Contextual Retrieval vs Naive RAG

A comprehensive implementation and comparison of two Retrieval-Augmented Generation (RAG) approaches on Chapter 8: Transformers from the Stanford NLP textbook (Jurafsky & Martin, Draft January 2026).

## 📋 Project Overview

This project implements and evaluates two RAG systems:

1. **Naive RAG**: Standard RAG pipeline with direct chunk embedding
2. **Contextual Retrieval**: Enhanced RAG with contextual enrichment before embedding

The goal is to compare their performance on answering questions about transformer architecture using ROUGE metrics.

## 🎯 Assignment Details

- **Chapter**: 8 - Transformers
- **Source Document**: `8.pdf` (Stanford NLP Textbook — Jurafsky & Martin, Draft Jan 6, 2026)
- **LLM Provider**: OpenAI (GPT-4o-mini)
- **Embedding Model**: OpenAI text-embedding-3-small
- **Vector Database**: FAISS (IndexFlatL2)
- **Evaluation Metric**: ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- **Web Framework**: Chainlit

## 🏗️ Project Structure

```
A6/
├── 8.pdf                                      # Source: Chapter 8 (Transformers)
├── README.md                                  # This file
├── requirements.txt                           # Python dependencies
│
├── 01-document-processing.ipynb              # Phase 1: Text extraction & QA generation
├── 02-naive-rag-implementation.ipynb         # Phase 2: Naive RAG pipeline
├── 03-contextual-retrieval-implementation.ipynb # Phase 3: Contextual RAG pipeline
├── 04-evaluation.ipynb                       # Phase 4: Evaluation & comparison
│
├── data/                                      # Generated data files
│   ├── chapter8_text.txt                     # Full chapter text (71,757 characters)
│   ├── chapter8_chunks.json                  # Naive RAG chunks (170 chunks)
│   ├── chapter8_contextualized_chunks.json   # Contextual chunks
│   └── chapter8_qa_pairs.json                # 20 QA pairs
│
├── vectorstore/                               # FAISS indexes
│   ├── naive_rag/
│   │   └── index.faiss                       # Naive RAG vector store
│   └── contextual_retrieval/
│       └── index.faiss                       # Contextual vector store
│
├── results/                                   # Evaluation results
│   ├── naive_rag_responses.json              # Naive RAG answers
│   ├── contextual_retrieval_responses.json   # Contextual answers
│   ├── evaluation_summary.json               # Performance summary
│   ├── rouge_comparison.png                  # Visualization
│   └── improvement_distribution.png          # Distribution plot
│
├── answer/                                    # Submission
│   └── response-st125988-chapter-8.json      # Final submission file
│
└── app/                                       # Chainlit web application
    ├── app.py                                # Main Chainlit app (single file)
    ├── requirements.txt                      # App dependencies
    ├── .env.example                          # API key template
    └── README.md                             # App documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Jupyter Notebook

### Installation

1. **Clone or navigate to the project directory**

```bash
cd /path/to/A6
```

2. **Create a virtual environment and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# OR
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

3. **Set up OpenAI API key**

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

### Running the Notebooks

Execute the notebooks **in order**:

```bash
source .venv/bin/activate
jupyter notebook
```

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01-document-processing.ipynb` | Extract text, create chunks, generate QA pairs |
| 2 | `02-naive-rag-implementation.ipynb` | Build Naive RAG pipeline |
| 3 | `03-contextual-retrieval-implementation.ipynb` | Build Contextual RAG pipeline |
| 4 | `04-evaluation.ipynb` | Evaluate and compare results |

## 📓 Notebook Guide

### 01-document-processing.ipynb

**Purpose**: Prepare data for RAG systems

**Tasks**:
- Load `8.pdf` using `PyMuPDFLoader` — **27 pages** extracted
- Extract and save full chapter text — **71,757 characters** saved to `data/chapter8_text.txt`
- Create chunks using `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50) — **170 chunks** produced
- Generate 20 QA pairs using GPT-4 and save to `data/chapter8_qa_pairs.json`

**Outputs**:
- `data/chapter8_text.txt`
- `data/chapter8_chunks.json` (170 chunks)
- `data/chapter8_qa_pairs.json` (20 QA pairs)

**Cost**: ~$0.05–0.10 (GPT-4 for QA generation)

---

### 02-naive-rag-implementation.ipynb

**Purpose**: Implement standard RAG pipeline

**Pipeline**:
1. Load 170 chunks from Phase 1
2. Create embeddings for each chunk (`text-embedding-3-small`, 1536-dim)
3. Build FAISS vector store (`IndexFlatL2`)
4. Implement retrieval function (top-3 chunks by L2 similarity)
5. Implement generation function (`gpt-4o-mini`, temp=0, max_tokens=500)
6. Evaluate on all 20 QA pairs

**Outputs**:
- `vectorstore/naive_rag/index.faiss`
- `results/naive_rag_responses.json`

**Cost**: ~$0.05–0.10 (embeddings + generation)

---

### 03-contextual-retrieval-implementation.ipynb

**Purpose**: Implement Contextual Retrieval with chunk enrichment

**Pipeline**:
1. Load 170 chunks from Phase 1
2. **Enrich each chunk** with a 1–2 sentence contextual prefix using `gpt-4o-mini` (async/parallel)
3. Create embeddings for the enriched chunks
4. Build a separate FAISS vector store
5. Implement retrieval & generation (identical settings to Naive RAG)
6. Evaluate on all 20 QA pairs

**Key Difference**: Each chunk is prepended with context such as:
> *"This chunk from Chapter 8 discusses [explanation]."*
> 
> [Original chunk text]

**Outputs**:
- `data/chapter8_contextualized_chunks.json`
- `vectorstore/contextual_retrieval/index.faiss`
- `results/contextual_retrieval_responses.json`

**Cost**: ~$0.10–0.15 (contextual enrichment + embeddings + generation)

---

### 04-evaluation.ipynb

**Purpose**: Compare both approaches using ROUGE metrics

**Analysis**:
1. Load results from both RAG systems (20 pairs each)
2. Calculate ROUGE-1, ROUGE-2, ROUGE-L scores (F1, with Porter stemmer)
3. Generate comparison tables and visualizations
4. Identify top improvements and degradations per question
5. Create submission JSON file

**Outputs**:
- `results/evaluation_summary.json`
- `results/rouge_comparison.png`
- `results/improvement_distribution.png`
- `answer/response-st125988-chapter-8.json`

**Cost**: Free (local computation only)

## 🤖 Model Specifications

### Retriever

| Component | Specification |
|-----------|---------------|
| **Embedding Model** | OpenAI `text-embedding-3-small` |
| **Dimension** | 1536 |
| **Vector Database** | FAISS (`IndexFlatL2`) |
| **Similarity Metric** | L2 distance → similarity score |
| **Top-k** | 3 chunks |

### Generator

| Component | Specification |
|-----------|---------------|
| **LLM** | OpenAI `gpt-4o-mini` |
| **Temperature** | 0 (deterministic) |
| **Max Tokens** | 500 |
| **System Role** | Answer questions about Chapter 8: Transformers |
| **Context Window** | 128k tokens |

### Contextual Enrichment

| Component | Specification |
|-----------|---------------|
| **LLM** | OpenAI `gpt-4o-mini` |
| **Temperature** | 0 |
| **Max Tokens** | 150 |
| **Purpose** | Add contextual prefixes to chunks |
| **Processing** | Async (parallel API calls via `asyncio`) |

## 💰 Cost Estimation

| Task | Cost |
|------|------|
| QA Pair Generation (GPT-4) | $0.05–0.10 |
| Contextual Enrichment (GPT-4o-mini) | $0.10–0.15 |
| Embeddings (text-embedding-3-small) | $0.01 |
| Answer Generation (GPT-4o-mini) | $0.05–0.10 |
| **Total** | **$0.20–0.36** |

Web app usage (per 100 queries): ~$0.02–0.04

## 📊 Results

Results were generated by running `04-evaluation.ipynb` on all 20 QA pairs.

### ROUGE Score Comparison

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|---------|---------|---------|
| Naive RAG | **0.4471** | **0.2839** | **0.3902** |
| Contextual Retrieval | 0.4102 | 0.2372 | 0.3376 |
| **Change** | **−8.25%** | **−16.46%** | **−13.48%** |

> **Overall finding**: Naive RAG outperformed Contextual Retrieval by **8.25%** on ROUGE-1 for this chapter.

### Per-Question Breakdown

| Outcome | Count |
|---------|-------|
| Contextual Retrieval performed **better** | 7 / 20 |
| Contextual Retrieval performed **worse** | 13 / 20 |
| No change | 0 / 20 |

**Average ROUGE-1 delta**: −0.0369 (Contextual − Naive)

### Notable Cases

**Top improvements (Contextual > Naive):**

| Question | Naive ROUGE-1 | Contextual ROUGE-1 | Δ |
|----------|--------------|-------------------|---|
| What is the standard architecture for building large language models? | 0.3284 | **1.0000** | +0.6716 |
| What does the transformer's embedding matrix E do? | 0.3210 | 0.6207 | +0.2997 |
| What are the components of the language modeling head? | 0.2927 | 0.3939 | +0.1013 |

**Top degradations (Naive > Contextual):**

| Question | Naive ROUGE-1 | Contextual ROUGE-1 | Δ |
|----------|--------------|-------------------|---|
| What does multi-head attention layer also known as? | **0.8333** | 0.2632 | −0.5702 |
| What concept distinguishes transformers from feedforward layers? | **0.9032** | 0.3571 | −0.5461 |
| What happens after the N transformer blocks? | **0.8182** | 0.3778 | −0.4404 |

### Discussion

While Contextual Retrieval is designed to improve retrieval quality by prepending context to chunks, the results for Chapter 8 show that Naive RAG produced higher ROUGE scores on average. A likely reason is that Chapter 8 is a **focused, single-topic chapter** on transformers — the plain chunks already carry sufficient semantic signal for the embedding model, and the added contextual prefix may introduce noise or slight paraphrasing that shifts answers away from the ground-truth phrasing used in the evaluation. Contextual Retrieval tends to benefit more on broad, multi-domain corpora where chunks lack standalone context.

Contextual Retrieval did show clear wins on definitional questions (e.g., the exact transformer definition), where the added framing helped the model recover the precise phrasing. For short, factual questions with near-verbatim ground-truth answers, the added verbosity from enrichment can hurt ROUGE recall.

## 🌐 Web Application

A Chainlit-based web interface for interactive question answering about Chapter 8.

### Features

- Interactive chat interface
- Real-time answers powered by **Contextual Retrieval**
- Source citations with chunk text, page numbers, and similarity scores
- Clean, user-friendly design

### Running the App

```bash
cd app

source ../.venv/bin/activate    # macOS/Linux
# OR
.venv\Scripts\activate          # Windows

export OPENAI_API_KEY='your-key-here'

chainlit run app.py -w
```

Open your browser to `http://localhost:8000`

## 🖥️ App Demo

The Chainlit web application provides a conversational interface to query Chapter 8 content using Contextual Retrieval.

### Demo

![Demo](demo.gif)

### App Interaction Flow

```
User Question
     │
     ▼
Query Embedding (text-embedding-3-small)
     │
     ▼
FAISS Vector Search (Contextual Index, top-3 chunks)
     │
     ▼
GPT-4o-mini Generation (with retrieved context)
     │
     ▼
Answer + Source Citations displayed in chat
```

### Key App Behaviors

- **Source display**: Each response cites the top retrieved chunk(s), including page number and similarity score
- **Deterministic answers**: Temperature=0 ensures reproducible outputs
- **Context-aware**: Retrieval uses the contextually enriched vector store for improved recall on broad questions

## 📝 Key Implementation Details

### Chunking Strategy

- **Method**: `RecursiveCharacterTextSplitter` (LangChain)
- **Chunk Size**: 500 characters
- **Overlap**: 50 characters
- **Result**: 170 chunks from 27 pages (71,757 characters)
- **Rationale**: Balances context preservation with retrieval precision

### Contextual Enrichment

Each chunk is enriched with a 1–2 sentence contextual prefix generated by `gpt-4o-mini`:

```
"This chunk from Chapter 8 (Transformers) discusses [explanation]."

[Original chunk text]
```

Enrichment runs asynchronously via `asyncio`, significantly reducing total processing time.

### ROUGE Evaluation

- **Metrics**: ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence)
- **Score Type**: F1-measure
- **Stemming**: Enabled (Porter stemmer via `rouge_score` library)
- **Pairs evaluated**: 20

## 🔧 Troubleshooting

**`ModuleNotFoundError: No module named 'openai'`** — Activate the virtual environment and run `pip install -r requirements.txt`.

**`AuthenticationError: Invalid API key`** — Run `export OPENAI_API_KEY='your-actual-key-here'`.

**`FileNotFoundError: data/chapter8_chunks.json`** — Run notebooks in order: 01 → 02 → 03 → 04.

**FAISS index errors** — Delete existing stores and rerun notebooks 02 and 03:
```bash
rm -rf vectorstore/naive_rag/index.faiss
rm -rf vectorstore/contextual_retrieval/index.faiss
```

**Jupyter kernel crashes** — Increase available memory or reduce batch size for embedding creation.

## 📚 References

- **Source Chapter**: `8.pdf` — Jurafsky & Martin, *Speech and Language Processing*, Draft January 6, 2026
- **Contextual Retrieval**: https://www.anthropic.com/engineering/contextual-retrieval
- **LangChain Documentation**: https://python.langchain.com/
- **OpenAI API**: https://platform.openai.com/docs
- **FAISS**: https://faiss.ai/
- **Chainlit**: https://docs.chainlit.io/

## 🎓 Learning Outcomes

This project demonstrates:

1. **Document Processing**: PDF extraction (27 pages → 170 chunks), text chunking, QA generation
2. **Embedding Techniques**: Converting text to dense 1536-dim vectors with `text-embedding-3-small`
3. **Vector Databases**: FAISS `IndexFlatL2` for efficient similarity search
4. **RAG Architectures**: Naive vs. Contextual retrieval — trade-offs in single-topic vs. broad corpora
5. **Async Programming**: Parallel API calls for efficient contextual enrichment
6. **Evaluation Metrics**: ROUGE-1/2/L for text similarity assessment
7. **Web Development**: Building interactive NLP applications with Chainlit
8. **Cost Optimization**: Balancing model quality and API costs (~$0.20–0.36 total)

## 👥 Author

**Student ID**: st125988

**Course**: AT82.03 — Artificial Intelligence: Natural Language Understanding

**Assignment**: A6 — Contextual Retrieval vs Naive RAG

**Chapter**: 8 — Transformers
