# chunk_compare.py

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![Sentence Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-orange) ![License](https://img.shields.io/badge/License-MIT-green)

> A hands-on comparison of two text chunking strategies — **fixed-size** and **paragraph-based** — using semantic embeddings and cosine similarity to evaluate retrieval quality.

---

## 📌 Project Overview

`chunk_compare.py` is a Python script that demonstrates how different text chunking strategies affect the quality of semantic search results. It splits a source document using two distinct strategies, encodes all chunks into vector embeddings with a lightweight sentence transformer model, and then runs a set of test queries — surfacing the top-2 most similar chunks for each strategy side-by-side.

This project is ideal for anyone exploring **Retrieval-Augmented Generation (RAG)** pipelines, vector databases, or information retrieval systems who wants to make an informed choice about chunking before scaling up.

---

## 🎯 Objectives

- Implement and compare two practical chunking strategies on the same source text
- Generate semantic embeddings using a compact, high-quality model
- Rank chunk relevance against multiple queries using cosine similarity
- Provide a clear, reproducible framework for evaluating chunking trade-offs

---

## 🧩 Chunking Strategies

### 1. Fixed-Size Chunking
Splits the document into chunks of exactly **300 characters** with a **50-character overlap** between consecutive chunks.

| Parameter | Value |
|-----------|-------|
| Chunk size | 300 characters |
| Overlap | 50 characters |

**How it works:** A sliding window moves through the raw text. Each chunk starts 250 characters after the previous one begins, ensuring context is never abruptly cut off at a boundary.

### 2. Paragraph-Based Chunking
Splits the document at **natural paragraph boundaries** (double newlines `\n\n`), preserving the logical structure of the source text.

**How it works:** The text is split on paragraph delimiters, and empty segments are discarded. Each resulting chunk corresponds to one coherent paragraph of the original document.

---

## ⚙️ How the Script Works

```
Source Text
    │
    ├──► Fixed-Size Chunker (300 chars / 50 overlap)
    │         └──► Chunk List A
    │
    └──► Paragraph Chunker (split on \n\n)
              └──► Chunk List B
                       │
              ┌────────┴────────┐
              │                 │
    Encode with all-MiniLM-L6-v2 (Sentence Transformers)
              │                 │
    Embeddings A        Embeddings B
              │                 │
         ┌────┴──────────────────┴────┐
         │     For Each Query (×3)    │
         │  Cosine Similarity Ranking │
         │    Return Top-2 Chunks     │
         └────────────────────────────┘
```

1. **Load** the source document into memory
2. **Chunk** the text using both strategies independently
3. **Encode** all chunks into dense vector embeddings via `sentence-transformers`
4. **Query** — for each of 3 test queries, compute cosine similarity against every chunk embedding
5. **Rank & Display** the top-2 most relevant chunks from each strategy

---

## 🚀 How to Run

### Prerequisites

```bash
pip install sentence-transformers scikit-learn numpy
```

### Run the Script

```bash
python chunk_compare.py
```

No arguments required — the source text and queries are defined within the script.

---

## 📋 Example Output Structure

```
========================================
Query: "What is the impact of climate change on agriculture?"
========================================

--- Fixed-Size Chunking (Top 2) ---

  [Rank 1] Similarity: 0.847
  "...rising temperatures and unpredictable rainfall patterns
  are placing increasing pressure on crop yields worldwide,
  particularly in sub-Saharan Africa where..."

  [Rank 2] Similarity: 0.791
  "...farmers have begun adopting drought-resistant seed
  varieties as a direct response to the shifting growing
  seasons caused by warming trends..."

--- Paragraph-Based Chunking (Top 2) ---

  [Rank 1] Similarity: 0.883
  "Rising temperatures and unpredictable rainfall patterns are
  placing increasing pressure on crop yields worldwide. Regions
  like sub-Saharan Africa face the steepest challenges, with
  projections suggesting output could fall by up to 20% by 2050."

  [Rank 2] Similarity: 0.812
  "Governments and NGOs have responded by funding research into
  drought-resistant seed varieties and precision irrigation,
  helping farmers adapt to shifting growing seasons."

========================================
```

---

## ⚖️ Strategy Comparison

| Dimension | Fixed-Size (300 / 50) | Paragraph-Based |
|---|---|---|
| **Boundary awareness** | None — cuts mid-sentence | Full — respects natural breaks |
| **Chunk consistency** | Uniform length | Variable length |
| **Context preservation** | Partial (via overlap) | High |
| **Similarity scores** | Slightly lower (fragmented context) | Generally higher |
| **Best for…** | Uniform, densely packed text; PDFs; OCR output | Well-structured prose; articles; documentation |
| **Risk** | Splitting key concepts across chunks | Very long paragraphs create oversized chunks |

### When to Use Fixed-Size Chunking
- Your source text **lacks consistent structure** (e.g., raw OCR, transcripts, scraped web content)
- You need **predictable memory usage** in a vector store with chunk-size limits
- The document is **one continuous block** with no natural paragraph breaks
- You are working with **multilingual or code-heavy** content where paragraph semantics are unreliable

### When to Use Paragraph-Based Chunking
- Your source text is **well-written prose** — articles, reports, documentation, books
- **Semantic completeness per chunk** is a priority (each chunk should express one idea)
- You are feeding chunks directly into an **LLM prompt** and need coherent context windows
- Your retrieval metric is **cosine similarity** — paragraph chunks typically score higher because they are topically unified

---

## 🗂️ Project Structure

```
chunk_compare.py       # Main script
README.md              # This file
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Model | `all-MiniLM-L6-v2` |
| Provider | Sentence Transformers (Hugging Face) |
| Embedding dimensions | 384 |
| Max input tokens | 256 |
| Use case | Semantic similarity, clustering, retrieval |

The model is downloaded automatically on first run via the `sentence-transformers` library.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
