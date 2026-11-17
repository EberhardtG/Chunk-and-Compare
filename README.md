# 📚 Chunking Strategy Comparison for Semantic Search

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Sentence Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-orange)
![Semantic Search](https://img.shields.io/badge/Search-Semantic-green)
![License](https://img.shields.io/badge/License-MIT-green)

A practical demonstration of how different document chunking strategies impact semantic search performance using embeddings and cosine similarity.

---

# 🎯 Project Goal

When building Retrieval-Augmented Generation (RAG) systems, documents must be broken into smaller chunks before they can be embedded and searched effectively.

This project compares two common chunking strategies:

- Fixed-size chunking
- Paragraph-based chunking

The script embeds both sets of chunks using Sentence Transformers and evaluates which approach retrieves more relevant content for several real-world queries.

---

# 🚀 Features

✅ Fixed-size chunking (300 characters)

✅ 50-character overlap

✅ Paragraph-based chunking

✅ Sentence Transformer embeddings

✅ Cosine similarity search

✅ Top 2 results returned per query

✅ Side-by-side ranking comparison

✅ Retrieval score analysis

✅ Data-driven chunking evaluation

---

# 🧩 Chunking Strategies

## Fixed-Size Chunking

The document is divided into chunks of:

```text
300 characters
```

with:

```text
50 characters overlap
```

between chunks.

### Advantages

- Consistent chunk sizes
- Predictable vector storage requirements
- Works well for unstructured text

### Drawbacks

- May split important concepts
- Context can be fragmented

---

## Paragraph-Based Chunking

The document is split using natural paragraph boundaries:

```python
re.split(r"\n\s*\n", text.strip())
```

Each parag*aph becomes an independent chunk.
*### Advantages

-*Preserves semantic meaning
- Maint*ins complete*thoughts
- Often improves*similarity scores

### Drawbacks

* Une*en chunk sizes
- Large*paragraphs can create oversized ch*nks

---

* ⚙️ Workflow

```text
Document*    │
    ├── Fixed*Chunking*    │       │
    │      *└── Chunk Set A
    │
*   └── Paragraph Chunking
        *   │*            └── Chunk Set B

     *              │
                  * ▼

         Sentence Transformer
*         (all-MiniLM*L6-v2)

                    │
    *               ▼

            Vect*r Embeddings

                    **                    ▼

           *  Query Search

                  **│
                    ▼

         *Cosine Similarity Ranking

       *            │
                    *

           Top 2 Results Returne**```

---

# 🧠 Emb*dding Model

The project uses:

``*text
all-MiniLM*L6-v2
```

from the Sentence Trans*ormers library.

### Model*Properties*
| Property | Value |
|-----------*-----------|
| Dimensions*| 384 |
| Framework*| Sentence Transformers |
| Simila**ty Metric | Cosine Similarity |
| *se*Cases | Search* Retrieval, Clustering |

This*model balances speed and retrieval*quality, making it a*popular choice for R*G systems.

---

#*🔍 Test Queries

The script evalua*es*three different queries:

```text*How does FastAPI validate data?

W*at are*embeddings used for?

How does Str*amlit manage state*
```

For each query, the program:*
*. Creates a*query embedding
2. Com*ares it against every chunk
3. Cal*ulates*cosine similarity scores
4. Ranks *he chunks*5. Returns the top 2 matches*
---

# 📊 Example*Output

```text
QUERY:
*ow does FastAPI validate data?

--* Fixed (*00 chars) ---

Result 1
Score* 0.8124

Result 2*Score: 0.744*

--- Paragraph ---

Result 1
*core: 0.9037

Result *
Score: 0.8422
*``

---

# 📈 Analysis*
The project now*performs score*based analysis using actual*retrieval results.

For each query* the script records*

```text
Best Fixed Chunk*Score
Best Paragraph Chunk Score
W*nning*Strategy
```

Example:

*``text
Query:
How does FastAPI val*date*data?

Best Fixed Score:
*.8124

Best Paragraph Score:
0*9037

Winner:
Paragraph Chunking
*``

This provides objective*evidence when comparing retrieval *uality.

---

* 🏆 Findings

*or structured educational content,*paragraph chunk*ng generally performs better becau*e:

- Individual*paragraphs focus on one*topic
- Semantic boundaries remain*intact
- More*context is preserved
- Similar*ty scores tend to be higher

Fixed*size chunk*ng remains useful when:

- Working*with OCR text
- Processing*transcripts
- Handling poorly form*tted*documents
- En*orcing strict chunk-size limits

-*-

* 📁 Project Structure

```*ext
chunk_compare/
│
├── chunk*compare.py
├── README.md
*── requirements.txt
```

---

#*📦 Installation

Install required*packages:

```bash
pip install sen*ence-transformers*torch numpy
```

---

#*▶️ Run

```bash
python chunk_compa*e*py
```

---

# 🎓 Learning Outcome*

This project demonstrates:

* Document*chunking strategies
- Semantic sea*ch fundamentals*- Embedding generation
- Cosine si*ilarity*ranking
- Retrieval evaluation
- R*G pipeline design
- Search quality*analysis

*t provides a practical example*of how chunk*ng decisions directly affect retri*val performance*and why*chunk selection is one*of the most important design choic*s in Retrieval*Augmented Generation systems.

---*
# 📄 License*
This project is licensed*under the MIT License and*is intended for educational and po*tfolio*use.
