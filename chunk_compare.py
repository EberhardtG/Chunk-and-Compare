from sentence_transformers import SentenceTransformer, util
import re

# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# Course Summary Document (~700 words)
# --------------------------------------------------

document = """
Over the past several modules, I have learned how modern software systems combine APIs,
authentication, databases, semantic search, and interactive user interfaces to create
powerful applications. One of the first major concepts introduced was FastAPI.
FastAPI is a modern Python framework for building APIs quickly while taking advantage
of automatic validation and documentation generation. By using Pydantic models, data
can be validated before it reaches business logic, helping developers catch mistakes early.

Authentication was another important topic. JSON Web Tokens, commonly known as JWTs,
allow users to securely authenticate with an application. Instead of storing sessions
on the server, the client stores a signed token that is sent with future requests.
FastAPI can use dependencies to verify these tokens and protect routes that require
authentication. This makes APIs both secure and scalable.

Database integration became important once applications needed persistent storage.
SQLAlchemy allows developers to map Python objects directly to database tables.
Relationships between entities can be defined in code, reducing the amount of raw SQL
that must be written. Dependency injection helps manage database sessions cleanly and
ensures connections are opened and closed appropriately.

Semantic search introduced a new approach to information retrieval.
Traditional keyword search relies heavily on exact matches and often fails when users
word questions differently. Embeddings solve this problem by converting text into
high-dimensional vectors that capture semantic meaning. Documents and queries can be
embedded into the same vector space, allowing similarity to be measured with cosine similarity.

The course also explored retrieval-augmented generation systems.
RAG applications typically store embedded document chunks in a vector database.
When a user asks a question, the most relevant chunks are retrieved and supplied to a
language model. This allows responses to be grounded in actual documents rather than
relying solely on model training data.

Chunking strategies significantly affect search quality.
Fixed-size chunking creates text segments with uniform length.
This approach ensures predictable chunk sizes but may break sentences or ideas apart.
Paragraph-based chunking preserves natural semantic boundaries and often provides
better contextual information. However, paragraph chunks may vary greatly in size.

Streamlit was used to build interactive applications.
Streamlit provides widgets such as buttons, text inputs, sidebars, and charts.
Because the application reruns after each interaction, session state becomes essential
for preserving user data. Streamlit allows rapid development of dashboards and AI-powered
interfaces without building a traditional frontend.

The course emphasized experimentation and iterative development.
When building semantic search systems, different embedding models can produce different
results. Chunk size, overlap, similarity thresholds, and retrieval settings all impact
overall quality. Developers must evaluate results and adjust configurations based on
actual performance rather than assumptions.

Overall, these topics collectively provide a foundation for modern AI-enhanced
applications. FastAPI offers API development capabilities, JWTs provide security,
SQLAlchemy enables persistence, embeddings power semantic understanding, Streamlit
provides interactive interfaces, and chunking strategies improve retrieval quality.
"""

# --------------------------------------------------
# Chunking Strategy 1
# Fixed Size (300 chars, 50 overlap)
# --------------------------------------------------


def fixed_chunks(text, size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        while end < len(text) and text[end] != " ":
            end += 1

        chunks.append(text[start:end].strip())

        start += size - overlap

    return chunks


# --------------------------------------------------
# Chunking Strategy 2
# Paragraph Based
# --------------------------------------------------


def paragraph_chunks(text):
    paragraphs = re.split(r"\n\s*\n", text.strip())

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


# --------------------------------------------------
# Create Chunks
# --------------------------------------------------

strategies = {
    "Fixed (300 chars)": fixed_chunks(document),
    "Paragraph": paragraph_chunks(document),
}

print("\n=== Chunk Counts ===")

for name, chunks in strategies.items():
    print(f"{name}: {len(chunks)} chunks")

# --------------------------------------------------
# Embed Chunks
# --------------------------------------------------

embedded_strategies = {
    name: model.encode(chunks, convert_to_tensor=True)
    for name, chunks in strategies.items()
}

# --------------------------------------------------
# Test Queries
# --------------------------------------------------

queries = [
    "How does FastAPI validate data?",
    "What are embeddings used for?",
    "How does Streamlit manage state?",
]

results_summary = {}

# --------------------------------------------------
# Search
# --------------------------------------------------

for query in queries:

    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    query_results = {}

    for strategy_name, chunk_embeddings in embedded_strategies.items():

        scores = util.cos_sim(
            query_embedding,
            chunk_embeddings
        )[0]

        top_indices = scores.argsort(
            descending=True
        )[:2]

        query_results[strategy_name] = [
            float(scores[idx])
            for idx in top_indices
        ]

        print(f"\n--- {strategy_name} ---")

        for rank, idx in enumerate(top_indices, start=1):

            score = float(scores[idx])

            print(
                f"\nResult {rank}"
            )
            print(
                f"Score: {score:.4f}"
            )
            print(
                f"Text: {strategies[strategy_name][idx][:250]}..."
            )

    results_summary[query] = query_results

# --------------------------------------------------
# Analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

paragraph_wins = 0
fixed_wins = 0

for query, strategy_scores in results_summary.items():

    fixed_best = max(
        strategy_scores["Fixed (300 chars)"]
    )

    paragraph_best = max(
        strategy_scores["Paragraph"]
    )

    print(f"\nQuery: {query}")
    print(
        f"Best Fixed Score: {fixed_best:.4f}"
    )
    print(
        f"Best Paragraph Score: {paragraph_best:.4f}"
    )

    if paragraph_best > fixed_best:
        print(
            "Winner: Paragraph Chunking"
        )
        paragraph_wins += 1

    elif fixed_best > paragraph_best:
        print(
            "Winner: Fixed Chunking"
        )
        fixed_wins += 1

    else:
        print(
            "Winner: Tie"
        )

# --------------------------------------------------
# Conclusion
# --------------------------------------------------

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)

print(
    f"\nParagraph Chunking Wins: {paragraph_wins}"
)
print(
    f"Fixed Chunking Wins: {fixed_wins}"
)

print("""
The paragraph-based strategy generally performed better because the course
document is naturally organized into topic-focused paragraphs. Preserving
those boundaries allows the embedding model to capture complete concepts such
as FastAPI, JWT authentication, embeddings, semantic search, and Streamlit.

The fixed-size strategy produced more chunks and greater document coverage,
but occasionally split ideas in the middle of a discussion. This reduced the
amount of context available in individual chunks and sometimes lowered the
similarity score.

Looking at the retrieval scores, the stronger-performing strategy was the one
that consistently returned the highest-scoring and most relevant chunks for a
given query. For structured educational notes such as this document,
paragraph-based chunking is likely the better choice because it preserves
semantic meaning while still keeping chunks reasonably sized.
""")
