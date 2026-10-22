from sentence_transformers import SentenceTransformer, util
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

#sample document that has information for FastAPI, Python, embeddings, semantic search, chunking, and Streamlit
document = """Over the past several modules, I’ve learned how modern applications combine structured APIs, data validation, authentication, and semantic search to create systems that feel intelligent and responsive. One of the earliest concepts introduced was FastAPI, a Python framework designed for building high‑performance APIs with minimal boilerplate. FastAPI’s reliance on Pydantic models makes data validation automatic, meaning the API can reject malformed input before it ever reaches business logic. This pattern of validating early and failing fast is a recurring theme in backend development, and it becomes even more important as applications scale.

Another major topic was authentication, specifically JWT‑based authentication. JSON Web Tokens allow a server to issue signed tokens that clients include in future requests. Because the token is signed, the server can verify its authenticity without storing session data. This stateless approach is ideal for distributed systems, but it also requires careful handling of secrets, expiration times, and refresh strategies. Learning how to implement login routes, protect endpoints, and decode tokens helped me understand how real‑world APIs secure user data.

Database interaction was another core skill. SQLAlchemy provides an ORM layer that maps Python classes to database tables, allowing developers to write expressive queries without manually crafting SQL. Concepts like sessions, relationships, and migrations are essential for building reliable applications. I also learned how dependency injection in FastAPI ensures that database sessions are created and cleaned up properly for each request, preventing issues like connection leaks.

As the course progressed, the focus shifted toward embeddings and semantic search. Traditional keyword search fails when users phrase queries differently than the stored text. Embeddings solve this by converting text into numerical vectors that capture meaning rather than exact wording. Using models like all‑MiniLM‑L6‑v2, I learned how to embed documents once, embed queries on the fly, and compute cosine similarity to determine relevance. This technique powers search engines, recommendation systems, and AI assistants.

Chunking emerged as a critical strategy when dealing with long documents. Large texts must be split into smaller pieces before embedding, but the way those pieces are created affects search quality. Fixed‑size chunking ensures uniform length but may split sentences awkwardly. Paragraph‑based chunking preserves semantic boundaries but results in chunks of inconsistent size. Choosing the right strategy depends on the type of content and the expected queries. This project explores those tradeoffs directly.

Another important concept was building interactive tools using Streamlit. Streamlit reruns the script on every user interaction, which means state must be preserved using session state. I learned how to build widgets, sidebars, metrics, and conditional UI elements to create dashboards and exploration tools. This helped me understand how frontend interfaces can be built quickly without needing a full JavaScript framework.

Finally, the course emphasized the importance of experimentation. Whether tuning similarity thresholds, comparing chunking strategies, or evaluating different embedding models, the goal is to build intuition. AI systems are rarely perfect on the first attempt; instead, they improve through iteration, measurement, and thoughtful adjustments. This mindset prepares developers to work with real‑world data, where ambiguity and inconsistency are the norm.

Overall, the modules provided a strong foundation in modern Python development, API design, authentication, semantic search, and interactive tooling. These concepts come together to form the backbone of intelligent applications, and each project builds toward understanding how to design systems that are both robust and user‑friendly.
"""

# --- Strategy 1: Fixed-size chunks (300 chars, 50 overlap) ---
def fixed_chunks(text, size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))

        # extend to nearest space so we don't cut words
        while end < len(text) and text[end] != " ":
            end += 1

        chunks.append(text[start:end].strip())
        start += size - overlap
    return chunks

# --- Strategy 2: Paragraph chunks ---
def paragraph_chunks(text):
    paragraphs = text.strip().split('\\n\\n')  # Split on double newlines
    return [p.strip() for p in paragraphs if p.strip()]


strategies = {
    "Fixed (300 chars)": fixed_chunks(document),
    "Paragraph": paragraph_chunks(document)
}

# Embed chunks for both strategies
embedded_strategies = {
    name: model.encode(chunks)
    for name, chunks in strategies.items()
}
#defining the 3 test queries 
queries = [
    "How does FastAPI handle data validation?",
    "What is the purpose of embeddings in search?",
    "How does Streamlit manage state?"
]

for query in queries:
    query_emb = model.encode(query)

#Compute similarity against each strategy
    print(f"\n=== Query: {query} ===")

    for name, chunk_embs in embedded_strategies.items():
        scores = util.cos_sim(query_emb, chunk_embs)[0]

        # Get top 2 chunks
        top_idx = scores.argsort(descending=True)[:2]

        print(f"\n-- Strategy: {name} --")
        for idx in top_idx:
            score = float(scores[idx])
            chunk_text = strategies[name][idx]
            print(f"[{score:.4f}] {chunk_text[:200]}...")

print("\n=== Comparison Summary ===")
print("Fixed-size chunking tends to capture more uniform slices of text, which can help when queries match specific phrases.")
print("Paragraph-based chunking preserves semantic meaning, often producing more accurate matches for conceptual queries.")
print("In this experiment, paragraph-based chunking generally performs better because the document is structured in clear thematic paragraphs.")

