#### *100+ LLM Questions and Answers Every AI Engineer Needs to Know*

---

# 6. Advanced Search Algorithms

### 6.1 What are architecture patterns for information retrieval & semantic search?

**Key Architecture Patterns:**

1. **Traditional IR Architecture:**
   - **Inverted Indexing:** Documents are tokenized and normalized into an inverted index for fast term-based lookups.
   - **Query Processing & Ranking:** Uses algorithms like TF-IDF or BM25 to score and rank results based on term frequency and document relevance.

2. **Semantic (Neural) Search Architecture:**
   - **Embedding Generation:** Leverages deep models (e.g., BERT, Sentence Transformers) to convert text into dense vector representations.
   - **Vector Indexing & Retrieval:** Utilizes Approximate Nearest Neighbor (ANN) algorithms (e.g., FAISS, HNSW, Annoy) to perform efficient similarity searches in high-dimensional spaces.

3. **Hybrid Search Architecture:**
   - **Combined Lexical and Semantic Retrieval:** Merges traditional keyword matching with vector similarity to cover both exact matches and conceptual relevance.
   - **Multi-Stage Pipeline:** Often starts with a broad candidate retrieval using an inverted index, followed by a semantic re-ranking phase using neural models.

4. **Distributed & Microservices Architecture:**
   - **Scalability & Modularity:** Each component (indexing, embedding, query processing, ranking) can be deployed as an independent service, allowing scalability and easier maintenance.
   - **Real-Time Updates:** Incorporates streaming data pipelines (using tools like Apache Kafka or Flink) for dynamic, up-to-date indexing.

These patterns are selected based on the trade-offs between latency, scalability, and retrieval quality, enabling systems to efficiently handle both traditional term-based queries and more complex semantic queries.

---

### 6.2 Why it’s important to have very good search

---

### 6.3 How can you achieve efficient and accurate search results in large-scale datasets?

---

### 6.4 Consider a scenario where a client has already built a RAG-based system that is not giving accurate results, upon investigation you find out that the retrieval system is not accurate, what steps you will take to improve it?

---

### 6.5 Explain the keyword-based retrieval method

---

### 6.6 How to fine-tune re-ranking models?

---

### 6.7 Explain most common metric used in information retrieval and when it fails?

---

### 6.8 If you were to create an algorithm for a Quora-like question-answering system, with the objective of ensuring users find the most pertinent answers as quickly as possible, which evaluation metric would you choose to assess the effectiveness of your system?

---

### 6.9 I have a recommendation system, which metric should I use to evaluate the system?

---

### 6.10 Compare different information retrieval metrics and which one to use when?

---

### 6.11 How does hybrid search works?

---

### 6.12 If you have search results from multiple methods, how would you merge and homogenize the rankings into a single result set?

---

### 6.13 How to handle multi-hop/multifaceted queries?

---

### 6.14 What are different techniques to be used to improved retrieval?

---
