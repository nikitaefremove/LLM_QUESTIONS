#### *100+ LLM Questions and Answers Every AI Engineer Needs to Know*

---

# 15. Prompt Hacking

## Table of Contents

- [15.1 How to optimize cost of overall LLM System?](#151-how-to-optimize-cost-of-overall-llm-system)
- [15.2 What are mixture of expert models (MoE)?](#152-what-are-mixture-of-expert-models-moe)  
- [15.3 How to build production grade RAG system, explain each component in detail?](#153-how-to-build-production-grade-rag-system-explain-each-component-in-detail)  
- [15.4 What is FP8 variable and what are its advantages of it?](#154-what-is-fp8-variable-and-what-are-its-advantages-of-it)  
- [15.5 How to train LLM with low precision training without compromising on accuracy?](#155-how-to-train-llm-with-low-precision-training-without-compromising-on-accuracy)  
- [15.6 How to calculate size of KV cache?](#156-how-to-calculate-size-of-kv-cache)  
- [15.7 Explain dimension of each layer in multi headed transformation attention block](#157-explain-dimension-of-each-layer-in-multi-headed-transformation-attention-block)  
- [15.8 How do you make sure that attention layer focuses on the right part of the input?](#158-how-do-you-make-sure-that-attention-layer-focuses-on-the-right-part-of-the-input)  

---

### 15.1 How to optimize cost of overall LLM System?

1. **Model Pruning**: Remove less important weights or neurons to reduce model size and computational cost.
2. **Quantization**: Convert high-precision weights to lower precision (e.g., FP32 to INT8) to reduce memory and computation requirements.
3. **Distillation**: Train a smaller model to mimic a larger model, reducing inference cost while maintaining performance.
4. **Efficient Architectures**: Use architectures like sparse transformers or mixture of experts (MoE) to reduce unnecessary computations.
5. **Batch Processing**: Process multiple inputs in parallel to maximize GPU utilization and reduce per-inference cost.
6. **Caching**: Cache frequent or repeated queries to avoid redundant computations.
7. **Dynamic Batching**: Adjust batch sizes dynamically based on workload to optimize resource usage.
8. **Cloud Optimization**: Use spot instances, auto-scaling, and cost-effective cloud services to manage infrastructure costs.
9. **Model Sharding**: Distribute model layers across multiple devices to balance load and reduce memory overhead.
10. **Early Exit**: Allow the model to exit early for simpler inputs, reducing computation for easier tasks.
11. **Use Pre-trained Models**: Leverage pre-trained models and fine-tune them for specific tasks instead of training from scratch.
12. **Monitor and Optimize**: Continuously monitor resource usage and optimize based on real-world performance data.

---

### 15.2 What are mixture of expert models (MoE)?

Mixture of Expert models (MoE) is a machine learning approach where multiple specialized sub-models (experts) are combined to handle different parts of the input space. A gating network determines which expert(s) to activate for a given input, allowing the model to dynamically allocate resources based on the input's characteristics. This architecture is particularly useful for scaling large models efficiently, as it enables conditional computation—only a subset of experts is activated per input, reducing computational cost while maintaining high performance. MoE is commonly used in large-scale neural networks, such as in transformer-based models, to improve efficiency and scalability.

---

### 15.3 How to build production grade RAG system, explain each component in detail?

Building a **production-grade Retrieval-Augmented Generation (RAG) system** requires robust architecture, scalable components, and efficient retrieval mechanisms. Below is a breakdown of each component:

---

#### **1. Data Ingestion & Preprocessing**

- **Purpose:** Convert raw data into structured, searchable chunks.
- **Key Steps:**
  - **Data Collection:** Crawl, scrape, or integrate APIs for acquiring structured/unstructured data.
  - **Text Preprocessing:** Tokenization, stopword removal, stemming, and lemmatization.
  - **Chunking:** Split large documents into semantically meaningful chunks (e.g., 256–512 tokens).
  - **Embedding Generation:** Convert text into vector representations using models like `OpenAI embeddings`, `BGE`, `FAISS`, `Hugging Face Transformers`, etc.
  - **Metadata Tagging:** Enhance retrieval with metadata (e.g., timestamps, source tags).

---

#### **2. Vector Database (Retrieval Engine)**

- **Purpose:** Efficiently store and retrieve relevant text embeddings.
- **Popular Choices:** FAISS, Pinecone, Weaviate, Qdrant, Chroma.
- **Key Considerations:**
  - **Indexing Method:** HNSW (Hierarchical Navigable Small World), IVF-Flat for fast approximate nearest neighbor (ANN) search.
  - **Hybrid Search:** Combine vector similarity (`cosine/dot product`) with keyword-based retrieval (BM25).
  - **Metadata Filtering:** Allow filtering results based on additional attributes.
  - **Scalability:** Choose distributed storage (e.g., Pinecone, Weaviate) for large-scale deployments.

---

#### **3. Query Processing & Augmentation**

- **Purpose:** Improve user queries for better retrieval accuracy.
- **Techniques:**
  - **Query Rewriting/Re-ranking:** Use LLMs to reframe ambiguous queries.
  - **Query Expansion:** Add synonyms or context to improve recall.
  - **Multi-Hop Retrieval:** Retrieve context from multiple sources iteratively.
  - **Personalization:** Adapt retrieval based on user history/preferences.

---

#### **4. LLM Response Generation**

- **Purpose:** Generate context-aware responses using retrieved documents.
- **Steps:**
  - **Context Injection:** Format retrieved passages and pass them to the LLM.
  - **Prompt Engineering:** Use structured templates (`e.g., system messages, role-based prompting`).
  - **Chain-of-Thought (CoT) Reasoning:** Improve logical coherence in responses.
  - **Re-ranking of Retrieved Context:** Use `Cross-Encoder` models to prioritize the most relevant chunks.

---

#### **5. Post-processing & Guardrails**

- **Purpose:** Ensure safe, accurate, and structured responses.
- **Key Aspects:**
  - **Fact-Checking & Validation:** Cross-check LLM outputs with retrieved sources.
  - **Toxicity & Bias Filtering:** Use moderation APIs (e.g., OpenAI, Perspective API).
  - **Response Formatting:** Generate outputs in JSON, markdown, or structured formats for UI integration.
  - **Confidence Scoring:** Provide a certainty level for each response.

---

#### **6. API Layer & Deployment**

- **Purpose:** Expose the RAG system as a scalable API.
- **Best Practices:**
  - **FastAPI / Flask** for serving.
  - **Asynchronous Processing** (Celery, RabbitMQ, Kafka) for high throughput.
  - **Rate Limiting & Caching** (Redis, Cloudflare) to optimize performance.
  - **Logging & Monitoring** (Prometheus, ELK Stack) for debugging and analytics.

---

#### **7. Feedback Loop & Continuous Improvement**

- **Purpose:** Improve system accuracy over time.
- **Strategies:**
  - **Human-in-the-loop Feedback:** Collect user interactions and retrain embeddings.
  - **Retrieval Fine-tuning:** Optimize embeddings & indexing parameters.
  - **Adaptive Prompting:** Adjust prompt templates based on response effectiveness.
  - **Data Drift Detection:** Monitor vector distributions and retrain models when needed.

---

#### **End-to-End Flow**

1. **User Query → Query Preprocessing**  
2. **Query → Vector Search in DB**  
3. **Top-k Relevant Docs → Re-ranking**  
4. **Filtered Context → LLM for Response Generation**  
5. **Generated Response → Post-processing & Guardrails**  
6. **Final Response → User API/UI**  
7. **Feedback → System Improvement**

---

### 15.4 What is FP8 variable and what are its advantages of it?

FP8 (8-bit floating point) is a reduced-precision floating-point format used in machine learning, particularly for training and inference of large models like LLMs. It uses 8 bits to represent floating-point numbers, typically with 1 sign bit, 4 exponent bits, and 3 mantissa bits (or similar configurations).

**Advantages:**

1. **Reduced Memory Usage:** FP8 requires less memory compared to FP16 or FP32, enabling larger models or batch sizes within the same memory constraints.
2. **Lower Bandwidth:** Reduces data transfer bandwidth between memory and compute units, improving efficiency.
3. **Faster Computation:** Smaller bit-width allows for faster arithmetic operations, speeding up training and inference.
4. **Energy Efficiency:** Lower precision reduces power consumption, making it suitable for edge devices or large-scale deployments.

FP8 is particularly useful in scenarios where performance and efficiency are critical, such as in deep learning accelerators or GPUs.

---

### 15.5 How to train LLM with low precision training without compromising on accuracy?

To train LLMs with low precision (e.g., FP16 or BF16) without compromising accuracy:

1. **Mixed Precision Training**: Use frameworks like NVIDIA's Apex or PyTorch's native AMP (Automatic Mixed Precision). This maintains FP32 for master weights and optimizer states, while using FP16/BF16 for forward/backward passes, reducing memory usage and speeding up training.

2. **Loss Scaling**: Scale the loss to prevent underflow in gradients during backpropagation, especially with FP16. Frameworks like AMP handle this automatically.

3. **BF16 for Stability**: BF16 offers a wider dynamic range than FP16, reducing the risk of overflow/underflow while maintaining similar memory benefits.

4. **Gradient Accumulation**: Accumulate gradients over multiple mini-batches to simulate larger batch sizes, improving stability with low precision.

5. **Optimizer Choice**: Use optimizers like AdamW or LAMB, which are robust to low precision and help maintain accuracy.

6. **Regularization**: Apply techniques like dropout or weight decay to prevent overfitting, which can be more critical with low precision.

7. **Learning Rate Tuning**: Adjust learning rates carefully, as low precision can affect gradient updates. Use learning rate warm-up and schedulers.

8. **Checkpointing**: Save intermediate model states to recover from potential instability during training.

---

### 15.6 How to calculate size of KV cache?

The size of the **Key-Value (KV) cache** in transformer-based models depends on several factors, including **batch size, sequence length, number of layers, number of attention heads, and head dimension**.  

#### **Formula for KV Cache Size**

```KV Cache Size = Batch Size × Number of Layers × 2 × Sequence Length × Number of Heads × Head Dimension × Data Type Size```

#### **Breakdown of Terms:**

- **Batch Size (B):** Number of sequences processed in parallel.
- **Number of Layers (L):** Total transformer layers in the model.
- **Factor of 2:** Because both **keys (K) and values (V)** are stored.
- **Sequence Length (S):** The current token count stored in cache.
- **Number of Attention Heads (H):** Total attention heads in the model.
- **Head Dimension (D):** The embedding dimension per attention head (e.g., for GPT-3, `D = 64`).
- **Data Type Size:** Depends on precision (FP32 = 4 bytes, FP16/BF16 = 2 bytes, INT8 = 1 byte).

---

#### **Example Calculation (GPT-3 175B)**

#### **Given Parameters:**

- **Batch Size (B) = 1**
- **Number of Layers (L) = 96**
- **Number of Heads (H) = 96**
- **Head Dimension (D) = 128**
- **Sequence Length (S) = 2048 (full context window)**
- **Precision = FP16 (2 bytes per value)**

#### **Applying the Formula:**

`1 × 96 × 2 × 2048 × 96 × 128 × 2 bytes
= 96 × 2 × 2048 × 96 × 128 × 2
= 9.6 GiB`

#### **Optimizations to Reduce KV Cache Size**

1. **Lower Precision** – Using `BF16/FP16` instead of `FP32` reduces memory usage.
2. **Flash Attention** – Reduces memory overhead by recomputing attention on demand.
3. **Sliding Window Attention** – Retains only the most relevant tokens instead of full history.
4. **Sparse Attention Mechanisms** – Limits attention to a subset of tokens.

---

### 15.7 Explain dimension of each layer in multi headed transformation attention block

In a multi-headed attention block, the dimensions of each layer are as follows:

1. **Input Embedding**: Shape `(batch_size, seq_len, d_model)`, where `d_model` is the dimension of the input embeddings.

2. **Linear Projections (Q, K, V)**: Each head projects the input into query (Q), key (K), and value (V) matrices. The shape for each is `(batch_size, seq_len, d_k)`, where `d_k = d_model / h` and `h` is the number of heads.

3. **Scaled Dot-Product Attention**: The attention scores are computed as `Q @ K.T / sqrt(d_k)`, resulting in a shape of `(batch_size, h, seq_len, seq_len)`.

4. **Attention Output**: The attention output is computed as `attention_scores @ V`, resulting in a shape of `(batch_size, h, seq_len, d_k)`.

5. **Concatenation**: The outputs from all heads are concatenated along the last dimension, resulting in a shape of `(batch_size, seq_len, d_model)`.

6. **Final Linear Projection**: A final linear layer projects the concatenated output back to the original dimension `(batch_size, seq_len, d_model)`.

---

### 15.8 How do you make sure that attention layer focuses on the right part of the input?

Ensuring that the **attention layer** focuses on the right part of the input involves a combination of **architectural choices, training strategies, and optimization techniques**. Here are the key methods:

#### **1. Positional Encoding & Token Embeddings**  

- Since transformers do not have inherent order awareness, **positional encodings** (sinusoidal or learned) provide **context on token positions**, ensuring proper focus.

#### **2. Attention Masking**  

- **Causal Masking:** Used in autoregressive models (e.g., GPT) to prevent attending to future tokens.  
- **Padding Masking:** Prevents attention to padding tokens, ensuring focus remains on meaningful inputs.  

#### **3. Pretraining on High-Quality Data**  

- Training on **diverse, well-labeled, and structured datasets** helps the model **learn meaningful attention patterns** rather than spurious correlations.

#### **4. Supervised Fine-tuning with Attention Guidance**  

- **Providing labeled examples** where correct focus regions are known (e.g., using human annotations) ensures attention aligns with task-specific needs.

#### **5. Multi-Head Self-Attention (MHSA)**  

- **Different attention heads** capture **different aspects of the input** (e.g., syntax, semantics), leading to a more robust focus distribution.

#### **6. Reinforcement Learning with Human Feedback (RLHF)**  

- Techniques like **reward models** guide attention towards **more interpretable and correct outputs**, improving focus over iterations.

#### **7. Attention Visualization & Interpretability Checks**  

- Using tools like **attention heatmaps (e.g., BertViz, Captum)** to analyze where the model is focusing and adjusting training accordingly.

#### **8. Adversarial Training & Robustness Checks**  

- Testing with adversarial examples ensures that attention remains stable under **input perturbations**.

---
