#### *100+ LLM Questions and Answers Every AI Engineer Needs to Know*

---

# 1. Prompt Engineering & Basics of LLM

### 1.1 What is the difference between Predictive/Discriminative AI and Generative AI?

Predictive/Discriminative AI focuses on learning the boundary between classes or predicting outcomes based on input data. It models the conditional probability ( P(Y|X) ), where ( Y ) is the target variable and ( X ) is the input. Examples include classification and regression tasks.

Generative AI, on the other hand, aims to model the joint probability ( P(X, Y) ) or ( P(X) ) to generate new data samples that resemble the training data. It can create new content, such as text, images, or audio, by learning the underlying data distribution. Examples include GANs, VAEs, and language models like GPT.

---

### 1.2 What is LLM, and how are LLMs trained?

LLM stands for Large Language Model, a type of AI model designed to understand and generate human-like text. LLMs are trained on vast amounts of text data using unsupervised learning, typically employing transformer architectures. The training process involves predicting the next word in a sequence (autoregressive modeling) or filling in missing words (masked language modeling). The model learns patterns, grammar, and context from the data, enabling it to generate coherent and contextually relevant text. Training requires significant computational resources, often involving GPUs or TPUs, and large-scale datasets like Common Crawl or Wikipedia. Fine-tuning on specific tasks or domains can further enhance performance.

---

### 1.3 What is a token in the language model?

A token in a language model is the smallest unit of text that the model processes. It can be as short as a single character or as long as a word, depending on the tokenization method used. For example, the word "unhappiness" might be split into tokens like "un", "happiness". Tokens are the input format for the model, which then processes them to generate predictions or outputs.

---

### 1.4 How to estimate the cost of running SaaS-based and Open Source LLM models?

For SaaS-based LLMs (API-based like OpenAI, Anthropic, or Google), calculate token pricing (cost per 1M input/output tokens), average tokens per request, and total monthly queries. Include additional costs for fine-tuning, rate limits, and potential overages.

For Open Source LLMs (self-hosted like Llama, Mistral, or Falcon), factor in infrastructure costs, including GPU instances (cloud or on-premise), storage, bandwidth, and operational costs (MLOps, monitoring, scaling). Cloud GPU pricing varies by provider (e.g., AWS, GCP, Azure), and fine-tuning/inference costs depend on model size, batch processing, and quantization.

To calculate the number of GPUs required for a large language model (LLM), consider the model size, GPU memory, and computational needs for inference or training. The key factor is VRAM, as model parameters must fit into memory. In FP32 precision, each parameter takes 4 bytes, meaning a 1B parameter model requires ~4GB VRAM, while FP16 reduces it to ~2GB, and quantization (e.g., INT8) can further cut memory usage. For inference, the total VRAM should accommodate the model, activation memory, and batch size overhead. For training, memory requirements increase due to optimizer states and gradients, typically requiring 2-3x more VRAM than inference. GPUs like A100 (80GB), H100 (94GB), or consumer-grade options like RTX 4090 (24GB) set practical limits on model deployment. To determine the number of GPUs, divide the required memory by the available per-GPU VRAM and ensure inter-GPU communication via NVLink or high-bandwidth interconnects for multi-GPU setups.

---

### 1.5 Explain the Temperature parameter and how to set it

The temperature parameter in LLMs controls the randomness of token selection during text generation. A higher temperature (e.g., 1.0–2.0) increases randomness, making outputs more diverse and creative but also less predictable. A lower temperature (e.g., 0.1–0.3) makes the model more deterministic, focusing on high-probability tokens and producing more factual, repetitive, and structured responses. To set it correctly, use low temperature (0.1–0.3) for tasks requiring precision, such as factual answers, coding, or scientific explanations, moderate temperature (0.5–0.8) for balanced responses with some creativity while maintaining coherence, and high temperature (1.0–2.0) for storytelling, brainstorming, or when diversity is more important than accuracy. In production, adjusting temperature dynamically based on user intent can optimize response quality.

---

### 1.6 What are different decoding strategies for picking output tokens?

Different decoding strategies for picking output tokens in LLMs include:

- Greedy Decoding: Chooses the token with the highest probability at each step, leading to deterministic and often repetitive outputs. It’s fast but lacks diversity.  
- Beam Search: Maintains multiple candidate sequences (beams) and selects the best one based on cumulative probability. It balances diversity and accuracy but is computationally expensive.  
- Top-k Sampling: Selects the next token from the top-k most likely tokens, introducing randomness by limiting the pool of candidates. This increases diversity while maintaining coherence.  
- Top-p (Nucleus) Sampling: Chooses the next token from the smallest set of tokens whose cumulative probability exceeds a threshold p. This allows for dynamic variability while avoiding low-probability tokens.  
- Temperature Sampling: Modifies the probabilities by scaling them according to the temperature value, making the model either more deterministic or more random, depending on the setting.  
- Random Sampling: Picks the next token randomly according to the probability distribution. This method provides maximum diversity but can lead to incoherent outputs.  

Each strategy has its use cases depending on the desired balance between randomness, coherence, and computational efficiency.

---

### 1.7 What are different ways you can define stopping criteria in large language model?

Different ways to define stopping criteria in large language models include:

- Maximum Token Length: The generation stops after producing a predefined number of tokens, ensuring output length control.
- End-of-Sequence Token: The model stops when it generates a special token (e.g., `<EOS>`) that indicates the end of a meaningful sequence.
- Maximum Generation Time: The generation stops after a specified time limit, which can help control latency and computational costs.
- Probability Threshold: The model stops if the probability of the next token falls below a certain threshold, indicating low-confidence or incoherent outputs.
- Repetition Penalty: Stops if the model generates too many repeated tokens or phrases within a sequence to avoid redundancy.
- Perplexity Threshold: Stops when the model’s perplexity (a measure of prediction uncertainty) exceeds a certain threshold, often indicating diminishing quality of output.
- Custom Criteria: Domain-specific or task-specific stopping conditions, such as specific key phrases, semantic coherence, or user-defined quality checks, can be set.  

These criteria help control the model's behavior, ensuring efficiency, relevance, and quality of generated text.

---

### 1.8 How to use stop sequences in LLMs?

Stop sequences in LLMs are predefined strings or tokens that signal the model to stop generating further text. They are useful for ensuring that the output is confined to a certain format or does not continue indefinitely. Here's how to use them:

1. **Define Stop Sequences**: Choose specific tokens or sequences that should trigger the model to stop, such as `<EOS>`, `</s>`, or other custom strings like `END`, `STOP`, or even punctuation marks like `\n` or `.` depending on the task.

2. **Configure Model Parameters**: When making an inference request, set the stop sequences as part of the model's generation parameters. In API-based models, this can be done by including the `stop` parameter, followed by a list of stop sequences.

3. **Multiple Stop Sequences**: You can define multiple stop sequences. The model will stop as soon as it generates any of the predefined sequences.

4. **Use in Structured Outputs**: For tasks like question-answering or code generation, stop sequences ensure that the model halts after completing a logical unit, such as after an answer or code block.

5. **Adjust According to Context**: In some cases, you may need to dynamically adjust stop sequences depending on context or specific requirements, such as stopping after a certain phrase or when a new section starts.

By using stop sequences, you can improve the control over the generated text, ensuring that it meets the expected format and length.

---

### 1.9 Explain the basic structure prompt engineering

Prompt engineering is the process of designing input prompts that guide large language models (LLMs) to generate desired outputs. The basic structure of prompt engineering typically involves several key elements:

1. **Instruction/Task**: The first part of the prompt is a clear and specific instruction or task that tells the model what to do. This can range from asking a question, providing a command, or setting a context for the model to follow. For example, "Write a short story about a space adventure" or "Summarize the following article."

2. **Context/Background Information**: Providing relevant context helps the model understand the scope and nuances of the task. This could include details, definitions, examples, or references. For instance, "In the year 2500, humans have colonized Mars…" or "Here is a brief overview of the article you need to summarize."

3. **Input/Query**: The input or query is the specific content or data the model will work with. This could be a question, a piece of text, an image description, or structured data. It is important that the input is clear and unambiguous. For example, "What is the capital of France?" or "Here is a product description: 'This laptop has a 16GB RAM, Intel i7 processor...'"

4. **Constraints/Output Instructions**: You can also set constraints or guidelines for the output, such as word limits, format, or style. For example, "Provide a concise 2-sentence summary" or "Generate the text in a formal tone."

5. **Examples (optional)**: Providing examples of desired outputs can help guide the model, especially in tasks that require specific formats or responses. For instance, "Example 1: 'The capital of France is Paris.' Example 2: 'The capital of Spain is Madrid.'"

By structuring the prompt in a clear and purposeful way, you guide the model towards generating high-quality, relevant, and accurate responses.

---

### 1.10 Explain in-context learning

In-context learning refers to the ability of a large language model (LLM) to learn and adapt to specific tasks or behaviors directly from the input provided during inference, without requiring any explicit retraining. This process leverages the model’s understanding of patterns and context within the prompt to generate appropriate responses or solve problems.

Here’s how it works:

1. **Contextual Input**: The model is given an input prompt containing relevant examples or patterns that provide context for the task. This can include instructions, examples, or any other data that the model can infer meaning from. The model does not learn permanently; it adapts based on the given input context during the current interaction.

2. **Task Understanding**: Based on the examples and instructions provided in the context, the model uses its pre-trained knowledge to generalize and perform the desired task. For example, if a user provides several examples of translations, the model can infer that it should translate similar sentences without being explicitly retrained for that specific task.

3. **No Need for Fine-Tuning**: Unlike traditional machine learning, where models are retrained with labeled data to adapt to new tasks, in-context learning allows the model to adapt instantly based on the current input. It uses the provided context to understand the user’s needs and generate relevant outputs accordingly.

4. **Prompt Flexibility**: The model can learn from a wide range of examples and instructions, and the effectiveness of in-context learning often depends on the quality and clarity of the prompt. The more structured and consistent the context, the more likely the model will generate accurate results.

In summary, in-context learning enables LLMs to leverage patterns and examples within a single prompt to adapt to a task, making it a powerful tool for quickly solving problems without requiring additional training or model updates.

---

### 1.11 Explain type of prompt engineering

Prompt engineering involves designing and optimizing input prompts to guide the behavior of large language models (LLMs). Here are the main types:

1. **Zero-Shot Prompting**: The model generates a response without any prior examples, relying solely on the input prompt. Example: "Translate this English sentence to French: 'Hello, how are you?'"

2. **Few-Shot Prompting**: The model is given a few examples in the prompt to guide its response. Example: "Translate English to French: 'Hello' -> 'Bonjour', 'Goodbye' -> 'Au revoir'. Now translate: 'How are you?'"

3. **Chain-of-Thought (CoT) Prompting**: Encourages the model to break down complex tasks into intermediate steps. Example: "Q: If Alice has 3 apples and Bob gives her 2 more, how many does she have? A: Alice starts with 3 apples. Bob gives her 2 more. 3 + 2 = 5. So, Alice has 5 apples."

4. **Instruction-Based Prompting**: Directly instructs the model to perform a specific task. Example: "Summarize the following text in one sentence: [text]"

5. **Role-Playing Prompting**: Assigns a role to the model to shape its responses. Example: "You are a helpful assistant. Explain quantum computing in simple terms."

6. **Template-Based Prompting**: Uses structured templates to guide the model's output. Example: "Fill in the blanks: The capital of France is ____."

7. **Meta-Prompting**: Prompts the model to generate or refine its own prompts. Example: "Create a prompt that would help a student understand the concept of recursion."

Each type serves different use cases and can be combined for better performance.

---

### 1.12 What are some of the aspect to keep in mind while using few-shots prompting?

When using few-shot prompting, consider the following aspects:

1. **Relevance**: Ensure the examples provided are highly relevant to the task. Irrelevant examples can mislead the model.

2. **Diversity**: Include diverse examples to cover a range of scenarios, helping the model generalize better.

3. **Clarity**: Make sure the examples are clear and unambiguous. Ambiguity can lead to incorrect outputs.

4. **Consistency**: Use consistent formatting and structure across examples to help the model understand the pattern.

5. **Task Alignment**: Align the examples with the specific task or domain. Misalignment can reduce the model's effectiveness.

6. **Brevity**: Keep examples concise. Overly verbose examples can dilute the model's focus.

7. **Order**: The order of examples can influence the model's output. Place the most representative examples first.

8. **Context**: Provide sufficient context in the examples to guide the model, especially for complex tasks.

9. **Bias**: Be aware of potential biases in the examples, as the model may amplify them in its responses.

10. **Evaluation**: Continuously evaluate the model's performance and refine the examples as needed to improve accuracy.

---

### 1.13 What are certain strategies to write good prompt?

1. **Be Clear and Specific**: Clearly define the task and provide specific instructions to guide the model. Avoid ambiguity.

2. **Use Examples**: Include examples (few-shot learning) to demonstrate the desired output format or style.

3. **Iterate and Refine**: Test and refine prompts based on the model's responses to improve accuracy and relevance.

4. **Control Length**: Specify the desired length of the response (e.g., "in one sentence" or "in 100 words") to avoid overly verbose or too brief answers.

5. **Leverage Context**: Provide relevant context or background information to help the model generate more accurate responses.

6. **Use Constraints**: Apply constraints (e.g., "list three reasons" or "use formal language") to guide the model's output.

7. **Break Down Complex Tasks**: Divide complex tasks into smaller, manageable sub-tasks with separate prompts if needed.

8. **Experiment with Formats**: Try different prompt formats (e.g., questions, commands, or fill-in-the-blank) to see what works best.

9. **Avoid Bias**: Be mindful of potential biases in the prompt that could influence the model's output.

10. **Use System Messages**: For conversational models, use system messages to set the tone or role (e.g., "You are a helpful assistant").

---

### 1.14 What is hallucination, and how can it be controlled using prompt engineering?

Hallucination in LLMs refers to the generation of incorrect, nonsensical, or fabricated information that is not grounded in the input data or reality. To control hallucination using prompt engineering:

1. **Explicit Instructions**: Provide clear, detailed instructions to guide the model's output, such as "Answer based on verified facts only."
2. **Contextual Constraints**: Include specific context or constraints in the prompt to limit the scope of the response.
3. **Iterative Refinement**: Use iterative prompting to refine the output, correcting errors in subsequent prompts.
4. **Source Anchoring**: Reference specific sources or data in the prompt to ensure the model grounds its response in factual information.
5. **Temperature Adjustment**: Lower the temperature parameter to reduce randomness and encourage more deterministic, fact-based responses.

---

### 1.15 How to improve the reasoning ability of LLM through prompt engineering?

To improve LLM reasoning through prompt engineering:  

1. **Chain-of-Thought (CoT):** *Guide step-by-step reasoning instead of asking for direct answers.*  
2. **Few-Shot CoT:** *Provide examples before posing a new question.*  
3. **Explicit Thinking Instruction:** *Use "Let's think step by step."*  
4. **Self-Consistency:** *Generate multiple answers and pick the most consistent one.*  
5. **Ask for Explanations:** *Request reasoning before the final answer.*  
6. **Role-Playing:** *Frame the model as an expert (e.g., "You are a mathematician").*  
7. **Scratchpad Method:** *Encourage writing intermediate steps.*  
8. **Decomposition:** *Break problems into smaller subtasks.*  
9. **Analogies:** *Link problems to familiar concepts.*  
10. **Self-Reflection:** *Ask the model to review and refine its response.*

---

### 1.16 How to improve LLM reasoning if your COT prompt fails? 

1. **Self-Consistency:** Generate multiple answers, pick the most frequent.  
2. **Refine Prompt:** Make instructions clearer, more structured.  
3. **Stepwise Decomposition:** Break the task into smaller steps.  
4. **Self-Reflection:** Ask the model to verify and refine its answer.  
5. **Alternative Methods:** Try **Tree-of-Thought (ToT)** or **Scratchpad.**  
6. **Examples & Analogies:** Relate to familiar concepts.  
7. **Adjust Temperature:** Lower for logic, higher for creativity.  
8. **External Validation:** Cross-check with another model or tool.  

---
