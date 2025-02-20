#### *100+ LLM Questions and Answers Every AI Engineer Needs to Know*

---

# 14. Prompt Hacking

## Table of Contents

- [14.1 What is prompt hacking and why should we bother about it?](#141-what-is-prompt-hacking-and-why-should-we-bother-about-it)  
- [14.2 What are the different types of prompt hacking?](#142-what-are-the-different-types-of-prompt-hacking)  
- [14.3 What are the different defense tactics from prompt hacking?](#143-what-are-the-different-defense-tactics-from-prompt-hacking)  

---

### 14.1 What is prompt hacking and why should we bother about it?

Prompt hacking refers to the manipulation of input prompts to a language model (LLM) to elicit unintended or harmful outputs. This can include bypassing safety filters, generating biased or toxic content, or extracting sensitive information. It is a concern because it can undermine the reliability, safety, and ethical use of LLMs, potentially leading to misuse in real-world applications. Addressing prompt hacking is crucial to ensure the responsible deployment of AI systems.

---

### 14.2 What are the different types of prompt hacking?

Prompt hacking refers to techniques used to manipulate or exploit the behavior of large language models (LLMs) by crafting specific inputs. The main types include:

1. **Prompt Injection**: Inserting malicious or misleading instructions into the input to alter the model's output.
2. **Prompt Leaking**: Extracting sensitive information or internal prompts from the model by crafting specific queries.
3. **Jailbreaking**: Bypassing the model's safety or ethical constraints to generate restricted content.
4. **Adversarial Attacks**: Crafting inputs that cause the model to produce incorrect or nonsensical outputs.
5. **Data Extraction**: Exploiting the model to reveal training data or other sensitive information.
6. **Instruction Manipulation**: Altering the model's behavior by rephrasing or restructuring the input instructions.

---

### 14.3 What are the different defense tactics from prompt hacking?

Prompt hacking refers to adversarial attacks on LLMs to manipulate outputs, bypass safeguards, or extract sensitive information. Key defense tactics include:

1. **Input Sanitization** – Filtering, tokenizing, and normalizing input to detect harmful patterns.
2. **Few-shot Prompting with Guardrails** – Using structured prompts with constraints to guide safe responses.
3. **Output Filtering** – Post-processing model responses using regex, classifiers, or heuristics.
4. **Reinforcement Learning with Human Feedback (RLHF)** – Training the model to avoid harmful completions.
5. **Fine-tuned Access Control** – Restricting model capabilities based on user roles and context.
6. **Rate Limiting & Monitoring** – Detecting anomalies in input patterns and response frequencies.
7. **Adversarial Testing & Red-Teaming** – Running penetration tests to identify vulnerabilities.
8. **Prompt Injection Detection** – Using classifiers to identify prompt manipulations.
9. **Context Length & Memory Restrictions** – Limiting retained input to prevent long-range attacks.
10. **Multi-agent Verification** – Cross-checking outputs with additional AI models for inconsistencies.

---
