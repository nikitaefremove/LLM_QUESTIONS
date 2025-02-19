#### *100+ LLM Questions and Answers Every AI Engineer Needs to Know*

---

# 11. Hallucination Control Techniques

## Table of Contents

- [11.1 What are different forms of hallucinations?](#111-what-are-different-forms-of-hallucinations)
- [11.2 How to control hallucinations at various levels?](#112-how-to-control-hallucinations-at-various-levels)

### 11.1 What are different forms of hallucinations?

Hallucinations in LLMs can take several forms:

1. **Factual Hallucinations**: The model generates incorrect or fabricated facts.
2. **Contextual Hallucinations**: The model produces information that is inconsistent with the given context.
3. **Logical Hallucinations**: The model generates statements that are logically inconsistent or nonsensical.
4. **Temporal Hallucinations**: The model provides incorrect or anachronistic timelines or events.
5. **Referential Hallucinations**: The model incorrectly references entities, sources, or data that do not exist or are irrelevant.
6. **Contradictory Hallucinations**: The model produces statements that contradict previously generated content or known facts.

---

### 11.2 How to control hallucinations at various levels?

To control hallucinations in LLMs at various levels, consider the following techniques:

1. **Input Level**:
   - **Prompt Engineering**: Use clear, specific, and well-structured prompts to guide the model.
   - **Contextual Anchoring**: Provide relevant context or examples to reduce ambiguity.

2. **Model Level**:
   - **Fine-Tuning**: Fine-tune the model on domain-specific data to improve relevance and accuracy.
   - **Temperature Adjustment**: Lower the temperature to reduce randomness and increase determinism.

3. **Output Level**:
   - **Post-Processing**: Implement filters or rules to detect and correct hallucinations.
   - **Verification Mechanisms**: Use external knowledge bases or fact-checking tools to validate outputs.

4. **System Level**:
   - **Ensemble Methods**: Combine multiple models to cross-verify outputs.
   - **Human-in-the-Loop**: Incorporate human review for critical applications.

---
