#### *100+ LLM Questions and Answers Every AI Engineer Needs to Know*

---

# 13. Agent-Based System

## Table of Contents

- [13.1 Explain the basic concepts of an agent and the types of strategies available to implement agents](#131-explain-the-basic-concepts-of-an-agent-and-the-types-of-strategies-available-to-implement-agents)
- [13.2 Why do we need agents and what are some common strategies to implement agents?](#132-why-do-we-need-agents-and-what-are-some-common-strategies-to-implement-agents)
- [13.3 Explain ReAct prompting with a code example and its advantages](#133-explain-react-prompting-with-a-code-example-and-its-advantages)  
- [13.4 Explain Plan and Execute prompting strategy](#134-explain-plan-and-execute-prompting-strategy)  
- [13.5 Explain OpenAI functions strategy with code examples](#135-explain-openai-functions-strategy-with-code-examples)  
- [13.6 Explain the difference between OpenAI functions vs LangChain Agents](#136-explain-the-difference-between-openai-functions-vs-langchain-agents)  

---

### 13.1 Explain the basic concepts of an agent and the types of strategies available to implement agents

An **agent** in AI is an autonomous entity that perceives its environment through sensors and acts upon it using actuators to achieve specific goals. Agents can be simple (e.g., reflex agents) or complex (e.g., learning agents).

#### Basic Concepts

1. **Autonomy**: Operates without direct human intervention.
2. **Perception**: Uses sensors to gather data from the environment.
3. **Action**: Uses actuators to perform actions.
4. **Goal-Driven**: Acts to achieve specific objectives.
5. **Rationality**: Makes decisions to maximize performance.

#### Types of Strategies

1. **Simple Reflex Agents**: Act based on current percepts using condition-action rules.
2. **Model-Based Reflex Agents**: Maintain an internal state to track the environment.
3. **Goal-Based Agents**: Use goal information to choose actions.
4. **Utility-Based Agents**: Maximize utility or performance metrics.
5. **Learning Agents**: Improve performance over time through experience.

---

### 13.2 Why do we need agents and what are some common strategies to implement agents?

We need agents in systems to enable autonomous, goal-oriented behavior, where agents can perceive their environment, make decisions, and act to achieve specific objectives. Agents are particularly useful in complex, dynamic, or distributed environments where centralized control is impractical.

Common strategies to implement agents include:

1. **Reactive Agents**: These agents respond to environmental changes in real-time using simple if-then rules or condition-action pairs. They are lightweight and fast but lack long-term planning.

2. **Deliberative Agents**: These agents use symbolic reasoning and planning to make decisions. They maintain an internal model of the world and use algorithms like A* or STRIPS to plan actions.

3. **Hybrid Agents**: Combining reactive and deliberative approaches, these agents balance quick responses with long-term planning. They often use layered architectures like BDI (Belief-Desire-Intention).

4. **Learning Agents**: These agents improve their behavior over time using machine learning techniques like reinforcement learning, supervised learning, or unsupervised learning.

5. **Multi-Agent Systems (MAS)**: In MAS, multiple agents interact, collaborate, or compete to achieve individual or collective goals. Strategies include negotiation, coordination, and game theory.

6. **Utility-Based Agents**: These agents make decisions by maximizing a utility function, which quantifies the desirability of different outcomes.

---

### 13.3 Explain ReAct prompting with a code example and its advantages

ReAct (Reasoning and Acting) prompting is a technique used in large language models (LLMs) to improve their ability to perform complex tasks by combining reasoning and action steps. It allows the model to break down a problem into smaller reasoning steps and take actions (e.g., API calls, lookups) to gather information before producing a final answer.

#### Advantages

1. **Improved Reasoning**: ReAct allows the model to reason through complex problems by breaking them into smaller, manageable steps.
2. **Dynamic Information Gathering**: The model can take actions (e.g., API calls) to fetch real-time data, enhancing its ability to provide accurate and up-to-date answers.
3. **Transparency**: The step-by-step reasoning and actions make the model's decision-making process more transparent and interpretable.
4. **Flexibility**: ReAct can be applied to a wide range of tasks, from simple Q&A to complex problem-solving scenarios.

---

### 13.4 Explain Plan and Execute prompting strategy

The **Plan-and-Execute** prompting strategy is a structured approach for LLM-based agent systems to solve complex tasks by breaking them into manageable steps. It consists of two main phases:

1. **Planning Phase**  
   - The LLM is prompted to generate a **high-level plan** before executing any actions.  
   - The plan outlines the sequence of subtasks required to complete the goal.  
   - Example prompt: *"Given the user request, break down the task into step-by-step actions before execution."*

2. **Execution Phase**  
   - The agent follows the generated plan, executing each step iteratively.  
   - It may use additional LLM calls, tool invocations, or reasoning mechanisms to complete each subtask.  
   - Example prompt: *"Now execute step 1 of the plan and provide the result before proceeding to the next step."*

#### Advantages

- **Improves reasoning**: Reduces hallucinations by enforcing structured thinking.  
- **Enhances control**: Allows validation and correction of the plan before execution.  
- **Scalability**: Works well for multi-step and tool-augmented tasks.

---

### 13.5 Explain OpenAI functions strategy with code examples

#### **OpenAI Functions Strategy**  

The **OpenAI functions strategy** is used to extend LLM capabilities by integrating structured API calls or tool execution within conversations. Instead of generating free-form text, the model predicts which function to call and with what arguments, allowing interaction with external systems.

---

#### **How It Works**  

1. **Define available functions**  
   - Specify function names, parameters, and descriptions.  
2. **Invoke the model with function calling mode enabled**  
   - The model selects a function and generates structured JSON output.  
3. **Execute the function in the backend**  
   - Process the response and return results to the user.  

---

#### **Example Code: OpenAI Functions in Python**  

#### **Step 1: Define Functions**

```python
import openai
import json

def get_weather(location: str):
    """Mock function to return weather data"""
    return {"location": location, "temperature": "22°C", "condition": "Sunny"}

functions = [
    {
        "name": "get_weather",
        "description": "Retrieve weather information for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
]
```

---

#### **Step 2: Call LLM with Function Mode Enabled**

```python
response = openai.ChatCompletion.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "What's the weather in Bali?"}],
    functions=functions
)

function_call = response["choices"][0]["message"].get("function_call")
if function_call:
    function_name = function_call["name"]
    arguments = json.loads(function_call["arguments"])
    
    # Execute the function dynamically
    if function_name == "get_weather":
        result = get_weather(**arguments)

    print(result)  # Return the result to the user
```

---

#### **Key Advantages**

- **Structured API Calls**: Prevents LLM from hallucinating function outputs.  
- **Better Tool Use**: The model decides when and how to call external tools.  
- **Scalability**: Works well with multiple function calls in complex pipelines.  

This approach is widely used in **AI agents, automation systems, and LLM-powered assistants**.

---

### 13.6 Explain the difference between OpenAI functions vs LangChain Agents

**OpenAI Functions** and **LangChain Agents** both facilitate interactions with LLMs but serve different purposes:  

- **OpenAI Functions**:  
  - Allow LLMs to call predefined functions with structured outputs.  
  - Best suited for deterministic API calls, structured responses, and controlled workflows.  
  - Example: Calling a weather API, database lookup, or executing a fixed function based on the model's response.  

- **LangChain Agents**:  
  - More flexible, enabling LLMs to reason dynamically and decide which tools (functions, APIs, databases) to invoke.  
  - Use **ReAct (Reasoning + Acting)** paradigms, iterating through thought, action, and observation loops.  
  - Best for complex, multi-step workflows requiring adaptability, such as multi-tool orchestration and autonomous decision-making.  

**Key Difference**: OpenAI Functions provide structured, predictable function execution, while LangChain Agents offer dynamic decision-making with tool selection.

---
