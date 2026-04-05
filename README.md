# 🧠 AI Evaluation Framework (Q&A Use Case)

This project is a simple, hands-on **AI evaluation framework** to compare how different LLMs behave on the same task.

It evaluates multiple models (via Ollama) on a Q&A dataset and scores their outputs using rule-based heuristics.

---

## 🚀 What this project does

- Reads a dataset of **questions + context + expected answers**
- Sends the same inputs to multiple LLMs
- Collects model outputs
- Automatically evaluates responses using: Correctness, Grounding, Completeness, Hallucination detection
- Assigns a **failure type** to each response

---

## 🏗️ Architecture

### 1. Dataset Layer
Defines:
- The task
- Expected behavior
- Ground truth

### 2. Model Layer
Runs multiple models via Ollama:
- gemma
- qwen
- llama3.1

Each model independently generates outputs.

### 3. Evaluation Layer
A rule-based scoring system that:
- Compares outputs with expected answers
- Detects failure patterns
- Assigns structured labels

---

## 📊 Example Output

| Question | Model | Output | Failure Type |
|---------|------|--------|-------------|
| Is it in stock? | gemma | Yes, it is in stock | Not grounded |
| Is it in stock? | qwen | Stock not mentioned | Correct |

---

## 🧠 Key Learnings

- AI systems fail in **patterns**, not randomly
- Hallucination happens when context is missing
- Models tend to be **overconfident**
- Accuracy alone is not enough — **evaluation matters**

---

## ⚖️ What this is (and what it is not)

### ✅ This is:
- A **multi-model evaluation pipeline**
- A **rule-based evaluator**
- A way to understand **AI failure modes**

### ❌ This is NOT:
- Training a model
- Fine-tuning
- LLM-as-a-Judge (yet)

---

## 🧪 How to run

### 1. Install dependencies

```bash
pip install pandas requests
````

### 2. Install Ollama

Download from: [https://ollama.com](https://ollama.com)

### 3. Pull models

```bash
ollama pull gemma
ollama pull qwen
ollama pull llama3.1
```

### 4. Run evaluation

```bash
python run_eval.py
```

### 5. Output

Results are saved in:

```
evaluated_outputs.csv
```

---

## 🔍 What to look for

* Which model hallucinates more?
* Which model is more cautious?
* Which model handles edge cases better?

---

## 🚧 Limitations

* Rule-based scoring is approximate
* Cannot fully capture reasoning quality
* May misclassify borderline cases
* Relies on predefined expected answers

---

## 🔮 Potential Next Step: LLM-as-a-Judge

The next evolution of this framework is to replace rule-based scoring with **LLM-based evaluation**.

### What is LLM-as-a-Judge?

Instead of manually scoring outputs, use another LLM to evaluate responses.
