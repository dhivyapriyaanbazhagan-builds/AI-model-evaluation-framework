import re
import pandas as pd
import requests
from difflib import SequenceMatcher

API_URL = "http://localhost:11434/api/generate"
TAGS_URL = "http://localhost:11434/api/tags"

# Edit this list to match the models you pulled locally
MODELS = [
 "gemma3:latest",
 "qwen3:latest",
 "llama3.1:latest",
]

SYSTEM_PROMPT = """
You are answering an e-commerce Q&A task.
Use only the provided context.
If the answer is not in the context, say: "I don't know."
Do not invent facts.
Be concise and grounded.
"""

def normalize_text(text: str) -> str:
 text = str(text).lower().strip()
 text = re.sub(r"\s+", " ", text)
 return text

def tokenize(text: str):
 return set(re.findall(r"[a-z0-9₹$%]+", normalize_text(text)))

def similarity(a: str, b: str) -> float:
 return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def contains_dont_know(text: str) -> bool:
 t = normalize_text(text)
 return any(phrase in t for phrase in [
 "i don't know",
 "dont know",
 "do not know",
 "not mentioned",
 "not specified",
 "cannot determine",
 "can't determine",
 ])

def is_yes(text: str) -> bool:
 t = normalize_text(text)
 return t.startswith("yes") or " yes " in f" {t} "

def is_no(text: str) -> bool:
 t = normalize_text(text)
 return t.startswith("no") or " no " in f" {t} "

def extract_numbers(text: str):
 # returns only the numeric substrings
 return re.findall(r"\d+(?:\.\d+)?", str(text))

def get_installed_models():
 try:
    resp = requests.get(TAGS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [m["name"] for m in data.get("models", [])]
 except Exception as e:
    print(f"[warn] Could not fetch installed models: {e}")
    return []

def build_prompt(question: str, context: str) -> str:
 return f"""Question: {question}
Context: {context}

Answer:"""

def call_ollama(model: str, question: str, context: str) -> str:
 payload = {
 "model": model,
 "system": SYSTEM_PROMPT,
 "prompt": build_prompt(question, context),
 "stream": False,
 "options": {
 "temperature": 0,
 "top_p": 1
 }
 }

 resp = requests.post(API_URL, json=payload, timeout=120)
 resp.raise_for_status()
 data = resp.json()
 return data.get("response", "").strip()

def score_output(question: str, context: str, expected: str, output: str):
 e = normalize_text(expected)
 o = normalize_text(output)

 expected_unknown = any(phrase in e for phrase in [
 "not mentioned",
 "not explicitly stated",
 "not in the context",
 "i don't know",
 "unknown"
 ])

 expected_yes = e.startswith("yes") or " yes " in f" {e} "
 expected_no = e.startswith("no") or " no " in f" {e} "
 expected_numbers = extract_numbers(expected)
 output_numbers = extract_numbers(output)

 # defaults
 correctness = 1
 grounding = 1
 completeness = 1
 hallucination = "N"
 failure_type = "Incorrect"

 # 1) If the expected answer is "unknown" and the model says it doesn't know
 if expected_unknown and contains_dont_know(output):
  return 3, 3, 3, "N", "Correct"

 # 2) If the model refuses / doesn't know when the answer is present
 if contains_dont_know(output) and not expected_unknown:
  return 1, 2, 1, "N", "Overcautious / Missing Answer"

 # 3) Yes/no mismatches
 if expected_yes and is_no(output):
  return 1, 1, 2, "Y", "Incorrect / Hallucination"
 if expected_no and is_yes(output):
  return 1, 1, 2, "Y", "Incorrect / Hallucination"

 # 4) Numeric mismatch
 if expected_numbers:
  if not any(num in output_numbers for num in expected_numbers):
   return 1, 1, 1, "Y", "Numeric Error / Hallucination"

 # 5) Similarity / overlap heuristics
 sim = similarity(expected, output)
 exp_tokens = tokenize(expected)
 out_tokens = tokenize(output)
 overlap = len(exp_tokens.intersection(out_tokens)) / max(len(exp_tokens), 1)

 if sim >= 0.80 or overlap >= 0.70:
  correctness, grounding, completeness = 3, 3, 3
  hallucination, failure_type = "N", "Correct"
 elif sim >= 0.55 or overlap >= 0.45:
  correctness, grounding, completeness = 2, 2, 2
  hallucination, failure_type = "N", "Partial"
 else:
  correctness, grounding, completeness = 1, 1, 1
  hallucination = "Y" if len(o) > 0 else "N"
  failure_type = "Hallucination" if hallucination == "Y" else "Incorrect"

 # 6) Catch weak guesses
 if any(word in o for word in ["maybe", "likely", "probably"]) and not expected_unknown:
  failure_type = "Overconfident / Unsupported Guess"
 grounding = min(grounding, 2)

 return correctness, grounding, completeness, hallucination, failure_type

def run_eval(input_csv="dataset.csv", output_csv="evaluated_outputs.csv"):
 df = pd.read_csv(input_csv)

# 🔧 FIX: clean column names (removes hidden spaces)
 df.columns = df.columns.str.strip()

# 🔍 DEBUG: see what columns are actually present
 print("Columns found:", df.columns.tolist())
 print(df.head(3))

# ✅ Ensure required columns exist
 required_cols = ["Question", "Context", "Expected Answer"]
 for col in required_cols:
  if col not in df.columns:
   raise ValueError(f"Missing column: {col}")

 installed = set(get_installed_models())
 active_models = [m for m in MODELS if m in installed]

 if not active_models:
  raise RuntimeError(
   "None of the requested models are installed in Ollama. "
   "Run `ollama pull <model_name>` first."
  )

 print(f"Running models: {active_models}")

 for model in active_models:
  safe = model.replace(":", "_").replace("/", "_")

  output_col = f"output_{safe}"
  c_col = f"correctness_{safe}"
  g_col = f"grounding_{safe}"
  comp_col = f"completeness_{safe}"
  h_col = f"hallucination_{safe}"
  f_col = f"failure_type_{safe}"

  outputs = []
  correctness_scores = []
  grounding_scores = []
  completeness_scores = []
  hallucinations = []
  failure_types = []

  for idx, row in df.iterrows():
   question = str(row["Question"])
   context = str(row["Context"])
   expected = str(row["Expected Answer"]) if "Expected Answer" in df.columns else ""

   try:
    answer = call_ollama(model, question, context)
   except Exception as e:
    answer = f"ERROR: {e}"

   outputs.append(answer)

   if answer.startswith("ERROR:"):
    correctness_scores.append(None)
    grounding_scores.append(None)
    completeness_scores.append(None)
    hallucinations.append(None)
    failure_types.append("Error")
   else:
    correctness, grounding, completeness, hallucination, failure_type = score_output(
    question, context, expected, answer
    )
    correctness_scores.append(correctness)
    grounding_scores.append(grounding)
    completeness_scores.append(completeness)
    hallucinations.append(hallucination)
    failure_types.append(failure_type)

    print(f"[{model}] row {idx + 1}/{len(df)} done")

  df[output_col] = outputs
  df[c_col] = correctness_scores
  df[g_col] = grounding_scores
  df[comp_col] = completeness_scores
  df[h_col] = hallucinations
  df[f_col] = failure_types
 
 df.to_csv(output_csv, index=False)
 print(f"Saved results to {output_csv}")

if __name__ == "__main__":
 run_eval()