import json
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoTokenizer

# Load tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
except Exception as e:
    print(f"Failed to load tokenizer {model_name}: {e}")
    print("Falling back to word count.")
    tokenizer = None

# Load data
input_file = "extracted_comparison_samples.json"
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit(1)

with open(input_file, "r") as f:
    data = json.load(f)

filtered_data = {}
stats = {
    "lengths": {
        "lam10p0": [],
        "lam-10p0": [],
        "BASELINE": []
    },
    "max_count": {
        "lam10p0": 0,
        "lam-10p0": 0,
        "BASELINE": 0
    },
    "min_count": {
        "lam10p0": 0,
        "lam-10p0": 0,
        "BASELINE": 0
    }
}

for doc_id, item in data.items():
    resps = item["responses"]
    
    # Check if keys exist first to avoid errors
    if "lam10p0" not in resps or "BASELINE" not in resps:
        continue

    # Calculate lengths
    def get_len(text):
        if tokenizer:
            return len(tokenizer.encode(text))
        return len(text.split())

    l_lam10 = get_len(resps["lam10p0"]["response"])
    l_lam_10 = get_len(resps.get("lam-10p0", {"response": ""})["response"])
    l_base = get_len(resps["BASELINE"]["response"])
    
    stats["lengths"]["lam10p0"].append(l_lam10)
    stats["lengths"]["lam-10p0"].append(l_lam_10)
    stats["lengths"]["BASELINE"].append(l_base)
    
    current_lengths = {
        "lam10p0": l_lam10,
        "lam-10p0": l_lam_10,
        "BASELINE": l_base
    }
    
    # Find max and min
    # Handle ties? This takes the first one. Acceptable for now.
    max_model = max(current_lengths, key=current_lengths.get)
    min_model = min(current_lengths, key=current_lengths.get)
    
    stats["max_count"][max_model] += 1
    stats["min_count"][min_model] += 1
        
    is_lam10_correct = resps["lam10p0"]["exact_match"] == 1.0
    is_baseline_wrong = resps["BASELINE"]["exact_match"] == 0.0
    
    if is_lam10_correct and is_baseline_wrong:
        filtered_data[doc_id] = item

print(f"Processed {len(data)} samples.")
print(f"Found {len(filtered_data)} samples where lam10p0 is correct and BASELINE is wrong.")

# Save filtered data
with open("target_samples.json", "w") as f:
    json.dump(filtered_data, f, indent=2)
print("Saved filtered samples to target_samples.json")

# Plotting
models = ["lam10p0", "lam-10p0", "BASELINE"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # blue, orange, green

# 1. Who has the most tokens
plt.figure(figsize=(10, 6))
max_counts = [stats["max_count"][m] for m in models]
plt.bar(models, max_counts, color=colors)
for i, v in enumerate(max_counts):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.title(f"Count of questions where model has the MAX {'token' if tokenizer else 'word'} count\n(All Samples)")
plt.xlabel("Model")
plt.ylabel("Count of Questions")
plt.savefig("max_tokens_distribution.png")
plt.close()

# 2. Who has the least tokens
plt.figure(figsize=(10, 6))
min_counts = [stats["min_count"][m] for m in models]
plt.bar(models, min_counts, color=colors)
for i, v in enumerate(min_counts):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.title(f"Count of questions where model has the MIN {'token' if tokenizer else 'word'} count\n(All Samples)")
plt.xlabel("Model")
plt.ylabel("Count of Questions")
plt.savefig("min_tokens_distribution.png")
plt.close()

# 3. Average Length Comparison
plt.figure(figsize=(10, 6))
avg_lens = [np.mean(stats["lengths"][m]) if stats["lengths"][m] else 0 for m in models]
plt.bar(models, avg_lens, color=colors, alpha=0.7)
for i, v in enumerate(avg_lens):
    plt.text(i, v + 0.5, f"{v:.1f}", ha='center')
plt.title(f"Average Response Length ({'Token' if tokenizer else 'Word'} Count)\n(All Samples)")
plt.ylabel(f"Avg {'Token' if tokenizer else 'Word'} Count")
plt.savefig("avg_length_comparison.png")
plt.close()

print("Plots saved: max_tokens_distribution.png, min_tokens_distribution.png, avg_length_comparison.png")
