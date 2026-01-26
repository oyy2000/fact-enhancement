import re
import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Configuration
JSON_FILE_PATH = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/large_model_rewrites_unified/paired_responses.json"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

target_words = [
    "from", "From", "above", "step", "Step", "bracket", "So", "Thus", "Therefore", 
    "Since", "First", "first", "Next", "next", "Finally", "Consequently", 
    "Calculate", "calculate", "Let", "let", "Assume", "assume", "Hence", "then", "Then"
]

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Handle potential wrapper keys like 'samples' or 'data'
        for key in ['samples', 'data', 'instances']:
            if key in data:
                return data[key]
        return [data] # Treat as single item list if no list found
    return data

def analyze_samples(data, tokenizer):
    print(f"Analyzing {len(data)} samples...")
    
    before_lengths = []
    after_lengths = []
    
    # Separate counters for \n and \n\n
    before_single_n = []
    before_double_n = []
    
    after_single_n = []
    after_double_n = []
    
    # Word frequency counters
    # Structure: {word: [count_sample_1, count_sample_2, ...]}
    word_counts_before = {word: [] for word in target_words}
    word_counts_after = {word: [] for word in target_words}

    after_longer_count = 0
    total_samples = 0
    
    for item in tqdm(data):
        if 'resp_before' not in item or 'resp_after' not in item:
            continue
            
        resp_before = item['resp_before']
        resp_after = item['resp_after']
        
        # Tokenize
        tokens_before = tokenizer.encode(resp_before, add_special_tokens=False)
        tokens_after = tokenizer.encode(resp_after, add_special_tokens=False)
        
        len_before = len(tokens_before)
        len_after = len(tokens_after)
        
        before_lengths.append(len_before)
        after_lengths.append(len_after)
        
        # Newline counts
        # Count \n\n
        cnt_nn_before = resp_before.count('\n\n')
        cnt_nn_after = resp_after.count('\n\n')
        
        # Count single \n (remove \n\n then count remaining \n)
        cnt_n_before = resp_before.replace('\n\n', '').count('\n')
        cnt_n_after = resp_after.replace('\n\n', '').count('\n')
        
        before_double_n.append(cnt_nn_before)
        before_single_n.append(cnt_n_before)
        
        after_double_n.append(cnt_nn_after)
        after_single_n.append(cnt_n_after)
        
        # Word counts
        for word in target_words:
            # Use regex to find whole words, case-sensitive
            pattern = r'\b' + re.escape(word) + r'\b'
            
            count_b = len(re.findall(pattern, resp_before))
            word_counts_before[word].append(count_b)
            
            count_a = len(re.findall(pattern, resp_after))
            word_counts_after[word].append(count_a)

        # Comparison
        if len_after > len_before:
            after_longer_count += 1
            
        total_samples += 1
        
    print("\n" + "="*40)
    print("ANALYSIS RESULTS")
    print("="*40)
    print(f"Total Samples Processed: {total_samples}")
    print("-" * 20)
    
    print(f"Average Token Length (Before): {np.mean(before_lengths):.2f}")
    print(f"Average Token Length (After) : {np.mean(after_lengths):.2f}")
    print(f"Length Change (After - Before): {np.mean(after_lengths) - np.mean(before_lengths):.2f}")
    print("-" * 20)
    
    print(f"Average Double Newlines (\\n\\n) (Before): {np.mean(before_double_n):.2f}")
    print(f"Average Double Newlines (\\n\\n) (After) : {np.mean(after_double_n):.2f}")
    print(f"Average Single Newlines (\\n)    (Before): {np.mean(before_single_n):.2f}")
    print(f"Average Single Newlines (\\n)    (After) : {np.mean(after_single_n):.2f}")
    print("-" * 20)

    print("Target Word Frequencies (Average count per sample):")
    print(f"{'Word':<15} | {'Before':<10} | {'After':<10} | {'Diff (A-B)':<10}")
    print("-" * 55)
    for word in target_words:
        avg_b = np.mean(word_counts_before[word])
        avg_a = np.mean(word_counts_after[word])
        diff = avg_a - avg_b
        print(f"{word:<15} | {avg_b:<10.4f} | {avg_a:<10.4f} | {diff:<10.4f}")
    print("-" * 20)
    
    print(f"Samples where After is longer than Before (by tokens): {after_longer_count}")
    print(f"Percentage: {after_longer_count / total_samples * 100:.2f}%")
    print("="*40)

def main():
    if not os.path.exists(JSON_FILE_PATH):
        print(f"Error: File not found at {JSON_FILE_PATH}")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer {MODEL_NAME}: {e}")
        return

    data = load_data(JSON_FILE_PATH)
    analyze_samples(data, tokenizer)

if __name__ == "__main__":
    main()
