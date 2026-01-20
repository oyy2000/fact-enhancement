import os
import json
import glob

base_path = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples/gsm8k_cot_zeroshot"

dirs = {
    "lam10p0": "Qwen2.5-3B-Instruct_L17_lam10p0",
    "lam-10p0": "Qwen2.5-3B-Instruct_L17_lam-10p0",
    "BASELINE": "Qwen2.5-3B-Instruct_L17_BASELINE"
}

results = {}

for label, dirname in dirs.items():
    full_dir_path = os.path.join(base_path, dirname, "Qwen__Qwen2.5-3B-Instruct")
    # Find the jsonl file
    jsonl_files = glob.glob(os.path.join(full_dir_path, "samples_*.jsonl"))
    if not jsonl_files:
        print(f"No jsonl file found in {full_dir_path}")
        continue
    
    # assuming the first one is correct if multiple
    jsonl_file = jsonl_files[0] 
    print(f"Processing {label}: {jsonl_file}")

    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            doc_id = entry['doc_id']
            
            if doc_id not in results:
                results[doc_id] = {
                    "doc_id": doc_id,
                    "question": entry.get('doc', {}).get('question', ''),
                    "target": entry.get('target', ''),
                    "responses": {}
                }
            
            # Extract response
            # resps is [[response_string]]
            resp = entry['resps'][0][0] if entry['resps'] and entry['resps'][0] else ""
            exact_match = entry.get('exact_match', 0.0)

            results[doc_id]["responses"][label] = {
                "response": resp,
                "exact_match": exact_match
            }

output_file = "extracted_comparison_samples.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Comparison saved to {output_file}")
