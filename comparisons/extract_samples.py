import json
import random

file_path_14b = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family/gsm8k_cot_zeroshot/Qwen2.5-14B-Instruct_L1_BASELINE/Qwen__Qwen2.5-14B-Instruct/samples_gsm8k_cot_zeroshot_2026-01-11T20-35-52.632309.jsonl"
file_path_3b = "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family/gsm8k_cot_zeroshot/Qwen2.5-3B-Instruct_L1_BASELINE/Qwen__Qwen2.5-3B-Instruct/samples_gsm8k_cot_zeroshot_2026-01-11T18-39-08.765074.jsonl"
output_file = "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/comparisons/extracted_comparison_samples.json"

def load_jsonl_by_id(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            obj = json.loads(line)
            # Assuming 'doc_id' is the unique identifier
            if 'doc_id' in obj:
                data[obj['doc_id']] = obj
    return data

def main():
    print(f"Loading 14B samples from {file_path_14b}...")
    data_14b = load_jsonl_by_id(file_path_14b)
    
    print(f"Loading 3B samples from {file_path_3b}...")
    data_3b = load_jsonl_by_id(file_path_3b)

    # Find common IDs where both have exact_match == 1.0 or both have exact_match == 0.0
    common_ids_correct = []
    common_ids_incorrect = []
    
    # Get all common IDs first to iterate over
    potential_ids = list(set(data_14b.keys()) & set(data_3b.keys()))
    
    for doc_id in potential_ids:
        sample_14b = data_14b[doc_id]
        sample_3b = data_3b[doc_id]
        
        # Check if both have exact_match == 1.0
        # The key might be numeric 1.0 or 1, so we handle comparison carefully 
        # based on the head output it looks like a float 0.0
        em_14b = sample_14b.get('exact_match', 0)
        em_3b = sample_3b.get('exact_match', 0)
        
        if em_14b == 1.0 and em_3b == 1.0:
            common_ids_correct.append(doc_id)
        elif em_14b == 0.0 and em_3b == 0.0:
            common_ids_incorrect.append(doc_id)
            
    print(f"Found {len(common_ids_correct)} common samples with exact_match=1.0 for both.")
    print(f"Found {len(common_ids_incorrect)} common samples with exact_match=0.0 for both.")

    selected_ids_correct = sorted(common_ids_correct)[:10]
    selected_ids_incorrect = sorted(common_ids_incorrect)[:10]
    
    selected_ids = selected_ids_correct + selected_ids_incorrect
    print(f"Selecting {len(selected_ids_correct)} correct samples and {len(selected_ids_incorrect)} incorrect samples.")

    extracted_data = []

    for doc_id in selected_ids:
        sample_14b = data_14b[doc_id]
        sample_3b = data_3b[doc_id]

        # Extract question (should be the same in both)
        question = sample_14b.get('doc', {}).get('question', '')
        
        # Extract responses
        # Assuming resps is a list of lists or strings, taking the first one
        resp_14b = sample_14b.get('resps', [['']])[0]
        if isinstance(resp_14b, list):
             resp_14b = resp_14b[0]
             
        resp_3b = sample_3b.get('resps', [['']])[0]
        if isinstance(resp_3b, list):
             resp_3b = resp_3b[0]

        # Extract exact_match (we know they are equal for the selected IDs)
        exact_match = sample_14b.get('exact_match', 0)

        entry = {
            "doc_id": doc_id,
            "question": question,
            "response_14B": resp_14b,
            "response_3B": resp_3b,
            "exact_match": exact_match
        }
        extracted_data.append(entry)

    with open(output_file, 'w') as f:
        json.dump(extracted_data, f, indent=4, ensure_ascii=False)
    
    print(f"Successfully extracted {len(extracted_data)} samples to {output_file}")

if __name__ == "__main__":
    main()
