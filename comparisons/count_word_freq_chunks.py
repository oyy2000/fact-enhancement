import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os

target_words = [
    "from", "From", "above", "step", "Step", "bracket", "So", "Thus", "Therefore", 
    "Since", "First", "first", "Next", "next", "Finally", "Consequently", 
    "Calculate", "calculate", "Let", "let", "Assume", "assume", "Hence", "then", "Then"
]

file_paths = [
    "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out_qwen_family_calibrated_prm_calibrated_split_calibrated/results_chunk_0.json",
    "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out_qwen_family_calibrated_prm_calibrated_split_calibrated/results_chunk_1.json",
    "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out_qwen_family_calibrated_prm_calibrated_split_calibrated/results_chunk_2.json",
    "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out_qwen_family_calibrated_prm_calibrated_split_calibrated/results_chunk_3.json",
    "/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/prm_out_qwen_family_calibrated_prm_calibrated_split_calibrated/results_chunk_4.json"
]

def find_step_texts(data):
    """
    Recursively search for 'step_texts' in the dictionary.
    """
    if isinstance(data, dict):
        if 'step_texts' in data:
            return data['step_texts']
        for key, value in data.items():
            result = find_step_texts(value)
            if result is not None:
                return result
    return None

def count_words_in_text(text, word_counts):
    for word in target_words:
        # Use regex to find whole words, case-sensitive
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, text)
        word_counts[word] += len(matches)

def main():
    # Dictionary to store counts per model
    # Key: model_name, Value: {word: count}
    all_results = {}

    for i, file_path in enumerate(file_paths):
        # Use provided name or default to Chunk X
        current_model_name = f"Chunk {i}"
        
        print(f"Processing {current_model_name} from {file_path}...")
        
        current_counts = {word: 0 for word in target_words}
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Simple check if there's a model name key at top level
            if isinstance(data, dict):
                top_keys = list(data.keys())
                if len(top_keys) == 1:
                     # e.g. "Qwen2.5-0.5B-Instruct" from previous inspect output
                     current_model_name = top_keys[0]

            step_texts = find_step_texts(data)
            
            if step_texts is None:
                print(f"Warning: 'step_texts' not found in {file_path}")
            else:
                for sample_steps in step_texts:
                    if isinstance(sample_steps, list):
                        for step in sample_steps:
                            if isinstance(step, str):
                                count_words_in_text(step, current_counts)
                    elif isinstance(sample_steps, str):
                         count_words_in_text(sample_steps, current_counts)
                
            print(f"Finished processing {file_path}. Found keys: {current_model_name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            current_model_name = f"Error_Chunk_{i}" # Fallback
        
        all_results[current_model_name] = current_counts

    print("All Model Counts:", json.dumps(all_results, indent=2))

    # --- Plotting Grouped Bar Chart ---
    
    models = list(all_results.keys())
    
    x = np.arange(len(target_words))  # the label locations
    width = 0.8 / len(models)  # the width of the bars
    
    plt.figure(figsize=(15, 8))
    
    # Generate a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        counts = [all_results[model][word] for word in target_words]
        offset = width * i - (width * len(models) / 2) + (width / 2)
        plt.bar(x + offset, counts, width, label=model, color=colors[i])

    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency in Step Texts by Model')
    plt.xticks(x, target_words, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    output_image = "/common/users/sl2148/Public/yang_ouyang/word_frequency_plot_grouped.png"
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

    # Save counts to json
    output_json = "/common/users/sl2148/Public/yang_ouyang/word_frequency_counts_grouped.json"
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Counts saved to {output_json}")

if __name__ == "__main__":
    main()
