import json
import glob

final_results = {}
FOLDER = "./prm_out_qwen_family"
#"./prm_out/"   # ⭐ 你想要的 folder

for f in sorted(glob.glob(f"{FOLDER}/results_chunk_*.json")):
    part = json.load(open(f))
    for model in part:
        final_results.setdefault(model, {})
        for L in part[model]:
            final_results[model].setdefault(L, {})
            for lam in part[model][L]:
                final_results[model][L][lam] = part[model][L][lam]

with open(f"{FOLDER}/results_merged.json", "w") as f:
    json.dump(final_results, f, indent=4)

print(f"🎉 All chunks merged → {FOLDER}/results_merged.json")