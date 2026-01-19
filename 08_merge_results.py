import json
import glob
import os

FOLDERS = [
    # "./prm_out_qwen_family_3",
    "/common/users/sl2148/Public/yang_ouyang/projects/lm-evaluation-harness/lm_eval/models/eval_grid_qwen_family_steered_calibrated_sweep/prm",
]

final_results = {}
seen = set()  # (model, L, lam)

for folder in FOLDERS:
    files = sorted(glob.glob(os.path.join(folder, "results_chunk_*.json")))
    print(f"📂 Scanning {folder}, found {len(files)} files")

    for f in files:
        part = json.load(open(f))
        for model in part:
            final_results.setdefault(model, {})
            for L in part[model]:
                final_results[model].setdefault(L, {})
                for lam, entry in part[model][L].items():
                    key = (model, L, lam)
                    if key in seen:
                        continue  # 🚫 去重
                    seen.add(key)
                    final_results[model][L][lam] = entry

OUT = f"{FOLDERS[0]}/results_merged.json"
with open(OUT, "w") as f:
    json.dump(final_results, f, indent=4)

print(f"🎉 Merged {len(seen)} unique (model, L, λ) → {OUT}")
