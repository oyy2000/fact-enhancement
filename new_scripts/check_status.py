#!/usr/bin/env python3
import subprocess, os, json
from pathlib import Path

out = {}

r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                    capture_output=True, text=True)
out["gpu"] = r.stdout.strip().split("\n")

r = subprocess.run(["bash", "-c", "ps aux | grep figure1_sampling_vllm | grep python | grep -v grep"],
                    capture_output=True, text=True)
out["processes"] = r.stdout.strip().split("\n") if r.stdout.strip() else []

log_path = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/figure1_sampling_data/log_32B_vllm.txt")
if log_path.exists():
    content = log_path.read_text(errors="replace")
    lines = content.replace("\r", "\n").split("\n")
    lines = [l for l in lines if l.strip()]
    out["log_tail"] = lines[-10:] if len(lines) > 10 else lines
    out["log_size"] = log_path.stat().st_size
else:
    out["log_tail"] = ["log file not found"]

data_dir = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/figure1_sampling_data")
jsonl = data_dir / "Qwen_Qwen2.5-32B-Instruct" / "gsm8k_samples.jsonl"
out["jsonl_exists"] = jsonl.exists()
if jsonl.exists():
    out["jsonl_lines"] = sum(1 for _ in open(jsonl))

status_out = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_scripts/status_32B.json")
status_out.write_text(json.dumps(out, indent=2))
print("Status written to", status_out)
