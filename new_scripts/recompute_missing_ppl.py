#!/usr/bin/env python3
"""
Recompute PPL (and NLL = log(PPL)) for grid entries that are missing it.

Missing entries:
  - All L10 (31 dirs): no chunk files exist at all
  - 4 existing chunks without ppl: chunk_28 (L35/lam-10p0), chunk_49 (L17/lam2p5),
    chunk_73 (L31/BASELINE), chunk_77 (L17/lam2p0)

PPL computation matches 06_PRM_plots_2.py ModelEvaluator.compute_ppl_and_rank:
  - Full-text PPL: tokenize(generated_text), forward pass, CE loss on all tokens,
    PPL = exp(mean_CE).

Usage:
  python new_scripts/recompute_missing_ppl.py --gpus 0,1,2,3,4,5,6,7
  python new_scripts/recompute_missing_ppl.py --dry-run
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, time
import multiprocessing as mp
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss

BASE = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement")
GRID = (
    BASE
    / "lm-evaluation-harness/lm_eval/models"
    / "eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples"
    / "gsm8k_cot_zeroshot"
)
COPY_PRM = (
    BASE
    / "lm-evaluation-harness/lm_eval/models"
    / "eval_grid_less_tokens_3B_lib_manual_same_lib_1000samples copy"
    / "prm_results copy"
)
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


# ── Discover jobs ────────────────────────────────────────────────────────────

def discover_jobs() -> list[dict]:
    """Return list of jobs that need PPL computed."""
    jobs: list[dict] = []

    # 1. Existing chunks missing ppl — we know exactly which ones
    MISSING_PPL_CHUNKS = {
        28: ("Qwen2.5-3B-Instruct", "L35", "lam-10p0"),
        49: ("Qwen2.5-3B-Instruct", "L17", "lam2p5"),
        73: ("Qwen2.5-3B-Instruct", "L31", "BASELINE"),
        77: ("Qwen2.5-3B-Instruct", "L17", "lam2p0"),
    }
    for idx, (m, L, lam) in MISSING_PPL_CHUNKS.items():
        cf = COPY_PRM / f"results_chunk_{idx}.json"
        if cf.exists():
            jobs.append({
                "source": "chunk",
                "chunk_file": str(cf),
                "model": m, "layer": L, "lam": lam,
            })

    # 2. Grid dirs with no chunk — skip for now (only recompute missing ppl in existing chunks)
    # To also compute grid dirs, uncomment below:
    # KNOWN_CHUNK_LAYERS = {"L17", "L31", "L35"}
    # for d in sorted(GRID.iterdir()):
    #     ...

    return jobs


# ── PPL computation ──────────────────────────────────────────────────────────

def compute_ppl_batch(texts: list[str], model, tokenizer, device: str,
                      batch_size: int = 4) -> list[float]:
    """Compute per-sample PPL matching ModelEvaluator.compute_ppl_and_rank."""
    loss_fct = CrossEntropyLoss(reduction="none")
    ppls = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        for text in batch_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=4096).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, :-1, :]
                labels = inputs.input_ids[:, 1:]
                shift_logits = logits.reshape(-1, logits.size(-1))
                shift_labels = labels.reshape(-1)
                loss = loss_fct(shift_logits, shift_labels)
                ppl = torch.exp(loss.mean()).item()
            ppls.append(ppl)
    return ppls


def worker(gpu_id: int, job_queue: mp.Queue, result_queue: mp.Queue):
    """GPU worker: load model once, process jobs from queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[GPU{gpu_id}] Loading model {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(device).eval()
    print(f"[GPU{gpu_id}] Model loaded.", flush=True)

    while True:
        job = job_queue.get()
        if job is None:  # poison pill
            break

        key = f"{job['model']}/{job['layer']}/{job['lam']}"
        t0 = time.time()

        try:
            if job["source"] == "chunk":
                # Load texts from existing chunk
                data = json.load(open(job["chunk_file"]))
                entry = data[job["model"]][job["layer"]][job["lam"]]
                texts = entry.get("generated_text", [])
            else:
                # Load texts from grid jsonl
                texts = []
                with open(job["jsonl"]) as f:
                    for line in f:
                        row = json.loads(line)
                        if row.get("filter") == "strict-match":
                            continue
                        try:
                            resp = row["resps"][0][0].strip()
                            if resp:
                                texts.append(resp)
                        except (KeyError, IndexError):
                            continue

            if not texts:
                print(f"[GPU{gpu_id}] {key}: no texts found, skipping", flush=True)
                result_queue.put((job, [], []))
                continue

            ppls = compute_ppl_batch(texts, model, tokenizer, device)
            nlls = [math.log(p) if p > 0 else float("inf") for p in ppls]

            dt = time.time() - t0
            print(f"[GPU{gpu_id}] {key}: {len(ppls)} samples, "
                  f"mean_ppl={sum(ppls)/len(ppls):.4f}, "
                  f"mean_nll={sum(nlls)/len(nlls):.4f}, "
                  f"{dt:.1f}s", flush=True)

            result_queue.put((job, ppls, nlls))

        except Exception as e:
            print(f"[GPU{gpu_id}] {key}: ERROR {e}", flush=True)
            import traceback; traceback.print_exc()
            result_queue.put((job, [], []))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    jobs = discover_jobs()
    print(f"Total jobs: {len(jobs)}")
    from collections import Counter
    c = Counter(j["source"] for j in jobs)
    print(f"  chunk (missing ppl): {c.get('chunk', 0)}")
    print(f"  grid (no chunk): {c.get('grid', 0)}")

    if args.dry_run:
        for j in jobs:
            print(f"  {j['model']}/{j['layer']}/{j['lam']} ({j['source']})")
        return

    # Start workers
    job_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    workers = []
    for gid in gpu_ids:
        p = mp.Process(target=worker, args=(gid, job_queue, result_queue))
        p.start()
        workers.append(p)

    # Enqueue jobs
    for j in jobs:
        job_queue.put(j)
    # Poison pills
    for _ in gpu_ids:
        job_queue.put(None)

    # Collect results
    results = []
    for _ in range(len(jobs)):
        job, ppls, nlls = result_queue.get()
        results.append((job, ppls, nlls))

    for p in workers:
        p.join()

    # Save results
    print("\n=== Saving results ===", flush=True)

    # 1. Update existing chunks that were missing ppl
    for job, ppls, nlls in results:
        if job["source"] == "chunk" and ppls:
            data = json.load(open(job["chunk_file"]))
            entry = data[job["model"]][job["layer"]][job["lam"]]
            entry["ppl"] = ppls
            entry["nll"] = nlls
            with open(job["chunk_file"], "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Updated {job['chunk_file']}", flush=True)

    # 2. Create new chunk files for grid-only entries
    next_chunk_idx = max(
        (int(re.search(r"(\d+)", f.name).group(1))
         for f in COPY_PRM.glob("results_chunk_*.json")),
        default=-1,
    ) + 1

    for job, ppls, nlls in results:
        if job["source"] == "grid" and ppls:
            # Build a minimal chunk structure matching existing format
            chunk_data = {
                job["model"]: {
                    job["layer"]: {
                        job["lam"]: {
                            "generated_text": [],  # will load from jsonl
                            "ppl": ppls,
                            "nll": nlls,
                        }
                    }
                }
            }
            # Also store generated_text for completeness
            texts = []
            with open(job["jsonl"]) as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("filter") == "strict-match":
                        continue
                    try:
                        resp = row["resps"][0][0].strip()
                        if resp:
                            texts.append(resp)
                    except (KeyError, IndexError):
                        continue
            chunk_data[job["model"]][job["layer"]][job["lam"]]["generated_text"] = texts

            out_path = COPY_PRM / f"results_chunk_{next_chunk_idx}.json"
            with open(out_path, "w") as f:
                json.dump(chunk_data, f, indent=2)
            print(f"  Wrote {out_path}", flush=True)
            next_chunk_idx += 1

    print(f"\nDone. {len([r for r in results if r[1]])} jobs completed successfully.", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
