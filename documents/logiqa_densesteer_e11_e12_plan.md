# LogiQA-based DenseSteer for E11 / E12 (re-run from scratch)

## Why restart

E11/E12 measure **non-math** generalization (GPQA, MMLU, BBH, HotpotQA, …). Using steering vectors trained only on **GSM8K** math CoT is a domain mismatch. This pipeline builds **DenseSteer** (GPT dense rewrite) and **InFamilySteer** (3B vs 7B pairing) from **LogiQA** logical reasoning CoT so the contrastive signal matches the evaluation family.

## Pipeline (ordered)

| Step | Script | Output |
|------|--------|--------|
| 0a | `new_scripts/my_scripts/logiqa_generate_cot.py` (3B, `train` split) | `exps/logiqa_densesteer/samples/Qwen_Qwen2.5-3B-Instruct_logiqa_cot_train.jsonl` |
| 0b | Same with `--model Qwen/Qwen2.5-7B-Instruct` (same `--limit`, same split) | `...7B...jsonl` |
| 1a | `new_scripts/my_scripts/logiqa_pair_infamily.py` | `exps/logiqa_densesteer/paired/Qwen3B_Qwen7B_logiqa.json` |
| 1b | `00_gpt_modification.py` with `--in_jsonl` = 3B jsonl, **`--rewrite_last_n 100000`**, `--prompt_style old`, `--out_json` → | `exps/logiqa_densesteer/gpt_rewrites/.../rewritten_old.json` |
| 2a | `new_scripts/my_scripts/logiqa_extract_steering.py` `--in_path` rewritten JSON | DenseSteer vector under `exps/logiqa_densesteer/vectors/...` |
| 2b | `logiqa_extract_steering.py` `--in_path` paired JSON | InFamily vector under `exps/logiqa_densesteer/vectors/...` |
| 3 | **Calibrate** λ (and optionally layer) on `logiqa` task or held-out split — do **not** reuse GSM8K λ=4.0 blindly. |
| 4 | `new_scripts/my_scripts/e11_e12_logiqa_steer_exp.py` | Fresh E11/E12 under `exps/steer_runs_e11_e12_logiqa/`; calibrate with `--experiments CALIB_LOGIQA` |
| 5 | Update `documents/rebuttal_experiment_data.xlsx` E11/E12 rows (note “LogiQA-calibrated steering”). |

## Stopping old MuSiQue

MuSiQue PIDs from the previous session are gone if processes exited. To stop any stray job: `pkill -f longbench_musique` (only if you intend to kill all such runs).

## HotpotQA fallback (when LogiQA is unavailable)

Set **`CALIB_DATASET=hotpotqa`** when running the driver script. This uses `hotpotqa_generate_cot.py` (LongBench `hotpotqa`), writes under `exps/hotpotqa_densesteer/`, and vectors land in `exps/hotpotqa_densesteer/vectors/...`.

**Caveat:** LongBench HotpotQA often has **no train split**; the generator falls back to `test`/`validation`. Then calibration data can **overlap** E12 HotpotQA evaluation — prefer LogiQA when you can.

**Calibration:** `e11_e12_logiqa_steer_exp.py --steering_suite hotpotqa --experiments CALIB_HOTPOTQA` (use `--batch_size 1` if OOM).

**E11/E12:** same script with `--steering_suite hotpotqa` (outputs default to `exps/steer_runs_e11_e12_hotpotqa/`).

## One-shot driver

`new_scripts/my_scripts/run_logiqa_densesteer_pipeline.sh` — default `CALIB_DATASET=logiqa`. For HotpotQA: `CALIB_DATASET=hotpotqa ./new_scripts/my_scripts/run_logiqa_densesteer_pipeline.sh`. Set `OPENAI_API_KEY` for GPT dense rewrite. Adjust `CUDA_VISIBLE_DEVICES` / `GPU3B` / `GPU7B` / `HOTPOT_LIMIT` as needed.

## Notes

- Calibration uses **train** split by default to avoid leaking LogiQA **test** into vector construction.
- `00_gpt_modification.py` defaults `--rewrite_last_n 2`; you **must** raise it or only the last two lines get GPT rewrites.
- After calibration, adjust lambdas in `make_steer_config()` inside `e11_e12_logiqa_steer_exp.py` (or extend the script with CLI lambdas).
