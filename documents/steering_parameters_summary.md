# Steering Parameters Summary

Collected from `exps/steer_runs/` through `exps/steer_runs_6/` experiment_status.json files.
Recorded in `11_steer_all_datasets_exp.py` STEERING_CONFIG.

## GPT_REWRITE Mode

Vector source: `gpt_rewrites_unified_new/{model_sanitized}/vectors_50_old/{model_sanitized}_applied/steering_vector.pt`

| Model | Layer | Lambda | Source Runs |
|-------|-------|--------|-------------|
| Qwen/Qwen2.5-3B-Instruct | 6 | 4.0 | steer_runs |
| Qwen/Qwen2.5-1.5B-Instruct | 27 | 2.5 | steer_runs |
| meta-llama/Llama-3.2-1B-Instruct | 8 | -2.0 | steer_runs |
| meta-llama/Llama-3.2-3B-Instruct | 24 | -2.0 | steer_runs |

## LARGE_MODEL Mode

Vector source: `large_model_rewrites_unified_new/{model_sanitized}/vectors_50_paired_{rewrite_model_sanitized}/{model_sanitized}_applied/steering_vector.pt`

| Model | Layer | Lambda | Rewrite Model | Source Runs |
|-------|-------|--------|---------------|-------------|
| Qwen/Qwen2.5-3B-Instruct | 6 | 0.45 | Qwen/Qwen2.5-7B-Instruct | steer_runs_2, steer_runs_3 |
| Qwen/Qwen2.5-1.5B-Instruct | 2 | -0.5 | Qwen/Qwen2.5-7B-Instruct | steer_runs_2 |
| meta-llama/Llama-3.2-1B-Instruct | 14 | -1.0 | meta-llama/Llama-3.1-8B-Instruct | steer_runs_2 |
| meta-llama/Llama-3.2-3B-Instruct | 22 | -0.5 | meta-llama/Llama-3.1-8B-Instruct | steer_runs, steer_runs_2~6 |

## Previous Math Eval Results (exact_match)

### GPT_REWRITE

| Model | L | lam | hendrycks_math_500 | Olympiad | AMC | AIME |
|-------|---|-----|-------------------|----------|-----|------|
| Qwen2.5-3B | 6 | 4.0 | 0.646 | ~0.207 | 0.425 | 0.1 |
| Qwen2.5-1.5B | 27 | 2.5 | ~0.408 | ~0.108 | 0.325 | 0 |
| Llama-3.2-1B | 8 | -2.0 | ~0.156 | ~0.037 | 0.025 | 0 |
| Llama-3.2-3B | 24 | -2.0 | ~0.434 | - | 0.25 | 0 |

### LARGE_MODEL

| Model | L | lam | hendrycks_math_500 | Olympiad | AMC | AIME |
|-------|---|-----|-------------------|----------|-----|------|
| Qwen2.5-3B | 6 | 0.45 | 0.598 | ~0.2 | 0.375 | 0 |
| Qwen2.5-1.5B | 2 | -0.5 | ~0.45 | ~0.115 | 0.2 | ~0.033 |
| Llama-3.2-1B | 14 | -1.0 | 0.142 | ~0.028 | 0 | 0 |
| Llama-3.2-3B | 22 | -0.5 | ~0.444 | ~0.105 | 0.25 | ~0.067 |

## Pending New Evals: MMLU & GPQA

Tasks: `mmlu`, `gpqa_main`

Run command:
```bash
python 11_steer_all_datasets_exp.py --experiment_mode ALL --gpus 0 1 2 3
```

Output: `exps/steer_runs_mmlu_gpqa/`

Total jobs: 4 models x 2 modes x 2 tasks = 16 evaluation runs.

## steer_hf Backend

- Model class: `SteerHFLM` registered as `steer_hf` in lm_eval
- Location: `exps/lm-evaluation-harness/lm_eval/models/steer_hf.py`
- Uses `steering_vectors` library with `SteeringVector.apply(model, multiplier=steer_lambda)`
- model_args format: `pretrained={model},dtype=float16,steer_layer={L},steer_lambda={lam},steer_vec_path={path}`
