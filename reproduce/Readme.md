# Reproduction Entry Points

Run experiments from the repository root:

```bash
python reproduce/run_exp.py ...
```

The entry point loads the repository-local PEFT fork from `peft/src` before any site-packages PEFT installation.

## Hydra Syntax

- `model=t5base` replaces the default model config.
- `+peft=care_lora` adds a PEFT config group.
- `+init=default` adds an initialization config group.
- `++dataset_name=mnli` sets the dataset key.
- `peft.lora_r=16` overrides a field in the selected PEFT config.
- `model.track_cuda_peak=false` overrides a field in the selected model config.

Supported public dataset keys are `sst2`, `cola`, `mrpc`, `mnli`, `qnli`, `boolq`, `cb`, `copa`, `rte`, `wic`, `metamathqa`, `opencodeinstruct`, and `smoltalk`.

## Minimal Smoke Run

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce/run_exp.py \
  model=t5base \
  +peft=care_lora \
  +init=default \
  ++dataset_name=sst2 \
  peft.lora_r=16 \
  peft.lora_alpha=32 \
  seed=0 \
  wandb.project=CARE-LoRA \
  wandb.name=smoke_t5base_sst2_care_lora \
  model.epochs=1 \
  model.per_device_batch_size=2 \
  model.real_batch_size=2 \
  +model.evaluation_strategy=no \
  +model.save_strategy=no \
  model.track_cuda_peak=false
```

## Final Evaluation Flags

For Mistral runs, enable the final generation evaluator that matches the training dataset:

- `++dataset_name=metamathqa model.final_gsm8k_eval=true model.final_gsm_eval_dataset=gsm8k`
- `++dataset_name=opencodeinstruct model.final_humaneval_eval=true`
- `++dataset_name=smoltalk model.final_ifeval_eval=true`

HumanEval uses EvalPlus. IFEval uses `lm-eval`. Both dependencies are included in the environment files.

## Gradient Similarity Diagnostic

Enable the diagnostic by prepending the bootstrap directory to `PYTHONPATH`:

```bash
CARE_LORA_GRAD_SIMILARITY=true \
CARE_LORA_GRAD_FIRST_N_STEPS=100 \
CARE_LORA_GRAD_STOP_AFTER_FIRST_N=true \
PYTHONPATH="reproduce/care_lora_grad_probe_bootstrap:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 python reproduce/run_exp.py \
  model=mistral_7b_v03 \
  +peft=care_lora \
  +init=default \
  ++dataset_name=metamathqa \
  peft.lora_r=16 \
  peft.lora_alpha=32 \
  seed=0 \
  wandb.project=CARE-LoRA \
  wandb.name=grad_similarity \
  model.track_cuda_peak=false
```

Optional diagnostic controls:

- `CARE_LORA_GRAD_PROGRESS_PERCENTAGES=5,10,25,50,75,100`
- `CARE_LORA_GRAD_FIRST_N_STEPS=0`
- `CARE_LORA_GRAD_STOP_AFTER_FIRST_N=false`

Logs are written under `results/<project>/<run_name>/<seed>/logs/`.
