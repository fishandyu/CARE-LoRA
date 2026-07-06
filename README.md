# CARE-LoRA

CARE-LoRA implements **Compressed Activation REconstruction for Memory-Efficient LoRA** on top of a repository-local PEFT fork. The method reduces training-time LoRA activation storage by saving a low-rank activation representation and reconstructing the required backward quantities inside custom LoRA operators.

The repository is organized for reproducing the experiments reported in the paper:

- T5-base on GLUE and SuperGLUE classification tasks.
- Mistral-7B-v0.3 on math, code, and instruction-following tasks.
- CARE-LoRA gradient-similarity diagnostics.
- Standard LoRA with Transformer-block gradient checkpointing as a memory baseline.

## Installation

Use the default CUDA 12.1 environment unless your GPU requires the CUDA 12.8 wheel stack.

```bash
conda env create -f environment.yml
conda activate care-lora
pip install -e peft
```

For recent NVIDIA GPUs that require CUDA 12.8 wheels:

```bash
conda env create -f environment_5090.yml
conda activate care-lora
pip install -e peft
```

`reproduce/run_exp.py` always prepends `peft/src` to `sys.path`, so runs use the local PEFT fork.

## Methods

Hydra PEFT configs are selected with `+peft=<name>`:

| Config | Description |
| --- | --- |
| `lora` | Standard LoRA baseline. |
| `care_lora` | CARE-LoRA. |
| `lorafa` | LoRA-FA baseline. |
| `loract` | LoRAct activation-compression baseline. |
| `lora_gradckpt` | Standard LoRA with block-level gradient checkpointing. |
| `pissa`, `dora`, `adalora` | Additional PEFT baselines retained from the local PEFT stack. |

Set either `peft.lora_r=<rank>` or `peft.lora_relative_r=<fraction>` in every PEFT run. The common paper settings use `peft.lora_r=8 peft.lora_alpha=16` for standard LoRA-family baselines and `peft.lora_r=16 peft.lora_alpha=32` for the memory-budget CARE-LoRA runs. Use the same rank/alpha across methods for equal-rank ablations.

## Datasets

Training datasets are downloaded through Hugging Face Datasets and cached under `reproduce/data_cache/` or `reproduce/processed_datasets/`, both ignored by git.

Available `dataset_name` values:

- GLUE: `sst2`, `cola`, `mrpc`, `mnli`, `qnli`.
- SuperGLUE: `boolq`, `cb`, `copa`, `rte`, `wic`.
- Mistral math: `metamathqa`, with final GSM8K/GSM-Hard evaluation controlled by model flags. `gsm8k` is also exposed as a loader for standalone checks.
- Mistral code: `opencodeinstruct`, with final HumanEval evaluation controlled by model flags.
- Mistral instruction following: `smoltalk`, with final IFEval evaluation controlled by model flags. `smoltalk_smol_magpie_ultra` is the exact SmolTalk subset alias.

Optional local dataset/tokenizer overrides:

```bash
export CARE_LORA_METAMATHQA_LOCAL=/path/to/metamathqa_saved_dataset
export CARE_LORA_OPENCODEINSTRUCT_LOCAL=/path/to/opencodeinstruct_saved_dataset
export CARE_LORA_SMOLTALK_LOCAL=/path/to/smoltalk_saved_dataset
export CARE_LORA_HF_MODEL_LOCAL=/path/to/model_or_tokenizer
```

## T5-Base GLUE/SuperGLUE

Run from the repository root. Replace `mnli` with any listed GLUE/SuperGLUE task and replace `care_lora` with a baseline config as needed.

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce/run_exp.py \
  model=t5base \
  +peft=care_lora \
  +init=default \
  ++dataset_name=mnli \
  peft.lora_r=16 \
  peft.lora_alpha=32 \
  seed=0 \
  wandb.project=CARE-LoRA \
  wandb.name=t5base_mnli_care_lora_r16 \
  model.track_cuda_peak=true
```

For a standard LoRA baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce/run_exp.py \
  model=t5base \
  +peft=lora \
  +init=default \
  ++dataset_name=mnli \
  peft.lora_r=8 \
  peft.lora_alpha=16 \
  seed=0 \
  wandb.project=CARE-LoRA \
  wandb.name=t5base_mnli_lora_r8 \
  model.track_cuda_peak=true
```

## Mistral Experiments

Math training on MetaMathQA with final GSM8K evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce/run_exp.py \
  model=mistral_7b_v03 \
  +peft=care_lora \
  +init=default \
  ++dataset_name=metamathqa \
  peft.lora_r=16 \
  peft.lora_alpha=32 \
  seed=0 \
  wandb.project=CARE-LoRA \
  wandb.name=mistral_metamathqa_care_lora_r16 \
  model.final_gsm8k_eval=true \
  model.final_gsm_eval_dataset=gsm8k \
  model.track_cuda_peak=true
```

Code training on OpenCodeInstruct with final HumanEval evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce/run_exp.py \
  model=mistral_7b_v03 \
  +peft=care_lora \
  +init=default \
  ++dataset_name=opencodeinstruct \
  peft.lora_r=16 \
  peft.lora_alpha=32 \
  seed=0 \
  wandb.project=CARE-LoRA \
  wandb.name=mistral_opencode_care_lora_r16 \
  model.final_humaneval_eval=true \
  model.track_cuda_peak=true
```

Instruction tuning on SmolTalk with final IFEval evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce/run_exp.py \
  model=mistral_7b_v03 \
  +peft=care_lora \
  +init=default \
  ++dataset_name=smoltalk \
  peft.lora_r=16 \
  peft.lora_alpha=32 \
  seed=0 \
  wandb.project=CARE-LoRA \
  wandb.name=mistral_smoltalk_care_lora_r16 \
  model.final_ifeval_eval=true \
  model.track_cuda_peak=true
```

Set `model.track_cuda_peak=false` for wall-clock speed measurements where memory instrumentation overhead should be excluded.

## Gradient Similarity

The diagnostic runs the normal CARE-LoRA training path and records exact-vs-CARE-LoRA LoRA-A gradient similarity at selected optimizer steps. It is not used for memory or speed measurements.
Run this diagnostic with one process and `lora_dropout=0`, which is the default in the provided PEFT configs.

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
  wandb.name=mistral_metamathqa_grad_similarity \
  model.track_cuda_peak=false
```

Raw diagnostic logs are written to:

```text
results/<project>/<run_name>/<seed>/logs/care_lora_gradient_similarity/
```

## Diffusion Experiments

Diffusion experiment code is released separately at https://github.com/fishandyu/CARE-LoRA-Diffusion.

## Runtime Outputs

By default, outputs are stored under:

```text
results/<project>/<run_name>/<seed>/
```

Override the root with either Hydra or an environment variable:

```bash
python reproduce/run_exp.py ... artifacts.runtime_dir=/path/to/results
CARE_LORA_RUNTIME_DIR=/path/to/results python reproduce/run_exp.py ...
```

## Acknowledgement

This codebase is built from the LoRA-GA project structure and local PEFT workflow. Please also cite or acknowledge LoRA-GA when this repository is useful to your work:

- LoRA-GA GitHub: https://github.com/Outsider565/LoRA-GA
