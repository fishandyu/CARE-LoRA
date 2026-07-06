# Local PEFT Fork

This directory contains the repository-local PEFT source used by CARE-LoRA.
`reproduce/run_exp.py` prepends `peft/src` to `sys.path`, so experiments use
this fork rather than a site-packages PEFT installation.

The fork keeps the upstream PEFT framework and adds the CARE-LoRA LoRA runtime
path through `LoraConfig(use_care_lora=True)`. Install it in editable mode from
the repository root:

```bash
pip install -e peft
```

The original PEFT project is available at https://github.com/huggingface/peft.
