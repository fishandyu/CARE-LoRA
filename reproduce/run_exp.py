import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Older Transformers/Datasets versions still pass resume_download to
# huggingface_hub. The hub already resumes downloads by default, so suppress the
# warning to keep training logs readable.
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

import sys


def _configure_streaming_stdio_for_pipes() -> None:
    """Use line buffering for stdout/stderr when logs are piped through tee."""
    for _name in ("stdout", "stderr"):
        _stream = getattr(sys, _name, None)
        if _stream is None:
            continue
        try:
            if hasattr(_stream, "reconfigure"):
                _stream.reconfigure(line_buffering=True)
        except (OSError, ValueError, AttributeError):
            pass


_configure_streaming_stdio_for_pipes()

from pathlib import Path

# Always prefer the repository-local PEFT source tree over any site-packages installation.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
_LOCAL_PEFT_SRC = _REPO_ROOT / "peft" / "src"
if _LOCAL_PEFT_SRC.exists():
    local_peft_src_str = str(_LOCAL_PEFT_SRC)
    if sys.path[:1] != [local_peft_src_str]:
        try:
            sys.path.remove(local_peft_src_str)
        except ValueError:
            pass
        sys.path.insert(0, local_peft_src_str)

from peft import get_peft_model, LoraConfig, AdaLoraConfig, TaskType
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import (
    train_text_to_text_model,
    model_inference,
    initialize_text_to_text_model,
    transform_dataset,
    evaluate_gsm8k_test_accuracy,
    evaluate_gsm_hard_accuracy,
    evaluate_humaneval_pass1,
    evaluate_humaneval_plus_pass1,
    evaluate_ifeval,
    CausalLMDataCollator,
    Seq2SeqDataCollatorStripLength,
    _is_trainer_log_main_process,
    _should_disable_tqdm,
)
import json
import re
import math
import inspect
from datasets import load_dataset
import wandb
from data import *
from typing import List
import torch
from copy import deepcopy
import logging
from tqdm import tqdm, trange
from typing import Tuple, List, Dict
from peft.tuners.lora.layer import Linear as LoraLinear
from callback import JsonlMetricsCallback
from hydra.utils import get_original_cwd


log = logging.getLogger(__name__)


def _assert_local_peft_imported():
    import peft

    peft_init = Path(peft.__file__).resolve()
    expected_root = (_REPO_ROOT / "peft" / "src").resolve()
    log.info(f"[run_exp] Using peft from: {peft_init}")
    try:
        peft_init.relative_to(expected_root)
    except ValueError as e:
        raise RuntimeError(
            f"Imported peft from unexpected path: {peft_init}. Expected repository-local source under {expected_root}."
        ) from e


def maybe_hf_login(cfg: DictConfig):
    """
    Optional Hugging Face login.

    - No hard-coded tokens (privacy).
    - Token is read from cfg.hf_token or env vars: HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HUGGINGFACE_TOKEN.
    - If missing or login fails, continue without crashing.
    """
    token = None
    try:
        token = cfg.get('hf_token', None)
    except Exception:
        token = None
    token = token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        return
    try:
        from huggingface_hub import login as huggingface_login
        huggingface_login(token=token, add_to_git_credential=False)
        log.info('Hugging Face login: token provided via config/env (not printed).')
    except Exception as e:
        log.warning(f'Hugging Face login failed (will continue): {e}')


def _safe_wandb_init(
    *,
    project: str,
    run_name: str,
    group: str,
    config: dict,
    wandb_dir: str,
) -> bool:
    """
    Robust wandb init:
      1) online
      2) offline fallback
      3) disabled fallback (never crash training)
    """
    try:
        wandb.init(
            project=project,
            name=run_name,
            group=group,
            config=config,
            dir=wandb_dir,
        )
        return getattr(wandb, "run", None) is not None
    except Exception as e:
        log.warning("[wandb] online init failed, fallback to offline: %s: %s", type(e).__name__, e)

    try:
        wandb.init(
            project=project,
            name=run_name,
            group=group,
            config=config,
            dir=wandb_dir,
            mode="offline",
        )
        log.warning("[wandb] running in offline mode.")
        return getattr(wandb, "run", None) is not None
    except Exception as e:
        log.warning("[wandb] offline init failed, disable wandb: %s: %s", type(e).__name__, e)
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        return False


_GLUE_TASKS = {"mrpc", "cola", "sst2", "qnli", "mnli"}
_SUPERGLUE_TASKS = {"boolq", "cb", "copa", "rte", "wic"}
_DEFAULT_EVAL_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "eval_datasets",
)
_DEFAULT_HUMANEVAL_DATA_ROOT = os.path.join(_DEFAULT_EVAL_DATA_ROOT, "humaneval")


def _metric_float(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return None


def _strip_eval_prefix(key: str) -> str:
    key = str(key)
    return key[5:] if key.startswith("eval_") else key


def _safe_metric_component(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.@+-]+", "_", str(text)).strip("_")


def _public_metric_name(name: str) -> str:
    return {
        "acc": "accuracy",
        "metrics": "metrics",
        "mcc": "matthews_correlation",
        "f1": "f1",
        "macro-f1": "macro_f1",
        "loss": "loss",
    }.get(str(name), _safe_metric_component(name))


def _metric_family(dataset_name: str) -> str:
    task = str(dataset_name).lower()
    if task in _GLUE_TASKS:
        return "glue"
    if task in _SUPERGLUE_TASKS:
        return "superglue"
    if task in {"metamathqa", "gsm8k"}:
        return "math"
    if task == "opencodeinstruct":
        return "code"
    if task in {"smoltalk_smol_magpie_ultra", "smoltalk"}:
        return "instruct"
    return "final"


def _prefer_existing_eval_path(configured_path: str, fallback_path: str) -> str:
    configured_path = str(configured_path or "")
    if configured_path and os.path.exists(configured_path):
        return configured_path
    return fallback_path


def _primary_metric_from_trainer_metrics(dataset_name: str, metrics: Dict) -> Tuple[str, float]:
    task = str(dataset_name).lower()
    priority = {
        "mrpc": ("f1", "metrics", "acc"),
        "cola": ("mcc", "metrics", "acc"),
        "cb": ("macro-f1", "metrics", "acc"),
        "sst2": ("acc", "metrics"),
        "qnli": ("acc", "metrics"),
        "mnli": ("acc", "metrics"),
        "boolq": ("acc", "metrics"),
        "copa": ("acc", "metrics"),
        "rte": ("acc", "metrics"),
        "wic": ("acc", "metrics"),
    }.get(task, ("metrics", "acc", "loss"))
    by_base = {_strip_eval_prefix(k): v for k, v in (metrics or {}).items()}
    for name in priority:
        value = _metric_float(by_base.get(name))
        if value is not None:
            return name, value
    for k, v in by_base.items():
        value = _metric_float(v)
        if value is not None:
            return str(k), value
    return "unknown", 0.0


def _build_final_trainer_eval_payload(dataset_name: str, metrics: Dict, trainer_global_step: int) -> Tuple[Dict[str, float], Dict[str, object]]:
    task = str(dataset_name).lower()
    family = _metric_family(task)
    payload: Dict[str, float] = {}
    summary_extra: Dict[str, object] = {}
    for raw_key, raw_value in (metrics or {}).items():
        value = _metric_float(raw_value)
        if value is None:
            continue
        base_key = _safe_metric_component(_strip_eval_prefix(raw_key))
        payload[f"final_eval/{base_key}"] = value
        payload[f"final_eval/{task}/{base_key}"] = value
        if family in {"glue", "superglue"}:
            public_name = _public_metric_name(_strip_eval_prefix(raw_key))
            payload[f"{family}/{task}_{public_name}"] = value

    primary_name, primary_value = _primary_metric_from_trainer_metrics(task, metrics or {})
    primary_component = _public_metric_name(primary_name)
    payload["final_eval/primary_metric"] = float(primary_value)
    payload[f"final_eval/{task}/primary_metric"] = float(primary_value)
    payload[f"{family}/{task}_{primary_component}"] = float(primary_value)
    payload["final_eval/trainer_global_step"] = float(int(trainer_global_step))
    summary_extra["final_eval/primary_metric_name"] = primary_name
    summary_extra["final_eval/dataset_name"] = task
    summary_extra["final_eval/family"] = family
    return payload, summary_extra


def _wandb_log_and_summarize(wandb_enabled: bool, payload: Dict[str, float], summary_extra: Dict[str, object] = None) -> None:
    if not payload:
        return
    if not (_is_trainer_log_main_process() and wandb_enabled and getattr(wandb, "run", None) is not None):
        return
    try:
        wandb.log(payload)
        wandb.summary.update(payload)
        if summary_extra:
            wandb.summary.update(summary_extra)
    except Exception as e:
        log.warning("[wandb final metrics] log failed: %s: %s", type(e).__name__, e)


def seed_everything(seed: int, *, strict_determinism: bool = False):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Lightweight reproducibility defaults:
    # - cudnn.deterministic=True reduces nondeterministic cuDNN choices.
    # - cudnn.benchmark=False avoids input-shape timing based algorithm changes.
    # strict_determinism=True is available for runs that require stronger global
    # determinism controls.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _cfg_get(cfg: DictConfig, key: str, default=None):
    """OmegaConf DictConfig safe-get helper."""
    try:
        return cfg.get(key, default)
    except Exception:
        return default


_FINAL_GENERATION_EVAL_FLAGS = (
    "final_gsm8k_eval",
    "final_humaneval_eval",
    "final_humaneval_plus_eval",
    "final_ifeval_eval",
)


def _has_final_generation_eval(cfg: DictConfig) -> bool:
    model_cfg = _cfg_get(cfg, "model", None)
    if model_cfg is None:
        return False
    return any(bool(_cfg_get(model_cfg, flag, False)) for flag in _FINAL_GENERATION_EVAL_FLAGS)


def _final_eval_peft_merge_method(cfg: DictConfig) -> str | None:
    peft_cfg = _cfg_get(cfg, "peft", None)
    if peft_cfg is None or not bool(_cfg_get(peft_cfg, "use_peft", False)):
        return None
    if bool(_cfg_get(peft_cfg, "adalora", False)):
        return None
    if bool(_cfg_get(peft_cfg, "use_loraplus", False)):
        return None
    if bool(_cfg_get(peft_cfg, "dora", False)):
        return "dora"
    if bool(_cfg_get(peft_cfg, "pissa", False)):
        return "pissa"
    if bool(_cfg_get(peft_cfg, "use_care_lora", False)):
        return "care_lora"
    if bool(_cfg_get(peft_cfg, "use_loract", False)):
        return "loract"
    if bool(_cfg_get(peft_cfg, "use_lorafa", False)):
        return "lorafa"
    return "lora"


def _maybe_merge_peft_for_final_generation_eval(
    model: torch.nn.Module,
    cfg: DictConfig,
    *,
    wandb_enabled: bool,
) -> torch.nn.Module:
    """Merge LoRA-family PEFT adapters once before post-train generation eval.

    This only affects final generation metrics after training. Trainer eval and the
    training path keep their original PEFT adapter behavior.
    """
    method = _final_eval_peft_merge_method(cfg)
    if method is None:
        return model
    if not _has_final_generation_eval(cfg):
        return model
    if not _is_trainer_log_main_process():
        return model
    if not hasattr(model, "merge_and_unload"):
        log.warning(
            "[final eval PEFT merge] skipped for method=%s: model has no merge_and_unload(); final generation eval will use the adapter path.",
            method,
        )
        return model

    try:
        model.eval()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        log.info(
            "[final eval PEFT merge] merging method=%s adapter into base model before generation eval.",
            method,
        )
        with torch.no_grad():
            merged_model = model.merge_and_unload(progressbar=False, safe_merge=False)
        merged_model.eval()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        log.info(
            "[final eval PEFT merge] merge complete for method=%s; final generation eval will use the merged base model.",
            method,
        )
        _wandb_log_and_summarize(
            wandb_enabled,
            {"final_eval/peft_merged_for_generation": 1.0},
            {"final_eval/peft_merged_method": method},
        )
        return merged_model
    except Exception as e:
        log.warning(
            "[final eval PEFT merge] failed for method=%s; final generation eval will fall back to the unmerged adapter path: %s: %s",
            method,
            type(e).__name__,
            e,
        )
        _wandb_log_and_summarize(
            wandb_enabled,
            {"final_eval/peft_merged_for_generation": 0.0},
            {"final_eval/peft_merged_method": method},
        )
        return model

def _call_dataset_func(dataset_func, cfg: DictConfig):
    """Call a dataset loader with optional keyword arguments from ``cfg.dataset``."""
    dataset_cfg = _cfg_get(cfg, "dataset", None)
    if dataset_cfg is None:
        return dataset_func()
    dataset_kwargs = OmegaConf.to_container(dataset_cfg, resolve=True)
    if not isinstance(dataset_kwargs, dict) or not dataset_kwargs:
        return dataset_func()
    sig = inspect.signature(dataset_func)
    accepted = {
        k: v
        for k, v in dataset_kwargs.items()
        if k in sig.parameters
    }
    return dataset_func(**accepted) if accepted else dataset_func()


@torch.no_grad()
def collect_svd_x_cache(
    model: torch.nn.Module,
    *,
    dataset,
    model_type: str,
    tokenizer,
    max_length: int,
    bsz: int,
    iters: int,
    max_samples_per_layer: int,
    device: torch.device,
):
    """Collect a small CPU cache of LoRA-layer inputs X for SVD-based A init.

    This collects the *input activation to the LoRA linear layer* (same X used by lora_A)
    across a few batches, and stores per-layer samples on CPU as float32.

    Returns:
        dict[str, torch.Tensor]: mapping from module name -> X_samples (M x in_features)
    """

    # The dataset is already preprocessed to contain input_ids/labels via transform_dataset.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz)

    # Build hooks for all LoRA linear modules
    cache: Dict[str, List[torch.Tensor]] = {}
    counts: Dict[str, int] = {}
    hooks = []

    def _make_hook(mod_name: str):
        def _hook(mod, inp, out):
            if not inp:
                return
            x = inp[0]
            if x is None:
                return
            # x: [*, in_features]
            try:
                x = x.detach()
            except Exception:
                return
            if x.numel() == 0:
                return

            in_features = x.shape[-1]
            x2 = x.reshape(-1, in_features)
            # Move to CPU float32 for SVD
            x2 = x2.to(dtype=torch.float32, device="cpu")

            cur = counts.get(mod_name, 0)
            if cur >= max_samples_per_layer:
                return
            remain = max_samples_per_layer - cur
            if x2.shape[0] > remain:
                x2 = x2[:remain].contiguous()
            cache.setdefault(mod_name, []).append(x2)
            counts[mod_name] = cur + x2.shape[0]

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    # Run a few forward passes
    model_was_training = model.training
    model.eval()
    try:
        it = 0
        for batch in dataloader:
            it += 1
            if it > iters:
                break
            # batch values are already tensors (set_transform). Move to device.
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        if model_was_training:
            model.train()

    # Concatenate lists into single tensor per layer
    out_cache: Dict[str, torch.Tensor] = {}
    for k, chunks in cache.items():
        if not chunks:
            continue
        out_cache[k] = torch.cat(chunks, dim=0)
    return out_cache


def find_all_linear_modules(model) -> List[str]:
    r"""
    Collect short ``nn.Linear`` module names for PEFT ``target_modules`` matching.

    Output heads and token embeddings are excluded by default, which matches the
    experimental setup used for both encoder-decoder and causal language models.
    """
    linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)


def find_hidden_state_size(model):
    """
    Resolve the hidden size used by ``lora_relative_r``.

    Prefer model config fields and fall back to the first linear layer if needed.
    """
    base = model
    # PeftModel and wrappers usually expose the backbone config through get_base_model().
    cfg = getattr(base, "config", None)
    if cfg is None and hasattr(base, "get_base_model"):
        try:
            cfg = getattr(base.get_base_model(), "config", None)
        except Exception:
            cfg = None
    if cfg is not None:
        for key in ("hidden_size", "d_model", "n_embd", "dim"):
            v = getattr(cfg, key, None)
            if isinstance(v, int) and v > 0:
                return v
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            return min(module.weight.shape)
    return None


def set_trainable_for_lorafa(model, train_embeddings: bool = False):
    """LoRA-FA training strategy: freeze base + LoRA-A, train only LoRA-B."""
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.requires_grad_(True)
        elif "lora_A" in n:
            p.requires_grad_(False)

    if train_embeddings:
        for n, p in model.named_parameters():
            if ("lm_head" in n) or ("embed_tokens" in n):
                p.requires_grad_(True)


def set_trainable_for_lora(model, train_embeddings: bool = False):
    """
    Standard LoRA training strategy:
      - freeze the base model
      - train LoRA-A and LoRA-B
      - optionally train lm_head / embed_tokens

    We enforce this explicitly here because this repo does several custom re-init / adapter mutations,
    and a silent trainability mismatch would make LoRA collapse into the same effective behavior as LoRA-FA.
    """
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            p.requires_grad_(True)

    if train_embeddings:
        for n, p in model.named_parameters():
            if ("lm_head" in n) or ("embed_tokens" in n):
                p.requires_grad_(True)


def apply_lora_gradient_checkpointing(
    model: torch.nn.Module,
    *,
    fraction: float = 0.9,
) -> Dict[str, object]:
    """Enable checkpointing on native Transformer blocks selected by ``fraction``.

    A LoRA linear cannot usefully checkpoint itself: its input would be a checkpoint
    boundary and would still have to be saved for recomputation.  Instead, reuse the
    backbone's native gradient-checkpointing units (for example, Mistral decoder
    blocks).  The checkpoint scope is each *entire selected block*, so attention,
    MLP, normalization, and LoRA activations inside it are all recomputed in backward.
    LoRA-layer coverage is only the deterministic selection metric; it is not the
    runtime checkpoint granularity.  For Mistral-7B, ``fraction=0.9`` selects 29/32
    complete decoder blocks (which happen to contain 203/224 LoRA linears).

    This function is deliberately applied after the ordinary LoRA injection and
    initialization.  It does not replace LoRA modules, alter their forward, or create
    a separate Trainer/optimizer path.
    """
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"lora_gradckpt_fraction must be in (0, 1], got {fraction!r}.")

    all_lora_ids = {
        id(module) for module in model.modules() if isinstance(module, LoraLinear)
    }
    if not all_lora_ids:
        raise RuntimeError("lora_gradckpt requested, but the model has no LoraLinear modules.")

    checkpoint_units = []
    for name, module in model.named_modules():
        # transformers may attach a ``gradient_checkpointing`` attribute to helper
        # modules which never consume it.  Restrict candidates to the native wrapper
        # base class whose __call__ actually invokes torch checkpoint.
        is_native_checkpoint_unit = any(
            cls.__name__ == "GradientCheckpointingLayer"
            for cls in type(module).__mro__
        )
        if not is_native_checkpoint_unit:
            continue
        contained_ids = {
            id(child) for child in module.modules() if isinstance(child, LoraLinear)
        }
        if contained_ids:
            checkpoint_units.append((name, module, contained_ids))

    if not checkpoint_units:
        raise RuntimeError(
            "lora_gradckpt requested, but no native checkpoint-capable Transformer "
            "blocks containing LoRA layers were found."
        )

    covered_ids = set().union(*(item[2] for item in checkpoint_units))
    if covered_ids != all_lora_ids:
        raise RuntimeError(
            "lora_gradckpt cannot cover every LoRA linear with the backbone's native "
            f"checkpoint units: uncovered_lora_layers={len(all_lora_ids - covered_ids)}."
        )

    # Native checkpoint granularity is a whole Transformer block, not an individual
    # LoRA linear. Choose the deterministic module-order prefix whose unique LoRA
    # coverage is closest to the requested fraction. On Mistral-7B this is equivalent
    # to selecting a block fraction because every decoder block has seven LoRA linears:
    # fraction=0.9 -> 29/32 complete blocks -> 203/224 covered LoRA linears.
    target_lora_count = fraction * len(all_lora_ids)
    prefix_coverage = set()
    prefix_options = []
    for count, item in enumerate(checkpoint_units, start=1):
        prefix_coverage.update(item[2])
        prefix_options.append(
            (abs(len(prefix_coverage) - target_lora_count), -count, set(prefix_coverage))
        )
    _, negative_selected_count, selected_lora_ids = min(prefix_options)
    selected = checkpoint_units[: -negative_selected_count]

    enable_gradient_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
    if enable_gradient_checkpointing is None:
        raise RuntimeError("The backbone does not expose gradient_checkpointing_enable().")
    enable_parameters = inspect.signature(enable_gradient_checkpointing).parameters
    if "gradient_checkpointing_kwargs" not in enable_parameters:
        raise RuntimeError(
            "This transformers version cannot configure non-reentrant gradient "
            "checkpointing, which is required for this PEFT baseline."
        )
    enable_gradient_checkpointing(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # gradient_checkpointing_enable() enables every supported unit.  Narrow it to the
    # deterministic prefix selected above; all non-selected units keep ordinary forward.
    selected_ids = {id(item[1]) for item in selected}
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = id(module) in selected_ids

    enabled_ids = {
        id(module)
        for module in model.modules()
        if hasattr(module, "gradient_checkpointing")
        and bool(module.gradient_checkpointing)
    }
    if enabled_ids != selected_ids:
        raise RuntimeError(
            "Failed to restrict gradient checkpointing to the selected Transformer blocks."
        )

    selected_lora_count = len(selected_lora_ids)
    effective_fraction = selected_lora_count / len(all_lora_ids)
    report: Dict[str, object] = {
        "requested_fraction": fraction,
        "effective_fraction": effective_fraction,
        "checkpointed_blocks": len(selected),
        "total_checkpointable_blocks": len(checkpoint_units),
        "checkpointed_lora_layers": selected_lora_count,
        "total_lora_layers": len(all_lora_ids),
        "checkpointed_block_names": tuple(item[0] for item in selected),
        "use_reentrant": False,
    }
    model._lora_gradckpt_report = report
    log.info(
        "[lora-gradckpt] native block checkpointing installed: "
        "blocks=%d/%d, covered_lora_layers=%d/%d (requested=%.2f%%, effective=%.2f%%), "
        "use_reentrant=False",
        report["checkpointed_blocks"],
        report["total_checkpointable_blocks"],
        report["checkpointed_lora_layers"],
        report["total_lora_layers"],
        100.0 * fraction,
        100.0 * effective_fraction,
    )
    log.info(
        "[lora-gradckpt] selected block range: first=%s, last=%s",
        selected[0][0],
        selected[-1][0],
    )
    return report


def set_trainable_for_care_lora(model, train_embeddings: bool = False):
    """
    CARE-LoRA training strategy:
      - freeze the base model
      - train LoRA-A and LoRA-B with the configured optimizer
      - use the CARE-LoRA fused PEFT path for activation reconstruction
      - optionally train lm_head / embed_tokens
    """
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    for n, p in model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            p.requires_grad_(True)

    if train_embeddings:
        for n, p in model.named_parameters():
            if ("lm_head" in n) or ("embed_tokens" in n):
                p.requires_grad_(True)


def sync_lora_runtime_modes(
    model,
    *,
    use_lorafa: bool,
    use_care_lora: bool = False,
    use_loract: bool = False,
    loract_rank: int = 64,
    care_lora_pinv_lambda: float = 1e-6,
):
    """Force every LoRA linear module to use the intended runtime mode."""
    num_linear = 0
    num_lorafa = 0
    num_care_lora = 0
    num_loract = 0
    for _, module in model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        num_linear += 1
        adapter_names = list(module.lora_A.keys()) if hasattr(module, 'lora_A') else []
        for adapter_name in adapter_names:
            module.use_care_lora[adapter_name] = (
                bool(use_care_lora)
                and (not bool(use_lorafa))
                and (not bool(use_loract))
            )
            if hasattr(module, "use_loract") and isinstance(module.use_loract, dict):
                module.use_loract[adapter_name] = (
                    bool(use_loract)
                    and (not bool(use_lorafa))
                    and (not bool(use_care_lora))
                )
            if hasattr(module, "loract_rank") and isinstance(module.loract_rank, dict):
                module.loract_rank[adapter_name] = int(loract_rank)
            if hasattr(module, "care_lora_pinv_lambda") and isinstance(module.care_lora_pinv_lambda, dict):
                module.care_lora_pinv_lambda[adapter_name] = float(care_lora_pinv_lambda)
            module.use_lorafa[adapter_name] = (
                bool(use_lorafa)
                and (not bool(use_care_lora))
                and (not bool(use_loract))
            )
            if bool(use_loract):
                num_loract += 1
            elif bool(use_care_lora):
                num_care_lora += 1
            elif bool(use_lorafa):
                num_lorafa += 1
        module._lora_runtime_mode = (
            'care_lora' if bool(use_care_lora) else (
                'loract' if bool(use_loract) else ('lorafa' if bool(use_lorafa) else 'lora')
            )
        )
    log.info(
        f"[runtime-mode-sync] LoraLinear modules={num_linear}, forced_care_lora_layers={num_care_lora}, "
        f"forced_loract_layers={num_loract}, forced_lorafa_layers={num_lorafa}"
    )


@torch.no_grad()
def reinit_lora_modules(name, module, init_config, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    lora_r = min(module.lora_A.default.weight.shape)
    a_dim = max(module.lora_A.default.weight.shape)
    b_dim = max(module.lora_B.default.weight.shape)
    if init_config.mode == "simple":
        match init_config.lora_A:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=init_config.lora_A_std
                )
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A.default.weight)
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config.lora_A}")
        match init_config.lora_B:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=init_config.lora_B_std
                )
            case "kaiming":
                torch.nn.init.kaiming_normal_(module.lora_B.default.weight)
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_B.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_B.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_B.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_B.default.weight)
            case _:
                raise ValueError(f"Unknown lora_B initialization: {init_config.lora_B}")
        # Backward-compat alias: some old commands used init.weight="stable".
        scale_mode = init_config.get("scale", None) or init_config.get("weight", None) or ""
        if scale_mode == "stable":
            m, n = module.weight.shape
            gamma = init_config.stable_gamma
            module.lora_B.default.weight.data *= (m**0.25) / gamma**0.5
            module.lora_A.default.weight.data *= (n**0.25) / gamma**0.5
    elif init_config.mode == "svd":
        U, S, V = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
        V = V.T
        m, n = module.weight.shape
        scale_mode = getattr(init_config, "scale", None) or getattr(init_config, "weight", None) or "default"
        if scale_mode == "default":
            S = S / module.scaling["default"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif scale_mode == "stable":
            m, n = module.weight.shape
            gamma = init_config.stable_gamma
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif scale_mode == "unit":
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r]).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :]).contiguous()
            )
        elif scale_mode == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).T.contiguous()
            )
    elif init_config.mode == "svd_x":
        # SVD on the *input activation X* to this LoRA layer, then set A as top-r right singular vectors.
        # This initializes A as an (approx.) orthonormal projector of X into an r-dim subspace.
        svd_x_cache = kwargs.get("svd_x_cache", None)
        if svd_x_cache is None:
            raise ValueError("init.mode=svd_x requires svd_x_cache (collected in run_exp before reinit_lora)")
        X = svd_x_cache.get(name, None)
        if X is None or (not isinstance(X, torch.Tensor)) or X.numel() == 0:
            log.warning(f"[svd_x] No cached X found for layer '{name}', fallback to kaiming for lora_A.")
            torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
        else:
            # X is on CPU float32, shape [M, in_features]
            try:
                # torch.linalg.svd returns U, S, Vh
                _, _, Vh = torch.linalg.svd(X, full_matrices=False)
                A = Vh[:lora_r, :].contiguous()  # [r, in_features]
                module.lora_A.default.weight.data.copy_(A.to(device=module.lora_A.default.weight.device, dtype=module.lora_A.default.weight.dtype))
            except Exception as e:
                log.warning(f"[svd_x] SVD failed for layer '{name}' ({type(e).__name__}: {e}); fallback to kaiming.")
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))

        # Initialize B (default zeros unless specified)
        b_init = init_config.get("lora_B", None) or "zeros"
        if b_init == "gaussian":
            std = init_config.get("lora_B_std", 0.01)
            torch.nn.init.normal_(module.lora_B.default.weight, mean=0.0, std=std)
        elif b_init == "kaiming":
            torch.nn.init.kaiming_normal_(module.lora_B.default.weight)
        elif b_init == "fan_out_kaiming":
            torch.nn.init.kaiming_normal_(module.lora_B.default.weight, mode="fan_out")
        elif b_init == "xavier":
            torch.nn.init.xavier_normal_(module.lora_B.default.weight)
        elif b_init == "zeros":
            torch.nn.init.zeros_(module.lora_B.default.weight)
        elif b_init == "unit":
            torch.nn.init.normal_(module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim**0.5))
        elif b_init == "orthogonal":
            torch.nn.init.orthogonal_(module.lora_B.default.weight)
        else:
            raise ValueError(f"Unknown lora_B initialization for svd_x: {b_init}")
    elif init_config.mode == "gradient":
        named_grad = kwargs["named_grads"]
        grad_name = ".".join(name.split(".")[2:]) + ".weight"
        grads = named_grad[grad_name]
        U, S, V = torch.svd_lowrank(grads.cuda().float(), q=4 * lora_r, niter=4)
        V = V.T
        # set direction
        if init_config.direction == "ArBr":
            B = U[:, 0 : 2 * lora_r : 2]
            A = V[1 : 2 * lora_r : 2, :]
        elif init_config.direction == "A2rBr":
            B = U[:, :lora_r]
            A = V[lora_r : 2 * lora_r, :]
        elif init_config.direction == "ArB2r":
            B = U[:, lora_r : 2 * lora_r]
            A = V[:lora_r, :]
        scaling_factor = module.scaling["default"]
        if init_config.scale == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config.scale == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config.scale == "stable":
            m, n = grads.shape # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config.stable_gamma
            B = B * m**0.25 / gamma**0.5
            A = A * m**0.25 / gamma**0.5
        elif init_config.scale == "weightS":
            _, S, _ = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
            S = S / module.scaling["default"]
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s
        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().cuda())

    with torch.no_grad():
        # consider dtype not in init_config
        if "dtype" not in init_config:
            pass
        elif init_config.dtype == "bf16":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.bfloat16
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.bfloat16
            )
        elif init_config.dtype == "fp32":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.float32
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.float32
            )

        A_w = module.lora_A.default.weight
        B_w = module.lora_B.default.weight
        offset = (B_w @ A_w).to(module.weight.data.device)
        scaling_factor = module.scaling["default"]
        offset *= scaling_factor

        if "norm_clip" in init_config and init_config.norm_clip:
            ratio = torch.max(torch.abs(module.weight.data)) / torch.max(torch.abs(offset))
            if ratio < 1:
                offset *= ratio
                module.lora_A.default.weight.data *= ratio**0.5
                module.lora_B.default.weight.data *= ratio**0.5
                log.warning(f"Clipping offset by {ratio}")

        try:
            module.weight.data -= offset
        except Exception as e:
            raise RuntimeError(f"Failed to subtract LoRA init offset for module {name}: {e}")
        try:
            module.lora_B.default.weight.requires_grad_(True)
        except Exception:
            pass




def reinit_lora(model, init_config, **kwargs):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
        disable=_should_disable_tqdm(),
    ):
        if isinstance(module, LoraLinear):
            reinit_lora_modules(name, module, init_config, **kwargs)

    return model


def cast_lora_ab_dtype(model, init_config, *, reason: str = "LoRA dtype alignment"):
    """Cast only LoRA A/B parameters to the dtype requested by init_config."""
    if "dtype" not in init_config:
        return model
    dtype_name = str(init_config.dtype).lower()
    if dtype_name == "bf16":
        target_dtype = torch.bfloat16
    elif dtype_name == "fp32":
        target_dtype = torch.float32
    else:
        return model

    n_cast = 0
    with torch.no_grad():
        for module in model.modules():
            try:
                lora_a = module.lora_A.default.weight
                lora_b = module.lora_B.default.weight
            except Exception:
                continue
            if lora_a.dtype != target_dtype:
                lora_a.data = lora_a.data.to(target_dtype)
                n_cast += 1
            if lora_b.dtype != target_dtype:
                lora_b.data = lora_b.data.to(target_dtype)
                n_cast += 1
    log.info("[%s] cast LoRA A/B tensors to %s: changed=%d", reason, target_dtype, n_cast)
    return model


def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_gradient(
    model, dataset, batch_size: int = 4, collate_fn=None
) -> Dict[str, List[torch.Tensor]]:
    r"""
    Estimate the gradient of the model on the given dataset
    """
    log.info("Estimating gradient")
    model.train()
    named_grads = {}
    hooks = []
    for name, param in model.named_parameters():
        hook = param.register_hook(get_record_gradient_hook(model, named_grads))
        hooks.append(hook)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    num = 0
    for batch in tqdm(
        dataloader,
        desc="Estimating gradient",
        disable=_should_disable_tqdm(),
    ):
        num += 1
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        get_record_gradient_hook(model, named_grads)(None)  # get gradient of last layer
        # make sure the gradient is cleared
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad = None
    for n, g in named_grads.items():
        named_grads[n] /= num
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    return named_grads





def _fix_single_node_ddp_master_addr_for_c10d() -> None:
    """
    Use IPv4 loopback for single-node torchrun when MASTER_ADDR is ambiguous.

    Multi-node training must provide a routable MASTER_ADDR; this helper leaves
    NNODES!=1 launches unchanged.
    """
    import socket

    try:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return
    if ws <= 1:
        return
    nnodes_raw = os.environ.get("NNODES", os.environ.get("PET_NNODES", "1"))
    try:
        nnodes = int(nnodes_raw)
    except ValueError:
        nnodes = 1
    if nnodes != 1:
        return

    addr = (os.environ.get("MASTER_ADDR") or "").strip()
    host = socket.gethostname()
    try:
        fqdn = socket.getfqdn()
    except Exception:
        fqdn = host

    ambiguous_local_name = ("." not in addr and addr.lower() != "localhost") if addr else False
    force_loopback = (
        (not addr)
        or addr in (host, fqdn)
        or ambiguous_local_name
        or addr.lower() == "localhost"
    )
    if force_loopback:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        log.info(
            "[dist] single-node DDP (WORLD_SIZE=%s): set MASTER_ADDR=127.0.0.1 "
            "to avoid hostname/localhost IPv6 resolution warnings from c10d.",
            ws,
        )


def _build_runtime_dir(cfg, base_dir: str, project: str, run_name: str, seed: int) -> str:
    runtime_root = None
    try:
        runtime_root = cfg.artifacts.get("runtime_dir", None)
    except Exception:
        runtime_root = None
    runtime_root = runtime_root or os.environ.get(
        "CARE_LORA_RUNTIME_DIR",
        os.path.join(base_dir, "results"),
    )
    safe_run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_name))
    runtime_dir = os.path.join(runtime_root, str(project), safe_run_name, str(seed))
    os.makedirs(runtime_dir, exist_ok=True)
    return runtime_dir

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_exp(cfg: DictConfig):
    BASE_DIR = get_original_cwd()
    _fix_single_node_ddp_master_addr_for_c10d()

    log.info(OmegaConf.to_yaml(cfg))
    seed_everything(
        cfg.seed,
        strict_determinism=bool(_cfg_get(cfg, "strict_determinism", False)),
    )
    maybe_hf_login(cfg)
    _assert_local_peft_imported()
    model_name = cfg.model.name
    model_type = cfg.model.type
    dataset_name = cfg.dataset_name
    dataset_func = DATASET_MAP[dataset_name]
    use_peft = cfg.peft.use_peft
    # Keep optional PEFT variants disabled unless the YAML or CLI enables them.
    if_use_rslora = bool(_cfg_get(cfg.peft, "use_rslora", False))
    lora_r = cfg.peft.lora_r
    lora_relative_r = cfg.peft.lora_relative_r
    lora_target_modules = cfg.peft.lora_target_modules
    train_embeddings = cfg.peft.train_embeddings
    use_lorafa = bool(cfg.peft.get("use_lorafa", False))
    use_care_lora = bool(cfg.peft.get("use_care_lora", False))
    use_loract = bool(cfg.peft.get("use_loract", False))
    use_lora_gradckpt = bool(cfg.peft.get("use_lora_gradckpt", False))
    lora_gradckpt_fraction = float(cfg.peft.get("lora_gradckpt_fraction", 0.9))
    full_gradient_checkpointing = bool(
        cfg.model.get(
            "gradient_checkpointing",
            _cfg_get(cfg, "gradient_checkpointing", False),
        )
    )
    loract_rank = int(cfg.peft.get("loract_rank", cfg.peft.get("loract_k", 64)))
    care_lora_pinv_lambda = float(cfg.peft.get("care_lora_pinv_lambda", 1e-6))
    use_pissa = bool(cfg.peft.get("pissa", False))
    init_lora_weights = cfg.peft.get(
        "init_lora_weights",
        "pissa_niter_4" if use_pissa else True,
    )
    if use_loract:
        if use_lorafa or use_care_lora:
            raise ValueError(
                "use_loract=True cannot be combined with use_lorafa/use_care_lora."
            )
        if cfg.peft.get("dora", False):
            raise ValueError("use_loract=True is incompatible with DoRA in this repo.")
        if cfg.peft.get("adalora", False):
            raise ValueError("use_loract=True is incompatible with AdaLoRA in this repo.")
        if getattr(cfg.peft, "use_loraplus", False):
            raise ValueError("use_loract=True should not be combined with use_loraplus in this repo.")
        if loract_rank < 1:
            raise ValueError("loract_rank must be >= 1.")

    if use_lorafa:
        if cfg.peft.get("dora", False):
            raise ValueError("use_lorafa=True is incompatible with DoRA in this repo.")
        if cfg.peft.get("adalora", False):
            raise ValueError("use_lorafa=True is incompatible with AdaLoRA in this repo.")

    if use_care_lora:
        if use_lorafa:
            raise ValueError("use_care_lora=True cannot be combined with use_lorafa.")
        if cfg.peft.get("dora", False):
            raise ValueError("use_care_lora=True is incompatible with DoRA in this repo.")
        if cfg.peft.get("adalora", False):
            raise ValueError("use_care_lora=True is incompatible with AdaLoRA in this repo.")
        if getattr(cfg.peft, "use_loraplus", False):
            raise ValueError("use_care_lora=True should not be combined with use_loraplus in this repo.")

    if use_lora_gradckpt:
        incompatible_modes = (
            use_lorafa
            or use_care_lora
            or use_loract
            or use_pissa
            or bool(cfg.peft.get("dora", False))
            or bool(cfg.peft.get("adalora", False))
            or bool(cfg.peft.get("use_loraplus", False))
        )
        if incompatible_modes:
            raise ValueError(
                "use_lora_gradckpt=True is a standard-LoRA-only baseline and cannot "
                "be combined with LoRA-FA/CARE-LoRA/LoRAct/PiSSA/DoRA/AdaLoRA/LoRA+."
            )
        if not (0.0 < lora_gradckpt_fraction <= 1.0):
            raise ValueError(
                "lora_gradckpt_fraction must be in (0, 1], got "
                f"{lora_gradckpt_fraction!r}."
            )
        if full_gradient_checkpointing:
            raise ValueError(
                "peft.use_lora_gradckpt and model.gradient_checkpointing cannot both "
                "be enabled: the latter checkpoints all supported Transformer blocks."
            )

    if use_pissa:
        if not (isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa")):
            raise ValueError(
                "PiSSA requires peft.init_lora_weights to be 'pissa' or 'pissa_niter_[number of iters]'. "
                f"Got {init_lora_weights!r}."
            )
        if use_lorafa or use_care_lora or use_loract:
            raise ValueError("PiSSA is a standalone LoRA initialization here; do not combine it with LoRA-FA/CARE-LoRA/LoRAct modes.")
        if cfg.peft.get("dora", False):
            raise ValueError("PiSSA cannot be combined with DoRA in this repo. Use +peft=pissa or +peft=dora, not both.")
        if cfg.peft.get("adalora", False):
            raise ValueError("PiSSA cannot be combined with AdaLoRA in this repo.")

    if cfg.dry_run:
        return
    if use_peft:
        assert (lora_r is not None) ^ (
            lora_relative_r is not None
        ), "Please specify lora_r or lora_relative_r"
        assert lora_target_modules is not None, "Please specify lora_target_modules"
    else:
        lora_r = None
        lora_target_modules = None
        lora_relative_r = None
        train_embeddings = True
    use_flash_attention_2 = bool(_cfg_get(cfg.model, "use_flash_attention_2", False))
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "use_peft": use_peft,
        "lora_r": lora_r,
        "lora_target_modules": str(lora_target_modules),
        "lora_relative_r": lora_relative_r,
        "train_embeddings": train_embeddings,
        "use_lorafa": use_lorafa,
        "use_care_lora": use_care_lora,
        "use_loract": use_loract,
        "loract_rank": loract_rank,
        "care_lora_pinv_lambda": care_lora_pinv_lambda,
        "use_pissa": use_pissa,
        "init_lora_weights": init_lora_weights,
        "use_flash_attention_2": use_flash_attention_2,
    }
    if use_lora_gradckpt:
        config.update(
            {
                "use_lora_gradckpt": True,
                "lora_gradckpt_fraction": lora_gradckpt_fraction,
            }
        )
    if cfg.wandb.name:
        name = cfg.wandb.name
    else:
        name = "_".join([f"{k}={v}" for k, v in config.items()])
    project = cfg.wandb.project
    run_name = cfg.wandb.name if cfg.wandb.name else name

    # Runtime outputs are kept under results/ by default and can be redirected
    # with CARE_LORA_RUNTIME_DIR.
    local_run_dir = _build_runtime_dir(cfg, BASE_DIR, project, run_name, cfg.seed)
    local_log_dir = os.path.join(local_run_dir, "logs")
    os.makedirs(local_log_dir, exist_ok=True)

    # Python logging -> run.log on the main process.
    log_file = os.path.join(local_log_dir, "run.log")
    if _is_trainer_log_main_process():
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_file
                for h in root_logger.handlers):
            root_logger.addHandler(fh)

    # Store the resolved config beside run logs for reproducibility.
    if _is_trainer_log_main_process():
        with open(os.path.join(local_log_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))

    # Keep W&B local files inside the run log directory.
    wandb_dir = os.path.join(local_log_dir)

    # Initialize W&B only on the main process.
    wandb_enabled = False
    if _is_trainer_log_main_process():
        wandb_enabled = _safe_wandb_init(
            project=project,
            run_name=run_name,
            group=cfg.dataset_name,
            config=config,
            wandb_dir=wandb_dir,
        )

    train_set, val_set, _ = _call_dataset_func(dataset_func, cfg)
    model, tokenizer = initialize_text_to_text_model(
        model_name,
        model_type,
        cfg.model.bf16,
        use_peft=False,
        use_flash_attention_2=use_flash_attention_2,
    )
    additional_kwargs = {}
    if use_peft and cfg.init.mode == "gradient":
        if isinstance(train_set, list):
            temp_set = train_set[: cfg.init.bsz * cfg.init.iters]
        else:
            temp_set = train_set.select(range(cfg.init.bsz * cfg.init.iters))
        temp_set = transform_dataset(
            model_type=model_type,
            dataset=temp_set,
            tokenizer=tokenizer,
            max_length=cfg.init.max_length,
        )
        if model_type == "CausalLM":
            gradient_collator = CausalLMDataCollator(tokenizer)
        else:
            gradient_collator = Seq2SeqDataCollatorStripLength(tokenizer, model)
        named_grads = estimate_gradient(
            model,
            temp_set,
            cfg.init.bsz,
            collate_fn=gradient_collator,
        )
        additional_kwargs["named_grads"] = named_grads

    # For input-SVD initialization (svd_x), we will collect layer-wise input activations X
    # after PEFT injection (so LoRA modules exist) and before reinit_lora.

    if lora_target_modules == "all":
        lora_target_modules = find_all_linear_modules(model)
    else:
        lora_target_modules = list(lora_target_modules) if lora_target_modules else []
    if lora_relative_r is not None:
        hidden_size = find_hidden_state_size(model)
        lora_r = int(hidden_size * lora_relative_r)
        log.info(f"lora_r is set to {hidden_size} * {lora_relative_r} = {lora_r}")
    if use_peft and cfg.peft.get("dora", False):
        log.info("Using Dora")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            use_rslora=if_use_rslora,
            use_dora=True,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        sync_lora_runtime_modes(
            model,
            use_lorafa=use_lorafa,
            use_care_lora=False,
            care_lora_pinv_lambda=care_lora_pinv_lambda,
        )
        if use_lorafa:
            set_trainable_for_lorafa(model, train_embeddings=train_embeddings)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft and cfg.peft.get("adalora", False):
        log.info("Using AdaLora")
        _adalora_task = (
            TaskType.CAUSAL_LM if str(model_type) == "CausalLM" else TaskType.SEQ_2_SEQ_LM
        )
        peft_config = AdaLoraConfig(
            task_type=_adalora_task,
            target_r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            total_step=int(len(train_set)/cfg.model.real_batch_size)*cfg.model.epochs,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        sync_lora_runtime_modes(
            model,
            use_lorafa=use_lorafa,
            use_care_lora=False,
            care_lora_pinv_lambda=care_lora_pinv_lambda,
        )
        if use_lorafa:
            set_trainable_for_lorafa(model, train_embeddings=train_embeddings)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            init_lora_weights=init_lora_weights,
            use_rslora=if_use_rslora,
            use_lorafa=use_lorafa,
            use_care_lora=use_care_lora,
            care_lora_pinv_lambda=care_lora_pinv_lambda,
            use_loract=use_loract,
            loract_rank=loract_rank,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        sync_lora_runtime_modes(
            model,
            use_lorafa=use_lorafa,
            use_care_lora=use_care_lora,
            use_loract=use_loract,
            loract_rank=loract_rank,
            care_lora_pinv_lambda=care_lora_pinv_lambda,
        )
        if use_care_lora or use_loract:
            set_trainable_for_care_lora(model, train_embeddings=train_embeddings)
        elif use_lorafa:
            set_trainable_for_lorafa(model, train_embeddings=train_embeddings)
        else:
            set_trainable_for_lora(model, train_embeddings=train_embeddings)

        # ===== svd_x init: collect per-layer X caches (CPU) =====
        if (not use_pissa) and cfg.init.mode == "svd_x":
            # Select a small subset for estimation
            bsz = int(_cfg_get(cfg.init, "bsz", 1))
            iters = int(_cfg_get(cfg.init, "iters", 8))
            max_length = int(_cfg_get(cfg.init, "max_length", 512))
            n_examples = max(1, bsz * iters)
            if isinstance(train_set, list):
                temp_set = train_set[:n_examples]
            else:
                temp_set = train_set.select(range(min(n_examples, len(train_set))))

            temp_set = transform_dataset(
                model_type=model_type,
                dataset=temp_set,
                tokenizer=tokenizer,
                max_length=max_length,
            )

            max_samples_per_layer = int(
                _cfg_get(cfg.init, "max_samples_per_layer", _cfg_get(cfg.init, "max_samples", 8192))
            )
            log.info(
                f"[svd_x] Collecting LoRA input activations: bsz={bsz}, iters={iters}, max_length={max_length}, "
                f"max_samples_per_layer={max_samples_per_layer}"
            )
            svd_x_cache = collect_svd_x_cache(
                model,
                dataset=temp_set,
                model_type=model_type,
                tokenizer=tokenizer,
                max_length=max_length,
                bsz=bsz,
                iters=iters,
                max_samples_per_layer=max_samples_per_layer,
                device=model.device,
            )
            additional_kwargs["svd_x_cache"] = svd_x_cache

        if use_pissa:
            # PiSSA initialization is performed inside PEFT's get_peft_model(...):
            # it decomposes the base weight, writes the residual back to the frozen
            # base layer, and initializes LoRA A/B from the principal components.
            # Calling this repo's reinit_lora() afterwards would overwrite A/B and
            # subtract a second adapter offset from the residual weight. Cast only
            # LoRA A/B afterward so PiSSA differs from LoRA by initialization, not
            # by adapter dtype under bf16 base-model runs.
            log.info("[PiSSA] Using PEFT init_lora_weights=%s; skipped repo reinit_lora().", init_lora_weights)
            cast_lora_ab_dtype(model, cfg.init, reason="PiSSA dtype alignment")
        else:
            reinit_lora(model, cfg.init, **additional_kwargs)
        sync_lora_runtime_modes(
            model,
            use_lorafa=use_lorafa,
            use_care_lora=use_care_lora,
            use_loract=use_loract,
            loract_rank=loract_rank,
            care_lora_pinv_lambda=care_lora_pinv_lambda,
        )
        if use_care_lora or use_loract:
            set_trainable_for_care_lora(model, train_embeddings=train_embeddings)
        elif use_lorafa:
            set_trainable_for_lorafa(model, train_embeddings=train_embeddings)
        else:
            set_trainable_for_lora(model, train_embeddings=train_embeddings)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    else:
        # full finetune
        all_param = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rate = {
            "trainable_params": trainable_params,
            "orig_params": all_param,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": 1,
        }
    if use_lora_gradckpt:
        apply_lora_gradient_checkpointing(
            model,
            fraction=lora_gradckpt_fraction,
        )
    log.info(rate)
    # Log parameter counts to the W&B summary on the main process.
    if _is_trainer_log_main_process() and wandb_enabled and getattr(wandb, "run", None) is not None:
        wandb.summary.update(rate)

    metrics_jsonl = os.path.join(local_log_dir, "metrics.jsonl")
    if os.path.exists(metrics_jsonl) and _is_trainer_log_main_process():
        os.remove(metrics_jsonl)
    jsonl_cb = JsonlMetricsCallback(
        metrics_jsonl,
        log_every_n_steps=max(1, int(cfg.model.get("logging_steps", 10))),
    )

    training_loop = train_text_to_text_model
    model, trainer_global_step, final_trainer_metrics = training_loop(
        f"{project}/{run_name}",
        train_set,
        val_set,
        model,
        tokenizer,
        model_type,
        num_train_epochs=cfg.model.epochs,
        per_device_batch_size=cfg.model.per_device_batch_size,
        real_batch_size=cfg.model.real_batch_size,
        bf16=cfg.model.bf16,
        fp16=cfg.model.get("fp16", False),
        eval_epochs=cfg.model.eval_epochs,
        early_stopping_patience=cfg.model.early_stopping_patience,

        evaluation_strategy=cfg.model.get("evaluation_strategy", "steps"),
        # Checkpoint saving is disabled by default unless the model config or CLI enables it.
        save_strategy=cfg.model.get("save_strategy", "no"),
        save_steps=cfg.model.get("save_steps", None),
        save_total_limit=cfg.model.get("save_total_limit", 1),
        load_best_model_at_end=cfg.model.get("load_best_model_at_end", False),
        run_final_trainer_eval=cfg.model.get("run_final_trainer_eval", True),
        **(
            {"do_eval": cfg.model.do_eval}
            if "do_eval" in cfg.model
            else {}
        ),
        enable_early_stopping=cfg.model.get("enable_early_stopping", False),
        **(
            {"metric_for_best_model": cfg.model.metric_for_best_model}
            if "metric_for_best_model" in cfg.model
            else {}
        ),
        **(
            {"greater_is_better": cfg.model.greater_is_better}
            if "greater_is_better" in cfg.model
            else {}
        ),

        max_length=cfg.model.max_length,
        logging_steps=cfg.model.logging_steps,
        use_loraplus=cfg.peft.use_loraplus,
        loraplus_lr_ratio=cfg.peft.loraplus_lr_ratio,
        learning_rate=cfg.model.learning_rate,
        optim=cfg.model.get("optim", "adamw_torch"),
        weight_decay=cfg.model.get("weight_decay", 0.0),
        max_grad_norm=cfg.model.get("max_grad_norm", 1.0),
        adam_beta1=cfg.model.get("adam_beta1", 0.9),
        adam_beta2=cfg.model.get("adam_beta2", 0.999),
        adam_epsilon=cfg.model.get("adam_epsilon", 1e-8),
        warmup_ratio=cfg.model.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.model.get("lr_scheduler_type", "cosine"),
        gradient_checkpointing=full_gradient_checkpointing,
        seed=cfg.seed,
        wandb_enabled=wandb_enabled,
        callbacks=[jsonl_cb],
        dataset_name=str(cfg.dataset_name),
        metric_task=cfg.dataset_name,
        use_lorafa=use_lorafa,
        use_care_lora=use_care_lora,
        use_loract=use_loract,
        use_dora=bool(cfg.peft.get("dora", False)),
        loract_rank=loract_rank,
        runtime_dir=local_run_dir,
        dataloader_num_workers=cfg.model.get("dataloader_num_workers", 4),
        dataloader_persistent_workers=cfg.model.get("dataloader_persistent_workers", True),
        group_by_length=cfg.model.get("group_by_length", True),
        speed_over_logging=cfg.model.get("speed_over_logging", False),
        enable_compute_metrics=cfg.model.get("enable_compute_metrics", None),
        # Peak-memory instrumentation records saved-tensor and CUDA allocator statistics.
        track_cuda_peak=cfg.model.get("track_cuda_peak", True),
        cuda_allow_tf32=cfg.model.get("cuda_allow_tf32", True),
    )

    if _is_trainer_log_main_process():
        final_eval_report = {
            "dataset_name": str(cfg.dataset_name),
            "trainer_global_step": int(trainer_global_step),
            "trainer_final_metrics": final_trainer_metrics,
        }
        try:
            with open(
                os.path.join(local_log_dir, "final_trainer_eval.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(final_eval_report, f, indent=2, ensure_ascii=False)
        except Exception as e_json:
            log.warning("Failed to write final_trainer_eval.json: %s", e_json)
        if final_trainer_metrics:
            final_payload, final_summary = _build_final_trainer_eval_payload(
                str(cfg.dataset_name),
                final_trainer_metrics,
                int(trainer_global_step),
            )
            log.info("[final trainer eval wandb] %s", final_payload)
            _wandb_log_and_summarize(wandb_enabled, final_payload, final_summary)

    model = _maybe_merge_peft_for_final_generation_eval(
        model,
        cfg,
        wandb_enabled=wandb_enabled,
    )

    # Run GSM8K / GSM-Hard generation evaluation after training when requested.
    if bool(_cfg_get(cfg.model, "final_gsm8k_eval", False)) and _is_trainer_log_main_process():
        if str(cfg.model.type) != "CausalLM":
            log.warning(
                "[final_gsm8k_eval] skipped: only the CausalLM path is implemented, model.type=%s",
                cfg.model.type,
            )
        else:
            try:
                final_gsm_dataset = str(_cfg_get(cfg.model, "final_gsm_eval_dataset", "gsm8k")).lower()
                evaluator = evaluate_gsm_hard_accuracy if final_gsm_dataset in ("gsm-hard", "gsm_hard", "gsmhard") else evaluate_gsm8k_test_accuracy
                acc, n_ex, n_ok = evaluator(
                    model,
                    tokenizer,
                    str(cfg.model.type),
                    max_source_length=int(_cfg_get(cfg.model, "final_gsm8k_max_source_length", 512)),
                    max_new_tokens=int(_cfg_get(cfg.model, "final_gsm8k_max_new_tokens", 512)),
                )
                log.info(
                    "[final %s test] accuracy=%.6f (%d/%d) | trainer_global_step=%s",
                    final_gsm_dataset,
                    acc,
                    n_ok,
                    n_ex,
                    trainer_global_step,
                )
                gsm8k_report = {
                    "metric_name": f"{final_gsm_dataset}_numeric_accuracy",
                    "description": "GSM8K extracts the final numeric answer after the official #### marker when available, otherwise it falls back to final-number matching. GSM-Hard compares the final generated number with the target value.",
                    "dataset": final_gsm_dataset,
                    "accuracy": float(acc),
                    "num_correct": int(n_ok),
                    "num_examples": int(n_ex),
                    "trainer_global_step": int(trainer_global_step),
                    "max_source_length": int(
                        _cfg_get(cfg.model, "final_gsm8k_max_source_length", 512)
                    ),
                    "max_new_tokens": int(
                        _cfg_get(cfg.model, "final_gsm8k_max_new_tokens", 512)
                    ),
                }
                try:
                    with open(
                        os.path.join(local_log_dir, f"{final_gsm_dataset.replace('-', '_')}_test_eval.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(gsm8k_report, f, indent=2, ensure_ascii=False)
                except Exception as e_json:
                    log.warning("Failed to write gsm8k_test_eval.json: %s", e_json)

                final_gsm_component = _safe_metric_component(final_gsm_dataset)
                _wandb_log_and_summarize(
                    wandb_enabled,
                    {
                        f"math/{final_gsm_component}_accuracy": float(acc),
                        f"math/{final_gsm_component}_num_correct": float(n_ok),
                        f"math/{final_gsm_component}_num_examples": float(n_ex),
                    },
                )
            except Exception as e:
                log.warning(
                    "[final_gsm8k_eval] failed after training: %s: %s",
                    type(e).__name__,
                    e,
                )

    if bool(_cfg_get(cfg.model, "final_humaneval_eval", False)) and _is_trainer_log_main_process():
        try:
            he_report = evaluate_humaneval_pass1(
                model,
                tokenizer,
                str(cfg.model.type),
                output_dir=local_log_dir,
                max_source_length=int(_cfg_get(cfg.model, "final_humaneval_max_source_length", 1024)),
                max_new_tokens=int(_cfg_get(cfg.model, "final_humaneval_max_new_tokens", 512)),
                evalplus_parallel=int(_cfg_get(cfg.model, "final_humaneval_parallel", 1)),
                evalplus_cache_dir=_prefer_existing_eval_path(
                    str(_cfg_get(cfg.model, "evalplus_cache_dir", "")),
                    _DEFAULT_HUMANEVAL_DATA_ROOT,
                ),
            )
            try:
                with open(
                    os.path.join(local_log_dir, "humaneval_report.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(he_report, f, indent=2, ensure_ascii=False)
            except Exception as e_json:
                log.warning("Failed to write humaneval_report.json: %s", e_json)

            pass1 = None
            if isinstance(he_report.get("pass_at_k"), dict):
                base_metrics = he_report["pass_at_k"].get("base")
                if isinstance(base_metrics, dict):
                    pass1 = base_metrics.get("pass@1")
            for key in ("humaneval", "base"):
                if isinstance(he_report.get(key), dict):
                    pass1 = he_report[key].get("pass@1")
                    if pass1 is not None:
                        break
            if pass1 is None and "pass@1" in he_report:
                pass1 = he_report["pass@1"]
            log.info("[final HumanEval] report=%s", he_report)
            if pass1 is not None:
                _wandb_log_and_summarize(
                    wandb_enabled,
                    {
                        "code/humaneval_pass@1": float(pass1),
                    },
                )
        except Exception as e:
            log.warning(
                "[final_humaneval_eval] failed after training: %s: %s",
                type(e).__name__,
                e,
            )


    if bool(_cfg_get(cfg.model, "final_humaneval_plus_eval", False)) and _is_trainer_log_main_process():
        try:
            he_report = evaluate_humaneval_plus_pass1(
                model,
                tokenizer,
                str(cfg.model.type),
                output_dir=local_log_dir,
                max_source_length=int(_cfg_get(cfg.model, "final_humaneval_max_source_length", 1024)),
                max_new_tokens=int(_cfg_get(cfg.model, "final_humaneval_max_new_tokens", 512)),
                evalplus_parallel=int(_cfg_get(cfg.model, "final_humaneval_parallel", 1)),
                evalplus_cache_dir=_prefer_existing_eval_path(
                    str(_cfg_get(cfg.model, "evalplus_cache_dir", "")),
                    _DEFAULT_HUMANEVAL_DATA_ROOT,
                ),
            )
            try:
                with open(
                    os.path.join(local_log_dir, "humaneval_plus_report.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(he_report, f, indent=2, ensure_ascii=False)
            except Exception as e_json:
                log.warning("Failed to write humaneval_plus_report.json: %s", e_json)

            pass1 = None
            if isinstance(he_report.get("pass_at_k"), dict):
                plus_metrics = he_report["pass_at_k"].get("plus")
                if isinstance(plus_metrics, dict):
                    pass1 = plus_metrics.get("pass@1")
            for key in ("humaneval_plus", "plus"):
                if isinstance(he_report.get(key), dict):
                    pass1 = he_report[key].get("pass@1")
                    if pass1 is not None:
                        break
            if pass1 is None and "pass@1" in he_report:
                pass1 = he_report["pass@1"]
            log.info("[final HumanEval+] report=%s", he_report)
            if pass1 is not None:
                _wandb_log_and_summarize(
                    wandb_enabled,
                    {
                        "code/humaneval_plus_pass@1": float(pass1),
                    },
                )
        except Exception as e:
            log.warning(
                "[final_humaneval_plus_eval] failed after training: %s: %s",
                type(e).__name__,
                e,
            )

    if bool(_cfg_get(cfg.model, "final_ifeval_eval", False)) and _is_trainer_log_main_process():
        try:
            ife_report = evaluate_ifeval(
                model,
                tokenizer,
                str(cfg.model.type),
                output_dir=local_log_dir,
                data_path=str(_cfg_get(cfg.model, "ifeval_data_path", "")) or None,
                tokenizer_name=str(_cfg_get(cfg.model, "name", "")),
                max_source_length=_cfg_get(cfg.model, "ifeval_max_source_length", None),
                max_new_tokens=int(_cfg_get(cfg.model, "ifeval_max_new_tokens", 1280)),
                max_examples=_cfg_get(cfg.model, "ifeval_max_examples", None),
                apply_chat_template=bool(_cfg_get(cfg.model, "ifeval_apply_chat_template", True)),
            )
            log.info("[final IFEval] report=%s", ife_report)
            ife_payload: Dict[str, float] = {
                "instruct/ifeval_prompt_level_strict_acc": float(ife_report["prompt_level_strict_acc"]),
                "instruct/ifeval_inst_level_strict_acc": float(ife_report["inst_level_strict_acc"]),
                "instruct/ifeval_prompt_level_loose_acc": float(ife_report["prompt_level_loose_acc"]),
                "instruct/ifeval_inst_level_loose_acc": float(ife_report["inst_level_loose_acc"]),
            }
            _wandb_log_and_summarize(wandb_enabled, ife_payload)
        except Exception as e:
            if bool(_cfg_get(cfg.model, "final_ifeval_strict", False)):
                raise
            log.warning(
                "[final_ifeval_eval] failed after training: %s: %s",
                type(e).__name__,
                e,
            )

    # Note: checkpoints are saved under runtime_dir/trainer_output during training, then that folder
    # is removed at the end in train_text_to_text_model so results/ does not grow.

    if _is_trainer_log_main_process() and wandb_enabled and getattr(wandb, "run", None) is not None:
        wandb.finish()


if __name__ == "__main__":
    run_exp()
