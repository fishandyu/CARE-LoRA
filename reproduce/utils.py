import torch
import os
import sys
import re
import shutil
import threading
import time
import json
import subprocess
import inspect
import argparse
import typing as tp
import random
from collections import Counter

from local_peft import ensure_local_peft_first

ensure_local_peft_first()

import numpy as np
import pandas as pd
from contextlib import nullcontext
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    DataCollatorForSeq2Seq,
)
from transformers import PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput
from datasets import Dataset, load_dataset, load_metric, load_from_disk
from datasets.utils import DownloadConfig
from torch.utils.data import DataLoader
from lora_plus import LoraPlusTrainingArguments, LoraPlusTrainer
import logging
import wandb
from peft import PeftModel
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_REPRODUCE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EVAL_DATA_ROOT = os.path.join(_REPRODUCE_DIR, "eval_datasets")
_DEFAULT_IFEVAL_DATA_DIR = os.path.join(_DEFAULT_EVAL_DATA_ROOT, "ifeval", "train")
_DEFAULT_IFEVAL_JSONL_PATH = os.path.join(_DEFAULT_EVAL_DATA_ROOT, "ifeval", "train.jsonl")
_DEFAULT_HUMANEVAL_DATA_ROOT = os.path.join(_DEFAULT_EVAL_DATA_ROOT, "humaneval")
_DEFAULT_EVALPLUS_RUNTIME_CACHE_ROOT = os.path.join(
    _REPRODUCE_DIR,
    "processed_datasets",
    "evalplus_runtime_cache",
)
_DEFAULT_NLTK_DATA_ROOT = os.path.join(_DEFAULT_EVAL_DATA_ROOT, "nltk_data")


def _existing_default_ifeval_path() -> tp.Optional[str]:
    for local_path in (_DEFAULT_IFEVAL_DATA_DIR, _DEFAULT_IFEVAL_JSONL_PATH):
        if os.path.exists(local_path):
            return local_path
    return None


def _prepend_default_nltk_data_path() -> None:
    if not os.path.isdir(_DEFAULT_NLTK_DATA_ROOT):
        return
    current = os.environ.get("NLTK_DATA", "")
    paths = [p for p in current.split(os.pathsep) if p]
    if _DEFAULT_NLTK_DATA_ROOT not in paths:
        os.environ["NLTK_DATA"] = os.pathsep.join([_DEFAULT_NLTK_DATA_ROOT, *paths])
    try:
        import nltk

        if _DEFAULT_NLTK_DATA_ROOT not in nltk.data.path:
            nltk.data.path.insert(0, _DEFAULT_NLTK_DATA_ROOT)
    except Exception:
        pass


def _distributed_world_size() -> int:
    """Return the torchrun / torch.distributed world size; single-process runs return 1."""
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def _is_trainer_log_main_process() -> bool:
    """Return whether this process should write W&B logs and exclusive files."""
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None and str(lr).strip() != "":
        try:
            return int(lr) == 0
        except ValueError:
            return True
    rk = os.environ.get("RANK")
    if rk is not None and str(rk).strip() != "":
        try:
            return int(rk) == 0
        except ValueError:
            return True
    return True


def _should_disable_tqdm(*, is_main_process: tp.Optional[bool] = None) -> bool:
    """
    Decide whether tqdm progress bars should be disabled.

    Priority:
    1) Explicit env override via CARE_LORA_DISABLE_TQDM
    2) Non-main DDP ranks: always disable
    3) Else: show tqdm on the main process.
    """
    _override = str(os.environ.get("CARE_LORA_DISABLE_TQDM", "")).strip().lower()
    if _override in {"1", "true", "yes", "on"}:
        return True
    if _override in {"0", "false", "no", "off"}:
        return False

    if is_main_process is None:
        is_main_process = _is_trainer_log_main_process()
    if not bool(is_main_process):
        return True

    return False


def _install_care_lora_attention_mask_context(model: torch.nn.Module, *, enabled: bool) -> None:
    """
    Provide CausalLM ``attention_mask`` to repository-local CARE-LoRA LoRA layers.

    Transformer Linear modules do not receive ``attention_mask`` directly, while CARE-LoRA's
    data-aware M* fit can use it to exclude padding token rows. A model-level
    forward pre-hook is the least invasive bridge: it updates a process-local
    context before submodules run. Ordinary LoRA / LoRA-FA / LoRAct ignore this context.
    """
    if not bool(enabled):
        try:
            import peft.tuners.lora.layer as lora_layer

            lora_layer.clear_care_lora_attention_mask()
        except Exception:
            pass
        return
    if getattr(model, "_care_lora_attention_mask_hook_installed", False):
        return
    try:
        import peft.tuners.lora.layer as lora_layer
    except Exception as e:
        log.warning("[care_lora attention mask] failed to import local LoRA layer module: %s", e)
        return

    def _pre_hook(module, args, kwargs):
        del module, args
        mask = kwargs.get("attention_mask") if isinstance(kwargs, dict) else None
        lora_layer.set_care_lora_attention_mask(mask if torch.is_tensor(mask) else None)

    try:
        handle = model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    except TypeError as e:
        log.warning(
            "[care_lora attention mask] PyTorch forward_pre_hook does not support with_kwargs; "
            "CARE-LoRA M* will keep using all token rows. Error: %s",
            e,
        )
        return
    model._care_lora_attention_mask_hook_handle = handle
    model._care_lora_attention_mask_hook_installed = True
    log.info("[care_lora attention mask] enabled: M* fit uses attention_mask==1 rows when shapes match.")


def _patch_transformers_flash_attn2_unavailable() -> None:
    """
    Make Transformers treat FlashAttention-2 as unavailable.

    Some environments keep flash-attn package metadata even when the CUDA
    extension cannot be imported. Patch the availability checks before loading a
    CausalLM so Transformers falls back to sdpa/eager attention cleanly.
    """
    import transformers.utils.import_utils as _import_utils
    import transformers.utils as _transformers_utils

    def _false() -> bool:
        return False

    _import_utils.is_flash_attn_2_available = _false  # type: ignore[method-assign]
    _import_utils.is_flash_attn_greater_or_equal_2_10 = _false  # type: ignore[method-assign]
    if hasattr(_transformers_utils, "is_flash_attn_2_available"):
        _transformers_utils.is_flash_attn_2_available = _false  # type: ignore[method-assign]
    if hasattr(_transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
        _transformers_utils.is_flash_attn_greater_or_equal_2_10 = _false  # type: ignore[method-assign]


def _preload_flash_attn_compat() -> None:
    """
    Disable FA2 discovery if flash-attn is installed but its extension fails.
    """
    import sys

    try:
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_func  # noqa: F401
    except ModuleNotFoundError:
        return
    except ImportError:
        for k in list(sys.modules.keys()):
            if k == "flash_attn" or k.startswith("flash_attn."):
                del sys.modules[k]
        log.warning(
            "flash_attn is installed but its CUDA extension cannot be imported. "
            "Disabled Transformers FlashAttention-2 discovery for this process; "
            "training will use sdpa/eager attention."
        )
        _patch_transformers_flash_attn2_unavailable()


def _get_peft_cfg0(model):
    """Return the first PEFT config (usually adapter 'default') if model is a PeftModel."""
    cfg = getattr(model, "peft_config", None)
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        if len(cfg) == 0:
            return None
        return cfg.get("default", next(iter(cfg.values())))
    return cfg


def _runtime_startup_report(
    trainer,
    model,
    *,
    use_lorafa: bool,
    use_care_lora: bool,
    use_loraplus: bool,
    use_loract: bool = False,
):
    """Print a structured summary of the run setup."""
    trainer_name = getattr(trainer, "__class__", type(trainer)).__name__
    log.info("\n" + "=" * 60)
    log.info(
        "Trainer: %s | use_care_lora=%s | use_loract=%s | use_lorafa=%s | use_loraplus=%s",
        trainer_name,
        bool(use_care_lora),
        bool(use_loract),
        bool(use_lorafa),
        bool(use_loraplus),
    )

    cfg0 = _get_peft_cfg0(model)
    if cfg0 is not None:
        lora_r = getattr(cfg0, "r", getattr(cfg0, "lora_r", None))
        log.info(
            "PEFT config: "
            + f"type={type(cfg0).__name__}, "
            + f"lora_r={lora_r}, "
            + f"lora_alpha={getattr(cfg0, 'lora_alpha', None)}, "
            + f"init_lora_weights={getattr(cfg0, 'init_lora_weights', None)}, "
            + f"use_rslora={getattr(cfg0, 'use_rslora', None)}, "
            + f"use_dora={getattr(cfg0, 'use_dora', None)}, "
            + f"use_care_lora={getattr(cfg0, 'use_care_lora', None)}, "
            + f"use_loract={getattr(cfg0, 'use_loract', None)}, "
            + f"loract_rank={getattr(cfg0, 'loract_rank', None)}, "
            + f"use_lorafa={getattr(cfg0, 'use_lorafa', None)}, "
            + f"care_lora_pinv_lambda={getattr(cfg0, 'care_lora_pinv_lambda', None)}, "
        )
    else:
        log.info("PEFT config: <none> (not a PeftModel or peft_config not found)")

    named_params = list(model.named_parameters())
    total_params = sum(p.numel() for _, p in named_params)
    trainable_params = [(n, p) for n, p in named_params if p.requires_grad]
    trainable_numel = sum(p.numel() for _, p in trainable_params)

    log.info(
        f"Params: trainable={trainable_numel} / total={total_params} "
        f"({(100.0 * trainable_numel / max(total_params, 1)):.4f}%) | "
        f"trainable_tensors={len(trainable_params)}"
    )

    a_train = [n for n, _ in trainable_params if "lora_A" in n]
    b_train = [n for n, _ in trainable_params if "lora_B" in n]
    other_train = [n for n, _ in trainable_params if ("lora_A" not in n and "lora_B" not in n)]

    log.info(
        f"Trainable tensors (by name): lora_A={len(a_train)}, lora_B={len(b_train)}, other={len(other_train)}"
    )
    if a_train:
        log.info(f"  e.g., trainable lora_A: {a_train[:3]}")
    if b_train:
        log.info(f"  e.g., trainable lora_B: {b_train[:3]}")

    if use_lorafa:
        runtime_lorafa_layers = 0
        for _, module in model.named_modules():
            use_lorafa_dict = getattr(module, "use_lorafa", {})
            if isinstance(use_lorafa_dict, dict) and bool(use_lorafa_dict.get("default", False)):
                runtime_lorafa_layers += 1
        if runtime_lorafa_layers == 0:
            raise RuntimeError(
                "LoRA-FA was requested globally, but zero LoRA layers have runtime use_lorafa[default]=True. "
                "Expected LoRA-FA runtime flags were not propagated to the LoRA layers."
            )
    if use_care_lora:
        runtime_care_lora_layers = 0
        for _, module in model.named_modules():
            use_care_lora_dict = getattr(module, "use_care_lora", {})
            if isinstance(use_care_lora_dict, dict) and bool(use_care_lora_dict.get("default", False)):
                runtime_care_lora_layers += 1
        if runtime_care_lora_layers == 0:
            raise RuntimeError(
                "CARE-LoRA was requested globally, but zero LoRA layers have runtime use_care_lora[default]=True. "
                "Expected CARE-LoRA runtime flags were not propagated to the LoRA layers."
            )
    if use_loract:
        runtime_loract_layers = 0
        for _, module in model.named_modules():
            d = getattr(module, "use_loract", {})
            if isinstance(d, dict) and bool(d.get("default", False)):
                runtime_loract_layers += 1
        if runtime_loract_layers == 0:
            raise RuntimeError(
                "LoRAct was requested globally, but zero LoRA layers have runtime use_loract[default]=True. "
                "This means the forward path is NOT running the LoRAct branch, so activation saving will not match."
            )

    try:
        if hasattr(trainer, "create_optimizer"):
            trainer.create_optimizer()
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            log.info("Optimizer: <none yet> (will be created at train start)")
        else:
            opt_param_ids = {id(p) for g in opt.param_groups for p in g.get("params", [])}
            a_in_opt = [n for n, p in named_params if ("lora_A" in n and id(p) in opt_param_ids)]
            b_in_opt = [n for n, p in named_params if ("lora_B" in n and id(p) in opt_param_ids)]
            log.info(f"Optimizer contains: lora_A={len(a_in_opt)}, lora_B={len(b_in_opt)}")
            if a_in_opt:
                log.info(f"  e.g., optimizer lora_A: {a_in_opt[:3]}")
            if b_in_opt:
                log.info(f"  e.g., optimizer lora_B: {b_in_opt[:3]}")

            if use_care_lora and len(a_in_opt) == 0:
                log.warning("[CARE-LoRA guard] Detected no lora_A in optimizer param groups; CARE-LoRA expects A to be trainable.")
            if use_care_lora and len(b_in_opt) == 0:
                log.warning("[CARE-LoRA guard] Detected no lora_B in optimizer param groups; CARE-LoRA is misconfigured.")
            if use_loract and len(a_in_opt) == 0:
                log.warning("[LoRAct guard] Detected no lora_A in optimizer param groups; expected trainable A.")
            if use_loract and len(b_in_opt) == 0:
                log.warning("[LoRAct guard] Detected no lora_B in optimizer param groups; misconfigured.")
            if use_lorafa and len(a_in_opt) > 0:
                log.warning("[LoRA-FA guard] Detected lora_A in optimizer param groups, but LoRA-FA expects A to be frozen.")
            if use_lorafa and len(b_in_opt) == 0:
                log.warning("[LoRA-FA guard] Detected no lora_B in optimizer param groups; training likely misconfigured.")
            if (
                (not use_lorafa)
                and (not use_care_lora)
                and (not use_loract)
                and cfg0 is not None
                and type(cfg0).__name__ == "LoraConfig"
            ):
                if len(a_in_opt) == 0:
                    log.warning("[LoRA guard] Detected no lora_A in optimizer param groups; standard LoRA would silently degenerate toward B-only training.")
                if len(b_in_opt) == 0:
                    log.warning("[LoRA guard] Detected no lora_B in optimizer param groups; standard LoRA is misconfigured.")
    except Exception as e:
        log.warning(f"Optimizer membership check skipped: {type(e).__name__}: {e}")

    log.info("=" * 60 + "\n")


class ReservedPeakTrackerCallback(TrainerCallback):
    """Track allocated bytes at the exact moment reserved bytes reach a new peak."""

    def __init__(self):
        self.max_reserved_bytes = 0
        self.allocated_at_max_reserved_bytes = 0
        self.max_reserved_global_step = -1
        self.max_reserved_stage = "unknown"

    def _sample(self, state, stage: str):
        if not torch.cuda.is_available():
            return
        try:
            cur_alloc = torch.cuda.memory_allocated()
            cur_reserved = torch.cuda.memory_reserved()
        except Exception:
            return
        if cur_reserved > self.max_reserved_bytes:
            self.max_reserved_bytes = cur_reserved
            self.allocated_at_max_reserved_bytes = cur_alloc
            self.max_reserved_global_step = int(getattr(state, "global_step", -1))
            self.max_reserved_stage = str(stage)

    def on_step_end(self, args, state, control, **kwargs):
        self._sample(state, stage="train_step_end")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._sample(state, stage="train_step_begin")
        return control

    def on_substep_end(self, args, state, control, **kwargs):
        self._sample(state, stage="train_substep_end")
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._sample(state, stage="epoch_begin")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        self._sample(state, stage="epoch_end")
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        self._sample(state, stage="train_begin")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._sample(state, stage="train_end")
        return control

    def on_prediction_step(self, args, state, control, **kwargs):
        self._sample(state, stage="prediction_step")
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        self._sample(state, stage="evaluate")
        return control


class CudaPeakPairSampler:
    """Continuously sample (allocated, reserved) to keep a same-time peak pair."""

    def __init__(self, poll_interval_s: float = 0.005):
        self.poll_interval_s = float(poll_interval_s)
        self._stop_event = threading.Event()
        self._thread = None
        self.max_reserved_bytes = 0
        self.allocated_at_max_reserved_bytes = 0
        self.max_allocated_bytes = 0
        self.reserved_at_max_allocated_bytes = 0

    def _run(self):
        while not self._stop_event.is_set():
            try:
                cur_alloc = torch.cuda.memory_allocated()
                cur_reserved = torch.cuda.memory_reserved()
                if cur_reserved > self.max_reserved_bytes:
                    self.max_reserved_bytes = cur_reserved
                    self.allocated_at_max_reserved_bytes = cur_alloc
                if cur_alloc > self.max_allocated_bytes:
                    self.max_allocated_bytes = cur_alloc
                    self.reserved_at_max_allocated_bytes = cur_reserved
            except Exception:
                pass
            time.sleep(self.poll_interval_s)

    def start(self):
        if not torch.cuda.is_available():
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(0.1, 10 * self.poll_interval_s))
        self._thread = None


class SavedTensorActivationTracker:
    """
    Track peak live CUDA bytes of autograd-saved tensors.

    Implementation detail:
      - saved_tensors_hooks hooks into autograd's save_for_backward.
      - We approximate "activation bytes" by summing **unique storage** bytes
        of CUDA tensors saved for backward.

    Important: multiple saved tensors can be views into the same underlying
    `untyped_storage()` (same physical allocation). Summing `numel*elem_size`
    per tensor **double-counts** shared storage and can make `activation_mib`
    larger than the true allocated residual, yielding negative `other_mib`.
    We therefore refcount by `untyped_storage().data_ptr()` and add/remove
    `storage.nbytes()` only when the refcount transitions 0->1 / 1->0.
    """

    def __init__(
        self,
        *,
        excluded_ptrs: Optional[set[int]] = None,
        excluded_storage_ptrs: Optional[set[int]] = None,
        excluded_shapes: Optional[set[tuple[int, ...]]] = None,
        include_last_dims: Optional[set[int]] = None,
    ) -> None:
        self.live_bytes: int = 0
        self.peak_live_bytes: int = 0
        self.excluded_ptrs: set[int] = excluded_ptrs or set()
        # Exclude by underlying allocation: views of parameters can have a
        # different tensor.data_ptr() than the Parameter but share storage.
        self.excluded_storage_ptrs: set[int] = excluded_storage_ptrs or set()
        # Shape exclusions keep CARE-LoRA M* out of activation accounting because
        # it is already counted in the LoRA bucket.
        self.excluded_shapes: set[tuple[int, ...]] = excluded_shapes or set()
        self.include_last_dims: Optional[set[int]] = include_last_dims
        # storage_ptr -> refcount of saved tensors referencing this storage
        self._storage_refcount: dict[int, int] = {}
        # storage_ptr -> nbytes (constant for that storage)
        self._storage_nbytes: dict[int, int] = {}
        # LoRA-adapter saved activation accounting at the same peak as total saved tensors.
        self.lora_activation_live_bytes: int = 0
        self.lora_activation_bytes_at_peak: int = 0
        self._lora_activation_storage_refcount: dict[int, int] = {}
        self._lora_activation_storage_nbytes: dict[int, int] = {}
        try:
            from peft.tuners.lora import layer as _lora_layer

            self._is_lora_activation_context_active = _lora_layer.is_lora_activation_context_active
        except Exception:
            self._is_lora_activation_context_active = None

    def pack(self, tensor):
        storage_key: Optional[int] = None
        lora_storage_key: Optional[int] = None
        storage_size = 0
        added = 0
        lora_added = 0
        try:
            if torch.is_tensor(tensor) and tensor.is_cuda:
                storage = tensor.untyped_storage()
                storage_key = int(storage.data_ptr())
                storage_size = int(storage.nbytes())
                ptr = int(tensor.data_ptr())
                shape = tuple(int(x) for x in tensor.shape)
                excluded = (
                    (storage_key in self.excluded_storage_ptrs)
                    or (ptr in self.excluded_ptrs)
                    or (shape in self.excluded_shapes)
                )
                if not excluded:
                    self._storage_refcount[storage_key] = int(self._storage_refcount.get(storage_key, 0)) + 1
                    self._storage_nbytes[storage_key] = storage_size
                    if self._storage_refcount[storage_key] == 1:
                        added = storage_size
                    is_lora_activation = (
                        self._is_lora_activation_context_active is not None
                        and self._is_lora_activation_context_active()
                    )
                    if is_lora_activation:
                        lora_storage_key = storage_key
                        self._lora_activation_storage_refcount[lora_storage_key] = (
                            int(self._lora_activation_storage_refcount.get(lora_storage_key, 0)) + 1
                        )
                        self._lora_activation_storage_nbytes[lora_storage_key] = storage_size
                        if self._lora_activation_storage_refcount[lora_storage_key] == 1:
                            lora_added = storage_size
                else:
                    storage_key = None
                    storage_size = 0
        except Exception:
            storage_key = None
            lora_storage_key = None
            storage_size = 0
            added = 0
            lora_added = 0
        self.live_bytes += added
        self.lora_activation_live_bytes += lora_added
        if self.live_bytes > self.peak_live_bytes:
            self.peak_live_bytes = self.live_bytes
            self.lora_activation_bytes_at_peak = self.lora_activation_live_bytes
        # Return a packed payload; hook will call unpack() later.
        return (tensor, storage_key, storage_size, lora_storage_key)

    def unpack(self, packed):
        try:
            tensor, storage_key, storage_size, lora_storage_key = packed
        except ValueError:
            tensor, storage_key, storage_size = packed
            lora_storage_key = None
        try:
            if storage_key is not None:
                rc = int(self._storage_refcount.get(storage_key, 0))
                if rc > 0:
                    self._storage_refcount[storage_key] = rc - 1
                    if self._storage_refcount[storage_key] == 0:
                        sz = int(self._storage_nbytes.get(storage_key, storage_size))
                        self.live_bytes -= sz
                        del self._storage_refcount[storage_key]
                        if storage_key in self._storage_nbytes:
                            del self._storage_nbytes[storage_key]
            if lora_storage_key is not None:
                lora_rc = int(self._lora_activation_storage_refcount.get(lora_storage_key, 0))
                if lora_rc > 0:
                    self._lora_activation_storage_refcount[lora_storage_key] = lora_rc - 1
                    if self._lora_activation_storage_refcount[lora_storage_key] == 0:
                        lora_sz = int(self._lora_activation_storage_nbytes.get(lora_storage_key, storage_size))
                        self.lora_activation_live_bytes -= lora_sz
                        del self._lora_activation_storage_refcount[lora_storage_key]
                        if lora_storage_key in self._lora_activation_storage_nbytes:
                            del self._lora_activation_storage_nbytes[lora_storage_key]
            if self.live_bytes < 0:
                self.live_bytes = 0
            if self.lora_activation_live_bytes < 0:
                self.lora_activation_live_bytes = 0
        except Exception:
            pass
        return tensor


class CausalLMDataCollator:
    """Dynamically pad causal-LM features and pad labels with -100."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Suppress the fast-tokenizer padding advisory; inputs are already tokenized.
        _dw = getattr(self.tokenizer, "deprecation_warnings", None)
        if isinstance(_dw, dict):
            _dw["Asking-to-pad-a-fast-tokenizer"] = True

    def __call__(self, features):
        labels = [list(f["labels"]) for f in features]
        inputs = [{k: v for k, v in f.items() if k not in {"labels", "length"}} for f in features]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            pad_len = max_len - len(lab)
            if pad_len < 0:
                lab = lab[:max_len]
                pad_len = 0
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                padded_labels.append(([-100] * pad_len) + lab)
            else:
                padded_labels.append(lab + ([-100] * pad_len))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


class Seq2SeqDataCollatorStripLength:
    """Wrap HF seq2seq collator while stripping helper columns such as `length`."""

    def __init__(self, tokenizer, model):
        self.inner = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

    def __call__(self, features):
        cleaned = [{k: v for k, v in f.items() if k != "length"} for f in features]
        return self.inner(cleaned)


def _causal_lm_combined_text_for_train(x: str, y: str, tokenizer: PreTrainedTokenizer) -> str:
    """Build the full CausalLM training sequence used by _causal_lm_encode_batched."""
    return x + " " + y + tokenizer.eos_token


def causal_lm_training_sequence_token_count(
    x: str, y: str, tokenizer: PreTrainedTokenizer
) -> int:
    """Return the untruncated token count of the full CausalLM training sequence."""
    s = _causal_lm_combined_text_for_train(x, y, tokenizer)
    enc = tokenizer(s, padding=False, truncation=False)
    return len(enc["input_ids"])


def _causal_lm_encode_batched(batch, tokenizer, max_length=-1, ignore_masked_token=True):
    combined_text = [
        _causal_lm_combined_text_for_train(x, y, tokenizer)
        for x, y in zip(batch["x"], batch["y"])
    ]
    answer_starts = [len(x + " ") for x in batch["x"]]
    tokenizer_kwargs = dict(
        padding=False,
        truncation=True,
        max_length=max_length if max_length != -1 else None,
    )
    try:
        encodings = tokenizer(
            combined_text,
            return_offsets_mapping=True,
            **tokenizer_kwargs,
        )
        offset_mapping = encodings.pop("offset_mapping")
    except (NotImplementedError, TypeError, ValueError):
        encodings = tokenizer(combined_text, **tokenizer_kwargs)
        offset_mapping = None
        prefix_texts = [x + " " for x in batch["x"]]
        x_enc = tokenizer(prefix_texts, **tokenizer_kwargs)

    labels = []
    for row_idx, (ids, attn) in enumerate(zip(encodings["input_ids"], encodings["attention_mask"])):
        l = list(ids)
        if offset_mapping is not None:
            answer_start = int(answer_starts[row_idx])
            for i, (_start, end) in enumerate(offset_mapping[row_idx]):
                if int(end) <= answer_start:
                    l[i] = -100
        else:
            # Slow-tokenizer fallback for tokenizers without offset mappings.
            x_ids = x_enc["input_ids"][row_idx]
            prefix_len = min(len(x_ids), len(l))
            for i in range(prefix_len):
                l[i] = -100
        if ignore_masked_token:
            # attention_mask is all 1s here because we avoid eager padding; keep for compatibility
            pass
        labels.append(l)

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    }


def _seq2seq_encode_batched(batch, tokenizer, max_length=None, ignore_masked_token=True):
    inputs = tokenizer(
        batch["x"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    outputs = tokenizer(
        batch["y"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    labels = []
    for ids in outputs["input_ids"]:
        labels.append(list(ids))

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }


def causalLMEncode(example, tokenizer, max_length=-1, ignore_masked_token=True):
    is_list_input = isinstance(example["x"], list)
    # Combine text and add EOS token
    combined_text = (
        [
            x + " " + y + tokenizer.eos_token
            for (x, y) in zip(example["x"], example["y"])
        ]
        if is_list_input
        else example["x"] + " " + example["y"] + tokenizer.eos_token
    )
    # Tokenize combined text
    encodings = tokenizer(
        combined_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length if max_length != -1 else None,
    )
    # Match _causal_lm_encode_batched: the supervised prefix is x + " ".
    input_text_length = (
        [
            len(
                tokenizer(
                    example["x"][i] + " ", return_tensors="pt"
                )["input_ids"][0]
            )
            for i in range(len(example["x"]))
        ]
        if is_list_input
        else len(
            tokenizer(example["x"] + " ", return_tensors="pt")["input_ids"][0]
        )
    )
    if input_text_length[0] >= max_length:
        log.warning(
            f"Input text length >= max_length: {input_text_length} >= {max_length}. "
            "Consider increasing max_length to avoid truncation."
        )
    # Create labels
    labels = encodings["input_ids"].clone()
    if is_list_input:
        for i, l in enumerate(input_text_length):
            labels[i, :l] = -100
    else:
        labels[0, :input_text_length] = -100
    if ignore_masked_token:
        labels[encodings["attention_mask"] == 0] = -100
    # Update example dictionary
    results = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
        # "input_text_length": input_text_length,
    }

    return results


def SeqToSeqEncode(example, tokenizer, max_length=None, ignore_masked_token=True):
    inputs = tokenizer(
        example["x"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    outputs = tokenizer(
        example["y"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    labels = outputs["input_ids"]

    # Mask padding positions so pad tokens do not contribute to the loss.
    if ignore_masked_token:
        labels = labels.clone()
        labels[outputs["attention_mask"] == 0] = -100

    results = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
        # "decoder_attention_mask": outputs["attention_mask"],
    }
    return results


def preprocess_dataset(
    dataset: tp.Union[Dataset, tp.List[tp.Tuple[str, str]], tp.List[tp.Dict[str, str]]]
) -> Dataset:
    if isinstance(dataset, list) and isinstance(dataset[0], tuple):
        dataset = Dataset.from_pandas(pd.DataFrame(dataset, columns=["x", "y"]))
    elif isinstance(dataset, list) and isinstance(dataset[0], dict):
        dataset = Dataset.from_dict(
            {k: [dic[k] for dic in dataset] for k in dataset[0]}
        )
    elif isinstance(dataset, dict):
        dataset = Dataset.from_dict(dataset)
    elif isinstance(dataset, Dataset):
        pass
    else:
        raise ValueError("Wrong format")
    return dataset


def initialize_text_to_text_model(
    model_name: str,
    model_type: str,
    bf16: bool,
    use_peft: bool = True,
    tokenizer: str = None,
    flash_attention: bool = False,
    use_flash_attention_2: bool = False,
):
    if model_type == "CausalLM":
        _preload_flash_attn_compat()
        # CausalLM attention implementation selection.
        want_fa2 = bool(use_flash_attention_2 or flash_attention)
        if flash_attention and not use_flash_attention_2:
            log.info(
                "CausalLM: flash_attention=True is treated as use_flash_attention_2=True."
            )
        impl_candidates: tp.List[str] = []
        if want_fa2:
            try:
                from transformers.utils.import_utils import is_flash_attn_2_available

                if is_flash_attn_2_available():
                    impl_candidates.append("flash_attention_2")
                else:
                    log.warning(
                        "CausalLM: FlashAttention-2 was requested but is unavailable; using sdpa/eager."
                    )
            except Exception as e:
                log.warning(
                    "CausalLM: FlashAttention-2 availability check failed (%s); using sdpa/eager.",
                    e,
                )
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            impl_candidates.append("sdpa")
        impl_candidates.append("eager")
        # Preserve order while removing duplicates.
        _seen: tp.Set[str] = set()
        impl_order: tp.List[str] = []
        for _impl in impl_candidates:
            if _impl not in _seen:
                _seen.add(_impl)
                impl_order.append(_impl)

        last_err: Optional[Exception] = None
        model = None
        chosen: Optional[str] = None
        for impl in impl_order:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                    device_map="auto" if use_peft else None,
                    attn_implementation=impl,
                )
                chosen = impl
                break
            except (TypeError, ValueError, RuntimeError) as e:
                last_err = e
                log.warning(
                    "CausalLM: attn_implementation=%s failed, trying the next candidate: %s",
                    impl,
                    e,
                )
        if model is None or chosen is None:
            raise RuntimeError(
                "CausalLM: from_pretrained failed for all attention implementations (tried: "
                + ", ".join(impl_order)
                + "）"
            ) from last_err
        if want_fa2 and chosen == "flash_attention_2":
            log.info(
                "CausalLM: attn_implementation=flash_attention_2"
            )
        else:
            log.info("CausalLM: attn_implementation=%s", chosen)
    elif model_type == "ConditionalGeneration":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if bf16 else torch.float32,
            device_map="auto" if use_peft else None,
        )
    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model_type == "CausalLM":
        tokenizer.padding_side = "right"
        model.config.pad_token_id = tokenizer.pad_token_id
        try:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass
    return model, tokenizer


def _pick_first_label_token(label_ids: np.ndarray) -> np.ndarray:
    """Extract one label token per sample from label_ids with shape (B, L)."""
    if label_ids.ndim != 2:
        raise ValueError(f"label_ids must be 2D, got shape={label_ids.shape}")
    y = label_ids[:, 0].copy()
    if np.any(y == -100):
        for i in range(label_ids.shape[0]):
            row = label_ids[i]
            idx = np.where(row != -100)[0]
            if len(idx) > 0:
                y[i] = row[idx[0]]
    return y.astype(np.int64)


def _predict_label_from_candidates(logits0: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Argmax over the candidate label-token set at decoder position 0."""
    candidates = np.array(sorted(set(int(x) for x in candidates.tolist() if int(x) != -100)), dtype=np.int64)
    if candidates.size == 0:
        return np.argmax(logits0, axis=-1).astype(np.int64)

    # Restrict logits to the candidate-token subspace.
    sub = logits0[:, candidates]                      # (B, C)
    best_idx = np.argmax(sub, axis=1)                 # (B,)
    pred = candidates[best_idx]                       # (B,)
    return pred.astype(np.int64)


def _mcc_binary(y_true01: np.ndarray, y_pred01: np.ndarray) -> float:
    y_true01 = y_true01.astype(np.int64)
    y_pred01 = y_pred01.astype(np.int64)
    tp = int(np.sum((y_true01 == 1) & (y_pred01 == 1)))
    tn = int(np.sum((y_true01 == 0) & (y_pred01 == 0)))
    fp = int(np.sum((y_true01 == 0) & (y_pred01 == 1)))
    fn = int(np.sum((y_true01 == 1) & (y_pred01 == 0)))
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def _f1_binary_positive1(y_true01: np.ndarray, y_pred01: np.ndarray) -> float:
    """Binary F1 with class 1 as the positive class."""
    y_true01 = y_true01.astype(np.int64)
    y_pred01 = y_pred01.astype(np.int64)
    tp = int(np.sum((y_true01 == 1) & (y_pred01 == 1)))
    fp = int(np.sum((y_true01 == 0) & (y_pred01 == 1)))
    fn = int(np.sum((y_true01 == 1) & (y_pred01 == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def _f1_macro_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: tp.Sequence[int],
) -> float:
    """Multiclass macro-F1 implemented as one-vs-rest F1 over ``labels``."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    f1s: list[float] = []
    for c in labels:
        c = int(c)
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0.0:
            f1s.append(0.0)
        else:
            f1s.append(float(2.0 * prec * rec / (prec + rec)))
    return float(np.mean(f1s)) if f1s else 0.0


def _mrpc_equivalent_token_id(tokenizer) -> tp.Tuple[int, int]:
    """Return first-token ids for the MRPC labels used by data.load_mrpc."""
    pos_ids = tokenizer("equivalent", add_special_tokens=False).input_ids
    neg_ids = tokenizer("different", add_special_tokens=False).input_ids
    if not pos_ids or not neg_ids:
        raise ValueError("MRPC F1: tokenizer returned empty ids for label strings.")
    pos_id = int(pos_ids[0])
    neg_id = int(neg_ids[0])
    return pos_id, neg_id


def _cb_label_token_ids(tokenizer) -> tp.Tuple[int, int, int]:
    """Return first-token ids for the CB labels used by data.load_cb."""
    words = ("entailment", "contradiction", "neutral")
    ids: list[int] = []
    for w in words:
        tids = tokenizer(w, add_special_tokens=False).input_ids
        if not tids:
            raise ValueError(f"CB macro-F1: tokenizer returned empty ids for label string {w!r}.")
        ids.append(int(tids[0]))
    if len(set(ids)) != 3:
        raise ValueError(
            "CB macro-F1: first-token ids for entailment/contradiction/neutral must be distinct; "
            f"got {dict(zip(words, ids))}."
        )
    return int(ids[0]), int(ids[1]), int(ids[2])



def _infer_candidate_token_ids_from_dataset(dataset: Dataset, tokenizer) -> tp.Optional[np.ndarray]:
    """Infer fixed first-token label candidates from raw text labels BEFORE set_transform."""
    try:
        if dataset is None or "y" not in dataset.column_names:
            return None
        uniq_labels = []
        seen = set()
        for y in dataset["y"]:
            y = str(y)
            if y not in seen:
                seen.add(y)
                uniq_labels.append(y)
        if len(uniq_labels) < 2 or len(uniq_labels) > 8:
            return None
        token_ids = []
        for y in uniq_labels:
            ids = tokenizer(y, add_special_tokens=False).input_ids
            if len(ids) == 0:
                continue
            token_ids.append(int(ids[0]))
        token_ids = sorted(set(token_ids))
        if len(token_ids) < 2:
            return None
        return np.array(token_ids, dtype=np.int64)
    except Exception:
        return None


def _infer_candidate_token_ids_from_tokenized_dataset(dataset: Dataset) -> tp.Optional[np.ndarray]:
    """Infer first-label-token candidates directly from tokenized `labels` for robustness."""
    try:
        if dataset is None or "labels" not in dataset.column_names:
            return None
        labels_col = dataset["labels"]
        first_tokens: list[int] = []

        for row in labels_col:
            # row is usually a Python list (variable-length) after dataset.map tokenization
            if row is None:
                continue
            if isinstance(row, np.ndarray):
                vals = row.tolist()
            elif isinstance(row, (list, tuple)):
                vals = list(row)
            else:
                # scalar fallback
                vals = [row]

            tok = -100
            if len(vals) > 0:
                tok = int(vals[0])
                if tok == -100:
                    for v in vals:
                        iv = int(v)
                        if iv != -100:
                            tok = iv
                            break
            if tok != -100:
                first_tokens.append(int(tok))

        if len(first_tokens) == 0:
            return None

        cand = np.array(sorted(set(first_tokens)), dtype=np.int64)
        if cand.size < 2:
            return None
        return cand
    except Exception:
        return None


def _build_preprocess_logits_for_metrics(candidate_token_ids: tp.Optional[np.ndarray]):
    """Reduce eval payload before HF gathers it across the whole validation set."""
    if candidate_token_ids is None:
        def _preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            return logits[:, 0, :]
        return _preprocess_logits_for_metrics

    cand_cpu = torch.tensor(candidate_token_ids, dtype=torch.long)
    _device_cache: dict[str, torch.Tensor] = {}

    def _preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits0 = logits[:, 0, :]
        key = str(logits0.device)
        local_cand = _device_cache.get(key)
        if local_cand is None or local_cand.device != logits0.device:
            local_cand = cand_cpu.to(device=logits0.device)
            _device_cache[key] = local_cand

        # Return predicted token ids with shape (B,) instead of candidate logits.
        sub = logits0.index_select(dim=-1, index=local_cand)   # (B, C)
        best_idx = torch.argmax(sub, dim=-1)                   # (B,)
        pred_token_ids = local_cand[best_idx]                  # (B,)
        return pred_token_ids.to(dtype=torch.long)

    return _preprocess_logits_for_metrics


def _decode_pred_from_maybe_compressed_logits(
    logits0: np.ndarray,
    y_true: np.ndarray,
    candidate_token_ids: tp.Optional[np.ndarray],
):
    """Support full-vocab logits, compressed candidate logits, or already-decoded token ids."""
    if logits0.ndim == 1:
        return logits0.astype(np.int64)

    if logits0.ndim == 2 and logits0.shape[1] == 1:
        return logits0.reshape(-1).astype(np.int64)

    if candidate_token_ids is not None and logits0.ndim == 2 and logits0.shape[1] == len(candidate_token_ids):
        best_idx = np.argmax(logits0, axis=1)
        return np.asarray(candidate_token_ids, dtype=np.int64)[best_idx]

    candidates = np.unique(y_true)
    return _predict_label_from_candidates(logits0, candidates)


def _normalize_eval_predictions_for_metrics(preds):
    """Normalize predictions returned by Trainer.evaluate / preprocess_logits_for_metrics."""
    if preds.ndim == 1:
        return preds
    if preds.ndim == 2:
        return preds
    return preds[:, 0, :]


def make_compute_metrics(candidate_token_ids: tp.Optional[np.ndarray] = None):
    def _compute_metrics(p: PredictionOutput):
        preds = p.predictions
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        logits0 = _normalize_eval_predictions_for_metrics(logits)
        label_ids = p.label_ids
        y_true = _pick_first_label_token(label_ids)
        y_pred = _decode_pred_from_maybe_compressed_logits(
            logits0, y_true, candidate_token_ids
        )
        acc = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
        return {"metrics": acc, "acc": acc}
    return _compute_metrics


def make_compute_metrics_cola_mcc(candidate_token_ids: tp.Optional[np.ndarray] = None):
    def _compute_metrics_cola_mcc(p: PredictionOutput):
        preds = p.predictions
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        logits0 = _normalize_eval_predictions_for_metrics(logits)
        label_ids = p.label_ids
        y_true_tok = _pick_first_label_token(label_ids)
        y_pred_tok = _decode_pred_from_maybe_compressed_logits(
            logits0, y_true_tok, candidate_token_ids
        )

        cand = np.array(
            sorted(set(int(x) for x in np.unique(y_true_tok).tolist() if int(x) != -100)),
            dtype=np.int64,
        )
        if cand.size != 2:
            acc = float(np.mean(y_pred_tok == y_true_tok)) if len(y_true_tok) else 0.0
            return {"metrics": 0.0, "mcc": 0.0, "acc": float(acc)}

        neg_tok, pos_tok = cand[0], cand[1]
        y_true01 = (y_true_tok == pos_tok).astype(np.int64)
        y_pred01 = (y_pred_tok == pos_tok).astype(np.int64)

        mcc = _mcc_binary(y_true01, y_pred01)
        acc = float(np.mean(y_pred_tok == y_true_tok)) if len(y_true_tok) else 0.0
        return {"metrics": float(mcc), "mcc": float(mcc), "acc": float(acc)}

    return _compute_metrics_cola_mcc


def make_compute_metrics_mrpc_f1(
    candidate_token_ids: tp.Optional[np.ndarray],
    tokenizer,
):
    """MRPC primary metric is F1 with equivalent / GLUE label 1 as the positive class."""

    def _compute_metrics_mrpc_f1(p: PredictionOutput):
        preds = p.predictions
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        logits0 = _normalize_eval_predictions_for_metrics(logits)
        label_ids = p.label_ids
        y_true_tok = _pick_first_label_token(label_ids)
        y_pred_tok = _decode_pred_from_maybe_compressed_logits(
            logits0, y_true_tok, candidate_token_ids
        )

        pos_id, neg_id = _mrpc_equivalent_token_id(tokenizer)
        cand = np.array(
            sorted(set(int(x) for x in np.unique(y_true_tok).tolist() if int(x) != -100)),
            dtype=np.int64,
        )
        if cand.size != 2 or pos_id == neg_id:
            acc = float(np.mean(y_pred_tok == y_true_tok)) if len(y_true_tok) else 0.0
            return {"metrics": 0.0, "f1": 0.0, "acc": float(acc)}

        if pos_id not in cand or neg_id not in cand:
            acc = float(np.mean(y_pred_tok == y_true_tok)) if len(y_true_tok) else 0.0
            log.warning(
                "[MRPC F1] Label token ids from batch do not match tokenizer('equivalent'/'different') "
                f"first tokens (pos_id={pos_id}, neg_id={neg_id}, cand={cand.tolist()}). "
                "Falling back to sorted-token positive class (may not match GLUE sign)."
            )
            neg_tok, pos_tok = cand[0], cand[1]
            y_true01 = (y_true_tok == pos_tok).astype(np.int64)
            y_pred01 = (y_pred_tok == pos_tok).astype(np.int64)
        else:
            y_true01 = (y_true_tok == pos_id).astype(np.int64)
            y_pred01 = (y_pred_tok == pos_id).astype(np.int64)

        f1 = _f1_binary_positive1(y_true01, y_pred01)
        acc = float(np.mean(y_pred_tok == y_true_tok)) if len(y_true_tok) else 0.0
        return {"metrics": float(f1), "f1": float(f1), "acc": float(acc)}

    return _compute_metrics_mrpc_f1


def make_compute_metrics_cb_macro_f1(
    candidate_token_ids: tp.Optional[np.ndarray],
    tokenizer,
):
    """SuperGLUE CB primary metric is three-class macro-F1."""

    def _compute_metrics_cb_macro_f1(p: PredictionOutput):
        preds = p.predictions
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        logits0 = _normalize_eval_predictions_for_metrics(logits)
        label_ids = p.label_ids
        y_true_tok = _pick_first_label_token(label_ids)
        y_pred_tok = _decode_pred_from_maybe_compressed_logits(
            logits0, y_true_tok, candidate_token_ids
        )

        try:
            ent_id, con_id, neu_id = _cb_label_token_ids(tokenizer)
        except ValueError as e:
            acc = float(np.mean(y_pred_tok == y_true_tok)) if len(y_true_tok) else 0.0
            log.warning("[CB macro-F1] %s Falling back to metrics=0.", e)
            return {"metrics": 0.0, "macro-f1": 0.0, "acc": float(acc)}

        tok2cls = {ent_id: 0, con_id: 1, neu_id: 2}
        cand = np.array(
            sorted(set(int(x) for x in np.unique(y_true_tok).tolist() if int(x) != -100)),
            dtype=np.int64,
        )
        if cand.size != 3 or any(int(t) not in tok2cls for t in cand.tolist()):
            acc = float(np.mean(y_pred_tok == y_true_tok)) if len(y_true_tok) else 0.0
            log.warning(
                "[CB macro-F1] Label token ids from batch do not match tokenizer("
                "'entailment'/'contradiction'/'neutral') first tokens "
                f"(expected ids={sorted(tok2cls.keys())}, cand={cand.tolist()}). "
                "Falling back to metrics=0."
            )
            return {"metrics": 0.0, "macro-f1": 0.0, "acc": float(acc)}

        def _tok_to_cls(a: np.ndarray) -> np.ndarray:
            out = np.full(len(a), -1, dtype=np.int64)
            for i, t in enumerate(a.astype(np.int64).tolist()):
                if int(t) in tok2cls:
                    out[i] = tok2cls[int(t)]
            return out

        y_true_c = _tok_to_cls(y_true_tok)
        y_pred_c = _tok_to_cls(y_pred_tok)
        macro = _f1_macro_multiclass(y_true_c, y_pred_c, labels=(0, 1, 2))
        acc = float(np.mean(y_pred_tok == y_true_tok)) if len(y_true_tok) else 0.0
        mf = float(macro)
        return {"metrics": mf, "macro-f1": mf, "acc": float(acc)}

    return _compute_metrics_cb_macro_f1


def compute_metrics(p: PredictionOutput):
    """Default token-label accuracy metric for seq2seq classification tasks."""
    preds = p.predictions
    logits = preds[0] if isinstance(preds, (tuple, list)) else preds  # (B, L, V)
    label_ids = p.label_ids                                          # (B, L)

    logits0 = _normalize_eval_predictions_for_metrics(logits)        # (B, V)

    y_true = _pick_first_label_token(label_ids)                      # (B,)
    candidates = np.unique(y_true)                                   # (C,)

    y_pred = _predict_label_from_candidates(logits0, candidates)      # (B,)

    acc = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
    return {"metrics": acc, "acc": acc}


def compute_metrics_cola_mcc(p: PredictionOutput):
    """CoLA primary metric is Matthews correlation coefficient."""
    preds = p.predictions
    logits = preds[0] if isinstance(preds, (tuple, list)) else preds
    label_ids = p.label_ids

    logits0 = _normalize_eval_predictions_for_metrics(logits)
    y_true_tok = _pick_first_label_token(label_ids)
    candidates = np.unique(y_true_tok)

    y_pred_tok = _predict_label_from_candidates(logits0, candidates)

    # Map tokens to {0, 1}; MCC is invariant to swapping the class names.
    cand = np.array(sorted(set(int(x) for x in candidates.tolist() if int(x) != -100)), dtype=np.int64)
    if cand.size != 2:
        acc = float(np.mean(y_pred_tok == y_true_tok)) if len(y_true_tok) else 0.0
        return {"metrics": 0.0, "mcc": 0.0, "acc": float(acc)}


    neg_tok, pos_tok = cand[0], cand[1]
    y_true01 = (y_true_tok == pos_tok).astype(np.int64)
    y_pred01 = (y_pred_tok == pos_tok).astype(np.int64)

    mcc = _mcc_binary(y_true01, y_pred01)
    acc = float(np.mean(y_pred_tok == y_true_tok)) if len(y_true_tok) else 0.0
    pred_pos_rate = float(np.mean(y_pred01)) if len(y_pred01) else 0.0
    true_pos_rate = float(np.mean(y_true01)) if len(y_true01) else 0.0
    tp = int(np.sum((y_true01 == 1) & (y_pred01 == 1)))
    tn = int(np.sum((y_true01 == 0) & (y_pred01 == 0)))
    fp = int(np.sum((y_true01 == 0) & (y_pred01 == 1)))
    fn = int(np.sum((y_true01 == 1) & (y_pred01 == 0)))

    return {"metrics": float(mcc), "mcc": float(mcc), "acc": float(acc)}



def transform_dataset(model_type, tokenizer, dataset, max_length):
    """
    Tokenize once up front instead of tokenizing inside set_transform for every batch.
    Also add a length column for group_by_length batching.
    """
    if {"input_ids", "attention_mask", "labels"}.issubset(set(dataset.column_names)):
        extra_columns = [
            c
            for c in dataset.column_names
            if c not in {"input_ids", "attention_mask", "labels", "length"}
        ]
        if extra_columns:
            dataset = dataset.remove_columns(extra_columns)
        if "length" not in dataset.column_names:
            def _add_pretokenized_length(batch):
                batch["length"] = [len(x) for x in batch["input_ids"]]
                return batch

            dataset = dataset.map(
                _add_pretokenized_length,
                batched=True,
                desc="Adding length column",
            )
        return dataset

    remove_columns = list(dataset.column_names)

    def _add_length(batch):
        batch["length"] = [len(x) for x in batch["input_ids"]]
        return batch

    if model_type == "CausalLM":
        dataset = dataset.map(
            lambda batch: _causal_lm_encode_batched(batch, tokenizer, max_length),
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing causal LM dataset",
        )
        dataset = dataset.map(
            _add_length,
            batched=True,
            desc="Adding length column",
        )
    elif model_type == "ConditionalGeneration":
        dataset = dataset.map(
            lambda batch: _seq2seq_encode_batched(batch, tokenizer, max_length),
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing seq2seq dataset",
        )
        dataset = dataset.map(
            _add_length,
            batched=True,
            desc="Adding length column",
        )
    else:
        raise ValueError("Wrong model type")

    return dataset


def _optimizer_state_bytes_measured(optimizer) -> Optional[int]:
    """
    Sum of CUDA tensor bytes in optimizer.state (actual Adam / AdamW moment estimates, etc.).
    Used to correct theoretical 2*fp32-per-numel accounting when it undercounts (e.g. 8bit Adam,
    fused optimizers, or extra slots). Returns None if unavailable.
    """
    if optimizer is None:
        return None
    opt = optimizer
    # Unwrap a few common trainer wrappers (best-effort).
    for _ in range(4):
        inner = getattr(opt, "optimizer", None)
        if inner is not None and inner is not opt:
            opt = inner
            continue
        break
    try:
        total = 0
        st = getattr(opt, "state", None)
        if st is None:
            return None
        for state in st.values():
            if not isinstance(state, dict):
                continue
            for v in state.values():
                if torch.is_tensor(v) and v.is_cuda:
                    total += int(v.numel() * v.element_size())
        return int(total)
    except Exception:
        return None


def _compute_cuda_memory_breakdown(
    model: torch.nn.Module,
    trainer,  # Trainer with .optimizer (possibly wrapped)
    peak_alloc_bytes: int,
    peak_reserved_bytes: int,
    # Optional: same-time pair at RESERVED peak (used only for diagnostics of the
    # allocator cache/fragmentation behavior; does NOT affect the main quantities
    # we log to W&B).
    reserved_peak_alloc_bytes: Optional[int] = None,
    reserved_peak_reserved_bytes: Optional[int] = None,
    use_care_lora: bool = False,
    use_loract: bool = False,
    use_lorafa: bool = False,
    use_dora: bool = False,
    activation_saved_peak_bytes: Optional[int] = None,
    lora_activation_peak_bytes: Optional[int] = None,
) -> Dict[str, float]:
    """Split the CUDA allocated-memory peak into model, LoRA, activation, and residual buckets."""
    breakdown_version = "reserved_peak_semantic_v17"
    bytes_per_mib = 1024**2
    peak_reserved_mib = float(peak_reserved_bytes) / float(bytes_per_mib)

    _method_bits = [
        bool(use_care_lora),
        bool(use_loract),
        bool(use_lorafa),
        bool(use_dora),
    ]
    if sum(_method_bits) > 1:
        log.warning(
            "[mem-breakdown] multiple method flags are True; memory accounting assumes a single active method."
        )

    # 1. Static non-LoRA state: parameters plus CUDA buffers.
    lora_param_names = {
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
    }

    base_model_params_bytes = 0
    for name, p in model.named_parameters():
        if not torch.is_tensor(p) or not p.is_cuda:
            continue
        parts = set(name.split("."))
        if not parts.intersection(lora_param_names):
            base_model_params_bytes += p.numel() * p.element_size()
    base_model_params_bytes = int(base_model_params_bytes)

    base_model_buffers_bytes = 0
    for name, buf in model.named_buffers():
        if not torch.is_tensor(buf):
            continue
        if not buf.is_cuda:
            continue
        parts = set(name.split("."))
        if parts.intersection(lora_param_names):
            continue
        base_model_buffers_bytes += buf.numel() * buf.element_size()
    base_model_buffers_bytes = int(base_model_buffers_bytes)

    base_model_static_bytes = int(base_model_params_bytes) + int(base_model_buffers_bytes)

    # ------------------------------------------------------------------
    # 2. LoRA parameter bytes by method semantics.
    # ------------------------------------------------------------------
    lora_params_bytes = 0
    lora_a_numel = 0
    lora_a_bytes = 0
    lora_b_numel = 0
    lora_b_bytes = 0
    lora_dora_numel = 0
    lora_dora_bytes = 0
    for name, p in model.named_parameters():
        if not torch.is_tensor(p) or not p.is_cuda:
            continue
        parts = set(name.split("."))
        if parts.intersection(lora_param_names):
            bytes_i = p.numel() * p.element_size()
            lora_params_bytes += bytes_i
            if ("lora_A" in parts) or ("lora_embedding_A" in parts):
                lora_a_numel += p.numel()
                lora_a_bytes += bytes_i
            if ("lora_B" in parts) or ("lora_embedding_B" in parts):
                lora_b_numel += p.numel()
                lora_b_bytes += bytes_i
            if "lora_magnitude_vector" in parts:
                lora_dora_numel += p.numel()
                lora_dora_bytes += bytes_i

    lora_params_bytes = int(lora_params_bytes)

    # 2.5. CARE-LoRA extra state: one saved M* per LoRA layer.
    care_lora_m_linear_bytes_peak = 0
    if use_care_lora:
        try:
            for n, p in model.named_parameters():
                if "lora_A" not in n:
                    continue
                if not torch.is_tensor(p) or not p.is_cuda:
                    continue
                # lora_A weight is [r, in_features] in PEFT row form.
                if p.ndim != 2:
                    continue
                r, in_features = int(p.shape[0]), int(p.shape[1])
                # Linear CARE-LoRA keeps one saved M* with the same logical shape as lora_A.
                # Account it with lora_A's element size: fp32 LoRA uses fp32 M*, while
                # lower-precision LoRA stores the rounded lower-precision M*.
                elem = int(p.element_size())
                care_lora_m_linear_bytes_peak += r * in_features * elem
        except Exception:
            care_lora_m_linear_bytes_peak = 0
    care_lora_m_linear_bytes_peak = int(care_lora_m_linear_bytes_peak)

    care_lora_m_bytes_peak = int(care_lora_m_linear_bytes_peak)

    # ------------------------------------------------------------------
    # 3/4. Gradient + optimizer-state by METHOD-SEMANTICS (paper accounting).
    #
    # We intentionally do NOT rely on runtime lazy allocation in optimizer.state,
    # because that can undercount method cost when some states are materialized
    # late / outside standard optimizer objects.
    #
    # Method accounting (matching the user's table):
    # - LoRA:    optimize A + B with Adam
    # - LoRA-FA: optimize B with Adam, A frozen
    # ------------------------------------------------------------------
    def _adam_state_bytes(numel: int) -> int:
        # exp_avg + exp_avg_sq, fp32
        return 2 * 4 * int(numel)

    # Pick the accounting mode string (for logs only).
    if use_care_lora:
        mode = "care_lora"
    elif use_loract:
        mode = "loract"
    elif use_dora:
        mode = "dora"
    elif use_lorafa:
        mode = "lorafa"
    else:
        mode = "lora"

    if use_care_lora or use_loract:
        # CARE-LoRA and LoRAct optimize A and B with the standard optimizer.
        optimizer_state_bytes = _adam_state_bytes(lora_a_numel + lora_b_numel)
        gradients_bytes = lora_a_bytes + lora_b_bytes
    elif use_dora:
        # DoRA: A + B plus a learnable magnitude vector are optimized by the standard optimizer.
        optimizer_state_bytes = _adam_state_bytes(lora_a_numel + lora_b_numel + lora_dora_numel)
        gradients_bytes = lora_a_bytes + lora_b_bytes + lora_dora_bytes
    elif use_lorafa:
        # LoRA-FA: only B participates in optimizer/autograd
        optimizer_state_bytes = _adam_state_bytes(lora_b_numel)
        gradients_bytes = lora_b_bytes
    else:
        # Standard LoRA
        optimizer_state_bytes = _adam_state_bytes(lora_a_numel + lora_b_numel)
        gradients_bytes = lora_a_bytes + lora_b_bytes

    # Keep optional non-LoRA trainables (e.g. embeddings) in the accounting.
    extra_trainable_bytes = 0
    extra_trainable_numel = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not torch.is_tensor(p) or not p.is_cuda:
            continue
        parts = set(name.split("."))
        if parts.intersection(lora_param_names):
            continue
        extra_trainable_bytes += p.numel() * p.element_size()
        extra_trainable_numel += p.numel()
    if extra_trainable_numel > 0:
        gradients_bytes += extra_trainable_bytes
        optimizer_state_bytes += _adam_state_bytes(extra_trainable_numel)

    optimizer_state_bytes_theoretical = int(optimizer_state_bytes)
    measured_opt_bytes = _optimizer_state_bytes_measured(
        trainer.optimizer if trainer is not None else None
    )
    if measured_opt_bytes is not None:
        if measured_opt_bytes != optimizer_state_bytes_theoretical:
            log.info(
                "[mem-breakdown] optimizer_state theoretical=%d MiB vs measured=%d MiB (using max)",
                int(optimizer_state_bytes_theoretical // bytes_per_mib),
                int(measured_opt_bytes // bytes_per_mib),
            )
        optimizer_state_bytes = max(optimizer_state_bytes_theoretical, int(measured_opt_bytes))
    else:
        optimizer_state_bytes = optimizer_state_bytes_theoretical

    optimizer_state_bytes = int(optimizer_state_bytes)
    gradients_bytes = int(gradients_bytes)

    log.info(
        "[mem-breakdown-components] mode=%s | lora_A_numel=%d | lora_B_numel=%d | "
        "base_params_mib=%.2f | base_buffers_mib=%.2f | "
        "dora_magnitude_mib=%.4f | care_lora_M_mib=%.4f | lora_params_mib=%.2f | optimizer_state_mib=%.2f | gradients_mib=%.2f | extra_trainable_numel=%d",
        mode,
        int(lora_a_numel),
        int(lora_b_numel),
        float(base_model_params_bytes / bytes_per_mib),
        float(base_model_buffers_bytes / bytes_per_mib),
        float(lora_dora_bytes / bytes_per_mib),
        float(care_lora_m_bytes_peak / bytes_per_mib),
        float(lora_params_bytes / bytes_per_mib),
        float(optimizer_state_bytes / bytes_per_mib),
        float(gradients_bytes / bytes_per_mib),
        int(extra_trainable_numel),
    )

    bpm = float(bytes_per_mib)
    optimizer_state_mib = float(optimizer_state_bytes) / bpm
    gradients_mib = float(gradients_bytes) / bpm
    lora_params_mib = float(lora_params_bytes) / bpm
    dora_magnitude_mib = float(lora_dora_bytes) / bpm
    care_lora_m_mib_peak = float(care_lora_m_bytes_peak) / bpm
    care_lora_m_linear_mib_peak = float(care_lora_m_linear_bytes_peak) / bpm

    # 7-8. Split saved-tensor activations from the residual bucket.
    if activation_saved_peak_bytes is None:
        activation_saved_peak_bytes = 0
    activation_saved_peak_bytes = int(activation_saved_peak_bytes)

    optimizer_state_bytes_full_lora = _adam_state_bytes(
        lora_a_numel + lora_b_numel + lora_dora_numel + extra_trainable_numel
    )
    gradients_bytes_full_lora = (lora_a_bytes + lora_b_bytes + lora_dora_bytes) + extra_trainable_bytes
    accounted_static_fixed = (
        int(base_model_static_bytes)
        + int(lora_params_bytes)
        + int(optimizer_state_bytes_full_lora)
        + int(gradients_bytes_full_lora)
        + int(care_lora_m_bytes_peak)
    )
    activation_workspace_bytes = max(0, int(peak_alloc_bytes) - accounted_static_fixed)
    activation_workspace_mib = float(activation_workspace_bytes) / bpm

    lora_bucket_bytes = (
        int(lora_params_bytes)
        + int(optimizer_state_bytes)
        + int(gradients_bytes)
        + int(care_lora_m_bytes_peak)
    )
    accounted_static_method = int(base_model_static_bytes) + int(lora_bucket_bytes)

    peak_alloc_mib = float(int(peak_alloc_bytes)) / bpm
    # model_params_mib means static non-LoRA parameters plus CUDA buffers.
    model_params_mib = float(base_model_static_bytes) / bpm
    lora_mib_para_opti_grad = float(lora_bucket_bytes) / bpm

    # Same-peak residual bytes used to close the full breakdown.
    residual_method_bytes = int(peak_alloc_bytes) - accounted_static_method
    if residual_method_bytes < 0:
        log.warning(
            "[mem-breakdown] peak_alloc < method static sum by %d bytes (timing or accounting mismatch).",
            -residual_method_bytes,
        )
    residual_bytes = max(0, residual_method_bytes)

    activation_saved_raw_bytes = int(activation_saved_peak_bytes)
    activation_saved_mib_raw = float(activation_saved_raw_bytes) / bpm

    if activation_saved_raw_bytes > 0:
        # Clamp activation to residual so the decomposition closes.
        activation_bytes_split = min(activation_saved_raw_bytes, residual_bytes)
        other_bytes = residual_bytes - activation_bytes_split
        activation_mib_source = "saved_tensors_peak_clamped_to_residual"
        if activation_saved_raw_bytes > residual_bytes:
            log.info(
                "[mem-breakdown] saved_tensors_peak (%d B) > residual (%d B); clamping activation to residual.",
                activation_saved_raw_bytes,
                residual_bytes,
            )
    else:
        # Without tracker data, assign the full residual to activation.
        activation_bytes_split = residual_bytes
        other_bytes = 0
        activation_mib_source = "residual_all_activation_no_tracker"
        log.warning(
            "[mem-breakdown] activation_saved_peak_bytes==0: activation_mib=residual, other_mib=0."
        )

    activation_mib = float(activation_bytes_split) / bpm
    other_mib = float(other_bytes) / bpm
    if lora_activation_peak_bytes is None:
        lora_activation_peak_bytes = 0
    lora_activation_z_bytes_split = min(
        max(0, int(lora_activation_peak_bytes)),
        int(activation_bytes_split),
    )
    lora_activation_extra_care_lora_m_bytes = int(care_lora_m_bytes_peak) if use_care_lora else 0
    lora_activation_bytes_split = int(lora_activation_z_bytes_split) + int(lora_activation_extra_care_lora_m_bytes)
    lora_activation_mib = float(lora_activation_bytes_split) / bpm

    # Closure check; MiB values may differ only by floating-point rounding.
    _sum_mib = model_params_mib + lora_mib_para_opti_grad + activation_mib + other_mib
    if abs(_sum_mib - peak_alloc_mib) > 1e-3:
        log.warning(
            "[mem-breakdown] closure mismatch: model+lora+act+other=%.6f vs peak=%.6f MiB",
            float(_sum_mib),
            float(peak_alloc_mib),
        )

    log.info(
        "[mem-breakdown] activation_mib_source=%s | saved_tensors_peak_raw_mib=%.4f | "
        "activation_in_split_mib=%.4f | lora_activation_at_peak_mib=%.4f | "
        "residual_mib=%.4f | activation_workspace(ref)_mib=%.4f | other_mib=%.4f",
        activation_mib_source,
        activation_saved_mib_raw,
        float(activation_mib),
        float(lora_activation_mib),
        float(residual_bytes) / bpm,
        float(activation_workspace_mib),
        float(other_mib),
    )

    # ------------------------------------------------------------------
    # 9. Reserved but not allocated at ALLOCATED peak (diagnostic only).
    #    This bucket includes allocator cache, fragmentation, inactive split blocks,
    #    and similar reserved-but-not-live bytes. We keep it for logging/analysis,
    #    but the main decomposition we care about is strictly on the allocated side.
    # ------------------------------------------------------------------
    # Optional diagnostic: reserved-but-not-allocated at the RESERVED peak.
    other_reserved_at_reserved_peak_mib_diag = None
    if reserved_peak_alloc_bytes is not None and reserved_peak_reserved_bytes is not None:
        other_reserved_at_reserved_peak_bytes = max(
            0,
            int(reserved_peak_reserved_bytes) - int(reserved_peak_alloc_bytes),
        )
        other_reserved_at_reserved_peak_mib_diag = other_reserved_at_reserved_peak_bytes / bytes_per_mib

    out = {
        "breakdown_version": breakdown_version,
        "mode": str(mode),
        # model_params_mib = params_only + base buffers.
        "model_params_only_mib": float(base_model_params_bytes) / bpm,
        "model_buffers_mib": float(base_model_buffers_bytes) / bpm,
        "model_params_mib": float(model_params_mib),
        "lora_params_mib": float(lora_params_mib),
        "dora_magnitude_mib": float(dora_magnitude_mib),
        "optimizer_state_mib": float(optimizer_state_mib),
        "gradients_mib": float(gradients_mib),
        "care_lora_m_mib_peak": float(care_lora_m_mib_peak),
        "care_lora_m_linear_mib_peak": float(care_lora_m_linear_mib_peak),
        "lora_mib_para_opti_grad": float(lora_mib_para_opti_grad),
        "residual_mib": float(residual_bytes) / bpm,
        "activation_saved_tensors_peak_mib": float(activation_saved_mib_raw),
        "activation_mib": float(activation_mib),
        "lora_activation_mib": float(lora_activation_mib),
        "other_mib": float(other_mib),
    }
    if other_reserved_at_reserved_peak_mib_diag is not None:
        out["other_reserved_at_reserved_peak_mib"] = float(other_reserved_at_reserved_peak_mib_diag)
    return out


def _unwrap_trainer_model(trainer: tp.Any) -> torch.nn.Module:
    """Return the unwrapped model for post-training generation evaluation."""
    m = trainer.model
    acc = getattr(trainer, "accelerator", None)
    if acc is not None and hasattr(acc, "unwrap_model"):
        try:
            return acc.unwrap_model(m)
        except Exception:
            return m
    return m


def train_text_to_text_model(
    run_name: str,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_type: str,
    per_device_batch_size: int = 1,
    real_batch_size: int = 32,
    max_length: int = None,
    callbacks: list = None,
    **kwargs,
) -> tp.Tuple[torch.nn.Module, int, tp.Dict[str, tp.Any]]:
    # Preprocess the dataset
    train_dataset = preprocess_dataset(train_dataset)
    valid_dataset = preprocess_dataset(valid_dataset)

    # Infer fixed label-token candidates; prefer tokenized-label inference for correctness.
    task = (kwargs.get("metric_task") or "").lower()
    candidate_token_ids_raw = _infer_candidate_token_ids_from_dataset(valid_dataset, tokenizer)

    # Samples per optimizer step =
    # per_device_train_batch_size * world_size * gradient_accumulation_steps.
    ws = _distributed_world_size()
    per_optimizer_step = per_device_batch_size * ws
    assert (
        real_batch_size % per_optimizer_step == 0
    ), (
        f"real_batch_size ({real_batch_size}) must be divisible by "
        f"per_device_batch_size * WORLD_SIZE ({per_device_batch_size} * {ws} = {per_optimizer_step})"
    )
    accu_step = real_batch_size // per_optimizer_step
    log.info(
        "[batch/ddp] WORLD_SIZE=%s per_device_train_batch_size=%s -> "
        "grad_accumulation_steps=%s (global batch per optimizer step=%s)",
        ws,
        per_device_batch_size,
        accu_step,
        per_device_batch_size * ws * accu_step,
    )

    # Avoid interleaved progress bars from non-main DDP ranks.
    if not _is_trainer_log_main_process():
        try:
            from datasets import disable_progress_bar

            disable_progress_bar()
        except Exception:
            pass

    train_dataset, valid_dataset = transform_dataset(
        model_type, tokenizer, train_dataset, max_length
    ), transform_dataset(model_type, tokenizer, valid_dataset, max_length)
    candidate_token_ids = _infer_candidate_token_ids_from_tokenized_dataset(valid_dataset)
    if candidate_token_ids is None:
        candidate_token_ids = candidate_token_ids_raw

    if model_type == "CausalLM":
        # TF32 can improve matmul throughput on supported GPUs.
        if torch.cuda.is_available() and bool(kwargs.get("cuda_allow_tf32", True)):
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                log.info("CausalLM: cuda.matmul.allow_tf32 / cudnn.allow_tf32 = True")
            except Exception:
                pass
        data_collator = CausalLMDataCollator(tokenizer)
    else:
        data_collator = Seq2SeqDataCollatorStripLength(tokenizer=tokenizer, model=model)

    eval_steps = max(
        1, int(len(train_dataset) * kwargs.get("eval_epochs", 1)) // real_batch_size
    )

    total_train_steps = max(
        1,
        int(np.ceil(len(train_dataset) / max(real_batch_size, 1))) * int(kwargs.get("num_train_epochs", 3))
    )

    requested_logging_steps = int(kwargs.get("logging_steps", 1))

    dataloader_num_workers = int(kwargs.get("dataloader_num_workers", 4))
    dataloader_persistent_workers = bool(
        kwargs.get("dataloader_persistent_workers", dataloader_num_workers > 0)
    )
    group_by_length = bool(kwargs.get("group_by_length", True))

    _disable_tqdm = kwargs.get("disable_tqdm", None)
    if _disable_tqdm is None:
        _disable_tqdm = _should_disable_tqdm()
    wandb_enabled = bool(kwargs.get("wandb_enabled", True))

    # Special for loraplus and custom LoRA paths.
    use_loraplus = kwargs.get("use_loraplus", False)
    use_lorafa = kwargs.get("use_lorafa", False)
    use_care_lora = kwargs.get("use_care_lora", False)
    use_loract = bool(kwargs.get("use_loract", False))
    use_dora = bool(kwargs.get("use_dora", False))
    if use_dora:
        dataset_for_dora = str(kwargs.get("dataset_name", kwargs.get("metric_task", ""))).lower()
        dora_chunked_datasets = {
            "metamathqa",
            "opencodeinstruct",
            "smoltalk",
        }
        dora_row_chunk_mib_by_dataset = {
            "metamathqa": 32.0,
            "opencodeinstruct": 64.0,
            "smoltalk": 128.0,
        }
        dora_enable_chunked_ops = dataset_for_dora in dora_chunked_datasets
        dora_row_chunk_mib = dora_row_chunk_mib_by_dataset.get(dataset_for_dora, 32.0)
        dora_force_narrow_output = dataset_for_dora in dora_chunked_datasets
        dora_narrow_output_ratio = 1.0
        try:
            import peft.tuners.lora.dora as _dora_layer

            _dora_layer.set_dora_enable_chunked_ops(dora_enable_chunked_ops)
            _dora_layer.set_dora_row_chunk_cast_mib(dora_row_chunk_mib)
            _dora_layer.set_dora_force_row_chunk_for_narrow_output(dora_force_narrow_output)
            _dora_layer.set_dora_narrow_output_ratio(dora_narrow_output_ratio)
            log.info(
                "[dora] chunked_ops=%s | row-chunk cast threshold=%.1f MiB | force_narrow_output=%s | narrow_output_ratio=%.1f (dataset_name=%s)",
                bool(dora_enable_chunked_ops),
                dora_row_chunk_mib,
                bool(dora_force_narrow_output),
                dora_narrow_output_ratio,
                dataset_for_dora or "unknown",
            )
        except Exception as e:
            log.warning("[dora] failed to set row-chunk policy: %s", e)
    TrainingArgumentsClass = (
        LoraPlusTrainingArguments if use_loraplus else Seq2SeqTrainingArguments
    )
    # Use the standard HF trainer unless LoRA+ requires its custom trainer.
    TrainerClass = LoraPlusTrainer if use_loraplus else Seq2SeqTrainer
    if use_loraplus:
        additional_kwargs = {
            "loraplus_lr_ratio": kwargs.get("loraplus_lr_ratio", 1.0),
        }
        log.info(
            f"Begin training using LoraPlusTrainer with additional kwargs: {additional_kwargs}"
        )
    else:
        additional_kwargs = {}
        log.info(
            f"Begin training using {TrainerClass.__name__} | optim={kwargs.get('optim', 'adamw_torch')} | weight_decay={kwargs.get('weight_decay', 0.0)}"
        )

    # Training/runtime directories.
    runtime_dir = kwargs.get("runtime_dir")
    os.makedirs(runtime_dir, exist_ok=True)
    output_dir = os.path.join(runtime_dir, "trainer_output")
    logging_dir = os.path.join(runtime_dir, "hf_logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    _eval_strategy = str(kwargs.get("evaluation_strategy", "no")).lower()
    _periodic_eval = _eval_strategy != "no"
    _run_final_trainer_eval = bool(kwargs.get("run_final_trainer_eval", True))
    _do_eval_kw = kwargs.get("do_eval", None)
    if _do_eval_kw is None:
        _do_eval_flag = _periodic_eval
    else:
        _do_eval_flag = bool(_do_eval_kw)
    # Trainer only needs do_eval during periodic evaluation.
    _trainer_eval_dataset = (
        valid_dataset
        if (_periodic_eval or (_eval_strategy == "no" and _run_final_trainer_eval))
        else None
    )

    # When save_strategy=steps and save_steps is omitted, align save_steps with eval_steps.
    _save_strategy = str(kwargs.get("save_strategy", "no")).lower()
    _save_steps_kw = kwargs.get("save_steps", None)
    if _save_strategy == "steps":
        if _save_steps_kw is not None:
            save_steps_arg = max(1, int(_save_steps_kw))
        else:
            save_steps_arg = eval_steps
            log.info(
                "[checkpoint] save_strategy=steps, save_steps not set → save_steps=%d (same as eval_steps). "
                "Trainer saves after every eval; expect a pause after eval logs while checkpoints are written. "
                "Set save_strategy=no (default in run_exp) or set a larger save_steps to reduce stalls.",
                eval_steps,
            )
    else:
        save_steps_arg = None

    _training_args_init_params = inspect.signature(TrainingArgumentsClass.__init__).parameters
    _evaluation_strategy_arg = (
        {"eval_strategy": kwargs.get("evaluation_strategy", "no")}
        if "eval_strategy" in _training_args_init_params
        else {"evaluation_strategy": kwargs.get("evaluation_strategy", "no")}
    )

    training_args = TrainingArgumentsClass(
        output_dir=output_dir,
        num_train_epochs=kwargs.get(
            "num_train_epochs", 3
        ),  # total number of training epochs
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", per_device_batch_size),
        gradient_accumulation_steps=accu_step,
        logging_dir=logging_dir,  # directory for runtime logs
        logging_steps=requested_logging_steps,  # when to print log
        bf16=kwargs.get("bf16", False),
        fp16=kwargs.get("fp16", False),
        gradient_checkpointing=kwargs.get("gradient_checkpointing", False),
        optim=kwargs.get("optim", "adamw_torch"),
        max_grad_norm=kwargs.get("max_grad_norm", 1.0),
        **_evaluation_strategy_arg,
        eval_steps=eval_steps if kwargs.get("evaluation_strategy", "no") != "no" else None,
        save_steps=save_steps_arg,
        save_strategy=kwargs.get("save_strategy", "no"),
        save_total_limit=kwargs.get("save_total_limit", None),
        load_best_model_at_end=kwargs.get("load_best_model_at_end", False),
        metric_for_best_model=kwargs.get(
            "metric_for_best_model",
            (
                "eval_metrics"
                if task in ("cola", "mrpc", "cb")
                else "eval_loss"
            ),
        ),
        greater_is_better=kwargs.get(
            "greater_is_better",
            True if task in ("cola", "mrpc", "cb") else False,
        ),
        do_eval=_do_eval_flag,
        learning_rate=kwargs.get("learning_rate", 5e-4),
        remove_unused_columns=False,  # Dataset is pre-tokenized; keep labels/helper columns intact
        eval_accumulation_steps=kwargs.get("eval_accumulation_steps", None),
        label_names=[
            "labels"
        ],  # Peft are not compatible with HF's default label names yet
        # Ref: https://discuss.huggingface.co/t/eval-with-trainer-not-running-with-peft-lora-model/53286
        weight_decay=kwargs.get("weight_decay", 0.0),
        warmup_ratio=kwargs.get("warmup_ratio", 0.03),
        lr_scheduler_type=kwargs.get("lr_scheduler_type", "cosine"),
        adam_beta1=kwargs.get("adam_beta1", 0.9),
        adam_beta2=kwargs.get("adam_beta2", 0.999),
        adam_epsilon=kwargs.get("adam_epsilon", 1e-8),
        seed=kwargs.get("seed", 42),
        # Keep sampler order deterministic across runs explicitly.
        # CARE-LoRA extra compute scales with effective token count N, so even modest
        # batch-length order drift can amplify wall-time variance vs LoRA.
        data_seed=kwargs.get("data_seed", kwargs.get("seed", 42)),
        report_to=(["wandb"] if _is_trainer_log_main_process() and wandb_enabled else []),
        run_name=run_name,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_persistent_workers=dataloader_persistent_workers,
        group_by_length=group_by_length,
        dataloader_pin_memory=kwargs.get("dataloader_pin_memory", True),
        disable_tqdm=_disable_tqdm,
        **additional_kwargs,
    )

    # Peak-memory profiling installs saved_tensors_hooks and a sampler thread.
    # It does not change any method's forward/backward implementation.
    track_cuda_peak = not (str(kwargs.get("track_cuda_peak", "true")).lower() in {"false", "0", "no"})
    if use_care_lora:
        try:
            import peft.tuners.lora.layer as _lora_layer

            _lora_layer._CARE_LORA_USE_SPEED_PATH = False
            log.info(
                "[care_lora] saved-state layout=%s (track_cuda_peak=%s; profiling flag does not change impl)",
                "save_for_backward hook-observable",
                bool(track_cuda_peak),
            )
        except Exception as e:
            log.warning("[care_lora] failed to set main path impl flag: %s", e)
    callback_list = []
    peak_tracker_cb = None
    peak_pair_sampler = None
    enable_early_stopping = bool(kwargs.get("enable_early_stopping", False)) and kwargs.get("evaluation_strategy", "no") != "no"
    if enable_early_stopping:
        callback_list.append(
            EarlyStoppingCallback(
                early_stopping_patience=kwargs.get("early_stopping_patience", 1)
            )
        )
    if track_cuda_peak and torch.cuda.is_available():
        peak_tracker_cb = ReservedPeakTrackerCallback()
        peak_pair_sampler = CudaPeakPairSampler(
            poll_interval_s=float(kwargs.get("cuda_peak_poll_interval_s", 0.005))
        )
        callback_list.append(peak_tracker_cb)
    if callbacks:
        callback_list.extend(callbacks)

    if task == "cola":
        cm = make_compute_metrics_cola_mcc(candidate_token_ids)
    elif task == "mrpc":
        cm = make_compute_metrics_mrpc_f1(candidate_token_ids, tokenizer)
    elif task == "cb":
        cm = make_compute_metrics_cb_macro_f1(candidate_token_ids, tokenizer)
    else:
        cm = make_compute_metrics(candidate_token_ids)
    preprocess_logits_for_metrics = _build_preprocess_logits_for_metrics(candidate_token_ids)

    # Enable Trainer classification metrics by default for seq2seq tasks.
    enable_compute_metrics = kwargs.get("enable_compute_metrics", None)
    if enable_compute_metrics is None:
        enable_compute_metrics = str(model_type) == "ConditionalGeneration"
    enable_compute_metrics = bool(enable_compute_metrics)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=_trainer_eval_dataset,
        data_collator=data_collator,
        compute_metrics=cm if enable_compute_metrics else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if enable_compute_metrics else None,
        callbacks=callback_list,
    )
    trainer_init_params = inspect.signature(TrainerClass.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if candidate_token_ids is not None:
        log.info(f"[eval fast-path] Using compressed first-token logits with {len(candidate_token_ids)} fixed label candidates: {candidate_token_ids.tolist()}")

    trainer = TrainerClass(**trainer_kwargs)

    _install_care_lora_attention_mask_context(
        model,
        enabled=(
            model_type == "CausalLM"
            and bool(use_care_lora)
        ),
    )

    # PEFT + gradient checkpointing requires input gradients for frozen-backbone paths.
    if bool(kwargs.get("gradient_checkpointing", False)):
        try:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                log.info(
                    "gradient_checkpointing=True: called model.enable_input_require_grads()."
                )
        except Exception as e:
            log.warning("gradient_checkpointing: enable_input_require_grads failed: %s", e)

    # Paper-friendly sanity report (helps ensure baselines/method are configured as intended)
    _runtime_startup_report(
        trainer,
        model,
        use_care_lora=use_care_lora,
        use_lorafa=use_lorafa,
        use_loraplus=use_loraplus,
        use_loract=use_loract,
    )

    # ===== GPU peak memory tracking =====
    # Record the peak allocated/reserved CUDA memory during training.
    if track_cuda_peak and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
            if peak_pair_sampler is not None:
                peak_pair_sampler.start()
        except Exception:
            pass

    activation_tracker = None
    if track_cuda_peak and torch.cuda.is_available():
        excluded_ptrs: set[int] = set()
        excluded_storage_ptrs: set[int] = set()
        excluded_shapes: set[tuple[int, ...]] = set()
        try:
            exclude_care_lora_m_shapes = bool(use_care_lora)
            for name, p in model.named_parameters():
                if torch.is_tensor(p) and p.is_cuda:
                    excluded_ptrs.add(int(p.data_ptr()))
                    excluded_storage_ptrs.add(int(p.untyped_storage().data_ptr()))
                # CARE-LoRA's saved M* is not a parameter, but it often shares the
                # same shape as lora_A [r, in_features]. Exclude that shape for
                # CARE-LoRA runs because M* is already counted in the LoRA bucket.
                if exclude_care_lora_m_shapes and ("lora_A" in name or "lora_embedding_A" in name) and torch.is_tensor(p):
                    try:
                        shape = tuple(int(x) for x in p.shape)
                        excluded_shapes.add(shape)
                        # Some implementations may store a transposed view.
                        if len(shape) == 2:
                            excluded_shapes.add((shape[1], shape[0]))
                    except Exception:
                        pass
            for _, b in model.named_buffers():
                if torch.is_tensor(b) and b.is_cuda:
                    excluded_ptrs.add(int(b.data_ptr()))
                    excluded_storage_ptrs.add(int(b.untyped_storage().data_ptr()))
        except Exception:
            pass
        activation_tracker = SavedTensorActivationTracker(
            excluded_ptrs=excluded_ptrs,
            excluded_storage_ptrs=excluded_storage_ptrs,
            excluded_shapes=excluded_shapes,
            include_last_dims=None,
        )
    train_ctx = (
        torch.autograd.graph.saved_tensors_hooks(
            activation_tracker.pack,
            activation_tracker.unpack,
        )
        if activation_tracker is not None
        else nullcontext()
    )
    lora_activation_tracking_setter = None
    if activation_tracker is not None:
        try:
            from peft.tuners.lora import layer as _lora_layer

            lora_activation_tracking_setter = _lora_layer.set_lora_activation_tracking_enabled
        except Exception:
            lora_activation_tracking_setter = None
    # Peaks taken *after train, before eval* so they align with `activation_mib`
    # (saved_tensors_hooks only wrap `trainer.train()`). Reading max_memory after
    # `evaluate()` mixes in no_grad forward activations that are not counted by
    # the autograd saved-tensor tracker, which inflates `other_mib` spuriously.
    cuda_peak_alloc_train_bytes: Optional[int] = None
    cuda_peak_reserved_train_bytes: Optional[int] = None

    _t_train_wall0 = time.perf_counter()
    try:
        if lora_activation_tracking_setter is not None:
            lora_activation_tracking_setter(True)
        with train_ctx:
            trainer.train()
    finally:
        if lora_activation_tracking_setter is not None:
            lora_activation_tracking_setter(False)
    _train_wall_seconds = float(time.perf_counter() - _t_train_wall0)
    _train_time_minutes = _train_wall_seconds / 60.0
    if _is_trainer_log_main_process():
        log.info(
            "[train wall time] %.4f min (%.2f s) | trainer.train() only (excludes post-train eval / GSM8K)",
            _train_time_minutes,
            _train_wall_seconds,
        )
        try:
            if getattr(wandb, "run", None) is not None:
                wandb.log({"train/time": float(_train_time_minutes)})
                wandb.summary["train/time"] = float(_train_time_minutes)
        except Exception as e_wb:
            log.warning("wandb log train/time failed: %s", e_wb)

    if track_cuda_peak and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            pa = int(torch.cuda.max_memory_allocated())
            pr = int(torch.cuda.max_memory_reserved())
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    local = torch.tensor([pa, pr], device="cuda", dtype=torch.long)
                    world = int(dist.get_world_size())
                    gathered = [torch.zeros_like(local) for _ in range(world)]
                    dist.all_gather(gathered, local)
                    stack = torch.stack(gathered, dim=0)
                    best = int(torch.argmax(stack[:, 0]).item())
                    pa = int(stack[best, 0].item())
                    pr = int(stack[best, 1].item())
            except Exception:
                pass
            cuda_peak_alloc_train_bytes = pa
            cuda_peak_reserved_train_bytes = pr
        except Exception:
            pass

    # When evaluation_strategy=no, optionally run one validation pass after training.
    final_eval_metrics: tp.Dict[str, tp.Any] = {}
    if (
        str(kwargs.get("evaluation_strategy", "no")).lower() == "no"
        and bool(kwargs.get("run_final_trainer_eval", True))
    ):
        try:
            final_eval_metrics = dict(trainer.evaluate() or {})
            log.info(f"[final eval] {final_eval_metrics}")
        except Exception as e:
            log.warning(f"Final evaluation failed: {type(e).__name__}: {e}")

    if peak_pair_sampler is not None:
        try:
            peak_pair_sampler.stop()
        except Exception:
            pass

    if track_cuda_peak and torch.cuda.is_available():
        try:
            sampled_alloc_at_reserved = getattr(peak_tracker_cb, "allocated_at_max_reserved_bytes", 0) or 0
            sampled_reserved_peak = getattr(peak_tracker_cb, "max_reserved_bytes", 0) or 0
            peak_stage = str(getattr(peak_tracker_cb, "max_reserved_stage", "unknown"))
            peak_step = int(getattr(peak_tracker_cb, "max_reserved_global_step", -1))

            # Decomposition uses peak allocated **during training only** (captured
            # before `evaluate()`), so it matches `activation_mib` (saved_tensors only
            # while `trainer.train()` runs). Post-train `evaluate()` can raise the
            # historical max without contributing to `activation_mib`, which would
            # otherwise distort `other_mib`.
            if cuda_peak_alloc_train_bytes is not None:
                peak_alloc = int(cuda_peak_alloc_train_bytes)
                peak_reserved = int(cuda_peak_reserved_train_bytes or 0)
                hist_peak_alloc = peak_alloc
                hist_peak_reserved = peak_reserved
            else:
                hist_peak_alloc = int(torch.cuda.max_memory_allocated())
                hist_peak_reserved = int(torch.cuda.max_memory_reserved())
                peak_alloc = hist_peak_alloc
                peak_reserved = hist_peak_reserved
                try:
                    import torch.distributed as dist

                    if dist.is_available() and dist.is_initialized():
                        local_peaks = torch.tensor(
                            [
                                peak_alloc,
                                peak_reserved,
                                sampled_alloc_at_reserved,
                                sampled_reserved_peak,
                                hist_peak_alloc,
                                hist_peak_reserved,
                            ],
                            device="cuda",
                            dtype=torch.long,
                        )
                        world = dist.get_world_size()
                        gathered = [torch.zeros_like(local_peaks) for _ in range(world)]
                        dist.all_gather(gathered, local_peaks)
                        stack = torch.stack(gathered, dim=0)

                        best_breakdown_idx = int(torch.argmax(stack[:, 0]).item())
                        peak_alloc = stack[best_breakdown_idx, 0].item()
                        peak_reserved = stack[best_breakdown_idx, 1].item()

                        best_sampled_idx = int(torch.argmax(stack[:, 3]).item())
                        sampled_alloc_at_reserved = stack[best_sampled_idx, 2].item()
                        sampled_reserved_peak = stack[best_sampled_idx, 3].item()
                        best_hist_idx = int(torch.argmax(stack[:, 5]).item())
                        hist_peak_alloc = stack[best_hist_idx, 4].item()
                        hist_peak_reserved = stack[best_hist_idx, 5].item()
                except Exception:
                    pass

            sampler_alloc_at_reserved = getattr(peak_pair_sampler, "allocated_at_max_reserved_bytes", 0) or 0
            sampler_reserved_peak = getattr(peak_pair_sampler, "max_reserved_bytes", 0) or 0
            sampler_peak_alloc = getattr(peak_pair_sampler, "max_allocated_bytes", 0) or 0
            sampler_reserved_at_alloc = getattr(peak_pair_sampler, "reserved_at_max_allocated_bytes", 0) or 0

            peak_alloc_mib = peak_alloc / (1024**2)
            peak_reserved_mib = peak_reserved / (1024**2)
            sampled_alloc_at_reserved_mib = sampled_alloc_at_reserved / (1024**2)
            sampled_reserved_peak_mib = sampled_reserved_peak / (1024**2)
            hist_peak_alloc_mib = hist_peak_alloc / (1024**2)
            hist_peak_reserved_mib = hist_peak_reserved / (1024**2)
            log.info(
                f"[CUDA peak@reserved-sampled] allocated_at_reserved_peak={sampled_alloc_at_reserved_mib:.2f} MiB | "
                f"reserved_peak_sampled={sampled_reserved_peak_mib:.2f} MiB | step={peak_step} | stage={peak_stage}"
            )
            log.info(
                f"[CUDA peak@history] allocated_peak={hist_peak_alloc_mib:.2f} MiB | reserved_peak={hist_peak_reserved_mib:.2f} MiB"
            )
            log.info(
                f"[CUDA peak@breakdown] allocated_peak={peak_alloc_mib:.2f} MiB | reserved_peak={peak_reserved_mib:.2f} MiB "
                f"(train only; matches activation_mib scope)."
            )
            if peak_pair_sampler is not None and int(sampler_peak_alloc) > 0:
                log.info(
                    "[CUDA peak@sampler] max_allocated=%.4f MiB (background poll; decomposition uses torch.cuda.max_memory_allocated / DDP-reduced peak above)",
                    float(sampler_peak_alloc) / (1024**2),
                )
            # Best-effort W&B logging: total + detailed breakdown
            try:
                breakdown = _compute_cuda_memory_breakdown(
                    model,
                    trainer,
                    peak_alloc,
                    peak_reserved,
                    reserved_peak_alloc_bytes=sampled_alloc_at_reserved,
                    reserved_peak_reserved_bytes=sampled_reserved_peak,
                    use_care_lora=use_care_lora,
                    use_loract=use_loract,
                    use_lorafa=use_lorafa,
                    use_dora=use_dora,
                    activation_saved_peak_bytes=(
                        int(getattr(activation_tracker, "peak_live_bytes", 0))
                        if activation_tracker is not None
                        else None
                    ),
                    lora_activation_peak_bytes=(
                        int(getattr(activation_tracker, "lora_activation_bytes_at_peak", 0))
                        if activation_tracker is not None
                        else None
                    ),
                )
                log.info(
                    "[CUDA breakdown] allocated=%.6f MiB | reserved=%.6f MiB | model_params=%.6f | "
                    "lora(para+opti+grad)=%.6f | activation=%.6f | lora_activation=%.6f | other=%.6f"
                    % (
                        float(peak_alloc_mib),
                        float(peak_reserved_mib),
                        breakdown["model_params_mib"],
                        breakdown["lora_mib_para_opti_grad"],
                        breakdown["activation_mib"],
                        breakdown["lora_activation_mib"],
                        breakdown["other_mib"],
                    )
                )
                wandb_payload = {
                    "cuda_peak/allocated_mib": float(peak_alloc_mib),
                    "cuda_peak/model_params_mib": float(breakdown["model_params_mib"]),
                    "cuda_peak/lora_mib_para_opti_grad": float(breakdown["lora_mib_para_opti_grad"]),
                    "cuda_peak/activation_mib": float(breakdown["activation_mib"]),
                    "cuda_peak/lora_activation_mib": float(breakdown["lora_activation_mib"]),
                    "cuda_peak/other_mib": float(breakdown["other_mib"]),
                }
                # Use Trainer's default W&B step ordering.
                if _is_trainer_log_main_process():
                    wandb.log(wandb_payload)
                    # Also push to summary so they appear in run summary
                    wandb.summary.update(wandb_payload)
            except Exception as e_breakdown:
                log.warning(f"CUDA breakdown or wandb log failed: {e_breakdown}")
                try:
                    if _is_trainer_log_main_process():
                        _partial = {
                            "cuda_peak/allocated_mib": float(peak_alloc_mib),
                        }
                        wandb.log(_partial)
                        wandb.summary.update(_partial)
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"CUDA peak memory tracking failed: {type(e).__name__}: {e}")

    # Remove trainer_output after run so results/ does not accumulate checkpoints
    try:
        if output_dir and os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
            log.info(f"[cleanup] Removed trainer_output: {output_dir}")
    except Exception as e:
        log.warning(f"Failed to remove trainer_output: {type(e).__name__}: {e}")

    global_step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    return _unwrap_trainer_model(trainer), global_step, final_eval_metrics


def model_inference(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_text: str,
    model_type: str,
    max_source_length: tp.Optional[int] = 768,
    max_target_length: int = 256,
    append_space: bool = True,
):
    if model_type == "CausalLM":
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "return_token_type_ids": False,
        }
        if max_source_length is not None and int(max_source_length) > 0:
            tokenizer_kwargs["max_length"] = int(max_source_length)
            tokenizer_kwargs["truncation"] = True
        else:
            tokenizer_kwargs["truncation"] = False
        inputs = tokenizer(input_text + (" " if append_space else ""), **tokenizer_kwargs)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            # Greedy decoding for reproducible final-generation metrics.
            outputs = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_target_length,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )
        pred_text = tokenizer.decode(
            outputs.sequences[0][len(inputs["input_ids"][0]) :],
            skip_special_tokens=True,
        )
    elif model_type == "ConditionalGeneration":
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_target_length)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pred_text


def _extract_gsm8k_numeric_answer(text: str) -> tp.Optional[float]:
    """Extract the final GSM8K numeric answer.

    GSM8K gold answers mark the final value with ``####``. For model outputs,
    match the common lm-evaluation-harness flexible-extract behavior by falling
    back to the last generated number when the marker is absent.
    """
    match = re.search(r"####\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)", text or "")
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return _extract_last_number(text)


def _extract_last_number(text: str) -> tp.Optional[float]:
    """Extract the last plain decimal/integer from generated text."""
    if text is None:
        return None
    matches = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", str(text))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _numeric_equal(pred: tp.Optional[float], gt: float, *, rel_tol: float = 1e-4, abs_tol: float = 1e-4) -> bool:
    if pred is None:
        return False
    try:
        return bool(np.isclose(float(pred), float(gt), rtol=rel_tol, atol=abs_tol))
    except Exception:
        return False


@torch.no_grad()
def evaluate_gsm_hard_accuracy(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    model_type: str,
    *,
    max_source_length: int = 512,
    max_new_tokens: int = 512,
) -> tp.Tuple[float, int, int]:
    """Evaluate on GSM-Hard (reasoning-machines/gsm-hard), using numeric answer match."""
    if model_type != "CausalLM":
        raise ValueError(f"evaluate_gsm_hard_accuracy only supports CausalLM, got model_type={model_type!r}")

    try:
        dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
    except Exception as e:
        msg = str(e).lower()
        is_net = "connection" in msg or "couldn't reach" in msg or "offline" in msg
        if not is_net:
            raise
        dataset = load_dataset(
            "reasoning-machines/gsm-hard",
            split="train",
            download_config=DownloadConfig(local_files_only=True),
        )

    model.eval()
    n = 0
    correct = 0
    for example in tqdm(dataset, desc="gsm_hard_eval", disable=_should_disable_tqdm()):
        question = str(example["input"])
        prompt = f"Q: {question}\nA: "
        pred_text = model_inference(
            model,
            tokenizer,
            prompt,
            model_type,
            max_source_length=max_source_length,
            max_target_length=max_new_tokens,
        )
        pred = _extract_last_number(pred_text)
        correct += int(_numeric_equal(pred, float(example["target"])))
        n += 1
    return float(correct) / float(max(n, 1)), n, correct


@torch.no_grad()
def evaluate_gsm8k_test_accuracy(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    model_type: str,
    *,
    max_source_length: int = 512,
    max_new_tokens: int = 512,
) -> tp.Tuple[float, int, int]:
    """Evaluate final accuracy on GSM8K main/test with numeric answer matching."""
    from data import load_gsm8k

    if model_type != "CausalLM":
        raise ValueError(f"evaluate_gsm8k_test_accuracy only supports CausalLM, got model_type={model_type!r}")

    model.eval()
    _, _, test_set = load_gsm8k()
    n = 0
    correct = 0
    for example in tqdm(
        test_set,
        desc="gsm8k_test_eval",
        disable=_should_disable_tqdm(),
    ):
        pred_text = model_inference(
            model,
            tokenizer,
            example["x"],
            model_type,
            max_source_length=max_source_length,
            max_target_length=max_new_tokens,
        )
        gt = _extract_gsm8k_numeric_answer(example["y"])
        pred = _extract_gsm8k_numeric_answer(pred_text)
        correct += int(_numeric_equal(pred, float(gt)) if gt is not None else False)
        n += 1
    acc = float(correct) / float(max(n, 1))
    return acc, n, correct


HUMANEVAL_PREFIX_TEMPLATE = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code:
Notes: respond only with the Python code completion that should be appended after the given code
do not repeat the function signature or any imports already provided in the code
do not add explanations or Markdown, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{prompt}
```

### Response:
```python
"""


def _post_process_humaneval_completion(text: str) -> str:
    text = (text or "").replace("```python", "```")
    if "```" in text:
        parts = text.split("```")
        fenced_parts = [part for part in parts[1::2] if part.strip()]
        code_parts = [
            part
            for part in parts
            if re.search(r"^\s*(def\s+|from\s+\S+\s+import\s+|import\s+)", part, flags=re.MULTILINE)
        ]
        if code_parts:
            text = code_parts[0]
        elif fenced_parts:
            text = fenced_parts[0]
        else:
            nonempty_parts = [part for part in parts if part.strip()]
            text = nonempty_parts[0] if nonempty_parts else parts[0]
    text = text.replace("```", "").replace("\t", "    ")
    lines = [ll.rstrip() for ll in text.splitlines() if ll.strip()]
    if not lines:
        return ""
    try:
        def_idx = next(i for i, line in enumerate(lines) if re.match(r"^\s*def\s+", line))
        leading_code = []
        for line in lines[:def_idx]:
            stripped = line.strip()
            if re.match(r"^(from\s+\S+\s+import\s+|import\s+)", stripped):
                leading_code.append(line)
            elif stripped.startswith("@"):
                leading_code.append(line)
        lines = leading_code + lines[def_idx:]
    except StopIteration:
        pass
    spaces = [len(re.match(r"^( *)", line).group(1)) for line in lines]
    try:
        first_def_idx = next(i for i, line in enumerate(lines) if re.match(r"^\s*def\s+", line))
        base = spaces[first_def_idx]
    except StopIteration:
        base = spaces[0] if spaces else 0
    normalized = []
    for line, sp in zip(lines, spaces):
        normalized.append(line[base:] if sp >= base else line.lstrip())
    return "\n".join(normalized)


def _common_leading_spaces(lines: tp.List[str]) -> int:
    spaces = [len(re.match(r"^( *)", line).group(1)) for line in lines if line.strip()]
    return min(spaces) if spaces else 0


def _indent_humaneval_completion_body(text: str) -> str:
    """Return a HumanEval completion body indented under the prompted function."""
    lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    base = _common_leading_spaces(lines)
    body_lines = []
    for line in lines:
        sp = len(re.match(r"^( *)", line).group(1))
        stripped = line[base:] if sp >= base else line.lstrip()
        body_lines.append("    " + stripped)
    return "\n".join(body_lines) + "\n"


def _normalize_humaneval_body_lines(lines: tp.List[str]) -> tp.List[str]:
    lines = [line.rstrip() for line in lines if line.strip()]
    if not lines:
        return []
    base = _common_leading_spaces(lines)
    normalized = []
    for line in lines:
        sp = len(re.match(r"^( *)", line).group(1))
        stripped = line[base:] if sp >= base else line.lstrip()
        normalized.append("    " + stripped)
    return normalized


def _extract_target_function_body(code: str, entry_point: str) -> tp.Optional[str]:
    lines = (code or "").splitlines()
    if not lines or not entry_point:
        return None
    target_re = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(")
    try:
        def_idx = next(i for i, line in enumerate(lines) if target_re.match(line))
    except StopIteration:
        return None

    def_indent = len(re.match(r"^( *)", lines[def_idx]).group(1))
    body_start = None
    body_end = len(lines)
    for idx in range(def_idx + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        sp = len(re.match(r"^( *)", line).group(1))
        if sp <= def_indent:
            body_end = idx
            break
        if body_start is None:
            body_start = idx
    if body_start is None:
        return None

    leading_imports = []
    for line in lines[:def_idx]:
        stripped = line.strip()
        if re.match(r"^(from\s+\S+\s+import\s+|import\s+)", stripped):
            leading_imports.append("    " + stripped)

    body_lines = []
    for line in lines[body_start:body_end]:
        if not line.strip():
            body_lines.append("")
            continue
        body_lines.append(line)
    body_lines = _normalize_humaneval_body_lines(body_lines)
    completion_lines = leading_imports + body_lines
    if not completion_lines:
        return None
    return "\n".join(completion_lines) + "\n"


def _get_evalplus_runtime_cache_dir(dataset_name: str) -> str:
    override = os.environ.get("CARE_LORA_EVALPLUS_RUNTIME_CACHE_DIR", "").strip()
    if override:
        return os.path.join(override, dataset_name)
    return os.path.join(_DEFAULT_EVALPLUS_RUNTIME_CACHE_ROOT, dataset_name)


def _make_humaneval_evalplus_sample(
    task_id: str,
    model_code: str,
    entry_point: str,
) -> tp.Tuple[tp.Dict[str, str], str]:
    """EvalPlus canonical HumanEval sample: completion appended to prompt."""
    completion = _extract_target_function_body(model_code, entry_point)
    if completion is not None:
        return {"task_id": task_id, "completion": completion}, "full_function_to_completion"
    return {
        "task_id": task_id,
        "completion": _indent_humaneval_completion_body(model_code),
    }, "body_completion"


def _evalplus_status_success(entry: tp.Any, success_value: str = "success") -> bool:
    if isinstance(entry, (list, tuple)) and entry:
        return str(entry[0]) == str(success_value)
    if isinstance(entry, dict):
        for key in ("status", "result"):
            if key in entry:
                return str(entry[key]) == str(success_value)
    return str(entry) == str(success_value)


def _add_evalplus_pass_at_k(report: tp.Dict[str, tp.Any], evalplus_module) -> None:
    """Normalize EvalPlus raw task results into ``pass_at_k`` metrics."""
    eval_items = list((report.get("eval") or {}).values())
    if not eval_items:
        return

    estimate_pass_at_k = getattr(evalplus_module, "estimate_pass_at_k", None)
    if estimate_pass_at_k is None:
        return
    success_value = str(getattr(evalplus_module, "SUCCESS", "success"))

    total = np.array(
        [
            int(res.get("nfiles", len(res.get("base", []) or [])))
            for res in eval_items
        ],
        dtype=np.int64,
    )
    if total.size == 0:
        return

    base_correct = np.array(
        [
            sum(
                1
                for item in (res.get("base", []) or [])
                if _evalplus_status_success(item, success_value)
            )
            for res in eval_items
        ],
        dtype=np.int64,
    )

    pass_at_k: tp.Dict[str, tp.Dict[str, float]] = {}
    base_metrics = {
        f"pass@{k}": float(estimate_pass_at_k(total, base_correct, k).mean())
        for k in (1, 10, 100)
        if int(total.min()) >= k
    }
    if base_metrics:
        pass_at_k["base"] = base_metrics

    plus_correct = []
    has_plus = False
    for res in eval_items:
        base_rows = res.get("base", []) or []
        plus_rows = res.get("plus", []) or []
        if plus_rows:
            has_plus = True
            plus_correct.append(
                sum(
                    1
                    for base_item, plus_item in zip(base_rows, plus_rows)
                    if _evalplus_status_success(base_item, success_value)
                    and _evalplus_status_success(plus_item, success_value)
                )
            )
        else:
            plus_correct.append(0)
    if has_plus:
        plus_correct_arr = np.array(plus_correct, dtype=np.int64)
        plus_metrics = {
            f"pass@{k}": float(estimate_pass_at_k(total, plus_correct_arr, k).mean())
            for k in (1, 10, 100)
            if int(total.min()) >= k
        }
        if plus_metrics:
            pass_at_k["plus"] = plus_metrics

    if pass_at_k:
        report["pass_at_k"] = pass_at_k
        if "base" in pass_at_k:
            report["base"] = pass_at_k["base"]
        if "plus" in pass_at_k:
            report["plus"] = pass_at_k["plus"]


@torch.no_grad()
def evaluate_humaneval_pass1(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    model_type: str,
    *,
    output_dir: str,
    max_source_length: int = 1024,
    max_new_tokens: int = 512,
    evalplus_parallel: int = 1,
    plus: bool = False,
    evalplus_cache_dir: tp.Optional[str] = None,
) -> tp.Dict[str, tp.Any]:
    """
    Generate HumanEval code completions and score them with EvalPlus.

    Samples are written in EvalPlus's canonical HumanEval ``completion`` format,
    so EvalPlus evaluates ``problem["prompt"] + completion``. If the model still
    returns a full function, the target function body is converted back to a
    completion before evaluation.

    ``plus=False`` is the current Code experiment setting: HumanEval prompts with
    the base HumanEval tests only. ``plus=True`` keeps the previous HumanEval+
    behavior and reports the stricter plus-test pass@1 metric.
    """
    if model_type != "CausalLM":
        raise ValueError(f"evaluate_humaneval_pass1 only supports CausalLM, got model_type={model_type!r}")
    try:
        import evalplus.data.humaneval as evalplus_humaneval
        import evalplus.data.utils as evalplus_data_utils
        import evalplus.evaluate as evalplus_evaluate_module
        from evalplus.data import get_human_eval_plus, write_jsonl
        from evalplus.evaluate import evaluate as evalplus_evaluate
    except ImportError as e:
        raise ImportError(
            "HumanEval/HumanEval+ evaluation requires evalplus. Use the pinned version: "
            "pip install evalplus==0.2.0 appdirs==1.4.4 multipledispatch==1.0.0 tempdir==0.7.1 wget==3.2"
        ) from e

    os.makedirs(output_dir, exist_ok=True)
    evalplus_runtime_cache_dir = _get_evalplus_runtime_cache_dir("humaneval")
    os.makedirs(evalplus_runtime_cache_dir, exist_ok=True)
    evalplus_data_utils.CACHE_DIR = evalplus_runtime_cache_dir
    evalplus_humaneval.CACHE_DIR = evalplus_runtime_cache_dir
    try:
        evalplus_evaluate_module.CACHE_DIR = evalplus_runtime_cache_dir
    except Exception:
        pass
    evalplus_override_path = None
    if evalplus_cache_dir:
        os.makedirs(evalplus_cache_dir, exist_ok=True)
        plus_path = os.path.join(evalplus_cache_dir, "HumanEvalPlus-v0.1.9.jsonl")
        if os.path.isfile(plus_path):
            evalplus_override_path = plus_path
            evalplus_humaneval.HUMANEVAL_OVERRIDE_PATH = plus_path
    # EvalPlus serves the HumanEval prompts and can run either the base tests
    # only or the additional HumanEval+ tests, controlled by ``base_only`` below.
    problems = get_human_eval_plus()
    suffix = "humaneval_plus" if plus else "humaneval"
    samples_path = os.path.join(output_dir, f"{suffix}_samples.jsonl")
    results_path = os.path.join(output_dir, f"{suffix}_eval.json")
    evalplus_results_path = samples_path.replace(".jsonl", "_eval_results.json")

    model.eval()
    samples = []
    sample_format_counts: Counter = Counter()
    for task_id, problem in tqdm(problems.items(), desc=f"{suffix}_codegen", disable=_should_disable_tqdm()):
        prompt = HUMANEVAL_PREFIX_TEMPLATE.format(prompt=problem["prompt"])
        pred_text = model_inference(
            model,
            tokenizer,
            prompt,
            model_type,
            max_source_length=max_source_length,
            max_target_length=max_new_tokens,
        )
        sample, sample_format = _make_humaneval_evalplus_sample(
            task_id,
            _post_process_humaneval_completion(pred_text),
            str(problem.get("entry_point", "")),
        )
        samples.append(sample)
        sample_format_counts[sample_format] += 1
    write_jsonl(samples_path, samples)

    for stale_path in {results_path, evalplus_results_path}:
        if os.path.exists(stale_path):
            os.remove(stale_path)

    try:
        evalplus_flags = argparse.Namespace(
            dataset="humaneval",
            samples=samples_path,
            base_only=not bool(plus),
            parallel=int(evalplus_parallel),
            i_just_wanna_run=False,
            test_details=False,
            min_time_limit=0.2,
            gt_time_limit_factor=4.0,
            mini=False,
        )
        evalplus_evaluate(evalplus_flags)
    except TypeError:
        try:
            evalplus_evaluate(
                dataset="humaneval",
                samples=samples_path,
                parallel=int(evalplus_parallel),
                output_file=results_path,
                i_just_wanna_run=True,
                base_only=not bool(plus),
            )
        except TypeError:
            runner = (
                "import argparse, sys; "
                "import evalplus.data.utils as du; "
                "import evalplus.data.humaneval as he; "
                "import evalplus.evaluate as ev; "
                "cache_dir, samples, override, plus_flag, parallel = sys.argv[1:6]; "
                "du.CACHE_DIR = cache_dir; he.CACHE_DIR = cache_dir; ev.CACHE_DIR = cache_dir; "
                "he.HUMANEVAL_OVERRIDE_PATH = override or None; "
                "flags = argparse.Namespace("
                "dataset='humaneval', samples=samples, base_only=(plus_flag != '1'), "
                "parallel=int(parallel), i_just_wanna_run=False, test_details=False, "
                "min_time_limit=0.2, gt_time_limit_factor=4.0, mini=False"
                "); "
                "ev.evaluate(flags)"
            )
            cmd = [
                sys.executable,
                "-c",
                runner,
                evalplus_runtime_cache_dir,
                samples_path,
                evalplus_override_path or "",
                "1" if bool(plus) else "0",
                str(int(evalplus_parallel)),
            ]
            subprocess.run(cmd, check=True)

    report: tp.Dict[str, tp.Any] = {
        "samples_path": samples_path,
        "results_path": results_path,
        "evalplus_results_path": evalplus_results_path,
        "evalplus_data_dir": evalplus_cache_dir,
        "evalplus_runtime_cache_dir": evalplus_runtime_cache_dir,
        "sample_format_counts": dict(sample_format_counts),
    }
    # EvalPlus 0.2.0 writes ``*_samples_eval_results.json`` by default; keep a
    # stable ``*_eval.json`` copy for local logs and downstream report parsing.
    raw_results_path = results_path if os.path.exists(results_path) else evalplus_results_path
    if os.path.exists(raw_results_path):
        with open(raw_results_path, "r", encoding="utf-8") as f:
            report.update(json.load(f))
        _add_evalplus_pass_at_k(report, evalplus_evaluate_module)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def evaluate_humaneval_plus_pass1(*args, **kwargs) -> tp.Dict[str, tp.Any]:
    kwargs["plus"] = True
    return evaluate_humaneval_pass1(*args, **kwargs)


def _load_ifeval_rows_from_path(path: str):
    if os.path.isdir(path):
        try:
            return load_from_disk(path)
        except Exception:
            return load_dataset(path, split="train")
    if os.path.isfile(path):
        if path.endswith(".jsonl"):
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        raise ValueError(f"Unsupported IFEval data file: {path}")
    raise FileNotFoundError(f"IFEval data_path does not exist: {path}")


def _load_ifeval_rows(data_path: tp.Optional[str] = None):
    path = str(data_path or "").strip()
    if path:
        return _load_ifeval_rows_from_path(path)

    local_path = _existing_default_ifeval_path()
    if local_path:
        log.info("Using local IFEval data: %s", local_path)
        return _load_ifeval_rows_from_path(local_path)

    try:
        return load_dataset("google/IFEval", split="train")
    except Exception as e:
        ename = type(e).__name__
        msg = str(e).lower()
        is_net = ename in (
            "ConnectionError",
            "ConnectTimeout",
            "Timeout",
            "OfflineModeIsEnabled",
        ) or "couldn't reach" in msg or "connection" in msg
        if not is_net:
            raise
        log.warning("google/IFEval download failed (%s: %s); retrying with local dataset cache.", ename, e)
        return load_dataset(
            "google/IFEval",
            split="train",
            download_config=DownloadConfig(local_files_only=True),
        )


def _manual_render_single_turn_chat_prompt(
    prompt: str,
    *,
    tokenizer_name: str,
    eos_token: str,
) -> str:
    name = str(tokenizer_name or "").lower()
    eos = eos_token or "</s>"
    del name
    return f"<s>[INST] {prompt} [/INST]"


def _render_ifeval_prompt(
    tokenizer,
    prompt: str,
    *,
    tokenizer_name: str,
    apply_chat_template: bool,
) -> str:
    prompt = str(prompt or "")
    if not bool(apply_chat_template):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, str) and rendered:
            return rendered
    except Exception:
        pass
    return _manual_render_single_turn_chat_prompt(
        prompt,
        tokenizer_name=tokenizer_name,
        eos_token=getattr(tokenizer, "eos_token", None) or "</s>",
    )


def _import_ifeval_lm_eval_utils():
    _prepend_default_nltk_data_path()
    try:
        from lm_eval.tasks.ifeval.utils import (
            InputExample,
            test_instruction_following_loose,
            test_instruction_following_strict,
        )

        return InputExample, test_instruction_following_strict, test_instruction_following_loose
    except Exception:
        return None


def _import_ifeval_registry():
    _prepend_default_nltk_data_path()
    for module_name in (
        "lm_eval.tasks.ifeval.instructions_registry",
        "instruction_following_eval.instructions_registry",
    ):
        try:
            module = __import__(module_name, fromlist=["INSTRUCTION_DICT"])
            if hasattr(module, "INSTRUCTION_DICT"):
                return module
        except Exception:
            continue
    return None


def _ifeval_check_with_registry(doc: tp.Dict[str, tp.Any], response: str, *, loose: bool) -> tp.Tuple[bool, tp.List[bool]]:
    registry = _import_ifeval_registry()
    if registry is None:
        raise ImportError(
            "IFEval scoring requires lm-evaluation-harness or the official google-research "
            "instruction_following_eval package. Install one of them, e.g. `pip install lm-eval`, "
            "or place google-research's instruction_following_eval on PYTHONPATH."
        )

    if loose:
        r = str(response or "").split("\n")
        response_remove_first = "\n".join(r[1:]).strip()
        response_remove_last = "\n".join(r[:-1]).strip()
        response_remove_both = "\n".join(r[1:-1]).strip()
        revised_response = str(response or "").replace("*", "")
        candidate_responses = [
            str(response or ""),
            revised_response,
            response_remove_first,
            response_remove_last,
            response_remove_both,
            response_remove_first.replace("*", ""),
            response_remove_last.replace("*", ""),
            response_remove_both.replace("*", ""),
        ]
    else:
        candidate_responses = [str(response or "")]

    instruction_ids = list(doc.get("instruction_id_list") or [])
    kwargs_list = list(doc.get("kwargs") or [{} for _ in instruction_ids])
    prompt = str(doc.get("prompt", "") or "")
    followed: tp.List[bool] = []
    for idx, instruction_id in enumerate(instruction_ids):
        instruction_cls = registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        raw_kwargs = kwargs_list[idx] if idx < len(kwargs_list) and isinstance(kwargs_list[idx], dict) else {}
        clean_kwargs = {k: v for k, v in raw_kwargs.items() if v is not None}
        instruction.build_description(**clean_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        ok = False
        for candidate in candidate_responses:
            if str(candidate).strip() and instruction.check_following(candidate):
                ok = True
                break
        followed.append(bool(ok))
    return all(followed), followed


def _ifeval_check_one(doc: tp.Dict[str, tp.Any], response: str) -> tp.Dict[str, tp.Any]:
    lm_eval_utils = _import_ifeval_lm_eval_utils()
    if lm_eval_utils is not None:
        InputExample, strict_fn, loose_fn = lm_eval_utils
        inp = InputExample(
            key=int(doc.get("key", 0) or 0),
            instruction_id_list=list(doc.get("instruction_id_list") or []),
            prompt=str(doc.get("prompt", "") or ""),
            kwargs=list(doc.get("kwargs") or []),
        )
        strict = strict_fn(inp, response)
        loose = loose_fn(inp, response)
        return {
            "prompt_level_strict_acc": bool(strict.follow_all_instructions),
            "inst_level_strict_acc": [bool(x) for x in strict.follow_instruction_list],
            "prompt_level_loose_acc": bool(loose.follow_all_instructions),
            "inst_level_loose_acc": [bool(x) for x in loose.follow_instruction_list],
        }

    strict_all, strict_inst = _ifeval_check_with_registry(doc, response, loose=False)
    loose_all, loose_inst = _ifeval_check_with_registry(doc, response, loose=True)
    return {
        "prompt_level_strict_acc": bool(strict_all),
        "inst_level_strict_acc": [bool(x) for x in strict_inst],
        "prompt_level_loose_acc": bool(loose_all),
        "inst_level_loose_acc": [bool(x) for x in loose_inst],
    }


@torch.no_grad()
def evaluate_ifeval(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    model_type: str,
    *,
    output_dir: str,
    data_path: tp.Optional[str] = None,
    tokenizer_name: str = "",
    max_source_length: tp.Optional[int] = None,
    max_new_tokens: int = 1280,
    max_examples: tp.Optional[int] = None,
    apply_chat_template: bool = True,
) -> tp.Dict[str, tp.Any]:
    """Generate google/IFEval responses and compute strict/loose prompt/instruction accuracy."""
    if model_type != "CausalLM":
        raise ValueError(f"evaluate_ifeval only supports CausalLM, got model_type={model_type!r}")
    if _import_ifeval_lm_eval_utils() is None and _import_ifeval_registry() is None:
        raise ImportError(
            "IFEval scoring backend is not installed. Install `lm-eval==0.4.8` "
            "or put google-research's instruction_following_eval package on PYTHONPATH."
        )
    os.makedirs(output_dir, exist_ok=True)
    resolved_data_path = str(data_path or "") or _existing_default_ifeval_path() or "google/IFEval"
    rows_obj = _load_ifeval_rows(data_path)
    rows = list(rows_obj)
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    if max_source_length is None or str(max_source_length).strip().lower() in {"", "none", "null"}:
        source_limit = None
    else:
        source_limit = int(max_source_length)

    samples_path = os.path.join(output_dir, "ifeval_generations.jsonl")
    model.eval()
    generated: tp.List[tp.Dict[str, tp.Any]] = []
    for idx, row in enumerate(tqdm(rows, desc="ifeval_generate", disable=_should_disable_tqdm())):
        doc = dict(row)
        prompt = str(doc.get("prompt", "") or "")
        rendered_prompt = _render_ifeval_prompt(
            tokenizer,
            prompt,
            tokenizer_name=tokenizer_name,
            apply_chat_template=bool(apply_chat_template),
        )
        response = model_inference(
            model,
            tokenizer,
            rendered_prompt,
            model_type,
            max_source_length=source_limit,
            max_target_length=int(max_new_tokens),
            append_space=False,
        )
        generated.append(
            {
                "key": int(doc.get("key", idx) or idx),
                "prompt": prompt,
                "rendered_prompt": rendered_prompt,
                "instruction_id_list": list(doc.get("instruction_id_list") or []),
                "kwargs": list(doc.get("kwargs") or []),
                "response": response,
            }
        )

    with open(samples_path, "w", encoding="utf-8") as f:
        for row in generated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    prompt_strict: tp.List[bool] = []
    prompt_loose: tp.List[bool] = []
    inst_strict: tp.List[bool] = []
    inst_loose: tp.List[bool] = []
    per_sample: tp.List[tp.Dict[str, tp.Any]] = []
    for row in generated:
        metrics = _ifeval_check_one(row, str(row.get("response", "") or ""))
        prompt_strict.append(bool(metrics["prompt_level_strict_acc"]))
        prompt_loose.append(bool(metrics["prompt_level_loose_acc"]))
        inst_strict.extend(bool(x) for x in metrics["inst_level_strict_acc"])
        inst_loose.extend(bool(x) for x in metrics["inst_level_loose_acc"])
        per_sample.append(
            {
                "key": row["key"],
                "prompt_level_strict_acc": bool(metrics["prompt_level_strict_acc"]),
                "prompt_level_loose_acc": bool(metrics["prompt_level_loose_acc"]),
                "inst_level_strict_acc": [bool(x) for x in metrics["inst_level_strict_acc"]],
                "inst_level_loose_acc": [bool(x) for x in metrics["inst_level_loose_acc"]],
            }
        )

    def _mean_bool(xs: tp.Sequence[bool]) -> float:
        return float(sum(1 for x in xs if x)) / float(max(len(xs), 1))

    report = {
        "metric_name": "ifeval",
        "dataset": str(resolved_data_path),
        "num_prompts": int(len(generated)),
        "num_instructions": int(len(inst_strict)),
        "prompt_level_strict_acc": _mean_bool(prompt_strict),
        "inst_level_strict_acc": _mean_bool(inst_strict),
        "prompt_level_loose_acc": _mean_bool(prompt_loose),
        "inst_level_loose_acc": _mean_bool(inst_loose),
        "samples_path": samples_path,
        "max_source_length": source_limit,
        "max_new_tokens": int(max_new_tokens),
        "apply_chat_template": bool(apply_chat_template),
        "per_sample": per_sample,
    }
    with open(os.path.join(output_dir, "ifeval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
