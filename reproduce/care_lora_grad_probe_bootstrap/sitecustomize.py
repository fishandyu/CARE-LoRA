"""Process-local bootstrap for CARE-LoRA gradient-similarity diagnostics.

Python imports sitecustomize from the explicitly prepended bootstrap directory.
The wrapper injects the diagnostic Trainer callback without editing the
production training entry, CARE-LoRA operator, or ordinary configuration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    return int(str(os.environ.get(name, default)).strip())


def _env_float(name: str, default: float) -> float:
    return float(str(os.environ.get(name, default)).strip())


def _env_percentages(name: str) -> list[float]:
    raw = str(os.environ.get(name, ",".join(str(v) for v in range(5, 101, 5))))
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _install() -> None:
    if not _env_bool("CARE_LORA_GRAD_SIMILARITY", False):
        return

    reproduce_dir = Path(__file__).resolve().parent.parent
    reproduce_str = str(reproduce_dir)
    if reproduce_str not in sys.path:
        sys.path.insert(0, reproduce_str)

    import utils
    from care_lora_gradient_similarity import CareLoraGradientSimilarityCallback

    original = utils.train_text_to_text_model
    if getattr(original, "_care_lora_grad_similarity_wrapped", False):
        return

    first_n_steps = _env_int("CARE_LORA_GRAD_FIRST_N_STEPS", 0)
    stop_after_first_n = _env_bool("CARE_LORA_GRAD_STOP_AFTER_FIRST_N", False)
    percentages = _env_percentages("CARE_LORA_GRAD_PROGRESS_PERCENTAGES")
    eps = _env_float("CARE_LORA_GRAD_EPS", 1.0e-12)

    def wrapped_train_text_to_text_model(*args, **kwargs):
        if not bool(kwargs.get("use_care_lora", False)):
            raise RuntimeError("CARE-LoRA gradient diagnostics require use_care_lora=True.")
        if str(kwargs.get("track_cuda_peak", True)).strip().lower() not in {"false", "0", "no"}:
            raise RuntimeError(
                "Diagnostic steps retain full X and must not be used as CARE-LoRA memory results; "
                "set model.track_cuda_peak=false."
            )
        runtime_dir = kwargs.get("runtime_dir")
        if not runtime_dir:
            raise RuntimeError("CARE-LoRA gradient diagnostics require runtime_dir for raw-data output.")
        callbacks = list(kwargs.get("callbacks") or [])
        callbacks.append(
            CareLoraGradientSimilarityCallback(
                output_dir=os.path.join(runtime_dir, "logs", "care_lora_gradient_similarity"),
                progress_percentages=percentages,
                first_n_steps=first_n_steps,
                stop_after_first_n_steps=stop_after_first_n,
                eps=eps,
            )
        )
        kwargs["callbacks"] = callbacks
        return original(*args, **kwargs)

    wrapped_train_text_to_text_model._care_lora_grad_similarity_wrapped = True
    utils.train_text_to_text_model = wrapped_train_text_to_text_model


_install()
