"""Opt-in gradient-fidelity diagnostics for the repository's plain CARE-LoRA path.

The production CARE-LoRA autograd function is intentionally not modified here. This
callback installs hooks only for sampled optimizer steps, computes the exact
counterfactual LoRA-A gradient from the retained input X, and observes (without
replacing) the approximate A gradient returned by CARE-LoRA.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn
from transformers import TrainerCallback


log = logging.getLogger(__name__)


def _is_main_process() -> bool:
    for key in ("LOCAL_RANK", "RANK"):
        value = os.environ.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(value) == 0
        except ValueError:
            return True
    return True


def _world_size() -> int:
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def _json_dump(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _jsonl_append_many(path: str, payloads: Iterable[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class CareLoraGradientSimilarityCallback(TrainerCallback):
    """Measure exact-vs-CARE-LoRA LoRA-A gradients along the unchanged CARE-LoRA trajectory.

    Sampling is defined in optimizer-step units. During a sampled step, each
    layer accumulates exact and approximate gradients across every micro-batch;
    global metrics are computed from those accumulated matrices immediately
    before optimizer.step(). Raw per-layer sufficient statistics are kept in
    JSONL so the global metrics can be reproduced without storing enormous
    gradient matrices.
    """

    def __init__(
        self,
        *,
        output_dir: str,
        progress_percentages: Iterable[float] = tuple(range(5, 101, 5)),
        first_n_steps: int = 0,
        stop_after_first_n_steps: bool = False,
        eps: float = 1.0e-12,
    ) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.progress_percentages = sorted({float(v) for v in progress_percentages})
        self.first_n_steps = max(0, int(first_n_steps))
        self.stop_after_first_n_steps = bool(stop_after_first_n_steps)
        self.eps = float(eps)
        if self.eps <= 0.0:
            raise ValueError("CARE-LoRA gradient-similarity eps must be positive.")
        if any((p <= 0.0 or p > 100.0) for p in self.progress_percentages):
            raise ValueError("CARE-LoRA gradient-similarity progress percentages must be in (0, 100].")
        if self.stop_after_first_n_steps and self.first_n_steps < 1:
            raise ValueError("stop_after_first_n_steps=True requires first_n_steps >= 1.")

        self.global_jsonl = os.path.join(self.output_dir, "global_metrics.jsonl")
        self.layers_jsonl = os.path.join(self.output_dir, "layer_metrics.jsonl")
        self.metadata_json = os.path.join(self.output_dir, "metadata.json")

        self._layer_specs: Dict[str, Dict[str, Any]] = {}
        self._progress_targets: Dict[int, List[float]] = {}
        self._max_steps = 0
        self._active_step: Optional[int] = None
        self._active_reasons: List[str] = []
        self._forward_handles: List[Any] = []
        self._parameter_handles: List[Any] = []
        self._exact_grads: Dict[str, torch.Tensor] = {}
        self._approx_grads: Dict[str, torch.Tensor] = {}
        self._exact_calls: Dict[str, int] = {}
        self._approx_calls: Dict[str, int] = {}
        self._num_records = 0

    @staticmethod
    def _active_care_lora_adapter(module: nn.Module) -> Optional[str]:
        use_care_lora = getattr(module, "use_care_lora", None)
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if not isinstance(use_care_lora, dict) or lora_a is None or lora_b is None:
            return None
        active = getattr(module, "active_adapters", [])
        if isinstance(active, str):
            active = [active]
        candidates = [name for name in active if bool(use_care_lora.get(name, False))]
        if not candidates:
            candidates = [name for name, enabled in use_care_lora.items() if bool(enabled)]
        if len(candidates) > 1:
            raise RuntimeError(
                "CARE-LoRA gradient diagnostics require exactly one active CARE-LoRA adapter per layer; "
                f"found {candidates!r}."
            )
        return candidates[0] if candidates else None

    def _discover_layers(self, model: nn.Module) -> None:
        specs: Dict[str, Dict[str, Any]] = {}
        for module_name, module in model.named_modules():
            adapter = self._active_care_lora_adapter(module)
            if adapter is None:
                continue
            A = module.lora_A[adapter].weight
            B = module.lora_B[adapter].weight
            if not A.requires_grad:
                raise RuntimeError(f"Diagnostic CARE-LoRA layer has frozen A: {module_name}")
            dropout = module.lora_dropout[adapter]
            dropout_p = float(getattr(dropout, "p", 0.0))
            if not isinstance(dropout, nn.Identity) and dropout_p != 0.0:
                raise RuntimeError(
                    "CARE-LoRA gradient diagnostics currently require lora_dropout=0 so the externally "
                    f"captured X exactly matches CARE-LoRA's input; layer={module_name}, p={dropout_p}."
                )
            specs[module_name] = {
                "module": module,
                "adapter": adapter,
                "A": A,
                "B": B,
                "scaling": float(module.scaling[adapter]),
                "shape": [int(v) for v in A.shape],
            }
        if not specs:
            raise RuntimeError(
                "CARE-LoRA gradient diagnostics found zero active CARE-LoRA LoRA linear layers. "
                "Use the plain peft=care_lora path and verify runtime mode synchronization."
            )
        self._layer_specs = specs

    def _build_progress_targets(self, max_steps: int) -> Dict[int, List[float]]:
        targets: Dict[int, List[float]] = {}
        for percentage in self.progress_percentages:
            step = max(1, min(max_steps, int(math.ceil(max_steps * percentage / 100.0))))
            targets.setdefault(step, []).append(float(percentage))
        return targets

    def _write_metadata(self) -> None:
        if not _is_main_process():
            return
        scheduled = [
            {"optimizer_step": int(step), "percentages": percentages}
            for step, percentages in sorted(self._progress_targets.items())
        ]
        layer_shapes = {
            name: {
                "adapter": spec["adapter"],
                "a_grad_shape": spec["shape"],
            }
            for name, spec in self._layer_specs.items()
        }
        _json_dump(
            self.metadata_json,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "definition": "exact and approximate LoRA-A gradients accumulated over one optimizer step",
                "trajectory": "CARE-LoRA approximate gradient only; exact gradient is diagnostic and never returned",
                "max_optimizer_steps": int(self._max_steps),
                "progress_percentages": self.progress_percentages,
                "progress_schedule": scheduled,
                "first_n_steps": int(self.first_n_steps),
                "stop_after_first_n_steps": bool(self.stop_after_first_n_steps),
                "eps": float(self.eps),
                "num_care_lora_lora_layers": len(self._layer_specs),
                "world_size": _world_size(),
                "raw_layer_statistics": ["dot", "norm_exact_sq", "norm_approx_sq"],
                "layers": layer_shapes,
            },
        )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if _world_size() != 1:
            raise RuntimeError(
                "CARE-LoRA gradient diagnostics currently support single-process runs only. "
                "The provided Mistral bash uses CUDA_VISIBLE_DEVICES=0."
            )
        if model is None:
            raise RuntimeError("Trainer did not provide a model to CARE-LoRA gradient diagnostics.")
        self._max_steps = max(1, int(state.max_steps))
        self._discover_layers(model)
        self._progress_targets = self._build_progress_targets(self._max_steps)
        if _is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)
            for path in (self.global_jsonl, self.layers_jsonl):
                if os.path.exists(path):
                    os.remove(path)
            self._write_metadata()
            log.info(
                "[care_lora-grad-sim] enabled | layers=%d | max_steps=%d | first_n=%d | "
                "progress_targets=%s | raw_dir=%s",
                len(self._layer_specs),
                self._max_steps,
                self.first_n_steps,
                sorted(self._progress_targets),
                self.output_dir,
            )
            try:
                import wandb

                if getattr(wandb, "run", None) is not None:
                    wandb.define_metric("grad_similarity/optimizer_step")
                    wandb.define_metric(
                        "grad_similarity/*",
                        step_metric="grad_similarity/optimizer_step",
                    )
            except Exception as exc:
                log.warning("[care_lora-grad-sim] wandb metric definition skipped: %s", exc)
        return control

    def _sample_reasons(self, optimizer_step: int) -> List[str]:
        reasons: List[str] = []
        if self.first_n_steps > 0 and optimizer_step <= self.first_n_steps:
            reasons.append("first_n")
        for percentage in self._progress_targets.get(optimizer_step, []):
            reasons.append(f"progress_{percentage:g}pct")
        return reasons

    def _accumulate(self, store: Dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
        value_fp32 = value.detach()
        if value_fp32.dtype != torch.float32:
            value_fp32 = value_fp32.to(dtype=torch.float32)
        previous = store.get(name)
        if previous is None:
            store[name] = value_fp32.clone()
        else:
            previous.add_(value_fp32)

    def _record_exact(
        self,
        name: str,
        x: torch.Tensor,
        B: torch.Tensor,
        scaling: float,
        grad_output: torch.Tensor,
    ) -> None:
        if self._active_step is None:
            return
        with torch.no_grad():
            x2 = x.reshape(-1, x.shape[-1])
            go2 = grad_output.reshape(-1, grad_output.shape[-1])
            if int(x2.shape[0]) != int(go2.shape[0]):
                raise RuntimeError(
                    f"CARE-LoRA diagnostic row mismatch at {name}: X={tuple(x2.shape)}, dY={tuple(go2.shape)}"
                )
            xf = x2 if x2.dtype == torch.float32 else x2.to(torch.float32)
            gof = go2 if go2.dtype == torch.float32 else go2.to(torch.float32)
            Bf = B.detach() if B.dtype == torch.float32 else B.detach().to(torch.float32)
            grad_z = gof @ Bf
            grad_z.mul_(float(scaling))
            exact_grad = grad_z.mT @ xf
            self._accumulate(self._exact_grads, name, exact_grad)
            self._exact_calls[name] = self._exact_calls.get(name, 0) + 1

    def _make_forward_hook(self, name: str, spec: Dict[str, Any]):
        def hook(module, inputs, output):
            if self._active_step is None or not torch.is_grad_enabled():
                return
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"CARE-LoRA diagnostic expected tensor input at layer {name}.")
            if not torch.is_tensor(output) or not output.requires_grad:
                raise RuntimeError(f"CARE-LoRA diagnostic expected differentiable tensor output at layer {name}.")
            # Retain only a detached reference. Shared q/k/v or gate/up inputs keep
            # shared storage; no clone is made. The closure dies after this backward.
            x_saved = inputs[0].detach()
            B = spec["B"]
            scaling = float(spec["scaling"])

            def output_grad_hook(grad_output):
                self._record_exact(name, x_saved, B, scaling, grad_output)
                # Returning None preserves grad_output exactly.
                return None

            output.register_hook(output_grad_hook)

        return hook

    def _make_parameter_hook(self, name: str):
        def hook(grad):
            if self._active_step is not None:
                self._accumulate(self._approx_grads, name, grad)
                self._approx_calls[name] = self._approx_calls.get(name, 0) + 1
            # Returning None preserves the CARE-LoRA gradient exactly.
            return None

        return hook

    def _install_probe_hooks(self) -> None:
        if self._forward_handles or self._parameter_handles:
            raise RuntimeError("CARE-LoRA diagnostic hooks were already installed.")
        for name, spec in self._layer_specs.items():
            self._forward_handles.append(
                spec["module"].register_forward_hook(self._make_forward_hook(name, spec))
            )
            self._parameter_handles.append(spec["A"].register_hook(self._make_parameter_hook(name)))

    def _remove_probe_hooks(self) -> None:
        for handle in self._forward_handles:
            handle.remove()
        for handle in self._parameter_handles:
            handle.remove()
        self._forward_handles.clear()
        self._parameter_handles.clear()

    def _reset_accumulators(self) -> None:
        self._exact_grads.clear()
        self._approx_grads.clear()
        self._exact_calls.clear()
        self._approx_calls.clear()

    def on_step_begin(self, args, state, control, **kwargs):
        optimizer_step = int(state.global_step) + 1
        reasons = self._sample_reasons(optimizer_step)
        if not reasons:
            return control
        if self._active_step is not None:
            raise RuntimeError(
                f"CARE-LoRA diagnostic step {self._active_step} was not finalized before step {optimizer_step}."
            )
        self._active_step = optimizer_step
        self._active_reasons = reasons
        self._reset_accumulators()
        self._install_probe_hooks()
        log.info("[care_lora-grad-sim] sampling optimizer_step=%d reasons=%s", optimizer_step, reasons)
        return control

    def _finalize_active_step(self) -> Dict[str, Any]:
        if self._active_step is None:
            return {}
        optimizer_step = int(self._active_step)
        self._remove_probe_hooks()

        expected = set(self._layer_specs)
        exact_names = set(self._exact_grads)
        approx_names = set(self._approx_grads)
        if exact_names != expected or approx_names != expected:
            raise RuntimeError(
                "Incomplete CARE-LoRA gradient diagnostic capture at optimizer step "
                f"{optimizer_step}: missing_exact={sorted(expected - exact_names)[:8]}, "
                f"missing_approx={sorted(expected - approx_names)[:8]}."
            )
        mismatched_calls = {
            name: (self._exact_calls.get(name, 0), self._approx_calls.get(name, 0))
            for name in expected
            if self._exact_calls.get(name, 0) != self._approx_calls.get(name, 0)
        }
        if mismatched_calls:
            raise RuntimeError(
                f"Exact/approx micro-batch counts differ at step {optimizer_step}: "
                f"{list(mismatched_calls.items())[:8]}"
            )

        names = sorted(expected)
        scalar_rows = []
        for name in names:
            exact = self._exact_grads[name]
            approx = self._approx_grads[name]
            if exact.shape != approx.shape:
                raise RuntimeError(
                    f"Exact/approx A-gradient shape mismatch at {name}: {exact.shape} vs {approx.shape}"
                )
            scalar_rows.append(
                torch.stack(
                    [
                        torch.sum(exact * approx),
                        torch.sum(exact * exact),
                        torch.sum(approx * approx),
                    ]
                )
            )
        # One device synchronization for every layer's sufficient statistics.
        values = torch.stack(scalar_rows).detach().cpu().tolist()
        timestamp = datetime.now().isoformat(timespec="seconds")
        progress = 100.0 * optimizer_step / max(self._max_steps, 1)
        layer_records: List[Dict[str, Any]] = []
        total_dot = 0.0
        total_exact_sq = 0.0
        total_approx_sq = 0.0
        for name, (dot, exact_sq, approx_sq) in zip(names, values):
            dot = float(dot)
            exact_sq = max(0.0, float(exact_sq))
            approx_sq = max(0.0, float(approx_sq))
            exact_norm = math.sqrt(exact_sq)
            approx_norm = math.sqrt(approx_sq)
            valid = exact_norm > self.eps and approx_norm > self.eps
            cosine = dot / (exact_norm * approx_norm + self.eps) if valid else None
            norm_ratio = approx_norm / (exact_norm + self.eps) if exact_norm > self.eps else None
            layer_records.append(
                {
                    "optimizer_step": optimizer_step,
                    "training_progress_percent": progress,
                    "sample_reasons": self._active_reasons,
                    "layer": name,
                    "adapter": self._layer_specs[name]["adapter"],
                    "a_grad_shape": self._layer_specs[name]["shape"],
                    "micro_batches": int(self._exact_calls[name]),
                    "dot": dot,
                    "norm_exact_sq": exact_sq,
                    "norm_approx_sq": approx_sq,
                    "cosine": cosine,
                    "norm_ratio": norm_ratio,
                    "valid": valid,
                    "_time": timestamp,
                }
            )
            total_dot += dot
            total_exact_sq += exact_sq
            total_approx_sq += approx_sq

        exact_norm = math.sqrt(max(0.0, total_exact_sq))
        approx_norm = math.sqrt(max(0.0, total_approx_sq))
        valid = exact_norm > self.eps and approx_norm > self.eps
        global_record = {
            "optimizer_step": optimizer_step,
            "training_progress_percent": progress,
            "sample_reasons": self._active_reasons,
            "num_layers": len(names),
            "micro_batches_per_layer": sorted({int(self._exact_calls[name]) for name in names}),
            "dot": total_dot,
            "norm_exact_sq": total_exact_sq,
            "norm_approx_sq": total_approx_sq,
            "global_cosine": (
                total_dot / (exact_norm * approx_norm + self.eps) if valid else None
            ),
            "global_norm_ratio": (
                approx_norm / (exact_norm + self.eps) if exact_norm > self.eps else None
            ),
            "valid": valid,
            "_time": timestamp,
        }
        if _is_main_process():
            _jsonl_append_many(self.layers_jsonl, layer_records)
            _jsonl_append_many(self.global_jsonl, [global_record])
            log.info(
                "[care_lora-grad-sim] step=%d progress=%.4f%% cosine=%s norm_ratio=%s "
                "micro_batches=%s",
                optimizer_step,
                progress,
                global_record["global_cosine"],
                global_record["global_norm_ratio"],
                global_record["micro_batches_per_layer"],
            )
            try:
                import wandb

                if getattr(wandb, "run", None) is not None:
                    payload = {
                        "grad_similarity/optimizer_step": optimizer_step,
                        "grad_similarity/progress_percent": progress,
                    }
                    if global_record["global_cosine"] is not None:
                        payload["grad_similarity/global_cosine"] = global_record["global_cosine"]
                    if global_record["global_norm_ratio"] is not None:
                        payload["grad_similarity/global_norm_ratio"] = global_record["global_norm_ratio"]
                    wandb.log(payload)
                    wandb.summary["grad_similarity/raw_data_dir"] = self.output_dir
            except Exception as exc:
                log.warning("[care_lora-grad-sim] wandb logging skipped: %s", exc)

        self._num_records += 1
        self._active_step = None
        self._active_reasons = []
        self._reset_accumulators()
        return global_record

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self._active_step is not None:
            expected_step = int(state.global_step) + 1
            if int(self._active_step) != expected_step:
                raise RuntimeError(
                    f"CARE-LoRA diagnostic active step={self._active_step}, Trainer next step={expected_step}."
                )
            self._finalize_active_step()
        if (
            self.stop_after_first_n_steps
            and self.first_n_steps > 0
            and int(state.global_step) + 1 >= self.first_n_steps
        ):
            # Trainer still executes the current optimizer.step(), then exits.
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._remove_probe_hooks()
        self._active_step = None
        self._active_reasons = []
        self._reset_accumulators()
        if _is_main_process():
            log.info(
                "[care_lora-grad-sim] finished | records=%d | global=%s | layers=%s",
                self._num_records,
                self.global_jsonl,
                self.layers_jsonl,
            )
        return control
