# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import contextlib
import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

from .config import LoraConfig
from .dora import (
    DoraConv2dLayer,
    DoraLinearLayer,
    get_dora_enable_chunked_ops,
    get_dora_force_row_chunk_for_narrow_output,
    get_dora_narrow_output_ratio,
    get_dora_row_chunk_cast_bytes,
)

# CARE-LoRA saved-state layout selector:
# - False (default): put Z through ``save_for_backward`` so saved_tensors_hooks
#   observe the custom saved state during profiling.
# - True: store the same mathematical state directly on ``ctx`` when hook-based
#   profiling is not required.
_CARE_LORA_USE_SPEED_PATH: bool = False
_LORA_ACTIVATION_TRACKING_ENABLED: bool = False
_LORA_ACTIVATION_CONTEXT_DEPTH: int = 0

_TORCH_SOLVE_EX = getattr(torch.linalg, "solve_ex", None)
_AUTOCAST_HAS_DEVICE_ARG = True
try:
    torch.is_autocast_enabled("cuda")
except TypeError:
    _AUTOCAST_HAS_DEVICE_ARG = False
except Exception:
    _AUTOCAST_HAS_DEVICE_ARG = False


class _LoraActivationCaptureContext:
    def __enter__(self):
        global _LORA_ACTIVATION_CONTEXT_DEPTH
        _LORA_ACTIVATION_CONTEXT_DEPTH += 1

    def __exit__(self, exc_type, exc, tb):
        global _LORA_ACTIVATION_CONTEXT_DEPTH
        _LORA_ACTIVATION_CONTEXT_DEPTH = max(0, _LORA_ACTIVATION_CONTEXT_DEPTH - 1)
        return False


_LORA_ACTIVATION_CAPTURE_CONTEXT = _LoraActivationCaptureContext()


def set_lora_activation_tracking_enabled(enabled: bool) -> None:
    global _LORA_ACTIVATION_TRACKING_ENABLED
    global _LORA_ACTIVATION_CONTEXT_DEPTH
    _LORA_ACTIVATION_TRACKING_ENABLED = bool(enabled)
    if not enabled:
        _LORA_ACTIVATION_CONTEXT_DEPTH = 0


def is_lora_activation_context_active() -> bool:
    return bool(_LORA_ACTIVATION_TRACKING_ENABLED and _LORA_ACTIVATION_CONTEXT_DEPTH > 0)


def _linalg_amp_off_context(t: torch.Tensor):
    """Disable autocast around fp32 linear algebra used by CARE-LoRA solves."""
    if t.is_cuda:
        # Keep this branch lightweight because it is hit once per CARE-LoRA layer.
        # Semantics are unchanged: enter autocast(False) only when CUDA autocast is currently on.
        if _AUTOCAST_HAS_DEVICE_ARG:
            cuda_amp_on = bool(torch.is_autocast_enabled("cuda"))  # torch>=2.1
        else:
            cuda_amp_on = bool(torch.is_autocast_enabled())  # older API
        if not cuda_amp_on:
            return contextlib.nullcontext()
        return torch.amp.autocast("cuda", enabled=False)
    if t.device.type == "mps":
        try:
            return torch.amp.autocast("mps", enabled=False)
        except Exception:
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def _cuda_autocast_is_enabled() -> bool:
    """Fast compatibility wrapper for CUDA autocast state checks."""
    if _AUTOCAST_HAS_DEVICE_ARG:
        return bool(torch.is_autocast_enabled("cuda"))
    return bool(torch.is_autocast_enabled())


def _loract_rank(value: int, n_rows: int, n_cols: int) -> int:
    """Return the effective LoRAct decomposition rank for a flattened activation matrix."""
    return max(1, min(int(value), int(n_rows), int(n_cols)))


@torch.no_grad()
def _loract_qb_compress(
    x2: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    LoRAct sampling-based orthogonal decomposition for the flattened activation X.

    This follows the reference LoRAct RQB path with subsampling and exactly one QR:
      H_k = sampled rows from X
      Y = X H_k^T
      Q = qr(Y)
      V = Q^T X
      X ≈ Q V
    """
    n_rows, n_cols = int(x2.shape[0]), int(x2.shape[1])
    with _linalg_amp_off_context(x2):
        x_f = x2 if x2.dtype == torch.float32 else x2.to(torch.float32)
        k = _loract_rank(rank, n_rows, n_cols)
        idx = torch.randperm(n_rows, device=x2.device)[:k]
        h_k = x_f.index_select(0, idx)
        y = x_f @ h_k.t()
        q, _ = torch.linalg.qr(y.nan_to_num(0.0), mode="reduced")
        v = q.t() @ x_f
    return q.to(dtype=x2.dtype), v.to(dtype=x2.dtype)


def _as_fp32_no_copy(t: torch.Tensor) -> torch.Tensor:
    """
    Return ``t`` itself when already fp32; otherwise cast to fp32.

    This keeps CARE-LoRA math unchanged while skipping no-op
    ``.to(torch.float32)`` calls on tensors that are already fp32.
    """
    return t if t.dtype == torch.float32 else t.to(dtype=torch.float32)


def _round_up_capacity(n: int, *, granularity: int = 128) -> int:
    """
    Round ``n`` up to a fixed block size for workspace growth.

    For variable-length batches, growing capacity exactly to each new ``n`` can cause
    frequent realloc/free churn on CUDA allocator and lead to run-to-run wall-time
    jitter. Block growth keeps numerics identical (only storage capacity changes)
    while making runtime behavior more stable.

    Currently used by the (N, r)-shaped ``dZ`` backward workspace which MUST
    be a dedicated per-module buffer (it flows back into autograd as
    ``grad_z``), so the cross-layer shared workspace pattern does not apply.
    """
    g = max(1, int(granularity))
    n_i = max(1, int(n))
    return ((n_i + g - 1) // g) * g


# Workspace growth policy is capacity-only and does not change numerical results.
# Larger blocks reduce CUDA caching-allocator churn when N/token counts fluctuate,
# which lowers step-to-step wall-time variance in packed CausalLM training.
_CARE_LORA_GRAD_Z_GROW_GRANULARITY = 1024
_CARE_LORA_LINALG_WORKSPACES: dict[tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_CARE_LORA_EYE_WORKSPACES: dict[tuple[int, torch.device], torch.Tensor] = {}
_CARE_LORA_GRAD_Z_WORKSPACES: dict[tuple[int, torch.device, torch.dtype], tuple[int, torch.Tensor]] = {}
_CARE_LORA_MASKED_Z_WORKSPACES: dict[tuple[int, torch.device, torch.dtype], tuple[int, torch.Tensor]] = {}
_CARE_LORA_CURRENT_ATTENTION_MASK_SHAPE: Optional[tuple[int, ...]] = None
_CARE_LORA_CURRENT_ATTENTION_VALID_ROWS: Optional[torch.Tensor] = None
_CARE_LORA_CURRENT_ATTENTION_VALID_ROWS_BY_DEVICE: dict[torch.device, torch.Tensor] = {}
_CARE_LORA_CURRENT_ATTENTION_VALID_COUNT: Optional[int] = None


def set_care_lora_attention_mask(attention_mask: Optional[torch.Tensor]) -> None:
    """
    Stash the current model-level attention mask for CARE-LoRA's data-aware M* fit.

    LoRA Linear modules do not receive ``attention_mask`` in their forward
    signature, so the training harness sets this once at the outer model forward.
    CARE-LoRA then uses it only when the mask shape exactly matches the LoRA input
    leading dimensions, and only for the M* least-squares fit.
    """
    global _CARE_LORA_CURRENT_ATTENTION_MASK_SHAPE
    global _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS
    global _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS_BY_DEVICE
    global _CARE_LORA_CURRENT_ATTENTION_VALID_COUNT
    if attention_mask is None or not torch.is_tensor(attention_mask) or attention_mask.numel() == 0:
        clear_care_lora_attention_mask()
        return
    flat_mask = attention_mask.reshape(-1)
    try:
        if flat_mask.dtype == torch.bool:
            valid_count = int(flat_mask.sum().item())
        else:
            valid_count = int(torch.count_nonzero(flat_mask).item())
        if valid_count == int(flat_mask.numel()):
            clear_care_lora_attention_mask()
            return
    except Exception:
        valid_count = None
    valid_rows = flat_mask if flat_mask.dtype == torch.bool else (flat_mask != 0)
    _CARE_LORA_CURRENT_ATTENTION_MASK_SHAPE = tuple(int(v) for v in attention_mask.shape)
    _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS = valid_rows
    _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS_BY_DEVICE = {}
    _CARE_LORA_CURRENT_ATTENTION_VALID_COUNT = valid_count


def clear_care_lora_attention_mask() -> None:
    """Clear the process-local CARE-LoRA attention-mask context."""
    global _CARE_LORA_CURRENT_ATTENTION_MASK_SHAPE
    global _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS
    global _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS_BY_DEVICE
    global _CARE_LORA_CURRENT_ATTENTION_VALID_COUNT
    _CARE_LORA_CURRENT_ATTENTION_MASK_SHAPE = None
    _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS = None
    _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS_BY_DEVICE = {}
    _CARE_LORA_CURRENT_ATTENTION_VALID_COUNT = None


def _care_lora_prepare_linalg_workspace(
    holder: nn.Module, r: int, in_f: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preallocate (r×in) Z^T X and (r×r) SPD system buffer ``H`` for CARE-LoRA.

    Reusing these removes repeated ``torch.mm`` output allocations. The buffers
    are shared per (r, in_features, device), then cached on each module as a
    direct reference. They never escape the forward M* solve.

    The SPD system is solved with ``torch.linalg.solve`` on ``h_buf``; the returned ``M``
    does not alias ``c_buf``.
    """
    key = (int(r), int(in_f), device)
    meta = getattr(holder, "_care_lora_linalg_workspace_meta", None)
    if meta == key:
        workspace = getattr(holder, "_care_lora_linalg_workspace", None)
        if workspace is not None:
            return workspace

    workspace = _CARE_LORA_LINALG_WORKSPACES.get(key)
    if workspace is None:
        c_buf = torch.empty(r, in_f, dtype=torch.float32, device=device)
        h_buf = torch.empty(r, r, dtype=torch.float32, device=device)
        # ``eye_r`` is a constant identity used by the fused
        # ``addmm(λI, Z^T, Z)`` path that assembles ``H = Z^T Z + λI``.
        eye_r = torch.eye(r, dtype=torch.float32, device=device)
        workspace = (c_buf, h_buf, eye_r)
        _CARE_LORA_LINALG_WORKSPACES[key] = workspace

    holder._care_lora_linalg_workspace_meta = key
    holder._care_lora_linalg_workspace = workspace
    return workspace


def _care_lora_attention_valid_rows_for_x(x: torch.Tensor, n_rows: int) -> Optional[torch.Tensor]:
    """Return a flattened boolean valid-token mask for ``x`` if the current attention mask matches."""
    valid = _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS
    mask_shape = _CARE_LORA_CURRENT_ATTENTION_MASK_SHAPE
    if valid is None or mask_shape is None or not torch.is_tensor(valid):
        return None
    if x.ndim < 3:
        return None
    expected_shape = tuple(int(v) for v in x.shape[:-1])
    if tuple(int(v) for v in mask_shape) != expected_shape:
        return None
    if int(valid.numel()) != int(n_rows):
        return None
    if valid.device != x.device:
        cached = _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS_BY_DEVICE.get(x.device)
        if cached is None:
            cached = valid.to(device=x.device, non_blocking=True)
            _CARE_LORA_CURRENT_ATTENTION_VALID_ROWS_BY_DEVICE[x.device] = cached
        valid = cached
    return valid


def _care_lora_masked_z_left(z_f: torch.Tensor, row_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Return ``z_f`` with invalid rows zeroed, reusing one small (N×r) buffer.

    This is equivalent to ``z_f * row_mask[:, None]`` but avoids allocating a
    fresh masked-Z tensor in every LoRA layer. The buffer is separate from the
    backward ``dZ`` workspace because both paths are live in the same forward pass.
    """
    if row_mask is None:
        return z_f
    n, r = int(z_f.shape[0]), int(z_f.shape[1])
    key = (r, z_f.device, z_f.dtype)
    entry = _CARE_LORA_MASKED_Z_WORKSPACES.get(key)
    if entry is None or n > int(entry[0]):
        cap_n = _round_up_capacity(n, granularity=_CARE_LORA_GRAD_Z_GROW_GRANULARITY)
        buf = torch.empty((cap_n, r), dtype=z_f.dtype, device=z_f.device)
        _CARE_LORA_MASKED_Z_WORKSPACES[key] = (cap_n, buf)
        z_left = buf[:n]
    else:
        z_left = entry[1][:n]
    torch.mul(z_f, row_mask.reshape(-1, 1), out=z_left)
    return z_left


def _care_lora_m_fit_row_mask_for_x(
    x: torch.Tensor,
    z_f: torch.Tensor,
    stats_holder: Optional[nn.Module],
) -> Optional[torch.Tensor]:
    """
    Return the token-row mask used only for the M* least-squares fit.

    Forward values and backward ``Z^T dZ`` still use all rows; only H/C for
    ``min_M ||ZM-X||`` exclude padding rows when an exact-shape mask is available.
    """
    total_rows = int(z_f.shape[0])
    valid = _care_lora_attention_valid_rows_for_x(x, total_rows)
    if valid is None:
        if stats_holder is not None:
            stats_holder._care_lora_last_m_total_rows = total_rows
            stats_holder._care_lora_last_m_fit_rows = total_rows
            stats_holder._care_lora_last_m_masked_rows = 0
        return None

    fit_rows = _CARE_LORA_CURRENT_ATTENTION_VALID_COUNT
    if fit_rows is None:
        try:
            fit_rows = int(valid.sum().item())
        except Exception:
            fit_rows = total_rows
    if stats_holder is not None:
        stats_holder._care_lora_last_m_total_rows = total_rows
        stats_holder._care_lora_last_m_fit_rows = fit_rows
        stats_holder._care_lora_last_m_masked_rows = max(0, total_rows - fit_rows)
    if fit_rows == total_rows:
        return None
    return valid


def _care_lora_get_eye_r(holder: nn.Module, r: int, device: torch.device) -> torch.Tensor:
    """Fetch the shared (r×r) identity used by the fused addmm path.

    Falls back to allocating one if the workspace was not prepared yet.
    """
    meta = getattr(holder, "_care_lora_linalg_workspace_meta", None)
    if meta is not None and meta[0] == r and meta[2] == device:
        workspace = getattr(holder, "_care_lora_linalg_workspace", None)
        if workspace is not None:
            return workspace[2]
    key = (int(r), device)
    eye_r = _CARE_LORA_EYE_WORKSPACES.get(key)
    if eye_r is None:
        eye_r = torch.eye(r, dtype=torch.float32, device=device)
        _CARE_LORA_EYE_WORKSPACES[key] = eye_r
    return eye_r


def _care_lora_prepare_bwd_workspace(
    holder: nn.Module,
    r: int,
    in_f: int,
    out_f: int,
    device: torch.device,
    grad_b_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preallocate (r×r) ``Z^T dZ``, (r×in) fp32 ``grad_A``, and (out×r) ``grad_B`` buffers for CARE-LoRA backward.

    Using ``torch.mm(..., out=buf)`` matches a fresh ``@`` / ``torch.mm`` on CUDA bit-for-bit
    (same rationale as ``_care_lora_prepare_linalg_workspace``).
    """
    meta = getattr(holder, "_care_lora_bwd_workspace_meta", None)
    if meta != (r, in_f, out_f, device, grad_b_dtype):
        holder._care_lora_bwd_workspace_meta = (r, in_f, out_f, device, grad_b_dtype)
        holder._care_lora_bwd_workspace_t_zu = torch.empty(r, r, dtype=torch.float32, device=device)
        holder._care_lora_bwd_workspace_grad_a_fp32 = torch.empty(r, in_f, dtype=torch.float32, device=device)
        holder._care_lora_bwd_workspace_grad_b = torch.empty(out_f, r, dtype=grad_b_dtype, device=device)
    return (
        holder._care_lora_bwd_workspace_t_zu,
        holder._care_lora_bwd_workspace_grad_a_fp32,
        holder._care_lora_bwd_workspace_grad_b,
    )


def _care_lora_prepare_grad_z_workspace(
    holder: nn.Module, n: int, r: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Reuse the internal (N×r) ``dZ = dY @ B`` buffer used in CARE-LoRA linear backward.

    ``dZ`` never escapes the custom backward, so one grow-only shared buffer per
    (r, device, dtype) avoids both repeated allocation and per-layer resident
    workspace memory without changing the matmul itself.
    """
    del holder
    key = (int(r), device, dtype)
    entry = _CARE_LORA_GRAD_Z_WORKSPACES.get(key)
    if entry is None or n > int(entry[0]):
        cap_n = _round_up_capacity(n, granularity=_CARE_LORA_GRAD_Z_GROW_GRANULARITY)
        buf = torch.empty((cap_n, r), dtype=dtype, device=device)
        _CARE_LORA_GRAD_Z_WORKSPACES[key] = (cap_n, buf)
        return buf[:n]
    return entry[1][:n]


def _care_lora_build_h_c(
    z_f: torch.Tensor,
    x_f: torch.Tensor,
    pinv_lambda: float,
    c_buf: torch.Tensor,
    h_buf: torch.Tensor,
    stats_holder: Optional[nn.Module],
    row_mask: Optional[torch.Tensor] = None,
) -> None:
    """
    Assemble ``H = Z^T Z + λ I`` into ``h_buf`` and ``C = Z^T X`` into ``c_buf`` (fp32).

    When ``row_mask`` is provided, only valid token rows contribute to H/C without
    copying the wide ``X_valid`` matrix.
    """
    lam0 = float(pinv_lambda)
    z_left = _care_lora_masked_z_left(z_f, row_mask)
    zt = z_left.mT
    eye_r = None
    if stats_holder is not None:
        eye_r = _care_lora_get_eye_r(stats_holder, int(z_f.shape[1]), z_f.device)
    if eye_r is None or lam0 == 0.0:
        torch.mm(zt, z_f, out=h_buf)
        if lam0 != 0.0:
            h_buf.diagonal().add_(lam0)
    else:
        torch.addmm(eye_r, zt, z_f, beta=lam0, alpha=1.0, out=h_buf)
    torch.mm(zt, x_f, out=c_buf)


def _care_lora_solve_from_buffers(
    h_buf: torch.Tensor,
    c_buf: torch.Tensor,
    z_f: torch.Tensor,
    x_f: torch.Tensor,
    pinv_lambda: float,
    stats_holder: Optional[nn.Module],
    z_left_f: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Solve ``H M = C`` given pre-filled ``H`` / ``C`` buffers (``use_ws`` path).

    Use ``solve_ex`` first, then the ridge / Cholesky / pinv fallback when needed.
    """
    lam0 = float(pinv_lambda)
    c = c_buf

    # ---------- Hot path: linalg.solve_ex (avoid implicit host sync in linalg.solve) ----------
    try:
        solve_ex = _TORCH_SOLVE_EX
        if solve_ex is not None:
            return solve_ex(h_buf, c, check_errors=False)[0]
        return torch.linalg.solve(h_buf, c)
    except Exception:
        if stats_holder is not None:
            try:
                stats_holder._care_lora_solve_fallback_count = int(
                    getattr(stats_holder, "_care_lora_solve_fallback_count", 0)
                ) + 1
            except Exception:
                pass
        # ---------- Slow path: exception-only fallback (rare pathological cases). ----------
        z_left = z_f if z_left_f is None else z_left_f
        ztz = z_left.t() @ z_f
        c = z_left.t() @ x_f
        r, in_f = int(ztz.shape[0]), int(x_f.shape[1])
        if not (torch.isfinite(ztz).all() and torch.isfinite(c).all()):
            return torch.zeros((r, in_f), dtype=torch.float32, device=z_f.device)

        trace = torch.trace(ztz)
        tr = float(trace.item()) if torch.isfinite(trace) else 0.0
        scale_ridge = max(lam0, 1e-8, (1e-7 * tr / max(r, 1)) if tr > 0 else 1e-8)

        ridge_eps = [0.0]
        e = float(scale_ridge)
        for _ in range(14):
            ridge_eps.append(e)
            e *= 10.0**0.5

        for re in ridge_eps:
            lam_eff = lam0 + float(re)
            h = ztz.clone()
            h.diagonal().add_(lam_eff)
            L, info = torch.linalg.cholesky_ex(h, upper=False)
            if int(info.item()) == 0:
                M = torch.cholesky_solve(c, L)
            else:
                try:
                    M = torch.linalg.solve(h, c)
                except Exception:
                    continue
            if torch.isfinite(M).all():
                return M

        h = ztz.clone()
        h.diagonal().add_(lam0 + float(ridge_eps[-1]))
        try:
            M = torch.linalg.pinv(h, hermitian=True, rtol=1e-4, atol=1e-6) @ c
            if torch.isfinite(M).all():
                return M
        except Exception:
            pass
        return torch.zeros((r, in_f), dtype=torch.float32, device=z_f.device)


def _care_lora_stable_m_star(
    z_f: torch.Tensor,
    x_f: torch.Tensor,
    pinv_lambda: float,
    *,
    c_buf: Optional[torch.Tensor] = None,
    h_buf: Optional[torch.Tensor] = None,
    eye_buf: Optional[torch.Tensor] = None,
    stats_holder: Optional[nn.Module] = None,
    row_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Solve H M = C with H = Z^T Z + λ I, C = Z^T X (fp32, under no_grad).
    ``row_mask`` restricts H/C to valid token rows while keeping the same output shape.

    Main path:
        Preallocated ``c_buf`` / ``h_buf`` feed ``torch.mm(..., out=)`` and
        in-place ``H`` assembly; ``solve`` returns fresh ``M`` storage.

    Fallback path:
        Ill-conditioned systems use ridge, Cholesky, and pinv fallbacks.

    Optional ``c_buf`` / ``h_buf`` (both or none):
        ``torch.addmm(λI, Z^T, Z, out=h_buf)`` assembles ``H = Z^T Z + λI`` in a single
        GEMM call, and ``torch.mm(..., out=c_buf)`` computes ``Z^T X``.
    """
    use_ws = c_buf is not None and h_buf is not None
    lam0 = float(pinv_lambda)
    if use_ws:
        z_left = _care_lora_masked_z_left(z_f, row_mask)
        zt = z_left.mT
        eye_r = eye_buf
        if eye_r is None and stats_holder is not None:
            eye_r = _care_lora_get_eye_r(stats_holder, int(z_f.shape[1]), z_f.device)
        if eye_r is None or lam0 == 0.0:
            torch.mm(zt, z_f, out=h_buf)
            if lam0 != 0.0:
                h_buf.diagonal().add_(lam0)
        else:
            torch.addmm(eye_r, zt, z_f, beta=lam0, alpha=1.0, out=h_buf)
        torch.mm(zt, x_f, out=c_buf)
        try:
            solve_ex = _TORCH_SOLVE_EX
            if solve_ex is not None:
                return solve_ex(h_buf, c_buf, check_errors=False)[0]
            return torch.linalg.solve(h_buf, c_buf)
        except Exception:
            if stats_holder is not None:
                try:
                    stats_holder._care_lora_solve_fallback_count = int(
                        getattr(stats_holder, "_care_lora_solve_fallback_count", 0)
                    ) + 1
                except Exception:
                    pass
            return _care_lora_solve_from_buffers(h_buf, c_buf, z_f, x_f, lam0, None, z_left_f=z_left)

    z_left = _care_lora_masked_z_left(z_f, row_mask)
    ztz = z_left.t() @ z_f
    c = z_left.t() @ x_f

    # ---------- Hot path: linalg.solve_ex (avoid implicit host sync in linalg.solve) ----------
    # NOTE:
    #   We intentionally avoid per-step finite checks here, because ``Tensor.item()`` /
    #   Python-side bool branching on CUDA tensors introduces host-device sync on every
    #   layer/step and can dominate wall time for small-r CARE-LoRA.
    #   Under normal finite training trajectories this keeps the exact same H/C solve path
    #   and returns numerically identical M* to the previous hot path.
    try:
        solve_ex = _TORCH_SOLVE_EX
        h_spd = ztz.clone()
        h_spd.diagonal().add_(lam0)
        if solve_ex is not None:
            return solve_ex(h_spd, c, check_errors=False)[0]
        return torch.linalg.solve(h_spd, c)
    except Exception:
        if stats_holder is not None:
            try:
                stats_holder._care_lora_solve_fallback_count = int(
                    getattr(stats_holder, "_care_lora_solve_fallback_count", 0)
                ) + 1
            except Exception:
                pass
        # ---------- Slow path: exception-only fallback (rare pathological cases). ----------
        r, in_f = int(ztz.shape[0]), int(x_f.shape[1])
        if not (torch.isfinite(ztz).all() and torch.isfinite(c).all()):
            return torch.zeros((r, in_f), dtype=torch.float32, device=z_f.device)

        trace = torch.trace(ztz)
        tr = float(trace.item()) if torch.isfinite(trace) else 0.0
        scale_ridge = max(lam0, 1e-8, (1e-7 * tr / max(r, 1)) if tr > 0 else 1e-8)

        ridge_eps = [0.0]
        e = float(scale_ridge)
        for _ in range(14):
            ridge_eps.append(e)
            e *= 10.0**0.5

        for re in ridge_eps:
            lam_eff = lam0 + float(re)
            h = ztz.clone()
            h.diagonal().add_(lam_eff)
            L, info = torch.linalg.cholesky_ex(h, upper=False)
            if int(info.item()) == 0:
                M = torch.cholesky_solve(c, L)
            else:
                try:
                    M = torch.linalg.solve(h, c)
                except Exception:
                    continue
            if torch.isfinite(M).all():
                return M

        h = ztz.clone()
        h.diagonal().add_(lam0 + float(ridge_eps[-1]))
        try:
            M = torch.linalg.pinv(h, hermitian=True, rtol=1e-4, atol=1e-6) @ c
            if torch.isfinite(M).all():
                return M
        except Exception:
            pass
        return torch.zeros((r, in_f), dtype=torch.float32, device=z_f.device)


class _LoraFaLinearFn(torch.autograd.Function):
    """
    Fused LoRA-FA path for linear layers.

    LoRA-FA freezes A. We still need exact gradients for:
      - x (to preserve gradient flow to earlier layers)
      - B (the only trainable LoRA parameter)

    Forward:
      z = x @ A^T
      y = z @ B^T * scaling

    Backward (exact):
      grad_B = dY^T @ z
      grad_x = (dY @ B) @ A

    Memory semantics:
      - Save only z (detached) to avoid retaining the full X autograd graph.
      - A is frozen (requires_grad=False): keep a reference on ctx instead of
        save_for_backward(A) so autograd holds one fewer saved tensor slot; A
        must not be modified in-place between forward and backward.
      - Forward uses F.linear (same as nn.Linear). Backward uses explicit ``@``
        for the chain rule (``dL/dz = dL/dy @ B``, ``dL/dx = dL/dz @ A``), because
        ``F.linear(dy, W)`` computes ``dy @ W.T``, which would be wrong here.
    """

    @staticmethod
    def forward(ctx, x, A, B, scaling: float):
        x2 = x.reshape(-1, x.shape[-1])
        z = F.linear(x2, A)
        y2 = F.linear(z, B)
        # Keep only activation-like tensors in save_for_backward to reduce
        # saved_tensors_hooks overhead when peak-memory tracking is enabled.
        ctx.save_for_backward(z.detach())
        ctx.B = B
        ctx.A = A
        ctx.input_shape = x.shape
        ctx.scaling = float(scaling)
        return (y2 * scaling).view(*x.shape[:-1], B.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        A = ctx.A
        B = ctx.B
        grad_out2 = grad_output.reshape(-1, grad_output.shape[-1]) * ctx.scaling

        # Mixed-precision CausalLM training may send bf16 grad_output while LoRA
        # A/B remain fp32. Use an fp32 fallback for cross-dtype matmuls and keep
        # the original fast path when all operands share the same dtype.
        _same_dt = (
            grad_out2.dtype == z.dtype
            and grad_out2.dtype == A.dtype
            and grad_out2.dtype == B.dtype
        )

        if _same_dt:
            grad_B = grad_out2.t() @ z if ctx.needs_input_grad[2] else None
            # z = x @ A.T, dZ = dY @ B, and dX = dZ @ A.
            if ctx.needs_input_grad[0]:
                grad_z = grad_out2 @ B
                grad_x2 = grad_z @ A
                grad_x = grad_x2.view(*ctx.input_shape)
            else:
                grad_x = None
        else:
            go = grad_out2 if grad_out2.dtype == torch.float32 else grad_out2.to(torch.float32)
            zf = z if z.dtype == torch.float32 else z.to(torch.float32)
            Af = A if A.dtype == torch.float32 else A.to(torch.float32)
            Bf = B if B.dtype == torch.float32 else B.to(torch.float32)

            grad_B = (go.t() @ zf).to(dtype=B.dtype) if ctx.needs_input_grad[2] else None
            if ctx.needs_input_grad[0]:
                grad_z = go @ Bf
                grad_x2 = grad_z @ Af
                grad_x = grad_x2.view(*ctx.input_shape)
            else:
                grad_x = None
        return grad_x, None, grad_B, None


class _LorActALinearFn(torch.autograd.Function):
    """
    Reference-style LoRAct path for the LoRA A projection.

    The forward is exactly ``Z = X A^T``. For backward, this function does not
    save the full activation X; it saves a LoRAct low-rank decomposition
    ``X ≈ U @ V`` and reconstructs X only for ``grad(A)``.

    ``lora_B`` is intentionally left to ordinary PyTorch autograd in
    ``Linear.forward``. This mirrors the official patch-style LoRAct structure
    more closely than the previous fused A+B custom function.
    """

    @staticmethod
    def forward(ctx, x, A, rank: int):
        x2 = x.reshape(-1, x.shape[-1])
        z = F.linear(x2, A)

        u, v = _loract_qb_compress(x2, int(rank))
        if x2.is_cuda and x2.dtype != torch.float32 and A.dtype == torch.float32:
            # Keep compressed factors in the same precision family as fp32 LoRA weights.
            u = u.to(dtype=torch.float32)
            v = v.to(dtype=torch.float32)
        ctx.save_for_backward(u.detach(), v.detach())
        ctx.A = A
        ctx.input_shape = x.shape
        return z.view(*x.shape[:-1], A.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        u, v = saved[0], saved[1]
        A = ctx.A
        grad_z = grad_output.reshape(-1, grad_output.shape[-1])

        same_dt = grad_z.dtype == u.dtype == v.dtype == A.dtype
        if same_dt:
            x_hat = u @ v
            grad_x2 = grad_z @ A if ctx.needs_input_grad[0] else None
            grad_A = grad_z.t() @ x_hat if ctx.needs_input_grad[1] else None
        else:
            gz = grad_z if grad_z.dtype == torch.float32 else grad_z.to(torch.float32)
            uf = u if u.dtype == torch.float32 else u.to(torch.float32)
            vf = v if v.dtype == torch.float32 else v.to(torch.float32)
            Af = A if A.dtype == torch.float32 else A.to(torch.float32)

            x_hat = uf @ vf
            grad_x2 = gz @ Af if ctx.needs_input_grad[0] else None
            grad_A = (gz.t() @ x_hat).to(dtype=A.dtype) if ctx.needs_input_grad[1] else None

        grad_x = grad_x2.view(*ctx.input_shape) if grad_x2 is not None else None
        return grad_x, grad_A, None


class _CareLoraLinearFn(torch.autograd.Function):
    """
    Fused CARE-LoRA LoRA path for linear layers.

    Forward computes ``Z = X A^T`` and the data-dependent decoder
    ``M* = (Z^T Z + λI)^{-1} Z^T X``. Backward approximates ``grad(A)`` as
    ``(Z^T dZ)^T M*`` while keeping ``grad(B)`` exact.

    ``X_hat = Z M*`` is never materialized in backward; only ``Z``, ``M*`` and
    small ``r×r`` / ``r×in`` intermediates are used.

    Important memory semantics:
      - The full activation X is not saved for backward.
      - M* is computed during forward while X is available; backward saves only (Z, M*).

    """

    @staticmethod
    def forward(ctx, x, A, B, scaling: float, module, pinv_lambda: float):
        use_speed_path = bool(_CARE_LORA_USE_SPEED_PATH)
        pinv_lambda = float(pinv_lambda)
        x2 = x.reshape(-1, x.shape[-1])
        z = F.linear(x2, A)

        # Build data-dependent M* during forward, then save only (Z, M*) for backward.
        # Keep H/C assembly and the solve on the default CUDA stream so ordering is
        # explicit and the implementation stays compatible with autograd hooks.
        with _linalg_amp_off_context(z):
            # The custom autograd forward already runs outside autograd, so
            # solve-path temporaries do not need additional detach() wrappers.
            z_f = z if z.dtype == torch.float32 else z.to(torch.float32)
            x_f = x2 if x2.dtype == torch.float32 else x2.to(torch.float32)
            row_mask = _care_lora_m_fit_row_mask_for_x(x, z_f, module)
            # H = Z^T Z + λ I, C = Z^T X; fallback handles pathological singular/NaN systems.
            r_i, in_i = int(z_f.shape[1]), int(x_f.shape[1])
            c_b, h_b, eye_b = _care_lora_prepare_linalg_workspace(module, r_i, in_i, z_f.device)
            M = _care_lora_stable_m_star(
                z_f,
                x_f,
                pinv_lambda,
                c_buf=c_b,
                h_buf=h_b,
                eye_buf=eye_b,
                stats_holder=module,
                row_mask=row_mask,
            )
            if use_speed_path:
                if x2.dtype == torch.float32 or A.dtype == torch.float32:
                    # Preserve M* in fp32 when LoRA A/B are fp32 under CUDA autocast.
                    M_fp32 = M
                else:
                    # Store M* in activation dtype and cast to fp32 in backward.
                    M_fp32 = M.to(dtype=x2.dtype).to(torch.float32)
                M_store = None
            else:
                preserve_fp32_m_store = (
                    x2.dtype != torch.float32
                    and A.dtype == torch.float32
                    and z.is_cuda
                )
                if preserve_fp32_m_store:
                    # Keep M* fp32 when LoRA A/B are fp32 under CUDA autocast.
                    M_store = M
                else:
                    M_store = M if M.dtype == x2.dtype else M.to(dtype=x2.dtype)
                M_fp32 = None

        # Hook-observable layout: keep activation tensor ``Z`` in ``save_for_backward``.
        #
        # Direct-ctx layout: cache ``z.detach()`` directly on ``ctx`` and use the
        # pre-rounded fp32 ``M*`` above.
        # Also stash the fp32 view of Z that was already built for the linalg
        # solve; backward uses the same value to build grad_A.
        ctx.use_speed_path = use_speed_path
        ctx.z_dtype = z.dtype
        if use_speed_path:
            # In mixed bf16 training, backward uses ``z_fp32`` for grad_A/grad_B
            # and only needs the original z dtype for casting grad_x. Avoid
            # retaining the extra bf16 z reference in that common path.
            ctx.z_saved = z.detach() if (z.dtype == A.dtype and z.dtype == B.dtype) else None
            ctx.z_fp32 = z_f
        else:
            # Hook-observable layout: save the fp32 Z copy when backward math uses fp32.
            ctx.save_for_backward((z if z_f is z else z_f).detach())
            ctx.z_fp32 = None
        ctx.M_store = M_store
        ctx.M_fp32 = M_fp32
        ctx.A = A
        ctx.B = B
        ctx.input_shape = x.shape
        ctx.scaling = float(scaling)
        ctx.module = module

        # Compute the LoRA branch output after the M* solve. The matmul inputs
        # are unchanged, so the returned values and custom backward are identical.
        del x_f, M, row_mask
        y2 = F.linear(z, B)
        return (y2 * scaling).view(*x.shape[:-1], B.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        use_speed_path = bool(getattr(ctx, "use_speed_path", _CARE_LORA_USE_SPEED_PATH))
        if use_speed_path:
            z = ctx.z_saved
        else:
            (z,) = ctx.saved_tensors
        M_store = ctx.M_store
        M_fp32 = ctx.M_fp32
        # Reuse the fp32 view of ``z`` computed during the forward M* solve.
        z_fp32_saved = getattr(ctx, "z_fp32", None)
        z_dtype = ctx.z_dtype if hasattr(ctx, "z_dtype") else z.dtype
        A = ctx.A
        B = ctx.B
        scaling = float(ctx.scaling)
        grad_out2 = grad_output.reshape(-1, grad_output.shape[-1]) * scaling

        # Use the direct matmul path when all operands share dtype. Mixed-dtype
        # training uses an fp32 fallback for the LoRA branch matmuls.
        _same_dt = (
            z is not None
            and grad_out2.dtype == B.dtype
            and grad_out2.dtype == A.dtype
            and grad_out2.dtype == z.dtype
        )

        module = ctx.module
        z_meta = z if z is not None else z_fp32_saved
        r_i = int(z_meta.shape[1])
        in_i = int(A.shape[1])
        out_i = int(B.shape[0])
        need_grad_z = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        n_i = int(grad_out2.shape[0])

        if _same_dt:
            # y = z @ B^T => dZ = dY @ B; z = x @ A^T => dX = dZ @ A
            grad_z = None
            if need_grad_z:
                dz_buf = _care_lora_prepare_grad_z_workspace(module, n_i, r_i, z_meta.device, grad_out2.dtype)
                torch.mm(grad_out2, B, out=dz_buf)
                grad_z = dz_buf
            # Do not keep a module-level (N x in_features) dX workspace; its
            # resident footprint scales like storing the full activation.

            need_ga = ctx.needs_input_grad[1]
            need_gb = ctx.needs_input_grad[2]
            t_buf = None
            ga_buf = None
            gb_buf = None
            if need_ga or need_gb:
                t_buf, ga_buf, gb_buf = _care_lora_prepare_bwd_workspace(
                    module, r_i, in_i, out_i, z_meta.device, grad_out2.dtype
                )
            grad_B = None
            if need_gb:
                torch.mm(grad_out2.t(), z, out=gb_buf)
                grad_B = gb_buf
            grad_A = None
            if need_ga:
                # Reuse forward's fp32 z for the grad_A computation.
                z_f = z_fp32_saved if (z_fp32_saved is not None) else (
                    z if z.dtype == torch.float32 else z.to(torch.float32)
                )
                U_f = grad_z if grad_z.dtype == torch.float32 else grad_z.to(torch.float32)
                M_f = M_fp32 if (M_fp32 is not None) else (
                    M_store if M_store.dtype == torch.float32 else M_store.to(torch.float32)
                )
                torch.mm(z_f.t(), U_f, out=t_buf)
                torch.mm(t_buf.t(), M_f, out=ga_buf)
                grad_A = ga_buf.to(dtype=A.dtype)
                del z_f, U_f, M_f
            del grad_out2
            # Delay the wide dX materialization until after the small grad_A/grad_B
            # work, so it does not overlap with temporary fp32 casts used above.
            grad_x2 = grad_z @ A if ctx.needs_input_grad[0] else None
            grad_x = grad_x2.view(*ctx.input_shape) if grad_x2 is not None else None
        else:
            # Cross-dtype fallback: use explicit fp32 casts and reuse forward
            # fp32 tensors when available.
            go = grad_out2 if grad_out2.dtype == torch.float32 else grad_out2.to(torch.float32)
            Bf = B if B.dtype == torch.float32 else B.to(torch.float32)
            zf = z_fp32_saved if (z_fp32_saved is not None) else (
                z if z.dtype == torch.float32 else z.to(torch.float32)
            )
            grad_z = None
            if need_grad_z:
                dz_buf = _care_lora_prepare_grad_z_workspace(module, n_i, r_i, z_meta.device, go.dtype)
                torch.mm(go, Bf, out=dz_buf)
                grad_z = dz_buf

            need_ga = ctx.needs_input_grad[1]
            need_gb = ctx.needs_input_grad[2]
            gb_mm_dt = torch.result_type(go, zf)
            t_buf = None
            ga_buf = None
            gb_buf = None
            if need_ga or need_gb:
                t_buf, ga_buf, gb_buf = _care_lora_prepare_bwd_workspace(
                    module, r_i, in_i, out_i, z_meta.device, gb_mm_dt
                )
            grad_B = None
            if need_gb:
                torch.mm(go.t(), zf, out=gb_buf)
                grad_B = gb_buf.to(dtype=B.dtype) if gb_buf.dtype != B.dtype else gb_buf
            grad_A = None
            if need_ga:
                M_f = M_fp32 if (M_fp32 is not None) else (
                    M_store if M_store.dtype == torch.float32 else M_store.to(torch.float32)
                )
                torch.mm(zf.t(), grad_z, out=t_buf)
                torch.mm(t_buf.t(), M_f, out=ga_buf)
                grad_A = ga_buf.to(dtype=A.dtype)
                del M_f
            del grad_out2, go, zf, Bf
            # Delay the wide dX materialization until after the small grad_A/grad_B
            # work, so it does not overlap with temporary fp32 casts used above.
            if ctx.needs_input_grad[0]:
                Af = A if A.dtype == torch.float32 else A.to(torch.float32)
                grad_x2 = grad_z @ Af
                del Af
            else:
                grad_x2 = None
            grad_x = (
                (grad_x2 if grad_x2.dtype == z_dtype else grad_x2.to(dtype=z_dtype)).view(*ctx.input_shape)
                if grad_x2 is not None
                else None
            )

        # Release ctx-held workspace tensor as soon as backward finishes this node.
        ctx.M_store = None
        ctx.M_fp32 = None
        ctx.z_fp32 = None
        if use_speed_path:
            ctx.z_saved = None
        return grad_x, grad_A, grad_B, None, None, None


class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")

    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        # Mode switches per adapter
        self.use_care_lora: dict[str, bool] = {}
        self.use_loract: dict[str, bool] = {}
        self.use_lorafa: dict[str, bool] = {}
        self.care_lora_pinv_lambda: dict[str, float] = {}
        self.loract_rank: dict[str, int] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("lora_ga"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.lora_ga_init(adapter_name)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False
        self.set_adapter(self.active_adapters)

        # Decide which custom LoRA path this adapter uses.
        peft_config = self.kwargs.get("peft_config", None)
        use_loract = (
            bool(getattr(peft_config, "use_loract", False))
            and (not use_dora)
        )
        use_care_lora = (
            bool(getattr(peft_config, "use_care_lora", False))
            and (not use_dora)
            and (not use_loract)
        )
        use_lorafa = (
            bool(getattr(peft_config, "use_lorafa", False))
            and (not use_dora)
            and (not use_loract)
            and (not use_care_lora)
        )
        self.use_care_lora[adapter_name] = use_care_lora
        self.use_loract[adapter_name] = use_loract
        self.use_lorafa[adapter_name] = use_lorafa
        self.care_lora_pinv_lambda[adapter_name] = float(getattr(peft_config, "care_lora_pinv_lambda", 1e-6) if peft_config is not None else 1e-6)
        self.loract_rank[adapter_name] = int(getattr(peft_config, "loract_rank", 64) if peft_config is not None else 64)

        if use_lorafa:
            self.lora_A[adapter_name].weight.requires_grad_(False)
        else:
            self.lora_A[adapter_name].weight.requires_grad_(True)



    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def lora_ga_init(self, adapter_name):
        def get_float_weight(model: torch.nn.Module):
            model: torch.nn.Linear

            device = model.weight.device
            in_features = model.in_features
            with torch.no_grad():
                I = torch.eye(in_features).to(device)
                w = model(I)
                if hasattr(model, "bias") and isinstance(model.bias, torch.Tensor):
                    w -= model.bias
                w = torch.transpose(w, 0, 1)
            w.requires_grad = model.weight.requires_grad
            return w
        
        if "grad" not in self.kwargs.keys():
            return

        base_layer = self.get_base_layer()
        weight = self.get_base_layer().weight
        device = weight.device
        dtype = weight.dtype
        quant_flag = False
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            """
            for quantized model, it is needed to get the floating point weights through forward, 
            which may take 1-2 minutes (7bmodel, all linear)
            """
            quant_flag = True
            weight = get_float_weight(base_layer)
            dtype = weight.dtype
        grad = self.kwargs["grad"].to(device).to(torch.float32)
        weight = weight.to(torch.float32)
        lora_r = self.r[adapter_name]
        init_config = self.kwargs["peft_config"]
        try:
            U, S, V = torch.svd_lowrank(
                grad.float(), q=min(4 * lora_r, min(grad.shape)),
                niter=4
            )
            V = V.T
        except Exception as e:
            raise ValueError("error from torch.svd_lowrank")
        # set direction
        if init_config.direction == "ArBr":
            B = U[:, 0: 2 * lora_r: 2]
            A = V[1: 2 * lora_r: 2, :]
        elif init_config.direction == "A2rBr":
            B = U[:, :lora_r]
            A = V[lora_r: 2 * lora_r, :]
        elif init_config.direction == "ArB2r":
            B = U[:, lora_r: 2 * lora_r]
            A = V[:lora_r, :]
        elif init_config.direction == "random":
            import random
            random_list = random.sample(range(2 * lora_r), 2 * lora_r)
            indexes_A = random_list[0:lora_r]
            indexes_B = random_list[lora_r:2 * lora_r]
            print(f"indexes_A={indexes_A}")
            print(f"indexes_B={indexes_B}")
            B = U[:, indexes_B]
            A = V[indexes_A, :]
        scaling_factor = self.scaling["default"]
        if init_config.scale == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config.scale == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config.scale == "stable":
            m, n = grad.shape  # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config.stable_gamma
            B = B * m ** 0.25 / gamma ** 0.5
            A = A * m ** 0.25 / gamma ** 0.5
        elif init_config.scale == "weightS":
            _, S, _ = torch.svd_lowrank(weight.data.float(), q=4 * lora_r, niter=4)
            S = S / self.scaling["default"]
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s

        offset = B @ A
        # Training type
        # consider dtype not in init_config
        if not hasattr(init_config, "dtype"):
            pass
        elif init_config.dtype == "bf16":
            A = A.to(torch.bfloat16)
            B = B.to(torch.bfloat16)
        elif init_config.dtype == "fp32":
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        scaling_factor = self.scaling["default"]
        offset *= scaling_factor
        if hasattr(init_config, "norm_clip") and init_config.norm_clip:
            # for numerical stability, offset's largest value must be less then weight's largest value
            ratio = torch.max(torch.abs(weight.data)) / torch.max(
                torch.abs(offset)
            )
            if ratio < 1:
                offset *= ratio
                A *= ratio ** 0.5
                B *= ratio ** 0.5

        weight.data -= offset

        self.lora_A[adapter_name].weight.data = A.contiguous()
        self.lora_B[adapter_name].weight.data = B.contiguous()
        if not quant_flag:
            weight = weight.data
            weight = weight.to(dtype)
            self.get_base_layer().weight.data = weight
        else:
            has_bias = True if base_layer.bias is not None else False
            float_linear = torch.nn.Linear(base_layer.in_features, base_layer.out_features, has_bias)
            if has_bias and isinstance(base_layer.bias.data, torch.Tensor):
                float_linear.bias.data = base_layer.bias.data
            float_linear.weight.data = weight.data
            import bitsandbytes
            if isinstance(base_layer, bitsandbytes.nn.Linear8bitLt):
                new_base_layer = type(base_layer)(base_layer.in_features, base_layer.out_features, has_bias,
                                                  has_fp16_weights=False)
            else:
                new_base_layer = type(base_layer)(base_layer.in_features, base_layer.out_features, has_bias, )
            new_base_layer.load_state_dict(float_linear.state_dict())
            new_base_layer.to(device)
            base_layer.__dict__.update(new_base_layer.__dict__)
            del new_base_layer

    def olora_init(self, adapter_name):
        dtype = self.get_base_layer().weight.dtype
        if dtype in [torch.int8, torch.uint8]:
            weight_tensor = dequantize_module_weight(self.get_base_layer())
        elif dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weight_tensor = self.get_base_layer().weight
        else:
            raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")

        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor.to(torch.float32)
        Q, R = torch.linalg.qr(weight_tensor.data)

        Qr, Rr = Q[:, :r], R[:r]

        self.lora_A[adapter_name].weight.data = Rr.contiguous()
        self.lora_B[adapter_name].weight.data = Qr.contiguous()

        weight_tensor.data -= scale_factor * self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        weight_tensor = weight_tensor.to(dtype)
        self.get_base_layer().weight.data = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
            )
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
        lora_B = Vr @ torch.diag(torch.sqrt(Sr))
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def dora_init(self, adapter_name: str) -> None:
        if not self.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(self, "fan_in_fan_out", False))
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        place_on_cpu = self.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
        if self.ephemeral_gpu_offload:
            if lora_A.device.type == "cuda":
                lora_B = lora_B.to(lora_A.device)
            else:
                if lora_B.device.type != "cuda":
                    lora_B = lora_B.to("cuda")
                lora_A = lora_A.to(lora_B.device)
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling, place_on_cpu=place_on_cpu
        )
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        unique_adapters = set(self.active_adapters)
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            if sub_batch.dtype != lora_A.weight.dtype:
                sub_batch = sub_batch.to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                base_weight_dtype = base_layer.weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = (orig_weights + delta_weight).to(dtype=base_weight_dtype)
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = (dora_factor * (orig_weights + delta_weight)).to(dtype=base_weight_dtype)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = (base_layer.weight.data + delta_weight).to(dtype=base_weight_dtype)
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight.to(dtype=base_weight_dtype)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                use_care_lora_projection_path = self.use_care_lora.get(active_adapter, False)
                use_loract_path = self.use_loract.get(active_adapter, False)
                use_care_lora_autocast_input = False
                if (
                    use_care_lora_projection_path
                    and x.is_cuda
                    and x.dtype != lora_A.weight.dtype
                    and lora_A.weight.dtype == torch.float32
                    and _cuda_autocast_is_enabled()
                ):
                    # The autocast shortcut is valid only when dropout is a no-op.
                    use_care_lora_autocast_input = isinstance(dropout, nn.Identity) or float(
                        getattr(dropout, "p", 1.0)
                    ) == 0.0
                if not self.use_dora[active_adapter]:
                    x_cast = (
                        x
                        if (
                            x.dtype == lora_A.weight.dtype
                            or use_care_lora_autocast_input
                        )
                        else x.to(lora_A.weight.dtype)
                    )
                    x_drop = dropout(x_cast)

                    if use_loract_path:
                        # Reference-style LoRAct baseline: patch the LoRA-A projection
                        # only; LoRA-B remains ordinary autograd and sees exact Z.
                        rank = 64
                        if isinstance(getattr(self, "loract_rank", None), dict):
                            rank = int(self.loract_rank.get(active_adapter, rank))
                        need_loract_backward = torch.is_grad_enabled() and (
                            x_drop.requires_grad or lora_A.weight.requires_grad or lora_B.weight.requires_grad
                        )
                        if need_loract_backward:
                            if _LORA_ACTIVATION_TRACKING_ENABLED:
                                with _LORA_ACTIVATION_CAPTURE_CONTEXT:
                                    z_loract = _LorActALinearFn.apply(
                                        x_drop,
                                        lora_A.weight,
                                        int(rank),
                                    )
                                with _LORA_ACTIVATION_CAPTURE_CONTEXT:
                                    lora_delta = F.linear(z_loract, lora_B.weight)
                                result = result + lora_delta * scaling
                            else:
                                z_loract = _LorActALinearFn.apply(
                                    x_drop,
                                    lora_A.weight,
                                    int(rank),
                                )
                                result = result + F.linear(z_loract, lora_B.weight) * scaling
                        else:
                            result = result + F.linear(F.linear(x_drop, lora_A.weight), lora_B.weight) * scaling
                    elif self.use_care_lora.get(active_adapter, False):
                        # CARE-LoRA: forward builds M* from (Z, X); backward uses (Z, M*).
                        pinv_lambda = float(self.care_lora_pinv_lambda.get(active_adapter, 1e-6))
                        need_care_lora_backward = torch.is_grad_enabled() and (
                            x_drop.requires_grad or lora_A.weight.requires_grad or lora_B.weight.requires_grad
                        )
                        if need_care_lora_backward:
                            if _LORA_ACTIVATION_TRACKING_ENABLED:
                                with _LORA_ACTIVATION_CAPTURE_CONTEXT:
                                    lora_delta = _CareLoraLinearFn.apply(
                                        x_drop,
                                        lora_A.weight,
                                        lora_B.weight,
                                        scaling,
                                        self,
                                        pinv_lambda,
                                    )
                                result = result + lora_delta
                            else:
                                result = result + _CareLoraLinearFn.apply(
                                    x_drop,
                                    lora_A.weight,
                                    lora_B.weight,
                                    scaling,
                                    self,
                                    pinv_lambda,
                                )
                        else:
                            result = result + F.linear(F.linear(x_drop, lora_A.weight), lora_B.weight) * scaling
                    else:
                        if self.use_lorafa.get(active_adapter, False):
                            # LoRA-FA: A frozen, train only B, avoid saving X.
                            if _LORA_ACTIVATION_TRACKING_ENABLED:
                                with _LORA_ACTIVATION_CAPTURE_CONTEXT:
                                    lora_delta = _LoraFaLinearFn.apply(
                                        x_drop,
                                        lora_A.weight,
                                        lora_B.weight,
                                        scaling,
                                    )
                                result = result + lora_delta
                            else:
                                result = result + _LoraFaLinearFn.apply(
                                    x_drop,
                                    lora_A.weight,
                                    lora_B.weight,
                                    scaling,
                                )
                        else:
                            # Standard LoRA behavior
                            if _LORA_ACTIVATION_TRACKING_ENABLED:
                                with _LORA_ACTIVATION_CAPTURE_CONTEXT:
                                    z_lora = lora_A(x_drop)
                                with _LORA_ACTIVATION_CAPTURE_CONTEXT:
                                    lora_delta = lora_B(z_lora)
                                result = result + lora_delta * scaling
                            else:
                                result = result + lora_B(lora_A(x_drop)) * scaling
                else:
                    saved_x = None
                    dora_full_cast_bytes = (
                        x.numel() * lora_A.weight.element_size() if x.dtype != lora_A.weight.dtype else 0
                    )
                    use_dora_row_chunk_input = (
                        get_dora_enable_chunked_ops()
                        and x.is_cuda
                        and x.dtype != lora_A.weight.dtype
                        and isinstance(dropout, nn.Identity)
                        and (
                            (
                                get_dora_force_row_chunk_for_narrow_output()
                                and int(self.in_features) > int(self.out_features) * get_dora_narrow_output_ratio()
                            )
                            or int(dora_full_cast_bytes) >= get_dora_row_chunk_cast_bytes()
                        )
                    )
                    if use_dora_row_chunk_input:
                        x_cast = x
                        saved_x = x.detach()
                    else:
                        x_cast = x if x.dtype == lora_A.weight.dtype else x.to(lora_A.weight.dtype)
                        x_cast = dropout(x_cast)
                    result = self.lora_magnitude_vector[active_adapter](
                        x_cast,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=result,
                        saved_x=saved_x,
                    )

            if result.dtype != torch_result_dtype:
                result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Embedding(nn.Module, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)

        if use_dora:
            raise ValueError(f"{self.__class__.__name__} does not support DoRA yet, please set it to False")

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        peft_config = self.kwargs.get("peft_config", None)
        if peft_config is not None and (
            bool(getattr(peft_config, "use_care_lora", False))
            or bool(getattr(peft_config, "use_loract", False))
            or bool(getattr(peft_config, "use_lorafa", False))
        ):
            raise ValueError(
                "CARE-LoRA/LoRAct/LoRA-FA custom paths are implemented for Linear layers only in this repo. "
                "Please exclude Embedding layers from target_modules."
            )

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights + self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data = base_layer.weight.data + self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]
                after_A = self._embed(x, embedding_A)
                result = result + (after_A @ embedding_B) * scaling
            if result.dtype != torch_result_dtype:
                result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(nn.Module, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        peft_config = self.kwargs.get("peft_config", None)
        if peft_config is not None and (
            bool(getattr(peft_config, "use_care_lora", False))
            or bool(getattr(peft_config, "use_loract", False))
            or bool(getattr(peft_config, "use_lorafa", False))
        ):
            raise ValueError(
                "CARE-LoRA/LoRAct/LoRA-FA custom paths are implemented for Linear layers only in this repo. "
                "Please exclude Conv2d layers from target_modules."
            )

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def dora_init(self, adapter_name: str) -> None:
        if self.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraConv2dLayer(fan_in_fan_out=False)
        lora_A = self.lora_A[adapter_name].weight
        lora_B = self.lora_B[adapter_name].weight
        scaling = self.scaling[adapter_name]
        dora_layer.update_layer(base_layer=self.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling)
        self.lora_magnitude_vector[adapter_name] = dora_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                base_weight_dtype = base_layer.weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights = (orig_weights + delta_weight).to(dtype=base_weight_dtype)
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        orig_weights = (
                            dora_factor.view(-1, 1, 1, 1) * (orig_weights + delta_weight)
                        ).to(dtype=base_weight_dtype)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = (base_layer.weight.data + delta_weight).to(dtype=base_weight_dtype)
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(base_layer.weight, delta_weight, scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        new_weight = dora_factor.view(-1, 1, 1, 1) * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight.to(dtype=base_weight_dtype)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1, 1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                F.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x_cast = x if x.dtype == lora_A.weight.dtype else x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x_cast))) * scaling
                else:
                    x_cast = dropout(x_cast)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x_cast,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            if result.dtype != torch_result_dtype:
                result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
