# Copyright 2024-present the HuggingFace Inc. team.
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

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose


_DORA_ROW_CHUNK_CAST_BYTES = int(32 * 1024 * 1024)
_DORA_ENABLE_CHUNKED_OPS = False
_DORA_FORCE_ROW_CHUNK_FOR_NARROW_OUTPUT = False
_DORA_NARROW_OUTPUT_RATIO = 1.0


def set_dora_enable_chunked_ops(value: bool) -> None:
    """Control whether DoRA uses extra chunked forward/norm paths."""
    global _DORA_ENABLE_CHUNKED_OPS
    _DORA_ENABLE_CHUNKED_OPS = bool(value)


def get_dora_enable_chunked_ops() -> bool:
    return bool(_DORA_ENABLE_CHUNKED_OPS)


def set_dora_row_chunk_cast_mib(value: float) -> None:
    """Set the DoRA row-chunk trigger threshold in MiB for the current process."""
    global _DORA_ROW_CHUNK_CAST_BYTES
    _DORA_ROW_CHUNK_CAST_BYTES = int(float(value) * 1024 * 1024)


def get_dora_row_chunk_cast_bytes() -> int:
    return int(_DORA_ROW_CHUNK_CAST_BYTES)


def set_dora_force_row_chunk_for_narrow_output(value: bool) -> None:
    """Control whether DoRA row-chunks layers with in_features > out_features."""
    global _DORA_FORCE_ROW_CHUNK_FOR_NARROW_OUTPUT
    _DORA_FORCE_ROW_CHUNK_FOR_NARROW_OUTPUT = bool(value)


def get_dora_force_row_chunk_for_narrow_output() -> bool:
    return bool(_DORA_FORCE_ROW_CHUNK_FOR_NARROW_OUTPUT)


def set_dora_narrow_output_ratio(value: float) -> None:
    """Set the in/out feature ratio that triggers narrow-output row chunking."""
    global _DORA_NARROW_OUTPUT_RATIO
    _DORA_NARROW_OUTPUT_RATIO = max(1.0, float(value))


def get_dora_narrow_output_ratio() -> float:
    return float(_DORA_NARROW_OUTPUT_RATIO)


def _dora_weight_norm_chunked(
    weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
    chunk_rows: int = 1024,
) -> torch.Tensor:
    """Compute ||W + scaling * BA|| row-wise without materializing the full BA."""
    norms = []
    out_features = int(weight.shape[0])
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        lora_weight_chunk = lora_B[start:end] @ lora_A
        weight_chunk = weight[start:end].to(dtype=lora_weight_chunk.dtype, device=lora_weight_chunk.device)
        lora_weight_chunk.mul_(float(scaling)).add_(weight_chunk)
        norms.append(torch.linalg.norm(lora_weight_chunk, dim=1))
        del lora_weight_chunk, weight_chunk
    return torch.cat(norms, dim=0).detach()


def _dora_tensor_cache_identity(tensor: torch.Tensor) -> tuple:
    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.device),
        tensor.dtype,
        int(getattr(tensor, "_version", 0)),
    )


def _dora_autocast_cache_identity(device: torch.device) -> tuple:
    device_type = str(device).split(":", 1)[0]
    try:
        enabled = bool(torch.is_autocast_enabled(device_type))
    except TypeError:
        enabled = bool(torch.is_autocast_enabled()) if device_type == "cuda" else False
    except Exception:
        enabled = False

    autocast_dtype = None
    if enabled:
        try:
            autocast_dtype = torch.get_autocast_dtype(device_type)
        except Exception:
            if device_type == "cuda":
                try:
                    autocast_dtype = torch.get_autocast_gpu_dtype()
                except Exception:
                    autocast_dtype = None
            elif device_type == "cpu":
                try:
                    autocast_dtype = torch.get_autocast_cpu_dtype()
                except Exception:
                    autocast_dtype = None
    return device_type, enabled, autocast_dtype


def _dora_linear_weight_norm_cache_key(
    *,
    weight: torch.Tensor,
    lora_A_weight: torch.Tensor,
    lora_B_weight: torch.Tensor,
    scaling: float,
    fan_in_fan_out: bool,
    use_chunked_ops: bool,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> tuple:
    return (
        _dora_tensor_cache_identity(weight),
        _dora_tensor_cache_identity(lora_A_weight),
        _dora_tensor_cache_identity(lora_B_weight),
        float(scaling),
        bool(fan_in_fan_out),
        bool(use_chunked_ops),
        compute_dtype,
        str(device),
        _dora_autocast_cache_identity(device),
    )


def _compute_dora_linear_weight_norm(
    *,
    weight: torch.Tensor,
    lora_A_weight: torch.Tensor,
    lora_B_weight: torch.Tensor,
    scaling: float,
    fan_in_fan_out: bool,
    use_chunked_ops: bool,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    weight = transpose(weight, fan_in_fan_out).to(device=device)
    lora_A = lora_A_weight.to(dtype=compute_dtype, device=device)
    lora_B = lora_B_weight.to(dtype=compute_dtype, device=device)
    if use_chunked_ops:
        return _dora_weight_norm_chunked(weight, lora_A, lora_B, float(scaling))

    weight_compute = weight.to(dtype=compute_dtype, device=device)
    lora_weight = lora_B @ lora_A
    return torch.linalg.norm(weight_compute + float(scaling) * lora_weight, dim=1).detach()


def _dora_forward_chunked(
    x: torch.Tensor,
    base_result: torch.Tensor | None,
    weight: torch.Tensor,
    z: torch.Tensor,
    lora_B: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scaling: float,
    chunk_rows: int = 1024,
) -> torch.Tensor:
    """Compute ``base_result + DoRA branch`` in output chunks."""
    out_features = int(lora_B.shape[0])
    chunk_rows = max(1, int(chunk_rows))
    result = (
        base_result
        if base_result is not None
        else torch.empty(*x.shape[:-1], out_features, dtype=x.dtype, device=x.device)
    )
    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        weight_chunk = weight[start:end].to(dtype=x.dtype, device=x.device)
        base_chunk = F.linear(x, weight_chunk)
        base_chunk.mul_(mag_norm_scale[..., start:end] - 1)
        lora_chunk = F.linear(z, lora_B[start:end])
        lora_chunk.mul_(mag_norm_scale[..., start:end]).mul_(float(scaling))
        base_chunk.add_(lora_chunk)
        if base_result is not None:
            base_chunk.add_(
                base_result[..., start:end].to(dtype=base_chunk.dtype, device=base_chunk.device)
            )
        result[..., start:end].copy_(base_chunk.to(dtype=result.dtype))
        del weight_chunk, base_chunk, lora_chunk
    return result


def _dora_forward_row_chunked(
    x: torch.Tensor,
    base_result: torch.Tensor,
    weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    mag_norm_scale: torch.Tensor,
    scaling: float,
    row_chunk: int = 2048,
    out_chunk: int = 4096,
    out_of_place: bool = False,
) -> torch.Tensor:
    """DoRA forward for very wide inputs without materializing full fp32 X."""
    input_shape = x.shape
    x2 = x.reshape(-1, input_shape[-1])
    result = torch.empty_like(base_result) if out_of_place else base_result
    base_result2 = base_result.reshape(-1, base_result.shape[-1])
    result2 = result.reshape(-1, result.shape[-1])
    n_rows = int(x2.shape[0])
    out_features = int(lora_B.shape[0])
    row_chunk = max(1, int(row_chunk))
    out_chunk = max(1, int(out_chunk))
    for row_start in range(0, n_rows, row_chunk):
        row_end = min(row_start + row_chunk, n_rows)
        x_chunk = x2[row_start:row_end].to(dtype=lora_A.dtype, device=lora_A.device)
        z_chunk = F.linear(x_chunk, lora_A)
        for out_start in range(0, out_features, out_chunk):
            out_end = min(out_start + out_chunk, out_features)
            weight_chunk = weight[out_start:out_end].to(dtype=x_chunk.dtype, device=x_chunk.device)
            base_chunk = F.linear(x_chunk, weight_chunk)
            base_chunk.mul_(mag_norm_scale[..., out_start:out_end] - 1)
            lora_chunk = F.linear(z_chunk, lora_B[out_start:out_end])
            lora_chunk.mul_(mag_norm_scale[..., out_start:out_end]).mul_(float(scaling))
            base_chunk.add_(lora_chunk)
            base_chunk.add_(base_result2[row_start:row_end, out_start:out_end].to(dtype=base_chunk.dtype))
            result2[row_start:row_end, out_start:out_end].copy_(base_chunk.to(dtype=result2.dtype))
            del weight_chunk, base_chunk, lora_chunk
        del x_chunk, z_chunk
    return result


class _DoraLinearFn(torch.autograd.Function):
    """
    Memory-lean DoRA linear branch.

    The native autograd expression for DoRA saves output-sized intermediates
    such as ``base_no_bias`` and ``lora_result``. This function saves only the
    input activation and recomputes those tensors in backward.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        x_saved,
        base_result,
        base_weight,
        lora_A_weight,
        lora_B_weight,
        magnitude,
        precomputed_weight_norm,
        scaling: float,
        fan_in_fan_out: bool,
    ):
        weight = transpose(base_weight, fan_in_fan_out).to(device=x.device)
        compute_dtype = lora_A_weight.dtype
        lora_A = lora_A_weight.to(dtype=compute_dtype, device=x.device)
        lora_B = lora_B_weight.to(dtype=compute_dtype, device=x.device)
        use_chunked_ops = get_dora_enable_chunked_ops()
        if torch.is_tensor(precomputed_weight_norm):
            weight_norm = precomputed_weight_norm.detach().to(dtype=compute_dtype, device=x.device)
        else:
            if use_chunked_ops:
                weight_norm = _dora_weight_norm_chunked(weight, lora_A, lora_B, float(scaling))
            else:
                weight_compute = weight.to(dtype=compute_dtype, device=x.device)
                lora_weight = lora_B @ lora_A
                weight_norm = torch.linalg.norm(weight_compute + float(scaling) * lora_weight, dim=1).detach()
        mag_norm_scale = (magnitude.to(dtype=compute_dtype, device=x.device) / weight_norm).view(1, -1)

        has_base_result = torch.is_tensor(base_result)
        full_cast_bytes = x.numel() * torch.empty((), dtype=compute_dtype).element_size()
        use_row_chunk = (
            use_chunked_ops
            and x.dtype != compute_dtype
            and has_base_result
            and (
                (
                    get_dora_force_row_chunk_for_narrow_output()
                    and int(x.shape[-1]) > int(lora_B.shape[0]) * get_dora_narrow_output_ratio()
                )
                or int(full_cast_bytes) >= _DORA_ROW_CHUNK_CAST_BYTES
            )
        )
        use_row_chunk_out_of_place = (
            use_row_chunk
            # Very narrow output projections can make in-place row-chunk CopySlices
            # backward hit a PyTorch shape check. Write those outputs out-of-place
            # while keeping the memory-saving path for wide layers.
            and int(lora_B.shape[0]) < 1024
            and int(x.shape[-1]) > int(lora_B.shape[0])
        )
        if has_base_result and not use_row_chunk_out_of_place:
            ctx.mark_dirty(base_result)
        if use_row_chunk:
            result = _dora_forward_row_chunked(
                x,
                base_result,
                weight,
                lora_A,
                lora_B,
                mag_norm_scale,
                float(scaling),
                out_of_place=use_row_chunk_out_of_place,
            )
        else:
            x_compute = x if x.dtype == compute_dtype else x.to(compute_dtype)
            z = F.linear(x_compute, lora_A)
            if use_chunked_ops:
                result = _dora_forward_chunked(
                    x_compute,
                    base_result if has_base_result else None,
                    weight,
                    z,
                    lora_B,
                    mag_norm_scale,
                    float(scaling),
                )
            else:
                weight_compute = weight.to(dtype=x_compute.dtype, device=x_compute.device)
                base_no_bias = F.linear(x_compute, weight_compute)
                lora_result = F.linear(z, lora_B)
                base_no_bias.mul_(mag_norm_scale - 1)
                lora_result.mul_(mag_norm_scale).mul_(float(scaling))
                base_no_bias.add_(lora_result)
                if has_base_result:
                    base_result.add_(base_no_bias.to(dtype=base_result.dtype))
                    result = base_result
                else:
                    result = base_no_bias

        if use_row_chunk:
            x_for_backward = x_saved.detach() if torch.is_tensor(x_saved) else x.detach()
        else:
            x_for_backward = x_compute
        ctx.save_for_backward(x_for_backward)
        ctx.has_base_result = bool(has_base_result)
        ctx.base_result_dtype = base_result.dtype if has_base_result else None
        ctx.compute_dtype = compute_dtype
        ctx.use_chunked_ops = bool(use_chunked_ops)
        ctx.use_row_chunk = bool(use_row_chunk)
        ctx.base_weight = base_weight
        ctx.lora_A_weight = lora_A_weight
        ctx.lora_B_weight = lora_B_weight
        ctx.magnitude = magnitude
        ctx.weight_norm = weight_norm
        ctx.scaling = float(scaling)
        ctx.fan_in_fan_out = bool(fan_in_fan_out)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        base_weight = ctx.base_weight
        lora_A_weight = ctx.lora_A_weight
        lora_B_weight = ctx.lora_B_weight
        magnitude = ctx.magnitude
        weight_norm = ctx.weight_norm
        scaling = ctx.scaling

        compute_dtype = getattr(ctx, "compute_dtype", x.dtype)
        weight = transpose(base_weight, ctx.fan_in_fan_out).to(device=x.device)
        lora_A = lora_A_weight.to(dtype=compute_dtype, device=x.device)
        lora_B = lora_B_weight.to(dtype=compute_dtype, device=x.device)
        mag_norm_scale = (
            magnitude.to(dtype=compute_dtype, device=x.device) / weight_norm.to(dtype=compute_dtype)
        ).view(1, -1)

        if bool(getattr(ctx, "use_row_chunk", False)):
            input_shape = x.shape
            x_saved2 = x.reshape(-1, x.shape[-1])
            grad2_all = grad_output.reshape(-1, grad_output.shape[-1])
            n_rows = int(x_saved2.shape[0])
            out_features = int(lora_B.shape[0])
            row_chunk = 2048
            out_chunk = 4096
            reduce_dims = tuple(range(grad_output.dim() - 1))
            mag_norm_flat = mag_norm_scale.reshape(-1)
            weight_norm_flat = weight_norm.to(dtype=compute_dtype).reshape(-1)

            need_grad_x = bool(ctx.needs_input_grad[0])
            need_grad_A = bool(ctx.needs_input_grad[4])
            need_grad_B = bool(ctx.needs_input_grad[5])
            need_grad_z = need_grad_x or need_grad_A

            grad_x_out = torch.empty_like(x_saved2) if need_grad_x else None
            grad_A_work = torch.zeros_like(lora_A) if need_grad_A else None
            grad_B_work = torch.zeros_like(lora_B) if need_grad_B else None
            grad_magnitude_work = torch.zeros_like(magnitude.to(dtype=compute_dtype, device=x.device))

            for row_start in range(0, n_rows, row_chunk):
                row_end = min(row_start + row_chunk, n_rows)
                x_chunk = x_saved2[row_start:row_end].to(dtype=compute_dtype, device=x.device)
                grad_out_rows = grad2_all[row_start:row_end].to(dtype=compute_dtype)
                z_chunk = F.linear(x_chunk, lora_A)
                grad_z_chunk = torch.zeros_like(z_chunk) if need_grad_z else None
                grad_x_chunk = torch.zeros_like(x_chunk) if need_grad_x else None

                for out_start in range(0, out_features, out_chunk):
                    out_end = min(out_start + out_chunk, out_features)
                    weight_chunk = weight[out_start:out_end].to(dtype=compute_dtype, device=x.device)
                    base_chunk = F.linear(x_chunk, weight_chunk)
                    lora_inner_chunk = F.linear(z_chunk, lora_B[out_start:out_end])
                    lora_inner_chunk.mul_(float(scaling)).add_(base_chunk)
                    gm_chunk = (
                        grad_out_rows[:, out_start:out_end] * lora_inner_chunk
                    ).sum(dim=0) / weight_norm_flat[out_start:out_end]
                    grad_magnitude_work[out_start:out_end].add_(gm_chunk)
                    del base_chunk, lora_inner_chunk

                    grad_out_chunk = grad_out_rows[:, out_start:out_end]
                    base_scale = (mag_norm_flat[out_start:out_end] - 1).to(dtype=compute_dtype)
                    lora_scale = (mag_norm_flat[out_start:out_end] * float(scaling)).to(dtype=compute_dtype)
                    grad_lora_chunk = grad_out_chunk * lora_scale
                    if need_grad_x:
                        grad_base_chunk = grad_out_chunk * base_scale
                        grad_x_chunk.add_(grad_base_chunk @ weight_chunk)
                        del grad_base_chunk
                    if need_grad_z:
                        grad_z_chunk.add_(grad_lora_chunk @ lora_B[out_start:out_end])
                    if need_grad_B:
                        grad_B_work[out_start:out_end].add_(grad_lora_chunk.t() @ z_chunk)
                    del weight_chunk, grad_lora_chunk

                if need_grad_x:
                    grad_x_chunk.add_(grad_z_chunk @ lora_A)
                    grad_x_out[row_start:row_end].copy_(grad_x_chunk.to(dtype=grad_x_out.dtype))
                if need_grad_A:
                    grad_A_work.add_(grad_z_chunk.t() @ x_chunk)
                del x_chunk, z_chunk, grad_z_chunk, grad_x_chunk

            grad_x = grad_x_out.view(input_shape) if need_grad_x else None
            grad_A = grad_A_work.to(dtype=lora_A_weight.dtype) if need_grad_A else None
            grad_B = grad_B_work.to(dtype=lora_B_weight.dtype) if need_grad_B else None
            grad_magnitude = grad_magnitude_work.to(dtype=magnitude.dtype)
            grad_base_result = None
            if bool(getattr(ctx, "has_base_result", False)):
                base_result_dtype = getattr(ctx, "base_result_dtype", None)
                grad_base_result = (
                    grad_output if grad_output.dtype == base_result_dtype else grad_output.to(base_result_dtype)
                )
            return grad_x, None, grad_base_result, None, grad_A, grad_B, grad_magnitude, None, None, None

        x = x if x.dtype == compute_dtype else x.to(compute_dtype)
        x2 = x.reshape(-1, x.shape[-1])
        grad2 = grad_output.reshape(-1, grad_output.shape[-1]).to(dtype=x.dtype)

        z = F.linear(x, lora_A)
        z2 = z.reshape(-1, z.shape[-1])
        reduce_dims = tuple(range(grad_output.dim() - 1))

        if not bool(getattr(ctx, "use_chunked_ops", False)):
            weight_compute = weight.to(dtype=x.dtype, device=x.device)
            base_no_bias = F.linear(x, weight_compute)
            lora_result = F.linear(z, lora_B)
            dora_inner = base_no_bias + lora_result * float(scaling)
            grad_magnitude = (
                grad_output.to(dtype=x.dtype) * dora_inner
            ).sum(dim=reduce_dims) / weight_norm.to(dtype=x.dtype)
            grad_magnitude = grad_magnitude.to(dtype=magnitude.dtype)

            grad_base = grad2 * (mag_norm_scale - 1).to(dtype=x.dtype)
            grad_lora = grad2 * (mag_norm_scale * float(scaling)).to(dtype=x.dtype)
            grad_x = None
            grad_A = None
            grad_B = None
            grad_z = None

            if ctx.needs_input_grad[0]:
                grad_x_base = grad_base @ weight_compute
                grad_z = grad_lora @ lora_B
                grad_x_lora = grad_z @ lora_A
                grad_x = (grad_x_base + grad_x_lora).view_as(x)

            if ctx.needs_input_grad[4] or ctx.needs_input_grad[5]:
                if grad_z is None:
                    grad_z = grad_lora @ lora_B
                if ctx.needs_input_grad[5]:
                    grad_B = (grad_lora.t() @ z2).to(dtype=lora_B_weight.dtype)
                if ctx.needs_input_grad[4]:
                    grad_A = (grad_z.t() @ x2).to(dtype=lora_A_weight.dtype)

            grad_base_result = None
            if bool(getattr(ctx, "has_base_result", False)):
                base_result_dtype = getattr(ctx, "base_result_dtype", None)
                grad_base_result = (
                    grad_output if grad_output.dtype == base_result_dtype else grad_output.to(base_result_dtype)
                )
            return grad_x, None, grad_base_result, None, grad_A, grad_B, grad_magnitude, None, None, None

        mag_norm_flat = mag_norm_scale.reshape(-1)
        weight_norm_flat = weight_norm.to(dtype=x.dtype).reshape(-1)

        need_grad_x = bool(ctx.needs_input_grad[0])
        need_grad_A = bool(ctx.needs_input_grad[4])
        need_grad_B = bool(ctx.needs_input_grad[5])
        need_grad_z = need_grad_x or need_grad_A

        grad_x2 = torch.zeros_like(x2) if need_grad_x else None
        grad_z = torch.zeros_like(z2) if need_grad_z else None
        grad_B_work = torch.empty_like(lora_B) if need_grad_B else None
        grad_magnitude_chunks = []

        chunk_rows = 1024
        out_features = int(lora_B.shape[0])
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)

            weight_chunk = weight[start:end].to(dtype=x.dtype, device=x.device)
            base_chunk = F.linear(x, weight_chunk)
            lora_chunk = F.linear(z, lora_B[start:end])
            lora_chunk.mul_(float(scaling)).add_(base_chunk)
            gm_chunk = (
                grad_output[..., start:end].to(dtype=x.dtype) * lora_chunk
            ).sum(dim=reduce_dims) / weight_norm_flat[start:end]
            grad_magnitude_chunks.append(gm_chunk)
            del base_chunk, lora_chunk

            grad_out_chunk = grad2[:, start:end]
            base_scale = (mag_norm_flat[start:end] - 1).to(dtype=x.dtype)
            lora_scale = (mag_norm_flat[start:end] * float(scaling)).to(dtype=x.dtype)
            grad_lora_chunk = grad_out_chunk * lora_scale

            if need_grad_x:
                grad_base_chunk = grad_out_chunk * base_scale
                grad_x2.add_(grad_base_chunk @ weight_chunk)
                del grad_base_chunk
            if need_grad_z:
                grad_z.add_(grad_lora_chunk @ lora_B[start:end])
            if need_grad_B:
                grad_B_work[start:end].copy_(grad_lora_chunk.t() @ z2)
            del grad_lora_chunk, weight_chunk

        grad_magnitude = torch.cat(grad_magnitude_chunks, dim=0).to(dtype=magnitude.dtype)
        grad_x = None
        grad_A = None
        grad_B = None
        if need_grad_x:
            grad_x2.add_(grad_z @ lora_A)
            grad_x = grad_x2.view_as(x)
        if need_grad_A:
            grad_A = (grad_z.t() @ x2).to(dtype=lora_A_weight.dtype)
        if need_grad_B:
            grad_B = grad_B_work.to(dtype=lora_B_weight.dtype)

        grad_base_result = None
        if bool(getattr(ctx, "has_base_result", False)):
            base_result_dtype = getattr(ctx, "base_result_dtype", None)
            grad_base_result = (
                grad_output if grad_output.dtype == base_result_dtype else grad_output.to(base_result_dtype)
            )
        return grad_x, None, grad_base_result, None, grad_A, grad_B, grad_magnitude, None, None, None


class DoraLinearLayer(nn.Module):
    def __init__(self, fan_in_fan_out):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out
        self._weight_norm_cache_key = None
        self._weight_norm_cache_value = None

    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def update_layer(self, *, base_layer, lora_A, lora_B, scaling, place_on_cpu=False) -> None:
        # temporarily convert fp16 to fp32, as fp16 can cause trouble on CPU with PyTorch < 2.2
        dtype_is_fp16 = lora_A.dtype == torch.float16
        if dtype_is_fp16:
            lora_A = lora_A.float()
            lora_B = lora_B.float()

        with gather_params_ctx(base_layer.parameters()):
            if base_layer.__class__.__name__ == "Linear4bit":
                # We have to create a copy of the base layer, otherwise, FSDP will throw an error. 8bit does not work
                # yet because Int8Params cannot be correctly deep-copied (attributes vanish)
                base_layer = deepcopy(base_layer)

            weight = dequantize_module_weight(base_layer)
            if weight.data.ndim == 4:  # For handling LoRAs applied to Conv2Ds.
                lora_weight = torch.mm(lora_B.flatten(start_dim=1), lora_A.flatten(start_dim=1))
                lora_weight = lora_weight.reshape(weight.shape)
            else:
                lora_weight = lora_B @ lora_A

            if dtype_is_fp16:
                lora_weight = lora_weight.half()
            weight_norm = self.get_weight_norm(weight.to(lora_A.device), lora_weight, scaling)

        if place_on_cpu:
            weight_norm = weight_norm.to("cpu")
        self.weight = nn.Parameter(weight_norm, requires_grad=True)
        self._weight_norm_cache_key = None
        self._weight_norm_cache_value = None

    def _get_weight_norm_cached(self, *, x, weight, lora_A_weight, lora_B_weight, scaling):
        compute_dtype = lora_A_weight.dtype
        use_chunked_ops = get_dora_enable_chunked_ops()
        cache_key = _dora_linear_weight_norm_cache_key(
            weight=weight,
            lora_A_weight=lora_A_weight,
            lora_B_weight=lora_B_weight,
            scaling=float(scaling),
            fan_in_fan_out=bool(self.fan_in_fan_out),
            use_chunked_ops=bool(use_chunked_ops),
            compute_dtype=compute_dtype,
            device=x.device,
        )
        cached = self._weight_norm_cache_value
        if self._weight_norm_cache_key == cache_key and torch.is_tensor(cached):
            return cached

        # The DoRA paper treats this norm as a detached constant. Caching it is exact
        # until base/LoRA weights change, which is tracked by the tensor _version fields.
        with torch.no_grad():
            weight_norm = _compute_dora_linear_weight_norm(
                weight=weight,
                lora_A_weight=lora_A_weight,
                lora_B_weight=lora_B_weight,
                scaling=float(scaling),
                fan_in_fan_out=bool(self.fan_in_fan_out),
                use_chunked_ops=bool(use_chunked_ops),
                compute_dtype=compute_dtype,
                device=x.device,
            ).detach()
        self._weight_norm_cache_key = cache_key
        self._weight_norm_cache_value = weight_norm
        return weight_norm

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None, saved_x=None):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        weight = dequantize_module_weight(base_layer)
        weight_norm = self._get_weight_norm_cached(
            x=x,
            weight=weight,
            lora_A_weight=lora_A.weight,
            lora_B_weight=lora_B.weight,
            scaling=float(scaling),
        )
        return _DoraLinearFn.apply(
            x,
            saved_x,
            base_result,
            weight,
            lora_A.weight,
            lora_B.weight,
            self.weight,
            weight_norm,
            float(scaling),
            bool(self.fan_in_fan_out),
        )

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep


class DoraConv2dLayer(DoraLinearLayer):
    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + scaling * lora_weight
        # the following is needed to have compatibility with the 4D weight tensors of Conv2D
        weight_norm = weight.norm(p=2, dim=(1, 2, 3), keepdim=True).transpose(1, 0)
        return weight_norm

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer):
        """
        For DoRA, calculate the extra output from LoRA with DoRA applied. This should be added on top of the base layer
        output.
        """
        weight = base_layer.weight
        lora_weight = torch.mm(lora_B.weight.flatten(start_dim=1), lora_A.weight.flatten(start_dim=1))
        lora_weight = lora_weight.reshape(weight.shape)
        magnitude = self.weight
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        weight_norm = weight_norm.detach()
        mag_norm_scale = magnitude / weight_norm
        result_dora = (mag_norm_scale - 1) * (
            F.conv2d(
                x,
                weight,
                bias=None,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
            )
        ) + mag_norm_scale * lora_B(lora_A(x)) * scaling

        return result_dora

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.dora." + rep
