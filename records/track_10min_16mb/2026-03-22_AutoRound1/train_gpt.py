"""
parameter golf submission: int6 STE + NorMuon + NTK rope extrapolation + SWA + sliding window + SmearGate + Muon WD + Seq2048 + OrthoInit + RoPE50k + 11 Layers + AdamW WD=0.04 + Partial RoPE + LN Scale + FA3 + BigramHash + Value Residual Learning + Shared VE128

- 11 layers
- int6 fake quantization w/ STE during training
- fp16 passthrough on embedding (no quant penalty on most sensitive tensors)
- 3x wider MLP (1536 hidden) enabled by int6 compression savings
- NorMuon optimizer (row-normalized Newton-Schulz updates) with decoupled weight decay = 0.04
- AdamW optimizer for scalar/embedding parameters with weight decay = 0.04
- NTK-aware RoPE extrapolation
- sliding window eval w/ stride=64
- stochastic weight averaging during warmdown (every 50 steps)
- optimizer tuning: momentum 0.99, higher LRs for 11L, longer warmdown
- SmearGate: blends token embeddings with their predecessors using a learned per-dim gate
- Sequence Length 2048: longer context per step
- Orthogonal Initialization: orthogonal init for non-zero-init Linear layers
- RoPE Base 50000: better position allocation for longer sequences
- Partial RoPE: Apply RoPE to only 16 dims per head
- LN Scale: scale RMSNorm outputs by 1/sqrt(layer_idx+1)
- FlashAttention 3 (cudnn SDP)
- BigramHash: 2048 buckets for token-pair context
- Value Residual Learning: mix layer 0's V vectors into all subsequent layers
- Shared VE128: shared 128-dim Value Embedding bypassed directly to layers 9 and 10, replacing c_v
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 50))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 50000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    grad_accum_override = int(os.environ.get("GRAD_ACCUM", 0))

    # eval-time RoPE extrapolation
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_rope_base = float(os.environ.get("EVAL_ROPE_BASE", 0))  # 0 = auto-compute NTK base

    # stochastic weight averaging: average checkpoints during warmdown
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.2))  # tight SWA: last 20% of warmdown only (PR #374)
    swa_every = int(os.environ.get("SWA_EVERY", 25))  # collect checkpoint every N steps


# -----------------------------
# NORMUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def normuon_update(
    grad: Tensor,
    momentum: Tensor,
    second_momentum: Tensor,
    beta: float = 0.95,
    beta2: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update.to(grad.dtype)
    if original_shape is not None:
        update = update.reshape(original_shape)
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10)))
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 beta2: float = 0.95, backend_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, backend_steps=backend_steps, weight_decay=weight_decay)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            pad_count = (-len(params)) % world_size
            params_pad = params + [torch.empty_like(params[-1]) for _ in range(pad_count)]
            for base_i in range(0, len(params_pad), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                    update = normuon_update(
                        p.grad,
                        state["momentum_buffer"],
                        state["second_momentum_buffer"],
                        beta=group["momentum"],
                        beta2=group["beta2"],
                        ns_steps=group["backend_steps"],
                    )
                    if group["weight_decay"] > 0.0:
                        p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                if distributed:
                    dist.all_gather(params_pad[base_i : base_i + world_size], params_pad[base_i + rank])
        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        # U+2581 LOWER ONE EIGHTH BLOCK = SentencePiece word boundary marker
        # NOTE: use chr(0x2581) to avoid LLM Unicode corruption during code generation
        _SP_SPACE = chr(0x2581)
        if piece.startswith(_SP_SPACE):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def swap_rope_base(model: nn.Module, new_base: float) -> float:
    """swap RoPE base frequency in all attention layers. returns the old base."""
    old_base = None
    for module in model.modules():
        if isinstance(module, Rotary):
            if old_base is None:
                old_base = float(module.inv_freq[0].item())
                # reconstruct: inv_freq[0] = 1.0 / (base ** (0 / dim))
                # but inv_freq[0] is always 1.0, so we need to look at inv_freq[1]
            dim = module.inv_freq.numel() * 2
            new_inv_freq = 1.0 / (new_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            module.inv_freq = new_inv_freq.to(module.inv_freq.device)
            # invalidate cache
            module._seq_len_cached = 0
            module._cos_cached = None
            module._sin_cached = None
    return old_base if old_base is not None else 10000.0


def compute_ntk_rope_base(train_seq_len: int, eval_seq_len: int, head_dim: int, base: float = 10000.0) -> float:
    """NTK-aware RoPE base scaling for length extrapolation."""
    if eval_seq_len <= train_seq_len:
        return base
    ratio = eval_seq_len / train_seq_len
    # NTK scaling: base * ratio^(dim / (dim - 2))
    return base * (ratio ** (head_dim / (head_dim - 2)))


def eval_val_sliding_window(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int = 64,
    eval_seq_len: int = 0,
    eval_rope_base: float = 0,
) -> tuple[float, float]:
    """sliding window eval with optional NTK RoPE extrapolation."""
    seq_len = eval_seq_len if eval_seq_len > 0 else args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    num_windows = max((total_tokens - seq_len) // stride + 1, 1)
    win_start = (num_windows * rank) // world_size
    win_end = (num_windows * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # scale down batch size for longer eval sequences to avoid OOM
    default_batch = max(1, 32 * args.train_seq_len // seq_len)
    eval_batch = int(os.environ.get("SW_EVAL_BATCH", default_batch))

    # swap rope base for NTK extrapolation if requested
    old_rope_base = None
    if eval_rope_base > 0:
        old_rope_base = swap_rope_base(base_model, eval_rope_base)

    base_model.eval()
    with torch.inference_mode():
        window_list = list(range(win_start, win_end))
        num_batches = (len(window_list) + eval_batch - 1) // eval_batch
        for batch_idx in range(num_batches):
            batch_wins = window_list[batch_idx * eval_batch : (batch_idx + 1) * eval_batch]

            inputs = torch.stack([
                val_tokens[w * stride : w * stride + seq_len]
                for w in batch_wins
            ]).to(device=device, dtype=torch.int64)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(inputs)

            scored_logits = logits[:, -stride:, :].reshape(-1, logits.size(-1))
            targets = torch.stack([
                val_tokens[w * stride + seq_len - stride + 1 : w * stride + seq_len + 1]
                for w in batch_wins
            ]).to(device=device, dtype=torch.int64).reshape(-1)

            loss = F.cross_entropy(scored_logits.float(), targets, reduction="sum")
            val_loss_sum += loss.to(torch.float64)
            val_token_count += float(targets.numel())

            prev_ids = torch.stack([
                val_tokens[w * stride + seq_len - stride : w * stride + seq_len]
                for w in batch_wins
            ]).to(device=device, dtype=torch.int64).reshape(-1)

            tgt_ids = targets
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    # restore original rope base
    if old_rope_base is not None:
        swap_rope_base(base_model, old_rope_base)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    base_model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16

INT6_QUANT_RANGE = 31
INT6_CLIP_Q = 0.9999984

# embedding gets fp16 passthrough - most quant-sensitive, no STE
FP16_PASSTHROUGH_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("FP16_PASSTHROUGH_PATTERNS", "tok_emb").split(",")
    if pattern
)


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor, quant_range: int = INT6_QUANT_RANGE) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT6_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(quant_range)).clamp_min(1.0 / float(quant_range))
        q = torch.clamp(torch.round(clipped / scale[:, None]), -quant_range, quant_range).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT6_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # fp16 passthrough for embedding
        if any(pattern in name for pattern in FP16_PASSTHROUGH_PATTERNS):
            passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            kept = t.to(dtype=torch.float16).contiguous()
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t, quant_range=INT6_QUANT_RANGE)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int6_fp16embed_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if self.training and w.ndim == 2:
            # fake int6 per-row quantization with STE
            with torch.no_grad():
                w32 = w.float()
                clip_abs = torch.quantile(w32.abs(), INT6_CLIP_Q, dim=1).clamp_min(1e-8)
                scale = clip_abs / INT6_QUANT_RANGE
                w_clipped = torch.clamp(w32, -clip_abs[:, None], clip_abs[:, None])
                w_q = (torch.round(w_clipped / scale[:, None]) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()  # STE
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float, use_shared_v: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        
        self.use_shared_v = use_shared_v
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        if not use_shared_v:
            self.c_v = CastedLinear(dim, kv_dim, bias=False)
        else:
            self.shared_v_scale = nn.Parameter(torch.ones(kv_dim, dtype=torch.float32))
            
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        
        # Partial RoPE: Apply RoPE to 16 dims
        self.rotary_dim = 16
        self.rotary = Rotary(self.rotary_dim, base=rope_base)

    def forward(self, x: Tensor, v_first: Tensor | None = None, shared_v: Tensor | None = None) -> tuple[Tensor, Tensor]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.use_shared_v:
            assert shared_v is not None
            scale = self.shared_v_scale.to(dtype=shared_v.dtype).view(self.num_kv_heads, 1, self.head_dim)
            v = shared_v * scale[None, ...]
        else:
            v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        
        v_out = v
        # Value Residual Learning: mix first layer's v
        if v_first is not None and not self.use_shared_v:
            v = 0.5 * v + 0.5 * v_first
            
        q_rot, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
        k_rot, k_pass = k[..., :self.rotary_dim], k[..., self.rotary_dim:]
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q_rot = apply_rotary_emb(q_rot, cos, sin)
        k_rot = apply_rotary_emb(k_rot, cos, sin)
        q = torch.cat((q_rot, q_pass), dim=-1)
        k = torch.cat((k_rot, k_pass), dim=-1)
        
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        # manual GQA: repeat KV heads to match Q heads (torch <2.5 compat)
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k[:, :, None, :, :].expand(bsz, self.num_kv_heads, rep, seqlen, self.head_dim).reshape(bsz, self.num_heads, seqlen, self.head_dim)
            v_attn = v[:, :, None, :, :].expand(bsz, self.num_kv_heads, rep, seqlen, self.head_dim).reshape(bsz, self.num_heads, seqlen, self.head_dim)
        else:
            v_attn = v
        y = F.scaled_dot_product_attention(q, k, v_attn, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y), v_out


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float, layer_idx: int, num_layers: int = 11):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        
        use_shared_v = layer_idx >= (num_layers - 2)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_shared_v)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        # LN Scale
        self.layer_scale = 1.0 / math.sqrt(layer_idx + 1)

    def forward(self, x: Tensor, x0: Tensor, v_first: Tensor | None = None, shared_v: Tensor | None = None) -> tuple[Tensor, Tensor]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        # Pre-LN only (Mix-LN if/else removed -- breaks torch.compile, 160ms vs 62ms/step)
        attn_out, v_current = self.attn(self.attn_norm(x) * self.layer_scale, v_first, shared_v)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * self.layer_scale)
        return x, v_current


class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
        num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float,
        logit_softcap: float, rope_base: float, qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size
        
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        # BigramHash: reduced to fit under 16MB (512 buckets × 64 dim + projection)
        bigram_buckets = int(os.environ.get("BIGRAM_BUCKETS", "512"))
        bigram_dim = int(os.environ.get("BIGRAM_DIM", "64"))
        self.bigram_buckets = bigram_buckets
        self.bigram_table = nn.Embedding(bigram_buckets, bigram_dim)
        self.bigram_proj = CastedLinear(bigram_dim, model_dim, bias=False)
        self.bigram_proj._zero_init = True
        
        # Shared VE128 (bypasses c_v in layers 9 and 10)
        self.shared_ve = nn.Embedding(vocab_size, 128)
        self.shared_ve_proj = CastedLinear(128, num_kv_heads * (model_dim // num_heads), bias=False)
        
        self.smear_gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
        
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, i, num_layers)
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        nn.init.normal_(self.bigram_table.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.shared_ve.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _transformer_forward(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        
        # BigramHash (reduced: 512 buckets × 64 dim + projection to model_dim)
        bigram_ids = (input_ids[:, :-1].to(torch.int64) * self.vocab_size + input_ids[:, 1:].to(torch.int64)) % self.bigram_buckets
        bigram_emb = self.bigram_proj(self.bigram_table(bigram_ids))
        bigram_emb = F.pad(bigram_emb, (0, 0, 1, 0), "constant", 0)
        x = x + bigram_emb.to(dtype=x.dtype)
        
        # SmearGate
        gate = torch.sigmoid(self.smear_gate).to(dtype=x.dtype)
        x_shifted = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        x = gate[None, None, :] * x + (1 - gate)[None, None, :] * x_shifted
        
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        
        # Shared VE128 direct mapping
        shared_v_flat = self.shared_ve_proj(self.shared_ve(input_ids))
        B, T, _ = shared_v_flat.shape
        num_kv_heads = self.blocks[0].attn.num_kv_heads
        head_dim = self.blocks[0].attn.head_dim
        shared_v = shared_v_flat.reshape(B, T, num_kv_heads, head_dim).transpose(1, 2).to(dtype=x.dtype)
        
        v_first = None
        for i in range(self.num_encoder_layers):
            x, v_current = self.blocks[i](x, x0, v_first, shared_v)
            if i == 0:
                v_first = v_current.detach() if not self.training else v_current
            skips.append(x)
            
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, _ = self.blocks[self.num_encoder_layers + i](x, x0, v_first, shared_v)
            
        return self.final_norm(x)

    def _compute_logits(self, x: Tensor) -> Tensor:
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight.to(x.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._transformer_forward(input_ids)
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        targets = target_ids.reshape(-1)
        logits = self._compute_logits(x_flat)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self._transformer_forward(input_ids)
        return self._compute_logits(x)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    use_compile = bool(int(os.environ.get("USE_TORCH_COMPILE", "1")))
    if use_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")

    # flexible grad accumulation - default to 8//world_size, allow override
    if args.grad_accum_override > 0:
        grad_accum_steps = args.grad_accum_override
    else:
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8 (or set GRAD_ACCUM)")
        grad_accum_steps = 8 // world_size

    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(True)
    enable_flash_sdp(False)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if use_compile:
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    else:
        compiled_model = base_model  # eager mode: no compile overhead, ~2x slower per step
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if hasattr(base_model, "smear_gate"):
        scalar_params.append(base_model.smear_gate)
    # BigramHash: proj weight -> Muon (2D matrix), table weight -> AdamW
    if hasattr(base_model, "bigram_proj") and hasattr(base_model.bigram_proj, "weight"):
        matrix_params.append(base_model.bigram_proj.weight)
    # Shared VE128: proj weight -> Muon (2D matrix), table weight -> AdamW
    if hasattr(base_model, "shared_ve_proj") and hasattr(base_model.shared_ve_proj, "weight"):
        matrix_params.append(base_model.shared_ve_proj.weight)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight, base_model.bigram_table.weight, base_model.shared_ve.weight], "lr": token_lr, "base_lr": token_lr, "weight_decay": 0.04}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = NorMuon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        beta2=args.muon_beta2, backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr, "weight_decay": 0.04}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr, "weight_decay": 0.04}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"mlp_mult:{args.mlp_mult} mlp_hidden:{args.mlp_mult * args.model_dim}")
    log0(f"optimizer:NorMuon momentum:{args.muon_momentum} beta2:{args.muon_beta2} weight_decay:{args.muon_weight_decay}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # SWA state
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    swa_collecting = False

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # SWA: start collecting checkpoints during warmdown
        if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                if master_process:
                    log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1

        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # apply SWA if we collected checkpoints
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"swa:applying averaged {swa_count} checkpoints")
        avg_state = {name: (t / swa_count).to(dtype=base_model.state_dict()[name].dtype)
                     for name, t in swa_state.items()}
        base_model.load_state_dict(avg_state, strict=True)

    # Remove MTP head before serialization (training-only, zero artifact cost)
    if hasattr(base_model, 'mtp_head'):
        del base_model.mtp_head

    # serialization + roundtrip
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        quant_blob = cctx.compress(quant_raw)
        compress_name = "zstd-22"
    else:
        quant_blob = zlib.compress(quant_raw, level=9)
        compress_name = "zlib-9"
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int6+{compress_name}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int6+{compress_name}: {quant_file_bytes + code_bytes} bytes")
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.ptz", "rb") as f:
        quant_blob_disk = f.read()
    if HAS_ZSTD:
        dctx = zstd.ZstdDecompressor()
        quant_decompressed = dctx.decompress(quant_blob_disk)
    else:
        quant_decompressed = zlib.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(quant_decompressed), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)


    # === TTT: Test-Time Training (PR #338 approach) ===
    # 3-epoch SGD on val data, freeze first 2 blocks, ~47s on 8xH100
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", "3"))
    ttt_lr = float(os.environ.get("TTT_LR", "0.002"))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", "2"))
    ttt_max_steps = int(os.environ.get("TTT_MAX_STEPS", "0"))  # 0 = unlimited

    if ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()

        # Freeze first N blocks
        for i in range(min(ttt_freeze_blocks, len(base_model.blocks))):
            for p in base_model.blocks[i].parameters():
                p.requires_grad_(False)

        ttt_opt = torch.optim.SGD(
            [p for p in base_model.parameters() if p.requires_grad],
            lr=ttt_lr, momentum=0.9,
        )

        # Use DDP-aware training: each rank processes its shard of val data
        ttt_seq = args.train_seq_len
        total_val = val_tokens.numel() - 1
        # Each rank takes every world_size-th chunk
        rank_starts = list(range(rank * ttt_seq, total_val - ttt_seq, world_size * ttt_seq))

        base_model.train()
        ttt_steps = 0
        for ttt_epoch in range(ttt_epochs):
            for ttt_start in rank_starts:
                ttt_x = val_tokens[ttt_start:ttt_start + ttt_seq].unsqueeze(0).to(device=device, dtype=torch.int64)
                ttt_y = val_tokens[ttt_start + 1:ttt_start + ttt_seq + 1].unsqueeze(0).to(device=device, dtype=torch.int64)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ttt_loss = base_model(ttt_x, ttt_y)
                ttt_loss.backward()
                ttt_opt.step()
                ttt_opt.zero_grad()
                ttt_steps += 1
                if ttt_max_steps > 0 and ttt_steps >= ttt_max_steps:
                    break
            if ttt_max_steps > 0 and ttt_steps >= ttt_max_steps:
                break

        base_model.eval()
        # Unfreeze all
        for p in base_model.parameters():
            p.requires_grad_(True)

        # Sync params across ranks after TTT
        if distributed:
            for p in base_model.parameters():
                dist.all_reduce(p.data, op=dist.ReduceOp.AVG)

        torch.cuda.synchronize()
        log0(f"ttt: {ttt_steps} steps x {ttt_epochs} epochs in {1000.0 * (time.perf_counter() - t_ttt):.0f}ms")

    # standard eval on quantized model
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_bpb:{q_val_bpb:.8f}")

    # sliding window eval
    if args.eval_stride > 0:
        torch.cuda.synchronize()
        t_sw = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding_window(
            args, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
        )
        torch.cuda.synchronize()
        log0(
            f"final_sliding_window stride:{args.eval_stride} "
            f"val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_sw):.0f}ms"
        )
        log0(
            f"final_sliding_window_exact stride:{args.eval_stride} "
            f"val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}"
        )

    # NTK RoPE extrapolation: eval at longer sequence length
    if args.eval_stride > 0 and args.eval_seq_len > args.train_seq_len:
        head_dim = args.model_dim // args.num_heads
        ntk_base = args.eval_rope_base if args.eval_rope_base > 0 else compute_ntk_rope_base(
            args.train_seq_len, args.eval_seq_len, head_dim, args.rope_base
        )
        log0(f"ntk_eval: eval_seq_len={args.eval_seq_len} ntk_rope_base={ntk_base:.1f}")
        torch.cuda.synchronize()
        t_ntk = time.perf_counter()
        ntk_val_loss, ntk_val_bpb = eval_val_sliding_window(
            args, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=args.eval_seq_len,
            eval_rope_base=ntk_base,
        )
        torch.cuda.synchronize()
        log0(
            f"final_ntk_sliding_window seq_len:{args.eval_seq_len} stride:{args.eval_stride} "
            f"rope_base:{ntk_base:.1f} val_loss:{ntk_val_loss:.4f} val_bpb:{ntk_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_ntk):.0f}ms"
        )
        log0(
            f"final_ntk_sliding_window_exact seq_len:{args.eval_seq_len} stride:{args.eval_stride} "
            f"rope_base:{ntk_base:.1f} val_loss:{ntk_val_loss:.8f} val_bpb:{ntk_val_bpb:.8f}"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()