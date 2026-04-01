"""MFU tracking utilities for DINOv3 / iBOT SSL training on H100 GPUs.

Implements the FLOP formulas derived in docs/dinov3-mfu-tracking-initial-brief-03-27-26.md.
Bidirectional attention (no causal mask saving), backward ≈ 2× forward.
"""

# H100 SXM5 BF16 peak TFLOPS — dense (no structured sparsity).
# NVIDIA's published spec-sheet number (1979 TFLOPS) uses 2:4 structured sparsity,
# which roughly doubles theoretical throughput. Almost all transformer and ViT training
# uses standard dense matmuls, so 989 TFLOPS is the correct denominator for MFU.
# Reference: NVIDIA H100 datasheet footnote; see also "ML Engineering" (Bekman et al.)
# which explicitly corrects for this. Dense = published / 2.
H100_BF16_TFLOPS = 989.0

# A100 SXM4 BF16 peak TFLOPS — dense (no structured sparsity).
# NVIDIA publishes 312 TFLOPS for A100 BF16, which is also the sparsity number; dense = 156.
A100_BF16_TFLOPS = 156.0


def vit_forward_flops(
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    ffn_ratio: float = 4.0,
) -> int:
    """MACs for one forward pass of a ViT on one image (bidirectional attention).

    Uses MAC convention: 1 MAC = 1 multiply-add (fvcore / DINOv2 paper convention).
    The DINOv2 paper reports ~17.4 GFLOPs for ViT-B/16 global crop using this convention.
    Hardware TFLOPS specs count each MAC as 2 FLOPs (1 multiply + 1 add); see
    compute_mfu() for the 2× conversion factor. Use H100_BF16_TFLOPS (989, dense)
    as the denominator — not NVIDIA's published 1979 which assumes 2:4 sparsity.

    Per-layer breakdown (all in MACs):
      QKV projections: 3 × seq_len × D²   (three [D→D] projections)
      O projection:    1 × seq_len × D²
      Attn QKᵀ + AV:  2 × seq_len² × D   (bidirectional, full square)
      FFN up + down:   2 × seq_len × D × ffn_dim
    """
    ffn_dim = int(hidden_dim * ffn_ratio)
    # Linear: Q, K, V, O projections each [seq_len, D] × [D, D] → seq_len × D² MACs each.
    #   QKV = 3 × seq_len × D², O = 1 × seq_len × D²  →  4 × seq_len × D²
    # Attention: QKᵀ and AV each need seq_len × seq_len × D MACs (per-head × num_heads).
    #   Combined: 2 × seq_len² × D
    # FFN: up [D→ffn_dim] + down [ffn_dim→D] each seq_len × D × ffn_dim MACs → 2× total.
    attn_linear = 4 * seq_len * hidden_dim * hidden_dim   # QKV + O projections
    attn_scores = 2 * seq_len * seq_len * hidden_dim       # QKᵀ + AV (bidirectional, full sq)
    ffn = 2 * seq_len * hidden_dim * ffn_dim               # FFN up + down
    return num_layers * (attn_linear + attn_scores + ffn)


def compute_dino_flops_per_image(
    global_crop_size: int = 224,
    local_crop_size: int = 96,
    patch_size: int = 16,
    n_global_crops: int = 2,
    n_local_crops: int = 8,
    hidden_dim: int = 768,
    num_layers: int = 12,
    ffn_ratio: float = 4.0,
    n_registers: int = 0,
    gram_enabled: bool = False,
    head_overhead_pct: float = 0.05,
) -> int:
    """Total MACs for one full training step per original image in the batch.

    Uses MAC convention (1 MAC = 1 multiply-add). To convert to hardware FLOPs
    for MFU computation, multiply by 2 — see compute_mfu().

    Accounts for:
      - Student forward (global + local crops) + backward (~2× forward)
      - Teacher forward (global crops only, no grad)
      - Optional gram teacher forward (global crops only, no grad)
      - Head overhead (~5% of backbone by default)

    Args:
        global_crop_size: Global crop image size in pixels (default 224).
        local_crop_size: Local crop image size in pixels (default 96).
        patch_size: ViT patch size (default 16).
        n_global_crops: Number of global crops per image (default 2).
        n_local_crops: Number of local crops per image (default 8).
        hidden_dim: ViT hidden dimension (default 768 for ViT-B).
        num_layers: Number of transformer layers (default 12 for ViT-B).
        ffn_ratio: FFN hidden dim ratio (default 4.0 for standard MLP, not SwiGLU).
        n_registers: Number of register tokens (default 0; satellite fork drops them).
        gram_enabled: Whether gram teacher is enabled (default False).
        head_overhead_pct: Fraction added for DINO/iBOT heads (default 0.05 = 5%).

    Returns:
        Total MACs per image as an integer.
    """
    global_seq = (global_crop_size // patch_size) ** 2 + 1 + n_registers
    local_seq = (local_crop_size // patch_size) ** 2 + 1 + n_registers

    global_fwd = vit_forward_flops(global_seq, hidden_dim, num_layers, ffn_ratio)
    local_fwd = vit_forward_flops(local_seq, hidden_dim, num_layers, ffn_ratio)

    student_fwd = n_global_crops * global_fwd + n_local_crops * local_fwd
    student_bwd = 2 * student_fwd          # backward ≈ 2× forward
    teacher_fwd = n_global_crops * global_fwd
    gram_fwd = n_global_crops * global_fwd if gram_enabled else 0

    backbone_flops = student_fwd + student_bwd + teacher_fwd + gram_fwd
    return int(backbone_flops * (1.0 + head_overhead_pct))


def compute_mfu(
    images_per_sec: float,
    macs_per_image: int,
    num_gpus: int,
    peak_tflops: float = H100_BF16_TFLOPS,
) -> float:
    """Compute Model FLOP Utilization (MFU) as a fraction 0.0–1.0.

    MFU = actual_hardware_flops_per_sec / peak_hardware_flops_per_sec

    The 2× factor converts MACs (from compute_dino_flops_per_image) to hardware FLOPs:
    hardware vendors count each multiply-add as 2 FLOPs (1 multiply + 1 add).
    H100_BF16_TFLOPS = 989.0 is the dense (no 2:4 sparsity) peak — NVIDIA's published
    1979 TFLOPS assumes structured sparsity that standard dense matmuls do not use.

    Args:
        images_per_sec: Total images processed per second across all GPUs.
        macs_per_image: MACs per image per step (from compute_dino_flops_per_image).
        num_gpus: Number of GPUs in the run.
        peak_tflops: Peak TFLOPS per GPU (default H100 BF16 dense = 989.0).

    Returns:
        MFU as a fraction (multiply by 100 to get percentage).
    """
    # 2× converts MACs → hardware FLOPs (1 MAC = 2 hardware FLOPs: 1 multiply + 1 add)
    actual_tflops = (images_per_sec * 2 * macs_per_image) / 1e12
    theoretical_peak = num_gpus * peak_tflops
    return actual_tflops / theoretical_peak
