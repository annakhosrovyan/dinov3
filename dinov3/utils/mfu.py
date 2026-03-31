"""MFU tracking utilities for DINOv3 / iBOT SSL training on H100 GPUs.

Implements the FLOP formulas derived in docs/dinov3-mfu-tracking-initial-brief-03-27-26.md.
Bidirectional attention (no causal mask saving), backward ≈ 2× forward.
"""

H100_BF16_TFLOPS = 1979.0  # H100 SXM5 BF16 peak TFLOPS
A100_BF16_TFLOPS = 312.0   # A100 SXM4 BF16 peak TFLOPS (reference)


def vit_forward_flops(
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    ffn_ratio: float = 4.0,
) -> int:
    """FLOPs for one forward pass of a ViT on one image (bidirectional attention).

    Each multiply-add counts as 2 FLOPs.

    Per-layer breakdown:
      QKV projection:  seq_len × D × 3D × 2 = 6 × seq_len × D²
      O projection:    seq_len × D × D × 2   = 2 × seq_len × D²
      Attn scores:     seq_len² × D × 2      (QKᵀ, full square, bidirectional)
      Attn weighted:   seq_len² × D × 2      (softmax(QKᵀ)V)
      FFN up+down:     seq_len × D × ffn_dim × 2 × 2 = 4 × seq_len × D × ffn_dim
    """
    ffn_dim = int(hidden_dim * ffn_ratio)
    # Convention: 1 FLOP = 1 multiply-add (MAC), matching fvcore / DINOv2 paper ~17.4 GFLOPs.
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
    """Total FLOPs for one full training step per original image in the batch.

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
        Total FLOPs per image as an integer.
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
    flops_per_image: int,
    num_gpus: int,
    peak_tflops: float = H100_BF16_TFLOPS,
) -> float:
    """Compute Model FLOP Utilization (MFU) as a fraction 0.0–1.0.

    Args:
        images_per_sec: Total images processed per second across all GPUs.
        flops_per_image: FLOPs per image per step (from compute_dino_flops_per_image).
        num_gpus: Number of GPUs in the run.
        peak_tflops: Peak TFLOPS per GPU (default H100 BF16 = 1979.0).

    Returns:
        MFU as a fraction (multiply by 100 to get percentage).
    """
    actual_tflops = (images_per_sec * flops_per_image) / 1e12
    theoretical_peak = num_gpus * peak_tflops
    return actual_tflops / theoretical_peak
