"""
Unit tests for MFU computation correctness.
Run with: python -m pytest tests/test_mfu.py -v
"""
from dinov3.utils.mfu import vit_forward_flops, compute_dino_flops_per_image, compute_mfu, H100_BF16_TFLOPS


class TestVitForwardFlops:
    def test_global_crop_matches_dinov2_paper(self):
        """DINOv2 paper reports ~17.5 GFLOPs for ViT-B/16 global crop (197 tokens)."""
        flops = vit_forward_flops(seq_len=197, hidden_dim=768, num_layers=12, ffn_ratio=4.0)
        assert 16e9 <= flops <= 19e9, f"Expected ~17.4 GFLOPs, got {flops/1e9:.2f}"

    def test_local_crop_smaller_than_global(self):
        """Local crop (37 tokens) should be much cheaper than global (197 tokens)."""
        g = vit_forward_flops(197, 768, 12, 4.0)
        l = vit_forward_flops(37, 768, 12, 4.0)
        assert l < g / 3, "Local crop should be < 1/3 of global (37² << 197²)"

    def test_scales_quadratically_with_seq_len_dominated_by_attn(self):
        """Doubling seq_len gives > 2× FLOPs because of the quadratic attention term.

        For ViT-B (D=768, ffn_ratio=4), the FFN and linear projections dominate the
        attention scores at typical seq_len (100-200 tokens), so the ratio is just above 2.
        A ratio strictly > 2.0 (and <= 4.0) confirms the quadratic component is present.
        """
        f1 = vit_forward_flops(100, 768, 12, 4.0)
        f2 = vit_forward_flops(200, 768, 12, 4.0)
        # Linear terms (QKV+O, FFN) go 2×; attn scores go 4×; combined is between (2, 4).
        # For D=768, FFN dominates over attn at seq_len=100-200, so ratio ≈ 2.04.
        assert 2.0 < f2 / f1 <= 4.0, f"Ratio was {f2/f1:.2f}"

    def test_scales_linearly_with_num_layers(self):
        f6 = vit_forward_flops(197, 768, 6, 4.0)
        f12 = vit_forward_flops(197, 768, 12, 4.0)
        assert abs(f12 / f6 - 2.0) < 0.01, "FLOPs must scale exactly 2× with layers"

    def test_no_registers_vs_registers(self):
        """n_storage_tokens=0 in this satellite fork — confirms our 197-token baseline."""
        f_no_reg = vit_forward_flops(197, 768, 12, 4.0)   # 196 patches + 1 CLS
        f_4reg   = vit_forward_flops(201, 768, 12, 4.0)   # 196 + 1 + 4 registers
        assert f_4reg > f_no_reg, "More tokens = more FLOPs"


class TestComputeDinoFlopsPerImage:
    def test_default_config_range(self):
        """Default config (2 global + 8 local, gram disabled) should be ~221 GFLOPs."""
        flops = compute_dino_flops_per_image(
            global_crop_size=224, local_crop_size=96, patch_size=16,
            n_global_crops=2, n_local_crops=8,
            hidden_dim=768, num_layers=12, ffn_ratio=4.0,
            n_registers=0, gram_enabled=False, head_overhead_pct=0.05,
        )
        assert 200e9 <= flops <= 245e9, f"Expected ~221 GFLOPs, got {flops/1e9:.1f}"

    def test_gram_adds_flops(self):
        """Enabling gram teacher adds one more teacher-forward (2 global crops)."""
        base = compute_dino_flops_per_image(gram_enabled=False)
        with_gram = compute_dino_flops_per_image(gram_enabled=True)
        global_fwd = vit_forward_flops(197, 768, 12, 4.0)
        gram_added = with_gram - base
        # Gram adds 2 global forwards + head overhead
        expected_added = 2 * global_fwd
        assert abs(gram_added - expected_added * 1.05) / expected_added < 0.10, \
            f"Gram overhead was {gram_added/1e9:.1f} GFLOPs, expected ~{expected_added*1.05/1e9:.1f}"

    def test_no_local_crops_is_less_than_default(self):
        """Zero local crops should give fewer FLOPs than 8 local crops."""
        no_local = compute_dino_flops_per_image(n_local_crops=0)
        with_local = compute_dino_flops_per_image(n_local_crops=8)
        assert no_local < with_local

    def test_backward_is_2x_student_forward(self):
        """
        Verify the backward=2×forward assumption by decomposing the formula.
        With no local crops, 1 global, no gram, no overhead:
          student_fwd = 1 × global_fwd
          student_bwd = 2 × student_fwd
          teacher_fwd = 1 × global_fwd
          total = student_fwd + student_bwd + teacher_fwd = 4 × global_fwd
        """
        flops = compute_dino_flops_per_image(
            n_global_crops=1, n_local_crops=0, gram_enabled=False, head_overhead_pct=0.0
        )
        global_fwd = vit_forward_flops(197, 768, 12, 4.0)
        expected = 4 * global_fwd  # student_fwd + 2*student_fwd + teacher_fwd
        assert abs(flops - expected) / expected < 0.01, \
            f"Expected {expected/1e9:.2f} GFLOPs, got {flops/1e9:.2f}"


class TestComputeMfu:
    def test_mfu_is_fraction_between_0_and_1(self):
        flops = compute_dino_flops_per_image()
        mfu = compute_mfu(images_per_sec=512, flops_per_image=flops, num_gpus=8)
        assert 0.0 < mfu < 1.0, f"MFU={mfu:.4f} should be in (0, 1)"

    def test_mfu_scales_linearly_with_throughput(self):
        flops = compute_dino_flops_per_image()
        mfu_1x = compute_mfu(512, flops, 8)
        mfu_2x = compute_mfu(1024, flops, 8)
        assert abs(mfu_2x / mfu_1x - 2.0) < 0.01

    def test_mfu_scales_inversely_with_more_gpus(self):
        flops = compute_dino_flops_per_image()
        mfu_8  = compute_mfu(512, flops, 8)
        mfu_16 = compute_mfu(512, flops, 16)
        assert abs(mfu_8 / mfu_16 - 2.0) < 0.01

    def test_floor_estimate_at_expected_baseline(self):
        """
        At 512 img/s (8 GPUs × 64 img/GPU, ~1 s/step), MFU ≈ 0.7%.

        NOTE: The plan document cited ~7% here, but that is a calculation error.
        With H100_BF16_TFLOPS=1979 and flops_per_image≈226 GFLOPs:
          MFU = 512 × 226e9 / (8 × 1979e12) ≈ 0.73%
        The ~7% figure would require either H100=197.9 TFLOPS or ~5000 img/s throughput.
        1-second steps at bs=64/GPU give low absolute MFU because H100 peak is enormous;
        the real baseline (with compile, ~100ms steps) would be ~7% at ~5000 img/s.
        """
        flops = compute_dino_flops_per_image()
        mfu = compute_mfu(512, flops, 8, H100_BF16_TFLOPS)
        # Correct range for 512 img/s, 8× H100s, ~226 GFLOPs/image: ~0.73%
        assert 0.003 <= mfu <= 0.02, \
            f"MFU at 512 img/s should be ~0.7%, got {mfu*100:.2f}%"

    def test_perfect_mfu_is_1(self):
        """If images_per_sec = (num_gpus × peak_tflops × 1e12) / flops_per_image, MFU=1."""
        flops = 100e9
        peak = 1000.0  # 1000 TFLOPS per GPU
        gpus = 2
        perfect_ips = (gpus * peak * 1e12) / flops
        mfu = compute_mfu(perfect_ips, int(flops), gpus, peak_tflops=peak)
        assert abs(mfu - 1.0) < 1e-6
