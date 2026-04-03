# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Profiling utilities for DINOv3 performance optimization.
# All profiling behavior is opt-in via --profiling flag.

import logging
import os
import socket

import torch
import torch.cuda.nvtx as nvtx

logger = logging.getLogger("dinov3")


# ---------------------------------------------------------------------------
# NVTX context manager — zero-cost when profiling is off
# ---------------------------------------------------------------------------
class NVTXRange:
    """Thin NVTX wrapper that is a no-op when disabled."""

    __slots__ = ("msg", "enabled")

    def __init__(self, msg: str, enabled: bool = False):
        self.msg = msg
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            nvtx.range_push(self.msg)
        return self

    def __exit__(self, *args):
        if self.enabled:
            nvtx.range_pop()


def make_nvtx(enabled: bool):
    """Return a factory that creates NVTX ranges (no-ops when disabled)."""

    def _nvtx(msg: str) -> NVTXRange:
        return NVTXRange(msg, enabled=enabled)

    return _nvtx


# ---------------------------------------------------------------------------
# PyTorch Profiler setup
# ---------------------------------------------------------------------------
def build_profiler(
    output_dir: str,
    warmup: int = 5,
    active: int = 3,
    repeat: int = 1,
    rank: int = 0,
):
    """Build a PyTorch profiler with a warmup/active/repeat schedule.

    The profiler skips `warmup` steps, records `active` steps, and repeats
    that cycle `repeat` times.  Traces are exported as Chrome JSON to
    ``output_dir/profiler_traces/``.
    """
    trace_dir = os.path.join(output_dir, "profiler_traces")
    os.makedirs(trace_dir, exist_ok=True)

    def trace_handler(prof):
        trace_path = os.path.join(
            trace_dir,
            f"rank{rank}_step{prof.step_num}.json.gz",
        )
        prof.export_chrome_trace(trace_path)
        logger.info("Profiler trace exported: %s", trace_path)
        # Also dump a key-averages summary to a text file
        summary_path = os.path.join(
            trace_dir,
            f"rank{rank}_step{prof.step_num}_summary.txt",
        )
        summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
        with open(summary_path, "w") as f:
            f.write(summary)
        logger.info("Profiler summary exported: %s", summary_path)

    schedule = torch.profiler.schedule(
        wait=0,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # stack traces are expensive; enable manually if needed
        with_flops=True,
    )

    logger.info(
        "PyTorch profiler ready: warmup=%d, active=%d, repeat=%d, trace_dir=%s",
        warmup,
        active,
        repeat,
        trace_dir,
    )
    return profiler


# ---------------------------------------------------------------------------
# Graph-break diagnostics for torch.compile
# ---------------------------------------------------------------------------
def enable_graph_break_logging():
    """Enable verbose graph-break logging from torch._dynamo."""
    import torch._dynamo

    torch._dynamo.config.verbose = True
    if hasattr(torch._dynamo.config, "log_level"):
        torch._dynamo.config.log_level = logging.DEBUG
    logger.info("torch._dynamo graph-break verbose logging enabled")


# ---------------------------------------------------------------------------
# Extended metrics helpers
# ---------------------------------------------------------------------------
def get_memory_stats(device=None) -> dict:
    """Return current and peak reserved/allocated memory in MB."""
    if device is None:
        device = torch.cuda.current_device()
    return {
        "allocated_mem_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "reserved_mem_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
        "max_allocated_mem_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "max_reserved_mem_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
    }


def get_run_metadata(cfg) -> dict:
    """Return static run metadata for logging."""
    return {
        "node_name": socket.gethostname(),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "compile_enabled": bool(getattr(cfg.train, "compile", False)),
        "cudagraphs_enabled": bool(getattr(cfg.train, "cudagraphs", False)),
        "checkpointing_enabled": bool(getattr(cfg.train, "checkpointing", False)),
        "checkpointing_full_enabled": bool(getattr(cfg.train, "checkpointing_full", False)),
    }
