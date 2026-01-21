"""
Experiment run summary and analysis utilities.

Provides functions to analyze PyTorch profiler traces and experiment configurations
to generate performance summaries and metrics.
"""

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .scanner import find_profiler_traces, get_experiment_config_path

console = Console()


@dataclass
class ProfilerMetrics:
    """Metrics extracted from PyTorch profiler traces."""

    avg_step_time_ms: float
    gpu_utilization_pct: float
    cpu_time_pct: float
    gpu_time_pct: float
    memory_bandwidth_pct: float
    kernel_count: int
    total_kernels: int
    trace_duration_ms: float
    steps_profiled: int
    batch_aug_time_ms: float = 0.0
    mixing_time_ms: float = 0.0

    # Level 2 Profiling - Dataloader Components
    dataloader_io_wait_ms: float = 0.0
    dataloader_cpu_decode_ms: float = 0.0
    dataloader_preprocess_wait_ms: float = 0.0
    dataloader_cpu_xform_ms: float = 0.0

    # Level 2 Profiling - Model Components
    model_stem_ms: float = 0.0
    model_convnext_stage_0_ms: float = 0.0
    model_convnext_stage_1_ms: float = 0.0
    model_rope_stage_2_ms: float = 0.0
    model_rope_stage_3_ms: float = 0.0
    model_rope_stage_4_ms: float = 0.0
    model_downsample_0_ms: float = 0.0
    model_downsample_1_ms: float = 0.0
    model_downsample_2_ms: float = 0.0
    model_aggregation_ms: float = 0.0

    # Level 2 Profiling - Loss Components
    loss_core_loss_ms: float = 0.0
    loss_masking_ms: float = 0.0
    loss_weighting_ms: float = 0.0
    loss_aggregation_ms: float = 0.0

    # Level 2 Profiling - Backward/Optimizer/Comms Components
    backward_ddp_sync_ctx_ms: float = 0.0
    backward_scale_backward_ms: float = 0.0
    gradnorm_reforward_ms: float = 0.0
    optimizer_unscale_ms: float = 0.0
    optimizer_clip_grad_ms: float = 0.0
    optimizer_step_ms: float = 0.0
    optimizer_scaler_update_ms: float = 0.0
    optimizer_zero_grad_ms: float = 0.0
    optimizer_lr_step_ms: float = 0.0
    comms_allreduce_ms: float = 0.0

    # Level 2 Profiling - Augmentation Components (enhanced from Level 1)
    augmentation_selective_mixing_ms: float = 0.0
    augmentation_gpu_batch_augmentations_ms: float = 0.0

    # Level 3 Profiling - Host-side Dataloader Metrics
    median_batch_index_q_depth: float = 0.0
    median_preprocess_q_depth: float = 0.0
    median_processed_batch_q_depth: float = 0.0
    median_io_throughput_it_s: float = 0.0
    median_handoff_throughput_it_s: float = 0.0
    median_cache_hit_rate_pct: float = 0.0


@dataclass
class ExperimentConfig:
    """Key configuration settings from experiment."""

    batch_size: int
    accumulation_steps: int
    aug_pipeline_device: str
    gpu_compile_enabled: bool
    gpu_compile_mode: str
    profiler_enabled: bool
    model_type: str
    amp_level: str
    num_workers: int


@dataclass
class RunSummary:
    """Complete summary of an experiment run."""

    run_path: Path
    run_id: str
    config: ExperimentConfig
    profiler_metrics: ProfilerMetrics | None
    has_profiler_traces: bool
    trace_files_count: int
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_path": str(self.run_path),
            "run_id": self.run_id,
            "config": asdict(self.config),
            "profiler_metrics": asdict(self.profiler_metrics) if self.profiler_metrics else None,
            "has_profiler_traces": self.has_profiler_traces,
            "trace_files_count": self.trace_files_count,
            "error_message": self.error_message,
        }


def build_summary(run_path: Path, write_cache: bool = False) -> RunSummary:
    """
    Build a comprehensive summary of an experiment run.

    Args:
        run_path: Path to experiment run directory
        write_cache: Whether to cache the computed summary

    Returns:
        RunSummary object with all available metrics
    """
    run_path = Path(run_path).resolve()
    run_id = f"{run_path.parent.parent.name}/{run_path.parent.name}/{run_path.name}"

    # Check for cached summary
    cache_dir = run_path / ".linnaeus_cache"
    cache_file = cache_dir / "summary.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            # Ignore corrupted cache
            pass

    error_message = None
    profiler_metrics = None

    try:
        # Load experiment configuration
        config = load_experiment_config(run_path)

        # Find and analyze profiler traces
        trace_files = find_profiler_traces(run_path)
        has_traces = len(trace_files) > 0

        if has_traces:
            try:
                profiler_metrics = analyze_profiler_traces(trace_files)
            except Exception as e:
                error_message = f"Failed to analyze profiler traces: {str(e)}"

        summary = RunSummary(
            run_path=run_path,
            run_id=run_id,
            config=config,
            profiler_metrics=profiler_metrics,
            has_profiler_traces=has_traces,
            trace_files_count=len(trace_files),
            error_message=error_message,
        )

        # Cache if requested
        if write_cache:
            cache_dir.mkdir(exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(summary, f)

        return summary

    except Exception as e:
        return RunSummary(
            run_path=run_path,
            run_id=run_id,
            config=ExperimentConfig(
                batch_size=0,
                accumulation_steps=0,
                aug_pipeline_device="unknown",
                gpu_compile_enabled=False,
                gpu_compile_mode="unknown",
                profiler_enabled=False,
                model_type="unknown",
                amp_level="unknown",
                num_workers=0,
            ),
            profiler_metrics=None,
            has_profiler_traces=False,
            trace_files_count=0,
            error_message=f"Failed to build summary: {str(e)}",
        )


def load_experiment_config(run_path: Path) -> ExperimentConfig:
    """
    Load and parse experiment configuration.

    Args:
        run_path: Path to experiment run directory

    Returns:
        ExperimentConfig object
    """
    config_path = get_experiment_config_path(run_path)

    with open(config_path) as f:
        config_data = yaml.load(f, Loader=yaml.UnsafeLoader)

    return ExperimentConfig(
        batch_size=config_data.get("DATA", {}).get("BATCH_SIZE", 0),
        accumulation_steps=config_data.get("TRAIN", {}).get("ACCUMULATION_STEPS", 1),
        aug_pipeline_device=config_data.get("AUG", {}).get("PIPELINE_DEVICE", "cpu"),
        gpu_compile_enabled=config_data.get("AUG", {}).get("GPU_COMPILE", {}).get("ENABLED", False),
        gpu_compile_mode=config_data.get("AUG", {}).get("GPU_COMPILE", {}).get("MODE", "default"),
        profiler_enabled=config_data.get("DEBUG", {}).get("PROFILER", {}).get("ENABLED", False),
        model_type=config_data.get("MODEL", {}).get("TYPE", "unknown"),
        amp_level=config_data.get("TRAIN", {}).get("AMP_OPT_LEVEL", "O0"),
        num_workers=config_data.get("DATA", {}).get("NUM_WORKERS", 0),
    )


def analyze_profiler_traces(trace_files: list[Path]) -> ProfilerMetrics:
    """
    Analyze PyTorch profiler trace files to extract metrics.

    Args:
        trace_files: List of paths to .pt.trace.json files

    Returns:
        ProfilerMetrics object with aggregated statistics
    """
    if not trace_files:
        raise ValueError("No trace files provided")

    total_step_time = 0.0
    total_gpu_time = 0.0
    total_cpu_time = 0.0
    total_memory_time = 0.0
    total_kernel_count = 0
    total_duration = 0.0
    steps_profiled = 0
    total_batch_aug_time = 0.0
    total_mixing_time = 0.0

    # Level 2 profiling region totals
    total_dataloader_io_wait = 0.0
    total_dataloader_cpu_decode = 0.0
    total_dataloader_preprocess_wait = 0.0
    total_dataloader_cpu_xform = 0.0
    total_model_stem = 0.0
    total_model_convnext_stage_0 = 0.0
    total_model_convnext_stage_1 = 0.0
    total_model_rope_stage_2 = 0.0
    total_model_rope_stage_3 = 0.0
    total_model_rope_stage_4 = 0.0
    total_model_downsample_0 = 0.0
    total_model_downsample_1 = 0.0
    total_model_downsample_2 = 0.0
    total_model_aggregation = 0.0
    total_loss_core_loss = 0.0
    total_loss_masking = 0.0
    total_loss_weighting = 0.0
    total_loss_aggregation = 0.0
    total_augmentation_selective_mixing = 0.0
    total_augmentation_gpu_batch_augmentations = 0.0

    # NOTE: It's tempting to use locals() to accumulate into the total_* variables,
    # but mutating locals() does not reliably update local variables in Python.
    # Use an explicit mapping instead.
    l2_region_totals: dict[str, float] = {
        # Dataloader components
        "dataloader/io_wait": 0.0,
        "dataloader/cpu_decode": 0.0,
        "dataloader/preprocess_wait": 0.0,
        "dataloader/cpu_xform": 0.0,
        # Model components
        "model/stem": 0.0,
        "model/convnext_stage_0": 0.0,
        "model/downsample_0": 0.0,
        "model/convnext_stage_1": 0.0,
        "model/downsample_1": 0.0,
        "model/downsample_2": 0.0,
        # NOTE: mFormerV1 currently uses rope stages 3/4.
        "model/rope_stage_2": 0.0,  # legacy / reserved
        "model/rope_stage_3": 0.0,
        "model/rope_stage_4": 0.0,
        "model/aggregation": 0.0,
        # Loss components
        "loss/core_loss": 0.0,
        "loss/masking": 0.0,
        "loss/weighting": 0.0,
        "loss/aggregation": 0.0,
        # Backward/optimizer components
        "backward/ddp_sync_ctx": 0.0,
        "backward/scale_backward": 0.0,
        "gradnorm/reforward": 0.0,
        "optimizer/unscale": 0.0,
        "optimizer/clip_grad": 0.0,
        "optimizer/step": 0.0,
        "optimizer/scaler_update": 0.0,
        "optimizer/zero_grad": 0.0,
        "optimizer/lr_step": 0.0,
        # DDP comms (prefix-matched below)
        "comms/allreduce": 0.0,
        # Augmentation components
        "augmentation/selective_mixing": 0.0,
        "augmentation/gpu_batch_augmentations": 0.0,
    }

    for trace_file in trace_files:
        with open(trace_file) as f:
            trace_data = json.load(f)

        events = trace_data.get("traceEvents", [])

        # Extract step-level metrics
        step_events = [e for e in events if e.get("name", "").startswith("ProfilerStep")]
        steps_profiled += len(step_events)

        # Calculate step times
        for event in step_events:
            if "dur" in event:
                total_step_time += event["dur"] / 1000.0  # Convert microseconds to milliseconds

        # Count CUDA kernels
        cuda_events = [e for e in events if e.get("cat") == "kernel"]
        total_kernel_count += len(cuda_events)

        # Calculate GPU vs CPU time
        gpu_events = [e for e in events if e.get("cat") in ["kernel", "gpu_memcpy", "gpu_sync"]]
        cpu_events = [e for e in events if e.get("cat") == "cpu_op"]

        for event in gpu_events:
            if "dur" in event:
                total_gpu_time += event["dur"] / 1000.0

        for event in cpu_events:
            if "dur" in event:
                total_cpu_time += event["dur"] / 1000.0

        # Extract custom profiler regions (Level 2 / record_function), single pass.
        for event in events:
            name = event.get("name")
            dur_us = event.get("dur")
            if not name or dur_us is None:
                continue

            if name in l2_region_totals:
                l2_region_totals[name] += dur_us / 1000.0  # us -> ms
            elif name.startswith("comms/allreduce_"):
                l2_region_totals["comms/allreduce"] += dur_us / 1000.0  # us -> ms

        # Estimate memory-bound operations
        memory_events = [e for e in events if e.get("cat") == "gpu_memcpy" or "memory" in e.get("name", "").lower()]
        for event in memory_events:
            if "dur" in event:
                total_memory_time += event["dur"] / 1000.0

        # Extract custom profiler regions (Level 1)
        batch_aug_events = [e for e in events if e.get("name") == "gpu_batch_augmentations"]
        mixing_events = [e for e in events if e.get("name") == "gpu_selective_mixing"]

        for event in batch_aug_events:
            if "dur" in event:
                total_batch_aug_time += event["dur"] / 1000.0

        for event in mixing_events:
            if "dur" in event:
                total_mixing_time += event["dur"] / 1000.0

        # Calculate total trace duration
        if events:
            timestamps = [e.get("ts", 0) for e in events if "ts" in e]
            if timestamps:
                trace_start = min(timestamps)
                trace_end = max(timestamps)
                total_duration += (trace_end - trace_start) / 1000.0

    # Calculate averages and percentages
    step_divisor = max(steps_profiled, 1)
    avg_step_time = total_step_time / step_divisor
    total_active_time = total_gpu_time + total_cpu_time

    gpu_utilization = (total_gpu_time / total_duration * 100) if total_duration > 0 else 0
    cpu_time_pct = (total_cpu_time / total_active_time * 100) if total_active_time > 0 else 0
    gpu_time_pct = (total_gpu_time / total_active_time * 100) if total_active_time > 0 else 0
    memory_bandwidth_pct = (total_memory_time / total_gpu_time * 100) if total_gpu_time > 0 else 0

    # --- Level 3 Profiling: Analyze queue_stats.jsonl if present ---
    run_dir = trace_files[0].parent.parent.parent if trace_files else None  # Heuristic to get run dir
    queue_stats_files = list(run_dir.glob("**/queue_stats_rank*.jsonl")) if run_dir else []

    median_batch_q = 0.0
    median_preproc_q = 0.0
    median_ready_q = 0.0
    median_io_tput = 0.0
    median_handoff_tput = 0.0
    median_hit_rate = 0.0

    if queue_stats_files:
        try:
            import pandas as pd

            # For simplicity, we'll just analyze the first file found (usually rank 0)
            df = pd.read_json(queue_stats_files[0], lines=True)
            if not df.empty:
                median_batch_q = df["batch_index_q"].median()
                median_preproc_q = df["preprocess_q"].median()
                median_ready_q = df["processed_batch_q"].median()
                median_io_tput = df["io_throughput_it_s"].median()
                median_handoff_tput = df["handoff_throughput_it_s"].median()
                median_hit_rate = df["cache_hit_rate_pct"].median()
        except Exception:
            # Silently continue if pandas is not available or parsing fails
            pass

    # Convert L2 totals into per-step averages (ms/step), matching avg_step_time_ms.
    total_dataloader_io_wait = l2_region_totals["dataloader/io_wait"] / step_divisor
    total_dataloader_cpu_decode = l2_region_totals["dataloader/cpu_decode"] / step_divisor
    total_dataloader_preprocess_wait = l2_region_totals["dataloader/preprocess_wait"] / step_divisor
    total_dataloader_cpu_xform = l2_region_totals["dataloader/cpu_xform"] / step_divisor

    total_model_stem = l2_region_totals["model/stem"] / step_divisor
    total_model_convnext_stage_0 = l2_region_totals["model/convnext_stage_0"] / step_divisor
    total_model_downsample_0 = l2_region_totals["model/downsample_0"] / step_divisor
    total_model_convnext_stage_1 = l2_region_totals["model/convnext_stage_1"] / step_divisor
    total_model_downsample_1 = l2_region_totals["model/downsample_1"] / step_divisor
    total_model_downsample_2 = l2_region_totals["model/downsample_2"] / step_divisor
    total_model_rope_stage_2 = l2_region_totals["model/rope_stage_2"] / step_divisor
    total_model_rope_stage_3 = l2_region_totals["model/rope_stage_3"] / step_divisor
    total_model_rope_stage_4 = l2_region_totals["model/rope_stage_4"] / step_divisor
    total_model_aggregation = l2_region_totals["model/aggregation"] / step_divisor

    total_loss_core_loss = l2_region_totals["loss/core_loss"] / step_divisor
    total_loss_masking = l2_region_totals["loss/masking"] / step_divisor
    total_loss_weighting = l2_region_totals["loss/weighting"] / step_divisor
    total_loss_aggregation = l2_region_totals["loss/aggregation"] / step_divisor

    total_backward_ddp_sync_ctx = l2_region_totals["backward/ddp_sync_ctx"] / step_divisor
    total_backward_scale_backward = l2_region_totals["backward/scale_backward"] / step_divisor
    total_gradnorm_reforward = l2_region_totals["gradnorm/reforward"] / step_divisor
    total_optimizer_unscale = l2_region_totals["optimizer/unscale"] / step_divisor
    total_optimizer_clip_grad = l2_region_totals["optimizer/clip_grad"] / step_divisor
    total_optimizer_step = l2_region_totals["optimizer/step"] / step_divisor
    total_optimizer_scaler_update = l2_region_totals["optimizer/scaler_update"] / step_divisor
    total_optimizer_zero_grad = l2_region_totals["optimizer/zero_grad"] / step_divisor
    total_optimizer_lr_step = l2_region_totals["optimizer/lr_step"] / step_divisor
    total_comms_allreduce = l2_region_totals["comms/allreduce"] / step_divisor

    total_augmentation_selective_mixing = l2_region_totals["augmentation/selective_mixing"] / step_divisor
    total_augmentation_gpu_batch_augmentations = l2_region_totals["augmentation/gpu_batch_augmentations"] / step_divisor

    return ProfilerMetrics(
        avg_step_time_ms=avg_step_time,
        gpu_utilization_pct=gpu_utilization,
        cpu_time_pct=cpu_time_pct,
        gpu_time_pct=gpu_time_pct,
        memory_bandwidth_pct=memory_bandwidth_pct,
        kernel_count=total_kernel_count,
        total_kernels=total_kernel_count,
        trace_duration_ms=total_duration,
        steps_profiled=steps_profiled,
        batch_aug_time_ms=total_batch_aug_time / step_divisor,
        mixing_time_ms=total_mixing_time / step_divisor,
        # Level 2 profiling metrics
        dataloader_io_wait_ms=total_dataloader_io_wait,
        dataloader_cpu_decode_ms=total_dataloader_cpu_decode,
        dataloader_preprocess_wait_ms=total_dataloader_preprocess_wait,
        dataloader_cpu_xform_ms=total_dataloader_cpu_xform,
        model_stem_ms=total_model_stem,
        model_convnext_stage_0_ms=total_model_convnext_stage_0,
        model_convnext_stage_1_ms=total_model_convnext_stage_1,
        model_rope_stage_2_ms=total_model_rope_stage_2,
        model_rope_stage_3_ms=total_model_rope_stage_3,
        model_rope_stage_4_ms=total_model_rope_stage_4,
        model_downsample_0_ms=total_model_downsample_0,
        model_downsample_1_ms=total_model_downsample_1,
        model_downsample_2_ms=total_model_downsample_2,
        model_aggregation_ms=total_model_aggregation,
        loss_core_loss_ms=total_loss_core_loss,
        loss_masking_ms=total_loss_masking,
        loss_weighting_ms=total_loss_weighting,
        loss_aggregation_ms=total_loss_aggregation,
        backward_ddp_sync_ctx_ms=total_backward_ddp_sync_ctx,
        backward_scale_backward_ms=total_backward_scale_backward,
        gradnorm_reforward_ms=total_gradnorm_reforward,
        optimizer_unscale_ms=total_optimizer_unscale,
        optimizer_clip_grad_ms=total_optimizer_clip_grad,
        optimizer_step_ms=total_optimizer_step,
        optimizer_scaler_update_ms=total_optimizer_scaler_update,
        optimizer_zero_grad_ms=total_optimizer_zero_grad,
        optimizer_lr_step_ms=total_optimizer_lr_step,
        comms_allreduce_ms=total_comms_allreduce,
        augmentation_selective_mixing_ms=total_augmentation_selective_mixing,
        augmentation_gpu_batch_augmentations_ms=total_augmentation_gpu_batch_augmentations,
        # Level 3 profiling metrics
        median_batch_index_q_depth=median_batch_q,
        median_preprocess_q_depth=median_preproc_q,
        median_processed_batch_q_depth=median_ready_q,
        median_io_throughput_it_s=median_io_tput,
        median_handoff_throughput_it_s=median_handoff_tput,
        median_cache_hit_rate_pct=median_hit_rate,
    )


def format_pretty(summary: RunSummary) -> Panel:
    """
    Format summary as a rich console panel.

    Args:
        summary: RunSummary to format

    Returns:
        Rich Panel object for console display
    """
    # Configuration table
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="yellow")

    config_table.add_row("Batch Size", str(summary.config.batch_size))
    config_table.add_row("Accumulation Steps", str(summary.config.accumulation_steps))
    config_table.add_row("Model Type", summary.config.model_type)
    config_table.add_row("AMP Level", summary.config.amp_level)
    config_table.add_row("Aug Pipeline Device", summary.config.aug_pipeline_device)
    config_table.add_row("GPU Compile Enabled", str(summary.config.gpu_compile_enabled))
    if summary.config.gpu_compile_enabled:
        config_table.add_row("GPU Compile Mode", summary.config.gpu_compile_mode)
    config_table.add_row("Profiler Enabled", str(summary.config.profiler_enabled))

    content = [config_table]

    # Profiler metrics table
    if summary.profiler_metrics:
        metrics_table = Table(title="Profiler Metrics", show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        m = summary.profiler_metrics
        metrics_table.add_row("Avg Step Time", f"{m.avg_step_time_ms:.1f} ms")
        metrics_table.add_row("GPU Utilization", f"{m.gpu_utilization_pct:.1f}%")
        metrics_table.add_row("CPU Time", f"{m.cpu_time_pct:.1f}%")
        metrics_table.add_row("GPU Time", f"{m.gpu_time_pct:.1f}%")
        metrics_table.add_row("Memory Bandwidth", f"{m.memory_bandwidth_pct:.1f}%")
        metrics_table.add_row("Kernel Count", str(m.kernel_count))
        metrics_table.add_row("Steps Profiled", str(m.steps_profiled))
        metrics_table.add_row("Trace Duration", f"{m.trace_duration_ms:.1f} ms")

        # Add new augmentation-specific metrics
        if m.batch_aug_time_ms > 0:
            metrics_table.add_row("Batch Aug Time", f"{m.batch_aug_time_ms:.1f} ms")
        if m.mixing_time_ms > 0:
            metrics_table.add_row("Mixing Time", f"{m.mixing_time_ms:.1f} ms")

        content.append(metrics_table)

        # Add Level 2 Component Breakdown Table
        component_table = Table(title="Component Breakdown (Level 2)", show_header=True)
        component_table.add_column("Component", style="cyan")
        component_table.add_column("Time (ms)", style="yellow")
        component_table.add_column("% of Step", style="green")

        # Helper function to add component rows
        def add_component_row(name: str, time_ms: float):
            if time_ms > 0:
                pct_of_step = (time_ms / m.avg_step_time_ms * 100) if m.avg_step_time_ms > 0 else 0
                component_table.add_row(name, f"{time_ms:.1f}", f"{pct_of_step:.1f}%")

        # Dataloader components
        add_component_row("Dataloader I/O Wait", m.dataloader_io_wait_ms)
        add_component_row("Dataloader CPU Decode", m.dataloader_cpu_decode_ms)
        add_component_row("Dataloader Preprocess Wait", m.dataloader_preprocess_wait_ms)
        add_component_row("Dataloader CPU Transform", m.dataloader_cpu_xform_ms)

        # Model components
        add_component_row("Model Stem", m.model_stem_ms)
        add_component_row("Model ConvNeXt Stage 0", m.model_convnext_stage_0_ms)
        add_component_row("Model Downsample 0", m.model_downsample_0_ms)
        add_component_row("Model ConvNeXt Stage 1", m.model_convnext_stage_1_ms)
        add_component_row("Model Downsample 1", m.model_downsample_1_ms)
        add_component_row("Model Downsample 2", m.model_downsample_2_ms)
        add_component_row("Model RoPE Stage 2", m.model_rope_stage_2_ms)
        add_component_row("Model RoPE Stage 3", m.model_rope_stage_3_ms)
        add_component_row("Model RoPE Stage 4", m.model_rope_stage_4_ms)
        add_component_row("Model Aggregation", m.model_aggregation_ms)

        # Loss components
        add_component_row("Loss Core Loss", m.loss_core_loss_ms)
        add_component_row("Loss Masking", m.loss_masking_ms)
        add_component_row("Loss Weighting", m.loss_weighting_ms)
        add_component_row("Loss Aggregation", m.loss_aggregation_ms)

        # Backward/optimizer/comms components
        add_component_row("Backward DDP Sync Ctx", m.backward_ddp_sync_ctx_ms)
        add_component_row("Backward Scale+Backward", m.backward_scale_backward_ms)
        add_component_row("GradNorm Reforward", m.gradnorm_reforward_ms)
        add_component_row("Optimizer Unscale", m.optimizer_unscale_ms)
        add_component_row("Optimizer Clip Grad", m.optimizer_clip_grad_ms)
        add_component_row("Optimizer Step", m.optimizer_step_ms)
        add_component_row("Optimizer Scaler Update", m.optimizer_scaler_update_ms)
        add_component_row("Optimizer Zero Grad", m.optimizer_zero_grad_ms)
        add_component_row("Optimizer LR Step", m.optimizer_lr_step_ms)
        add_component_row("Comms AllReduce", m.comms_allreduce_ms)

        # Augmentation components (Level 2)
        add_component_row("Aug Selective Mixing", m.augmentation_selective_mixing_ms)
        add_component_row("Aug GPU Batch Augmentations", m.augmentation_gpu_batch_augmentations_ms)

        # Only add the table if it has content
        if component_table.row_count > 0:
            content.append(component_table)

        # Add Level 3 Host-side Dataloader Metrics Table if any values are non-zero
        l3_metrics_exist = any(
            [
                m.median_batch_index_q_depth > 0,
                m.median_preprocess_q_depth > 0,
                m.median_processed_batch_q_depth > 0,
                m.median_io_throughput_it_s > 0,
                m.median_handoff_throughput_it_s > 0,
                m.median_cache_hit_rate_pct > 0,
            ]
        )

        if l3_metrics_exist:
            l3_table = Table(title="Host-side Dataloader Metrics (Level 3)", show_header=True)
            l3_table.add_column("Metric", style="cyan")
            l3_table.add_column("Value", style="yellow")

            if m.median_batch_index_q_depth > 0:
                l3_table.add_row("Median Batch Index Q Depth", f"{m.median_batch_index_q_depth:.1f}")
            if m.median_preprocess_q_depth > 0:
                l3_table.add_row("Median Preprocess Q Depth", f"{m.median_preprocess_q_depth:.1f}")
            if m.median_processed_batch_q_depth > 0:
                l3_table.add_row("Median Processed Batch Q Depth", f"{m.median_processed_batch_q_depth:.1f}")
            if m.median_io_throughput_it_s > 0:
                l3_table.add_row("Median I/O Throughput", f"{m.median_io_throughput_it_s:.1f} it/s")
            if m.median_handoff_throughput_it_s > 0:
                l3_table.add_row("Median Handoff Throughput", f"{m.median_handoff_throughput_it_s:.1f} it/s")
            if m.median_cache_hit_rate_pct > 0:
                l3_table.add_row("Median Cache Hit Rate", f"{m.median_cache_hit_rate_pct:.1f}%")

            content.append(l3_table)

    elif summary.has_profiler_traces:
        content.append(f"[red]Error analyzing {summary.trace_files_count} trace files[/red]")
    else:
        content.append("[yellow]No profiler traces found[/yellow]")

    if summary.error_message:
        content.append(f"[red]Error: {summary.error_message}[/red]")

    # Combine content
    from rich.columns import Columns

    if len(content) > 1 and isinstance(content[0], Table) and isinstance(content[1], Table):
        display_content = Columns(content[:2])
        if len(content) > 2:
            remaining = "\n".join(str(c) for c in content[2:])
            display_content = f"{display_content}\n{remaining}"
    else:
        display_content = "\n".join(str(c) for c in content)

    return Panel(display_content, title=f"Run Summary: {summary.run_id}", border_style="blue")


def format_markdown(summary: RunSummary) -> str:
    """
    Format summary as markdown.

    Args:
        summary: RunSummary to format

    Returns:
        Markdown formatted string
    """
    lines = [
        f"# Run Summary: {summary.run_id}",
        "",
        "## Configuration",
        "",
        f"- **Batch Size**: {summary.config.batch_size}",
        f"- **Accumulation Steps**: {summary.config.accumulation_steps}",
        f"- **Model Type**: {summary.config.model_type}",
        f"- **AMP Level**: {summary.config.amp_level}",
        f"- **Aug Pipeline Device**: {summary.config.aug_pipeline_device}",
        f"- **GPU Compile Enabled**: {summary.config.gpu_compile_enabled}",
    ]

    if summary.config.gpu_compile_enabled:
        lines.append(f"- **GPU Compile Mode**: {summary.config.gpu_compile_mode}")

    lines.extend([f"- **Profiler Enabled**: {summary.config.profiler_enabled}", ""])

    if summary.profiler_metrics:
        m = summary.profiler_metrics
        lines.extend(
            [
                "## Profiler Metrics",
                "",
                f"- **Avg Step Time**: {m.avg_step_time_ms:.1f} ms",
                f"- **GPU Utilization**: {m.gpu_utilization_pct:.1f}%",
                f"- **CPU Time**: {m.cpu_time_pct:.1f}%",
                f"- **GPU Time**: {m.gpu_time_pct:.1f}%",
                f"- **Memory Bandwidth**: {m.memory_bandwidth_pct:.1f}%",
                f"- **Kernel Count**: {m.kernel_count}",
                f"- **Steps Profiled**: {m.steps_profiled}",
                f"- **Trace Duration**: {m.trace_duration_ms:.1f} ms",
            ]
        )

        # Add augmentation-specific metrics if present
        if m.batch_aug_time_ms > 0 or m.mixing_time_ms > 0:
            lines.extend(["", "### Augmentation Breakdown", ""])
            if m.batch_aug_time_ms > 0:
                lines.append(f"- **Batch Augmentations**: {m.batch_aug_time_ms:.1f} ms")
            if m.mixing_time_ms > 0:
                lines.append(f"- **Selective Mixing**: {m.mixing_time_ms:.1f} ms")

        # Add Level 2 Component Breakdown
        component_data = []

        # Helper to collect non-zero components
        def add_component(name: str, time_ms: float):
            if time_ms > 0:
                pct_of_step = (time_ms / m.avg_step_time_ms * 100) if m.avg_step_time_ms > 0 else 0
                component_data.append((name, time_ms, pct_of_step))

        # Collect all component data
        add_component("Dataloader I/O Wait", m.dataloader_io_wait_ms)
        add_component("Dataloader CPU Decode", m.dataloader_cpu_decode_ms)
        add_component("Dataloader Preprocess Wait", m.dataloader_preprocess_wait_ms)
        add_component("Dataloader CPU Transform", m.dataloader_cpu_xform_ms)
        add_component("Model Stem", m.model_stem_ms)
        add_component("Model ConvNeXt Stage 0", m.model_convnext_stage_0_ms)
        add_component("Model Downsample 0", m.model_downsample_0_ms)
        add_component("Model ConvNeXt Stage 1", m.model_convnext_stage_1_ms)
        add_component("Model Downsample 1", m.model_downsample_1_ms)
        add_component("Model Downsample 2", m.model_downsample_2_ms)
        add_component("Model RoPE Stage 2", m.model_rope_stage_2_ms)
        add_component("Model RoPE Stage 3", m.model_rope_stage_3_ms)
        add_component("Model RoPE Stage 4", m.model_rope_stage_4_ms)
        add_component("Model Aggregation", m.model_aggregation_ms)
        add_component("Loss Core Loss", m.loss_core_loss_ms)
        add_component("Loss Masking", m.loss_masking_ms)
        add_component("Loss Weighting", m.loss_weighting_ms)
        add_component("Loss Aggregation", m.loss_aggregation_ms)
        add_component("Backward DDP Sync Ctx", m.backward_ddp_sync_ctx_ms)
        add_component("Backward Scale+Backward", m.backward_scale_backward_ms)
        add_component("GradNorm Reforward", m.gradnorm_reforward_ms)
        add_component("Optimizer Unscale", m.optimizer_unscale_ms)
        add_component("Optimizer Clip Grad", m.optimizer_clip_grad_ms)
        add_component("Optimizer Step", m.optimizer_step_ms)
        add_component("Optimizer Scaler Update", m.optimizer_scaler_update_ms)
        add_component("Optimizer Zero Grad", m.optimizer_zero_grad_ms)
        add_component("Optimizer LR Step", m.optimizer_lr_step_ms)
        add_component("Comms AllReduce", m.comms_allreduce_ms)
        add_component("Aug Selective Mixing", m.augmentation_selective_mixing_ms)
        add_component("Aug GPU Batch Augmentations", m.augmentation_gpu_batch_augmentations_ms)

        if component_data:
            lines.extend(["", "### Component Breakdown (Level 2)", ""])
            lines.append("| Component | Time (ms) | % of Step |")
            lines.append("|-----------|-----------|-----------|")
            for name, time_ms, pct_of_step in component_data:
                lines.append(f"| {name} | {time_ms:.1f} | {pct_of_step:.1f}% |")

        # Add Level 3 Host-side Dataloader Metrics if any values are non-zero
        l3_metrics_exist = any(
            [
                m.median_batch_index_q_depth > 0,
                m.median_preprocess_q_depth > 0,
                m.median_processed_batch_q_depth > 0,
                m.median_io_throughput_it_s > 0,
                m.median_handoff_throughput_it_s > 0,
                m.median_cache_hit_rate_pct > 0,
            ]
        )

        if l3_metrics_exist:
            lines.extend(["", "### Host-side Dataloader Metrics (Level 3)", ""])
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            if m.median_batch_index_q_depth > 0:
                lines.append(f"| Median Batch Index Q Depth | {m.median_batch_index_q_depth:.1f} |")
            if m.median_preprocess_q_depth > 0:
                lines.append(f"| Median Preprocess Q Depth | {m.median_preprocess_q_depth:.1f} |")
            if m.median_processed_batch_q_depth > 0:
                lines.append(f"| Median Processed Batch Q Depth | {m.median_processed_batch_q_depth:.1f} |")
            if m.median_io_throughput_it_s > 0:
                lines.append(f"| Median I/O Throughput | {m.median_io_throughput_it_s:.1f} it/s |")
            if m.median_handoff_throughput_it_s > 0:
                lines.append(f"| Median Handoff Throughput | {m.median_handoff_throughput_it_s:.1f} it/s |")
            if m.median_cache_hit_rate_pct > 0:
                lines.append(f"| Median Cache Hit Rate | {m.median_cache_hit_rate_pct:.1f}% |")

        lines.append("")
    elif summary.has_profiler_traces:
        lines.extend(["## Profiler Metrics", "", f"❌ Error analyzing {summary.trace_files_count} trace files", ""])
    else:
        lines.extend(["## Profiler Metrics", "", "⚠️ No profiler traces found", ""])

    if summary.error_message:
        lines.extend(["## Errors", "", f"❌ {summary.error_message}", ""])

    return "\n".join(lines)
