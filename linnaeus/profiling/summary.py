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

        # Estimate memory-bound operations
        memory_events = [e for e in events if e.get("cat") == "gpu_memcpy" or "memory" in e.get("name", "").lower()]
        for event in memory_events:
            if "dur" in event:
                total_memory_time += event["dur"] / 1000.0

        # Extract custom profiler regions
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
    avg_step_time = total_step_time / max(steps_profiled, 1)
    total_active_time = total_gpu_time + total_cpu_time

    gpu_utilization = (total_gpu_time / total_duration * 100) if total_duration > 0 else 0
    cpu_time_pct = (total_cpu_time / total_active_time * 100) if total_active_time > 0 else 0
    gpu_time_pct = (total_gpu_time / total_active_time * 100) if total_active_time > 0 else 0
    memory_bandwidth_pct = (total_memory_time / total_gpu_time * 100) if total_gpu_time > 0 else 0

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
        batch_aug_time_ms=total_batch_aug_time,
        mixing_time_ms=total_mixing_time,
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

        lines.append("")
    elif summary.has_profiler_traces:
        lines.extend(["## Profiler Metrics", "", f"❌ Error analyzing {summary.trace_files_count} trace files", ""])
    else:
        lines.extend(["## Profiler Metrics", "", "⚠️ No profiler traces found", ""])

    if summary.error_message:
        lines.extend(["## Errors", "", f"❌ {summary.error_message}", ""])

    return "\n".join(lines)
