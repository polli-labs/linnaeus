"""
Experiment run comparison and diff utilities.

Provides functions to compare performance metrics between experiment runs
and generate side-by-side reports highlighting differences.
"""

from dataclasses import asdict, dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .summary import RunSummary

console = Console()


@dataclass
class MetricDiff:
    """Represents the difference between two metric values."""

    name: str
    value_a: Any
    value_b: Any
    diff_abs: float | None = None
    diff_pct: float | None = None
    significant: bool = False

    def __post_init__(self):
        """Calculate absolute and percentage differences for numeric values."""
        if isinstance(self.value_a, (int, float)) and isinstance(self.value_b, (int, float)):
            self.diff_abs = self.value_b - self.value_a
            if self.value_a != 0:
                self.diff_pct = (self.diff_abs / self.value_a) * 100
            else:
                self.diff_pct = None

            # Consider >10% change as significant
            self.significant = self.diff_pct is not None and abs(self.diff_pct) > 10


@dataclass
class RunComparison:
    """Complete comparison between two experiment runs."""

    run_a: RunSummary
    run_b: RunSummary
    config_diffs: dict[str, MetricDiff]
    profiler_diffs: dict[str, MetricDiff] | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_a": self.run_a.to_dict(),
            "run_b": self.run_b.to_dict(),
            "config_diffs": {k: asdict(v) for k, v in self.config_diffs.items()},
            "profiler_diffs": {k: asdict(v) for k, v in self.profiler_diffs.items()} if self.profiler_diffs else None,
            "summary": self.summary,
        }


def compare_runs(run_a: RunSummary, run_b: RunSummary) -> RunComparison:
    """
    Compare two experiment runs and generate diff report.

    Args:
        run_a: First run summary
        run_b: Second run summary

    Returns:
        RunComparison object with detailed differences
    """
    # Compare configuration settings
    config_diffs = {}
    config_a = asdict(run_a.config)
    config_b = asdict(run_b.config)

    for key in config_a.keys():
        diff = MetricDiff(name=key, value_a=config_a[key], value_b=config_b[key])
        config_diffs[key] = diff

    # Compare profiler metrics if both have them
    profiler_diffs = None
    if run_a.profiler_metrics and run_b.profiler_metrics:
        profiler_diffs = {}
        metrics_a = asdict(run_a.profiler_metrics)
        metrics_b = asdict(run_b.profiler_metrics)

        for key in metrics_a.keys():
            diff = MetricDiff(name=key, value_a=metrics_a[key], value_b=metrics_b[key])
            profiler_diffs[key] = diff

    # Generate summary
    summary = generate_summary(config_diffs, profiler_diffs)

    return RunComparison(run_a=run_a, run_b=run_b, config_diffs=config_diffs, profiler_diffs=profiler_diffs, summary=summary)


def generate_summary(config_diffs: dict[str, MetricDiff], profiler_diffs: dict[str, MetricDiff] | None) -> str:
    """
    Generate a high-level summary of the differences.

    Args:
        config_diffs: Configuration differences
        profiler_diffs: Profiler metric differences

    Returns:
        Summary string highlighting key changes
    """
    significant_changes = []

    # Check for significant config changes
    important_configs = {
        "batch_size": "Batch Size",
        "aug_pipeline_device": "Augmentation Device",
        "gpu_compile_enabled": "GPU Compilation",
        "gpu_compile_mode": "Compilation Mode",
    }

    for key, display_name in important_configs.items():
        if key in config_diffs:
            diff = config_diffs[key]
            if diff.value_a != diff.value_b:
                significant_changes.append(f"{display_name}: {diff.value_a} → {diff.value_b}")

    # Check for significant performance changes
    if profiler_diffs:
        perf_metrics = {
            "avg_step_time_ms": "Average Step Time",
            "gpu_utilization_pct": "GPU Utilization",
            "memory_bandwidth_pct": "Memory Bandwidth Usage",
            "batch_aug_time_ms": "Batch Augmentation Time",
            "mixing_time_ms": "Selective Mixing Time",
        }

        for key, display_name in perf_metrics.items():
            if key in profiler_diffs:
                diff = profiler_diffs[key]
                if diff.significant:
                    direction = "↑" if diff.diff_abs > 0 else "↓"
                    significant_changes.append(
                        f"{display_name}: {direction} {abs(diff.diff_pct):.1f}% ({diff.value_a:.1f} → {diff.value_b:.1f})"
                    )

    if not significant_changes:
        return "No significant differences detected."

    return "Significant changes: " + "; ".join(significant_changes)


def format_pretty(comparison: RunComparison) -> Panel:
    """
    Format comparison as a rich console panel.

    Args:
        comparison: RunComparison to format

    Returns:
        Rich Panel object for console display
    """
    # Summary header
    content = [f"[bold blue]{comparison.summary}[/bold blue]", ""]

    # Configuration differences
    config_table = Table(title="Configuration Differences", show_header=True)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Run A", style="yellow")
    config_table.add_column("Run B", style="green")
    config_table.add_column("Change", style="magenta")

    for key, diff in comparison.config_diffs.items():
        if diff.value_a != diff.value_b:
            change_str = ""
            if diff.diff_pct is not None:
                change_str = f"{diff.diff_pct:+.1f}%"
            elif diff.diff_abs is not None:
                change_str = f"{diff.diff_abs:+.1f}"

            config_table.add_row(key.replace("_", " ").title(), str(diff.value_a), str(diff.value_b), change_str)

    content.append(config_table)

    # Profiler metrics differences
    if comparison.profiler_diffs:
        metrics_table = Table(title="Performance Metrics Differences", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Run A", style="yellow")
        metrics_table.add_column("Run B", style="green")
        metrics_table.add_column("Change", style="magenta")
        metrics_table.add_column("% Change", style="red")

        for key, diff in comparison.profiler_diffs.items():
            if isinstance(diff.value_a, (int, float)) and isinstance(diff.value_b, (int, float)):
                change_style = "red" if diff.significant else "white"
                pct_change = f"{diff.diff_pct:+.1f}%" if diff.diff_pct is not None else "N/A"

                metrics_table.add_row(
                    key.replace("_", " ").title(),
                    f"{diff.value_a:.2f}",
                    f"{diff.value_b:.2f}",
                    f"[{change_style}]{diff.diff_abs:+.2f}[/{change_style}]",
                    f"[{change_style}]{pct_change}[/{change_style}]",
                )

        content.append(metrics_table)
    elif comparison.run_a.profiler_metrics or comparison.run_b.profiler_metrics:
        content.append("[yellow]⚠️ Cannot compare profiler metrics - one run missing traces[/yellow]")
    else:
        content.append("[yellow]⚠️ No profiler traces available for comparison[/yellow]")

    # Combine content
    display_content = "\n\n".join(str(c) for c in content)

    return Panel(display_content, title=f"Run Comparison: {comparison.run_a.run_id} vs {comparison.run_b.run_id}", border_style="blue")


def format_markdown(comparison: RunComparison) -> str:
    """
    Format comparison as markdown.

    Args:
        comparison: RunComparison to format

    Returns:
        Markdown formatted string
    """
    lines = [
        "# Run Comparison",
        "",
        f"**Run A**: {comparison.run_a.run_id}",
        f"**Run B**: {comparison.run_b.run_id}",
        "",
        "## Summary",
        "",
        comparison.summary,
        "",
        "## Configuration Differences",
        "",
        "| Setting | Run A | Run B | Change |",
        "|---------|--------|-------|--------|",
    ]

    for key, diff in comparison.config_diffs.items():
        if diff.value_a != diff.value_b:
            change_str = ""
            if diff.diff_pct is not None:
                change_str = f"{diff.diff_pct:+.1f}%"
            elif diff.diff_abs is not None:
                change_str = f"{diff.diff_abs:+.1f}"

            lines.append(f"| {key.replace('_', ' ').title()} | {diff.value_a} | {diff.value_b} | {change_str} |")

    if comparison.profiler_diffs:
        lines.extend(
            [
                "",
                "## Performance Metrics Differences",
                "",
                "| Metric | Run A | Run B | Change | % Change |",
                "|--------|--------|-------|--------|----------|",
            ]
        )

        for key, diff in comparison.profiler_diffs.items():
            if isinstance(diff.value_a, (int, float)) and isinstance(diff.value_b, (int, float)):
                pct_change = f"{diff.diff_pct:+.1f}%" if diff.diff_pct is not None else "N/A"
                significance = " ⚠️" if diff.significant else ""

                lines.append(
                    f"| {key.replace('_', ' ').title()} | {diff.value_a:.2f} | {diff.value_b:.2f} | "
                    f"{diff.diff_abs:+.2f} | {pct_change}{significance} |"
                )
    elif comparison.run_a.profiler_metrics or comparison.run_b.profiler_metrics:
        lines.extend(["", "## Performance Metrics", "", "⚠️ Cannot compare profiler metrics - one run missing traces"])
    else:
        lines.extend(["", "## Performance Metrics", "", "⚠️ No profiler traces available for comparison"])

    lines.append("")
    return "\n".join(lines)


def format_html(comparison: RunComparison) -> str:
    """
    Format comparison as self-contained HTML report.

    Args:
        comparison: RunComparison to format

    Returns:
        HTML formatted string
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Linnaeus Run Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .significant {{ color: #d32f2f; font-weight: bold; }}
        .improvement {{ color: #2e7d32; }}
        .regression {{ color: #d32f2f; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Run Comparison Report</h1>
        <p><strong>Run A:</strong> {comparison.run_a.run_id}</p>
        <p><strong>Run B:</strong> {comparison.run_b.run_id}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>{comparison.summary}</p>
    </div>
    
    <h2>Configuration Differences</h2>
    <table>
        <tr>
            <th>Setting</th>
            <th>Run A</th>
            <th>Run B</th>
            <th>Change</th>
        </tr>
"""

    for key, diff in comparison.config_diffs.items():
        if diff.value_a != diff.value_b:
            change_str = ""
            if diff.diff_pct is not None:
                change_str = f"{diff.diff_pct:+.1f}%"
            elif diff.diff_abs is not None:
                change_str = f"{diff.diff_abs:+.1f}"

            html += f"""
        <tr>
            <td>{key.replace("_", " ").title()}</td>
            <td>{diff.value_a}</td>
            <td>{diff.value_b}</td>
            <td>{change_str}</td>
        </tr>"""

    html += """
    </table>
"""

    if comparison.profiler_diffs:
        html += """
    <h2>Performance Metrics Differences</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Run A</th>
            <th>Run B</th>
            <th>Change</th>
            <th>% Change</th>
        </tr>
"""

        for key, diff in comparison.profiler_diffs.items():
            if isinstance(diff.value_a, (int, float)) and isinstance(diff.value_b, (int, float)):
                pct_change = f"{diff.diff_pct:+.1f}%" if diff.diff_pct is not None else "N/A"

                # Determine CSS class for styling
                css_class = ""
                if diff.significant:
                    if diff.diff_abs < 0:  # Improvement (lower is better for most metrics)
                        css_class = "improvement"
                    else:
                        css_class = "regression"

                html += f"""
        <tr>
            <td>{key.replace("_", " ").title()}</td>
            <td>{diff.value_a:.2f}</td>
            <td>{diff.value_b:.2f}</td>
            <td class="{css_class}">{diff.diff_abs:+.2f}</td>
            <td class="{css_class}">{pct_change}</td>
        </tr>"""

        html += """
    </table>
"""
    else:
        html += """
    <h2>Performance Metrics</h2>
    <p>⚠️ Cannot compare profiler metrics - missing traces from one or both runs</p>
"""

    html += """
</body>
</html>
"""

    return html
