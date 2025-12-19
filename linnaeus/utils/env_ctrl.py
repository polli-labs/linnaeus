"""Environment variable control and scenario management for Linnaeus.

This module provides centralized environment variable management with:
- Scenario-based defaults (single GPU, multi-GPU, DGX H100)
- Validation and type checking
- Pretty-printing and logging
- Export to file for reproducibility
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


# Safe defaults for single-GPU workstation
LINNAEUS_SAFE_DEFAULT_ENV = {
    # Core BLAS / threading (keep CPU noise low)
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "OPENCV_NUM_THREADS": "1",
    "HDF5_USE_THREADS": "0",
    # PyTorch CPU side
    "TORCH_INTRAOP_NUM_THREADS": "2",
    "TORCH_INTEROP_NUM_THREADS": "1",
    # Keep this minimal + forward-compatible. Torch parses this at CUDA init.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TORCH_COMPILE_DISABLE": "1",
    # NCCL (single-node, PCIe only)
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_DISABLE": "0",
    "NCCL_P2P_LEVEL": "PXB",
    "NCCL_BLOCKING_WAIT": "1",
    "NCCL_ALGO": "Ring,Tree",
    "NCCL_MIN_NCHANNELS": "4",
    "NCCL_MAX_NCHANNELS": "4",
    "NCCL_TOPO_DUMP_FILE": "/tmp/nccl_graph.xml",
    # Torch >=2.0 expects OFF|INFO|DETAIL (invalid values can break torch import)
    "TORCH_DISTRIBUTED_DEBUG": "OFF",
}

# High-end defaults for DGX H100
LINNAEUS_DGX_H100_ENV = {
    # Core CPU threads (2×64C AMD EPYC / Intel Sapphire Rapids)
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "TBB_NUM_THREADS": "4",
    "OPENCV_NUM_THREADS": "2",
    "HDF5_USE_THREADS": "0",
    "TORCH_INTRAOP_NUM_THREADS": "8",
    "TORCH_INTEROP_NUM_THREADS": "4",
    # Keep this minimal + forward-compatible. Torch parses this at CUDA init.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TORCH_COMPILE_DISABLE": "0",
    # NCCL - tuned for NVLink-Switch & InfiniBand
    "NCCL_IB_DISABLE": "0",
    "NCCL_NVLS_ENABLE": "1",
    "NCCL_COLLNET_ENABLE": "1",
    "NCCL_ALGO": "Tree,Ring",
    "NCCL_BLOCKING_WAIT": "0",
    "NCCL_MIN_NCHANNELS": "8",
    "NCCL_MAX_NCHANNELS": "16",
    "NCCL_P2P_DISABLE": "0",
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_NET_GDR_LEVEL": "2",
    # CUDA runtime
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    # Optional debug flags
    "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    "NCCL_TOPO_DUMP_FILE": "/tmp/nccl_dgx_h100.xml",
}

# Multi-GPU workstation defaults (between safe and DGX)
LINNAEUS_MULTI_GPU_ENV = {
    **LINNAEUS_SAFE_DEFAULT_ENV,
    # Adjust for multi-GPU
    "NCCL_P2P_DISABLE": "0",
    "NCCL_P2P_LEVEL": "PXB",
    "TORCH_INTRAOP_NUM_THREADS": "4",
    "TORCH_INTEROP_NUM_THREADS": "2",
}


SCENARIO_DEFAULTS = {
    "safe_defaults": LINNAEUS_SAFE_DEFAULT_ENV,
    "single_gpu_workstation": LINNAEUS_SAFE_DEFAULT_ENV,
    "multi_gpu_workstation": LINNAEUS_MULTI_GPU_ENV,
    "dgx_h100": LINNAEUS_DGX_H100_ENV,
}


def load_env_defaults(scenario: str = "safe_defaults") -> dict[str, str]:
    """Load environment defaults for a given scenario.

    Args:
        scenario: One of 'safe_defaults', 'single_gpu_workstation',
                 'multi_gpu_workstation', 'dgx_h100'

    Returns:
        Dictionary of environment variable names to values
    """
    if scenario not in SCENARIO_DEFAULTS:
        logger.warning(f"Unknown scenario '{scenario}', using safe_defaults")
        scenario = "safe_defaults"

    return SCENARIO_DEFAULTS[scenario].copy()


def load_yaml_env(path: str | Path) -> dict[str, str]:
    """Load environment variables from a YAML file.

    The YAML file should have a structure like:
    BLAS:
        OMP_NUM_THREADS: 1
        MKL_NUM_THREADS: 1
    TORCH:
        TORCH_INTRAOP_NUM_THREADS: 2

    Args:
        path: Path to YAML file

    Returns:
        Flattened dictionary of environment variables
    """
    if not path or not Path(path).exists():
        return {}

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Flatten nested structure
    flat = {}
    for section, vars in data.items():
        if isinstance(vars, dict):
            for k, v in vars.items():
                flat[k] = str(v)

    return flat


def merge_yaml_env(base: dict[str, str], yaml_path: str | Path | None) -> dict[str, str]:
    """Merge environment variables from YAML file into base dictionary.

    Args:
        base: Base environment variables
        yaml_path: Optional path to YAML file with overrides

    Returns:
        Merged environment variables
    """
    if not yaml_path:
        return base.copy()

    overrides = load_yaml_env(yaml_path)
    merged = base.copy()
    merged.update(overrides)
    return merged


def init_from_config(cfg) -> dict[str, str]:
    """Initialize environment from config, applying scenario defaults.

    Args:
        cfg: Linnaeus config object with ENV.SCENARIO field

    Returns:
        Resolved environment variables that were set
    """
    # Start with current environment
    current_env = dict(os.environ)

    # Load scenario defaults if specified
    scenario = getattr(cfg.ENV, "SCENARIO", "safe_defaults")
    defaults = load_env_defaults(scenario)

    # Apply any YAML overrides
    if hasattr(cfg.ENV, "YAML_OVERRIDES") and cfg.ENV.YAML_OVERRIDES:
        defaults = merge_yaml_env(defaults, cfg.ENV.YAML_OVERRIDES)

    # Only set variables that aren't already in environment
    resolved = {}
    for k, v in defaults.items():
        if k not in current_env:
            os.environ[k] = v
            resolved[k] = v
        else:
            resolved[k] = current_env[k]

    return resolved


def pretty_print_env(env: dict[str, str], title: str = "Resolved Environment Variables", output_dir: str | Path | None = None) -> None:
    """Pretty print environment variables as a table.

    Args:
        env: Dictionary of environment variables
        title: Table title
        output_dir: Optional directory for fallback ENV_VARS.txt file
    """
    try:
        console = Console()
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Variable", style="yellow", no_wrap=True)
        table.add_column("Value", style="green")

        # Group by category
        categories = {
            "BLAS/Threading": ["OMP_", "MKL_", "OPENBLAS_", "TBB_", "OPENCV_", "HDF5_"],
            "PyTorch": ["TORCH_", "PYTORCH_"],
            "NCCL": ["NCCL_"],
            "CUDA": ["CUDA_"],
            "Other": [],
        }

        categorized = {cat: [] for cat in categories}

        for k, v in sorted(env.items()):
            found = False
            for cat, prefixes in categories.items():
                if cat == "Other":
                    continue
                if any(k.startswith(p) for p in prefixes):
                    categorized[cat].append((k, v))
                    found = True
                    break
            if not found:
                categorized["Other"].append((k, v))

        # Print by category
        for cat, vars in categorized.items():
            if vars:
                table.add_row(f"[bold]{cat}[/bold]", "", style="bold blue")
                for k, v in vars:
                    table.add_row(f"  {k}", v)

        console.print(table)

        # Also write plain text fallback to ENV_VARS.txt for CI environments
        fallback_path = Path(output_dir) / "ENV_VARS.txt" if output_dir else "ENV_VARS.txt"
        write_env_dump(env, fallback_path)

    except Exception as e:
        # Fallback for environments without rich or ANSI support
        logger.warning(f"Rich table rendering failed ({e}), using plain text fallback")
        _print_env_plain(env, title)
        # Always write the plain text dump as fallback
        fallback_path = Path(output_dir) / "ENV_VARS.txt" if output_dir else "ENV_VARS.txt"
        write_env_dump(env, fallback_path)


def _print_env_plain(env: dict[str, str], title: str) -> None:
    """Plain text fallback for environment variable printing.

    Args:
        env: Dictionary of environment variables
        title: Table title
    """
    print(f"\n{title}")
    print("=" * len(title))

    # Group by category (same logic as pretty print)
    categories = {
        "BLAS/Threading": ["OMP_", "MKL_", "OPENBLAS_", "TBB_", "OPENCV_", "HDF5_"],
        "PyTorch": ["TORCH_", "PYTORCH_"],
        "NCCL": ["NCCL_"],
        "CUDA": ["CUDA_"],
        "Other": [],
    }

    categorized = {cat: [] for cat in categories}

    for k, v in sorted(env.items()):
        found = False
        for cat, prefixes in categories.items():
            if cat == "Other":
                continue
            if any(k.startswith(p) for p in prefixes):
                categorized[cat].append((k, v))
                found = True
                break
        if not found:
            categorized["Other"].append((k, v))

    # Print by category
    for cat, vars in categorized.items():
        if vars:
            print(f"\n{cat}:")
            for k, v in vars:
                print(f"  {k}: {v}")

    print()  # Final newline


def write_env_dump(env: dict[str, str], output_path: str | Path) -> None:
    """Write environment variables to a file for reproducibility.

    Args:
        env: Dictionary of environment variables
        output_path: Path to write the dump
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Linnaeus Environment Variables\n")
        f.write("# Generated at startup for reproducibility\n\n")

        # Group by category (same as pretty print)
        categories = {
            "BLAS/Threading": ["OMP_", "MKL_", "OPENBLAS_", "TBB_", "OPENCV_", "HDF5_"],
            "PyTorch": ["TORCH_", "PYTORCH_"],
            "NCCL": ["NCCL_"],
            "CUDA": ["CUDA_"],
            "Other": [],
        }

        categorized = {cat: [] for cat in categories}

        for k, v in sorted(env.items()):
            found = False
            for cat, prefixes in categories.items():
                if cat == "Other":
                    continue
                if any(k.startswith(p) for p in prefixes):
                    categorized[cat].append((k, v))
                    found = True
                    break
            if not found:
                categorized["Other"].append((k, v))

        # Write by category
        for cat, vars in categorized.items():
            if vars:
                f.write(f"\n# {cat}\n")
                for k, v in vars:
                    f.write(f'export {k}="{v}"\n')

    logger.info(f"Environment variables written to {output_path}")


def generate_markdown_table() -> str:
    """Generate markdown table of all environment variables for documentation.

    Returns:
        Markdown-formatted table string
    """
    lines = ["# Environment Variables Reference\n"]
    lines.append("| Variable | Safe Default | Multi-GPU | DGX H100 | Description |")
    lines.append("|----------|--------------|-----------|----------|-------------|")

    # Collect all unique variables
    all_vars = set()
    for scenario_env in SCENARIO_DEFAULTS.values():
        all_vars.update(scenario_env.keys())

    # Variable descriptions
    descriptions = {
        "OMP_NUM_THREADS": "OpenMP thread count",
        "MKL_NUM_THREADS": "Intel MKL thread count",
        "OPENBLAS_NUM_THREADS": "OpenBLAS thread count",
        "TBB_NUM_THREADS": "Intel TBB thread count",
        "OPENCV_NUM_THREADS": "OpenCV thread count",
        "HDF5_USE_THREADS": "HDF5 threading (0=disabled)",
        "TORCH_INTRAOP_NUM_THREADS": "PyTorch intra-op parallelism",
        "TORCH_INTEROP_NUM_THREADS": "PyTorch inter-op parallelism",
        "PYTORCH_CUDA_ALLOC_CONF": "CUDA allocator configuration",
        "TORCH_COMPILE_DISABLE": "Disable torch.compile (1=disabled)",
        "NCCL_IB_DISABLE": "Disable InfiniBand (1=disabled)",
        "NCCL_P2P_DISABLE": "Disable P2P communication",
        "NCCL_P2P_LEVEL": "P2P level (PXB/NVL)",
        "NCCL_BLOCKING_WAIT": "Blocking wait mode",
        "NCCL_ALGO": "NCCL algorithms to use",
        "NCCL_MIN_NCHANNELS": "Minimum NCCL channels",
        "NCCL_MAX_NCHANNELS": "Maximum NCCL channels",
        "NCCL_NVLS_ENABLE": "Enable NVLink-Switch",
        "NCCL_COLLNET_ENABLE": "Enable CollNet",
        "NCCL_NET_GDR_LEVEL": "GPUDirect RDMA level",
        "CUDA_DEVICE_MAX_CONNECTIONS": "Max CUDA device connections",
        "TORCH_DISTRIBUTED_DEBUG": "Distributed debug level",
        "NCCL_TOPO_DUMP_FILE": "NCCL topology dump path",
    }

    for var in sorted(all_vars):
        safe = LINNAEUS_SAFE_DEFAULT_ENV.get(var, "-")
        multi = LINNAEUS_MULTI_GPU_ENV.get(var, "-")
        dgx = LINNAEUS_DGX_H100_ENV.get(var, "-")
        desc = descriptions.get(var, "")

        lines.append(f"| {var} | {safe} | {multi} | {dgx} | {desc} |")

    return "\n".join(lines)


if __name__ == "__main__":
    # CLI usage for generating documentation
    import sys

    if "--md" in sys.argv:
        print(generate_markdown_table())
    else:
        # Demo the functionality
        print("Scenario: safe_defaults")
        env = load_env_defaults("safe_defaults")
        pretty_print_env(env)
