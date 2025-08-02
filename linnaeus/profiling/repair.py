"""
PyTorch profiler trace repair utilities.

This module provides automated detection and repair of corrupted PyTorch profiler
JSON traces, particularly addressing H100 DDP-related corruption patterns.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class ProfilerTraceRepair:
    """Handles detection and repair of corrupted profiler traces."""

    # Known corruption patterns and their fixes
    CORRUPTION_PATTERNS = [
        # Pattern 1: Empty Process Group Description (most common H100 DDP issue)
        (r'"Process Group Description"\s*:\s*,', r'"Process Group Description": "",'),
        # Pattern 2: Generic empty field pattern for DDP metadata
        (r'"(Process Group Description|Description|Name)"\s*:\s*,', r'"\1": "",'),
    ]

    # Suffix for repaired files
    REPAIRED_SUFFIX = ".repaired"

    @classmethod
    def detect_corruption(cls, file_path: Path) -> bool:
        """
        Detect if a JSON trace file is corrupted.
        
        Args:
            file_path: Path to the JSON trace file
            
        Returns:
            True if corruption detected, False otherwise
        """
        try:
            # Try to parse the JSON file
            with open(file_path, encoding='utf-8') as f:
                json.load(f)
            return False  # File parses correctly, no corruption
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error in {file_path}: {e}")
            # Check if it matches our known corruption patterns
            try:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()
                for pattern, _ in cls.CORRUPTION_PATTERNS:
                    if re.search(pattern, content):
                        return True
            except Exception:
                pass
            return True  # Some kind of corruption exists
        except Exception as e:
            logger.warning(f"Error checking {file_path}: {e}")
            return False

    @classmethod
    def repair_trace(cls, input_path: Path, output_path: Path | None = None,
                    validate: bool = True) -> tuple[bool, str | None]:
        """
        Repair a corrupted profiler trace file.
        
        Args:
            input_path: Path to corrupted trace file
            output_path: Optional output path (if None, uses .repaired suffix)
            validate: Whether to validate the repaired JSON
            
        Returns:
            (success, error_message) tuple
        """
        if output_path is None:
            # Add .repaired suffix before .json
            base = str(input_path).replace('.pt.trace.json', '')
            output_path = Path(f"{base}.pt.trace.repaired.json")

        try:
            logger.info(f"Repairing trace: {input_path}")

            # Read the file content
            with open(input_path, encoding='utf-8') as f:
                content = f.read()

            original_size = len(content)
            repairs_made = 0

            # Apply all known repair patterns
            for pattern, replacement in cls.CORRUPTION_PATTERNS:
                content, count = re.subn(pattern, replacement, content)
                repairs_made += count
                if count > 0:
                    logger.debug(f"Applied pattern {pattern}: {count} replacements")

            if repairs_made == 0:
                logger.warning(f"No known corruption patterns found in {input_path}")

            # Validate if requested
            if validate:
                try:
                    data = json.loads(content)
                    trace_events = len(data.get('traceEvents', []))
                    logger.info(f"Validation successful: {trace_events} trace events")
                except json.JSONDecodeError as e:
                    error_msg = f"Repaired JSON still invalid: {e}"
                    logger.error(error_msg)
                    return False, error_msg

            # Write the repaired content
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Repaired trace saved to: {output_path}")
            logger.debug(f"Size: {original_size} -> {len(content)} bytes, {repairs_made} repairs")

            return True, None

        except Exception as e:
            error_msg = f"Error repairing {input_path}: {e}"
            logger.error(error_msg)
            return False, error_msg

    @classmethod
    def repair_directory(cls, directory: Path, recursive: bool = True,
                        dry_run: bool = False) -> dict[str, list[Path]]:
        """
        Repair all corrupted traces in a directory.
        
        Args:
            directory: Directory to scan for traces
            recursive: Whether to scan recursively
            dry_run: If True, only detect but don't repair
            
        Returns:
            Dictionary with 'repaired', 'failed', and 'skipped' lists
        """
        results = {
            'repaired': [],
            'failed': [],
            'skipped': [],
            'already_repaired': []
        }

        # Find all trace files
        pattern = '**/*.pt.trace.json' if recursive else '*.pt.trace.json'
        trace_files = list(directory.glob(pattern))

        logger.info(f"Found {len(trace_files)} trace files in {directory}")

        for trace_file in trace_files:
            # Skip already repaired files
            if '.repaired.' in str(trace_file):
                results['already_repaired'].append(trace_file)
                continue

            # Check if repair already exists
            base = str(trace_file).replace('.pt.trace.json', '')
            repaired_path = Path(f"{base}.pt.trace.repaired.json")
            if repaired_path.exists():
                logger.debug(f"Repaired version already exists: {repaired_path}")
                results['skipped'].append(trace_file)
                continue

            # Check if file is corrupted
            if not cls.detect_corruption(trace_file):
                logger.debug(f"No corruption detected: {trace_file}")
                results['skipped'].append(trace_file)
                continue

            # Repair the file (unless dry run)
            if dry_run:
                logger.info(f"[DRY RUN] Would repair: {trace_file}")
                results['repaired'].append(trace_file)
            else:
                success, error = cls.repair_trace(trace_file)
                if success:
                    results['repaired'].append(trace_file)
                else:
                    results['failed'].append(trace_file)
                    logger.error(f"Failed to repair {trace_file}: {error}")

        # Log summary
        logger.info(f"Repair summary: {len(results['repaired'])} repaired, "
                   f"{len(results['failed'])} failed, {len(results['skipped'])} skipped")

        return results

    @classmethod
    def get_best_trace_path(cls, original_path: Path) -> Path:
        """
        Get the best available trace path (repaired if exists, otherwise original).
        
        Args:
            original_path: Path to original trace file
            
        Returns:
            Path to best available trace (repaired or original)
        """
        # Check for repaired version
        base = str(original_path).replace('.pt.trace.json', '')
        repaired_path = Path(f"{base}.pt.trace.repaired.json")

        if repaired_path.exists():
            logger.debug(f"Using repaired trace: {repaired_path}")
            return repaired_path

        return original_path


def auto_repair_trace(trace_path: Path) -> Path:
    """
    Automatically repair a trace if needed and return the best path.
    
    Args:
        trace_path: Path to trace file
        
    Returns:
        Path to best available trace (repaired or original)
    """
    # First check if repaired version already exists
    best_path = ProfilerTraceRepair.get_best_trace_path(trace_path)
    if best_path != trace_path:
        return best_path

    # Check if repair is needed
    if ProfilerTraceRepair.detect_corruption(trace_path):
        logger.info(f"Corruption detected, attempting auto-repair: {trace_path}")
        success, error = ProfilerTraceRepair.repair_trace(trace_path)
        if success:
            return ProfilerTraceRepair.get_best_trace_path(trace_path)
        else:
            logger.warning(f"Auto-repair failed, using original: {error}")

    return trace_path


def repair_run_traces(run_path: Path) -> dict[str, list[Path]]:
    """
    Repair all traces in an experiment run directory.
    
    Args:
        run_path: Path to experiment run directory
        
    Returns:
        Dictionary with repair results
    """
    # Check common profiler locations
    profiler_dirs = [
        run_path / "assets" / "profiler",
        run_path / "profiler"
    ]

    all_results = {
        'repaired': [],
        'failed': [],
        'skipped': [],
        'already_repaired': []
    }

    for profiler_dir in profiler_dirs:
        if profiler_dir.exists():
            results = ProfilerTraceRepair.repair_directory(profiler_dir, recursive=False)
            for key in all_results:
                all_results[key].extend(results.get(key, []))

    return all_results
