"""
Concurrent Trial Executor for parallel profiling execution.

Executes profiling trials concurrently across multiple GPUs with proper
isolation, error handling, and resource management.
"""

import concurrent.futures
import copy
import datetime
import hashlib
import inspect
import re
import subprocess
import os
import time
import logging
import yaml
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from .gpu_pool import GPUPoolManager
from linnaeus.utils.init_timing import extract_init_timing_payloads, summarize_init_timings

logger = logging.getLogger(__name__)


_DOCKER_SERVICE_NAME = "linnaeus-training"
_MODEL_CONFIG_PATH_RE = re.compile(r"Model config => (?P<path>/\S+)")
_EXPERIMENT_CONFIG_PATH_RE = re.compile(r"Full experiment config => (?P<path>/\S+)")
_ENV_VARS_WRITTEN_RE = re.compile(r"Environment variables written to (?P<path>/\S+)")


def _derive_experiment_dir(path: str) -> str | None:
    if not path:
        return None
    p = Path(path.strip())
    if p.suffix == ".yaml" and p.parent.name == "configs":
        return str(p.parent.parent)
    if p.parent.name == "logs":
        return str(p.parent.parent)
    return str(p)


def _extract_experiment_path(stdout_lines: list[str]) -> str | None:
    for line in stdout_lines:
        match = _MODEL_CONFIG_PATH_RE.search(line)
        if match:
            return _derive_experiment_dir(match.group("path"))
        match = _EXPERIMENT_CONFIG_PATH_RE.search(line)
        if match:
            return _derive_experiment_dir(match.group("path"))
        match = _ENV_VARS_WRITTEN_RE.search(line)
        if match:
            return _derive_experiment_dir(match.group("path"))
    return None


def _sanitize_identifier(value: str, *, fallback: str) -> str:
    """
    Sanitize a string to a conservative identifier:
    - lowercase
    - only [a-z0-9_-]
    - collapse invalid spans to single '-'
    """
    if not value:
        return fallback
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9_-]+", "-", lowered)
    lowered = lowered.strip("-_")
    return lowered or fallback


def _make_compose_project_name(*, trial_name: str, gpu_ids: list[int]) -> str:
    """
    Create a docker-compose safe project name.

    Requirements:
    - filesystem + docker safe: [a-z0-9_-]
    - unique across concurrent trials
    - stable within a single trial run
    - short-ish (keep under typical docker limits)
    """
    safe_trial = _sanitize_identifier(trial_name, fallback="trial")
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    gpu_ids_str = ",".join(str(gid) for gid in gpu_ids)
    digest = hashlib.sha1(f"{trial_name}|{gpu_ids_str}|{time.time_ns()}".encode("utf-8")).hexdigest()[:8]

    prefix = "linprof"
    suffix = f"g{len(gpu_ids)}_{timestamp}_{digest}"

    # Keep under typical limits (docker frequently truncates around 63 chars).
    max_trial_len = max(8, 63 - (len(prefix) + 1 + len(suffix) + 1))
    safe_trial = safe_trial[:max_trial_len]
    return f"{prefix}_{safe_trial}_{suffix}"


def _normalize_environment_for_gpu(environment: Any, *, gpu_ids: list[int]) -> Any:
    """
    Normalize a docker-compose service 'environment' entry to ensure:
    - CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES are set consistently for
      the selected host GPU ids
    - supports list[str] and dict[str, str] representations
    """
    if not gpu_ids:
        return environment

    gpu_ids_str = ",".join(str(gid) for gid in gpu_ids)
    container_ids = ",".join(str(idx) for idx in range(len(gpu_ids)))

    if environment is None:
        environment = []

    if isinstance(environment, dict):
        env_dict = dict(environment)
        env_dict["NVIDIA_VISIBLE_DEVICES"] = gpu_ids_str
        env_dict["CUDA_VISIBLE_DEVICES"] = container_ids
        return env_dict

    if isinstance(environment, list):
        cleaned: list[Any] = []
        for item in environment:
            if isinstance(item, str) and item.startswith("CUDA_VISIBLE_DEVICES="):
                continue
            if isinstance(item, str) and item.startswith("NVIDIA_VISIBLE_DEVICES="):
                continue
            cleaned.append(item)

        cleaned.append(f"CUDA_VISIBLE_DEVICES={container_ids}")
        cleaned.append(f"NVIDIA_VISIBLE_DEVICES={gpu_ids_str}")
        return cleaned

    # Unknown type: coerce to list and apply conservatively.
    return [f"CUDA_VISIBLE_DEVICES={container_ids}", f"NVIDIA_VISIBLE_DEVICES={gpu_ids_str}"]


def _pin_service_to_gpu_ids(service: Dict[str, Any], *, gpu_ids: list[int]) -> None:
    """
    Pin the docker-compose service to a specific host GPU id by setting
    `deploy.resources.reservations.devices[].device_ids = ["<gpu_id>", ...]` and
    removing `count` (they are mutually exclusive).

    On some docker-compose setups, relying on `NVIDIA_VISIBLE_DEVICES` alone is
    not sufficient to select the desired host GPU when the template uses a
    single-GPU reservation (e.g., `count: 1`). Without `device_ids`, compose may
    always pick GPU0, causing "GPU1" trials to accidentally run on GPU0 and
    collide/oom under concurrency.
    """
    deploy = service.get("deploy")
    if not isinstance(deploy, dict):
        return
    resources = deploy.get("resources")
    if not isinstance(resources, dict):
        return
    reservations = resources.get("reservations")
    if not isinstance(reservations, dict):
        return
    devices = reservations.get("devices")
    if not isinstance(devices, list):
        return

    for device in devices:
        if not isinstance(device, dict):
            continue
        capabilities = device.get("capabilities")
        has_gpu_cap = isinstance(capabilities, list) and "gpu" in capabilities
        if device.get("driver") == "nvidia" or has_gpu_cap:
            device.pop("count", None)
            device["device_ids"] = [str(gpu_id) for gpu_id in gpu_ids]
            return


class ConcurrentTrialExecutor:
    """
    Executes profiling trials concurrently across multiple GPUs.
    
    Features:
    - Thread pool execution with configurable workers
    - Automatic GPU assignment via GPUPoolManager
    - Error isolation and recovery
    - Progress tracking and reporting
    """
    
    def __init__(self, 
                 gpu_pool: GPUPoolManager,
                 max_workers: int = 2,
                 stagger_delay: float = 5.0,
                 status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        """
        Initialize concurrent trial executor.
        
        Args:
            gpu_pool: GPU pool manager instance
            max_workers: Maximum concurrent worker threads
            stagger_delay: Delay between trial starts (seconds)
        """
        self.gpu_pool = gpu_pool
        self.max_workers = max_workers
        self.stagger_delay = stagger_delay
        self.status_callback = status_callback
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.results = []
        self.active_trials = {}
        self.completed_count = 0
        self.total_count = 0
        
    def _emit_status(self, trial_name: str, payload: Dict[str, Any]) -> None:
        if self.status_callback:
            self.status_callback(trial_name, payload)

    def run_trial_on_gpu(self,
                        trial: Dict[str, Any],
                        template_data: Dict[str, Any],
                        output_dir: str,
                        timeout: int,
                        capture_debug_logs: bool = False,
                        gpu_ids: Optional[list[int]] = None,
                        required_gpus: int = 1,
                        modify_compose_fn=None) -> Dict[str, Any]:
        """
        Execute a single trial on a specific GPU.
        
        Args:
            trial: Trial configuration dict
            template_data: Docker compose template data
            output_dir: Output directory for trial results
            timeout: Trial timeout in seconds
            capture_debug_logs: Whether to capture debug logs on failure
            gpu_id: Specific GPU to use (None for auto-assignment)
            modify_compose_fn: Function to modify compose data for trial
            
        Returns:
            Result dictionary with status, timing, and output paths
        """
        trial_name = trial["name"]
        start_time = time.time()

        process = None
        compose_project_name: str | None = None
        compose_path: Path | None = None
        trial_output_dir: Path | None = None
        result: Dict[str, Any] | None = None
        
        # Acquire GPU if not specified
        acquired_gpu = False
        if gpu_ids is None:
            gpu_ids = self.gpu_pool.acquire_gpus(trial_name, count=required_gpus, timeout=3600)
            if gpu_ids is None:
                return {
                    'name': trial_name,
                    'status': 'timeout',
                    'error': f'Failed to acquire {required_gpus} GPU(s) within timeout',
                }
            acquired_gpu = True
        gpu_ids = list(gpu_ids)
            
        try:
            # Create per-trial output directory for artifacts (compose file, logs, etc.)
            output_root = Path(output_dir).resolve()
            trial_dir_name = _sanitize_identifier(trial_name, fallback="trial")
            trial_output_dir = output_root / trial_dir_name
            trial_output_dir.mkdir(parents=True, exist_ok=True)

            # Unique compose project name to avoid cross-trial collisions under concurrency.
            compose_project_name = _make_compose_project_name(trial_name=trial_name, gpu_ids=gpu_ids)

            # Ensure templating sees the selected GPU without mutating shared trial data.
            trial_for_compose = dict(trial)
            if len(gpu_ids) == 1:
                trial_for_compose["gpu_rank"] = gpu_ids[0]

            # Modify compose data for this trial without cross-thread mutation.
            if modify_compose_fn:
                logger.info(f"Calling modify_compose_fn for trial: {trial_for_compose}")
                template_copy = copy.deepcopy(template_data)
                try:
                    sig = inspect.signature(modify_compose_fn)
                    params = list(sig.parameters.values())
                    accepts_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
                    positional = [
                        p
                        for p in params
                        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    ]
                    if accepts_varargs or len(positional) >= 3:
                        compose_data = modify_compose_fn(template_copy, trial_for_compose, trial_output_dir)
                    else:
                        compose_data = modify_compose_fn(template_copy, trial_for_compose)
                except (TypeError, ValueError):
                    compose_data = modify_compose_fn(template_copy, trial_for_compose)
            else:
                compose_data = copy.deepcopy(template_data)

            # Update docker-compose data with GPU assignment.
            services = compose_data.setdefault("services", {})
            service = services.setdefault(_DOCKER_SERVICE_NAME, {})
            service["environment"] = _normalize_environment_for_gpu(
                service.get("environment"),
                gpu_ids=gpu_ids,
            )
            _pin_service_to_gpu_ids(service, gpu_ids=gpu_ids)

            # Write compose file to per-trial output dir so cleanup can always target the right project.
            compose_path = trial_output_dir / f"docker-compose.{compose_project_name}.yml"
            with open(compose_path, "w") as f:
                yaml.safe_dump(compose_data, f, default_flow_style=False)

            # Build docker command.
            docker_cmd = [
                "docker",
                "compose",
                "--project-name",
                compose_project_name,
                "-f",
                str(compose_path),
                "up",
                "--abort-on-container-exit",
                "--timeout",
                str(timeout),
            ]
            
            # Execute trial with proper UID/GID
            logger.info(f"Starting trial {trial_name} on GPUs {gpu_ids}")

            self._emit_status(
                trial_name,
                {
                    "status": "running",
                    "gpu_ids": gpu_ids,
                    "started_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
                },
            )
            
            # Set environment with current user's UID/GID
            env = os.environ.copy()
            env['UID'] = str(os.getuid())
            env['GID'] = str(os.getgid())
            env['CUDA_VISIBLE_DEVICES'] = ",".join(str(gid) for gid in gpu_ids)
            
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(trial_output_dir),
                env=env
            )
            
            # Track active process
            self.active_trials[trial_name] = {
                'process': process,
                'gpu_ids': gpu_ids,
                'compose_project_name': compose_project_name,
                'compose_file': str(compose_path),
                'start_time': start_time
            }
            
            # Wait for completion
            stdout, stderr = process.communicate(timeout=timeout)
            returncode = process.returncode

            experiment_path = None
            if stdout:
                experiment_path = _extract_experiment_path(stdout.splitlines())

            init_timings = {}
            if stdout:
                init_payloads = extract_init_timing_payloads(stdout.splitlines())
                init_timings = summarize_init_timings(init_payloads)
            
            # Process results
            result = {
                'name': trial_name,
                'status': 'completed' if returncode == 0 else 'error',
                'returncode': returncode,
                'gpu_ids': gpu_ids,
                'elapsed_time': time.time() - start_time,
                'stdout': stdout[-5000:] if stdout else '',  # Last 5000 chars
                'stderr': stderr[-5000:] if stderr else '',   # Last 5000 chars
                'compose_file': str(compose_path) if compose_path is not None else None,
                'trial_output_dir': str(trial_output_dir) if trial_output_dir is not None else None,
                'experiment_path': experiment_path,
            }
            if init_timings:
                result["init_timings"] = init_timings
            
            # Capture debug logs if requested
            if capture_debug_logs and returncode != 0:
                debug_log = self._capture_debug_log(trial, output_dir)
                if debug_log:
                    result['debug_log'] = debug_log
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Trial {trial_name} exceeded timeout of {timeout}s, terminating...")
            
            # Try graceful termination first
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    process.kill()
                    process.wait()
            
            result = {
                'name': trial_name,
                'status': 'timeout',
                'gpu_ids': gpu_ids,
                'elapsed_time': timeout,
                'error': f'Trial exceeded timeout of {timeout}s'
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Error running trial {trial_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            result = {
                'name': trial_name,
                'status': 'error',
                'gpu_ids': gpu_ids,
                'elapsed_time': time.time() - start_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
        finally:
            # Clean up active trial tracking
            if trial_name in self.active_trials:
                del self.active_trials[trial_name]
                
            # docker compose cleanup (best-effort), using the same project name + compose file used for 'up'.
            if compose_project_name and compose_path and compose_path.exists():
                self._cleanup_docker(compose_project_name, compose_path)
                # Intentionally keep the per-trial compose file. The top-level
                # profiling runner uses it to resolve host volume mappings (e.g.
                # /modelWorkshop -> /datasets/modelWorkshop) and then parse
                # throughput/VRAM/batch size signals from debug logs.

            # Release GPU back to the pool *after* docker cleanup so a newly
            # scheduled trial can't overlap with a still-tearing-down container.
            if acquired_gpu:
                self.gpu_pool.release_gpus(gpu_ids)
            
        if result is None:
            return {
                "name": trial_name,
                "status": "error",
                "gpu_ids": gpu_ids,
                "elapsed_time": time.time() - start_time,
                "error": "Unknown error (no result produced)",
            }

        self._emit_status(
            trial_name,
            {
                "status": result.get("status", "error"),
                "gpu_ids": gpu_ids,
                "completed_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
                "returncode": result.get("returncode"),
                "error": result.get("error"),
            },
        )

        return result
        
    def run_trials_concurrent(self,
                            trials: List[Dict[str, Any]],
                            template_data: Dict[str, Any],
                            output_dir: str,
                            timeout: int,
                            capture_debug_logs: bool = False,
                            modify_compose_fn=None) -> List[Dict[str, Any]]:
        """
        Run multiple trials concurrently across available GPUs.
        
        Args:
            trials: List of trial configurations
            template_data: Docker compose template data
            output_dir: Output directory for results
            timeout: Timeout per trial in seconds
            capture_debug_logs: Whether to capture debug logs
            modify_compose_fn: Function to modify compose data for each trial
            
        Returns:
            List of result dictionaries
        """
        self.results = []
        self.completed_count = 0
        self.total_count = len(trials)
        
        futures = []
        
        # Submit trials with staggered starts
        for i, trial in enumerate(trials):
            # Apply stagger delay between submissions
            if i > 0:
                time.sleep(self.stagger_delay)
                
            # Determine GPU assignment
            gpu_ids = trial.get('gpu_ids')
            required_gpus = trial.get('required_gpus', 1)

            if gpu_ids is None and trial.get('gpu_rank') is not None and required_gpus == 1:
                gpu_ids = [trial.get('gpu_rank')]
            
            # Submit trial
            future = self.executor.submit(
                self.run_trial_on_gpu,
                trial,
                template_data,
                output_dir,
                timeout,
                capture_debug_logs,
                gpu_ids,
                required_gpus,
                modify_compose_fn,
            )
            futures.append(future)
            
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                self.results.append(result)
                self.completed_count += 1
                
                # Log progress
                logger.info(f"Trial {result['name']} completed ({self.completed_count}/{self.total_count})")
                logger.info(f"Progress: {self.completed_count}/{self.total_count} trials "
                          f"({100*self.completed_count/self.total_count:.1f}%)")
                logger.info(
                    f"GPU utilization: {self.gpu_pool.available_count()}/{self.gpu_pool.total_gpus} available"
                )
                
            except Exception as e:
                logger.error(f"Error collecting trial result: {e}")
                self.completed_count += 1
                
        return self.results
        
    def _capture_debug_log(self, trial: Dict[str, Any], output_dir: str) -> str:
        """
        Capture debug log from failed trial.
        
        Args:
            trial: Trial configuration
            output_dir: Output directory
            
        Returns:
            Debug log content or error message
        """
        # Placeholder for backward compatibility; debug log capture is handled
        # by the profiling runner after resolving host paths from compose files.
        return ""
        
    def _cleanup_docker(self, compose_project_name: str, compose_path: Path) -> None:
        """
        Clean up docker-compose resources for a trial.

        Args:
            compose_project_name: docker compose project name (must match the 'up' invocation)
            compose_path: compose file path (must match the 'up' invocation)
        """
        try:
            subprocess.run(
                [
                    "docker",
                    "compose",
                    "--project-name",
                    compose_project_name,
                    "-f",
                    str(compose_path),
                    "down",
                    "--remove-orphans",
                    "--timeout",
                    "5",
                ],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass  # Best effort cleanup

    def _force_cleanup_docker(self, compose_project_name: str, compose_path: Path) -> None:
        """
        Force cleanup docker-compose resources for a trial (best-effort).

        This intentionally avoids killing arbitrary host PIDs. Any such behavior
        must be explicitly added behind a very strict guard.
        """
        try:
            self._cleanup_docker(compose_project_name, compose_path)
        except Exception as e:
            logger.error(f"Error during force cleanup: {e}")
            
    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        self.executor.shutdown(wait=True)
        
        # Force release any remaining GPU allocations
        self.gpu_pool.force_release_all()
