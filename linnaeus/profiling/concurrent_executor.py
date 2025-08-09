"""
Concurrent Trial Executor for parallel profiling execution.

Executes profiling trials concurrently across multiple GPUs with proper
isolation, error handling, and resource management.
"""

import concurrent.futures
import subprocess
import tempfile
import json
import os
import time
import logging
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil

from .gpu_pool import GPUPoolManager

logger = logging.getLogger(__name__)


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
                 stagger_delay: float = 5.0):
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
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.results = []
        self.active_trials = {}
        self.completed_count = 0
        self.total_count = 0
        
    def run_trial_on_gpu(self,
                        trial: Dict[str, Any],
                        template_data: Dict[str, Any],
                        output_dir: str,
                        timeout: int,
                        capture_debug_logs: bool = False,
                        gpu_id: Optional[int] = None,
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
        trial_name = trial['name']
        start_time = time.time()
        
        # Acquire GPU if not specified
        acquired_gpu = False
        if gpu_id is None:
            gpu_id = self.gpu_pool.acquire_gpu(trial_name, timeout=3600)
            if gpu_id is None:
                return {
                    'name': trial_name,
                    'status': 'timeout',
                    'error': 'Failed to acquire GPU within timeout'
                }
            acquired_gpu = True
            
        try:
            # Modify compose data for this trial
            if modify_compose_fn:
                logger.info(f"Calling modify_compose_fn for trial: {trial}")
                compose_data = modify_compose_fn(template_data, trial)
            else:
                compose_data = template_data.copy()
            
            # Update docker-compose data with GPU assignment
            service = compose_data['services'].get('linnaeus-training', {})
            if 'environment' not in service:
                service['environment'] = []
            
            # Ensure GPU assignment in environment
            gpu_env_set = False
            for i, env in enumerate(service.get('environment', [])):
                if isinstance(env, str) and env.startswith('CUDA_VISIBLE_DEVICES='):
                    service['environment'][i] = f'CUDA_VISIBLE_DEVICES={gpu_id}'
                    gpu_env_set = True
                    break
            if not gpu_env_set:
                service['environment'].append(f'CUDA_VISIBLE_DEVICES={gpu_id}')
            
            # Write compose file to temp location
            compose_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix=f'_{trial_name}_gpu{gpu_id}.yml',
                prefix='docker-compose-',
                dir='/tmp',
                delete=False
            )
            yaml.dump(compose_data, compose_file)
            compose_file.close()
            
            # Build docker command
            docker_cmd = [
                'docker', 'compose',
                '-f', compose_file.name,
                'up',
                '--abort-on-container-exit',
                '--timeout', str(timeout)
            ]
            
            # Execute trial with proper UID/GID
            logger.info(f"Starting trial {trial_name} on GPU {gpu_id}")
            
            # Set environment with current user's UID/GID
            env = os.environ.copy()
            env['UID'] = str(os.getuid())
            env['GID'] = str(os.getgid())
            
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=output_dir,
                env=env
            )
            
            # Track active process
            self.active_trials[trial_name] = {
                'process': process,
                'gpu_id': gpu_id,
                'start_time': start_time
            }
            
            # Wait for completion
            stdout, stderr = process.communicate(timeout=timeout)
            returncode = process.returncode
            
            # Process results
            result = {
                'name': trial_name,
                'status': 'completed' if returncode == 0 else 'error',
                'returncode': returncode,
                'gpu_id': gpu_id,
                'elapsed_time': time.time() - start_time,
                'stdout': stdout[-5000:] if stdout else '',  # Last 5000 chars
                'stderr': stderr[-5000:] if stderr else ''   # Last 5000 chars
            }
            
            # Capture debug logs if requested
            if capture_debug_logs and returncode != 0:
                result['debug_log'] = self._capture_debug_log(trial, output_dir)
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Trial {trial_name} exceeded timeout of {timeout}s, terminating...")
            
            # Try graceful termination first
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                process.kill()
                process.wait()
            
            # Clean up Docker containers explicitly
            self._force_cleanup_docker(trial_name, gpu_id, output_dir)
            
            result = {
                'name': trial_name,
                'status': 'timeout',
                'gpu_id': gpu_id,
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
                'gpu_id': gpu_id,
                'elapsed_time': time.time() - start_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
        finally:
            # Release GPU if we acquired it
            if acquired_gpu:
                self.gpu_pool.release_gpu(gpu_id)
                
            # Clean up active trial tracking
            if trial_name in self.active_trials:
                del self.active_trials[trial_name]
                
            # Clean up compose file
            if 'compose_file' in locals():
                try:
                    os.unlink(compose_file.name)
                except:
                    pass
                    
            # Docker cleanup
            self._cleanup_docker(trial_name, gpu_id)
            
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
            gpu_id = trial.get('gpu_rank')  # Manual assignment if specified
            
            # Submit trial
            future = self.executor.submit(
                self.run_trial_on_gpu,
                trial,
                template_data,
                output_dir,
                timeout,
                capture_debug_logs,
                gpu_id,
                modify_compose_fn
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
                logger.info(f"GPU utilization: {self.gpu_pool.available_gpus.qsize()}/{self.gpu_pool.total_gpus} available")
                
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
        # This would need to be implemented based on where linnaeus saves logs
        # For now, return placeholder
        return "Debug log capture not implemented"
        
    def _cleanup_docker(self, trial_name: str, gpu_id: int) -> None:
        """
        Clean up Docker containers and resources.
        
        Args:
            trial_name: Name of trial
            gpu_id: GPU ID used
        """
        try:
            # Stop and remove containers
            container_name = f"linnaeus-training-{trial_name}-gpu{gpu_id}"
            subprocess.run(
                ['docker', 'stop', container_name],
                capture_output=True,
                timeout=10
            )
            subprocess.run(
                ['docker', 'rm', container_name],
                capture_output=True,
                timeout=10
            )
        except:
            pass  # Best effort cleanup
            
    def _force_cleanup_docker(self, trial_name: str, gpu_id: int, output_dir: str) -> None:
        """
        Force cleanup Docker containers and GPU processes.
        
        Args:
            trial_name: Name of trial
            gpu_id: GPU ID used
            output_dir: Output directory (for compose project name)
        """
        try:
            # Get project name from output dir
            project_name = os.path.basename(output_dir)
            
            # Force stop all containers for this project
            subprocess.run(
                ['docker', 'compose', '-p', project_name, 'down', '--timeout', '5'],
                capture_output=True,
                timeout=10
            )
            
            # Kill any remaining GPU processes on this GPU
            # Get processes using this GPU
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader', f'-i', str(gpu_id)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        try:
                            # Only kill if we own the process
                            subprocess.run(['kill', '-9', pid], capture_output=True, timeout=2)
                        except:
                            logger.warning(f"Could not kill GPU process {pid} on GPU {gpu_id}")
                            
        except Exception as e:
            logger.error(f"Error during force cleanup: {e}")
            
    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        self.executor.shutdown(wait=True)
        
        # Force release any remaining GPU allocations
        self.gpu_pool.force_release_all()