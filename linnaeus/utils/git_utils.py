"""
Git utilities for checkpoint metadata and validation.
"""

import subprocess
from typing import Any


def get_git_info() -> dict[str, Any]:
    """
    Get current git information for checkpoint metadata.
    
    Returns:
        Dictionary containing git branch, commit hash, and dirty status.
    """
    git_info = {
        "branch": "unknown",
        "commit": "unknown",
        "dirty": False
    }
    
    try:
        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()
        
        # Check if working directory is dirty
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            check=False
        )
        if result.returncode != 0:
            git_info["dirty"] = True
        
        # Also check for untracked files
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            git_info["dirty"] = True
            
    except (subprocess.SubprocessError, FileNotFoundError):
        # Git not available or not in a git repo
        pass
    
    return git_info


def get_git_branch() -> str:
    """Get current git branch name."""
    return get_git_info()["branch"]


def get_git_commit() -> str:
    """Get current git commit hash."""
    return get_git_info()["commit"]