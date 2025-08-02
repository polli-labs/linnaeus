"""
linnaeus package

linnaeus is a codebase for training MetaFormer models on taxonomic classification
tasks with strong metadata enrichment capabilities.
"""

# Apply thread control settings as early as possible
# This must happen before any heavy imports to be effective
from linnaeus.utils.thread_ctrl import apply_thread_settings

apply_thread_settings()
