import unittest
from unittest.mock import patch, MagicMock, mock_open
import os

from yacs.config import CfgNode as CN

# Assuming linnaeus is in PYTHONPATH or adjust path accordingly
from linnaeus.utils.logging.wandb import initialize_wandb

# Helper function to create a mock config
def _get_mock_config(**kwargs) -> CN:
    """Creates a mock CfgNode for testing initialize_wandb."""
    config = CN()
    config.EXPERIMENT = CN()
    config.EXPERIMENT.WANDB = CN()
    config.EXPERIMENT.PROJECT = "test_project"
    config.EXPERIMENT.GROUP = "test_group"
    config.EXPERIMENT.NAME = "test_name"
    config.EXPERIMENT.TAGS = ["tag1", "tag2"]
    config.EXPERIMENT.NOTES = "Test notes"

    # Defaults that can be overridden by kwargs
    config.EXPERIMENT.WANDB.ENABLED = True
    config.EXPERIMENT.WANDB.KEY = None # No API key by default
    config.EXPERIMENT.WANDB.RUN_ID = ""
    config.EXPERIMENT.WANDB.RESUME = False # Corresponds to manual resume flag

    config.LOADING_FROM_CHECKPOINT = False # Indicates auto-resume if RUN_ID is present

    config.ENV = CN()
    config.ENV.OUTPUT = CN()
    config.ENV.OUTPUT.DIRS = CN()
    config.ENV.OUTPUT.DIRS.LOGS = "/tmp/test_logs" # For JSONL

    # Apply overrides
    for key_path, value in kwargs.items():
        keys = key_path.split('.')
        cfg_ptr = config
        for k_idx, k in enumerate(keys):
            if k_idx == len(keys) - 1:
                if isinstance(cfg_ptr, CN) and k in cfg_ptr:
                    cfg_ptr[k] = value
                elif hasattr(cfg_ptr, k): # For top-level like LOADING_FROM_CHECKPOINT
                     setattr(cfg_ptr, k, value)
                else: # If key doesn't exist, create it (e.g. RUN_ID under WANDB)
                    if isinstance(cfg_ptr, CN):
                        cfg_ptr[k] = value
                    else:
                        # This case should ideally be handled by ensuring CfgNode structure upfront
                        # For simplicity, we'll assume paths like EXPERIMENT.WANDB.X are valid
                        # If not, the CfgNode structure needs more detailed setup for arbitrary keys
                        pass
            else:
                if isinstance(cfg_ptr, CN) and k in cfg_ptr:
                    cfg_ptr = cfg_ptr[k]
                elif hasattr(cfg_ptr, k):
                    cfg_ptr = getattr(cfg_ptr, k)
                else:
                    # This indicates a path not predefined, error or more complex setup needed
                    raise AttributeError(f"Config path {key_path} not fully defined in mock setup for key {k}")
    return config

@patch('linnaeus.utils.logging.wandb.construct_wandb_config', return_value={'config_key': 'config_value'})
@patch('linnaeus.utils.logging.wandb.get_rank_safely')
@patch('linnaeus.utils.logging.wandb.wandb')
@patch('os.makedirs') # Mock os.makedirs for JSONL
@patch('builtins.open', new_callable=mock_open) # Mock open for JSONL
class TestInitializeWandb(unittest.TestCase):

    def setUp(self):
        """Reset global state in the tested module before each test."""
        # Reset JSONL handler to ensure JSONL setup runs if rank is 0
        # This is important because _jsonl_file_handler is a global variable
        # in the wandb module and can persist state between tests.
        if hasattr(initialize_wandb, '__globals__'):
            # This is a bit of a hack to access globals of the module
            # where initialize_wandb is defined. A cleaner way might involve
            # a reset function in the module itself if this becomes common.
            wandb_module_globals = initialize_wandb.__globals__
            wandb_module_globals['_jsonl_file_handler'] = None
            wandb_module_globals['_jsonl_filepath'] = None
            wandb_module_globals['_jsonl_lock'] = None
        else:
            # Fallback or warning if __globals__ isn't found, though it should be for a function
            print("Warning: Could not reset _jsonl_file_handler global state.")


    def test_wandb_disabled(self, mock_open_func, mock_makedirs, mock_wandb, mock_get_rank_safely, mock_construct_config):
        """Test that wandb.init is not called if WANDB.ENABLED is False."""
        mock_config = _get_mock_config(**{"EXPERIMENT.WANDB.ENABLED": False})
        mock_model = MagicMock()
        mock_dataset_metadata = MagicMock()

        initialize_wandb(mock_config, mock_model, mock_dataset_metadata)

        mock_wandb.init.assert_not_called()
        mock_makedirs.assert_not_called() # JSONL setup should also be skipped
        mock_open_func.assert_not_called()

    def test_new_run_rank_0(self, mock_open_func, mock_makedirs, mock_wandb, mock_get_rank_safely, mock_construct_config):
        """Test new run on rank 0: mode=online, resume=False."""
        mock_get_rank_safely.return_value = 0
        mock_config = _get_mock_config(
            **{
                "EXPERIMENT.WANDB.RUN_ID": "",
                "LOADING_FROM_CHECKPOINT": False,
                "EXPERIMENT.WANDB.RESUME": False,
            }
        )
        mock_model = MagicMock()
        mock_dataset_metadata = MagicMock()

        initialize_wandb(mock_config, mock_model, mock_dataset_metadata)

        mock_wandb.init.assert_called_once_with(
            project="test_project",
            group="test_group",
            name="test_name",
            tags=["tag1", "tag2"],
            notes="Test notes",
            config={'config_key': 'config_value'},
            mode="online",
            resume=False # For a new run
        )
        mock_makedirs.assert_called_once_with("/tmp/test_logs", exist_ok=True)
        mock_open_func.assert_called_once_with("/tmp/test_logs/metrics_log.jsonl", "a", encoding="utf-8")


    def test_new_run_rank_1(self, mock_open_func, mock_makedirs, mock_wandb, mock_get_rank_safely, mock_construct_config):
        """Test new run on rank 1: mode=offline, resume=False."""
        mock_get_rank_safely.return_value = 1
        mock_config = _get_mock_config(
            **{
                "EXPERIMENT.WANDB.RUN_ID": "",
                "LOADING_FROM_CHECKPOINT": False,
                "EXPERIMENT.WANDB.RESUME": False,
            }
        )
        mock_model = MagicMock()
        mock_dataset_metadata = MagicMock()

        initialize_wandb(mock_config, mock_model, mock_dataset_metadata)

        mock_wandb.init.assert_called_once_with(
            project="test_project",
            group="test_group",
            name="test_name",
            tags=["tag1", "tag2"],
            notes="Test notes",
            config={'config_key': 'config_value'},
            mode="offline",
            resume=False
        )
        # JSONL logging happens only on rank 0
        mock_makedirs.assert_not_called()
        mock_open_func.assert_not_called()

    def test_auto_resume_rank_0(self, mock_open_func, mock_makedirs, mock_wandb, mock_get_rank_safely, mock_construct_config):
        """Test auto-resume on rank 0: mode=online, resume=must."""
        mock_get_rank_safely.return_value = 0
        mock_config = _get_mock_config(
            **{
                "EXPERIMENT.WANDB.RUN_ID": "test_run_id_123",
                "LOADING_FROM_CHECKPOINT": True, # Auto-resume trigger
                "EXPERIMENT.WANDB.RESUME": False, # Manual resume is off, but auto should take precedence
            }
        )
        mock_model = MagicMock()
        mock_dataset_metadata = MagicMock()

        initialize_wandb(mock_config, mock_model, mock_dataset_metadata)

        mock_wandb.init.assert_called_once_with(
            project="test_project",
            group="test_group",
            name="test_name",
            tags=["tag1", "tag2"],
            notes="Test notes",
            config={'config_key': 'config_value'},
            id="test_run_id_123",
            mode="online",
            resume="must"
        )

    def test_auto_resume_rank_1(self, mock_open_func, mock_makedirs, mock_wandb, mock_get_rank_safely, mock_construct_config):
        """Test auto-resume on rank 1: mode=offline, resume=must."""
        mock_get_rank_safely.return_value = 1
        mock_config = _get_mock_config(
            **{
                "EXPERIMENT.WANDB.RUN_ID": "test_run_id_456",
                "LOADING_FROM_CHECKPOINT": True, # Auto-resume trigger
            }
        )
        mock_model = MagicMock()
        mock_dataset_metadata = MagicMock()

        initialize_wandb(mock_config, mock_model, mock_dataset_metadata)

        mock_wandb.init.assert_called_once_with(
            project="test_project",
            group="test_group",
            name="test_name",
            tags=["tag1", "tag2"],
            notes="Test notes",
            config={'config_key': 'config_value'},
            id="test_run_id_456",
            mode="offline", # Rank 1 is always offline unless resume + rank 0
            resume="must"
        )

    def test_manual_resume_rank_0(self, mock_open_func, mock_makedirs, mock_wandb, mock_get_rank_safely, mock_construct_config):
        """Test manual-resume on rank 0: mode=online, resume=must."""
        mock_get_rank_safely.return_value = 0
        mock_config = _get_mock_config(
            **{
                "EXPERIMENT.WANDB.RUN_ID": "test_run_id_789",
                "LOADING_FROM_CHECKPOINT": False, # Auto-resume is off
                "EXPERIMENT.WANDB.RESUME": True, # Manual resume is on
            }
        )
        mock_model = MagicMock()
        mock_dataset_metadata = MagicMock()

        initialize_wandb(mock_config, mock_model, mock_dataset_metadata)

        mock_wandb.init.assert_called_once_with(
            project="test_project",
            group="test_group",
            name="test_name",
            tags=["tag1", "tag2"],
            notes="Test notes",
            config={'config_key': 'config_value'},
            id="test_run_id_789",
            mode="online",
            resume="must"
        )

    def test_manual_resume_rank_1(self, mock_open_func, mock_makedirs, mock_wandb, mock_get_rank_safely, mock_construct_config):
        """Test manual-resume on rank 1: mode=offline, resume=must."""
        mock_get_rank_safely.return_value = 1
        mock_config = _get_mock_config(
            **{
                "EXPERIMENT.WANDB.RUN_ID": "test_run_id_abc",
                "LOADING_FROM_CHECKPOINT": False,
                "EXPERIMENT.WANDB.RESUME": True,
            }
        )
        mock_model = MagicMock()
        mock_dataset_metadata = MagicMock()

        initialize_wandb(mock_config, mock_model, mock_dataset_metadata)

        mock_wandb.init.assert_called_once_with(
            project="test_project",
            group="test_group",
            name="test_name",
            tags=["tag1", "tag2"],
            notes="Test notes",
            config={'config_key': 'config_value'},
            id="test_run_id_abc",
            mode="offline", # Rank 1 is always offline
            resume="must"
        )

    def test_new_run_with_id_no_resume_rank_0(self, mock_open_func, mock_makedirs, mock_wandb, mock_get_rank_safely, mock_construct_config):
        """Test new run with ID but no resume flags on rank 0: mode=online, resume=False."""
        mock_get_rank_safely.return_value = 0
        mock_config = _get_mock_config(
            **{
                "EXPERIMENT.WANDB.RUN_ID": "test_run_id_def",
                "LOADING_FROM_CHECKPOINT": False,
                "EXPERIMENT.WANDB.RESUME": False,
            }
        )
        mock_model = MagicMock()
        mock_dataset_metadata = MagicMock()

        initialize_wandb(mock_config, mock_model, mock_dataset_metadata)

        mock_wandb.init.assert_called_once_with(
            project="test_project",
            group="test_group",
            name="test_name",
            tags=["tag1", "tag2"],
            notes="Test notes",
            config={'config_key': 'config_value'},
            id="test_run_id_def",
            mode="online",
            resume=False # Explicitly not resuming
        )

    def test_new_run_with_id_no_resume_rank_1(self, mock_open_func, mock_makedirs, mock_wandb, mock_get_rank_safely, mock_construct_config):
        """Test new run with ID but no resume flags on rank 1: mode=offline, resume=False."""
        mock_get_rank_safely.return_value = 1
        mock_config = _get_mock_config(
            **{
                "EXPERIMENT.WANDB.RUN_ID": "test_run_id_ghi",
                "LOADING_FROM_CHECKPOINT": False,
                "EXPERIMENT.WANDB.RESUME": False,
            }
        )
        mock_model = MagicMock()
        mock_dataset_metadata = MagicMock()

        initialize_wandb(mock_config, mock_model, mock_dataset_metadata)

        mock_wandb.init.assert_called_once_with(
            project="test_project",
            group="test_group",
            name="test_name",
            tags=["tag1", "tag2"],
            notes="Test notes",
            config={'config_key': 'config_value'},
            id="test_run_id_ghi",
            mode="offline",
            resume=False # Explicitly not resuming
        )

if __name__ == '__main__':
    unittest.main()
