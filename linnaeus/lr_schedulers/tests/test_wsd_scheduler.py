import unittest
from unittest.mock import MagicMock

import torch
from torch.optim import SGD

# Assuming the schedulers are importable like this. Adjust if necessary.
from linnaeus.lr_schedulers.schedulers.stable_decay_scheduler import StableDecayScheduler
from linnaeus.lr_schedulers.schedulers.warmup_lr_scheduler import WarmupLRScheduler


# Basic model and optimizer for testing
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(1))

    def forward(self, x):
        return self.param * x


class TestWSDScheduler(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.initial_lr = 0.1
        self.optimizer = SGD(self.model.parameters(), lr=self.initial_lr)

    def test_01_stable_decay_scheduler_initialization(self):
        """Test that StableDecayScheduler initializes correctly (Fixes Issue 1)."""
        try:
            scheduler = StableDecayScheduler(
                optimizer=self.optimizer,
                stable_steps=100,
                decay_steps=100,
                stable_lr=0.05,
                min_lr=0.001,
                verbose=False,  # Test with False
            )
            self.assertIsNotNone(scheduler)
            scheduler = StableDecayScheduler(
                optimizer=self.optimizer,
                stable_steps=100,
                decay_steps=100,
                stable_lr=0.05,
                min_lr=0.001,
                verbose=True,  # Test with True
            )
            self.assertIsNotNone(scheduler)
        except TypeError as e:
            self.fail(f"StableDecayScheduler initialization failed with TypeError: {e}")
        except Exception as e:
            self.fail(f"StableDecayScheduler initialization failed with an unexpected exception: {e}")

    def test_02_warmup_stable_decay_step_counting_and_lr(self):
        """Test correct step counting and LR during stable phase (Fixes Issue 2)."""
        warmup_steps = 50
        warmup_lr_init = 0.01
        stable_lr_sds = 0.05  # Target LR for stable phase in SDS
        min_lr_sds = 0.001
        stable_steps_sds = 100
        decay_steps_sds = 100

        sds = StableDecayScheduler(
            optimizer=self.optimizer,
            stable_steps=stable_steps_sds,
            decay_steps=decay_steps_sds,
            stable_lr=stable_lr_sds,
            min_lr=min_lr_sds,
            verbose=False,  # Verbosity for SDS itself
        )

        # Mock WarmupLRScheduler's logger to check verbosity later if needed
        # For now, we primarily test LR values.
        # WarmupLRScheduler.logger = MagicMock() # Example if we wanted to mock its logger

        wlr_scheduler = WarmupLRScheduler(
            optimizer=self.optimizer, warmup_steps=warmup_steps, warmup_lr_init=warmup_lr_init, base_scheduler=sds
        )

        # Simulate steps
        # Phase 1: Warmup
        for i in range(warmup_steps):
            wlr_scheduler.step_update(i)
            current_lr = self.optimizer.param_groups[0]["lr"]
            expected_lr = warmup_lr_init + (stable_lr_sds - warmup_lr_init) * (i / warmup_steps)
            self.assertAlmostEqual(current_lr, expected_lr, places=6, msg=f"Warmup LR mismatch at step {i}")

        # Phase 2: Transition to StableDecayScheduler's stable phase
        # At this point, WarmupLRScheduler should pass step 0 to StableDecayScheduler
        # Test a few steps into the stable phase
        for i in range(5):  # Test 5 steps into stable phase
            global_step = warmup_steps + i
            wlr_scheduler.step_update(global_step)
            current_lr = self.optimizer.param_groups[0]["lr"]
            # SDS's get_lr() uses self.last_epoch + 1.
            # If step_update(0) was called, last_epoch becomes 0, so get_lr uses step 1.
            # The LR should be stable_lr_sds
            self.assertAlmostEqual(
                current_lr, stable_lr_sds, places=6, msg=f"Stable phase LR mismatch at global_step {global_step} (SDS step {i})"
            )

        # Check SDS internal state (optional, but good for debugging)
        self.assertEqual(sds.last_epoch, 4)  # After 5 steps (0 to 4) passed to SDS

        # Phase 3: Further into stable phase
        stable_phase_step_10 = 10  # 10th step *within* SDS's stable phase
        global_step_stable_10 = warmup_steps + stable_phase_step_10
        wlr_scheduler.step_update(global_step_stable_10)
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(
            current_lr,
            stable_lr_sds,
            places=6,
            msg=f"Stable phase LR mismatch at global_step {global_step_stable_10} (SDS step {stable_phase_step_10})",
        )
        self.assertEqual(sds.last_epoch, stable_phase_step_10)

    def test_03_logging_verbosity(self):
        """Test that StableDecayScheduler does not log when managed by WarmupLRScheduler if WarmupLRScheduler handles logging."""
        # This test relies on StableDecayScheduler.print_lr being removed from its step_update.
        # We will set verbose=True for StableDecayScheduler but expect no output from it directly
        # if WarmupLRScheduler is supposed to handle it.

        warmup_steps = 10
        stable_lr_sds = 0.05

        # Mock the print_lr method of StableDecayScheduler to ensure it's NOT called
        # after the fix (where it's removed from step_update).
        # If it were still present and called, this mock would register the call.
        # However, since we removed it, this test is more about ensuring that
        # setting verbose=True on SDS doesn't cause errors and that the structure is clean.

        sds = StableDecayScheduler(
            optimizer=self.optimizer,
            stable_steps=20,
            decay_steps=20,
            stable_lr=stable_lr_sds,
            min_lr=0.001,
            verbose=True,  # SDS verbose is true
        )
        sds.print_lr = MagicMock()  # Mock its print_lr specifically

        wlr_scheduler = WarmupLRScheduler(optimizer=self.optimizer, warmup_steps=warmup_steps, warmup_lr_init=0.01, base_scheduler=sds)

        # Simulate steps past warmup
        for i in range(warmup_steps + 5):  # Go 5 steps into SDS phase
            # We need to ensure WarmupLRScheduler's own logging is also handled.
            # The ticket implies WarmupLRScheduler is the source of truth for logging.
            # For this test, we're primarily concerned that SDS.print_lr isn't the one firing.
            # We can simulate the config for WarmupLRScheduler's logger:
            if not hasattr(self.optimizer.param_groups[0], "config"):
                self.optimizer.param_groups[0]["config"] = {"DEBUG": {"SCHEDULING": True}}

            wlr_scheduler.step_update(i)

        # Assert that StableDecayScheduler's specific print_lr was NOT called by its step_update
        # because it should have been removed.
        sds.print_lr.assert_not_called()

        # To actually check WarmupLRScheduler's logging, we would need to mock its logger
        # e.g. linnaeus.lr_schedulers.schedulers.warmup_lr_scheduler.logger.debug = MagicMock()
        # and then check `...logger.debug.assert_called()`
        # This part is more involved and depends on how WarmupLRScheduler's logger is accessed.
        # The critical part of the ticket (removing SDS.print_lr) is covered by it not being called.


if __name__ == "__main__":
    unittest.main()
