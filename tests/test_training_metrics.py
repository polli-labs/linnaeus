import unittest
import torch
from collections import defaultdict

from linnaeus.utils.metrics.tracker import MetricsTracker, Metric
from linnaeus.utils.metrics.step_metrics_logger import StepMetricsLogger
from linnaeus.ops_schedule.ops_schedule import OpsSchedule # May need a mock or simple config

# Mock config
class MockConfig:
    def __init__(self):
        self.DATA = type('DATA', (object,), {'TASK_KEYS_H5': ['taxa_L10']})
        self.METRICS = type('METRICS', (object,), {
            'TRACK_NULL_VS_NON_NULL': False,
            'NULL_VS_NON_NULL_TASKS': []
        })
        self.EXPERIMENT = type('EXPERIMENT', (object,), {
            'WANDB': type('WANDB', (object,), {'ENABLED': False})
        })
        self.SCHEDULE = type('SCHEDULE', (object,), {
            'METRICS': type('METRICS', (object,), {
                'CONSOLE_INTERVAL': 10,
                'WANDB_INTERVAL': 10,
                'PIPELINE_INTERVAL': 100,
                'LR_INTERVAL': 100,
            })
        })
        self.DEBUG = type('DEBUG', (object,), {
            'VALIDATION_METRICS': False,
            'WANDB_METRICS': False,
        })


class TestTrainingAccuracyMetrics(unittest.TestCase):

    def setUp(self):
        self.config = MockConfig()
        # OpsSchedule might need more elaborate mocking or a simple real instance
        self.ops_schedule = OpsSchedule(self.config, metrics_tracker=None)
        self.metrics_tracker = MetricsTracker(config=self.config, subset_maps={})
        self.step_logger = StepMetricsLogger(config=self.config, metrics_tracker=self.metrics_tracker, ops_schedule=self.ops_schedule)

    def test_training_accuracy_logging(self):
        epoch = 0
        current_step = 0
        total_dataloader_steps = 2 # Number of batches

        # Batch 1 Data
        # Task: taxa_L10, Batch size: 2, Num classes: 5
        # Sample 1: Pred L10=[0.1, 0.8, 0.05, 0.05, 0.0], Target L10=[0,1,0,0,0] -> Correct
        # Sample 2: Pred L10=[0.7, 0.1, 0.05, 0.05, 0.1], Target L10=[0,0,0,0,1] -> Incorrect
        outputs_b1 = {
            'taxa_L10': torch.tensor([
                [0.1, 0.8, 0.05, 0.05, 0.0],
                [0.7, 0.1, 0.05, 0.05, 0.1]
            ])
        }
        targets_b1 = {
            'taxa_L10': torch.tensor([
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1]
            ], dtype=torch.float)
        }
        # Expected acc1 for taxa_L10 in batch1 = 1/2 = 50.0

        # Batch 2 Data
        # Task: taxa_L10, Batch size: 2, Num classes: 5
        # Sample 1: Pred L10=[0.0, 0.0, 0.8, 0.1, 0.1], Target L10=[0,0,1,0,0] -> Correct
        # Sample 2: Pred L10=[0.1, 0.1, 0.0, 0.8, 0.0], Target L10=[0,0,1,0,0] -> Incorrect
        outputs_b2 = {
            'taxa_L10': torch.tensor([
                [0.0, 0.0, 0.8, 0.1, 0.1],
                [0.1, 0.1, 0.0, 0.8, 0.0]
            ])
        }
        targets_b2 = {
            'taxa_L10': torch.tensor([
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ], dtype=torch.float)
        }
        # Expected acc1 for taxa_L10 in batch2 = 1/2 = 50.0

        # Total correct = 1 (from b1) + 1 (from b2) = 2
        # Total samples = 2 (from b1) + 2 (from b2) = 4
        # Expected epoch acc1 for taxa_L10 = 2/4 = 50.0

        dummy_loss_components_b1 = {'total': 1.0, 'tasks': {'taxa_L10': 1.0}}
        dummy_loss_components_b2 = {'total': 0.8, 'tasks': {'taxa_L10': 0.8}}
        empty_subset_ids = {}

        # --- Simulate Batch 1 ---
        self.metrics_tracker.update_train_batch(outputs_b1, targets_b1, dummy_loss_components_b1, empty_subset_ids)
        # Values in metrics_tracker.phase_task_metrics["train"]["taxa_L10"]["acc1"].value are still 0.0 (init_value)
        # because _finalize_phase hasn't run.
        self.step_logger.log_step_metrics(
            current_step=current_step, epoch=epoch, step_idx=0, total_steps=total_dataloader_steps,
            batch_loss_dict=dummy_loss_components_b1, lr_value=0.001
        )
        current_step += 1

        # Check what step_logger accumulated after B1 (value should be 0.0 as finalize not called yet)
        # averaged_metrics_after_b1 = self.step_logger.get_averaged_wandb_metrics()
        # self.assertEqual(averaged_metrics_after_b1.get('train/acc1_taxa_L10', 0.0), 0.0)


        # --- Simulate Batch 2 ---
        self.metrics_tracker.update_train_batch(outputs_b2, targets_b2, dummy_loss_components_b2, empty_subset_ids)
        self.step_logger.log_step_metrics(
            current_step=current_step, epoch=epoch, step_idx=1, total_steps=total_dataloader_steps,
            batch_loss_dict=dummy_loss_components_b2, lr_value=0.001
        )
        current_step += 1

        # --- Finalize Epoch ---
        # This is when metrics_tracker.phase_task_metrics["train"]["taxa_L10"]["acc1"].value gets updated
        avg_epoch_loss = (dummy_loss_components_b1['total'] + dummy_loss_components_b2['total']) / 2.0
        self.metrics_tracker.finalize_train_epoch(epoch=epoch, avg_epoch_loss=avg_epoch_loss)

        # Now, metrics_tracker.phase_task_metrics["train"]["taxa_L10"]["acc1"].value should be 50.0

        # What step_logger has accumulated should be based on the values *at the time of logging*
        # which were based on metric_obj.value being 0.0 for both steps.
        averaged_metrics = self.step_logger.get_averaged_wandb_metrics()

        self.assertIn('train/acc1_taxa_L10', averaged_metrics)
        # Given StepMetricsLogger logs metric_obj.value which is updated at finalize_epoch,
        # the individual step logs for acc1 would have used the value *before* finalize_epoch (i.e. 0.0 for the first epoch).
        # So the average of these (0.0, 0.0) is 0.0.
        self.assertEqual(averaged_metrics.get('train/acc1_taxa_L10', -1.0), 0.0)

        # To test the *actual computed accuracy* that *should* be logged if we logged *after* finalize_epoch,
        # we can directly inspect the metrics_tracker or call get_wandb_metrics on it.
        final_tracker_metrics = self.metrics_tracker.get_wandb_metrics()
        self.assertIn('train/acc1_taxa_L10', final_tracker_metrics)
        self.assertEqual(final_tracker_metrics['train/acc1_taxa_L10'], 50.0)

        # The current bug is that these are -1 or absent.
        # If this test passes with 0.0, it means the pipeline up to WandB is okay for 0.0.
        # If it's -1, then the test setup itself or the components have an issue with -1.

if __name__ == '__main__':
    unittest.main()
