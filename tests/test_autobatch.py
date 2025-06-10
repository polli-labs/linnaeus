import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("yacs")
from yacs.config import CfgNode as CN

from linnaeus.utils import autobatch

torch = pytest.importorskip("torch")


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.use_checkpoint = False
        self.train_called = False
        self.eval_called = False

    def train(self, mode: bool = True):
        self.train_called = True
        return super().train(mode)

    def eval(self):
        self.eval_called = True
        return super().eval()

    def forward(self, images, aux_info):
        return {"t": torch.zeros(images.size(0), 2, device=images.device)}


class DummyOptimizer:
    def __init__(self, *args, **kwargs):
        self.zero_grad = MagicMock()
        self.step = MagicMock()


class DummyScaled:
    def __init__(self):
        self.backward = MagicMock()


class DummyScaler:
    def __init__(self, *args, **kwargs):
        self.scaled = DummyScaled()
        self.step = MagicMock()
        self.update = MagicMock()

    def scale(self, _tensor):
        return self.scaled


DUMMY_LOSS = (torch.tensor(1.0, requires_grad=True), {}, {})


def make_cfg():
    cfg = CN()
    cfg.DATA = CN()
    cfg.DATA.IMG_SIZE = 8
    cfg.DATA.TASK_KEYS_H5 = ["t"]
    cfg.DATA.META = CN()
    cfg.DATA.META.ACTIVE = False
    cfg.MODEL = CN()
    cfg.MODEL.IN_CHANS = 3
    cfg.MODEL.NUM_CLASSES = {"t": 2}
    cfg.TRAIN = CN()
    cfg.TRAIN.AMP_OPT_LEVEL = "O0"
    cfg.TRAIN.GRADIENT_CHECKPOINTING = CN()
    cfg.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_NORMAL_STEPS = False
    cfg.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_GRADNORM_STEPS = False
    cfg.LOSS = CN()
    cfg.LOSS.GRAD_WEIGHTING = CN()
    cfg.LOSS.GRAD_WEIGHTING.TASK = CN()
    cfg.LOSS.GRAD_WEIGHTING.TASK.GRADNORM_ENABLED = False
    return cfg


def fake_param():
    p = MagicMock()
    p.device = torch.device("cuda")
    return p


@patch.object(autobatch, "_create_temporary_optimizer", return_value=DummyOptimizer())
@patch.object(autobatch.torch.cuda.amp, "GradScaler", DummyScaler)
@patch.object(autobatch, "weighted_hierarchical_loss", return_value=DUMMY_LOSS)
@patch.object(autobatch.torch.cuda, "max_memory_allocated", return_value=100 << 20)
@patch.object(autobatch.torch.cuda, "reset_peak_memory_stats")
@patch.object(autobatch.torch.cuda, "synchronize")
@patch.object(autobatch.torch.cuda, "empty_cache")
def test_run_trial_train_and_val(
    empty_cache_mock,
    sync_mock,
    reset_mock,
    mem_mock,
    loss_mock,
    scaler_patch,
    opt_patch,
):
    cfg = make_cfg()
    model = DummyModel()

    with patch.object(autobatch, "next", return_value=fake_param()):
        usage = autobatch._run_trial(
            model_for_trial=model,
            config_for_trial=cfg,
            mode="train",
            batch_size=2,
            optimizer_main=None,
            criteria_train={"t": torch.nn.CrossEntropyLoss()},
            grad_weighting_main=None,
            scaler_main=None,
            criteria_val=None,
            steps_per_trial=1,
            logger_autobatch=logging.getLogger("test"),
        )

    assert model.train_called
    assert isinstance(usage, float)
    assert opt_patch.called
    assert loss_mock.called

    model.train_called = False
    model.eval_called = False
    with patch.object(autobatch, "next", return_value=fake_param()):
        usage = autobatch._run_trial(
            model_for_trial=model,
            config_for_trial=cfg,
            mode="val",
            batch_size=2,
            optimizer_main=None,
            criteria_train=None,
            grad_weighting_main=None,
            scaler_main=None,
            criteria_val={"t": torch.nn.CrossEntropyLoss()},
            steps_per_trial=1,
            logger_autobatch=logging.getLogger("test"),
        )

    assert model.eval_called
    assert isinstance(usage, float)


@patch.object(autobatch, "_create_temporary_optimizer", return_value=DummyOptimizer())
@patch.object(autobatch.torch.cuda.amp, "GradScaler", DummyScaler)
@patch.object(autobatch, "weighted_hierarchical_loss", return_value=DUMMY_LOSS)
@patch.object(autobatch.torch.cuda, "max_memory_allocated", return_value=0)
@patch.object(autobatch.torch.cuda, "reset_peak_memory_stats")
@patch.object(autobatch.torch.cuda, "synchronize")
@patch.object(autobatch.torch.cuda, "empty_cache")
def test_gradnorm_toggle(
    empty_cache_mock,
    sync_mock,
    reset_mock,
    mem_mock,
    loss_mock,
    scaler_patch,
    opt_patch,
):
    cfg = make_cfg()
    cfg.LOSS.GRAD_WEIGHTING.TASK.GRADNORM_ENABLED = True
    cfg.TRAIN.GRADIENT_CHECKPOINTING.ENABLED_GRADNORM_STEPS = True

    model = DummyModel()
    gradnorm = MagicMock()

    with patch.object(autobatch, "next", return_value=fake_param()):
        autobatch._run_trial(
            model_for_trial=model,
            config_for_trial=cfg,
            mode="train",
            batch_size=2,
            optimizer_main=None,
            criteria_train={"t": torch.nn.CrossEntropyLoss()},
            grad_weighting_main=gradnorm,
            scaler_main=None,
            criteria_val=None,
            steps_per_trial=1,
            logger_autobatch=logging.getLogger("test"),
        )

    assert gradnorm.update_gradnorm_weights_reforward.called
    assert model.use_checkpoint is False


def test_binary_search_and_broadcast():
    cfg = make_cfg()
    device_prop = SimpleNamespace(total_memory=2 << 30)

    def fake_run_trial(**kwargs):
        bs = kwargs["batch_size"]
        return bs * 0.1

    dist_mock = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 2,
        get_rank=lambda: 0,
        broadcast=MagicMock(),
    )

    model = DummyModel()
    with patch.object(autobatch, "dist", dist_mock), patch.object(
        autobatch.torch.cuda, "get_device_properties", return_value=device_prop
    ), patch.object(
        autobatch, "_run_trial", side_effect=fake_run_trial
    ), patch.object(
        autobatch, "next", return_value=fake_param()
    ):
        bs = autobatch.auto_find_batch_size(
            model=model,
            config=cfg,
            mode="train",
            optimizer_main=None,
            criteria_train={"t": torch.nn.CrossEntropyLoss()},
            grad_weighting_main=None,
            scaler_main=None,
            criteria_val=None,
            target_memory_fraction=0.5,
            max_batch_size=16,
        )

    assert bs == 10
    assert dist_mock.broadcast.called


@patch.object(autobatch, "_create_temporary_optimizer")
@patch('linnaeus.utils.autobatch.torch.cuda.amp.GradScaler')
@patch.object(autobatch, "weighted_hierarchical_loss", return_value=DUMMY_LOSS)
@patch.object(autobatch.torch.cuda, "max_memory_allocated", return_value=100 << 20)
@patch.object(autobatch.torch.cuda, "reset_peak_memory_stats")
@patch.object(autobatch.torch.cuda, "synchronize")
@patch.object(autobatch.torch.cuda, "empty_cache")
@patch.object(autobatch, "next", return_value=fake_param())
def test_run_trial_train_with_accumulation(
    mock_next,
    mock_empty_cache,
    mock_synchronize,
    mock_reset_peak_memory,
    mock_max_memory_allocated,
    mock_weighted_hierarchical_loss,
    MockGradScaler,
    mock_create_temporary_optimizer
):
    cfg = make_cfg()
    accumulation_steps = 2
    steps_per_trial = 1
    cfg.TRAIN.ACCUMULATION_STEPS = accumulation_steps

    mock_optimizer_instance = MagicMock()
    mock_optimizer_instance.zero_grad = MagicMock()
    mock_optimizer_instance.step = MagicMock()
    mock_create_temporary_optimizer.return_value = mock_optimizer_instance

    mock_scaler_instance = MagicMock()
    mock_scaler_instance.scale = MagicMock(return_value=MagicMock(backward=MagicMock()))
    mock_scaler_instance.step = MagicMock()
    mock_scaler_instance.update = MagicMock()
    MockGradScaler.return_value = mock_scaler_instance

    autobatch._run_trial(
        model_for_trial=DummyModel(),
        config_for_trial=cfg,
        mode='train',
        batch_size=2,
        optimizer_main=None,
        criteria_train={'t': torch.nn.CrossEntropyLoss()},
        grad_weighting_main=None,
        scaler_main=None,
        criteria_val=None,
        steps_per_trial=steps_per_trial,
        logger_autobatch=logging.getLogger('test_accumulation')
    )

    assert mock_optimizer_instance.zero_grad.call_count == steps_per_trial
    assert mock_weighted_hierarchical_loss.call_count == accumulation_steps * steps_per_trial
    assert mock_scaler_instance.scale.call_count == accumulation_steps * steps_per_trial
    assert mock_scaler_instance.scale.return_value.backward.call_count == accumulation_steps * steps_per_trial

    # Verify loss scaling for the first call
    expected_scaled_loss = DUMMY_LOSS[0].item() / accumulation_steps
    actual_scaled_loss = mock_scaler_instance.scale.call_args_list[0][0][0].item()
    assert actual_scaled_loss == expected_scaled_loss, \
        f"Expected scaled loss {expected_scaled_loss}, but got {actual_scaled_loss}"

    assert mock_scaler_instance.step.call_count == steps_per_trial
    assert mock_scaler_instance.update.call_count == steps_per_trial
