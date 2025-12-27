from linnaeus.utils.metrics.metric_ref import parse_metric_ref


def test_parse_metric_ref_explicit_slash():
    assert parse_metric_ref("val/loss") == ("val", "loss")
    assert parse_metric_ref("train/acc") == ("train", "acc")


def test_parse_metric_ref_legacy_underscore():
    assert parse_metric_ref("val_loss") == ("val", "loss")
    assert parse_metric_ref("train_loss") == ("train", "loss")


def test_parse_metric_ref_legacy_dot():
    assert parse_metric_ref("val.loss") == ("val", "loss")
    assert parse_metric_ref("train.top1") == ("train", "top1")


def test_parse_metric_ref_defaults():
    assert parse_metric_ref("loss") == ("val", "loss")
    assert parse_metric_ref("", default_phase="train") == ("train", "loss")

