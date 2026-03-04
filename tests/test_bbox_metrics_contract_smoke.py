import json

from tools.smoke.bbox_metrics_contract_smoke import main


def test_bbox_metrics_contract_smoke_emits_required_keys(tmp_path, capsys):
    rc = main(["--output-dir", str(tmp_path), "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["missing_metric_keys"] == []
    required = set(payload["required_metric_keys"])
    assert "train/bbox_valid_fraction" in required
    assert "train/bbox_area_fraction" in required
    assert "val/bbox_valid_fraction" in required
    assert "val/bbox_area_fraction" in required

