import pytest

torch = pytest.importorskip("torch")

from linnaeus.loss.taxonomy_label_smoothing import (
    TaxonomyAwareLabelSmoothingCE,
    build_taxonomy_smoothing_matrix,
)


def test_taxonomy_smoothing_matrix_rows_sum_to_one():
    distances = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    alpha = 0.2
    mat = build_taxonomy_smoothing_matrix(
        num_classes=3, distances=distances, alpha=alpha, beta=1.0, uniform_roots=False
    )
    row_sums = mat.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    assert torch.allclose(torch.diag(mat), torch.full((3,), 1.0 - alpha), atol=1e-6)
    assert torch.all(mat >= 0)


def test_taxonomy_smoothing_uniform_roots():
    distances = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    alpha = 0.3
    mat = build_taxonomy_smoothing_matrix(
        num_classes=3,
        distances=distances,
        alpha=alpha,
        beta=1.0,
        uniform_roots=True,
        root_class_ids=[0],
    )
    off_diag = mat[0].clone()
    off_diag[0] = 0.0
    expected = torch.full((3,), alpha / 2.0)
    expected[0] = 0.0
    assert torch.allclose(off_diag, expected, atol=1e-6)


def test_taxonomy_aware_loss_identity_matrix_matches_nll():
    # Identity matrix (alpha=0) => should match standard NLL for hard labels
    soft_labels = torch.eye(3, dtype=torch.float32)
    loss_fn = TaxonomyAwareLabelSmoothingCE(soft_labels, apply_class_weights=False)

    logits = torch.tensor([[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    # Compute expected negative log likelihood per sample
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = torch.tensor([ -log_probs[0, 0], -log_probs[1, 1] ])

    out = loss_fn(logits, targets)
    assert torch.allclose(out, expected, atol=1e-6)
