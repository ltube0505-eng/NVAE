import torch

from diagnose_inference_gap import (
    classify_gap,
    importance_statistics,
    parse_checkpoints,
    parse_iw_checkpoints,
)


def test_checkpoint_and_iw_parsing():
    assert list(parse_checkpoints(['st=/a.pt', 'relaxed=/b.pt']).keys()) == ['st', 'relaxed']
    assert parse_iw_checkpoints('1,10', 100) == [1, 10, 100]


def test_importance_statistics_uniform_weights_have_full_ess():
    log_weights = torch.zeros(10, 3)
    nll, ess_fraction, max_weight_fraction = importance_statistics(log_weights, [1, 10])
    assert torch.allclose(nll['1'], torch.zeros(3))
    assert torch.allclose(nll['10'], torch.zeros(3), atol=1e-6)
    assert torch.allclose(ess_fraction, torch.ones(3), atol=1e-6)
    assert torch.allclose(max_weight_fraction, torch.full((3,), 0.1), atol=1e-6)


def _metrics(**overrides):
    values = dict(
        refined_ess_fraction=0.4,
        last_iw_tightening=0.1,
        surrogate_drop=2.,
        amortization_gap=6.,
        residual_gap=4.,
        total_inference_gap=10.,
    )
    values.update(overrides)
    return values


def test_classification_checks_estimator_reliability_before_gap_split():
    assert classify_gap(_metrics(refined_ess_fraction=0.001)) == \
        'inconclusive_importance_sampling'
    assert classify_gap(_metrics(amortization_gap=0.01)) == 'surrogate_mismatch'
    assert classify_gap(_metrics(amortization_gap=7., residual_gap=3.)) == \
        'amortization_dominated'
    assert classify_gap(_metrics(amortization_gap=2., residual_gap=8.)) == \
        'posterior_family_dominated_proxy'
    assert classify_gap(_metrics(amortization_gap=5., residual_gap=5.)) == 'mixed'
