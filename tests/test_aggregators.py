"""All aggregators must honor one contract: (n, dim) -> ((dim,), (n,)|None)."""

import pytest
import torch

from hopmil.models.aggregators import build_aggregator

DIM = 16
N = 7


@pytest.mark.parametrize("name", ["mean", "max", "attention"])
def test_aggregator_contract(name):
    agg = build_aggregator(name, dim=DIM)
    x = torch.randn(N, DIM)
    z, w = agg(x)
    assert z.shape == (DIM,)
    if w is not None:
        assert w.shape == (N,)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)


def test_unknown_aggregator_raises():
    with pytest.raises(ValueError):
        build_aggregator("nope", dim=DIM)


@pytest.mark.slow
def test_hopfield_contract():
    pytest.importorskip("hflayers")
    agg = build_aggregator("hopfield", dim=DIM)
    z, _ = agg(torch.randn(N, DIM))
    assert z.shape == (DIM,)
