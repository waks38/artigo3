import torch

from hopmil.data import Bag, mil_collate


def test_mil_collate_keeps_variable_lengths():
    bags = [
        Bag(instances=torch.randn(3, 8), label=torch.tensor(1)),
        Bag(instances=torch.randn(5, 8), label=torch.tensor(0)),
    ]
    batch = mil_collate(bags)
    assert [b.shape[0] for b in batch["instances"]] == [3, 5]
    assert batch["labels"].tolist() == [1, 0]
