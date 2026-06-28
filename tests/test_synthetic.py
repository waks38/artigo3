import torch

from hopmil.data.synthetic import make_bags, make_ilse_bags, make_witness_bags


def _toy(n=200, n_classes=10):
    # n instances, label = index % n_classes; instance "image" is (1,4,4)
    labels = torch.arange(n) % n_classes
    instances = torch.randn(n, 1, 4, 4)
    return instances, labels


def test_fixed_size_one_witness_balanced():
    inst, lab = _toy()
    bags = make_witness_bags(
        inst, lab, target=3, num_bags=100, bag_size=15,
        num_witnesses=1, positive_ratio=0.5, seed=0,
    )
    assert len(bags) == 100
    assert all(b.instances.shape == (15, 1, 4, 4) for b in bags)
    pos = [b for b in bags if int(b.label) == 1]
    neg = [b for b in bags if int(b.label) == 0]
    assert len(pos) == 50 and len(neg) == 50              # balanced
    assert all(int(b.instance_labels.sum()) == 1 for b in pos)   # exactly 1 witness
    assert all(int(b.instance_labels.sum()) == 0 for b in neg)   # none in negatives


def test_label_matches_witness_presence():
    inst, lab = _toy()
    bags = make_witness_bags(inst, lab, target=3, num_bags=40, seed=1)
    for b in bags:
        assert int(b.label) == int(b.instance_labels.any())


def test_deterministic_in_seed():
    inst, lab = _toy()
    a = make_witness_bags(inst, lab, target=3, num_bags=20, seed=7)
    b = make_witness_bags(inst, lab, target=3, num_bags=20, seed=7)
    assert [int(x.label) for x in a] == [int(x.label) for x in b]
    assert torch.equal(a[0].instances, b[0].instances)


def test_invalid_num_witnesses_raises():
    inst, lab = _toy()
    for bad in (0, 20):  # must be in [1, bag_size]
        try:
            make_witness_bags(inst, lab, target=3, num_bags=5, bag_size=15, num_witnesses=bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_ilse_bags_variable_size_balanced_multiwitness():
    inst, lab = _toy()
    bags = make_ilse_bags(
        inst, lab, target=3, num_bags=200,
        mean_bag_size=10, var_bag_size=2, positive_ratio=0.5, seed=0,
    )
    assert len(bags) == 200
    pos = [b for b in bags if int(b.label) == 1]
    neg = [b for b in bags if int(b.label) == 0]
    assert len(pos) == 100 and len(neg) == 100                      # balanced
    assert all(int(b.instance_labels.sum()) >= 1 for b in pos)      # >= 1 witness
    assert all(int(b.instance_labels.sum()) == 0 for b in neg)      # none in negatives
    assert any(int(b.instance_labels.sum()) >= 2 for b in pos)      # multi-witness occurs
    sizes = {b.instances.shape[0] for b in bags}
    assert len(sizes) > 1                                           # variable bag size
    assert all(int(b.label) == int(b.instance_labels.any()) for b in bags)


def test_make_bags_dispatch():
    inst, lab = _toy()
    w = make_bags(inst, lab, target=3, bag_mode="witness", num_bags=10, seed=0)
    assert all(b.instances.shape[0] == 15 for b in w)               # fixed size
    i = make_bags(inst, lab, target=3, bag_mode="ilse", num_bags=10, seed=0)
    assert len({b.instances.shape[0] for b in i}) > 1               # variable size
    try:
        make_bags(inst, lab, target=3, bag_mode="bogus", num_bags=5)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
