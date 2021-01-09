from pytorch_nn_tools.metrics import accuracy
from torch import tensor


def test_topk_accuracy():
    output = tensor([[0.1, 0.2, 0.5], [0.8, 0.1, 0.2]])
    target = tensor([1, 0])
    results = accuracy.topk_accuracy(output, target, (1, 2, 3))
    assert results == [0.5, 1., 1.]
