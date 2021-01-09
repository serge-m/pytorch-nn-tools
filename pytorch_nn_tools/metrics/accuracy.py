from typing import List
import torch


def topk_accuracy(output, target, topk=(1,)) -> List:
    """
    Computes the accuracy over the k top predictions for a
    set of k.
    By default computes top-1 accuracy only.
    Accuracy is returned as a fraction from interval [0, 1].
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = [
            correct[:k].reshape(-1).float().sum(0, keepdim=True).div_(batch_size)
            for k in topk
        ]
        return res
