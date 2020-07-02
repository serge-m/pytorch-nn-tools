
def dummy_batch(m: nn.Module, size:Tuple) -> torch.Tensor:
    return next(m.parameters()).new(*size).requires_grad_(False).uniform_(-1., 1.)
