import torch.nn.functional as F
import torch

def sample_gumbel(shape=[], eps=1e-20, device=[]):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits=[], temperature=[], device=[]):
    gumbel_noise = sample_gumbel(shape=logits.size(), device=device)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits=[], temperature=[], hard=False, device=[]):
    y = gumbel_softmax_sample(logits, temperature, device)
    if hard:
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y