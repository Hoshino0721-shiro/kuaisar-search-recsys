"""Separate cheap LR and explicit-cross + deep ranker on causal numeric features."""
import torch
from torch import nn

COARSE_COLUMNS = [1, 2, 3, 4, 5, 6, 9, 10]


class CoarseLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(len(COARSE_COLUMNS), 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class FineDCN(nn.Module):
    """Original vector-parameter cross layers plus an independent deep branch.

    x_(l+1) = x_0 * (w_l^T x_l) + b_l + x_l.
    This is not DCN-v2 and does not contain a behavior-sequence encoder.
    """
    def __init__(self, input_dim=20, cross_layers=2):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim)) for _ in range(cross_layers)])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(cross_layers)])
        for w in self.weights:
            nn.init.normal_(w, std=0.01)
        self.deep = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Linear(input_dim + 32, 1)

    def forward(self, x):
        cross = x
        for weight, bias in zip(self.weights, self.biases):
            cross = x * (cross @ weight).unsqueeze(-1) + bias + cross
        return self.head(torch.cat([cross, self.deep(x)], dim=-1)).squeeze(-1)


def make_model(kind):
    if kind == 'coarse_lr':
        return CoarseLR()
    if kind == 'fine_dcn':
        return FineDCN()
    raise ValueError(f'Unknown ranker: {kind}')
