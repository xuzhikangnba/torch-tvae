"""A custom network for continuous, unlabeled vectors; no data download."""

import torch
from torch import nn
from tvae import TVAE


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 8))

    def forward(self, x):
        return self.network(x).chunk(2, dim=-1)


def main():
    torch.manual_seed(42)
    model = TVAE(latent_dim=4, encoder=Encoder(),
                 decoder=nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 12)),
                 likelihood="gaussian", observation_scale=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(64, 12)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        result = model.loss(x)
        result.loss.backward()
        optimizer.step()
    model.eval()
    print(f"loss={result.loss.item():.4f}; reconstruction={tuple(model.reconstruct(x).shape)}")


if __name__ == "__main__":
    main()
