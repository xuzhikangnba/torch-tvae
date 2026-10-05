"""Optional MNIST example. Labels are never passed to the model or objective."""

import argparse
from pathlib import Path

import torch

from .models import TVAE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mnist"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--df", type=float, default=5.0)
    parser.add_argument("--fixed-df", action="store_true")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None, help="Optional training subset size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu or e.g. cuda:0")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.latent_dim) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("epochs, batch size, latent dimension and limit must be positive")
    try:
        from torchvision import datasets, transforms
        from torchvision.utils import save_image
    except ImportError as exc:
        raise SystemExit("Install the optional example dependencies: pip install 'torch-tvae[examples]'") from exc
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                             transform=transforms.ToTensor())
    if args.limit is not None:
        indices = torch.randperm(len(dataset))[:args.limit].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    config = dict(input_dim=784, latent_dim=args.latent_dim, df=args.df,
                  learn_df=not args.fixed_df)
    model = TVAE(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        total = count = 0
        for x, _unused_labels in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model.loss(x, beta=args.beta)
            if not torch.isfinite(result.loss):
                raise RuntimeError("Non-finite loss; inspect the learning rate and model parameters")
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += result.loss.item() * len(x)
            count += len(x)
        print(f"epoch={epoch + 1} loss={total / count:.4f} df={model.prior.df.item():.4f}", flush=True)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, args.output_dir / "model.pt")
    examples = torch.stack([dataset[i][0] for i in range(min(16, len(dataset)))]).to(device)
    save_image(torch.cat([examples, model.reconstruct(examples)]), args.output_dir / "reconstructions.png", nrow=16)
    save_image(model.sample(64).reshape(-1, 1, 28, 28), args.output_dir / "samples.png", nrow=8)
    print(f"Saved checkpoint and previews to {args.output_dir}")


if __name__ == "__main__":
    main()
