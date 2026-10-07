"""Fresh train/test runs using the paper's stated optimizer and two-stage schedule.

The default interprets '500 training iterations' literally as 500 minibatch
updates, following 50 reconstruction-only epochs. ``--main-epochs`` selects an
explicit alternative. Every run records the split, loss normalization, missing
protocol assumptions, package source hashes, random states, and actual updates.
These are new experiments; no archived measurements are substituted for output.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .reference import TVaDE28, TVaDE128

DATASETS = ("mnist", "fashion_mnist", "oasis_coronal", "oasis_axial", "oasis_sagittal")
PRIOR_NAMES = {"gmm_means", "gmm_logvars", "logit_pi", "nu_rho"}


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_cauchy(images, gamma, generator=None):
    if gamma == 0:
        return images
    return (images + torch.empty_like(images).cauchy_(0, gamma, generator=generator)).clamp(0, 1)


def objective(model, images, *, pretrain=False, pretrain_latent="mean", kl_weight=1.0):
    mu, logvar = model.encode(images)
    z = mu if pretrain and pretrain_latent == "mean" else model.reparameterize(mu, logvar)
    decoded = model.decode(z).clamp(1e-6, 1 - 1e-6)
    # Both terms are per observation, avoiding a batch-size-dependent KL weight.
    reconstruction = F.binary_cross_entropy(decoded, images, reduction="sum") / len(images)
    prior = reconstruction.new_zeros(()) if pretrain else model.student_t_elbo_loss(z, mu, logvar)
    return reconstruction + kl_weight * prior, reconstruction, prior


def optimizer_for(model, args, pretrain=False):
    autoencoder = [p for name, p in model.named_parameters() if name not in PRIOR_NAMES]
    groups = [{"params": autoencoder, "lr": args.ae_lr, "name": "autoencoder"}]
    if not pretrain:
        groups.append({"params": [p for name, p in model.named_parameters() if name in PRIOR_NAMES],
                       "lr": args.mixture_lr, "name": "mixture"})
    cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    return cls(groups, weight_decay=args.weight_decay)


def data(args, device):
    from torchvision.datasets import MNIST, FashionMNIST
    root = args.data_root.resolve()
    records = {}
    if args.dataset.startswith("oasis_"):
        file = root / (args.dataset.removeprefix("oasis_") + ".npz")
        with np.load(file, allow_pickle=False) as archive:
            x = torch.from_numpy(archive["images"].astype("float32")).unsqueeze(1)
        if x.ndim != 4 or x.shape[1:] != (1, 128, 128):
            raise ValueError(f"Expected (N,128,128) MRI images, got {tuple(x.shape)}")
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("MRI images must be finite and already normalized to [0,1]")
        y = torch.full((len(x),), -1, dtype=torch.long)
        records["prepared_images_sha256"] = sha256(file)
        split = "seeded_80_20"
    else:
        cls = MNIST if args.dataset == "mnist" else FashionMNIST
        ds = cls(str(root), train=True, download=True)
        x, y = ds.data.unsqueeze(1).float() / 255, ds.targets.clone()
        records["training_pixels_sha256"] = hashlib.sha256(ds.data.numpy().tobytes()).hexdigest()
        split = args.split
    generator = torch.Generator().manual_seed(args.seed)
    if split == "official":
        test = cls(str(root), train=False, download=True)
        xtrain, ytrain, xtest, ytest = x, y, test.data.unsqueeze(1).float() / 255, test.targets.clone()
        train_indices = torch.arange(len(x))
        test_indices = torch.arange(len(xtest))
        records["test_pixels_sha256"] = hashlib.sha256(test.data.numpy().tobytes()).hexdigest()
    else:
        permutation = torch.randperm(len(x), generator=generator)
        boundary = int(.8 * len(x))
        train_indices, test_indices = permutation[:boundary], permutation[boundary:]
        xtrain, ytrain, xtest, ytest = x[train_indices], y[train_indices], x[test_indices], y[test_indices]
    if args.train_limit:
        xtrain, ytrain, train_indices = xtrain[:args.train_limit], ytrain[:args.train_limit], train_indices[:args.train_limit]
    if args.test_limit:
        xtest, ytest, test_indices = xtest[:args.test_limit], ytest[:args.test_limit], test_indices[:args.test_limit]
    records.update(split=split, train_samples=len(xtrain), test_samples=len(xtest),
                   train_indices=train_indices.tolist(), test_indices=test_indices.tolist(),
                   test_indices_scope="official test partition" if split == "official" else "training archive")
    return xtrain.to(device), ytrain.numpy(), xtest.to(device), ytest.numpy(), records


@torch.no_grad()
def embeddings(model, images, batch=256):
    model.eval()
    return np.concatenate([model.encode(x)[0].cpu().numpy() for x in images.split(batch)])


@torch.no_grad()
def native_labels(model, latent, device):
    z = torch.as_tensor(latent, device=device)
    nu = model.nu_c
    variance = model.gmm_logvars.exp() + 1e-6
    radius = ((z[:, None] - model.gmm_means[None]).square() / variance[None]).sum(-1)
    log_prob = (torch.lgamma((nu + model.D) / 2) - torch.lgamma(nu / 2)
                - model.D / 2 * torch.log(nu * math.pi) - .5 * variance.log().sum(-1)
                - (nu + model.D) / 2 * torch.log1p(radius / nu))
    return (log_prob + model.logit_pi.log_softmax(0)).argmax(-1).cpu().numpy()


def fit_gmm(latent, k, n_init=10):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(k, covariance_type="full", n_init=n_init, random_state=0).fit(latent)


@torch.no_grad()
def initialize_mixture(model, fitted):
    device = model.gmm_means.device
    model.gmm_means.copy_(torch.tensor(fitted.means_, device=device, dtype=torch.float32))
    variance = np.diagonal(fitted.covariances_, axis1=1, axis2=2).copy()
    model.gmm_logvars.copy_(torch.tensor(variance, device=device, dtype=torch.float32).clamp_min(1e-6).log())
    model.logit_pi.copy_(torch.tensor(fitted.weights_, device=device, dtype=torch.float32).clamp_min(1e-6).log())


@torch.no_grad()
def initialize_posterior_variance(model, policy):
    """A deterministic AE never trains its variance head: initialize it explicitly.

    Leaving that random head attached to a changing encoder can produce log
    variances >88 and overflow float32 exp before the VAE has trained one step.
    Unit variance is an initialization, not a clamp or a hidden change of loss.
    """
    if policy == "unit":
        model.encoder.fc_logvar.weight.zero_()
        model.encoder.fc_logvar.bias.zero_()
    elif policy != "preserve":
        raise ValueError(policy)


def metrics(latent, actual, predicted, k, *, silhouette_limit=None):
    from scipy.optimize import linear_sum_assignment
    from sklearn import config_context
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
    counts = np.bincount(predicted, minlength=k)
    result = {"cluster_counts": counts.tolist(), "occupied_clusters": int(np.count_nonzero(counts))}
    if np.all(actual >= 0):
        contingency = np.zeros((k, k), dtype=np.int64)
        np.add.at(contingency, (actual, predicted), 1)
        row, col = linear_sum_assignment(-contingency)
        result.update(accuracy=float(contingency[row, col].sum() / len(actual)),
                      ari=float(adjusted_rand_score(actual, predicted)),
                      nmi=float(normalized_mutual_info_score(actual, predicted)))
    valid = 1 < len(np.unique(predicted)) < len(predicted)
    result["silhouette"] = None
    result["davies_bouldin"] = None
    if valid:
        with config_context(working_memory=128):
            result["silhouette"] = float(silhouette_score(latent, predicted, sample_size=silhouette_limit, random_state=0))
        result["davies_bouldin"] = float(davies_bouldin_score(latent, predicted))
    result["silhouette_samples"] = min(len(latent), silhouette_limit or len(latent))
    return result


@torch.no_grad()
def evaluate(model, xtrain, ytrain, xtest, ytest, args, output, label):
    start = time.monotonic()
    model.eval()
    train_z, test_z = embeddings(model, xtrain), embeddings(model, xtest)
    fitted = fit_gmm(train_z, model.K, args.gmm_n_init)
    report = {"stage": label, "metrics": {}, "gmm_fit_partition": "train",
              "selection": "This explicitly named training endpoint; no test-accuracy checkpoint selection"}
    for split, z, y in [("train", train_z, ytrain), ("test", test_z, ytest)]:
        report["metrics"][split] = {
            "native": metrics(z, y, native_labels(model, z, xtrain.device), model.K,
                              silhouette_limit=2000 if split == "train" else None),
            "refit": metrics(z, y, fitted.predict(z), model.K,
                             silhouette_limit=2000 if split == "train" else None),
        }
    reconstruction = 0.0
    for x in xtest.split(128):
        mu, _ = model.encode(x)
        reconstructed = model.decode(mu).clamp(1e-6, 1 - 1e-6)
        reconstruction += float(F.binary_cross_entropy(reconstructed, x, reduction="sum"))
    report.update(test_reconstruction_bce_per_image=reconstruction / len(xtest),
                  nu_c=model.nu_c.cpu().tolist(), evaluation_seconds=time.monotonic() - start)
    native = model.decode(model.gmm_means).cpu().numpy()
    refit = model.decode(torch.tensor(fitted.means_, device=xtrain.device, dtype=torch.float32)).cpu().numpy()
    np.savez_compressed(output / f"{label}_prototypes.npz", native=native, refit=refit,
                        refit_means=fitted.means_, refit_covariances=fitted.covariances_,
                        refit_weights=fitted.weights_)
    write_json(output / f"{label}_metrics.json", report)
    print(json.dumps({"evaluation": label, "test": report["metrics"]["test"]}), flush=True)
    return report


def checkpoint(path, model, optimizer, state, args):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "progress": state,
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "cpu_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_rng(saved):
    torch.set_rng_state(saved["cpu_rng"].cpu())
    if saved["cuda_rng"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gamma", type=float, default=.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["official", "seeded_80_20"], default="official")
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--main-steps", type=int, default=500)
    schedule.add_argument("--main-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ae-lr", type=float, default=.001)
    parser.add_argument("--mixture-lr", type=float, default=.002)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--kl-weight", type=float, default=1.)
    parser.add_argument("--pretrain-latent", choices=["mean", "sample"], default="mean")
    parser.add_argument("--posterior-init", choices=["unit", "preserve"], default="unit")
    parser.add_argument("--gmm-n-init", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--from-pretrained", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0, help="Nonzero marks a functionality check, not a benchmark")
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--evaluate-every-epochs", type=int, default=0)
    args = parser.parse_args(argv)
    if args.gamma < 0 or args.pretrain_epochs < 0 or args.main_steps < 0 or (args.main_epochs is not None and args.main_epochs < 0):
        parser.error("Noise and training durations must be nonnegative")
    if min(args.batch_size, args.ae_lr, args.mixture_lr, args.gmm_n_init, args.threads) <= 0:
        parser.error("Batch size, learning rates, GMM restarts and threads must be positive")
    if args.resume and args.from_pretrained:
        parser.error("Choose either --resume or --from-pretrained")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    xtrain, ytrain, xtest, ytest, source = data(args, device)
    write_json(output / "data_manifest.json", source)
    nsteps = math.ceil(len(xtrain) / args.batch_size)
    main_target = args.main_steps if args.main_epochs is None else args.main_epochs * nsteps
    model = (TVaDE128(32, 2) if args.dataset.startswith("oasis_") else TVaDE28(8, 10)).to(device)
    import tvae
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(package_version=tvae.__version__, package_path=str(Path(tvae.__file__).resolve()),
                  trainer_sha256=sha256(__file__), model_module=model.__class__.__module__,
                  main_updates=main_target, steps_per_epoch=nsteps, torch_version=torch.__version__,
                  cuda_device=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  objective="batch mean of pixel-summed BCE + kl_weight * batch mean analytic prior loss",
                  training_corruption="Fresh clipped additive Cauchy noise at every visit",
                  evaluation_corruption="Fixed independent train/test noise, seed+10000/seed+20000",
                  mixture_initialization="Train-only full-covariance GMM after reconstruction pretraining; diagonal extracted for prior",
                  assumptions_not_specified_in_main_text=["latent dimensions 8/32 from archived models",
                      "BCE observation loss and gradient clipping at norm 5 from archived source",
                      "deterministic latent mean for reconstruction-only pretraining unless requested otherwise",
                      "train GMM initialization, 10 restarts, random_state=0", "fixed evaluation noise"],
                  primary_result="Final endpoint; never best test accuracy",
                  functionality_check=bool(args.train_limit or args.test_limit))
    if args.resume:
        previous = json.loads((output / "run_config.json").read_text())
        for key in ["dataset", "data_root", "gamma", "seed", "split", "pretrain_epochs", "batch_size", "ae_lr", "mixture_lr", "optimizer", "weight_decay", "kl_weight", "pretrain_latent", "posterior_init", "train_limit", "test_limit", "trainer_sha256"]:
            if previous[key] != config[key]:
                raise ValueError(f"Cannot silently change {key} when resuming")
        write_json(output / f"resume_config_{time.time_ns()}.json", config)
    else:
        write_json(output / "run_config.json", config)
    print(json.dumps({"dataset": args.dataset, "device": str(device), "train": len(xtrain), "test": len(xtest),
                      "pretrain_epochs": args.pretrain_epochs, "main_updates": main_target,
                      "package_path": config["package_path"]}), flush=True)
    # Isolated generators ensure evaluation does not perturb the training RNG.
    train_eval = add_cauchy(xtrain, args.gamma, torch.Generator(device=device).manual_seed(args.seed + 10000))
    test_eval = add_cauchy(xtest, args.gamma, torch.Generator(device=device).manual_seed(args.seed + 20000))
    state = {"phase": "pretrain", "pretrain_epoch": 0, "main_step": 0, "order": None, "offset": 0}
    optimizer = optimizer_for(model, args, pretrain=True)
    if args.resume or args.from_pretrained:
        file = output / "last.pt" if args.resume else args.from_pretrained
        saved = torch.load(file, map_location=device, weights_only=True)
        if args.from_pretrained:
            if saved["progress"]["phase"] != "pretrain_complete":
                raise ValueError("--from-pretrained requires pretrain.pt")
            for key in ["dataset", "data_root", "gamma", "seed", "split", "pretrain_epochs", "batch_size", "ae_lr", "optimizer", "weight_decay", "pretrain_latent", "train_limit", "test_limit"]:
                value = str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
                if saved["args"][key] != value:
                    raise ValueError(f"Pretraining mismatch: {key}")
        state = saved["progress"]
        model.load_state_dict(saved["model"])
        optimizer = optimizer_for(model, args, pretrain=state["phase"] in ("pretrain", "pretrain_complete"))
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved)
    start = time.monotonic()
    history = output / "training.jsonl"

    def record(row):
        row["elapsed_seconds_this_process"] = time.monotonic() - start
        with history.open("a") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(output / "status.json", row)
        print(json.dumps(row), flush=True)

    for epoch in range(state["pretrain_epoch"], args.pretrain_epochs):
        model.train()
        total, seen = 0., 0
        order = torch.randperm(len(xtrain), device=device)
        epoch_start = time.monotonic()
        for indices in order.split(args.batch_size):
            noisy = add_cauchy(xtrain[indices], args.gamma)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = objective(model, noisy, pretrain=True, pretrain_latent=args.pretrain_latent)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite pretraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
        state.update(pretrain_epoch=epoch + 1)
        record({"phase": "pretrain", "epoch": epoch + 1, "bce_per_image": total / seen,
                "epoch_seconds": time.monotonic() - epoch_start})
        checkpoint(output / "last.pt", model, optimizer, state, args)
    if state["phase"] == "pretrain":
        state["phase"] = "pretrain_complete"
        checkpoint(output / "pretrain.pt", model, optimizer, state, args)
    if state["phase"] == "pretrain_complete":
        initialize_posterior_variance(model, args.posterior_init)
        print("Initializing mixture from training embeddings", flush=True)
        initialize_mixture(model, fit_gmm(embeddings(model, train_eval), model.K, args.gmm_n_init))
        optimizer = optimizer_for(model, args)
        state.update(phase="main", order=None, offset=0)
        checkpoint(output / "last.pt", model, optimizer, state, args)
    accum, count = 0., 0
    while state["main_step"] < main_target:
        model.train()
        if state["order"] is None or state["offset"] >= len(xtrain):
            state["order"] = torch.randperm(len(xtrain), device=device)
            state["offset"] = 0
        indices = state["order"][state["offset"]:state["offset"] + args.batch_size]
        state["offset"] += len(indices)
        noisy = add_cauchy(xtrain[indices], args.gamma)
        optimizer.zero_grad(set_to_none=True)
        loss, reconstruction, prior = objective(model, noisy, kl_weight=args.kl_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite ELBO objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        state["main_step"] += 1
        accum += float(loss.detach()) * len(indices)
        count += len(indices)
        step = state["main_step"]
        if step % 100 == 0 or step == main_target:
            record({"phase": "main", "step": step, "effective_epochs": step / nsteps,
                    "objective_per_image": accum / count, "last_reconstruction": float(reconstruction.detach()),
                    "last_prior": float(prior.detach()), "nu_mean": float(model.nu_c.detach().mean())})
            accum, count = 0., 0
        if step % 500 == 0 or step == main_target:
            checkpoint(output / "last.pt", model, optimizer, state, args)
        if args.evaluate_every_epochs and step % (args.evaluate_every_epochs * nsteps) == 0:
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                evaluate(model, train_eval, ytrain, test_eval, ytest, args, output, f"main_step_{step}")
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        report = evaluate(model, train_eval, ytrain, test_eval, ytest, args, output, "final")
    checkpoint(output / "last.pt", model, optimizer, state, args)
    record({"phase": "complete", "pretrain_epochs": state["pretrain_epoch"], "main_steps": state["main_step"],
            "test_native": report["metrics"]["test"]["native"], "test_refit": report["metrics"]["test"]["refit"]})


if __name__ == "__main__":
    main()
