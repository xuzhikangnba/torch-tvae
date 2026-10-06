"""Portable adapters for the unmodified archived experiment source."""
from pathlib import Path
import importlib.util
import os
import sys

DATASETS = ('mnist', 'fashion_mnist', 'oasis_coronal', 'oasis_axial', 'oasis_sagittal')


def load_reference(dataset, data_root, device='cpu', threads=4, workers=0):
    # Called once in a fresh process: archived modules use unqualified imports.
    if dataset not in DATASETS:
        raise ValueError(dataset)
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif device.startswith('cuda:'):
        os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':', 1)[1]
    else:
        raise ValueError('device must be cpu or cuda:<physical GPU index>')
    import torch
    from torch.utils.data import DataLoader, Subset, random_split
    from torchvision import datasets, transforms
    from tvae.reference import TVaDE28, TVaDE128
    torch.set_num_threads(threads)
    root = Path(__file__).resolve().parent / 'reference' / dataset
    sys.path[:0] = [str(root/'v4.1_tVaDE_ModeB'), str(root/'robustness_eval'), str(root)]
    spec = importlib.util.spec_from_file_location('reference_train', root/'robustness_eval/v4.1/train_noisy.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    data_root = Path(data_root).resolve()

    def make_dataset(n):
        if dataset.startswith('oasis_'):
            from data.data_utils import OASIS2DDataset
            return OASIS2DDataset(view=dataset.removeprefix('oasis_'), root=str(data_root), target_size=128)
        cls = datasets.MNIST if dataset == 'mnist' else datasets.FashionMNIST
        ds = cls(root=str(data_root), train=True, transform=transforms.ToTensor(), download=True)
        # Same original-index selection, without needlessly decoding every image.
        counts = [0] * 10
        indices = []
        for i, y in enumerate(ds.targets.tolist()):
            if counts[y] < n:
                indices.append(i)
                counts[y] += 1
        return Subset(ds, indices)

    def make_loaders(ds, batch_size=32, train_split=.8):
        n = int(len(ds) * train_split)
        train, test = random_split(ds, [n, len(ds)-n])
        def loader(part, shuffle=False):
            return DataLoader(part, batch_size=batch_size, shuffle=shuffle,
                              num_workers=workers, pin_memory=device != 'cpu')
        return loader(train, True), loader(test), loader(train), loader(test)

    entry.create_balanced_mnist_subset = make_dataset
    entry.create_data_loaders = make_loaders
    entry.StudentTVaDE = TVaDE128 if dataset.startswith('oasis_') else TVaDE28
    return entry, torch.device('cpu' if device == 'cpu' else 'cuda:0')
