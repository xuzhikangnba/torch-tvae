#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split


def create_balanced_mnist_subset(num_samples_per_class: int = 12800):
    """
    Create a balanced subset of MNIST with equal samples per class.
    
    Args:
        num_samples_per_class: Number of samples to select per class
        
    Returns:
        Subset of MNIST dataset with balanced classes
    """
    data_transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root="./data", train=True, transform=data_transform, download=True)

    # Balanced subset selection
    indices = []
    class_counts = {i: 0 for i in range(10)}
    for idx, (_, label) in enumerate(mnist_data):
        if class_counts[label] < num_samples_per_class:
            indices.append(idx)
            class_counts[label] += 1
        if all(class_counts[i] >= num_samples_per_class for i in range(10)):
            break
    
    subset_data = Subset(mnist_data, indices)
    return subset_data


def create_data_loaders(subset_data, batch_size: int = 128, train_split: float = 0.8):
    """
    Create training and test data loaders.
    
    Args:
        subset_data: Dataset subset to split
        batch_size: Batch size for loaders
        train_split: Fraction of data for training
        
    Returns:
        train_loader, test_loader, train_loader_eval, test_loader_eval
    """
    train_size = int(train_split * len(subset_data))
    test_size = len(subset_data) - train_size
    train_data, test_data = random_split(subset_data, [train_size, test_size])

    # Training loaders (with shuffle)
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=torch.cuda.is_available()
    )

    # Eval loaders (NO shuffle, for stable saving)
    train_loader_eval = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=torch.cuda.is_available()
    )
    test_loader_eval = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader, train_loader_eval, test_loader_eval
