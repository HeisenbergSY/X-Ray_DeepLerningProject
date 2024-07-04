import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from transforms import HistogramEqualization  # Import the custom transform
from undersample import undersample_dataset  # Import the undersample function

def get_transforms(train=True, augment=True, normalize=True):
    transform_list = []
    if train and augment:
        transform_list.extend([
            HistogramEqualization(),  # Apply histogram equalization
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        ])
    else:
        transform_list.extend([
            transforms.Resize((224, 224))
        ])
    
    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)

def plot_class_distribution(dataset, title, model_name):
    if isinstance(dataset, Subset):
        indices = dataset.indices
        original_dataset = dataset.dataset
        targets = [original_dataset.samples[idx][1] for idx in indices]
    else:
        targets = [sample[1] for sample in dataset.samples]
    counter = Counter(targets)
    classes = dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes
    plt.figure(figsize=(8, 6))
    plt.bar(counter.keys(), counter.values(), tick_label=[classes[k] for k in counter.keys()])
    plt.title(f"{title} - Model: {model_name}")
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.show()

def get_data_loaders(train_dir, val_dir, test_dir, batch_size, model_name):
    train_transform = get_transforms(train=True, augment=True, normalize=True)
    original_transform = get_transforms(train=True, augment=False, normalize=False)
    val_test_transform = get_transforms(train=False, augment=False, normalize=True)

    def target_transform(label):
        # Ensure labels are binary (0 or 1)
        return torch.tensor(float(label), dtype=torch.float)

    train_dataset = ImageFolder(root=train_dir, transform=train_transform, target_transform=target_transform)
    original_train_dataset = ImageFolder(root=train_dir, transform=original_transform, target_transform=target_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_test_transform, target_transform=target_transform)
    test_dataset = ImageFolder(root=test_dir, transform=val_test_transform, target_transform=target_transform)

    # Plot class distribution before undersampling
    plot_class_distribution(train_dataset, "Class Distribution Before Undersampling", model_name)

    # Apply undersampling to the training dataset
    balanced_train_dataset = undersample_dataset(train_dataset)

    # Plot class distribution after undersampling
    plot_class_distribution(balanced_train_dataset, "Class Distribution After Undersampling", model_name)

    train_loader = DataLoader(dataset=balanced_train_dataset, batch_size=batch_size, shuffle=True)
    original_train_loader = DataLoader(dataset=original_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, original_train_loader, val_loader, test_loader, train_dataset.classes

def get_k_fold_data_loaders(train_dir, batch_size, k=3):
    transform = get_transforms(train=True, augment=True, normalize=True)
    
    def target_transform(label):
        # Ensure labels are binary (0 or 1)
        return torch.tensor(float(label), dtype=torch.float)

    dataset = ImageFolder(root=train_dir, transform=transform, target_transform=target_transform)
    kfold = KFold(n_splits=k, shuffle=True)

    k_fold_loaders = []

    for train_idx, val_idx in kfold.split(dataset):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)
        
        k_fold_loaders.append((train_loader, val_loader))
    
    return k_fold_loaders, dataset.classes
