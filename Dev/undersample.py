from collections import Counter
import numpy as np
from torch.utils.data import Subset

def undersample_dataset(dataset):
    targets = [sample[1] for sample in dataset.samples]
    counter = Counter(targets)
    min_count = min(counter.values())

    indices_per_class = {cls: [] for cls in counter.keys()}
    for idx, target in enumerate(targets):
        indices_per_class[target].append(idx)

    undersampled_indices = []
    for cls, indices in indices_per_class.items():
        undersampled_indices.extend(np.random.choice(indices, min_count, replace=False))

    return Subset(dataset, undersampled_indices)
