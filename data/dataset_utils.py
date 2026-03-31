import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import random

def get_transforms(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10' or dataset_name == 'cifar100' :
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset_name == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        transform_test = transform_train
    elif dataset_name == 'svhn':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif dataset_name == 'tinyimagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError("Unknown dataset")
    return transform_train, transform_test


def get_full_datasets(dataset_name, root, transform_train, transform_test):
    """
    full_train_view: transform_train
    full_test_view: transform_test
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'cifar10':
        train_part_tr = datasets.CIFAR10(f"{root}/cifar10", train=True, download=True, transform=transform_train)
        test_part_tr = datasets.CIFAR10(f"{root}/cifar10", train=False, download=True, transform=transform_train)
        full_train_view = ConcatDataset([train_part_tr, test_part_tr])

        train_part_te = datasets.CIFAR10(f"{root}/cifar10", train=True, download=True, transform=transform_test)
        test_part_te = datasets.CIFAR10(f"{root}/cifar10", train=False, download=True, transform=transform_test)
        full_test_view = ConcatDataset([train_part_te, test_part_te])

    elif dataset_name == 'cifar100':
        train_part_tr = datasets.CIFAR100(f"{root}/cifar100", train=True, download=True, transform=transform_train)
        test_part_tr = datasets.CIFAR100(f"{root}/cifar100", train=False, download=True, transform=transform_train)
        full_train_view = ConcatDataset([train_part_tr, test_part_tr])

        train_part_te = datasets.CIFAR100(f"{root}/cifar100", train=True, download=True, transform=transform_test)
        test_part_te = datasets.CIFAR100(f"{root}/cifar100", train=False, download=True, transform=transform_test)
        full_test_view = ConcatDataset([train_part_te, test_part_te])

    elif dataset_name == 'mnist':
        train_part = datasets.MNIST(f"{root}/Mnist", train=True, download=True, transform=transform_train)
        test_part = datasets.MNIST(f"{root}/Mnist", train=False, download=True, transform=transform_train)

        full_train_view = ConcatDataset([train_part, test_part])
        full_test_view = full_train_view

    elif dataset_name == 'svhn':
        train_part_tr = datasets.SVHN(f"{root}/SVHN", split='train', download=True, transform=transform_train)
        test_part_tr = datasets.SVHN(f"{root}/SVHN", split='test', download=True, transform=transform_train)
        full_train_view = ConcatDataset([train_part_tr, test_part_tr])

        train_part_te = datasets.SVHN(f"{root}/SVHN", split='train', download=True, transform=transform_test)
        test_part_te = datasets.SVHN(f"{root}/SVHN", split='test', download=True, transform=transform_test)
        full_test_view = ConcatDataset([train_part_te, test_part_te])

    elif dataset_name == 'tinyimagenet':
        tiny_root = os.path.join(f"{root}/tiny_imagenet")
        train_dir = os.path.join(tiny_root, 'train')
        val_dir = os.path.join(tiny_root, 'val')

        d1_tr = datasets.ImageFolder(train_dir, transform=transform_train)
        d2_tr = datasets.ImageFolder(val_dir, transform=transform_train)
        full_train_view = ConcatDataset([d1_tr, d2_tr])

        d1_te = datasets.ImageFolder(train_dir, transform=transform_test)
        d2_te = datasets.ImageFolder(val_dir, transform=transform_test)
        full_test_view = ConcatDataset([d1_te, d2_te])

    else:
        raise ValueError("Unknown dataset")

    return full_train_view, full_test_view

def get_labels_from_concat_dataset(concat_dataset):
    all_labels = []
    for dataset in concat_dataset.datasets:
        if hasattr(dataset, 'targets'):
            all_labels.append(np.array(dataset.targets))
        elif hasattr(dataset, 'labels'):
            all_labels.append(np.array(dataset.labels))
        elif hasattr(dataset, 'train_labels'):
            all_labels.append(dataset.train_labels.numpy())
        else:
            raise AttributeError("Dataset has no targets/labels attribute.")
    return np.concatenate(all_labels)


def generate_dirichlet_proportions(y_total, n_nets, alpha=0.5, min_require_size=10, seed=42):
    np.random.seed(seed)
    n_total = y_total.shape[0]
    K = np.max(y_total) + 1

    print(f"Generating Dirichlet Proportions based on TOTAL Data (Train+Test)...")

    while True:
        current_proportions = []
        simulated_idx_batch = [[] for _ in range(n_nets)]

        for k in range(K):
            idx_k = np.where(y_total == k)[0]
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            proportions = np.array(
                [p * (len(idx_j) < n_total / n_nets) for p, idx_j in zip(proportions, simulated_idx_batch)])
            p_sum = proportions.sum()
            if p_sum > 0:
                proportions = proportions / p_sum
            else:
                proportions = np.array([1 / n_nets] * n_nets)

            current_proportions.append(proportions)

            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch_split = np.split(idx_k, split_points)

            for i in range(n_nets):
                simulated_idx_batch[i].extend(idx_batch_split[i])

        min_size = min([len(idx_j) for idx_j in simulated_idx_batch])

        if min_size >= min_require_size:
            print(f"Proportions generated successfully. Min client size: {min_size}")
            return current_proportions
        else:
            continue


def partition_via_proportions(y, proportions_matrix, n_nets, seed=42):
    np.random.seed(seed)
    K = np.max(y) + 1
    net_dataidx_map = {}
    idx_batch = [[] for _ in range(n_nets)]

    for k in range(K):
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)

        proportions = proportions_matrix[k]
        split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch_split = np.split(idx_k, split_points)

        for i in range(n_nets):
            idx_batch[i] += idx_batch_split[i].tolist()

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def get_dataloader_PFL(dataset_name, n_users, batch_size, alpha=0.5, seed=42, root='./data', train_ratio=0.75):
    transform_train, transform_test = get_transforms(dataset_name)
    full_train_view, full_test_view = get_full_datasets(dataset_name, root, transform_train, transform_test)
    y_total = get_labels_from_concat_dataset(full_train_view)

    proportions_matrix = generate_dirichlet_proportions(
        y_total, n_users, alpha, min_require_size=10, seed=seed
    )

    print("Partitioning Total Data...")
    user_global_indices = partition_via_proportions(
        y_total, proportions_matrix, n_users, seed=seed
    )

    train_loaders = []
    test_loaders = []
    train_user_groups = {}

    for i in range(n_users):
        indices = user_global_indices[i]
        train_user_groups[i] = indices
        client_labels = y_total[indices]
        try:
            idx_train, idx_test = train_test_split(
                indices,
                train_size=train_ratio,
                shuffle=True,
                stratify=client_labels,
                random_state=seed
            )
        except ValueError:
            idx_train, idx_test = train_test_split(
                indices,
                train_size=train_ratio,
                shuffle=True,
                stratify=None,
                random_state=seed
            )

        # np.random.shuffle(indices)
        # num_samples = len(indices)
        # train_len = int(num_samples * train_ratio)
        # idx_train = indices[:train_len]
        # idx_test = indices[train_len:]

        subset_train = Subset(full_train_view, idx_train)
        subset_test = Subset(full_test_view, idx_test)
        loader_train = DataLoader(subset_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        loader_test = DataLoader(subset_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        train_loaders.append(loader_train)
        test_loaders.append(loader_test)

    return train_loaders, test_loaders, train_user_groups




class CustomDataset(Dataset):
    """用于根据指定的索引列表，从全局数据集中提取出客户端本地数据集"""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def ram_dom_gen(total, size):
    """复刻你提供的随机切割逻辑，并加入了上下界的健壮性保护"""
    if size == 1:
        return [total]
    nums = []
    total_tmp = total
    for i in range(size - 1):
        low = total // (size + 1)
        high = total // max(2, size - 1)
        if low >= high:
            high = low + 1  # 防止 randint 报错
        val = np.random.randint(low, high)
        nums.append(val)
        total_tmp -= val
    nums.append(total_tmp)
    return nums


def get_dataloader_PFL_label_skew(dataset_name, n_users, batch_size, num_labels_per_user=2, seed=1, train_ratio=0.75,
                                  root='./data/'):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. 自动加载对应的数据集
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        num_classes = 10
        # 提取 target 以供索引
        targets = np.array(train_dataset.targets + test_dataset.targets)
    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        num_classes = 10
        targets = np.array(train_dataset.targets.tolist() + test_dataset.targets.tolist())
    elif dataset_name.lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
        num_classes = 100
        targets = np.array(train_dataset.targets + test_dataset.targets)
    else:
        raise ValueError("不支持的数据集")

    # 合并 Train 和 Test 构成总池子
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    # 获取每个类别的全部样本索引
    class_indices = {i: np.where(targets == i)[0] for i in range(num_classes)}
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])  # 打乱该类的样本索引

    # 2. 核心逻辑：给每个用户分配标签
    users_labels = []
    for user in range(n_users):
        for j in range(num_labels_per_user):
            l = (user * num_labels_per_user + j) % num_classes
            users_labels.append((user, l))

    # 统计每个类被多少个客户端需要
    class_counts = {i: 0 for i in range(num_classes)}
    for _, l in users_labels:
        class_counts[l] += 1

    # 3. 使用你的 ram_dom_gen 生成每个类别被切割的具体数量
    class_splits = {}
    for c in range(num_classes):
        total_samples = len(class_indices[c])
        count = class_counts[c]
        if count > 0:
            class_splits[c] = ram_dom_gen(total_samples, count)
        else:
            class_splits[c] = []

    # 4. 根据生成的切割数量，向客户端发放大锅里的数据索引
    user_indices = {i: [] for i in range(n_users)}
    class_split_idx = {i: 0 for i in range(num_classes)}  # 记录该类分到第几份了

    for user, l in users_labels:
        # 当前用户分得该类的样本数量
        num_samples = class_splits[l][class_split_idx[l]]
        class_split_idx[l] += 1

        # 从该类大锅中切出相应数量的索引
        assigned_idx = class_indices[l][:num_samples].tolist()
        class_indices[l] = class_indices[l][num_samples:]  # 切完剩下的

        user_indices[user].extend(assigned_idx)

    # 5. 生成 Dataloader 并封装
    train_loaders = []
    test_loaders = []
    stats = []

    for user in range(n_users):
        indices = user_indices[user]
        random.shuffle(indices)  # 打乱该用户拿到的所有混合类别索引

        num_samples = len(indices)
        train_len = int(train_ratio * num_samples)  # 75% 训练，25% 测试

        train_idx = indices[:train_len]
        test_idx = indices[train_len:]

        train_ds = CustomDataset(full_dataset, train_idx)
        test_ds = CustomDataset(full_dataset, test_idx)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

        stats.append(num_samples)

    print(
        f"Data Generation Finish! Total clients: {n_users}, Pathological Non-IID ({num_labels_per_user} classes/client).")
    return train_loaders, test_loaders, stats

