#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import math

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os

"""
-------------
MNIST non-iid
-------------
"""


def get_1shard(ds, row_0: int, digit: int, samples: int):
    """return an array from `ds` of `digit` starting of
    `row_0` in the indices of `ds`"""

    row = row_0

    shard = list()

    while len(shard) < samples:
        if ds.train_labels[row] == digit:
            shard.append(ds.train_data[row].numpy())
        row += 1

    return row, shard


def get_F_1shard(ds, row_0: int, digit: int, samples: int):
    """return an array from `ds` of `digit` starting of
    `row_0` in the indices of `ds`"""

    row = row_0

    shard = list()

    while len(shard) < samples:
        if ds.targets[row] == digit:
            shard.append(ds.data[row].numpy())
        row += 1

    return row, shard


def create_MNIST_ds_1shard_per_client(n_clients, samples_train, samples_test):

    MNIST_train = datasets.MNIST(root="./data", train=True, download=True)
    MNIST_test = datasets.MNIST(root="./data", train=False, download=True)

    shards_train, shards_test = [], []
    labels = []

    for i in range(10):
        row_train, row_test = 0, 0
        for j in range(10):
            row_train, shard_train = get_1shard(
                MNIST_train, row_train, i, samples_train
            )
            row_test, shard_test = get_1shard(
                MNIST_test, row_test, i, samples_test
            )

            shards_train.append([shard_train])
            shards_test.append([shard_test])

            labels += [[i]]

    X_train = np.array(shards_train)
    X_test = np.array(shards_test)

    y_train = labels
    y_test = y_train

    folder = "./data/"
    train_path = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)


def create_FashionMNIST_shard(n_clients, samples_train, samples_test):

    MNIST_train = datasets.FashionMNIST(root="./data", train=True, download=True)
    MNIST_test = datasets.FashionMNIST(root="./data", train=False, download=True)

    shards_train, shards_test = [], []
    labels = []

    for i in range(10):
        row_train, row_test = 0, 0
        for j in range(10):
            row_train, shard_train = get_F_1shard(
                MNIST_train, row_train, i, samples_train
            )
            row_test, shard_test = get_F_1shard(
                MNIST_test, row_test, i, samples_test
            )

            shards_train.append([shard_train])
            shards_test.append([shard_test])

            labels += [[i]]

    X_train = np.array(shards_train)
    X_test = np.array(shards_test)

    y_train = labels
    y_test = y_train

    folder = "./data/"
    train_path = f"FashionMNIST_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"FashionMNIST_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)


def create_MNIST_small_niid(
    n_clients: int,
    samples_train: list,
    samples_test: list,
    clients_digits: list,
):

    MNIST_train = datasets.MNIST(root="./data", train=True, download=True)
    MNIST_test = datasets.MNIST(root="./data", train=False, download=True)

    X_train, X_test = [], []
    y_train, y_test = [], []

    for digits, n_train, n_test in zip(
        clients_digits, samples_train, samples_test
    ):

        client_samples_train, client_samples_test = [], []
        client_labels_train, client_labels_test = [], []

        n_train_per_shard = int(n_train / len(digits))
        n_test_per_shard = int(n_test / len(digits))

        for digit in digits:

            row_train, row_test = 0, 0
            _, shard_train = get_1shard(
                MNIST_train, row_train, digit, n_train_per_shard
            )
            _, shard_test = get_1shard(
                MNIST_test, row_test, digit, n_test_per_shard
            )

            client_samples_train += shard_train
            client_samples_test += shard_test

            client_labels_train += [digit] * n_train_per_shard
            client_labels_test += [digit] * n_test_per_shard

        X_train.append(client_samples_train)
        X_test.append(client_samples_test)

        y_train.append(client_labels_train)
        y_test.append(client_labels_test)

    folder = "./data/"
    train_path = f"MNIST_small_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((np.array(X_train), np.array(y_train)), output)

    test_path = f"MNIST_small_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((np.array(X_test), np.array(y_test)), output)


class MnistShardDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):

        with open(file_path, "rb") as pickle_file:
            dataset = pickle.load(pickle_file)
            self.features = np.vstack(dataset[0][k])

            vector_labels = list()
            for idx, digit in enumerate(dataset[1][k]):
                vector_labels += [digit] * len(dataset[0][k][idx])

            self.labels = np.array(vector_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        # 3D input 1x28x28
        x = torch.Tensor([self.features[idx]]) / 255
        y = torch.LongTensor([self.labels[idx]])[0]

        return x, y


def clients_set_MNIST_shard(file_name, n_clients, batch_size=100, shuffle=True):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = MnistShardDataset(file_name, k)
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )
        list_dl.append(dataset_dl)

    return list_dl


"""
-------
MNIST 
Dirichilet distribution
----
"""

def partition_MNIST_dataset(
    dataset,
    file_name: str,
    balanced: bool,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
):
    """Partition dataset into `n_clients`.
    Each client i has matrix[k, i] of data of class k"""

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    list_clients_shannon = []

    if balanced and train:
        n_samples = [600] * n_clients
    elif balanced and not train:
        n_samples = [100] * n_clients
    elif not balanced and train:
        n_samples = (
                [120] * 2 + [300] * 10 + [600] * 10 + [900] * 5 + [1200] * 3 + [120] * 8 + [300] * 20 + [600] * 20 + [
            900] * 15 + [1200] * 7
        )
    elif not balanced and not train:
        n_samples = [20] * 2 + [50] * 10 + [100] * 10 + [150] * 5 + [200] * 3 + [20] * 8 + [50] * 20 + [100] * 20 + [
            150] * 15 + [200] * 7


    list_idx = []
    for k in range(n_classes):

        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]             # 第一维：标签值 第二维：属于该标签的条目下标

    for idx_client, n_sample in enumerate(n_samples):
        # 客户下标    客户样本数

        clients_idx_i = []  # client_i 分到的数据条目下标
        client_samples = 0
        client_shannon = 0  # 香农指数

        for k in range(n_classes):

            if k < n_classes-1:
                samples_digit = int(matrix[idx_client, k] * n_sample)
            if k == n_classes-1:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            p_k = samples_digit / n_sample          # 样本比例
            if p_k != 0:
                client_shannon -= p_k * math.log(p_k)   # 香农指数计算

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit))   # 将标签k的数据随机选取分给client_i
            )

        # clients_idx_i 当前客户所持有数据 在数据集中的下标
        clients_idx_i = clients_idx_i.astype(int)

        list_clients_shannon.append(client_shannon)

        for idx_sample in clients_idx_i:

            list_clients_X[idx_client] += [dataset.data[idx_sample].numpy()]       # 客户idx_client 数据样本
            list_clients_y[idx_client] += [dataset.targets[idx_sample].tolist()]    # 客户idx_client 数据标签

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y, list_clients_shannon), output)


def create_MNIST_dirichlet(
    dataset_name: str,
    balanced: bool,
    alpha: float,
    n_clients: int,
    n_classes: int,
):

    from numpy.random import dirichlet

    # matrix = dirichlet([alpha] * n_classes, size=n_clients)
    matrix = np.concatenate((dirichlet([alpha * 100] * n_classes, size=20),
                             dirichlet([alpha * 10000] * n_classes, size=10),
                             dirichlet([alpha] * n_classes, size=70)))

    MNIST_train = datasets.MNIST(root="./data", train=True, download=True)
    MNIST_test = datasets.MNIST(root="./data", train=False, download=True)

    file_name_train = f"{dataset_name}_train_{n_clients}.pkl"
    partition_MNIST_dataset(
        MNIST_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
    partition_MNIST_dataset(
        MNIST_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )


def create_FashionMNIST_dirichlet(
    dataset_name: str,
    balanced: bool,
    alpha: float,
    n_clients: int,
    n_classes: int,
):

    from numpy.random import dirichlet

    # matrix = dirichlet([alpha] * n_classes, size=n_clients)
    matrix = np.concatenate((dirichlet([alpha * 100] * n_classes, size=20),
                             dirichlet([alpha * 10000] * n_classes, size=10),
                             dirichlet([alpha] * n_classes, size=70)))

    MNIST_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    MNIST_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    file_name_train = f"{dataset_name}_train_{n_clients}.pkl"
    partition_MNIST_dataset(
        MNIST_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
    partition_MNIST_dataset(
        MNIST_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )


class MNISTDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):

        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        yt = np.array(dataset[1][k])
        self.y = torch.from_numpy(yt).type(torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        # 3D input 1x28x28
        x = torch.Tensor(np.array([self.X[idx]])) / 255
        y = self.y[idx]

        return x, y


def clients_set_MNIST(file_name, n_clients, batch_size=64, shuffle=True):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = MNISTDataset(file_name, k)
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )
        list_dl.append(dataset_dl)

    return list_dl


"""
-------
CIFAR
Dirichilet distribution
----
"""


def partition_CIFAR_dataset(
    dataset,
    file_name: str,
    balanced: bool,
    matrix,
    n_clients: int,
    n_classes: int,
    train: bool,
):
    """Partition dataset into `n_clients`.
    Each client i has matrix[k, i] of data of class k"""

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]
    list_clients_shannon = []

    if balanced and train:
        n_samples = [500] * n_clients
    elif balanced and not train:
        n_samples = [100] * n_clients
    elif not balanced and train:
        n_samples = (
                [100] * 2 + [250] * 10 + [500] * 10 + [750] * 5 + [1000] * 3 + [100] * 8 + [250] * 20 + [500] * 20 + [
            750] * 15 + [1000] * 7
        )
    elif not balanced and not train:
        n_samples = [20] * 2 + [50] * 10 + [100] * 10 + [150] * 5 + [200] * 3 + [20] * 8 + [50] * 20 + [100] * 20 + [
            150] * 15 + [200] * 7

    list_idx = []
    for k in range(n_classes):

        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]             # 第一维：标签值 第二维：属于该标签的条目下标

    for idx_client, n_sample in enumerate(n_samples):
        # 客户下标    客户样本数

        clients_idx_i = []  # client_i 分到的数据条目下标
        client_samples = 0
        client_shannon = 0  # 香农指数

        for k in range(n_classes):

            if k < n_classes - 1:
                samples_digit = int(matrix[idx_client, k] * n_sample)
            if k == n_classes - 1:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            p_k = samples_digit / n_sample          # 样本比例
            if p_k != 0:
                client_shannon -= p_k * math.log(p_k)   # 香农指数计算

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit))   # 将标签k的数据随机选取分给client_i
            )

        # clients_idx_i 当前客户所持有数据 在数据集中的下标
        clients_idx_i = clients_idx_i.astype(int)

        list_clients_shannon.append(client_shannon)

        for idx_sample in clients_idx_i:

            list_clients_X[idx_client] += [dataset.data[idx_sample]]       # 客户idx_client 数据样本
            list_clients_y[idx_client] += [dataset.targets[idx_sample]]    # 客户idx_client 数据标签

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = "./data/"
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y, list_clients_shannon), output)


def create_CIFAR10_dirichlet(
    dataset_name: str,
    balanced: bool,
    alpha: float,
    n_clients: int,
    n_classes: int,
):
    """ 根据Dir(alpha)创建CIFAR-10数据集"""

    from numpy.random import dirichlet

    # matrix = dirichlet([alpha] * n_classes, size=n_clients)
    matrix = np.concatenate((dirichlet([alpha * 100] * n_classes, size=20),
                             dirichlet([alpha * 10000] * n_classes, size=10),
                             dirichlet([alpha] * n_classes, size=70)))

    CIFAR10_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    CIFAR10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset_name}_train_{n_clients}.pkl"
    partition_CIFAR_dataset(
        CIFAR10_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
    partition_CIFAR_dataset(
        CIFAR10_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )


def create_CIFAR100_dirichlet(
        dataset_name: str,
        balanced: bool,
        alpha: float,
        n_clients: int,
        n_classes: int,
):
    """ 根据Dir(alpha)创建CIFAR-100数据集"""

    from numpy.random import dirichlet

    # matrix = dirichlet([alpha] * n_classes, size=n_clients)
    matrix = np.concatenate((dirichlet([alpha * 100] * n_classes, size=20),
                             dirichlet([alpha * 10000] * n_classes, size=10),
                             dirichlet([alpha] * n_classes, size=70)))


    CIFAR100_train = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    CIFAR100_test = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    file_name_train = f"{dataset_name}_train_{n_clients}.pkl"
    partition_CIFAR_dataset(
        CIFAR100_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
    partition_CIFAR_dataset(
        CIFAR100_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )


class CIFARDataset(Dataset):
    """Convert the CIFAR pkl file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):

        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        yt = np.array(dataset[1][k])
        self.y = torch.from_numpy(yt).type(torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        # 3D input 32x32x3
        x = torch.Tensor(self.X[idx]).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        y = self.y[idx]

        return x, y


def clients_set_CIFAR(
    file_name: str, n_clients: int, batch_size: int, shuffle=True
):
    """Download for all the clients their respective dataset"""
    print("数据集文件已找到：" + file_name)

    list_dl = list()

    for k in range(n_clients):
        dataset_object = CIFARDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl


"""
---------
Upload any dataset
Puts all the function above together
---------
"""


def get_dataloaders(dataset, batch_size: int, shuffle=True):

    folder = "./data/"
    
    # 修改 新增香农指数返回列表
    clients_shannon = []

    if dataset == "MNIST_iid":

        n_clients = 100
        samples_train, samples_test = 600, 100

        mnist_trainset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_train_split = torch.utils.data.random_split(
            mnist_trainset, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_train_split
        ]

        mnist_testset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_test_split = torch.utils.data.random_split(
            mnist_testset, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_test_split
        ]

    elif dataset == "MNIST_shard":
        n_clients = 100
        samples_train, samples_test = 500, 80

        file_name_train = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            create_MNIST_ds_1shard_per_client(
                n_clients, samples_train, samples_test
            )

        list_dls_train = clients_set_MNIST_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        )

        list_dls_test = clients_set_MNIST_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )
    
    elif dataset == "FashionMNIST_iid":
        n_clients = 100
        samples_train, samples_test = 600, 100

        mnist_trainset = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_train_split = torch.utils.data.random_split(
            mnist_trainset, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(
                ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_train_split
        ]

        mnist_testset = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_test_split = torch.utils.data.random_split(
            mnist_testset, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(
                ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_test_split
        ]

    elif dataset == "FashionMNIST_shard":
        n_clients = 100
        samples_train, samples_test = 500, 80

        file_name_train = f"FashionMNIST_shard_train_{n_clients}_{samples_train}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"FashionMNIST_shard_test_{n_clients}_{samples_test}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            create_FashionMNIST_shard(
                n_clients, samples_train, samples_test
            )

        list_dls_train = clients_set_MNIST_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        )

        list_dls_test = clients_set_MNIST_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )

    elif dataset == "CIFAR10_iid":
        n_clients = 100
        samples_train, samples_test = 500, 100

        CIFAR10_train = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        CIFAR10_train_split = torch.utils.data.random_split(
            CIFAR10_train, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_train_split
        ]

        CIFAR10_test = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        CIFAR10_test_split = torch.utils.data.random_split(
            CIFAR10_test, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_test_split
        ]

    elif dataset == "CIFAR100_iid":
        n_clients = 100
        samples_train, samples_test = 500, 100

        CIFAR10_train = datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        CIFAR10_train_split = torch.utils.data.random_split(
            CIFAR10_train, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(
                ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_train_split
        ]

        CIFAR10_test = datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        CIFAR10_test_split = torch.utils.data.random_split(
            CIFAR10_test, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(
                ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_test_split
        ]

    elif dataset[:5] == "MNIST":

        n_classes = 10          # 类别?
        n_clients = 100         # client个数
        balanced = dataset[6:10] == "bbal"      # 数据分布是否均衡
        alpha = float(dataset[11:])             # 迪利克雷分布alpha参数，越小越不均匀

        file_name_train = f"{dataset}_train_{n_clients}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset}_test_{n_clients}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            print("无数据集文件！creating dataset alpha:", alpha)
            create_MNIST_dirichlet(
                dataset, balanced, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_MNIST(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_MNIST(
            path_test, n_clients, batch_size, True
        )
        clients_shannon = pickle.load(open(path_train, "rb"))[2]
        print(f"MNIST_nonIID数据集加载成功！")

    elif dataset[:6] == "FMNIST":

        n_classes = 10          # 类别?
        n_clients = 100         # client个数
        balanced = dataset[7:11] == "bbal"      # 数据分布是否均衡
        alpha = float(dataset[12:])             # 迪利克雷分布alpha参数，越小越不均匀

        file_name_train = f"{dataset}_train_{n_clients}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset}_test_{n_clients}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            print("无数据集文件！creating dataset alpha:", alpha)
            create_FashionMNIST_dirichlet(
                dataset, balanced, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_MNIST(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_MNIST(
            path_test, n_clients, batch_size, True
        )
        clients_shannon = pickle.load(open(path_train, "rb"))[2]
        print(f"FashionMNIST_nonIID数据集加载成功！")

    elif dataset[:8] == "CIFAR100":

        n_classes = 100         # 类别?
        n_clients = 100         # client个数
        alpha = float(dataset[9:])             # 迪利克雷分布alpha参数，越小越不均匀
        print(f"alpha: {alpha}")

        file_name_train = f"{dataset}_train_{n_clients}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset}_test_{n_clients}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            print("creating dataset alpha:", alpha)
            create_CIFAR100_dirichlet(
                dataset, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_CIFAR(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_CIFAR(
            path_test, n_clients, batch_size, True
        )
        clients_shannon = pickle.load(open(path_train, "rb"))[2]
        print(f"CIFAR100_nonIID数据集加载成功")

    elif dataset[:5] == "CIFAR":

        n_classes = 10          # 类别?
        n_clients = 100         # client个数
        balanced = dataset[8:12] == "bbal"      # 数据分布是否均衡
        alpha = float(dataset[13:])             # 迪利克雷分布alpha参数，越小越不均匀

        file_name_train = f"{dataset}_train_{n_clients}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset}_test_{n_clients}.pkl"
        path_test = folder + file_name_test

        if not os.path.isfile(path_train):
            print("无数据集文件！creating dataset alpha:", alpha)
            create_CIFAR10_dirichlet(
                dataset, balanced, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_CIFAR(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_CIFAR(
            path_test, n_clients, batch_size, True
        )
        clients_shannon = pickle.load(open(path_train, "rb"))[2]
        print(f"CIFAR10_nonIID数据集加载成功！")

    else:
        print("Dataset para Error!")
        import sys
        sys.exit()

    # Save in a file the number of samples owned per client
    list_len = list()
    for dl in list_dls_train:
        list_len.append(len(dl.dataset))
    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "wb") as output:
        pickle.dump(list_len, output)

    return list_dls_train, list_dls_test, clients_shannon
