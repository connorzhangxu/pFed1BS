# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/utils/train_utils.py
# credit goes to: Paul Pu Liang

from torchvision import datasets, transforms
# from FLAlgorithms.trainmodel.models import *
from data.sampling import noniid
import os
import json
from utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_lt
from utils.sampling import femnist_iid, femnist_noniid, femnist_noniid_unequal, femnist_noniid_lt
from utils.sampling import cifar_iid, cifar100_noniid, cifar10_noniid, cifar100_noniid_lt, cifar10_noniid_lt
from utils import femnist

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])


def get_data(args):
    if args.dataset == 'Mnist':
        dataset_train = datasets.MNIST('data/Mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/Mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, 10)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, 10, rand_set_all=rand_set_all)
    elif args.dataset == 'Cifar10':
        dataset_train = datasets.CIFAR10('data/Cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/Cifar10', train=False, download=True, transform=trans_cifar10_val)
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, 10)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, 10, rand_set_all=rand_set_all)
    elif args.dataset == 'Cifar100':
        args.num_classes = 100
        dataset_train = datasets.CIFAR100('data/Cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/Cifar100', train=False, download=True, transform=trans_cifar100_val)
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


# def get_model(args):
#     if args.model == 'cnn' and 'cifar100' in args.dataset:
#         net_glob = CNNCifar100(args=args).to(args.device)
#     elif args.model == 'cnn' and 'cifar10' in args.dataset:
#         net_glob = CNNCifar(args=args).to(args.device)
#     elif args.model == 'dnn' and 'mnist' in args.dataset:
#         net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
#     elif args.model == 'cnn' and 'femnist' in args.dataset:
#         net_glob = CNN_FEMNIST(args=args).to(args.device)
#     elif args.model == 'mlp' and 'cifar' in args.dataset:
#         net_glob = MLP(dim_in=3072, dim_hidden=512, dim_out=args.num_classes).to(args.device)
#     elif 'sent140' in args.dataset:
#         net_glob = model = RNNSent(args,'LSTM', 2, 25, 128, 1, 0.5, tie_weights=False).to(args.device)
#     else:
#         exit('Error: unrecognized model')
#     print(net_glob)

#     return net_glob

def get_data_proto(args, n_list, k_list):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = './data/' + args.dataset
    if args.dataset == 'Mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            # if args.unequal:
            #     # Chose uneuqal splits for every user
            #     user_groups = mnist_noniid_unequal(args, train_dataset, args.num_users)
            # else:
            # Chose euqal splits for every user
            user_groups, classes_list = mnist_noniid(args, train_dataset, args.num_users, n_list, k_list)
            user_groups_lt = mnist_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)
            classes_list_gt = classes_list

    elif args.dataset == 'FMnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = femnist.FEMNIST(args, data_dir, train=True, download=True,
                                        transform=apply_transform)
        test_dataset = femnist.FEMNIST(args, data_dir, train=False, download=True,
                                       transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from FMnist
            user_groups = femnist_iid(train_dataset, args.num_users)
            # print("TBD")
        else:
            # Sample Non-IID user data from FMnist
            if args.unequal:
                # Chose uneuqal splits for every user
                # user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                user_groups = femnist_noniid_unequal(args, train_dataset, args.num_users)
                # print("TBD")
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = femnist_noniid(args, args.num_users, n_list, k_list)
                user_groups_lt = femnist_noniid_lt(args, args.num_users, classes_list)

    elif args.dataset == 'Cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Cifar10
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
               # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = cifar10_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar10_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)

    elif args.dataset == 'Cifar100':
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=trans_cifar100_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=trans_cifar100_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Cifar100
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Cifar100
            # Chose euqal splits for every user
            user_groups, classes_list = cifar100_noniid(args, train_dataset, args.num_users, n_list, k_list)
            user_groups_lt = cifar100_noniid_lt(test_dataset, args.num_users, classes_list)

    return train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt