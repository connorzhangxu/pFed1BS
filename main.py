#!/usr/bin/env python
import argparse
import random
from FLAlgorithms.trainmodel.models import *
import torch
import numpy as np
import copy
import torch.utils.model_zoo as model_zoo

from FLAlgorithms.servers.serverFedOnebitReg import FedOnebitReg



def set_seed(seed=42):
    """固定所有随机种子"""
    random.seed(seed)  # Python随机模块
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU（多卡时）
    torch.backends.cudnn.deterministic = True  # 保证CUDA卷积结果确定
    torch.backends.cudnn.benchmark = False  # 关闭自动优化


# 模型URL
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}



def main(args):
    dataset = args.dataset
    model = args.model
    algorithm = args.algorithm
    device = args.device

    for i in range(args.times):
        print("---------------Running time:------------", i)
        # Generate model

        if model == "cnn":
            if dataset == "Mnist":
                model = Net().to(device), model
            elif dataset == "Cifar10":
                if args.qat == True:
                    model = QuantizedCifarNet().to(device), model
                else:
                    model = CifarNet().to(device), model
            elif dataset == "Cifar100":
                if args.qat == True:
                    model = QuantizedCifarNet().to(device), model
                else:
                    model = CifarNet().to(device), model
        if model == "dnn":
            if dataset == "Mnist":
                if args.qat == True:
                    model = QuantizedDNN().to(device), model
                else:
                    model = DNN().to(device), model
            elif dataset == "FMnist":
                if args.qat == True:
                    model = QuantizedDNN().to(device), model
                else:
                    model = DNN().to(device), model
            elif dataset == "Cifar10":
                if args.qat == True:
                    model = QuantizedDNN(3072,256,10).to(device), model
                else:
                    model = DNN(3072,256,10).to(device), model
            else:
                if args.qat == True:
                    model = QuantizedDNN(60, 20, 10).to(device), model
                else:
                    model = DNN(60, 20, 10).to(device), model

        if model == "VGG8":
            if dataset == "Cifar10":
                model = VGG(model).to(device), model
            elif dataset == "Cifar100":
                model = VGG(model, num_clasees=100).to(device), model
            else:
                print("Not deployment for", model, dataset, "task.")
                raise ValueError()

        if model == "VGG16":
            if dataset == "Cifar10":
                model = VGG(model).to(device), model
            elif dataset == "Cifar100":
                model = VGG(model, num_clasees=100).to(device), model
            else:
                print("Not deployment for", model, dataset, "task.")
                raise ValueError()

        if model == "linear":
            if dataset == "Boston":
                model = LinearRegression().to(device), model
                args.task_type = "regression"
            else:
                print("Not deployment for", model, dataset, "task.")
                raise ValueError()

        args.model = model

        if algorithm == "FedOnebitReg":
            server = FedOnebitReg(args)


        else:
            server.train()
            server.test()

    # Average data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Common
    parser.add_argument("--dataset", type=str, default="Mnist",
                        choices=["Mnist", "Cifar10", "Cifar100", "FMnist"])
    parser.add_argument("--model", type=str, default="dnn",
                        choices=["dnn", "cnn", "VGG8", "VGG16"])
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "regression"])
    parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--learning_rate", type=float, default=0.2, help="Local learning rate")
    parser.add_argument("--learning_rate", type=float, default=3000, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1,
                        help="Average moving parameter for pFedMe and FedMac, or Second learning rate of Per-FedAvg")
    parser.add_argument("--num_global_iters", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, choices=['FedOnebitReg'])
    # parser.add_argument("--num_users", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.1,
                        help="Personalized learning rate to calculate theta approximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="Num of Monte Carlo experiments")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_users', type=int, default=10, help="number of users")
    parser.add_argument('--seed', type=int, default=123, help="set seed")
    parser.add_argument('--frac', type=float, default=1.0)
    # FedMac
    parser.add_argument("--lamda", type=float, default=1e-4, help="Regularization term")
    parser.add_argument("--gamma", type=float, default=0.0003, help="Sparse Regularization term")
    parser.add_argument("--gamma_w", type=float, default=1e-08, help="Sparse Regularization term")
    parser.add_argument('--iid', type=int, default=0, help='Default set to IID. Set to 0 for non-IID.')
    # FedOnebit
    parser.add_argument('--sign_loss_weight', type=float, default=0.001)
    parser.add_argument('--cr', type=float, default=1.0)
    parser.add_argument('--hadamard', action='store_true', default=False, help="with hadamard or not")

    args = parser.parse_args()
    
    set_seed(args.seed) #123
    
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    # if args.gpu != -1:
    # print(args.gpu)
    # torch.cuda.set_device(args.gpu)

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))  # batch
    print("Learing rate       : {}".format(args.learning_rate))  # lr
    # print("Personal Learing rate       : {}".format(args.personal_learning_rate))  # plr
    print("Average Moving       : {}".format(args.beta))  # beta
    print("Number of selected users      : {}".format((args.num_users * args.frac)))  # S
    print("Number of global rounds       : {}".format(args.num_global_iters))  # T
    print("Number of local rounds       : {}".format(args.local_epochs))  # R
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Task Type       : {}".format(args.task_type))
    # print("Number of quantization bits         : {}".format(args.num_bits))
    # print("Quantization Awareness Training     : {}".format(args.qat))
    print("=" * 80)

    main(args=args)
