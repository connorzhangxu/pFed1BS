import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.newuserbase import User
from FLAlgorithms.utils.MyOptimizerReg import MyOptimizerReg
from FLAlgorithms.utils.quant import get_outer_A
from FLAlgorithms.utils.hadmard import fast_hadamard_transform


class UserOnebitReg(User):
    def __init__(self, args, numeric_id, train_data, test_data):
        super().__init__(args, numeric_id, train_data, test_data)

        if (args.model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        if (args.task_type == "regression"):
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = nn.NLLLoss()

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)

        self.weight = args.sign_loss_weight if 'sign_loss_weight' in args else 0.5

        self.optimizer = MyOptimizerReg(self.model.parameters(), lr=self.learning_rate, momentum=0.5, alpha=self.weight,
                                     weight_decay=1e-4)

    def quant_model(self, args, seed=42):
        # print("1:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
        temp_flat = torch.cat([param.view(-1) for param in self.model.parameters()], -1)
        # print("2:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
        if args.hadamard:
            outer_product = fast_hadamard_transform(x=temp_flat, seed=seed, com_rate=args.cr)
        else:
            outers_dict = get_outer_A(temp_flat, seed=seed, com_rate=args.cr)
            outer_product = torch.matmul(outers_dict, temp_flat)
        # print("3:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
        # torch.cuda.empty_cache()
        return torch.sign(outer_product)

    def train(self, args, avg_state_dict, seed=42, original_shape=None, padding_shape=None):

        LOSS = 0
        self.model.train()

        for epoch in range(1, self.local_epochs + 1):
            # self.model.train()
            # print(self.model.linear.weight)

            # print("1:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
            X, y = self.get_next_train_batch()
            # self.optimizer.zero_grad()
            # self.model.zero_grad()

            # print("2:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
            output = self.model(X)
            loss = self.loss(output, y)
            # print("3:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)

            loss.backward()
            # print("4:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
            if args.hadamard:
                self.optimizer.step_with_hadamard(local_net=self.model,
                                                  global_net=avg_state_dict,
                                                  original_shape=original_shape,
                                                  padding_shape=padding_shape,
                                                  com_rate=args.cr,
                                                  seed=seed)
            else:
                self.optimizer.step(local_net=self.model, global_net=avg_state_dict,
                                    seed=seed)
        # print("5:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
        # z_outer=self.quant_model(seed=seed)
        return LOSS
