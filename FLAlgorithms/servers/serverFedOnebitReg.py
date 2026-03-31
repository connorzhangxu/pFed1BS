import torch
import os
from FLAlgorithms.users.userFedOnebitReg import UserOnebitReg
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
from FLAlgorithms.utils.quant import quan_average_gradients, quan_average_gradients_with_hadamard
from FLAlgorithms.utils.comm_cost import get_quant_model_size
import copy


# Implementation for FedAvg Server

class FedOnebitReg(Server):
    def __init__(self, args):
        super().__init__(args)

        # Initialize data for all users
        data = read_data(args.dataset)
        total_users = len(data[0])
        for i in range(total_users):
            id, train, test = read_user_data(i, data, args.dataset)
            user = UserOnebitReg(args, id, train, test)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        self.communication_list = []
        self.client_models = [copy.deepcopy(user.model) for user in self.users]
        # print(len(self.client_models))
        self.args = args

        self.total_users = total_users
        print("Finished creating FedOnebit server.")

    def downlink_compress_and_aggregate(self, seed):
        seed = seed
        original_shape = None
        padding_shape = None

        if self.args.hadamard:
            # print("1:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
            w_local_named_params = [dict(self.client_models[0].named_parameters())]

            # print("2:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
            avg_state_dict, original_shape, padding_shape = quan_average_gradients_with_hadamard(
                w_locals=w_local_named_params,
                com_rate=self.args.cr,
                seed=seed
            )
        else:
            avg_state_dict = quan_average_gradients(dict(self.client_models[0].named_parameters()), com_rate=self.args.cr,
                                                    seed=seed)
        # print("3:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
        model_size = get_quant_model_size(avg_state_dict)
        comm_download_cost = self.num_users * model_size
        return avg_state_dict, original_shape, padding_shape, comm_download_cost

    # def set_parameters(self):
    #     assert (self.users is not None and len(self.users) > 0)
    #     for user in self.users:
    #         user.set_parameters(self.model)

    def train(self):
        seed = 42
        loss = []
        # self.set_parameters()

        avg_state_dict, original_shape, padding_shape, comm_download_cost = self.downlink_compress_and_aggregate(
            seed=seed)

        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            # loss_ = 0
            self.selected_users = self.select_users(glob_iter, round(self.num_users * self.frac))

            print("Number of users / total users:", round(self.num_users * self.frac), " / ", self.total_users)

            # avg_state_dict, original_shape, padding_shape, comm_download_cost = self.downlink_compress_and_aggregate(
            #     seed=seed)
            all_user_quant_weights = torch.zeros_like(avg_state_dict).to(self.args.device)
            for user in self.selected_users:
                user.train(args=self.args,
                           avg_state_dict=avg_state_dict,
                           seed=seed,
                           original_shape=original_shape,
                           padding_shape=padding_shape)  # * user.train_samples
                with torch.no_grad():
                    all_user_quant_weights += (user.train_samples * user.quant_model(args=self.args, seed=seed))
            all_user_quant_weights = all_user_quant_weights / len(self.selected_users)
            avg_state_dict = torch.sign(all_user_quant_weights)

            upload_cost = comm_download_cost
            self.communication_list.append(comm_download_cost + upload_cost)

            # Evaluate model each interation
            if self.task_type == "classification":
                self.evaluate()
            elif self.task_type == "regression":
                self.evaluate_regression()
            # loss_ /= self.total_train_samples
            # loss.append(loss_)
            # print(loss_)
        # print(loss)
        self.save_results()
        # self.save_model()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        par_sparsity = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            par_sparsity.append(c.sparsity * 1.0)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, par_sparsity

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        # self.cal_sparsity(self.model.parameters())
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        # self.rs_glob_sparsity.append(self.sparsity)
        # print("stats_train[1]",stats_train[3][0])
        print(f"Test Acc: {glob_acc * 100:.2f}%")
        # print("Global Trainning Accurancy: ", train_acc)
        print(f"Average Loss: {train_loss:.4f}")
        # print(f"Model Sparsity: {self.sparsity:.4f}")
