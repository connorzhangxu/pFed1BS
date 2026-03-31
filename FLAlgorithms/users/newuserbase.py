import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy


class User:
    """
    Base class for users in federated learning.
    """

    def __init__(self, args, id, train_data, test_data):

        self.device = args.device
        self.model = copy.deepcopy(args.model[0])
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.num_bits = args.num_bits
        self.sparsity = 1
        self.trainloader = train_data
        self.testloader = test_data
        self.train_samples = len(self.trainloader.dataset)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        # those parameters are for pFedMe and FedMac.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        if args.task_type == "regression":
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()


    def cal_sparsity(self, model, pri=0):
        non_zero, total = 0.0, 0.0
        for param in model:
            if param.requires_grad:
                total += len(torch.flatten(param).data.cpu().numpy())
                non_zero += torch.norm(param.data, p=0).data.cpu().numpy()
        self.sparsity = non_zero / total

    def cal_grad_norm(self, model):
        grad_norm = 0
        for param in model.parameters():
            if param.grad is None:
                print("param.grad is None")
            if param.grad is not None:
                # 累加梯度的平方
                grad_norm += torch.norm(param.grad, p=2).item() ** 2
        grad_norm = grad_norm ** 0.5
        return grad_norm

    def quantize_grad(self, model, qbits):
        for param in model.parameters():
            if param.grad is None:
                print("error:param.grad is None")
            if param.grad is not None:
                grad = param.grad.data
                # 将梯度缩放到合适的范围
                grad = grad / torch.max(torch.abs(grad)) * (2 ** (qbits - 1) - 1)
                # 进行量化
                quantized_grad = torch.round(grad)
                # 将量化后的梯度缩放回原始范围
                quantized_grad = quantized_grad / (2 ** (qbits - 1) - 1) * torch.max(torch.abs(grad))
                # 将量化后的梯度赋值给模型参数的梯度
                param.grad.data = quantized_grad

    '''set_parameters(self, model): 这个方法将传入的模型参数设置为当前客户端的模型参数。它通过遍历每个参数的对应项，将新模型的参数数据克隆到当前模型和本地模型的参数中。注释部分可能是用于更新本地权重的代码，但是被注释掉了。

get_parameters(self): 这个方法用于获取当前客户端模型的参数，并将其返回。在这个方法中，模型参数的梯度被分离(detach)，以避免梯度传播到之前的训练中。'''

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        # self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    '''clone_model_paramenter(self, param, clone_param): 这个方法用于将一个参数的数据克隆到另一个参数中。它通过遍历每一对参数和克隆参数，将原始参数的数据克隆到克隆参数中，并返回克隆后的参数。

get_updated_parameters(self): 这个方法返回了一个名为local_weight_updated的属性，可能用于存储更新后的本地权重。这个属性应该在其他地方被更新过。

update_parameters(self, new_params): 这个方法用于更新模型参数。它通过遍历模型的每个参数和新参数列表，将新参数的数据克隆到模型参数中，以更新模型的参数。

update_parameters_amp(self, update_model): 这个方法似乎类似于 update_parameters 方法，但是它接受一个完整的模型 update_model，而不是只接受参数列表。它也是通过遍历模型的每个参数和新模型的参数，将新模型的参数数据克隆到模型参数中，以更新模型的参数。'''

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def clone_model_parameter_test(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def update_parameters_amp(self, update_model):
        for param, new_param in zip(self.model.parameters(), update_model.parameters()):
            param.data = new_param.data.clone()

    def update_persionalized_model(self, persionalized_model):
        self.persionalized_model_bar = copy.deepcopy(persionalized_model)
        '''首先，它创建了一个空列表 grads 来存储参数的梯度。
然后，它遍历模型的每个参数。
对于每个参数，如果参数的梯度为 None，则将一个与参数数据形状相同的零张量添加到 grads 列表中，以保持与参数相同的形状。
如果参数的梯度不为 None，则将参数的梯度数据添加到 grads 列表中。
最后，返回包含参数梯度的 grads 列表。'''

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        test_acc = 0
        total_samples = 0
        with torch.no_grad():
            for batch in self.testloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                if x.dim() == 4:
                    if x.shape[1] != 3 and x.shape[3] == 3:
                        x = x.permute(0, 3, 1, 2)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                total_samples += y.shape[0]
        if total_samples == 0:
            return 0, 0
        return test_acc, total_samples

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        total_samples = 0
        with torch.no_grad():
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if x.dim() == 4:
                    if x.shape[1] != 3 and x.shape[3] == 3:
                        x = x.permute(0, 3, 1, 2)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss += self.loss(output, y)
                total_samples += y.shape[0]

        if total_samples == 0:
            return 0, 0, 0
        return train_acc, loss, total_samples

    def test_persionalized_model(self):
        self.model.eval()
        test_acc = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0]

    def test_regression(self):
        """Returns true labels, predictions, and number of samples."""
        self.model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                preds.append(pred.cpu())
                truths.append(y.cpu())

        preds = torch.cat(preds).flatten().numpy()
        truths = torch.cat(truths).flatten().numpy()
        return truths, preds, len(truths)
    def test_persionalized_model_amp(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0]

    def train_error_and_loss_persionalized_model_amp(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        # self.update_parameters_amp(self.persionalized_model_bar)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        # self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            # print(self.id + ", Train Accuracy:", train_acc)
            # print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
