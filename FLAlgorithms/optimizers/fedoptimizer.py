from torch.optim import Optimizer
import torch

'''这段代码定义了一个自定义的随机梯度下降（SGD）优化器类 MySGD，它继承自基类 Optimizer。让我解释一下其中的关键部分：

1.__init__ 方法：


2.初始化函数接受参数 params 和 lr，其中 params 是一个包含模型参数的迭代器，lr 是学习率。
3.它创建了一个包含默认参数的字典 defaults，其中包含了学习率 lr。
4.使用 super().__init__() 调用基类的构造函数，将参数 params 和 defaults 传递给基类，实现了对参数和默认参数的初始化。


5.step 方法：


6.这个方法执行一步优化，根据参数组中的梯度更新模型参数。
7.如果提供了 closure 函数，它会执行该函数并返回损失值。
8.然后它遍历参数组 param_groups 中的每个参数 p，对其应用梯度下降更新规则。
9.如果 beta 不等于 0，则应用带有动量的梯度下降更新规则，否则应用标准的梯度下降更新规则。
10.最后，它返回损失值（如果有的话）。

总的来说，这段代码实现了一个简单的随机梯度下降优化器，具有基本的学习率和动量参数。'''


class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, beta=0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if beta != 0:
                    p.data.add_(-beta * d_p)
                else:
                    p.data.add_(-group['lr'] * d_p)
        return loss


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001, gamma=0.0, rho=6e-05):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu, gamma=gamma, rho=rho)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                g_phi = torch.tanh(p.data / group['rho'])
                p.data = p.data - group['lr'] * (
                        p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data + group[
                    'gamma'] * g_phi)
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        # return  p.data
        return group['params']

class FedMacOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, gamma=0.001, mu=1, rho=6e-05):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, gamma=gamma, mu=mu, rho=rho)
        super(FedMacOptimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                g_phi = torch.tanh(p.data / group['rho'])
                p.data = p.data - group['lr'] * (
                        p.grad.data + group['lamda'] * (- localweight.data) + group['gamma'] * g_phi)
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        # return  p.data
        return group['params']


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])
