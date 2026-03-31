import torch
from FLAlgorithms.utils.hadmard import fast_hadamard_transform, fast_hadamard_transform_inverse
from FLAlgorithms.utils.quant import get_outer_A


class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.1, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.alpha = alpha
        super(MyOptimizer, self).__init__(params, defaults)

    def step(self, closure=None, local_net=None, global_net=None, outers_dict=None):
        # outer_dict_keys = [key for key in outers_dict.keys()]

        with torch.no_grad():
            for group in self.param_groups:
                momentum = group.get("momentum", 0.0)
                lr = group["lr"]
                weight_decay=group["weight_decay"]
                # ===============add grad============
                temp_flat = torch.cat([param.view(-1) for param in group['params']], -1)
                outer_product = torch.matmul(outers_dict, temp_flat)
                sign_outer_product = torch.sign(outer_product)
                temp_minus = sign_outer_product - global_net
                grad_outer_local = torch.matmul(outers_dict.T, temp_minus)
                # ===================================
                cumsum = [0]

                # for idx, param in enumerate(group['params']):
                for param in group['params']:
                    if param.grad is None:
                        continue
                    # ===============add grad============
                    # param_flat = param.view(-1)
                    # outer = outers_dict[outer_dict_keys[idx]]
                    # outer_product = torch.matmul(outer, param_flat)
                    # sign_outer_product = torch.sign(outer_product)
                    # global_param = global_net[outer_dict_keys[idx]]
                    # grad_outer = torch.matmul(outer.T, sign_outer_product - global_param)

                    num = cumsum[-1] + param.numel()
                    cumsum.append(num)
                    grad_outer = grad_outer_local[cumsum[-2]:cumsum[-1]]

                    grad = param.grad.data + self.alpha * grad_outer.view(param.shape)
                    # ===================================
                    # grad = param.grad.data
                    if weight_decay !=0:
                        grad.add_(param.data,alpha=weight_decay)
                    # ==========Add momentum buffer=============
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad)

                    # lr = group['lr']

                    # param.data = param.data - lr * grad
                    param.data.add_(-lr, buf)

    def step_with_hadamard(self, closure=None, local_net=None, global_net=None, original_shape=None,
                           padding_shape=None, com_rate=0.1, seed=42):

        with torch.no_grad():
            for group in self.param_groups:
                momentum = group.get("momentum", 0.0)
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                # ===============add grad============
                temp_flat = torch.cat([param.view(-1) for param in group['params']], -1)
                # print("before compression:",torch.cuda.memory_allocated(device='cuda:1')/1024**2)
                outer_product = fast_hadamard_transform(temp_flat, com_rate=com_rate, seed=seed)
                # print("after compression:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
                torch.cuda.empty_cache()
                sign_outer_product = torch.sign(outer_product).add(-global_net)
                temp_minus = sign_outer_product

                # print("before compression:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
                grad_outer_local = fast_hadamard_transform_inverse(temp_minus, original_n=original_shape,
                                                                   padding_shape=padding_shape, com_rate=com_rate,
                                                                   seed=seed)
                # print("after compression:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
                torch.cuda.empty_cache()
                grad_outer_local = grad_outer_local.squeeze(0)
                # ===================================
                cumsum = [0]
                params_with_grad = [p for p in group['params'] if p.grad is not None]
                # total_num=sum(p.numel() for p in group['params'] if p.grad is not None)
                # print(total_num,original_shape)
                for p in params_with_grad:
                    cumsum.append(cumsum[-1] + p.numel())

                for i, p in enumerate(params_with_grad):
                    begin = cumsum[i]
                    end = cumsum[i + 1]
                    grad_outer = grad_outer_local[begin:end]
                    # p.grad.data += self.alpha * grad_outer.view_as(p.grad.data)
                    full_grad = p.grad.data + self.alpha * grad_outer.view_as(p.grad.data)
                    # ===================================
                    # lr = group['lr']
                    # p.data = p.data - lr * p.grad.data
                    if weight_decay != 0:
                        full_grad.add_(p.data, alpha=weight_decay)
                    # ==========Add momentum buffer=============
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(full_grad).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(full_grad)
                    p.data.add_(buf, alpha=-lr)

    def zero_grad(self):

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()
