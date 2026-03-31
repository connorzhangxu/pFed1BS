import copy
import torch
import math
import copy
from .hadmard import fast_hadamard_transform


def get_outer_A(hparams, seed=42, com_rate=0.1):
    torch.manual_seed(seed)
    m = hparams.size()[0]
    # m1 = m // 10
    m1 = int(m * com_rate)
    scale = 1 / math.sqrt(m1)
    A_outer = scale * torch.randn(m1, m, device=hparams.device)
    return A_outer


def quan_average_gradients_with_hadamard(w_locals, com_rate=0.1, seed=42):
    # print("1:", torch.cuda.memory_allocated(device='cuda:1') / 1024 ** 2)
    shape_tensor = torch.cat([w_locals[0][k].view(-1) for k in w_locals[0].keys()], -1)
    w_local, padding_shape = fast_hadamard_transform(shape_tensor, return_padding_shape=True, com_rate=com_rate)
    original_shape = shape_tensor.shape[-1]

    for i in range(0, len(w_locals)):
        temp_flat = torch.cat([w_locals[i][k].view(-1) for k in w_locals[i].keys()], -1)
        outer_w = fast_hadamard_transform(seed=seed, x=temp_flat, com_rate=com_rate)

        w_local += outer_w
    w_local = w_local / len(w_locals)

    w_local_quant = torch.sign(w_local)

    return w_local_quant, original_shape, padding_shape


def quan_average_gradients(w, com_rate=0.1, seed=42):
    torch.manual_seed(seed)
    w_avg = copy.deepcopy(w)
    for key in w_avg.keys():
        w_avg[key]=torch.zeros_like(w_avg[key])

    shape_tensor = torch.cat([w[k].view(-1) for k in w_avg.keys()], -1)
    A_outer_global = get_outer_A(seed=seed, hparams=shape_tensor, com_rate=com_rate)
    temp_flat = torch.cat([w[k].view(-1) for k in w_avg.keys()], -1)
    outer_w = torch.matmul(A_outer_global, temp_flat)
    temp_flat_quan = torch.sign(outer_w)

    return temp_flat_quan


def quan_average_gradients_AWGN(w):
    w_avg = copy.deepcopy(w[0])
    index_to_size = []
    index_to_key = []
    cumsum = [0]
    for key in w_avg.keys():
        w_avg[key].zero_()
        index_to_size.append(w_avg[key].size())
        index_to_key.append(key)
        cumsum.append(cumsum[-1] + w_avg[key].numel())

    for i in range(0, len(w)):
        temp_flat = []
        for k in w_avg.keys():
            temp_flat.append(w[i][k].view(-1))

        temp_flat = torch.cat(temp_flat, -1)
        temp_flat_quan = AirComp_4QAM_AWGN(temp_flat)

        for j in range(len(cumsum) - 1):
            begin = cumsum[j]
            end = cumsum[j + 1]
            temp_flat_raw = temp_flat_quan[begin:end].view(*index_to_size[j])
            w[i][index_to_key[j]] = temp_flat_raw

    for k in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[k] += w[i][k]

        # w_avg[k] = torch.div(w_avg[k], len(w))
        mask_neg_all = w_avg[k] < 0.
        mask_pos_all = w_avg[k] > 0.
        quan_all = mask_neg_all.float() * (-1.0) + mask_pos_all.float() * 1.0
        w_avg[k].zero_().add_(quan_all)

    return w_avg


def AirComp_4QAM_AWGN(g_temp):
    x_input = g_temp.cpu().numpy()
    Quan = (x_input > 0.).astype(float) + (x_input < 0.).astype(float) * (-1.)
    x_output = Quan
    y_temp = torch.from_numpy(x_output)
    y = y_temp.to(device=g_temp.device).float()
    return y
