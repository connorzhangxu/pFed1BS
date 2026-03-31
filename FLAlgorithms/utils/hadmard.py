import torch
import math
def get_outer_A(hparams,seed=42,com_rate=0.1):
    torch.manual_seed(seed)
    m = hparams.size()[0]
    # m1 = m // 10
    m1 = int(m * com_rate)
    scale = 1 / math.sqrt(m1)
    A_outer = scale * torch.randn(m1, m, device=hparams.device)
    return A_outer
def compute_pad_width(x):
    n = x.shape[-1]
    n_padded = 1 << (n - 1).bit_length()
    # print((n_padded > 0) and ((n_padded & (n_padded - 1)) == 0))

    return n_padded - n


def pad_tensor(x):
    # padded_chunks = []
    pad_width = compute_pad_width(x)
    pad_shape = list(x.shape)
    pad_shape[-1] = pad_width
    y = torch.zeros(pad_shape).to(device=x.device)
    x_padded = torch.cat([x, y], dim=-1)

    ori_shape = x.shape[-1]

    return x_padded, ori_shape


def fast_hadamard_transform(x, seed=42, com_rate=0.5, return_padding_shape=False):
    # x,priginal_n=pad_to_pow2(x)
    x, original_n = pad_tensor(x)
    n = x.shape[-1]
    h = 1
    while h < n:
        x = (x.reshape(-1, h * 2)
             .reshape(-1, 2, h)
             .transpose(0, 1)
             .contiguous())
        x = torch.stack((x[0].add(x[1]), x[0].add(-x[1])), dim=1)
        # torch.cuda.empty_cache()
        x = x.reshape(-1, h * 2)
        h *= 2
    # x = x / (n ** 0.5)

    m = int(original_n * com_rate)
    x = x / (m ** 0.5)
    torch.manual_seed(seed)
    idx = torch.randperm(n)[:m].to(device=x.device)
    x_compressed = x[..., idx]
    if return_padding_shape:
        return x_compressed, n
    else:
        return x_compressed


def fast_hadamard_transform_inverse(x_compressed, original_n, padding_shape, seed=42, com_rate=0.5):
    # n = padding_shape
    m = int(original_n * com_rate)
    torch.manual_seed(seed)
    # print("1:", torch.cuda.memory_allocated() / 1024 ** 2)
    idx = torch.randperm(padding_shape)[:m].to(device=x_compressed.device)
    x_full = torch.zeros((x_compressed.shape[0], padding_shape)).to(device=x_compressed.device)
    x_full[..., idx] = x_compressed

    h = 1
    while h < padding_shape:
        x_full = (x_full.reshape(-1, h * 2)
                  .reshape(-1, 2, h)
                  .transpose(0, 1)
                  .contiguous())
        x_full = torch.stack((x_full[0] + x_full[1], x_full[0] - x_full[1]),
                             dim=1)
        x_full = x_full.reshape(-1, h * 2)
        h *= 2
    # x_full = x_full / (padding_shape ** 0.5)
    # print("2:", torch.cuda.memory_allocated() / 1024 ** 2)
    x_full = x_full / (m ** 0.5)

    x_full = x_full[..., :original_n]

    return x_full


def hard_thresholding(x, tau):
    return torch.sign(x) * torch.maximum(torch.abs(x) - tau, torch.zeros_like(x))



def biht(shape_tensor,b, max_iter=1000, lambda_=1e-3, tau=1e-3,seed=42):
    torch.manual_seed(seed)
    device = b.device
    A=get_outer_A(shape_tensor,seed=seed)
    A = A.to(device)
    b = b.to(device)

    M, N = A.shape
    x = torch.matmul(A.transpose(0, 1), b)  # A^T * b

    for t in range(max_iter):
        residual = b - torch.matmul(A, x)  # b - A * x
        gradient = torch.matmul(A.T, residual)  # A^T * residual
        x = x + lambda_ * gradient
        x = hard_thresholding(x, tau)

    return x


def biht_with_hadamard(b, original_shape,padding_shape,  com_rate=0.1,max_iter=1000, lambda_=1e-3, tau=1e-3, device="cuda"):
    x = fast_hadamard_transform_inverse(b, original_n=original_shape, padding_shape=padding_shape,com_rate=com_rate)  # A^T *

    for t in range(max_iter):
        residual = b - fast_hadamard_transform(x,com_rate=com_rate)  # b - A * x
        gradient = fast_hadamard_transform_inverse(residual, original_n=original_shape,
                                                   padding_shape=padding_shape,com_rate=com_rate)  # A^T * residual
        x = x + lambda_ * gradient
        x = hard_thresholding(x, tau)

    return x
