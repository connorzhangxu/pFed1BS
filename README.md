
# pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching


[![AAAIPaper](https://img.shields.io/badge/AAAI%202026-Paper-blue.svg)]([https://doi.org/10.1609/aaai.v40i25.39185])
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)]([https://arxiv.org/abs/2511.1314])

This repository contains the official PyTorch implementation of the AAAI 2026 paper **"Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching" (pFed1BS)**.

## 📖 Introduction

In bandwidth-limited networks (such as IoT, V2X communications, etc.), traditional Federated Learning (FL) faces severe challenges due to bidirectional communication overhead and client-side data heterogeneity (Non-IID). 
To significantly reduce communication costs while embracing data heterogeneity, we propose **pFed1BS**, a novel personalized federated learning framework.

- **Extreme Compression**: Achieves extreme bidirectional (uplink and downlink) one-bit compression through One-Bit Random Sketching, leveraging the Fast Hadamard Transform (FHT).
- **Personalized Alignment**: Introduces a sign-based regularizer that guides local personalized models to align with the global consensus, effectively preventing model performance collapse under Non-IID data settings.

## ⚙️ Requirements

- Python 3.8+
- PyTorch 1.9.0+
- NumPy

## 🚀 Quick Start & Arguments

Before running the code, please prepare/download the corresponding datasets in the `data/` directory. You can reproduce the experiments from the paper by modifying the default parameters in `main.py`.

> **⚠️ Important Note on Data Path:** 
> The dataset path is explicitly defined in the server base class. Before running the code, please make sure to **modify the `root` path** in the `Server` initialization method (located in the server base file, e.g., configuring `root='data/'` in `get_dataloader_PFL_label_skew()`) to match your local dataset directory.

**Key arguments corresponding to the paper's experimental setup:**
- `--algorithm`: Set to `FedOnebitReg` to run the proposed **pFed1BS** algorithm.
- `--dataset`: Supports `Mnist`, `FMnist`, `Cifar10`, `Cifar100`, etc.
- `--model`: Following the paper, use `dnn` (2-layer MLP) for MNIST/FMNIST, and VGG architectures (`VGG8` or `VGG16`) for CIFAR datasets.
- `--num_users`: The total number of clients is set to `20` in the paper.
- `--iid`: Set to `0` for the Non-IID setting (data partitioned by labels), which perfectly simulates the paper's environment.
- `--num_global_iters`: Total communication rounds. The paper uses `100` to `300` rounds depending on the dataset.
- `--sign_loss_weight`: Corresponds to the regularization parameter $\lambda$ in the paper. The optimal value is **`0.0005`**.
- `--cr`: Corresponds to the compression ratio $m/n$, fixed at **`0.1`** in the paper.
- `--hadamard`: Add this flag to enable the Fast Hadamard Transform (FHT) for efficient $\mathcal{O}(n \log n)$ dimensionality reduction.

---

## 💻 Example Commands

The following commands strictly follow the experimental parameters detailed in the paper (20 clients, Non-IID distribution, compression ratio 0.1, regularization parameter $\lambda=0.0005$).

### 🌟 Running the proposed pFed1BS (Algorithm: FedOnebitReg)

**1. MNIST Dataset (Model: MLP / dnn):**
```bash
python main.py --dataset Mnist --model dnn --algorithm FedOnebitReg \
    --num_global_iters 200 --local_epochs 10 --batch_size 64 \
    --num_users 20 --frac 1.0 --iid 0 --gpu 0 \
    --sign_loss_weight 0.0005 --cr 0.1 --hadamard
```

**2. Fashion-MNIST Dataset (Model: MLP / dnn):**
```bash
python main.py --dataset FMnist --model dnn --algorithm FedOnebitReg \
    --num_global_iters 200 --local_epochs 10 --batch_size 64 \
    --num_users 20 --frac 1.0 --iid 0 --gpu 0 \
    --sign_loss_weight 0.0005 --cr 0.1 --hadamard
```

**3. CIFAR-10 Dataset (Model: VGG8):**
```bash
python main.py --dataset Cifar10 --model VGG8 --algorithm FedOnebitReg \
    --num_global_iters 300 --local_epochs 10 --batch_size 64 \
    --num_users 20 --frac 1.0 --iid 0 --gpu 0 \
    --learning_rate 0.01 --sign_loss_weight 0.0005 --cr 0.1 --hadamard
```

**4. CIFAR-100 Dataset (Model: VGG16):**
```bash
python main.py --dataset Cifar100 --model VGG16 --algorithm FedOnebitReg \
    --num_global_iters 300 --local_epochs 10 --batch_size 64 \
    --num_users 20 --frac 1.0 --iid 0 --gpu 0 \
    --learning_rate 0.01 --sign_loss_weight 0.0005 --cr 0.1 --hadamard
```

---


## 📜 Citation

If you find this code or our framework useful in your research, please consider citing our AAAI 2026 paper:

```bibtex
@inproceedings{cheng2026personalized,
  title={Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching},
  author={Cheng, Jiacheng and Zhang, Xu and Qiu, Guanghui and Zhang, Yifang and Li, Yinchuan and Feng, Kaiyuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={25},
  pages={20499--20508},
  year={2026}
}
```
