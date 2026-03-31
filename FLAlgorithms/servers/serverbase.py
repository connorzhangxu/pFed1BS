import torch
import os
import numpy as np
from utils.model_utils import Metrics
import copy
from FLAlgorithms.users.newuserbase import User
from data.dataset_utils import get_dataloader_PFL,get_dataloader_PFL_label_skew


class Server:
    def __init__(self, args):

        # Set up the main attributes
        # self.device = args.device
        self.dataset = args.dataset
        self.num_glob_iters = args.num_global_iters
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(args.model[0])
        self.users = []
        self.selected_users = []
        self.K = args.K
        self.num_users = args.num_users
        self.frac = args.frac
        self.beta = args.beta
        self.lamda = args.lamda
        self.algorithm = args.algorithm
        self.rs_glob_sparsity, self.rs_sparsity_per = [], []
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc, self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.times = args.times
        self.sparsity = 1
#        self.num_bits = args.num_bits
 ###       self.quan_type = args.quan_type
        self.task_type = args.task_type

        # self.train_loaders, self.test_loaders, _ = get_dataloader_PFL(
        #     dataset_name=args.dataset,
        #     n_users=args.num_users,
        #     batch_size=args.batch_size,
        #     alpha=args.alpha,
            # seed=args.seed,
            # train_ratio=0.75,
            # root='/shared/rawdata/'
        # )
        self.train_loaders, self.test_loaders, _ = get_dataloader_PFL_label_skew(
            dataset_name=args.dataset,
            n_users=args.num_users,
            batch_size=args.batch_size,
            num_labels_per_user=20,  # 每个客户端2种标签，和你的脚本设定一致
            seed=args.seed if hasattr(args, 'seed') else 1,
            train_ratio=0.75,  # 75% 训练，25% 测试，和你的脚本设定一致
            root='data/Cifar100/'
        )

        # Initialize the server's grads to zeros
        # for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        # self.send_parameters()
        '''
        变量初始化：

        non_zero：用于跟踪模型中非零参数的数量。
        total：用于跟踪模型中所有参数的总数量。
        循环遍历模型参数：

        对模型中的每个参数进行迭代。
        检查参数是否需要梯度计算（param.requires_grad），如果需要，则执行以下操作：
        计算参数的总数量，通过将参数扁平化并计算其元素数量。
        计算参数的非零数量，通过计算参数向量的 L0 范数（非零元素数量）。
        将这些值累加到 non_zero 和 total 变量中。
        计算稀疏度：

        将非零参数数量除以总参数数量，得到稀疏度。
        将计算结果保存在 self.sparsity 属性中。
        这段代码的目的是计算给定模型的稀疏度，并将结果存储在 self.sparsity 属性中。'''

    def cal_sparsity(self, model):
        non_zero, total = 0.0, 0.0
        for param in model:
            if param.requires_grad:
                total += len(torch.flatten(param).data.cpu().numpy())
                non_zero += torch.norm(param.data, p=0).data.cpu().numpy()
        self.sparsity = non_zero / total

    # def transmit_grad_norms(self):
    #     grad_norms = []
    #     grad_norms.append(user.self.grad_norm)
    #     return grad_norms

    def select_top_k_users(self, glob_iter, num_users):
        grad_norms = self.transmit_grad_norms()
        top_k_indices = sorted(range(len(grad_norms)), key=lambda i: grad_norms[i], reverse=True)[:num_users]
        top_k_users = [self.users[i] for i in top_k_indices]
        return top_k_users

    def top_k_users(self, grad_norms, num_users):
        top_k_indices = sorted(range(len(grad_norms)), key=lambda i: grad_norms[i], reverse=True)[:num_users]
        top_k_users = [self.users[i] for i in top_k_indices]
        return top_k_users

    def divide_top_k_users(self, grad_norms, top_k_users):
        # Assuming each user has a corresponding gradient norm in the same order
        # Sort the top-k users based on gradient norms
        top_k_users_with_norms = sorted(zip(top_k_users, grad_norms), key=lambda x: x[1], reverse=True)
        # Divide the top-k users into four groups based on gradient norms
        max_2 = [user for user, _ in top_k_users_with_norms[:2]]
        next_max_2 = [user for user, _ in top_k_users_with_norms[2:4]]
        min_2 = [user for user, _ in top_k_users_with_norms[4:6]]
        other = [user for user, _ in top_k_users_with_norms[6:]]
        return max_2, next_max_2, min_2, other

    def quantize_parameters(self, params, qbits):
        scale = (2 ** qbits) - 1  # Scale for quantization
        quantized_params = torch.round(params * scale) / scale
        return quantized_params

    def add_quantize_parameters(self, user, ratio, qbits):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            quantized_user_param = self.quantize_parameters(user_param.data.clone(), qbits)
            server_param.data = server_param.data + quantized_user_param * ratio

    def aggregate_quantize_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = sum(user.train_samples for user in self.selected_users)
        for user in self.selected_users:
            for user in self.max_2_users:
                qbits = 32
                self.add_quantize_parameters(user, user.train_samples / total_train, qbits)
            for user in self.next_max_2_users:
                qbits = 16
                self.add_quantize_parameters(user, user.train_samples / total_train, qbits)
            for user in self.min_2_users:
                qbits = 8
                self.add_quantize_parameters(user, user.train_samples / total_train, qbits)
            for user in self.other_users:
                qbits = 4
                self.add_quantize_parameters(user, user.train_samples / total_train, qbits)

    '''aggregate_grads 方法：

        首先，通过 assert 语句确保了 self.users 不为 None，并且用户数量大于 0。
        接着，对模型的每个参数进行迭代，并将其梯度初始化为与参数形状相同的零张量，使用 torch.zeros_like(param.data)。
        然后，对于每个用户，调用 add_grad 方法，传入用户对象和梯度聚合的比例（每个用户的训练样本数占总训练样本数的比例）。
        add_grad 方法：

        首先，通过 user.get_grads() 获取用户的梯度。
        接着，对模型的每个参数进行迭代。
        对于每个参数，将用户的梯度乘以传入的比例，并加到该参数的梯度上。
        这样，aggregate_grads 方法将用户的梯度按比例聚合到模型的梯度中，用于联合学习中的参数更新。'''

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio
            '''
        send_parameters 方法：

        首先，通过 assert 语句确保了 self.users 不为 None，并且用户数量大于 0。
        然后，对每个用户进行迭代，并调用 user.set_parameters(self.model) 方法，将当前模型的参数发送给每个用户。
        add_parameters 方法：

        该方法用于将用户的参数按比例添加到服务器端的参数中。
        首先，通过 zip 函数同时迭代服务器端参数和用户参数。
        然后，对于每一对参数，将用户参数乘以传入的比例后，加到服务器端参数上。
        aggregate_parameters 方法：

        该方法用于聚合所有选定用户的参数。
        首先，通过 assert 语句确保了 self.users 不为 None，并且用户数量大于 0。
        接着，对每个模型参数进行迭代，并将其初始化为与参数形状相同的零张量，使用 torch.zeros_like(param.data)。
        然后，计算所有选定用户的总训练样本数。
        最后，对于每个选定用户，调用 add_parameters 方法，传入用户对象和参数聚合的比例（每个用户的训练样本数占总训练样本数的比例）。
        这样，aggregate_parameters 方法将选定用户的参数按比例聚合到服务器端的模型参数中，用于联合学习中的参数更新。'''

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        # if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            '''这段代码定义了一个计算余弦相似度的函数。让我逐步解释：
            cosine_similarity 方法：
            该方法计算两个向量 u 和 v 之间的余弦相似度。
            uv = np.average(u * v)：计算两个向量对应位置元素的乘积的平均值，即内积。
            uu = np.average(np.square(u))：计算向量 u 的平方的平均值，即向量 u 的二范数的平方。
            vv = np.average(np.square(v))：计算向量 v 的平方的平均值，即向量 v 的二范数的平方。
            d = np.sqrt(uu * vv)：计算向量 u 和向量 v 的二范数的乘积的平方根，即它们的范数的乘积。
            最后，通过以下条件判断：
            如果 d 为零，则返回 0.0，避免除以零的错误。
            否则，返回内积 uv 除以范数的乘积 d，得到两个向量的余弦相似度。
            这个函数避免了使用 sklearn 或 scipy 中的余弦距离函数，因为它们在范数为零时无法正常工作，而是直接通过数学计算来实现余弦相似度的计算'''

    def cosine_similarity(self, u, v):
        # we do not use cosine distance function from sklearn or scipy
        # because they cannot work when d = 0. This function will
        # return exactly same value as torch.nn.CosineSimilarity(dim=0)
        uv = np.average(u * v)
        uu = np.average(np.square(u))
        vv = np.average(np.square(v))
        d = np.sqrt(uu * vv)
        if d == 0:
            return 0.0
        else:
            return uv / d

    '''这个函数似乎是为了计算每个客户端（用户）之间的权重。让我解释一下：

        获取客户端参数：

        首先，通过循环遍历每个用户和他们的参数，将参数展平并连接成一个大的参数集合。
        初始化客户端权重：

        使用 alpha 初始化一个二维数组 client_weight，其中 alpha 是给每个客户端自身模型的权重，对角线上的值为 alpha，其余值为 (1 - alpha) / (num_clients - 1)，即每个客户端对其他客户端的权重平均分配。
        计算客户端之间的相似度：

        计算每对客户端之间的余弦相似度，由于相似矩阵是对称的，只需要计算一半即可，因此在对角线上方进行计算，并在对角线以下对称填充相似度值。
        计算客户端之间的权重：

        对于每个客户端 i，计算其与其他客户端的相似度加权和，并将其归一化为概率值，作为客户端 i 对其他客户端的权重。
        这个函数的目的似乎是基于客户端之间的模型相似度来调整它们之间的权重，以便在联合学习或分布式学习中更有效地整合各个客户端的模型更新。'''

    def get_client_weight(self, users):
        # 1. flatten and concat all data from each client
        flatten_params_set = []
        for user_id, user in enumerate(users):
            for user_parm in user.get_parameters():
                if user_id >= len(flatten_params_set):
                    flatten_params_set.append([])
                flatten_params_set[user_id].append(user_parm.detach().cpu().numpy().flatten())

        num_clients = len(flatten_params_set)
        flatten_params_set = [np.concatenate(flatten_param) for flatten_param in flatten_params_set]

        # 2. init client_weight with alpha
        client_weight = np.ones((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    client_weight[i][j] = self.alpha
                    # alpha 为自己给自己模型的权重
                else:
                    client_weight[i][j] = (1 - self.alpha) / (num_clients - 1)

        # 3. compute product between every clients
        product = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(num_clients):
                # The similarity matrix is symmetric, so only half of the calculation is required
                if i < j:
                    product[i][j] = self.cosine_similarity(flatten_params_set[i], flatten_params_set[j])
                elif i > j:
                    product[i][j] = product[j][i]

        # 4. compute client_weight between every clients
        for i in range(num_clients):
            product_exp_sum = 0
            for j in range(num_clients):
                if i != j:
                    product_exp_sum += np.exp(product[i][j] * self.mul)
            for j in range(num_clients):
                if i != j:
                    client_weight[i][j] = (1 - client_weight[i][i]) * np.exp(
                        product[i][j] * self.mul) / product_exp_sum
                    # 就是跟自己模型的距离越近的相似程度越高的 则权重就越大
        return client_weight

    '''这段代码展示了一个可能用于联邦学习框架的类的几个方法。让我们详细解释每个方法的功能：

        保存模型 (save_model)：

        这个方法用于将当前的模型保存到指定的目录中。首先，它确保模型保存的目录存在，如果不存在，则创建该目录。然后，它使用 torch.save() 将当前模型保存到文件中，文件名为 server.pt。
        加载模型 (load_model)：

        这个方法用于从指定的路径加载先前保存的模型。它检查该路径是否存在，然后使用 torch.load() 将模型从 server.pt 文件中加载。如果该路径不存在，会引发一个断言错误。
        模型是否存在 (model_exists)：

        这个方法检查 server.pt 文件是否存在，以判断是否已经保存了模型。它返回一个布尔值，指示该文件是否存在。
        选择用户 (select_users)：

        这个方法用于在一组可能的用户中随机选择指定数量的用户，可能用于联邦学习中的客户端选择。它接受两个参数：round（当前回合数）和 num_users（要选择的用户数量）。该方法将随机选择 num_users 个用户，但不允许重复选择（replace=False）。如果 num_users 的值大于可用的用户数，则会选择全部用户。
        这个代码片段展示了在分布式学习环境中可能会用到的几个关键功能，包括模型的保存和加载，以及在多个客户端之间选择用于训练的用户。'''

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        '''
        if (num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        # np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False)  # , p=pk)

    '''这两个方法似乎是用于个性化模型参数更新和聚合的。让我逐个解释：

        personalized_update_parameters 方法：

        这个方法接受两个参数：user（用户对象）和 ratio（比率）。
        它通过将用户的本地权重更新与服务器模型的参数进行聚合来实现个性化的模型更新。
        对于每个参数，它通过用户的本地权重更新和给定的比率来更新服务器的模型参数。
        personalized_aggregate_parameters 方法：

        这个方法用于聚合多个用户的模型参数。
        它首先深度复制了模型的参数作为之前的参数。
        然后它将所有模型参数初始化为零。
        接着，它计算了所有选择用户的训练样本总数。
        对于每个选定的用户，它调用 add_parameters 方法，根据用户的训练样本数量与总样本数量之比来添加模型参数。
        最后，它使用参数 beta 对之前的参数和当前参数进行加权平均，以更新服务器的模型参数。
        personalized_aggregate_parameters_amp 方法：

        这个方法似乎与 personalized_aggregate_parameters 类似，但使用了不同的聚合策略。
        它计算了每个客户端的权重，并根据权重对客户端的模型参数进行聚合。
        对于每个选定的用户，它遍历所有其他选定的用户，将其模型参数按照客户端权重加权相加。
        然后，它更新用户的个性化模型，并调用 update_parameters_amp 方法。
        这些方法的细节取决于整个系统的上下文，例如联邦学习框架或分布式机器学习系统的具体实现细节。'''

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self, user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        # if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            # self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

    def personalized_aggregate_parameters_amp(self):
        assert (self.users is not None and len(self.users) > 0)

        client_weight = self.get_client_weight(self.selected_users)
        # num_clients = len(client_weight)
        for i, user_i in enumerate(self.selected_users):
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data)
            for j, user_j in enumerate(self.selected_users):
                for server_param, user_param in zip(self.model.parameters(), user_j.get_parameters()):
                    server_param.data += user_param.data * client_weight[i][j]
            # for param in self.model.parameters():
            #     a = param.data
            user_i.update_persionalized_model(self.model)
            user_i.update_parameters_amp(self.model)

    # Save loss, accurancy to h5 fiel
    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p":
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        if self.algorithm == "FedMac" or self.algorithm == "FedMac_p":
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate) + "_" + str(
                self.gamma) + "gam" + "_" + str(self.gamma_w) + "gw"
        if self.algorithm == "FedSPSelect" or self.algorithm == "FedSPSelect_p":
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate) + "_" + str(
                self.gamma) + "gam" + "_" + str(self.gamma_w) + "gw"
        if self.algorithm == "FedAMP" or self.algorithm == "FedAMP_p":
            alg = alg + "_" + str(self.alpha) + "alpha_" + str(self.mul) + "mul"
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 & len(self.rs_train_acc) & len(self.rs_train_loss)):
            with h5py.File("./results/" + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_parameter_sparsity', data=self.rs_glob_sparsity)
                hf.close()

        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(
            self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p":
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        if self.algorithm == "FedMac" or self.algorithm == "FedMac_p":
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate) + "_" + str(
                self.gamma) + "gam" + "_" + str(self.gamma_w) + "gw"
        if self.algorithm == "FedAMP" or self.algorithm == "FedAMP_p":
            alg = alg + "_" + str(self.alpha) + "alpha_" + str(self.mul) + "mul"
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc_per) != 0 & len(self.rs_train_acc_per) & len(self.rs_train_loss_per)):
            with h5py.File("./results/" + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.create_dataset('rs_parameter_sparsity', data=self.rs_sparsity_per)
                hf.close()
            '''这两个函数都是用于评估模型性能的方法。

                test(self) 函数通过在给定的客户端上测试最新的模型，收集每个客户端的样本数、正确预测数、损失和参数稀疏性。然后将这些信息存储在相应的列表中，并返回客户端的ID、样本数、正确预测数和参数稀疏性。参数稀疏性是指每个客户端模型的参数中为零的比例。

                train_error_and_loss(self) 函数用于计算每个客户端在训练过程中的误差和损失。它通过调用每个客户端的 train_error_and_loss() 方法来获取误差、损失和样本数。然后将这些信息存储在相应的列表中，并返回客户端的ID、样本数、正确预测数和损失。

                这两个方法都通过循环遍历 self.users 中的客户端来收集数据。然后，它们将这些数据存储在列表中，并返回客户端的ID、样本数、正确预测数和损失。'''

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

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        par_sparsity = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            par_sparsity.append(c.sparsity * 1.0)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, par_sparsity

    def test_persionalized_model_amp(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        par_sparsity = []
        for c in self.users:
            ct, ns = c.test_persionalized_model_amp()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            par_sparsity.append(c.sparsity * 1.0)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct, par_sparsity

    def train_error_and_loss_persionalized_model_amp(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model_amp()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]
        return ids, num_samples, tot_correct, losses

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate_regression(self):
        num_samples = []
        losses_mse = []
        losses_mae = []
        r2_scores = []

        for c in self.users:
            # Assume user class has a `test_regression()` method that returns:
            #   y_true, y_pred, num_samples
            y_true, y_pred, ns = c.test_regression()
            
            y_true_tensor = torch.tensor(y_true)
            y_pred_tensor = torch.tensor(y_pred)

            # Mean Squared Error (MSE)
            mse_loss = torch.nn.functional.mse_loss(y_pred_tensor, y_true_tensor, reduction='mean').item()

            # Mean Absolute Error (MAE)
            mae_loss = torch.nn.functional.l1_loss(y_pred_tensor, y_true_tensor, reduction='mean').item()


            losses_mse.append(mse_loss * ns)
            losses_mae.append(mae_loss * ns)
            num_samples.append(ns)

        total_samples = np.sum(num_samples)
        avg_mse = np.sum(losses_mse) / total_samples
        avg_mae = np.sum(losses_mae) / total_samples

        print(f"Regression Evaluation: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}")

        return {
            "mse": avg_mse,
            "mae": avg_mae,
        }

    def evaluate(self):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.cal_sparsity(self.model.parameters())
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        self.rs_glob_sparsity.append(self.sparsity)
        # print("stats_train[1]",stats_train[3][0])
        print(f"Test Acc: {glob_acc*100:.2f}%")
        # print("Global Trainning Accurancy: ", train_acc)
        print(f"Average Loss: {train_loss:.4f}")
        print(f"Model Sparsity: {self.sparsity:.4f}")

    def evaluate_amp(self):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.cal_sparsity(self.model.parameters())
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        self.rs_glob_sparsity.append(self.sparsity)
        # print("stats_train[1]",stats_train[3][0])
        print("Average local Accurancy: ", glob_acc)
        print("Average local Trainning Accurancy: ", train_acc)
        print("Average local Trainning Loss: ", train_loss)
        print("local Model Sparsity: ", self.sparsity)

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        self.rs_sparsity_per.append(np.average(stats[3]))
        # print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ", train_loss)
        print("Average Personal Model Sparsity: ", np.average(stats[3]))

    def evaluate_personalized_model_amp(self):
        stats = self.test_persionalized_model_amp()
        stats_train = self.train_error_and_loss_persionalized_model_amp()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        self.rs_sparsity_per.append(np.average(stats[3]))
        # print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ", train_loss)
        print("Average Personal Model Sparsity: ", np.average(stats[3]))

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        self.cal_sparsity(self.model.parameters())
        self.rs_sparsity_per.append(np.average(stats[3]))
        # print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ", train_loss)
        print("Average Personal Model Sparsity: ", np.average(stats[3]))
