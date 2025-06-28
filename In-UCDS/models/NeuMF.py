import torch
import torch.nn as nn

class NeuMF(torch.nn.Module):
    def __init__(self, config, user_num, item_num, device):
        super(NeuMF, self).__init__()
        # 初始化模型参数
        self.config = config  # 模型配置字典
        self.num_users = user_num  # 用户数量
        self.num_items = item_num  # 物品数量
        self.latent_dim_mf = config['latent_dim_mf']  # MF部分的嵌入维度
        self.latent_dim_mlp = config['latent_dim_mlp']  # MLP部分的嵌入维度
        self.device = device  # 计算设备（CPU/GPU）
        # MLP部分的用户和物品嵌入层
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        # MF部分的用户和物品嵌入层
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)
        # 激活函数
        self.ac_func = nn.ReLU()
        # 构建MLP的全连接层
        self.fc_layers = torch.nn.ModuleList()
        # 遍历层配置创建线性层
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # 输出层：融合MLP和MF的特征
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        # 最终评分转换为概率
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        # 获取MLP部分的用户/物品嵌入向量
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        # 获取MF部分的用户/物品嵌入向量
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        # 拼接MLP部分的用户和物品向量
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        # 计算MF部分的元素积（哈达玛积）
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        # 前向传播通过全连接层
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)  # 线性变换
            mlp_vector = torch.nn.ReLU()(mlp_vector)  # 激活函数
        # 拼接MLP和MF的输出向量
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        # 通过输出层得到预测分数
        logits = self.affine_output(vector)
        # 使用sigmoid转换为概率值
        rating = self.logistic(logits)
        return rating
    # 用于自定义损失函数：获取目标用户的MLP和MF嵌入
    def inactive_embeddings_for_my_loss(self, samples: torch.LongTensor):
        device = self.embedding_user_mlp.weight.device  # 获取嵌入层所在设备
        samples = samples.to(device)  # 确保数据在相同设备
        # 拼接目标用户的MLP和MF嵌入
        return torch.cat([self.embedding_user_mlp(samples[:, 0]), self.embedding_user_mf(samples[:, 0])])
    # 用于自定义损失函数：获取邻居用户的嵌入（不计算梯度）
    @torch.no_grad()
    def neighbor_embeddings_for_my_loss(self, samples: torch.LongTensor):
        device = self.embedding_user_mlp.weight.device
        samples = samples.to(device)
        # 拼接邻居用户的MLP和MF嵌入
        return torch.cat([self.embedding_user_mlp(samples[:, 1]), self.embedding_user_mf(samples[:, 1])])

