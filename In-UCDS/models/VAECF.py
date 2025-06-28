import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class PolyLinear(nn.Module):
    """可配置的多层线性网络模块"""
    def __init__(self, layer_config: list, activation_fn, output_fn=None, input_dropout=None):
        super().__init__()
        assert len(layer_config) > 1  # 至少需要输入和输出层
        # 存储配置参数
        self.layer_config = layer_config  # 每层的维度配置 [input_dim, hidden_dim, ..., output_dim]
        self.activation_fn = activation_fn  # 激活函数
        self.output_fn = output_fn  # 输出激活函数（可选）
        self.n_layers = len(layer_config) - 1  # 线性层数量
        # 使用有序字典构建网络层
        layer_dict = OrderedDict()
        # 输入dropout（可选）
        if input_dropout is not None:
            layer_dict["input_dropout"] = nn.Dropout(p=input_dropout)
        # 构建线性层和激活函数
        for i, (d1, d2) in enumerate(zip(layer_config[:-1], layer_config[1:])):
            layer_dict[f"linear_{i}"] = nn.Linear(d1, d2)  # 线性变换层
            if i < self.n_layers - 1:  # 除最后一层外都添加激活函数
                layer_dict[f"activation_{i}"] = activation_fn
        # 输出激活函数（可选）
        if self.output_fn is not None:
            layer_dict["output_activation"] = self.output_fn
        # 将所有层组合成序列
        self.layers = nn.Sequential(layer_dict)
    
    def forward(self, x):
        """前向传播"""
        return self.layers(x)

class VAECF(nn.Module):
    """
    基于变分自编码器的协同过滤模型
    
    参考论文: Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """
    def __init__(self, config, user_num, item_num, device):
        super(VAECF, self).__init__()
        # 初始化模型参数
        self.num_users = user_num  # 用户数量
        self.num_items = item_num  # 物品数量
        self.latent_size = config['num_feat']  # 隐变量维度
        self.device = device  # 计算设备
        self.total_anneal_steps = config['total_anneal_steps']  # 退火步数
        self.anneal_cap = config['anneal_cap']  # 退火上限
        self.update_count = 0  # 更新计数器（用于退火）
        self.activation_fn = nn.Tanh()  # 激活函数（使用tanh）
        # 编码器（推断网络）：隐变量 → 均值和方差
        self.q_layers = PolyLinear(
            [self.num_items, 600, self.latent_size * 2],  # 输入层→隐藏层→输出层（均值和方差）
            activation_fn=self.activation_fn
        )
        # 解码器（生成网络）：隐变量 → 重建的物品概率分布
        self.p_layers = PolyLinear(
            [self.latent_size, 600, self.num_items],  # 输入层→隐藏层→输出层
            activation_fn=self.activation_fn
        )
        # Dropout层
        self.drop = nn.Dropout(config['dropout'])
        # Sigmoid激活函数（用于输出概率）
        self.logistic = nn.Sigmoid()
        # 初始化权重
        self.init_weights()
        # 用户嵌入层
        self.user_embeddings = nn.Embedding(self.num_users, self.latent_size)
    
    def init_weights(self):
        """初始化网络权重"""
        # 初始化编码器权重
        for layer in self.q_layers.modules():
            if isinstance(layer, nn.Linear):  # 线性层初始化
                nn.init.xavier_normal_(layer.weight)  # Xavier正态初始化权重
                nn.init.normal_(layer.bias, mean=0.0, std=0.001)  # 偏置小标准差初始化
        # 初始化解码器权重
        for layer in self.p_layers.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.normal_(layer.bias, mean=0.0, std=0.001)
    
    def encode(self, user_vector):
        """编码过程：用户向量 → 隐变量"""
        # 预处理：归一化 + dropout
        h = self.drop(F.normalize(user_vector.float()))
        # 通过编码器网络
        h = self.q_layers(h)
        # 分离均值和方差
        mu, logvar = h[:, :self.latent_size], h[:, self.latent_size:]
        # 重参数化技巧：从分布中采样隐变量
        std = torch.exp(0.5 * logvar)  # 标准差
        eps = torch.randn_like(std)  # 随机噪声
        z = eps.mul(std).add_(mu)  # z = μ + σ*ε
        # 计算KL散度（正则化项）
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return z, KLD
    
    def decode(self, z):
        """解码过程：隐变量 → 重建的物品概率分布"""
        return self.p_layers(z)
    
    def forward(self, user_indices, item_indices):
        """前向传播计算预测分数"""
        # 获取用户嵌入向量
        user_vectors = self.user_embeddings(user_indices)
        # 编码得到隐变量和KL散度
        z, KLD = self.encode(user_vectors)
        # 解码得到所有物品的分数
        scores = self.decode(z)
        # 提取指定物品的分数
        pred_scores = scores.gather(1, item_indices.unsqueeze(1)).squeeze(1)
        # 转换为概率
        pred_scores = self.logistic(pred_scores)
        return pred_scores
    
    def loss_function(self, recon_x, x, KLD, anneal=1.0):
        """计算损失函数（重建损失 + KL散度正则项）"""
        # 重建损失（负对数似然）
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        # 总损失 = 重建损失 + 退火系数 * KL散度
        return BCE + anneal * KLD
    
    def fit_epoch(self, device, train_loader, optimizer):
        """训练一个epoch"""
        self.train()  # 设置为训练模式
        total_loss, batch_count = 0, 0  # 记录总损失和批次数量
        for user_indices, item_indices, _ in train_loader:
            # 数据转移到设备
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            # 计算退火系数（随时间增加）
            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap
            # 梯度清零
            self.zero_grad()
            # 前向传播（获取重建分数和KL散度）
            recon_batch, KLD = self(user_indices, item_indices)
            # 计算损失
            loss = self.loss_function(recon_batch, user_indices, KLD, anneal)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 更新统计信息
            total_loss += loss.item()
            batch_count += 1
            self.update_count += 1  # 更新计数器
        return total_loss, batch_count
    
    def inactive_embeddings_for_my_loss(self, samples: torch.LongTensor):
        """获取目标用户的嵌入（用于自定义损失）"""
        device = self.user_embeddings.weight.device
        samples = samples.to(device)
        return self.user_embeddings(samples[:, 0])
    
    @torch.no_grad()
    def neighbor_embeddings_for_my_loss(self, samples: torch.LongTensor):
        """获取邻居用户的嵌入（用于自定义损失，不计算梯度）"""
        device = self.user_embeddings.weight.device
        samples = samples.to(device)
        return self.user_embeddings(samples[:, 1])