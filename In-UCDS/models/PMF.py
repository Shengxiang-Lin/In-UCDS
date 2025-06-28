import torch
import torch.nn as nn

class PMF(nn.Module):
    def __init__(self, config, user_num, item_num, device):
        super(PMF, self).__init__()
        # 初始化模型参数
        self.num_users = user_num  # 用户数量
        self.num_items = item_num  # 物品数量
        self.num_feat = config['num_feat']  # 隐特征维度
        self.device = device  # 计算设备（CPU/GPU）
        # 初始化用户隐向量矩阵（使用正态分布随机初始化）
        self.w_User = nn.Parameter(0.1 * torch.randn(user_num, self.num_feat, device=device))
        # 初始化物品隐向量矩阵（使用正态分布随机初始化）
        self.w_Item = nn.Parameter(0.1 * torch.randn(item_num, self.num_feat, device=device))
        # 使用sigmoid将预测分数转换为概率
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        """前向传播计算预测分数"""
        # 获取指定用户的隐向量
        user_vec = self.w_User[user_indices]
        # 获取指定物品的隐向量
        item_vec = self.w_Item[item_indices]
        # 计算用户-物品点积作为预测分数
        pred_scores = torch.sum(user_vec * item_vec, dim=1)
        # 使用sigmoid转换为概率值
        pred_scores = self.logistic(pred_scores)
        return pred_scores

    def inactive_embeddings_for_my_loss(self, samples: torch.LongTensor):
        """获取目标用户的嵌入（用于自定义损失）"""
        device = self.w_User.device  # 获取参数所在设备
        samples = samples.to(device)  # 确保数据在相同设备
        # 返回目标用户的隐向量
        return self.w_User[samples[:, 0]]

    @torch.no_grad()
    def neighbor_embeddings_for_my_loss(self, samples: torch.LongTensor):
        """获取邻居用户的嵌入（用于自定义损失，不计算梯度）"""
        device = self.w_User.device
        samples = samples.to(device)
        # 返回邻居用户的隐向量
        return self.w_User[samples[:, 1]]