import torch
import torch.nn as nn

class InUCDS(nn.Module):
    """用户聚类主导集(InUCDS)模型，用于为不活跃用户识别相似活跃用户"""
    def __init__(self, num_users, active_ids, inactive_ids, similarity_matrix, alpha_coef=1.1, tol=1e-6, max_iter=5):
        """
        初始化InUCDS模型
        参数:
        num_users (int): 用户总数
        active_ids (list): 活跃用户ID列表
        inactive_ids (list): 不活跃用户ID列表
        similarity_matrix (array): 用户相似度矩阵
        alpha_coef (float): 控制主导集规模的系数
        tol (float): 收敛容忍度
        max_iter (int): 最大迭代次数
        """
        super(InUCDS, self).__init__()  # 调用父类构造函数
        self.num_users = num_users  # 用户总数
        self.alpha_coef = alpha_coef  # 控制主导集规模的系数
        self.tol = tol  # 收敛容忍度
        self.max_iter = max_iter  # 最大迭代次数
        self.active_ids = active_ids  # 活跃用户ID列表
        self.inactive_ids = inactive_ids  # 不活跃用户ID列表
        # 将相似度矩阵转换为PyTorch张量
        self.similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)

    def compute_affinity_matrix(self, user_ids, target_idx):
        """
        计算亲和力矩阵
        参数:
        user_ids (list): 当前处理的用户ID列表
        target_idx (int): 目标用户(不活跃用户)在列表中的索引
        返回:
        torch.Tensor: 亲和力矩阵
        """
        # 将用户ID列表转换为张量
        user_ids_tensor = torch.tensor(user_ids, dtype=torch.long)
        # 获取子相似度矩阵
        similarity_submatrix = self.similarity_matrix[user_ids_tensor][:, user_ids_tensor]
        similarity_submatrix.fill_diagonal_(0)  # 将对角线置零
        similarity_submatrix = (similarity_submatrix + similarity_submatrix.T) / 2  # 确保对称
        similarity_submatrix = similarity_submatrix - similarity_submatrix.min()  # 最小值为0
        similarity_submatrix = similarity_submatrix / (similarity_submatrix.max() + 1e-6)  # 归一化到[0,1]
        # 创建掩码矩阵，用于移除目标用户
        mask = torch.ones(len(user_ids), dtype=torch.float32)
        mask[target_idx] = 0  # 目标用户位置置0
        I_S = torch.diag(mask)  # 转换为对角矩阵
        # 移除目标用户所在的行和列
        similarity_submatrix = torch.cat([similarity_submatrix[:target_idx], similarity_submatrix[target_idx + 1:]], dim=0)
        similarity_submatrix = torch.cat(
            [similarity_submatrix[:, :target_idx], similarity_submatrix[:, target_idx + 1:]], dim=1)
        # 计算最大特征值估计
        if similarity_submatrix.numel() == 0:
            lambda_max = 1e-3  # 空矩阵处理
        else:
            # 计算迹平均值作为最大特征值估计
            lambda_max = max(torch.trace(similarity_submatrix) / similarity_submatrix.shape[0], 1e-3)
        # 计算alpha参数
        alpha = self.alpha_coef * lambda_max
        # 重新获取完整的子相似度矩阵
        result = self.similarity_matrix[user_ids_tensor][:, user_ids_tensor]
        result.fill_diagonal_(0)  # 将对角线置零
        # 归一化处理
        min_result = result.min()
        max_result = result.max()
        if max_result > min_result:
            result = (result - min_result) / (max_result - min_result)
        # 应用alpha调整
        result -= alpha * I_S
        return result

    def replicator_dynamics(self, B, init_x):
        """
        复制动力学算法，用于求解主导集
        参数:
        B (torch.Tensor): 亲和力矩阵
        init_x (torch.Tensor): 初始权重向量
        返回:
        torch.Tensor: 收敛后的权重向量
        """
        x = init_x.clone()  # 克隆初始权重
        toll = self.tol  # 收敛容忍度
        max_iter = self.max_iter  # 最大迭代次数
        # 迭代求解
        for _ in range(max_iter):
            x_old = x.clone()  # 保存旧值
            x = x * (B @ x)  # 复制动力学更新规则
            x /= torch.norm(x, p=2, dim=0).detach()  # L2归一化
            # 检查收敛
            if torch.norm(x - x_old) < toll:
                break
        return x

    def extract_dominant_set(self, user_ids, target_user_id, neighbor_num):
        """
        提取主导集
        参数:
        user_ids (list): 用户ID列表
        target_user_id (int): 目标用户ID(不活跃用户)
        neighbor_num (int): 邻居数量
        返回:
        torch.Tensor: 主导集用户ID张量
        """
        # 获取目标用户在列表中的索引
        target_idx = user_ids.index(target_user_id)
        # 计算亲和力矩阵
        B = self.compute_affinity_matrix(user_ids, target_idx)
        # 初始化权重向量
        init_x = torch.ones(len(user_ids), dtype=torch.float32) * 1e-6
        init_x[target_idx] = 1.0  # 目标用户权重设为1
        # 运行复制动力学算法
        x = self.replicator_dynamics(B, init_x)
        # 按权重降序排序
        sorted_x, indices = torch.sort(x, descending=True)
        # 提取主导集
        dominant_ids = []
        for idx in indices:
            user_id = user_ids[idx.item()]
            # 跳过目标用户
            if user_id != target_user_id:
                dominant_ids.append(user_id)
            # 达到所需邻居数量时停止
            if len(dominant_ids) == neighbor_num:
                break
        # 返回主导集张量
        return torch.tensor(dominant_ids[:neighbor_num], dtype=torch.long)

    def generate_dominant_sets_for_all_inactive(self, neighbor_num):
        """
        为所有不活跃用户生成主导集
        参数:
        neighbor_num (int): 每个不活跃用户的邻居数量
        返回:
        torch.Tensor: 所有主导集对[不活跃用户, 活跃用户]的张量
        """
        samples = []
        # 遍历所有不活跃用户
        for inactive_user_id in self.inactive_ids:
            # 为当前不活跃用户提取主导集
            dominant_set = self.extract_dominant_set(
                [inactive_user_id] + self.active_ids,  # 用户列表(目标用户+所有活跃用户)
                inactive_user_id,  # 目标用户ID
                neighbor_num  # 邻居数量
            )
            # 将主导集对添加到样本列表
            for neighbor_id in dominant_set.tolist():
                samples.append([inactive_user_id, neighbor_id])
        # 转换为张量并返回
        return torch.tensor(samples, dtype=torch.long)