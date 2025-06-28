import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NGCF(nn.Module):
    def __init__(self, config, user_num, item_num, device):
        super(NGCF, self).__init__()
        # 初始化模型参数
        self.user_num = user_num  # 用户数量
        self.item_num = item_num  # 物品数量
        self.device = device  # 计算设备（CPU/GPU）
        # 从配置中获取模型参数
        self.emb_dim = config['embed_size']  # 嵌入维度
        self.layer_size = config['layer_size']  # 各层大小
        self.n_layers = len(self.layer_size)  # 层数
        # 图相关配置
        self.adj_type = config['adj_type']  # 邻接矩阵类型
        self.alg_type = config['alg_type']  # 算法类型
        self.n_fold = config['n_fold']  # 邻接矩阵分块数量
        # Dropout配置
        self.node_dropout_flag = config['node_dropout_flag']  # 是否使用节点dropout
        self.node_dropout_rate = config['node_dropout_rate']  # 节点dropout率
        self.decay = config['decay']  # L2正则化系数
        # 创建归一化邻接矩阵
        self.norm_adj = self._create_norm_adj().to(device)
        # 初始化用户和物品嵌入
        self.user_embedding = nn.Embedding(self.user_num, self.emb_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.emb_dim)
        # Xavier均匀初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        # 初始化模型权重
        self.weights = self._init_weights()
        # 输出层的sigmoid激活函数
        self.logistic = nn.Sigmoid()

    def _init_weights(self):
        """初始化模型各层的权重和偏置"""
        weights = nn.ParameterDict()  # 使用字典存储各层参数
        # 权重尺寸列表：[初始嵌入尺寸, 第1层尺寸, 第2层尺寸, ...]
        self.weight_size_list = [self.emb_dim] + self.layer_size
        # 为每一层初始化权重
        for k in range(self.n_layers):
            # 图卷积部分权重
            weights[f'W_gc_{k}'] = nn.Parameter(torch.empty(
                self.weight_size_list[k], self.weight_size_list[k + 1]
            ))
            # 图卷积部分偏置
            weights[f'b_gc_{k}'] = nn.Parameter(torch.zeros(1, self.weight_size_list[k + 1]))
            # 双交互部分权重
            weights[f'W_bi_{k}'] = nn.Parameter(torch.empty(
                self.weight_size_list[k], self.weight_size_list[k + 1]
            ))
            # 双交互部分偏置
            weights[f'b_bi_{k}'] = nn.Parameter(torch.zeros(1, self.weight_size_list[k + 1]))
            # Xavier均匀初始化权重
            nn.init.xavier_uniform_(weights[f'W_gc_{k}'])
            nn.init.xavier_uniform_(weights[f'W_bi_{k}'])
        return weights

    def _create_norm_adj(self):
        """创建归一化的邻接矩阵（示例中使用随机数据）"""
        # 生成随机的用户-物品交互矩阵（实际应用中应使用真实数据）
        user_item_data = np.random.choice([0, 1], size=(self.user_num, self.item_num), p=[0.9, 0.1])
        user_item_matrix = torch.FloatTensor(user_item_data)
        # 构建完整的邻接矩阵（用户+物品）
        adj_mat = torch.zeros(self.user_num + self.item_num, self.user_num + self.item_num)
        # 填充用户-物品交互区域
        adj_mat[:self.user_num, self.user_num:] = user_item_matrix
        # 填充物品-用户交互区域（转置）
        adj_mat[self.user_num:, :self.user_num] = user_item_matrix.T
        # 计算度矩阵的平方根倒数（用于对称归一化）
        rowsum = adj_mat.sum(dim=1)  # 每行求和（节点度）
        d_inv_sqrt = torch.pow(rowsum, -0.5)  # 度矩阵的-1/2次方
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # 处理无穷大值
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)  # 转换为对角矩阵
        # 对称归一化：D^{-1/2}AD^{-1/2}
        norm_adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_mat), d_mat_inv_sqrt)
        return norm_adj

    def _split_A_hat(self, X):
        """将邻接矩阵分割成多个块（用于分块计算）"""
        A_fold_hat = []  # 存储分割后的矩阵块
        # 计算每块的长度
        fold_len = (self.user_num + self.item_num) // self.n_fold
        # 分割矩阵
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len  # 当前块的起始索引
            if i_fold == self.n_fold - 1:  # 最后一块
                end = self.user_num + self.item_num  # 结束索引
            else:
                end = (i_fold + 1) * fold_len  # 结束索引
            # 添加当前块
            A_fold_hat.append(X[start:end, :])
        return A_fold_hat

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """对稀疏矩阵应用dropout"""
        if keep_prob == 1.0:  # 不应用dropout
            return X
        dropout_rate = 1 - keep_prob  # dropout率
        # 应用dropout（对非零元素）
        dropout_X = F.dropout(X, p=dropout_rate, training=self.training)
        return dropout_X

    def _split_A_hat_node_dropout(self, X):
        """分割邻接矩阵并应用节点dropout"""
        A_fold_hat = []  # 存储分割后的矩阵块
        fold_len = (self.user_num + self.item_num) // self.n_fold  # 每块长度
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len  # 起始索引
            if i_fold == self.n_fold - 1:  # 最后一块
                end = self.user_num + self.item_num  # 结束索引
            else:
                end = (i_fold + 1) * fold_len  # 结束索引
            temp = X[start:end]  # 获取当前块
            n_nonzero_temp = temp.nonzero().size(0)  # 计算非零元素数量
            # 应用dropout
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout_rate, n_nonzero_temp))
        return A_fold_hat

    def _create_ngcf_embed(self):
        """生成NGCF的嵌入表示"""
        # 根据是否使用节点dropout选择分割方式
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        # 初始嵌入（用户和物品拼接）
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]  # 存储所有层的嵌入
        # 逐层传播
        for k in range(self.n_layers):
            temp_embed = []  # 临时存储各块的嵌入
            # 分块计算消息传递
            for f in range(self.n_fold):
                # 邻接矩阵块乘以当前嵌入
                temp_embed.append(torch.matmul(A_fold_hat[f], ego_embeddings))
            # 合并所有块的嵌入
            side_embeddings = torch.cat(temp_embed, 0)
            # 图卷积部分
            sum_embeddings = F.leaky_relu(
                torch.matmul(side_embeddings, self.weights[f'W_gc_{k}']) + self.weights[f'b_gc_{k}']
            )
            # 双交互部分
            bi_embeddings = ego_embeddings * side_embeddings  # 逐元素相乘
            bi_embeddings = F.leaky_relu(
                torch.matmul(bi_embeddings, self.weights[f'W_bi_{k}']) + self.weights[f'b_bi_{k}']
            )
            # 更新嵌入：图卷积 + 双交互
            ego_embeddings = sum_embeddings + bi_embeddings
            # 应用dropout
            ego_embeddings = F.dropout(ego_embeddings, p=self.node_dropout_rate, training=self.training)
            # L2归一化
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            # 添加到所有嵌入列表
            all_embeddings.append(norm_embeddings)
        # 拼接所有层的嵌入
        all_embeddings = torch.cat(all_embeddings, 1)
        # 分割为用户嵌入和物品嵌入
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num], 0)
        return u_g_embeddings, i_g_embeddings

    def forward(self, user_indices, item_indices):
        """前向传播计算预测分数"""
        # 获取用户和物品嵌入
        u_g_embeddings, i_g_embeddings = self._create_ngcf_embed()
        # 获取指定用户和物品的嵌入
        user_emb = u_g_embeddings[user_indices]
        item_emb = i_g_embeddings[item_indices]
        # 计算点积得分
        scores = torch.sum(user_emb * item_emb, dim=1)
        # 通过sigmoid转换为概率
        pred_scores = self.logistic(scores)
        return pred_scores

    def bpr_loss(self, users, pos_items, neg_items):
        """计算BPR损失（贝叶斯个性化排序损失）"""
        # 正样本预测分数
        pos_scores = self.forward(users, pos_items)
        # 负样本预测分数
        neg_scores = self.forward(users, neg_items)
        # 计算BPR损失（最大化正负样本得分差）
        mf_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        # L2正则化项
        regularizer = (self.user_embedding(users).norm(2).pow(2)
                       + self.item_embedding(pos_items).norm(2).pow(2)
                       + self.item_embedding(neg_items).norm(2).pow(2)) / 2
        # 正则化损失
        emb_loss = self.decay * regularizer
        reg_loss = 0.0  # 其他正则化项（此处未使用）
        # 总损失 = BPR损失 + 正则化损失
        return mf_loss + emb_loss + reg_loss

    def inactive_embeddings_for_my_loss(self, samples: torch.LongTensor):
        """获取目标用户的嵌入（用于自定义损失）"""
        u_g_embeddings, _ = self._create_ngcf_embed()
        device = u_g_embeddings.device
        samples = samples.to(device)
        # 返回目标用户的嵌入
        return u_g_embeddings[samples[:, 0]]

    @torch.no_grad()
    def neighbor_embeddings_for_my_loss(self, samples: torch.LongTensor):
        """获取邻居用户的嵌入（用于自定义损失，不计算梯度）"""
        u_g_embeddings, _ = self._create_ngcf_embed()
        device = u_g_embeddings.device
        samples = samples.to(device)
        # 返回邻居用户的嵌入
        return u_g_embeddings[samples[:, 1]]