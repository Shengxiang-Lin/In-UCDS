# NeuMF (神经矩阵分解) 模型配置
neumf_config = {
    'num_epoch': 100,           # 训练总轮次
    'batch_size': 1024,          # 批次大小
    'optimizer': 'adam',         # 优化器类型
    'adam_lr': 1e-4,             # Adam优化器的学习率
    'latent_dim_mf': 32,         # 矩阵分解部分的隐向量维度
    'latent_dim_mlp': 32,        # 多层感知机部分的隐向量维度
    'num_negative': 4,           # 每个正样本对应的负样本数量
    'layers': [64,32,16,8],      # MLP层的神经元数量（第0层是用户和物品隐向量的拼接）
    'l2_regularization': 1e-5,   # L2正则化系数
    'device_id': 1               # 使用的GPU设备ID
}
# PMF (概率矩阵分解) 模型配置
pmf_config = {
    'num_epoch': 100,           # 训练总轮次
    'batch_size': 1024,          # 批次大小
    'optimizer': 'adam',         # 优化器类型
    'num_feat': 16,              # 特征维度
    'epsilon': 1,                # 噪声参数（高斯分布的标准差）
    '_lambda': 0.1,              # 正则化系数
    'momentum': 0.8,             # 动量参数（如果使用SGD优化器）
    'maxepoch': 20,              # 最大训练轮次（可能用于早期停止）
    'num_batches': 10,           # 每轮训练的批次数量
    'num_negative': 4,           # 每个正样本对应的负样本数量
    'l2_regularization': 1e-5,   # L2正则化系数
    'adam_lr': 1e-4              # Adam优化器的学习率
}
# VAECF (变分自编码协同过滤) 模型配置
vaecf_config = {
    'num_epoch': 100,            # 训练总轮次
    'batch_size': 1024,           # 批次大小
    'optimizer': 'adam',          # 优化器类型
    'num_feat': 32,               # 特征维度
    'total_anneal_steps': 10000,  # 退火总步数（KL散度权重调整）
    'anneal_cap': 0.2,            # 退火上限（KL散度最大权重）
    'dropout': 0.5,               # Dropout比率
    'num_negative': 4,            # 每个正样本对应的负样本数量
    'l2_regularization': 1e-5,    # L2正则化系数
    'adam_lr': 1e-4               # Adam优化器的学习率
}
# NGCF (神经图协同过滤) 模型配置
ngcf_config = {
    'num_epoch': 100,             # 训练总轮次
    'batch_size': 1024,            # 批次大小
    'optimizer': 'adam',           # 优化器类型
    'embed_size': 16,              # 嵌入向量大小
    'layer_size': [32],            # 图卷积层大小
    'adj_type': 'norm',            # 邻接矩阵归一化类型
    'alg_type': 'ngcf',            # 算法类型
    'n_fold': 10,                  # 邻接矩阵折叠数（用于内存优化）
    'node_dropout_flag': True,     # 是否启用节点dropout
    'node_dropout_rate': 0.5,      # 节点dropout比率
    'decay': 1e-5,                 # 权重衰减系数
    'num_negative': 4,             # 每个正样本对应的负样本数量
    'l2_regularization': 1e-5,     # L2正则化系数
    'adam_lr': 1e-4                # Adam优化器的学习率
}