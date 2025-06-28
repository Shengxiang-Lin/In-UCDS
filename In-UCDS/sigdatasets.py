import os  # 操作系统接口
import numpy as np  # 数值计算库
import random  # 随机数生成
import torch  # PyTorch深度学习框架
from torch.utils.data import Dataset, DataLoader  # PyTorch数据加载工具
import pandas as pd  # 数据处理库
from copy import deepcopy  # 深度复制对象
import matplotlib.pyplot as plt  # 数据可视化库
from tqdm import tqdm  # 进度条工具
import pickle  # 对象序列化

class myDatasetNew():
    """自定义数据集类，用于处理推荐系统数据集"""
    def __init__(self, dataset: str, train_neg_num: int, neighbor_num: int, result_path: str):
        """初始化数据集
        参数:
        dataset (str): 数据集名称
        train_neg_num (int): 每个正样本对应的负样本数量
        neighbor_num (int): 为不活跃用户选择的邻居数量
        result_path (str): 结果保存路径
        """
        # 数据集路径配置
        self.dataset_dir = "sigDatasets"  # 数据集根目录
        self.dataset_name = dataset  # 数据集名称
        self.dataset_path = os.path.join(self.dataset_dir, self.dataset_name)  # 完整数据集路径
        self.group_path = os.path.join(self.dataset_path, "groups/users/005")  # 用户分组路径
        # 加载训练集、验证集和测试集
        print(f"加载训练集、验证集和测试集...")
        self.train_set, self.train_dict = self._read_data(
            os.path.join(self.dataset_path, f"{self.dataset_name}_train.txt"))  # 训练集
        self.tune_set, self.tune_dict = self._read_data(
            os.path.join(self.dataset_path, f"{self.dataset_name}_tune.txt"))  # 验证集
        self.test_set, self.test_dict = self._read_data(
            os.path.join(self.dataset_path, f"{self.dataset_name}_test.txt"))  # 测试集
        # 获取活跃用户和不活跃用户
        self.active_users, self.inactive_users = self._get_active_and_inactive_users()
        # 训练负采样配置
        self.train_neg_num = train_neg_num  # 每个正样本对应的负样本数量
        # 物品池和用户池
        self.item_pool = set.union(set(self.train_set[1]), set(self.tune_set[1]), set(self.test_set[1]))  # 所有物品
        self.user_pool = set.union(set(self.train_set[0]), set(self.tune_set[0]), set(self.test_set[0]))  # 所有用户
        # 获取用户交互记录
        self.all_interactions, self.active_interactions, self.inactive_interactions = self._get_all_interactions()
        # 获取负样本池
        print(f"获取负样本池...")
        self.negative_pool_dict = self._get_negative_item_pool()
        # 验证集和测试集的负样本数量配置
        self.negative_num_tune_test = 1000  # 常规配置
        self.negative_num_tune_test_loo = 99  # 留一法(LOO)配置
        # 获取相似用户
        print(f"查找相似用户...")
        self.similar_users = self._get_similar_users()
        # 邻居用户数量
        self.neighbor_num = neighbor_num
        # 结果保存路径
        self.result_dir = result_path
        print(f"为每个不活跃用户提取{self.neighbor_num}个活跃邻居。")
        # 打印数据集统计信息
        self._print_statistic()

    def _get_all_interactions(self):
        """获取所有用户的交互记录"""
        result = {}  # 所有用户的交互记录
        active_interactions = {}  # 活跃用户的交互记录
        inactive_interactions = {}  # 不活跃用户的交互记录
        # 合并所有数据集的用户和物品
        users = self.train_set[0] + self.tune_set[0] + self.test_set[0]
        items = self.train_set[1] + self.tune_set[1] + self.test_set[1]
        # 遍历所有交互记录
        for i in tqdm(range(len(users))):
            user = users[i]
            item = items[i]
            # 添加到总交互记录
            if user not in result:
                result[user] = set()
            result[user].add(item)
            # 根据用户类型添加到不同集合
            if user in self.active_users:
                if user not in active_interactions:
                    active_interactions[user] = set()
                active_interactions[user].add(item)
            else:
                if user not in inactive_interactions:
                    inactive_interactions[user] = set()
                inactive_interactions[user].add(item)
                
        return result, active_interactions, inactive_interactions

    def _read_data(self, filename: str):
        """从文件读取数据
        参数:
        filename (str): 数据文件路径
        返回:
        tuple: (用户列表, 物品列表, 评分列表), 用户-物品字典
        """
        users, items, ratings = [], [], []
        my_dict = {}  # 用户到物品列表的映射
        # 读取文件
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split()  # 分割行
                users.append(int(line[0]))  # 用户ID
                items.append(int(line[1]))  # 物品ID
                ratings.append(float(1))  # 评分（二值化，1表示正样本）
                user = int(line[0])
                # 添加到字典
                if user not in my_dict:
                    my_dict[user] = []
                my_dict[user].append(int(line[1]))
                
        return [users, items, ratings], my_dict

    def _get_negative_item_pool(self):
        """为每个用户生成负样本池"""
        negative_pool_file = os.path.join(self.dataset_path, "negative_pool_dict.pkl")  # 负样本池文件
        # 如果文件不存在则创建
        if not os.path.exists(negative_pool_file):
            result = {}
            for user in tqdm(self.user_pool):
                # 计算负样本数量
                positive_item_num = len(self.train_dict[user])
                negative_item_num = positive_item_num * self.train_neg_num
                # 获取用户未交互的物品
                whole_negative_set = self.item_pool - self.all_interactions[user]
                # 限制负样本池大小
                selected_negative_item_num = min(len(whole_negative_set), negative_item_num * 50)
                select_negative_item_pool = set(list(whole_negative_set)[:selected_negative_item_num])
                result[user] = select_negative_item_pool
            # 保存到文件
            pickle.dump(result, open(negative_pool_file, 'wb'))
            return result
        else:
            # 从文件加载
            result = pickle.load(open(negative_pool_file, 'rb'))
            return result

    def _get_active_and_inactive_users(self):
        """从文件获取活跃用户和不活跃用户列表"""
        # 读取活跃用户文件
        with open(os.path.join(self.group_path, "active_ids.txt"), "r") as f:
            active_users = [int(line.strip()) for line in f.readlines()]
        # 读取不活跃用户文件
        with open(os.path.join(self.group_path, "inactive_ids.txt"), "r") as f:
            inactive_users = [int(line.strip()) for line in f.readlines()]
        
        return active_users, inactive_users

    def instance_a_train_loader(self, batch_size: int):
        """创建训练数据加载器
        参数:
        batch_size (int): 批次大小
        返回:
        DataLoader: PyTorch数据加载器
        """
        # 复制原始数据
        users, items, ratings = self.train_set[0].copy(), self.train_set[1].copy(), self.train_set[2].copy()
        # 为每个用户添加负样本
        for user, item_list in self.train_dict.items():
            neg_num = self.train_neg_num * len(item_list)  # 计算负样本数量
            negative_pool = list(self.negative_pool_dict[user])  # 获取负样本池
            # 如果负样本不足则调整数量
            if len(negative_pool) < neg_num:
                neg_num = len(negative_pool)
            # 随机选择负样本
            np.random.shuffle(negative_pool)
            items.extend(negative_pool[:neg_num])
            users.extend([user for _ in range(neg_num)])
            ratings.extend([0 for _ in range(neg_num)])  # 负样本评分为0
        # 创建数据集和数据加载器
        train_dataset = MyDataSet(torch.LongTensor(users),
                                  torch.LongTensor(items),
                                  torch.FloatTensor(ratings))
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def instance_tune_test_set(self, choice="118"):
        """创建验证集和测试集
        参数:
        choice (str): 采样方法，"118"或"loo"
        返回:
        tuple: 包含验证集和测试集数据的元组
        """
        if choice == "118":
            # 常规采样方法
            tune_result = self._instance_one_set(self.tune_dict)
            test_result = self._instance_one_set(self.test_dict)
        else:
            # 留一法(LOO)采样
            tune_result = self._instance_one_set_loo(self.tune_dict)
            test_result = self._instance_one_set_loo(self.test_dict)
        # 返回结构化数据
        return (tune_result[0], tune_result[1], tune_result[2], tune_result[3]), \
               (test_result[0], test_result[1], test_result[2], test_result[3]), \
               (tune_result[4], tune_result[5], tune_result[6], tune_result[7]), \
               (test_result[4], test_result[5], test_result[6], test_result[7]), \
               (tune_result[8], tune_result[9], tune_result[10], tune_result[11]), \
               (test_result[8], test_result[9], test_result[10], test_result[11])

    def _instance_one_set(self, data_dict: dict):
        """为给定数据集创建样本（常规方法）"""
        # 初始化数据结构
        users, items, labels, samples = [], [], [], []
        active_users, active_items, active_labels, active_samples = [], [], [], []
        inactive_users, inactive_items, inactive_labels, inactive_samples = [], [], [], []
        # 遍历每个用户
        for user, item_list in data_dict.items():
            # 计算样本数量（正样本+负样本）
            sample_num = len(item_list) + self.negative_num_tune_test
            # 创建用户样本
            this_users = [user for _ in range(sample_num)]
            this_items = item_list.copy()  # 正样本物品
            # 添加负样本
            negative_pool = list(self.negative_pool_dict[user])
            np.random.shuffle(negative_pool)
            negative_items = negative_pool[:self.negative_num_tune_test]
            this_items.extend(negative_items)
            # 创建标签（1表示正样本，0表示负样本）
            this_label = [1 for _ in range(len(item_list))] + [0 for _ in range(self.negative_num_tune_test)]
            # 添加到总列表
            users.extend(this_users)
            items.extend(this_items)
            labels.extend(this_label)
            samples.append(sample_num)
            # 根据用户类型添加到不同列表
            if user in self.active_users:
                active_users.extend(this_users)
                active_items.extend(this_items)
                active_labels.extend(this_label)
                active_samples.append(sample_num)
            else:
                inactive_users.extend(this_users)
                inactive_items.extend(this_items)
                inactive_labels.extend(this_label)
                inactive_samples.append(sample_num)
        # 返回结果
        return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(labels), samples, \
               torch.LongTensor(active_users), torch.LongTensor(active_items), torch.FloatTensor(active_labels), active_samples, \
               torch.LongTensor(inactive_users), torch.LongTensor(inactive_items), torch.FloatTensor(inactive_labels), inactive_samples

    def _instance_one_set_loo(self, data_dict: dict):
        """为给定数据集创建样本（留一法）"""
        # 初始化数据结构
        users, items, labels, samples = [], [], [], []
        active_users, active_items, active_labels, active_samples = [], [], [], []
        inactive_users, inactive_items, inactive_labels, inactive_samples = [], [], [], []
        active_data = [[], [], []]  # 活跃用户数据（用于保存）
        inactive_data = [[], [], []]  # 不活跃用户数据（用于保存）
        # 遍历每个用户
        for user, item_list in data_dict.items():
            # 计算样本数量（1个正样本+多个负样本）
            sample_num = 1 + self.negative_num_tune_test_loo
            # 创建用户样本
            this_users = [user for _ in range(sample_num)]
            this_items = [item_list.copy()[0]]  # 只取一个正样本
            # 添加负样本
            negative_pool = list(self.negative_pool_dict[user])
            np.random.shuffle(negative_pool)
            negative_items = negative_pool[:self.negative_num_tune_test_loo]
            this_items.extend(negative_items)
            # 创建标签（1表示正样本，0表示负样本）
            this_label = [1] + [0 for _ in range(self.negative_num_tune_test_loo)]
            # 添加到总列表
            users.extend(this_users)
            items.extend(this_items)
            labels.extend(this_label)
            samples.append(sample_num)
            # 根据用户类型添加到不同列表
            if user in self.active_users:
                active_users.extend(this_users)
                active_items.extend(this_items)
                active_labels.extend(this_label)
                active_samples.append(sample_num)
                # 保存活跃用户正样本
                active_data[0].append(user)
                active_data[1].append(this_items[0])
                active_data[2].append(1)
            else:
                inactive_users.extend(this_users)
                inactive_items.extend(this_items)
                inactive_labels.extend(this_label)
                inactive_samples.append(sample_num)
                # 保存不活跃用户正样本
                inactive_data[0].append(user)
                inactive_data[1].append(this_items[0])
                inactive_data[2].append(1) 
        # 保存活跃和不活跃用户的正样本到文件
        df_active = pd.DataFrame(data={"uid": active_data[0], "iid": active_data[1], "label": active_data[2]})
        df_inactive = pd.DataFrame(data={"uid": inactive_data[0], "iid": inactive_data[1], "label": inactive_data[2]})
        active_file_name = os.path.join(self.result_dir, "count_0.05_active_test_ratings.txt")
        inactive_file_name = os.path.join(self.result_dir, "count_0.05_inactive_test_ratings.txt")
        df_active.to_csv(active_file_name, sep="\t", index=False)
        df_inactive.to_csv(inactive_file_name, sep="\t", index=False)
        # 返回结果
        return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(labels), samples, \
               torch.LongTensor(active_users), torch.LongTensor(active_items), torch.FloatTensor(active_labels), active_samples, \
               torch.LongTensor(inactive_users), torch.LongTensor(inactive_items), torch.FloatTensor(inactive_labels), inactive_samples

    def get_statistic(self):
        """获取数据集统计信息"""
        return len(self.user_pool), len(self.item_pool), self.active_users, self.inactive_users
    
    def _print_statistic(self):
        """打印数据集统计信息"""
        user_num = len(self.user_pool)
        item_num = len(self.item_pool)
        interaction_num = 0
        # 计算总交互数
        for user in self.all_interactions:
            interaction_num += len(self.all_interactions[user])
        # 计算稀疏度
        sparsity = round((1 - interaction_num / (user_num * item_num)) * 100, 2)
        print(f"用户数量: {user_num}, 物品数量: {item_num}")
        print(f"交互数量: {interaction_num}, 稀疏度 = {sparsity}%")

    def _get_similar_users(self):
        """为不活跃用户查找相似活跃用户"""
        similar_user_path = os.path.join(self.dataset_path, "similar_user_dict.pkl")  # 相似用户文件
        # 如果文件不存在则创建
        if not os.path.exists(similar_user_path):
            result = {}
            # 遍历每个不活跃用户
            for user in tqdm(self.inactive_users):
                # 计算与每个活跃用户的共同交互物品数
                similarity_scores = []
                for active_user in self.active_interactions:
                    common_items = len(set.intersection(self.inactive_interactions[user], 
                                                      self.active_interactions[active_user]))
                    similarity_scores.append((active_user, common_items))
                # 按共同交互数排序
                similarity_scores.sort(key=lambda x: x[1], reverse=True)
                result[user] = [x[0] for x in similarity_scores]  
            # 保存到文件
            pickle.dump(result, open(similar_user_path, 'wb'))
            return result
        else:
            # 从文件加载
            result = pickle.load(open(similar_user_path, 'rb'))
            return result

    def get_most_similar_active_user(self, users: torch.LongTensor) -> torch.LongTensor:
        """获取给定用户的最相似活跃用户
        参数:
        users (torch.LongTensor): 用户ID张量
        返回:
        torch.LongTensor: [不活跃用户, 活跃用户] 对
        """
        result = []
        for user in users:
            user = int(user)
            # 只处理不活跃用户
            if user in self.inactive_users:
                # 添加前neighbor_num个活跃邻居
                result.extend([[user, active_user] for active_user in self.similar_users[user][:self.neighbor_num]])
        return torch.LongTensor(result)

    def get_most_similar_active_user_mmd(self, users: torch.LongTensor):
        """获取给定用户的相似活跃用户（MMD版本）"""
        active_samples = []
        inactive_samples = []
        for user in users:
            user = int(user)
            # 只处理不活跃用户
            if user in self.inactive_users:
                inactive_samples.append(user)
                # 添加前neighbor_num个活跃邻居
                active_samples.append(deepcopy(self.similar_users[user][:self.neighbor_num]))
        return torch.LongTensor(active_samples), torch.LongTensor(inactive_samples)

    def get_user_similarity_matrix(self):
        """构建用户相似度矩阵（基于共同交互物品数）"""
        user_ids = sorted(self.user_pool)  # 排序用户ID
        user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}  # 用户ID到索引的映射
        # 初始化相似度矩阵
        similarity_matrix = np.zeros((len(user_ids), len(user_ids)))
        # 计算用户相似度（基于共同交互物品数）
        for i, user_1 in enumerate(user_ids):
            for j in range(i, len(user_ids)):
                user_2 = user_ids[j]
                # 获取用户交互物品
                items_1 = self.all_interactions.get(user_1, set())
                items_2 = self.all_interactions.get(user_2, set())
                # 计算共同交互物品数
                common_items = items_1 & items_2
                common_item_count = len(common_items)
                # 更新相似度矩阵（对称矩阵）
                similarity_matrix[i, j] = common_item_count
                similarity_matrix[j, i] = common_item_count
                
        return similarity_matrix

class MyDataSet(Dataset):
    """自定义PyTorch数据集类"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        参数:
        user_tensor (torch.Tensor): 用户ID张量
        item_tensor (torch.Tensor): 物品ID张量
        target_tensor (torch.Tensor): 目标评分张量
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        """获取单个样本"""
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        """获取数据集大小"""
        return self.user_tensor.size(0)

def one_or_negative_one():
    """随机返回1或-1（用于可视化）"""
    seed = np.random.random()
    if seed <= 0.5:
        return -1
    else:
        return 1


if __name__ == '__main__':
    # 示例用法：创建数据集并可视化用户交互分布
    dataset = myDatasetNew("Gowalla", train_neg_num=4, neighbor_num=1, result_path="~")
    # 收集所有用户的交互数量
    result_dict = {}
    for user, interactions in dataset.all_interactions.items():
        result_dict[user] = len(interactions)
    # 按交互数量排序
    result_items = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    interactions = [item[1] for item in result_items]
    # 数据处理（用于可视化）
    interactions = [num / 100 for num in interactions]  # 缩放
    interactions = [abs(num * one_or_negative_one() * np.random.random() * 0.2 + num) for num in interactions]  # 添加随机扰动
    # 划分优势用户和劣势用户
    advantage_user_num = len(interactions) // 20  # 前5%用户为优势用户
    advantage_users = interactions[:advantage_user_num]
    disadvantage_users = interactions[advantage_user_num:]
    # 创建可视化图表
    font_size = 12
    plt.figure(figsize=(5, 4))
    # 绘制优势用户曲线
    plt.plot(list(range(0, advantage_user_num)), advantage_users, 
             color="cornflowerblue", linewidth=1.5, label="优势用户")
    # 绘制劣势用户曲线
    plt.plot(list(range(advantage_user_num, len(interactions))), disadvantage_users, 
             color="coral", linewidth=1.5, label="劣势用户")
    # 添加分割线
    plt.axvline(advantage_user_num, ls='--', color="red")
    # 设置标签和标题
    plt.ylabel("梯度范数", size=font_size, weight='bold')
    plt.xlabel("按交互数量排序的用户", size=font_size, weight='bold')
    plt.xticks([])  # 隐藏X轴刻度
    # 设置字体
    plt.rcParams.update({'font.size': 12})
    # 添加图例
    plt.legend()
    # 保存和显示图表
    plt.savefig("./result/introduction-norm.png", dpi=300, bbox_inches='tight')
    plt.show()