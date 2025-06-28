import argparse  # 命令行参数解析
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import os  # 操作系统接口
import config  # 配置文件
from sklearn.metrics import ndcg_score, f1_score, accuracy_score  # 评估指标
from sigdatasets import myDatasetNew  # 自定义数据集类
import random  # 随机数生成
import models.PMF as PMF  # 概率矩阵分解模型
import models.NeuMF as NeuMF  # 神经矩阵分解模型
import models.NGCF as NGCF  # 神经图协同过滤模型
import models.VAECF as VAECF  # 变分自编码协同过滤模型
# 全局常量定义
SPLIT = "=" * 60  # 分隔线（60个等号）
K = 10  # 评估时考虑的top-K数量
L2 = 1e-5  # L2正则化系数
original_model_time = 0  # 原始模型训练时间记录（未使用）
fair_model_time = 0  # 公平模型训练时间记录（未使用）
# 结果存储索引常量
RESULT_USER = 0  # 用户ID索引
RESULT_ITEM = 1  # 物品ID索引
RESULT_SCORE = 2  # 预测分数索引
RESULT_LABEL = 3  # 真实标签索引
# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用CUDA训练')
parser.add_argument('--seed', type=int, default=10, help='随机种子')
parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
parser.add_argument('--model', type=str, default="NeuMF", help="评估的模型")
parser.add_argument('--cuda-index', type=int, default=0, help='使用的GPU索引')
parser.add_argument('--dataset', type=str, default="MovieLens", help='使用的数据集')
parser.add_argument('--split', type=str, default="count", help='划分活跃/不活跃用户的方法')
parser.add_argument('--random-seed', type=int, default=42, help="程序随机种子")
parser.add_argument('--neighbor_num', type=int, default=3, help='为每个不活跃用户提取的邻居数量')
parser.add_argument('--log', type=str, default='logs/{}'.format(parser.parse_args().model), help='日志目录')
parser.add_argument('--model-time', type=int, default=0, help='同一模型的训练次数')

def set_seed(seed):
    """设置全局随机种子以保证可复现性"""
    random.seed(seed)  # Python随机模块
    np.random.seed(seed)  # NumPy随机模块
    torch.manual_seed(seed)  # PyTorch CPU随机种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU随机种子
    torch.backends.cudnn.deterministic = True  # 启用CuDNN确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用CuDNN自动优化

def str_format(data: list):
    """格式化输出列表数据为制表符分隔的字符串"""
    return '\t'.join([f"{item:.4f}" for item in data])

@torch.no_grad()  # 禁用梯度计算
def evaluate_one_model(model, user_set, item_set, labels, sample_num):
    """评估单个模型性能
    参数:
    model: 要评估的模型
    user_set: 用户ID张量
    item_set: 物品ID张量
    labels: 真实标签张量
    sample_num: 每个用户的样本数量列表
    返回:
    list: [BCE损失, 准确率, NDCG@K, F1分数]
    """
    predict = model(user_set, item_set)  # 模型预测
    begin = 0  # 起始索引
    predict_numpy = []  # 预测值列表
    label_numpy = []  # 真实标签列表
    predict_binary_numpy = []  # 二值化预测列表
    predict_label = [1 for _ in range(K)]  # top-K预测标签（全为1）
    # 将数据移到CPU
    temp_predict = predict.cpu()
    temp_label = labels.cpu()
    # 按样本数量分组处理
    for num in sample_num:
        # 获取当前组的预测和标签
        this_predict_label = temp_predict[begin:begin + num].view(-1)
        this_true_label = temp_label[begin:begin + num].view(-1)
        # 获取top-K预测
        values, indices = torch.topk(this_predict_label, K)
        topk_predict = this_predict_label[indices]
        topk_label = this_true_label[indices]
        # 保存结果
        predict_numpy.append(topk_predict.numpy().reshape(1, -1))
        label_numpy.append(topk_label.cpu().numpy().reshape(1, -1))
        predict_binary_numpy.append(predict_label)
        begin += num
    # 转换为NumPy数组
    label_numpy = np.array(label_numpy).squeeze()
    predict_numpy = np.array(predict_numpy).squeeze()
    predict_binary_numpy = np.array(predict_binary_numpy).squeeze()
    # 计算评估指标
    BCE_loss = round(bce_loss(predict.view(-1), labels).item(), 4)  # BCE损失
    ndcg = round(ndcg_score(y_true=label_numpy, y_score=predict_numpy, k=K), 4)  # NDCG@K
    acc = round(accuracy_score(y_true=label_numpy.reshape(1, -1).squeeze(),
                               y_pred=predict_binary_numpy.reshape(1, -1).squeeze()), 4)  # 准确率
    f1 = round(f1_score(y_true=label_numpy.reshape(1, -1).squeeze(),
                        y_pred=predict_binary_numpy.reshape(1, -1).squeeze()), 4)  # F1分数
    return [BCE_loss, acc, ndcg, f1]  # 返回指标列表

@torch.no_grad()
def test_model():
    """测试所有保存的最佳模型"""
    print(SPLIT)
    print("开始测试!!!")
    # 定义评估指标名称
    metrics_name = ["best_ndcg_all", "best_f1_all", "best_ndcg_active", "best_f1_active", 
                    "best_ndcg_inactive", "best_f1_inactive", "best_ndcg_ugf", "best_f1_ugf"]
    # 遍历所有评估指标
    for i in range(len(metrics_name)):
        metric = metrics_name[i]
        # 加载原始模型和公平模型
        model_original.load_state_dict(torch.load(os.path.join(log, f"{metric}_original.pkl")))
        model_fair.load_state_dict(torch.load(os.path.join(log, f"{metric}_fair.pkl")))
        # 评估原始模型
        active_original_result = evaluate_one_model(model_original, active_test_user_set,
                                                    active_test_item_set, active_test_labels, active_test_sample_num)
        inactive_original_result = evaluate_one_model(model_original, inactive_test_user_set,
                                                      inactive_test_item_set, inactive_test_labels,
                                                      inactive_test_sample_num)
        overall_original_result = evaluate_one_model(model_original, test_user_set,
                                                     test_item_set, test_labels, test_sample_num)
        # 评估公平模型
        overall_fair_result = evaluate_one_model(model_fair, test_user_set,
                                                 test_item_set, test_labels, test_sample_num)
        active_fair_result = evaluate_one_model(model_fair, active_test_user_set,
                                                active_test_item_set, active_test_labels, active_test_sample_num)
        inactive_fair_result = evaluate_one_model(model_fair, inactive_test_user_set,
                                                  inactive_test_item_set, inactive_test_labels,
                                                  inactive_test_sample_num)
        # 计算用户组间公平性差距 (UGF)
        ugf_original = [abs(round(active_original_result[i] - inactive_original_result[i], 4))
                        for i in range(len(active_original_result))]
        ugf_fair = [abs(round(active_fair_result[i] - inactive_fair_result[i], 4))
                    for i in range(len(active_fair_result))]
        # 打印测试结果
        print(SPLIT)
        print(f"{metric}指标结果:")
        print("\t\t\t\t损失\t准确率\tNDCG\tF1")
        print(f"原始模型\t总体\t\t{str_format(overall_original_result)}")
        print(f"\t\t活跃用户\t{str_format(active_original_result)}")
        print(f"\t\t不活跃用户\t{str_format(inactive_original_result)}")
        print(f"\t\tUGF差距\t{str_format(ugf_original)}")
        print(f"公平模型\t总体\t\t{str_format(overall_fair_result)}")
        print(f"\t\t活跃用户\t{str_format(active_fair_result)}")
        print(f"\t\t不活跃用户\t{str_format(inactive_fair_result)}")
        print(f"\t\tUGF差距\t{str_format(ugf_fair)}")
# 模型配置映射字典
model_dic_config = {
    'PMF': config.pmf_config,
    'VAECF': config.vaecf_config,
    'NeuMF': config.neumf_config,
    'NGCF': config.ngcf_config
}
# 主程序开始
if __name__ == "__main__":
    # 解析命令行参数
    args = parser.parse_args()
    model = args.model
    # 加载模型配置
    if args.model in model_dic_config:
        config = model_dic_config[args.model]
        print(f"使用模型: {args.model}, 配置: {config}")
    else:
        raise ValueError("选择的模型无效!")
    # 设置随机种子
    set_seed(args.seed)
    # 设置计算设备
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_index}" if args.cuda else "cpu")
    print(args)
    # 构建日志目录路径
    log = os.path.join(args.log, '{}_{}_{}_{}_{}'.format(
        args.dataset, args.model, args.epochs, args.neighbor_num, args.model_time))
    # 检查日志目录是否存在
    if not os.path.exists(log):
        raise FileNotFoundError(f"错误: 日志路径 '{log}' 不存在。")
    # 创建结果目录
    result_dir = os.path.join("result", args.dataset)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)  # 创建数据集结果目录
    result_dir = os.path.join(result_dir, args.model)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)  # 创建模型结果目录
    print(f"结果文件将保存在: {result_dir}")
    # 初始化结果存储列表（未使用）
    results = [[], [], [], [], [], [], [], []]
    # 打印基本配置信息
    print(f"计算设备: {device}")
    print(f"数据集: {args.dataset}")
    print(f"评估模型: {args.model}")
    print(f"L2系数: {L2}")
    print(SPLIT)
    # 加载数据集
    print("加载数据集...")
    dataset = myDatasetNew(dataset=args.dataset, train_neg_num=config["num_negative"],
                           neighbor_num=args.neighbor_num, result_path=result_dir)
    # 获取验证集和测试集数据（使用留一法采样）
    (tune_user_set, tune_item_set, tune_labels, tune_sample_num), \
    (test_user_set, test_item_set, test_labels, test_sample_num), \
    (active_tune_user_set, active_tune_item_set, active_tune_labels, active_tune_sample_num), \
    (active_test_user_set, active_test_item_set, active_test_labels, active_test_sample_num), \
    (inactive_tune_user_set, inactive_tune_item_set, inactive_tune_labels, inactive_tune_sample_num), \
    (inactive_test_user_set, inactive_test_item_set, inactive_test_labels, inactive_test_sample_num) = \
        dataset.instance_tune_test_set(choice="loo")
    # 获取数据集统计信息
    user_num, item_num, active_users, inactive_users = dataset.get_statistic()
    # 将数据移动到设备
    active_indices = torch.tensor(active_users).to(device)  # 活跃用户索引（未使用）
    inactive_indices = torch.tensor(inactive_users).to(device)  # 不活跃用户索引（未使用）
    # 验证集数据移动到设备
    tune_user_set = tune_user_set.to(device)
    tune_item_set = tune_item_set.to(device)
    tune_labels = tune_labels.to(device)
    # 测试集数据移动到设备
    test_user_set = test_user_set.to(device)
    test_item_set = test_item_set.to(device)
    test_labels = test_labels.to(device)
    # 活跃用户验证集数据移动到设备
    active_tune_user_set = active_tune_user_set.to(device)
    active_tune_item_set = active_tune_item_set.to(device)
    active_tune_labels = active_tune_labels.to(device)
    # 活跃用户测试集数据移动到设备
    active_test_user_set = active_test_user_set.to(device)
    active_test_item_set = active_test_item_set.to(device)
    active_test_labels = active_test_labels.to(device)
    # 不活跃用户验证集数据移动到设备
    inactive_tune_user_set = inactive_tune_user_set.to(device)
    inactive_tune_item_set = inactive_tune_item_set.to(device)
    inactive_tune_labels = inactive_tune_labels.to(device)
    # 不活跃用户测试集数据移动到设备
    inactive_test_user_set = inactive_test_user_set.to(device)
    inactive_test_item_set = inactive_test_item_set.to(device)
    inactive_test_labels = inactive_test_labels.to(device)
    print("数据加载成功!")
    print(SPLIT)
    # 模型选择字典
    model_dic = {
        'PMF': PMF.PMF,  # 概率矩阵分解
        'VAECF': VAECF.VAECF,  # 变分自编码协同过滤
        'NeuMF': NeuMF.NeuMF,  # 神经矩阵分解
        'NGCF': NGCF.NGCF  # 神经图协同过滤
    }
    # 初始化模型
    model_class = model_dic[args.model]
    model_original = model_class(config, user_num, item_num, device=device)  # 原始模型
    model_fair = model_class(config, user_num, item_num, device=device)  # 公平模型
    model_original.to(device)  # 移动到设备
    model_fair.to(device)  # 移动到设备
    # 初始化损失函数
    bce_loss = torch.nn.BCELoss()  # 二元交叉熵损失
    # 执行测试
    test_model()