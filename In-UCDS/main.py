import argparse  # 命令行参数解析
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 数据可视化库
import torch  # PyTorch深度学习框架
import os  # 操作系统接口
import config  # 配置文件
from tqdm import tqdm  # 进度条工具
from sklearn.metrics import ndcg_score, f1_score, accuracy_score  # 评估指标
from myloss import myLoss  # 自定义损失函数
from sigdatasets import myDatasetNew  # 自定义数据集类
import time  # 时间相关功能
import pandas as pd  # 数据处理库
import ucds  # UCDS算法实现
import random  # 随机数生成
import models.PMF as PMF  # 概率矩阵分解模型
import models.NeuMF as NeuMF  # 神经矩阵分解模型
import models.NGCF as NGCF  # 神经图协同过滤模型
import models.VAECF as VAECF  # 变分自编码协同过滤模型
# 全局常量定义
SPLIT = "=" * 60  # 分隔线（60个等号）
K = 10  # 评估时考虑的top-K数量
L2 = 1e-5  # 自定义损失中的L2系数
original_model_time = 0  # 原始模型训练时间记录
fair_model_time = 0  # 公平模型训练时间记录
# 最佳指标索引常量（用于结果记录）
BEST_NDCG_ALL = 0
BEST_F1_ALL = 1
BEST_NDCG_ACTIVE = 2
BEST_F1_ACTIVE = 3
BEST_NDCG_INACTIVE = 4
BEST_F1_INACTIVE = 5
BEST_NDCG_UGF = 6
BEST_F1_UGF = 7
# 结果存储索引常量
RESULT_USER = 0
RESULT_ITEM = 1
RESULT_SCORE = 2
RESULT_LABEL = 3
# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用CUDA训练')
parser.add_argument('--seed', type=int, default=10, help='随机种子')
parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
parser.add_argument('--model', type=str, default="NeuMF", help="训练模型")
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
# In-UCDS训练函数
def train_InUCDS(epoch):
    """训练原始模型和公平模型(In-UCDS)"""
    print(SPLIT)
    print(f"训练轮次: {epoch}")
    # 初始化损失记录列表: [总损失, BCE损失, 自定义损失]
    original_loss_list = [[], [], []]  
    fair_loss_list = [[], [], []]       
    # 使用进度条遍历数据加载器
    for batch_idx, data in tqdm(enumerate(sample_generator)):
        # 将数据移动到指定设备
        users, items, rating = data[0].to(device), data[1].to(device), data[2].to(device)
        # 获取活跃/不活跃用户样本
        active_inactive_samples = dataset.get_most_similar_active_user(users)
        active_inactive_samples = active_inactive_samples.to(device)
        # 设置模型为训练模式
        model_original.train()
        model_fair.train()
        # 清空梯度
        original_model_opt.zero_grad()
        fair_model_opt.zero_grad()
        # ----------------------------------------------------
        # 训练原始模型
        # ----------------------------------------------------
        original_model_predict = model_original(users, items)  # 原始模型预测
        original_model_bce_loss = bce_loss(original_model_predict.view(-1), rating)  # 计算BCE损失
        # 获取原始模型的嵌入向量
        inactive_embedding_samples_original = model_original.inactive_embeddings_for_my_loss(active_inactive_samples)
        neighbor_embedding_samples_original = model_original.neighbor_embeddings_for_my_loss(active_inactive_samples)
        # 计算原始模型的自定义损失
        original_model_my_loss = my_loss(inactive_embedding_samples_original, neighbor_embedding_samples_original)
        # 原始模型总损失（仅BCE）
        original_model_loss = original_model_bce_loss
        # 记录损失值
        original_loss_list[0].append(original_model_loss.item())
        original_loss_list[1].append(original_model_bce_loss.item())
        original_loss_list[2].append(original_model_my_loss.item())
        # 反向传播
        original_model_loss.backward()
        original_model_opt.step()  # 更新参数
        # ----------------------------------------------------
        # 训练公平模型 (In-UCDS)
        # ----------------------------------------------------
        fair_model_predict = model_fair(users, items)  # 公平模型预测
        fair_model_bce_loss = bce_loss(fair_model_predict.view(-1), rating)  # 计算BCE损失
        # 获取公平模型的嵌入向量
        inactive_embedding_samples = model_fair.inactive_embeddings_for_my_loss(samples)
        neighbor_embedding_samples = model_fair.neighbor_embeddings_for_my_loss(samples)
        # 计算公平模型的自定义损失
        fair_model_my_loss = my_loss(inactive_embedding_samples, neighbor_embedding_samples)
        # 公平模型总损失 = BCE损失 + 自定义损失
        fair_model_loss = fair_model_bce_loss + fair_model_my_loss
        # 记录损失值
        fair_loss_list[0].append(fair_model_loss.item())
        fair_loss_list[1].append(fair_model_bce_loss.item())
        fair_loss_list[2].append(fair_model_my_loss.item())
        # 反向传播
        fair_model_loss.backward()
        fair_model_opt.step()  # 更新参数
    # 打印本轮平均损失
    print(f"原始模型损失: {round(np.mean(original_loss_list[0]), 4)}, "
          f"BCE损失: {round(np.mean(original_loss_list[1]), 4)}, "
          f"自定义损失: {round(np.mean(original_loss_list[2]), 4)},")
    print(f"公平模型损失: {round(np.mean(fair_loss_list[0]), 4)}, "
          f"BCE损失: {round(np.mean(fair_loss_list[1]), 4)}, "
          f"自定义损失: {round(np.mean(fair_loss_list[2]), 4)},")

@torch.no_grad()  # 禁用梯度计算
def tune_new(epoch):
    """在验证集上评估模型性能"""
    # 评估原始模型在不同用户组的表现
    active_original_result = evaluate_one_model(model_original, active_tune_user_set,
                                                active_tune_item_set, active_tune_labels, active_tune_sample_num)
    inactive_original_result = evaluate_one_model(model_original, inactive_tune_user_set,
                                                  inactive_tune_item_set, inactive_tune_labels,
                                                  inactive_tune_sample_num)
    overall_original_result = evaluate_one_model(model_original, tune_user_set,
                                                 tune_item_set, tune_labels, tune_sample_num)
    # 评估公平模型在不同用户组的表现
    overall_fair_result = evaluate_one_model(model_fair, tune_user_set,
                                             tune_item_set, tune_labels, tune_sample_num)
    active_fair_result = evaluate_one_model(model_fair, active_tune_user_set,
                                            active_tune_item_set, active_tune_labels, active_tune_sample_num)
    inactive_fair_result = evaluate_one_model(model_fair, inactive_tune_user_set,
                                              inactive_tune_item_set, inactive_tune_labels, inactive_tune_sample_num)
    # 计算用户组间公平性差距 (UGF)
    ugf_original = [abs(round(active_original_result[i] - inactive_original_result[i], 4))
                    for i in range(len(active_original_result))]
    ugf_fair = [abs(round(active_fair_result[i] - inactive_fair_result[i], 4))
                for i in range(len(active_fair_result))]
    # 在后半段训练轮次保存最佳模型
    if epoch > args.epochs // 2:
        save_model(model=model_original, model_best_results=best_metric_result_original,
                   this_ndcg_all=overall_original_result[2], this_f1_all=overall_original_result[3],
                   this_ndcg_active=active_original_result[2], this_f1_active=active_original_result[3],
                   this_ndcg_inactive=inactive_original_result[2], this_f1_inactive=inactive_original_result[3],
                   this_ndcg_ugf=ugf_original[2], this_f1_ugf=ugf_original[3], model_name="original")
        save_model(model=model_fair, model_best_results=best_metric_result_fair,
                   this_ndcg_all=overall_fair_result[2], this_f1_all=overall_fair_result[3],
                   this_ndcg_active=active_fair_result[2], this_f1_active=active_fair_result[3],
                   this_ndcg_inactive=inactive_fair_result[2], this_f1_inactive=inactive_fair_result[3],
                   this_ndcg_ugf=ugf_fair[2], this_f1_ugf=ugf_fair[3], model_name="fair")
    # 打印验证结果
    print("验证结果:")
    print("\t\t\t\t损失\t准确率\tNDCG\tF1")
    print(f"原始模型\t总体\t\t{str_format(overall_original_result)}")
    print(f"\t\t活跃用户\t{str_format(active_original_result)}")
    print(f"\t\t不活跃用户\t{str_format(inactive_original_result)}")
    print(f"\t\tUGF差距\t{str_format(ugf_original)}")
    print(f"公平模型\t总体\t\t{str_format(overall_fair_result)}")
    print(f"\t\t活跃用户\t{str_format(active_fair_result)}")
    print(f"\t\t不活跃用户\t{str_format(inactive_fair_result)}")
    print(f"\t\tUGF差距\t{str_format(ugf_fair)}")

def save_model(model, model_best_results, this_ndcg_all, this_f1_all, this_ndcg_active, this_f1_active, this_ndcg_inactive,
               this_f1_inactive, this_ndcg_ugf, this_f1_ugf, model_name):
    """根据评估指标保存最佳模型"""
    # 更新最佳NDCG@K指标
    if this_ndcg_all > model_best_results[BEST_NDCG_ALL]:
        model_best_results[BEST_NDCG_ALL] = this_ndcg_all
        torch.save(model.state_dict(), os.path.join(log, f"best_ndcg_all_{model_name}.pkl"))
    # 更新最佳F1指标
    if this_f1_all > model_best_results[BEST_F1_ALL]:
        model_best_results[BEST_F1_ALL] = this_f1_all
        torch.save(model.state_dict(), os.path.join(log, f"best_f1_all_{model_name}.pkl"))
    # 更新活跃用户最佳NDCG@K
    if this_ndcg_active > model_best_results[BEST_NDCG_ACTIVE]:
        model_best_results[BEST_NDCG_ACTIVE] = this_ndcg_active
        torch.save(model.state_dict(), os.path.join(log, f"best_ndcg_active_{model_name}.pkl"))
    # 更新活跃用户最佳F1
    if this_f1_active > model_best_results[BEST_F1_ACTIVE]:
        model_best_results[BEST_F1_ACTIVE] = this_f1_active
        torch.save(model.state_dict(), os.path.join(log, f"best_f1_active_{model_name}.pkl"))
    # 更新不活跃用户最佳NDCG@K
    if this_ndcg_inactive > model_best_results[BEST_NDCG_INACTIVE]:
        model_best_results[BEST_NDCG_INACTIVE] = this_ndcg_inactive
        torch.save(model.state_dict(), os.path.join(log, f"best_ndcg_inactive_{model_name}.pkl"))
    # 更新不活跃用户最佳F1
    if this_f1_inactive > model_best_results[BEST_F1_INACTIVE]:
        model_best_results[BEST_F1_INACTIVE] = this_f1_inactive
        torch.save(model.state_dict(), os.path.join(log, f"best_f1_inactive_{model_name}.pkl"))
    # 更新最小UGF差距 (NDCG)
    if this_ndcg_ugf < model_best_results[BEST_NDCG_UGF]:
        model_best_results[BEST_NDCG_UGF] = this_ndcg_ugf
        torch.save(model.state_dict(), os.path.join(log, f"best_ndcg_ugf_{model_name}.pkl"))
    # 更新最小UGF差距 (F1)
    if this_f1_ugf < model_best_results[BEST_F1_UGF]:
        model_best_results[BEST_F1_UGF] = this_f1_ugf
        torch.save(model.state_dict(), os.path.join(log, f"best_f1_ugf_{model_name}.pkl"))

def str_format(data: list):
    """格式化输出列表数据"""
    return '\t'.join([f"{item:.4f}" for item in data])

@torch.no_grad()
def evaluate_one_model(model, user_set, item_set, labels, sample_num, if_test=False, result_num=0):
    """评估单个模型性能"""
    predict = model(user_set, item_set)  # 模型预测
    # 测试时保存详细结果
    if if_test:
        # 保存用户ID、物品ID、预测分数和真实标签
        results[result_num].append([int(user) for user in user_set])
        results[result_num].append([int(item) for item in item_set])
        results[result_num].append([float(score) for score in predict])
        results[result_num].append([float(label) for label in labels])
    
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
                                                     test_item_set, test_labels, test_sample_num,
                                                     if_test=True, result_num=i)
        # 评估公平模型
        overall_fair_result = evaluate_one_model(model_fair, test_user_set,
                                                 test_item_set, test_labels, test_sample_num)
        active_fair_result = evaluate_one_model(model_fair, active_test_user_set,
                                                active_test_item_set, active_test_labels, active_test_sample_num)
        inactive_fair_result = evaluate_one_model(model_fair, inactive_test_user_set,
                                                  inactive_test_item_set, inactive_test_labels,
                                                  inactive_test_sample_num)
        # 计算UGF差距
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
        # 保存详细结果到CSV
        df = pd.DataFrame(
            data={"uid": results[i][RESULT_USER],
                  "iid": results[i][RESULT_ITEM],
                  "score": results[i][RESULT_SCORE],
                  "label": results[i][RESULT_LABEL]})
        filename = os.path.join(result_dir, f"{args.model}_{metric}_rank.csv")
        df.to_csv(filename, sep='\t', index=False)
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
    # 创建日志目录
    log = os.path.join(args.log, '{}_{}_{}_{}_{}'.format(
        args.dataset, args.model, args.epochs, args.neighbor_num, args.model_time))
    # 处理已存在的日志目录
    if os.path.isdir(log):
        print(f"{log} 已存在，是否覆盖？5秒后继续，按Ctrl-C取消。")
        time.sleep(5)
        os.system('rm -rf %s/' % log)
    os.makedirs(log, exist_ok=True)
    print("创建日志目录:", log)
    # 创建结果目录
    result_dir = os.path.join("result", args.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    result_dir = os.path.join(result_dir, args.model)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    print(f"结果文件将保存在: {result_dir}")
    # 初始化结果存储列表
    results = [[], [], [], [], [], [], [], []]
    # 打印基本配置信息
    print(f"训练设备: {device}")
    print(f"数据集: {args.dataset}")
    print(f"训练模型: {args.model}")
    print(f"L2系数: {L2}")
    print(SPLIT)
    # 加载数据集
    print("加载数据集...")
    dataset = myDatasetNew(dataset=args.dataset, train_neg_num=config["num_negative"],
                           neighbor_num=args.neighbor_num, result_path=result_dir)
    # 获取用户相似度矩阵
    similarity_matrix = dataset.get_user_similarity_matrix()
    # 生成验证集和测试集
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
    active_indices = torch.tensor(active_users).to(device)
    inactive_indices = torch.tensor(inactive_users).to(device)
    tune_user_set = tune_user_set.to(device)
    tune_item_set = tune_item_set.to(device)
    tune_labels = tune_labels.to(device)
    test_user_set = test_user_set.to(device)
    test_item_set = test_item_set.to(device)
    test_labels = test_labels.to(device)
    active_tune_user_set = active_tune_user_set.to(device)
    active_tune_item_set = active_tune_item_set.to(device)
    active_tune_labels = active_tune_labels.to(device)
    active_test_user_set = active_test_user_set.to(device)
    active_test_item_set = active_test_item_set.to(device)
    active_test_labels = active_test_labels.to(device)
    inactive_tune_user_set = inactive_tune_user_set.to(device)
    inactive_tune_item_set = inactive_tune_item_set.to(device)
    inactive_tune_labels = inactive_tune_labels.to(device)
    inactive_test_user_set = inactive_test_user_set.to(device)
    inactive_test_item_set = inactive_test_item_set.to(device)
    inactive_test_labels = inactive_test_labels.to(device)
    print("数据加载成功!")
    print(SPLIT)
    # 初始化最佳指标记录器
    best_metric_result_original = [0, 0, 0, 0, 0, 0, 1e5, 1e5]  # 原始模型
    best_metric_result_fair = [0, 0, 0, 0, 0, 0, 1e5, 1e5]  # 公平模型
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
    # 初始化UCDS模型
    InUCDS = ucds.InUCDS(num_users=user_num, active_ids=active_users,
                         inactive_ids=inactive_users, similarity_matrix=similarity_matrix)
    # 为不活跃用户生成主导集
    samples = InUCDS.generate_dominant_sets_for_all_inactive(neighbor_num=args.neighbor_num)
    # 初始化损失函数
    bce_loss = torch.nn.BCELoss()  # 二元交叉熵损失
    my_loss = myLoss(l2=L2)  # 自定义损失
    # 初始化优化器
    original_model_opt = torch.optim.Adam(model_original.parameters(),
                                          lr=config['adam_lr'],
                                          weight_decay=config['l2_regularization'])
    fair_model_opt = torch.optim.Adam(model_fair.parameters(),
                                      lr=config['adam_lr'],
                                      weight_decay=config['l2_regularization'])
    # 创建数据加载器
    sample_generator = dataset.instance_a_train_loader(config['batch_size'])
    # 训练循环
    for epoch in range(args.epochs):
        train_InUCDS(epoch)  # 训练一个epoch
        tune_new(epoch)  # 验证模型性能
    # 最终测试
    test_model()