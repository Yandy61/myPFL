import ssl
import os
import sys
import torch
from py_func.read_db import get_dataloaders
from py_func.create_model import load_model


def train_process(filename:str, para:list):

    if len(para) < 11:
        print("Para Error!")
        sys.exit()

    # 获取输入参数
    dataset = para[0]           # 数据集
    mod = para[1]               # 模型
    lr = float(para[2])         # 学习率
    decay = float(para[3])      # 分组衰减
    decayP = float(para[4])     # +分组后学利率衰减
    lamda_d = float(para[5])    # 多样性权重
    decayD = float(para[6])     # 多样性权重衰减
    lamda_n = float(para[7])    # 正则系数
    decayN = float(para[8])     # 正则项系数衰减
    n_iter = int(para[9])       # 总轮次
    pre_train = int(para[10])   # pre轮


    # 内置默认参数
    seed = 0                # 模型种子
    n_SGD = 20              # 本地轮次
    p = 1.0                 # 选取比例
    batch_size = 50         # batch_size
    meas_perf_period = 5    # 汇报频率
    force = "True"          # 覆盖结果

    mu = 0                  # 正则系数
    n_clusters = 10         # +分组个数
    local_test = False      # +acc/loss测试时机
    beta = 0.6              # +个性化系数

    """获取超参数"""
    # 全局轮次、batch_size、acc汇报频率
    # n_iter, batch_size, meas_perf_period = get_hyperparams(dataset, n_SGD)

    """用于保存数据的文件名"""
    file_name = (
        f"post_lr{lr}_{decay}_{decayP}_ld{lamda_d}_{decayD}_ln{lamda_n}_{decayN}_iter{n_iter}_{pre_train}_{filename}"
    )

    print("==========>>> 超参数 <<<========")
    print("  全局轮次: ", n_iter)
    print("  Pre轮次: ", pre_train)
    print("  本地轮次: ", n_SGD)
    print("  batch size: ", batch_size)
    print("  汇报频率: ", meas_perf_period)
    print("  个性化系数: ", beta)
    print("  文件名: ", file_name)
    print("===============================")


    """获取数据集"""
    ssl._create_default_https_context = ssl._create_unverified_context
    list_dls_train, list_dls_test, shannon_list = get_dataloaders(
        dataset, batch_size)        # 数据集加载 ！
    # 按100个clients划分好了数据集
    # shannon_list : list(K,1)，元素为每个客户端所持有数据的香农多样性指数


    """根据数据集划分确定设备个数"""
    n_sampled = int(p * len(list_dls_train))
    # print("  number fo sampled clients: ", n_sampled)


    """加载初始模型"""
    model_0 = load_model("CIFAR10_CNN", seed)
    print(f"模型加载完成：{mod}")

    state_dict = torch.load(
        'experiments_res/final_model/CIFAR10_nbal_0.001_CIFAR10_CNN_lr0.1_1.0_1.0_ld0.5_0.995_ln0.2_0.998_iter600_400_sA1_pre400.pth')
    model_0.load_state_dict(state_dict)
    print("模型参数加载成功！")

    import pickle
    gradients = pickle.load(open("experiments_res/gradPre/CIFAR10_nbal_0.001_CIFAR10_CNN_lr0.1_1.0_1.0_ld0.5_0.995_ln0.2_0.998_iter600_400_sA1_grad.pkl", 'rb'))
    print("梯度信息加载成功！")

    if not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force:
        
        from py_func.FedProx import pFedGLAO

        pFedGLAO(
            model_0,            # 模型
            n_sampled,          # 每轮选取个数
            list_dls_train,     # 训练集
            list_dls_test,      # 测试集
            shannon_list,       # 多样性指数
            n_iter + 1,         # 全局轮次
            pre_train,          # +预训练轮次 
            n_SGD,              # 本地轮次
            lr,                 # 学习率
            file_name,          # pkl存储名
            decay,              # 分组时学习率衰减
            meas_perf_period,   # 修改meas_perf_period, acc汇报频率
            mu,                 # fedProx 正则系数
            lamda_d,            # 多样性权重 正则系数
            lamda_n,            # 损失惩罚项 正则系数
            decayN,             # 正则项衰减                `-1`:新尝试
            decayD,             # +多样性权重衰减
            n_clusters,         # +分组个数
            decayP,             # +分组后学利率衰减
            local_test,         # +acc/loss测试时机
            beta,               # +个性化系数
            gradients,          # ++上轮梯度
            1                   # ++仅分组后训练模式
        )


    print("\nEXPERIMENT IS FINISHED")


if __name__ == '__main__':

    folder = "./experiments_info/"
    filename = sys.argv[1]
    # filename = "sA0"
    paraList = []

    with open(folder + filename + ".txt") as file:
        paraList = file.readlines()

    for para in paraList:
        paras = para.split(" ")
        train_process(filename, paras)

    print("All Done!")
