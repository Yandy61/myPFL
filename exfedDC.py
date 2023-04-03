import ssl
import os
import sys
from py_func.read_db import get_dataloaders
from py_func.create_model import load_model


def train_process(filename:str, para:list):

    if len(para) < 6:
        print("Para Error!")
        sys.exit()

    # 获取输入参数
    dataset = para[0]           # 数据集
    mod = para[1]               # 模型
    lr = float(para[2])         # 学习率
    decay = float(para[3])      # 分组衰减
    n_iter = int(para[4])       # 总轮次
    alpha_coef = float(para[5])
    mu = float(para[6])
    dmu = float(para[7])

    # 内置默认参数
    seed = 0                # 模型种子
    n_SGD = 20              # 本地轮次
    p = 1.0                 # 选取比例
    batch_size = 50         # batch_size
    meas_perf_period = 5    # 汇报频率


    """获取超参数"""
    # 全局轮次、batch_size、acc汇报频率
    # n_iter, batch_size, meas_perf_period = get_hyperparams(dataset, n_SGD)

    """用于保存数据的文件名"""
    file_name = (
        f"fedDC_{dataset}_{mod}_lr{lr}_d{decay}_alpha{alpha_coef}_mu{mu}_dmu{dmu}_iter{n_iter}_{filename}"
    )

    print("==========>>> FedDC 超参数 <<<========")
    print("  全局轮次: ", n_iter)
    print("  本地轮次: ", n_SGD)
    print("  batch size: ", batch_size)
    print("  汇报频率: ", meas_perf_period)
    print("  alpha: ", alpha_coef)
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
    model_0 = load_model(mod, seed)
    print(f"模型加载完成：{mod}")

        
    from py_func.fedDC import feddc

    feddc(
        model_0,            # 模型
        n_sampled,          # 每轮选取个数
        list_dls_train,     # 训练集
        list_dls_test,      # 测试集
        n_iter + 1,         # 全局轮次
        n_SGD,              # 本地轮次
        lr,                 # 学习率
        file_name,          # pkl存储名
        decay,              # 分组时学习率衰减
        meas_perf_period,   # 修改meas_perf_period, acc汇报频率
        alpha_coef,          # fedDC 正则系数
        mu,
        dmu
    )


    print("\nEXPERIMENT IS FINISHED")


if __name__ == '__main__':

    folder = "./experiments_info/"
    filename = sys.argv[1]
    # filename = "dc0"
    paraList = []

    with open(folder + filename + ".txt") as file:
        paraList = file.readlines()

    for para in paraList:
        paras = para.split(" ")
        train_process(filename, paras)

    print("All Done!")
