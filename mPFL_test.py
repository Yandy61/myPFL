import sys

def train_process(filename:str, para:list):

    if len(para) < 8:
        print("Para Error!")
        sys.exit()

    # 获取输入参数
    dataset = para[0]           # 数据集
    lr = float(para[1])         # 学习率
    decay = float(para[2])      # 分组衰减
    decay1 = float(para[3])     # 分组衰减
    n_iter = int(para[4])       # 总轮次
    pre_train = int(para[5])    # pre轮次
    n_clusters = int(para[6])   # 聚类数量
    mu = float(para[7])         # fedprox系数
    
    # 内置默认参数
    seed = 0                # 模型种子
    n_SGD = 20              # 本地轮次
    p = 1.0                 # 选取比例
    batch_size = 50         # batch_size
    meas_perf_period = 10   # 汇报频率
    force = "True"          # 覆盖结果


    """获取超参数"""
    # 全局轮次、batch_size、acc汇报频率
    # n_iter, batch_size, meas_perf_period = get_hyperparams(dataset, n_SGD)


    print("  ========>>> 超参数 <<<========")
    print("  全局轮次: ", n_iter)
    print("  本地轮次: ", n_SGD)
    print("  batch size: ", batch_size)
    print("  分组前训练轮次: ", pre_train)
    print("  分组数量: ", n_clusters)
    print("  学习率: ", lr)
    print("  decay: ", decay)
    print("  =============================")


    """用于保存数据的文件名"""
    file_name = (
        f"{filename}_dataset{dataset}_lr{lr}_da{decay}_db{decay1}_iter{n_iter}_pre{pre_train}"
    )

    print("  " + file_name)

    print("EXPERIMENT IS FINISHED")


if __name__ == '__main__':

    folder = "./experiments_info/"
    filename = sys.argv[1]
    paraList = []

    with open(folder + filename + ".txt") as file:
        paraList = file.readlines()

    for para in paraList:
        paras = para.split(" ")
        train_process(filename, paras)

    print("All Done!")
