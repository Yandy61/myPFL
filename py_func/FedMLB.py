from .FedProx import *
import torch.nn.functional as F

def FedMLB(
    model,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay=1.0,
    metric_period=1
):
    """基于标签多样性与梯度方向的个性化联邦学习
    Parameters:
        - `model`: 模型
        - `training_sets`: 训练数据集列表,index与客户端index相对应
        - `testing_set`: 训练数据集列表,index与客户端index相对应
        - `n_iter`: 全局轮次
        - `n_SGD`: 本地轮次
        - `lr`: 初始学习率
        - `file_name`: 用于存储训练结果的文件名
        - `decay`: 分组初始化阶段衰减系数，仅应用一轮
        - `metric_period`: 训练数据记录频次

    returns :
        - `model`: 最终的全局模型
    """

    print("========>>> 正在初始化训练")

    # GPU选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 移动模型到cuda
    print(f"模型放入{device}")

    # 损失函数
    loss_f = loss_classifier

    ''' ------------------------变量初始化>>>>>>>>>>>>>>>>>>>>>>>> '''
    K = len(training_sets)  # clients总数
    
    n_samples = np.array([len(db.dataset) for db in training_sets])     # array(K)，每个client拥有的样本数量
    
    weights = n_samples / np.sum(n_samples)     # array(K)，每个client样本数量所占权重

    loss_hist = np.zeros((n_iter + 1, K))       # array(n+1,k),记录n轮、k个设备的loss(全局模型)
    acc_hist = np.zeros((n_iter + 1, K))        # array(n+1,k),记录n轮、k个设备的acc(全局模型)

    server_loss_hist = []       # list(n,1),记录全局模型n轮的loss
    server_acc_hist = []        # list(n,1),记录全局模型n轮的acc
    
    # 初始化第0轮设备loss、acc
    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(
            model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # 全局模型的loss和acc
    server_loss = np.dot(weights, loss_hist[0])     # 当前轮次全局模型的平均 loss
    server_acc = np.dot(weights, acc_hist[0])       # 当前轮次全局模型的平均 acc
    
    # 将初始loss、acc加入记录
    server_loss_hist.append(server_loss)
    server_acc_hist.append(server_acc)

    starttime = datetime.now()  # 记录训练开始时时间

    ''' <<<<<<<<<<<<<<<<<<<<<<<<变量初始化------------------------ '''

    print("========>>> 初始化完成")
    

    ''' ------------------------完整训练>>>>>>>>>>>>>>>>>>>>>>>> '''
    # 全局轮次循环
    for i in range(n_iter):

        i_time = datetime.now()     # 记录当前轮次开始时间

        clients_params = []     # 当前轮次 所有客户端模型参数（占内存）
        sampled_clients_for_grad = []   # 存储梯度的客户端列表

        # 根据轮次选择参与训练的客户 依据 聚合时的权重
        clients_list = np.arange(K)
        agre_weights = weights

        ''' ------------------------训练阶段>>>>>>>>>>>>>>>>>>>>>>>> '''
        # print(f"========>>> 第{i+1}轮(未分组)")
        for k in clients_list:

            client_index = '{:>{width}}'.format(k, width=2)
            nowtime = datetime.now()    # 记录当前时间
            t0 = str(nowtime - i_time)[2:7]
            print(
                f"\r====>>> {i+1}/{n_iter}:  Iter - {t0} - {client_index}", end = "")
            
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            mlb_training(
                local_model,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f
            )

            # 当前客户端最新模型参数
            list_params = list(local_model.parameters())
            list_params = [
                tens_param.detach() for tens_param in list_params
            ]

            clients_params.append(list_params)  # 存入当前轮次客户端模型参数列表
            sampled_clients_for_grad.append(k)  # 参与训练的客户端下标存入列表

        ''' <<<<<<<<<<<<<<<<<<<<<<<<训练阶段------------------------ '''

        ''' ------------------------聚合>>>>>>>>>>>>>>>>>>>>>>>> '''
        # 更新全局模型
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, agre_weights
        )

        # 使用新模型对客户端计算loss/acc
        if i % metric_period == 0:
            # loss_hist存储训练集loss
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    # 每个设备对新模型的loss
                    loss_dataset(model, dl, loss_f).detach()
                )
            # acc_hist存储测试集acc
            for k, dl in enumerate(testing_sets):
                # 每个设备对新模型的acc
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            # 记录server的loss、acc
            server_loss = np.dot(weights, loss_hist[i + 1])     # 当前轮，全局平均 loss
            server_acc = np.dot(weights, acc_hist[i + 1])       # 当前轮，全局平均 acc
            server_loss_hist.append(server_loss)                # 所有轮，全局平均 loss
            server_acc_hist.append(server_acc)                  # 所有轮，全局平均 acc

        nowtime = datetime.now()    # 记录当前时间
        t0 = str(nowtime - i_time)[2:7]
        if i % metric_period == 0:
            t = str(nowtime - starttime).split(".")[0]
            print(
                f"\r====>>> {i+1}/{n_iter}:  Iter - {t0}      Loss: {server_loss:.3f}   Acc: {server_acc:.3f}   —>Time - {t}", end = ""
            )
        else:
            print(
                f"\r====>>> {i+1}/{n_iter}:  Iter - {t0}", end = ""
            )
        ''' <<<<<<<<<<<<<<<<<<<<<<<<聚合------------------------ '''

        # 学习率衰减
        lr *= decay

        ''' ------------------------每轮覆盖存储loss/acc>>>>>>>>>>>>>>>>>>>>>>>> '''
        # n轮K个客户端的loss与acc,无需存储
        # save_pkl(loss_hist, "loss", file_name)
        # save_pkl(acc_hist, "acc", file_name)
        
        # 全局平均loss/acc
        save_pkl(server_loss_hist, "server_loss", file_name)
        save_pkl(server_acc_hist, "server_acc", file_name)
        ''' <<<<<<<<<<<<<<<<<<<<<<<<每轮覆盖存储loss/acc------------------------ '''

    # 训练结束时存储实验数据
    # save_pkl(loss_hist, "loss", file_name)
    # save_pkl(acc_hist, "acc", file_name)

    
    save_pkl(server_loss_hist, "server_loss", file_name)
    save_pkl(server_acc_hist, "server_acc", file_name)

    # 存储最终模型
    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def KD(input_p, input_q, T=1):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    p = F.softmax(input_p/T, dim = 1)
    q = F.log_softmax(input_q/T, dim = 1)
    result = kl_loss(q,p)
    return result    

def mlb_training(
        model, 
        optimizer, 
        train_data, 
        n_SGD: int, 
        loss_f, 
        select_level=-1, 
        lambda1=0.5,
        lambda2=0.5,
        temperature=1
    ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)  # 移动模型到cuda
    model_0 = deepcopy(model)

    for _ in range(n_SGD):

        features, labels = next(iter(train_data))

        features = features.to(device)  # 移动数据到cuda
        labels = labels.to(device)
        # 或者  labels = labels.cuda() if torch.cuda.is_available() else labels

        optimizer.zero_grad()

        ''' ----- '''
        out_of_local = model(features, return_feature=True)

        local_features = out_of_local[:-1]
        log_probs = out_of_local[-1]
        log_prob_branch = []
        ce_branch = []
        kl_branch = []
        num_branch = len(local_features)

        ## Compute loss from hybrid branches
        for it in range(num_branch):
            if select_level != -1 and select_level != it:
                continue
            this_log_prob = model_0(local_features[it], level=it + 1)
            this_ce = loss_f(this_log_prob, labels)
            this_kl = KD(this_log_prob, log_probs, temperature)
            log_prob_branch.append(this_log_prob)
            ce_branch.append(this_ce)
            kl_branch.append(this_kl)

        ce_loss = loss_f(log_probs, labels)

        loss = ce_loss + lambda1 * (sum(ce_branch)) / num_branch + lambda2 * (sum(kl_branch)) / num_branch
        
        loss.backward()
        optimizer.step()
