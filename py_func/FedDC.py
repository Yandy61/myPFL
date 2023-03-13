from FedProx import *
import copy

def feddc(
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay=1.0,
    metric_period=1,
    mu=0.0,
    alpha_coef=0.0
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
        - `mu`: fedProx 正则系数

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
    dc_loss_hist = np.zeros((n_iter + 1, K))       # array(n+1,k),记录n轮、k个设备的loss(全局模型)
    dc_acc_hist = np.zeros((n_iter + 1, K))        # array(n+1,k),记录n轮、k个设备的acc(全局模型)

    server_loss_hist = []       # list(n,1),记录全局模型n轮的loss
    server_acc_hist = []        # list(n,1),记录全局模型n轮的acc
    dc_server_loss_hist = []       # list(n,1),记录全局模型n轮的loss
    dc_server_acc_hist = []        # list(n,1),记录全局模型n轮的acc
    
    # 初始化第0轮设备loss、acc
    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(
            model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)
        
        dc_loss_hist[0,k] = loss_hist[0, k]
        dc_acc_hist[0,k] = acc_hist[0, k]

    # 全局模型的loss和acc
    server_loss = np.dot(weights, loss_hist[0])     # 当前轮次全局模型的平均 loss
    server_acc = np.dot(weights, acc_hist[0])       # 当前轮次全局模型的平均 acc
    
    # 将初始loss、acc加入记录
    server_loss_hist.append(server_loss)
    server_acc_hist.append(server_acc)

    dc_server_loss_hist.append(server_loss)
    dc_server_acc_hist.append(server_acc)

    '''********************************************************************'''
    weight_list = weights * K

    # list(K) ,上一轮的梯度列表, 参数类型
    gradients = get_gradients(model, [model] * K)
    
    # h_i
    parameter_drifts = copy.deepcopy(gradients)

    state_gadient_diffs = get_gradients(model, [model] * K+1)
    
    # # 上一轮的总梯度，模型类型
    # grad = get_grad(model, model)
    # grad_para = get_model_para(grad)

    starttime = datetime.now()  # 记录训练开始时时间

    ''' <<<<<<<<<<<<<<<<<<<<<<<<变量初始化------------------------ '''

    print("========>>> 初始化完成")
    

    ''' ------------------------完整训练>>>>>>>>>>>>>>>>>>>>>>>> '''
    # 全局轮次循环
    for i in range(n_iter):
        i_time = datetime.now()     # 记录当前轮次开始时间

        # previous_global_model = deepcopy(model)     # 上一轮的全局模型

        global_model_param = get_model_para(model)
        delta_g_sum = np.array(global_model_param - global_model_param)

        clients_params = []     # 当前轮次 所有客户端模型参数（占内存）
        # clients_models = []     # 当前轮次 所有客户端模型

        # 根据轮次选择参与训练的客户 依据 聚合时的权重
        clients_list = np.arange(K)
        agre_weights = weights
        
        ''' ------------------------训练阶段>>>>>>>>>>>>>>>>>>>>>>>> '''
        # print(f"========>>> 第{i+1}轮(未分组)")
        for k in clients_list:
            print(f"---Training client {k}")

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            '''********************** FedDC ******************'''
            local_update_last = state_gadient_diffs[k]
            global_update_last = state_gadient_diffs[-1]/weight_list[k]
            alpha = alpha_coef / weight_list[k]
            h_i_k = parameter_drifts[k]
            '''***********************************************'''

            ''' DONE: 修改本地优化函数，需考虑本地模型与全局模型的更新方向 '''
            fedDC_local_learning(
                local_model,
                alpha,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
                lr,
                global_update_last,
                local_update_last,
                h_i_k
            )

            # 当前客户端最新模型参数
            list_params = list(local_model.parameters())
            list_params = [
                tens_param.detach() for tens_param in list_params
            ]
            
            '''***********************************************'''
            delta_param_curr = list_params - global_model_param
            parameter_drifts[k] += delta_param_curr
            
            # ?????
            beta = 1/n_SGD/lr

            state_g = local_update_last - global_update_last + beta * (-delta_param_curr) 
            delta_g_cur = (state_g - state_gadient_diffs[k])*weight_list[k]
            delta_g_sum += delta_g_cur
            state_gadient_diffs[k] = state_g

            '''***********************************************'''

            clients_params.append(list_params)  # 存入当前轮次客户端模型参数列表
            # clients_models.append(deepcopy(local_model))    # 存入当前轮次客户端模型列表

        ''' <<<<<<<<<<<<<<<<<<<<<<<<训练阶段------------------------ '''

        ''' ------------------------聚合>>>>>>>>>>>>>>>>>>>>>>>> '''
        avgmodel = FedAvg_agregation_process(
            deepcopy(model), clients_params, agre_weights
        )

        '''***********************************************'''
        avg_mdl_param_sel = get_model_para(avgmodel)

        delta_g_cur = 1 / K * delta_g_sum

        state_gadient_diffs[-1] += delta_g_cur 

        # 加上了hi的全局模型？
        cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0) #

        # TODO cld_mdl_param加载到模型中，DC全局模型，cloud模型
        model = set_param_to_model(model,cld_mdl_param)

        # 使用新模型对客户端计算loss/acc
        if i % metric_period == 0:
            # loss_hist存储训练集loss
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    # 每个设备对新模型的loss
                    loss_dataset(avgmodel, dl, loss_f).detach()
                )
                dc_loss_hist[i + 1, k] = float(
                    # 每个设备对新模型的loss
                    loss_dataset(model, dl, loss_f).detach()
                )
            # acc_hist存储测试集acc
            for k, dl in enumerate(testing_sets):
                # 每个设备对新模型的acc
                acc_hist[i + 1, k] = accuracy_dataset(avgmodel, dl)
                dc_acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            # 记录server的loss、acc
            server_loss = np.dot(weights, loss_hist[i + 1])     # 当前轮，全局平均 loss
            server_acc = np.dot(weights, acc_hist[i + 1])       # 当前轮，全局平均 acc
            dc_server_loss = np.dot(weights, dc_loss_hist[i + 1])     # 当前轮，全局平均 loss
            dc_server_acc = np.dot(weights, dc_acc_hist[i + 1])       # 当前轮，全局平均 acc

            server_loss_hist.append(server_loss)                # 所有轮，全局平均 loss
            server_acc_hist.append(server_acc)                  # 所有轮，全局平均 acc
            dc_server_loss_hist.append(dc_server_loss)                # 所有轮，全局平均 loss
            dc_server_acc_hist.append(dc_server_acc)                  # 所有轮，全局平均 acc

        nowtime = datetime.now()    # 记录当前时间
        if i % metric_period == 0:
            t = str(nowtime - starttime).split(".")[0]
            print(
                f"========>>> 第{i+1}轮:   Loss: {dc_server_loss}    Server Test Accuracy: {dc_server_acc}   —>Time: {t}"
            )
        else:
            t = str(nowtime - i_time).split(".")[0]
            print(
                f"========>>> 第{i+1}轮:   Done  IterTime: {t}"
            )
        ''' <<<<<<<<<<<<<<<<<<<<<<<<聚合------------------------ '''


        # # 更新最新梯度gradients
        # gradients = get_gradients(
        #     previous_global_model, clients_models
        # )

        # # 更新总梯度
        # grad = get_grad(previous_global_model, model)

        # 学习率衰减
        lr *= decay

        ''' ------------------------每轮覆盖存储loss/acc>>>>>>>>>>>>>>>>>>>>>>>> '''
        # n轮K个客户端的loss与acc,无需存储
        # save_pkl(loss_hist, "loss", file_name)
        # save_pkl(acc_hist, "acc", file_name)
        
        # 全局平均loss/acc
        save_pkl(server_loss_hist, "server_loss", file_name+"_G_loss")
        save_pkl(server_acc_hist, "server_acc", file_name+"_G_acc")
        save_pkl(dc_server_loss_hist, "server_loss", file_name+"_Dc_loss")
        save_pkl(dc_server_acc_hist, "server_acc", file_name+"_Dc_acc")
        ''' <<<<<<<<<<<<<<<<<<<<<<<<每轮覆盖存储loss/acc------------------------ '''

    # 训练结束时存储实验数据
    save_pkl(server_loss_hist, "server_loss", file_name+"_G_loss")
    save_pkl(server_acc_hist, "server_acc", file_name+"_G_acc")
    save_pkl(dc_server_loss_hist, "server_loss", file_name+"_Dc_loss")
    save_pkl(dc_server_acc_hist, "server_acc", file_name+"_Dc_acc")

    # 存储最终模型
    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def get_model_para(model, flag=True):
    '''将模型转为参数list'''
    if not flag:
        return model

    list_params = list(model.parameters())
    list_params = [
        tens_param.detach() for tens_param in list_params
    ]

    return list_params

def fedDC_local_learning(
        model,
        alpha: float,
        optimizer,
        train_data,
        n_SGD: int,
        loss_f,
        lr,
        grad_global_pre:list,
        grad_local_pre:list,
        h_i_k:list
    ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)  # 移动模型到cuda
    model_0 = deepcopy(model)

    global_model_param = torch.tensor(get_model_para(model_0), dtype=torch.float32, device=device)
    hist_i = torch.tensor(h_i_k, dtype=torch.float32, device=device)
    state_update_diff = torch.tensor(-grad_local_pre+ grad_global_pre,  dtype=torch.float32, device=device)


    for _ in range(n_SGD):

        features, labels = next(iter(train_data))

        features = features.to(device)  # 移动数据到cuda
        labels = labels.to(device)
        # 或者  labels = labels.cuda() if torch.cuda.is_available() else labels

        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_f(predictions, labels)

        local_model_para = list(model.parameters())

        loss_r = alpha/2 * torch.sum((local_model_para - (global_model_param - hist_i))*(local_model_para - (global_model_param - hist_i)))
        loss_g = torch.sum(local_model_para * state_update_diff) / (lr * n_SGD)
        
        batch_loss = batch_loss + loss_r + loss_g

        batch_loss.backward()
        optimizer.step()

def set_param_to_model(model, model_param):
    """ 根据权重进行FedAvg聚合 """

    new_model = deepcopy(model)
    for layer_weigths in new_model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)

    for idx, layer_weights in enumerate(new_model.parameters()):
        contribution = model_param.data
        layer_weights.data.add_(contribution)

    return new_model