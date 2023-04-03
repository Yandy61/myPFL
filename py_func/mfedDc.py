from .FedProx import *
import copy

# GPU选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging

# # 创建一个logger对象，并设置日志记录级别为DEBUG
# logger = logging.getLogger('my_logger')
# logger.setLevel(logging.DEBUG)

# # 创建一个文件处理器，用于将日志写入指定文件中
# file_handler = logging.FileHandler('my_log_file.txt')

# # 设置文件处理器的日志记录级别为DEBUG
# file_handler.setLevel(logging.DEBUG)

# # 创建一个格式化器，用于定义日志的格式
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # 将格式化器添加到文件处理器中
# file_handler.setFormatter(formatter)

# # 将文件处理器添加到logger对象中
# logger.addHandler(file_handler)

# # 将print语句的输出记录到日志文件中
# logger.debug('This message will be logged to the file')

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
    alpha_coef=0.0,
    mu = 0.5,
    dmu = 0.998,
    da = 0.998
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

    fix:
        1.去掉weight_list

    """

    print("========>>> 正在初始化训练")

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
    
    n_par = len(get_mdl_params([model])[0])

    # # list(K) ,上一轮的梯度列表, 参数类型
    # gradients = get_gradients(model, [model] * K)
    
    # # h_i
    # parameter_drifts = copy.deepcopy(gradients)
    # 创建一个形状为(n_clnt, n_par)的数组，用于存储每个客户端的模型参数漂移值
    clients_parameter_drifts_inAll = np.zeros((K, n_par)).astype('float32')
    
    # 获取初始模型的参数列表
    init_par_list=get_mdl_params([model], n_par)[0]

    # 创建一个形状为(n_clnt+1, n_par)的数组，用于存储每个客户端和云端的状态梯度差异
    last_gadient = np.zeros((K+1, n_par)).astype('float32') #including cloud state
    # state_gadient_diffs = get_gradients(model, [model] * (K+1))
    
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

        # global_model_param = get_model_para(model)
        global_model_param = get_mdl_params([model], n_par)[0]
        
        delta_g_sum = np.zeros(n_par)

        clients_params = []     # 当前轮次 所有客户端模型参数（占内存）
        clients_models = []     # 当前轮次 所有客户端模型

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

            '''********************** FedDC ******************'''
            local_update_last = last_gadient[k]
            global_update_last = last_gadient[-1]
            alpha = alpha_coef
            hist_i = torch.tensor(clients_parameter_drifts_inAll[k], dtype=torch.float32, device=device) #h_i
            '''***********************************************'''

            ''' DONE: 修改本地优化函数，需考虑本地模型与全局模型的更新方向 '''
            fedDC_local_learning(
                local_model,
                alpha,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
                mu,
                global_update_last,
                local_update_last,
                hist_i,
                n_par,
                i,
                k
            )

            # 当前客户端最新模型参数
            list_params = list(local_model.parameters())
            list_params = [
                tens_param.detach() for tens_param in list_params
            ]
            
            '''***********************************************'''
            cur_client_params = get_mdl_params([local_model], n_par)[0]
            delta_param_curr = cur_client_params - global_model_param
            clients_parameter_drifts_inAll[k] += delta_param_curr
            last_gadient[k] = delta_param_curr
            delta_g_sum += delta_param_curr

            '''***********************************************'''

            clients_params.append(list_params)  # 存入当前轮次客户端模型参数列表
            # clients_models.append(deepcopy(local_model))    # 存入当前轮次客户端模型列表

        ''' <<<<<<<<<<<<<<<<<<<<<<<<训练阶段------------------------ '''

        ''' ------------------------聚合>>>>>>>>>>>>>>>>>>>>>>>> '''
        avgmodel = FedAvg_agregation_process(
            deepcopy(model), clients_params, agre_weights
        )

        '''***********************************************'''
        avg_mdl_param_sel = get_mdl_params([avgmodel],n_par)[0]

        delta_g_cur = 1 / K * delta_g_sum

        last_gadient[-1] = delta_g_cur 

        # 加上了hi的全局模型？
        cld_mdl_param = avg_mdl_param_sel + np.mean(clients_parameter_drifts_inAll, axis=0) #

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
        t0 = str(nowtime - i_time)[2:7]
        if i % metric_period == 0:
            t = str(nowtime - starttime).split(".")[0]
            print(
                f"\r====>>> {i+1}/{n_iter}:  Iter - {t0}      Loss: {dc_server_loss:.3f}   Acc: {dc_server_acc:.3f}   —>Time - {t}", end = ""
            )
        else:
            print(
                f"\r====>>> {i+1}/{n_iter}:  Iter - {t0}", end = ""
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
        mu *= dmu
        alpha_coef *= da
        
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

def get_mdl_params(model_list, n_par=None):
    
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

def fedDC_local_learning(
        model,
        alpha: float,
        optimizer,
        train_data,
        n_SGD: int,
        loss_f,
        mu,
        grad_global_pre,
        grad_local_pre,
        hist_i,
        n_par,
        iterindex,
        cindex
    ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)  # 移动模型到cuda
    model_0 = deepcopy(model)

    state_update_diff = torch.tensor(grad_local_pre - grad_global_pre,  dtype=torch.float32, device=device)

    global_model_param = torch.tensor(get_mdl_params([model_0],n_par)[0], dtype=torch.float32, device=device)


    for _ in range(n_SGD):

        features, labels = next(iter(train_data))

        features = features.to(device)  # 移动数据到cuda
        labels = labels.to(device)
        # 或者  labels = labels.cuda() if torch.cuda.is_available() else labels

        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_f(predictions, labels)

        local_parameter = None
        for param in model.parameters():
            if not isinstance(local_parameter, torch.Tensor):
                local_parameter = param.reshape(-1)
            else:
                local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)

        loss_r = alpha/2 * torch.sum((local_parameter - (global_model_param - hist_i))*(local_parameter - (global_model_param - hist_i)))
        loss_g = torch.sum(local_parameter * state_update_diff)*mu
        '''
        3.13 18:47  此处loss_g 与论文不同，按论文的方法造成无法收敛，现修改为源码中计算方法。
        '''

        batch_loss = batch_loss + loss_r + loss_g
        # batch_loss = batch_loss + loss_r

        batch_loss.backward()
        optimizer.step()

    # logger.debug(f'iter{iterindex}, client{cindex}, epoch{_}, batch_loss_{batch_loss:.3f}, loss_r_{loss_r:.3f}, loss_g_{loss_g:.3f}, Loss_{(batch_loss + loss_r + loss_g):.3f}')
    

def set_param_to_model(model, model_param):
    """ 根据权重进行FedAvg聚合 """
    dict_param = copy.deepcopy(dict(model.named_parameters()))
    idx = 0
    for name, param in model.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(model_param[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length
    
    model.load_state_dict(dict_param)    
    return model