import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from copy import deepcopy

import random
from datetime import datetime
import math
import os

from py_func.clustering import get_gradients, get_grad, get_similarity
import py_func.personal_func as pf

def FedAvg_agregation_process(model, clients_models_hist: list, weights_list: list):
    """ 根据权重进行FedAvg聚合 """

    new_model = deepcopy(model)
    for layer_weigths in new_model.parameters():
        layer_weigths.data.sub_(sum(weights_list) * layer_weigths.data)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution = client_hist[idx].data * weights_list[k]
            layer_weights.data.add_(contribution)

    return new_model


def accuracy_dataset(model, dataset):
    """ 计算模型model在dataset上的准确度 """

    correct = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 移动模型到cuda
    for features, labels in dataset:
        features = features.to(device)  # 移动数据到cuda
        labels = labels.to(device)

        predictions = model(features)
        _, predicted = predictions.max(1, keepdim=True)

        correct += torch.sum(predicted.view(-1, 1) ==
                             labels.view(-1, 1)).item()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy


def loss_dataset(model, train_data, loss_f):
    """ 计算model在train_data上的loss，loss_f为损失函数 """
    loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 移动模型到cuda
    for idx, (features, labels) in enumerate(train_data):
        features = features.to(device)  # 移动数据到cuda
        labels = labels.to(device)

        predictions = model(features)
        loss += loss_f(predictions, labels)

    loss /= idx + 1
    return loss


def loss_classifier(predictions, labels):
    """ 损失函数 """

    criterion = nn.CrossEntropyLoss() # 交叉熵函数
    return criterion(predictions, labels)


def n_params(model):
    """ 返回模型参数量 """

    n_params = sum(
        [
            np.prod([tensor.size()[k] for k in range(len(tensor.size()))])
            for tensor in list(model.parameters())
        ]
    )

    return n_params


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters"""

    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum(
        [
            torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
            for i in range(len(tensor_1))
        ]
    )

    return norm


def grad_similarity(grad_0, model_0, model):
    """ 基于梯度方向余弦相似度的正则项 """

    tensor_0 = list(grad_0.parameters())
    tensor_1 = list(model_0.parameters())
    tensor_2 = list(model.parameters())

    norm, norm_1, norm_2 = 0, 0, 0
    for i in range(len(tensor_2)):
        norm += torch.sum(tensor_0[i] * (tensor_2[i]-tensor_1[i]))
        norm_1 += torch.sum(tensor_0[i] ** 2)
        norm_2 += torch.sum((tensor_2[i]-tensor_1[i]) ** 2)

    if norm_1 == 0 or norm_2 == 0:
        norm.zero_()
    else:
        norm /= torch.sqrt(norm_1 * norm_2)

    return 1-norm


def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_f):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)  # 移动模型到cuda
    model_0 = deepcopy(model)

    for _ in range(n_SGD):

        features, labels = next(iter(train_data))

        features = features.to(device)  # 移动数据到cuda
        labels = labels.to(device)
        # 或者  labels = labels.cuda() if torch.cuda.is_available() else labels

        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_f(predictions, labels)
        batch_loss += mu / 2 * difference_models_norm_2(model, model_0) # 惩罚项

        batch_loss.backward()
        optimizer.step()


def direction_local_learning(model, pre_grad, lamda: float, optimizer, train_data, n_SGD: int, loss_f):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)  # 移动模型到cuda
    model_0 = deepcopy(model)

    for _ in range(n_SGD):

        features, labels = next(iter(train_data))

        features = features.to(device)  # 移动数据到cuda
        labels = labels.to(device)
        # 或者  labels = labels.cuda() if torch.cuda.is_available() else labels

        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_f(predictions, labels)
    
        # 损失函数惩罚项
        # cur_grad = get_grad(model_0, model)
        loss_cosin = grad_similarity(pre_grad, model_0, model)
        
        # print(f"loss_cosin:{loss_cosin}")
        
        batch_loss += lamda * loss_cosin

        batch_loss.backward()
        optimizer.step()


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    path = f"experiments_res/{directory}" 
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


def FedALP(
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    pre_train=400,
    decay=1.0,
    metric_period=1,
    n_clusters=10,
    decay1=0.999,
    local_test=False,
    beta = 0.56,
    mu=0.0
):
    """基于分组与自适应分层的个性化联邦学习
    Parameters:
        - `model`: 模型
        - `training_sets`: 训练数据集列表,index与客户端index相对应
        - `testing_set`: 训练数据集列表,index与客户端index相对应
        - `n_iter`: 全局轮次
        - `n_SGD`: 本地轮次
        - `lr`: 初始学习率
        - `file_name`: 用于存储训练结果的文件名
        - `pre_train`: 预训练轮次
        - `decay`: 分组初始化阶段衰减系数，仅应用一轮
        - `metric_period`: 训练数据记录频次
        - `n_clusters`: 分组个数
        - `decay1`: 分组后学习衰减率
        - `local_test`: 本地模型测试方法
        - `beta`: 个性化系数
        - `mu`: FedProx 正则系数

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
    sim_type = "cosine"
    K = len(training_sets)  # clients总数
    
    n_samples = np.array([len(db.dataset) for db in training_sets])     # array(K)，每个client拥有的样本数量
    
    weights = n_samples / np.sum(n_samples)     # array(K)，每个client样本数量所占权重

    loss_hist = np.zeros((n_iter + 1, K))       # array(n+1,k),记录n轮、k个设备的loss(全局模型)
    acc_hist = np.zeros((n_iter + 1, K))        # array(n+1,k),记录n轮、k个设备的acc(全局模型)

    p_loss_hist = np.zeros((n_iter + 1, K))     # array(n+1,k),记录n轮、k个设备的loss(簇模型)    
    p_acc_hist = np.zeros((n_iter + 1, K))      # array(n+1,k),记录n轮、k个设备的acc(簇模型)

    server_loss_hist = []       # list(n,1),记录全局模型n轮的loss
    server_acc_hist = []        # list(n,1),记录全局模型n轮的acc

    p_server_loss_hist = []     # list(n,1),记录分层分组模型n轮的全局平均loss
    p_server_acc_hist = []      # list(n,1),记录分层分组模型n轮的全局平均acc
    
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

    # list(K) ,上一轮的梯度列表
    gradients = []

    starttime = datetime.now()  # 记录训练开始时时间

    ''' <<<<<<<<<<<<<<<<<<<<<<<<变量初始化------------------------ '''

    print("========>>> 初始化完成")
    

    ''' ------------------------完整训练>>>>>>>>>>>>>>>>>>>>>>>> '''
    # 全局轮次循环
    for i in range(n_iter):

        i_time = datetime.now()     # 记录当前轮次开始时间

        previous_global_model = deepcopy(model) # 上一轮的全局模型

        clients_params = []     # 当前轮次 所有客户端模型参数（占内存）
        clients_models = []
        sampled_clients_for_grad = []   # 存储梯度的客户端列表

        ''' ------------------------预热阶段>>>>>>>>>>>>>>>>>>>>>>>> '''
        # 分组前训练
        if i < pre_train:
            # print(f"========>>> 第{i+1}轮(未分组)")
            for k in range(K):
                local_model = deepcopy(model)
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )

                # 当前客户端最新模型参数
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]

                clients_models.append(deepcopy(local_model))  # 存入当前轮次客户端模型列表
                clients_params.append(list_params)  # 存入当前轮次客户端模型参数列表
                sampled_clients_for_grad.append(k)  # 参与训练的客户端下标存入列表

        ''' <<<<<<<<<<<<<<<<<<<<<<<<预热阶段------------------------ '''


        ''' ------------------------个性化训练阶段>>>>>>>>>>>>>>>>>>>>>>>> '''
        # 分组后训练
        if i >= pre_train:

            ''' ------------------------分组初始化阶段>>>>>>>>>>>>>>>>>>>>>>>> '''
            if i == pre_train :
                # 分组的那一轮进行初始化
                lr = lr * decay
                decay = decay1

                # 可选：保存预训练模型
                torch.save(
                    model.state_dict(), f"saved_exp_info/final_model/pre_{pre_train}.pth"
                )

                print("========>>> 分组信息初始化")
                
                ''' //////////////////////分组一次,不然簇id会乱///////////////////////////'''
                from scipy.cluster.hierarchy import linkage, fcluster
                from py_func.clustering import get_matrix_similarity_from_grads

                # 根据梯度计算相似度矩阵
                sim_matrix = get_matrix_similarity_from_grads(
                    gradients, distance_type=sim_type
                )

                # 计算层次聚类linkage
                linkage_matrix = linkage(sim_matrix, "ward")

                # 按最大簇个数聚类，(n,1)，元素为聚类簇 id，id从1开始？
                clusters = fcluster(
                    linkage_matrix, n_clusters, criterion="maxclust")

                # 簇_client分布，array(n_clusters,K)，元素为客户权重
                distri_clusters = pf.get_clusters(
                    clusters, n_clusters, weights)

                # 初始化簇weight:(n_clusters,1)，元素为组内权重之和。 组模型列表:(n_clusters,1)，元素为组模型
                clusters_weight, clusters_model = pf.init_clusters_model(
                    distri_clusters, model)

                # 设备在簇内所占权重
                clients_weight_in_cluster = pf.get_weight_in_cluster(
                    distri_clusters)

                '''//////////////////////////////以上区间只能执行一次!!!/////////////////////////////////'''

                # 用于存储簇模型各层个性化权重：list(n_clusters,1) 元素为每组的层个性化权重向量
                clusters_layer_weights = pf.get_clusters_layer_weights(
                    pf.get_clusters_avg_grad(distri_clusters, np.array(gradients, dtype = object)), beta
                )

            ''' <<<<<<<<<<<<<<<<<<<<<<<<分组初始化阶段------------------------ '''

            if local_test:
                # 3.回发给各簇，加权融合 //已修正，该过程应在测试过acc、loss后进行，在每轮训练前开始
                clusters_model = pf.cluster_aggregation_process(
                    model, clusters_model, clusters_layer_weights)
            
            for k in range(K):
                local_model = deepcopy(
                    clusters_model[(clusters[k] - 1)])       # 本地模型是所属簇的模型
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
                # local_optimizer = optim.Adam(local_model.parameters(), lr=lr)

                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )

                # 当前客户端最新模型参数
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]
    
                clients_models.append(deepcopy(local_model))  # 存入当前轮次客户端模型列表
                clients_params.append(list_params)  # 存入当前轮次客户端模型参数列表
                sampled_clients_for_grad.append(k)  # 参与训练的客户端下标存入列表
        ''' <<<<<<<<<<<<<<<<<<<<<<<<个性化训练阶段------------------------ '''


        ''' ------------------------分组前聚合>>>>>>>>>>>>>>>>>>>>>>>> '''
        if i < pre_train:
            # print("========>>> 聚合(未分组)")
            
            # 更新全局模型
            model = FedAvg_agregation_process(
                deepcopy(model), clients_params, weights
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
            if i % metric_period == 0:
                t = str(nowtime - starttime).split(".")[0]
                print(
                    f"========>>> 第{i+1}轮(未分组):   Loss: {server_loss}    Server Test Accuracy: {server_acc}   —>Time: {t}"
                )
            else:
                t = str(nowtime - i_time).split(".")[0]
                print(
                    f"========>>> 第{i+1}轮(未分组):   Done  IterTime: {t}"
                )
        ''' <<<<<<<<<<<<<<<<<<<<<<<<分组前聚合------------------------ '''
        
        ''' ------------------------分组后聚合>>>>>>>>>>>>>>>>>>>>>>>> '''
        if i >= pre_train:
            # print("========>>> 聚合(分组)")

            ''' //////////////////////新的聚合方案////////////////////// '''
            # 1.每个簇内模型FedAvg聚合
            for idx in range(len(clusters_model)):
                # 组idx内设备下标 数组
                index_i = [x for x in np.where(distri_clusters[idx] != 0)[0]]

                # 组内模型按权重聚合
                clusters_model[idx] = FedAvg_agregation_process(
                    clusters_model[idx],
                    [clients_params[x] for x in index_i],
                    [clients_weight_in_cluster[x] for x in index_i]
                )

            # 2.全局聚合
            model = FedAvg_agregation_process(
                model, pf.model_list_to_params(clusters_model), clusters_weight
            )

            # 3.回发给各簇，加权融合 //已修正，该过程应在测试过acc、loss后进行，在每轮训练前开始
            if not local_test:
                clusters_model = pf.cluster_aggregation_process(
                    model, clusters_model, clusters_layer_weights)

            # 4.测试Acc Loss
            if i % metric_period == 0:

                for k, dl in enumerate(training_sets):
                    # 每个设备对新global模型的loss
                    loss_hist[i + 1, k] = float(
                        loss_dataset(model, dl, loss_f).detach()
                    )
                    # 每个设备对新cluster模型的loss
                    p_loss_hist[i + 1, k] = float(
                        loss_dataset(
                            clusters_model[(clusters[k] - 1)], dl, loss_f).detach()
                    )

                for k, dl in enumerate(testing_sets):
                    # 每个设备对新global模型的loss
                    acc_hist[i + 1, k] = accuracy_dataset(model, dl)
                    # 每个设备对新cluster模型的acc
                    p_acc_hist[i + 1, k] = accuracy_dataset(
                        clusters_model[(clusters[k] - 1)], dl)

                server_loss = np.dot(weights, loss_hist[i + 1])     # 当前轮，全局模型平均 loss
                server_acc = np.dot(weights, acc_hist[i + 1])       # 当前轮，全局模型平均 acc

                p_server_loss = np.dot(weights, p_loss_hist[i + 1]) # 当前轮，组模型 loss 全局平均
                p_server_acc = np.dot(weights, p_acc_hist[i + 1])   # 当前轮，组模型 acc 全局平均
                

                server_loss_hist.append(server_loss)    # 所有轮，全局模型平均 loss
                server_acc_hist.append(server_acc)      # 所有轮，全局模型平均 acc

                p_server_loss_hist.append(p_server_loss)    # 所有轮，组模型 loss 全局平均
                p_server_acc_hist.append(p_server_acc)      # 所有轮，组模型 acc 全局平均
                
            nowtime = datetime.now()
            if i % metric_period == 0:
                t = str(nowtime-starttime).split(".")[0]
                print(
                    f"========>>> 第{i+1}轮(分组):   Loss: {server_loss}    Server Test Accuracy: {server_acc}"
                )
                print(
                    f"========>>> 第{i+1}轮(分组): p_Loss: {p_server_loss}  p_Server Test Accuracy: {p_server_acc}   —>Time: {t}"
                )
            else:
                t = str(nowtime-i_time).split(".")[0]
                print(
                    f"========>>> 第{i+1}轮(分组):   Done  IterTime: {t}"
                )
        ''' <<<<<<<<<<<<<<<<<<<<<<<<分组后聚合------------------------ '''

        # 更新Tpre前的梯度gradients
        if i == pre_train-1:
            gradients = get_gradients(
                previous_global_model, clients_models
            )

        # 学习率衰减
        if(i >= pre_train):
            lr *= decay

        ''' ------------------------每轮覆盖存储loss/acc>>>>>>>>>>>>>>>>>>>>>>>> '''
        # n轮K个客户端的loss与acc,无需存储
        # save_pkl(loss_hist, "loss", "g_"+file_name)
        # save_pkl(acc_hist, "acc", "g_"+file_name)
        # save_pkl(p_loss_hist, "loss", "p_"+file_name)
        # save_pkl(p_acc_hist, "acc", "p_"+file_name)
        
        # 全局平均loss/acc
        save_pkl(server_loss_hist, "server_loss", "g_"+file_name)
        save_pkl(server_acc_hist, "server_acc", "g_"+file_name)
        save_pkl(p_server_loss_hist, "server_loss", "p_"+file_name)
        save_pkl(p_server_acc_hist, "server_acc", "p_"+file_name)
        ''' <<<<<<<<<<<<<<<<<<<<<<<<每轮覆盖存储loss/acc------------------------ '''

    # 训练结束时存储实验数据
    # save_pkl(loss_hist, "loss", "g_" + file_name)
    # save_pkl(acc_hist, "acc", "g_" + file_name)
    # save_pkl(p_loss_hist, "loss", "p_" + file_name)
    # save_pkl(p_acc_hist, "acc", "p_" + file_name)
    
    save_pkl(server_loss_hist, "server_loss", "g_" + file_name)
    save_pkl(server_acc_hist, "server_acc", "g_" + file_name)
    save_pkl(p_server_loss_hist, "server_loss", "p_" + file_name)
    save_pkl(p_server_acc_hist, "server_acc", "p_" + file_name)

    # save_pkl(linkage_hist, "linkage", file_name)

    # 存储最终模型
    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def pFedLDGD(
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    shannon_list:list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay=1.0,
    metric_period=1,
    mu=0.0,
    lamda_d=0.0,
    lamda_n=0.0,
    pattern=0,
    decay_norm=1.0
):
    """基于标签多样性与梯度方向的个性化联邦学习
    Parameters:
        - `model`: 模型
        - `training_sets`: 训练数据集列表,index与客户端index相对应
        - `testing_set`: 训练数据集列表,index与客户端index相对应
        - `shannon_list`: 香农多样性指数列表
        - `n_iter`: 全局轮次
        - `n_SGD`: 本地轮次
        - `lr`: 初始学习率
        - `file_name`: 用于存储训练结果的文件名
        - `decay`: 分组初始化阶段衰减系数，仅应用一轮
        - `metric_period`: 训练数据记录频次
        - `mu`: fedProx 正则系数
        - `lamda_d`: 多样性权重 正则系数
        - `lamda_n`: 本地目标惩罚项 正则系数
        - `pattern`: 算法模式切换,`0`:FedAvg/FedProx,`1`:基于标签多样性,`2`:基于更新方向,`3`:两算法结合
        - `decay_norm`: 本地目标惩罚项 衰减系数

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

    # # list(K) ,上一轮的梯度列表
    # gradients = get_gradients(model, [model] * K)
    
    # 上一轮的总梯度
    grad = get_grad(model, model)
    ###################################################################### 
    shannon_list = np.array(shannon_list)


    ''' DONE:初始化基于多样性的分组，并确定更新间隔 '''
    diversity_avg = np.mean(shannon_list)   # 多样性均值
    
    # 多样性高于均值的客户下标
    hd_clients = np.where(shannon_list >= diversity_avg)[0]
    # 多样性低于均值的客户下标
    ld_clients = np.where(shannon_list < diversity_avg)[0]

    # 计算高多样性客户组的权重
    hd_weights = np.array([weights[idx] for idx in hd_clients])
    hd_weights = hd_weights/np.sum(hd_weights)

    # 计算更新间隔 \tao
    hd_avg = np.mean([shannon_list[idx] for idx in hd_clients])
    ld_avg = np.mean([shannon_list[idx] for idx in ld_clients])
    tao = int(math.sqrt(hd_avg//ld_avg)) + 1


    # TODO:tao改为固定值
    if pattern == 1 and lamda_d == 2:
        tao = 2
        lamda_d = 0

    # TODO:将香农指数以权重形式影响聚合
    shannon_weights = lamda_d * shannon_list / np.sum(shannon_list) + (1-lamda_d) * weights

    if pattern == 1:
        print("hd:")
        print(hd_clients)
        print("ld:")
        print(ld_clients)
        print("hd_weights:")
        print(hd_weights)
        print(f"hd_avg={hd_avg},ld_avg={ld_avg},tao={tao}")
    ######################################################################

    starttime = datetime.now()  # 记录训练开始时时间

    ''' <<<<<<<<<<<<<<<<<<<<<<<<变量初始化------------------------ '''

    print("========>>> 初始化完成")
    

    ''' ------------------------完整训练>>>>>>>>>>>>>>>>>>>>>>>> '''
    # 全局轮次循环
    for i in range(n_iter):

        # TODO:将香农指数以权重形式影响聚合 ###################################################################
        shannon_weights = lamda_d * shannon_list / np.sum(shannon_list) + (1-lamda_d) * weights
        
        i_time = datetime.now()     # 记录当前轮次开始时间

        previous_global_model = deepcopy(model)     # 上一轮的全局模型

        clients_params = []     # 当前轮次 所有客户端模型参数（占内存）
        sampled_clients_for_grad = []   # 存储梯度的客户端列表

        ######################################################################
        # 根据轮次选择参与训练的客户 依据 聚合时的权重
        clients_list = np.arange(K)
        agre_weights = weights

        # TODO:将香农指数以权重形式影响聚合
        if pattern == -1 or pattern == 3:
            agre_weights = shannon_weights

        # 在使用基于多样性方法 且 不是全局聚合轮次的时候，仅高多样性组聚合
        if pattern == 1 and i % tao != 0:
            clients_list = hd_clients
            agre_weights = hd_weights
        ######################################################################
        
        ''' ------------------------训练阶段>>>>>>>>>>>>>>>>>>>>>>>> '''
        # print(f"========>>> 第{i+1}轮(未分组)")
        for k in clients_list:
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            ''' DONE: 修改本地优化函数，需考虑本地模型与全局模型的更新方向 '''
            if pattern < 2:
                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )

            else:
                direction_local_learning(
                    local_model,
                    grad,
                    lamda_n,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
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
        # print("========>>> 聚合(未分组)")
        ######################################################################
        ''' DONE: 修改全局聚合方法，问题：不参与聚合的客户端是否保留本地最新模型？ 解决：低多样性组仅在需要参与的时候进行本地训练 '''
        # 更新全局模型
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, agre_weights
        )
        ######################################################################

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
        if i % metric_period == 0:
            t = str(nowtime - starttime).split(".")[0]
            print(
                f"========>>> 第{i+1}轮:   Loss: {server_loss}    Server Test Accuracy: {server_acc}   —>Time: {t}"
            )
        else:
            t = str(nowtime - i_time).split(".")[0]
            print(
                f"========>>> 第{i+1}轮:   Done  IterTime: {t}"
            )
        ''' <<<<<<<<<<<<<<<<<<<<<<<<聚合------------------------ '''

        # 更新总梯度
        grad = get_grad(previous_global_model, model)

        # 学习率衰减
        lr *= decay
        lamda_n *= decay_norm
        #########################################################################
        if pattern == 3:
            lamda_d *= mu

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




def pFedGLAO(
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    shannon_list:list,
    n_iter: int,
    pre_train: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay=1.0,
    metric_period=1,
    mu=0.0,
    lamda_d=0.0,
    lamda_n=0.0,
    decay_norm=1.0,
    decayD=1.0,
    n_clusters=10,
    decayP=1.0,
    local_test=False,
    beta=0.6,
    p_grad=[],
    pattern=0
):
    """全局与局部动态优化的个性化联邦学习
    Parameters:
        - `model`: 模型
        - `training_sets`: 训练数据集列表,index与客户端index相对应
        - `testing_set`: 训练数据集列表,index与客户端index相对应
        - `shannon_list`: 香农多样性指数列表
        - `n_iter`: 全局轮次
        - `pre_train`: 预训练轮次   #
        - `n_SGD`: 本地轮次
        - `lr`: 初始学习率
        - `file_name`: 用于存储训练结果的文件名
        - `decay`: 分组初始化阶段衰减系数，仅应用一轮
        - `metric_period`: 训练数据记录频次
        - `mu`: fedProx 正则系数
        - `lamda_d`: 多样性权重 正则系数
        - `lamda_n`: 本地目标惩罚项 正则系数
        - `decay_norm`: 本地目标惩罚项 衰减系数
        - `n_clusters`: 分组个数    #
        - `decay1`: 分组后学习衰减率    #
        - `local_test`: 本地模型测试方法    #
        - `beta`: 个性化系数    #
        - `p_grad`: 上一轮梯度
        - `pattern`: 模式切换 0：完整训练 1：仅分组后

    returns :
        - `model`: 最终的全局模型
    """


    # TODO:保存分组前最后的模型和梯度

    print("========>>> 正在初始化训练")

    # GPU选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 移动模型到cuda
    print(f"模型放入{device}")

    # 损失函数
    loss_f = loss_classifier

    ''' ------------------------变量初始化>>>>>>>>>>>>>>>>>>>>>>>> '''
    sim_type = "cosine"
    K = len(training_sets)  # clients总数
    
    n_samples = np.array([len(db.dataset) for db in training_sets])     # array(K)，每个client拥有的样本数量
    
    weights = n_samples / np.sum(n_samples)     # array(K)，每个client样本数量所占权重

    loss_hist = np.zeros((n_iter + 1, K))       # array(n+1,k),记录n轮、k个设备的loss(全局模型)
    acc_hist = np.zeros((n_iter + 1, K))        # array(n+1,k),记录n轮、k个设备的acc(全局模型)

    p_loss_hist = np.zeros((n_iter + 1, K))     # array(n+1,k),记录n轮、k个设备的loss(簇模型)    
    p_acc_hist = np.zeros((n_iter + 1, K))      # array(n+1,k),记录n轮、k个设备的acc(簇模型)

    server_loss_hist = []       # list(n,1),记录全局模型n轮的loss
    server_acc_hist = []        # list(n,1),记录全局模型n轮的acc

    p_server_loss_hist = []     # list(n,1),记录分层分组模型n轮的全局平均loss
    p_server_acc_hist = []      # list(n,1),记录分层分组模型n轮的全局平均acc
    
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

    # list(K) ,上一轮的梯度列表
    gradients = []
    
    # 上一轮的总梯度
    grad = get_grad(model, model)
    
    # 每组的总梯度
    grads = []
    
    shannon_list = np.array(shannon_list)

    starttime = datetime.now()  # 记录训练开始时时间

    ''' <<<<<<<<<<<<<<<<<<<<<<<<变量初始化------------------------ '''

    print("========>>> 初始化完成")
    
    # 仅分组后训练时，预置grad
    if pattern == 1:
        gradients = p_grad
        print(f"跳过前{pre_train}轮训练...")

    ''' ------------------------完整训练>>>>>>>>>>>>>>>>>>>>>>>> '''
    # 全局轮次循环
    for i in range(n_iter):
        # 跳过前面轮次
        if pattern == 1 and i < pre_train :
            # print(f"\r ===> {i}/{n_iter} skip", end="")
            continue

        i_time = datetime.now()     # 记录当前轮次开始时间

        previous_global_model = deepcopy(model)     # 上一轮的全局模型

        clients_params = []     # 当前轮次 所有客户端模型参数（占内存）
        clients_models = []
        sampled_clients_for_grad = []   # 存储梯度的客户端列表

        ''' ------------------------预热阶段>>>>>>>>>>>>>>>>>>>>>>>> '''
        # 分组前训练
        if i < pre_train:

            # 将香农指数以权重形式影响聚合
            shannon_weights = lamda_d * shannon_list / np.sum(shannon_list) + (1-lamda_d) * weights
            agre_weights = shannon_weights

            for k in range(K):
                local_model = deepcopy(model)
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

                ''' DONE: 修改本地优化函数，考虑本地模型与全局模型的更新方向 '''
                direction_local_learning(
                    local_model,
                    grad,
                    lamda_n,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )

                # 当前客户端最新模型参数
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]

                clients_models.append(deepcopy(local_model))  # 存入当前轮次客户端模型列表
                clients_params.append(list_params)  # 存入当前轮次客户端模型参数列表
                sampled_clients_for_grad.append(k)  # 参与训练的客户端下标存入列表

        ''' <<<<<<<<<<<<<<<<<<<<<<<<预热阶段------------------------ '''


        ''' ------------------------个性化训练阶段>>>>>>>>>>>>>>>>>>>>>>>> '''
        # 分组后训练
        if i >= pre_train:

            ''' ------------------------分组初始化阶段>>>>>>>>>>>>>>>>>>>>>>>> '''
            if i == pre_train :
                # 分组的那一轮进行初始化
                lr = lr * decay
                decay = decayP

                if pattern == 0:
                    # 保存预训练模型
                    torch.save(
                        model.state_dict(), f"experiments_res/final_model/{file_name}_pre{pre_train}.pth"
                    )
                    # TODO:保存梯度信息
                    save_pkl(gradients, "gradPre", file_name + "_grad")
                
                ''' //////////////////////分组一次,不然簇id会乱///////////////////////////'''
                from scipy.cluster.hierarchy import linkage, fcluster
                from py_func.clustering import get_matrix_similarity_from_grads

                # 根据梯度计算相似度矩阵
                sim_matrix = get_matrix_similarity_from_grads(
                    gradients, distance_type=sim_type
                )

                # 计算层次聚类linkage
                linkage_matrix = linkage(sim_matrix, "ward")

                # 按最大簇个数聚类，(n,1)，元素为聚类簇 id，id从1开始？
                clusters = fcluster(
                    linkage_matrix, n_clusters, criterion="maxclust")

                # 簇_client分布，array(n_clusters,K)，元素为客户权重
                distri_clusters = pf.get_clusters(
                    clusters, n_clusters, weights)

                # 初始化簇weight:(n_clusters,1)，元素为组内权重之和。 组模型列表:(n_clusters,1)，元素为组模型
                clusters_weight, clusters_model = pf.init_clusters_model(
                    distri_clusters, model)

                # 设备在簇内所占权重
                clients_weight_in_cluster = pf.get_weight_in_cluster(
                    distri_clusters)

                '''//////////////////////////////以上区间只能执行一次!!!/////////////////////////////////'''

                # 用于存储簇模型各层个性化权重：list(n_clusters,1) 元素为每组的层个性化权重向量
                clusters_layer_weights = pf.get_clusters_layer_weights(
                    pf.get_clusters_avg_grad(distri_clusters, np.array(gradients, dtype = object)), beta
                )

                # 初始化组梯度
                for idx in range(n_clusters):
                    grads.append(deepcopy(grad))
            ''' <<<<<<<<<<<<<<<<<<<<<<<<分组初始化阶段------------------------ '''

            if local_test:
                # 3.回发给各簇，加权融合 //已修正，该过程应在测试过acc、loss后进行，在每轮训练前开始
                clusters_model = pf.cluster_aggregation_process(
                    model, clusters_model, clusters_layer_weights)
            
            for k in range(K):
                local_model = deepcopy(
                    clusters_model[(clusters[k] - 1)])       # 本地模型是所属簇的模型
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )
                # direction_local_learning(
                #     local_model,
                #     grads[(clusters[k] - 1)],
                #     lamda_n,
                #     local_optimizer,
                #     training_sets[k],
                #     n_SGD,
                #     loss_f,
                # )

                # 当前客户端最新模型参数
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]

                clients_models.append(deepcopy(local_model))  # 存入当前轮次客户端模型列表
                clients_params.append(list_params)  # 存入当前轮次客户端模型参数列表
                sampled_clients_for_grad.append(k)  # 参与训练的客户端下标存入列表
        ''' <<<<<<<<<<<<<<<<<<<<<<<<个性化训练阶段------------------------ '''


        ''' ------------------------分组前聚合>>>>>>>>>>>>>>>>>>>>>>>> '''
        if i < pre_train:
            
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
                    f"\r====>>> {i+1}/{n_iter}:  Iter - {t0}   Loss: {server_loss:.3f}   Acc: {server_acc:.3f}   —>Time - {t}", end = ""
                )
            else:
                print(
                    f"\r====>>> {i+1}/{n_iter}:  Iter - {t0}", end = ""
                )
        ''' <<<<<<<<<<<<<<<<<<<<<<<<分组前聚合------------------------ '''
        
        ''' ------------------------分组后聚合>>>>>>>>>>>>>>>>>>>>>>>> '''
        if i >= pre_train:

            ''' //////////////////////新的聚合方案////////////////////// '''
            # 1.每个簇内模型FedAvg聚合
            for idx in range(len(clusters_model)):
                # 组idx内设备下标 数组
                index_i = [x for x in np.where(distri_clusters[idx] != 0)[0]]

                # 组内模型按权重聚合
                
                tmodel = FedAvg_agregation_process(
                    clusters_model[idx],
                    [clients_params[x] for x in index_i],
                    [clients_weight_in_cluster[x] for x in index_i]
                )
                grads[idx] = get_grad(clusters_model[idx], tmodel)
                clusters_model[idx] = tmodel

            # 2.全局聚合
            model = FedAvg_agregation_process(
                model, pf.model_list_to_params(clusters_model), clusters_weight
            )

            # 3.回发给各簇，加权融合 //已修正，该过程应在测试过acc、loss后进行，在每轮训练前开始
            if not local_test:
                clusters_model = pf.cluster_aggregation_process(
                    model, clusters_model, clusters_layer_weights)

            # 4.测试Acc Loss
            if i % metric_period == 0:

                for k, dl in enumerate(training_sets):
                    # 每个设备对新global模型的loss
                    loss_hist[i + 1, k] = float(
                        loss_dataset(model, dl, loss_f).detach()
                    )
                    # 每个设备对新cluster模型的loss
                    p_loss_hist[i + 1, k] = float(
                        loss_dataset(
                            clusters_model[(clusters[k] - 1)], dl, loss_f).detach()
                    )

                for k, dl in enumerate(testing_sets):
                    # 每个设备对新global模型的loss
                    acc_hist[i + 1, k] = accuracy_dataset(model, dl)
                    # 每个设备对新cluster模型的acc
                    p_acc_hist[i + 1, k] = accuracy_dataset(
                        clusters_model[(clusters[k] - 1)], dl)

                server_loss = np.dot(weights, loss_hist[i + 1])     # 当前轮，全局模型平均 loss
                server_acc = np.dot(weights, acc_hist[i + 1])       # 当前轮，全局模型平均 acc

                p_server_loss = np.dot(weights, p_loss_hist[i + 1]) # 当前轮，组模型 loss 全局平均
                p_server_acc = np.dot(weights, p_acc_hist[i + 1])   # 当前轮，组模型 acc 全局平均
                

                server_loss_hist.append(server_loss)    # 所有轮，全局模型平均 loss
                server_acc_hist.append(server_acc)      # 所有轮，全局模型平均 acc

                p_server_loss_hist.append(p_server_loss)    # 所有轮，组模型 loss 全局平均
                p_server_acc_hist.append(p_server_acc)      # 所有轮，组模型 acc 全局平均
                
            nowtime = datetime.now()
            t0 = str(nowtime - i_time)[2:7]
            if i % metric_period == 0:
                t = str(nowtime-starttime).split(".")[0]
                print(
                    f"\r====>>> {i+1}/{n_iter}:  Iter - {t0}  Loss: {server_loss:.3f}  Acc: {server_acc:.3f}  p_Loss: {p_server_loss:.3f}  p_Acc: {p_server_acc:.3f}  —>Time - {t}", end = ""
                )
            else:
                print(
                    f"\r====>>> {i+1}/{n_iter}:  Iter - {t0}", end = ""
                )
        ''' <<<<<<<<<<<<<<<<<<<<<<<<分组后聚合------------------------ '''

        # 更新Tpre前的梯度gradients
        if i == pre_train-1:
            gradients = get_gradients(
                previous_global_model, clients_models
            )
        # 更新总梯度
        grad = get_grad(previous_global_model, model)

        # 学习率衰减
        if i >= pre_train:
            lr *= decay
        lamda_n *= decay_norm
        # TODO
        lamda_d *= decayD

        ''' ------------------------每轮覆盖存储loss/acc>>>>>>>>>>>>>>>>>>>>>>>> '''
        # n轮K个客户端的loss与acc,无需存储
        # save_pkl(loss_hist, "loss", "g_"+file_name)
        # save_pkl(acc_hist, "acc", "g_"+file_name)
        # save_pkl(p_loss_hist, "loss", "p_"+file_name)
        # save_pkl(p_acc_hist, "acc", "p_"+file_name)
        
        # 全局平均loss/acc
        save_pkl(server_loss_hist, "server_loss", "gloss_"+file_name)
        save_pkl(server_acc_hist, "server_acc", "gacc_"+file_name)
        save_pkl(p_server_loss_hist, "server_loss", "ploss_"+file_name)
        save_pkl(p_server_acc_hist, "server_acc", "pacc_"+file_name)
        ''' <<<<<<<<<<<<<<<<<<<<<<<<每轮覆盖存储loss/acc------------------------ '''
    ''' <<<<<<<<<<<<<<<<<<<<<<<<完整训练------------------------ '''

    # 训练结束时存储实验数据
    save_pkl(server_loss_hist, "server_loss", "gloss_"+file_name)
    save_pkl(server_acc_hist, "server_acc", "gacc_"+file_name)
    save_pkl(p_server_loss_hist, "server_loss", "ploss_"+file_name)
    save_pkl(p_server_acc_hist, "server_acc", "pacc_"+file_name)

    # 存储最终模型
    torch.save(
        model.state_dict(), f"experiments_res/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist