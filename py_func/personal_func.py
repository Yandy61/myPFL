from copy import deepcopy

import numpy as np
from scipy.cluster.hierarchy import fcluster

''' 从某一轮开始聚类，当连续(10)轮聚类结果一致时，采用该聚类方案 '''


''' 生成簇-设备列表 '''
def get_clusters(clusters, n_clusters: int, weights: np.array):
    n_clients = len(clusters)   # client数量
    distri_clusters = np.zeros((n_clusters, n_clients))
    for index in range(n_clusters):
        for client_idx in np.where(clusters == (index + 1))[0]:
            distri_clusters[index, client_idx] = weights[client_idx]
    return distri_clusters


''' 初始化簇模型信息：权重、模型 '''
def init_clusters_model(distri_clusters, model):
    clusters_num = len(distri_clusters)
    clusters_weight = []
    clusters_model = []  # 簇model表
    for index in range(clusters_num):
        clusters_weight.append(sum(distri_clusters[index]))
        clusters_model.append(deepcopy(model))
    return clusters_weight, clusters_model


''' model_list转params_list '''
def model_list_to_params(clusters_model):
    clusters_params = []
    for cluster_model in clusters_model:
        clusters_params.append(get_model_params(cluster_model))
    return clusters_params


''' model转 params_list '''
def get_model_params(model):
    # t_model = deepcopy(model)
    list_params = list(model.parameters())
    list_params = [
        tens_param.detach() for tens_param in list_params
    ]
    return list_params


''' 生成设备的簇内权重 '''
def get_weight_in_cluster(distri_clusters):
    clients_weight_in_cluster = np.zeros(len(distri_clusters[0]))
    for cluster_i in distri_clusters:
        for idx in range(len(clients_weight_in_cluster)):
            clients_weight_in_cluster[idx] += cluster_i[idx] / sum(cluster_i)
    return clients_weight_in_cluster.tolist()


''' 获得簇平均梯度 '''
def get_clusters_avg_grad(distri_clusters, local_model_grads):
    clusters_grads = []  # 所有簇包含的clients的梯度
    for cluster in distri_clusters:
        clusters_grads.append([local_model_grads[idx]*cluster[idx] for idx in np.where(cluster != 0)[0]])
                                                    # 乘权重 后面除以簇总权重
    clusters_avg_grad = np.array([grads[0] for grads in clusters_grads], dtype=object)   # 所有簇的平均梯度
    for idx, cluster_grads in enumerate(clusters_grads):
        for client_grad in np.array(cluster_grads[1:], dtype=object):
            clusters_avg_grad[idx] += client_grad
        clusters_avg_grad[idx] /= sum(distri_clusters[idx])

    return clusters_avg_grad


''' 获得簇模型各层个性化权重 '''
def get_clusters_layer_weights(clusters_avg_grad, beta = 0.56):
    clusters_layer_weights = []
    for idx, cluster_grads in enumerate(clusters_avg_grad):
        layers_change = []
        for layer in cluster_grads:
            flat = layer.reshape((-1))  # 将某层梯度展平
            layers_change.append(pow(sum(v * v for v in flat), 0.5) / len(flat))  # 某层所有梯度参数平方和求平均
            # 欧几里得度量（euclidean metric）（也称欧氏距离）
        layers_change /= max(layers_change) / beta              # 1.25 是暂定的超参数，可能要修改
        clusters_layer_weights.append(layers_change)
    return clusters_layer_weights


''' 全局模型与簇模型按层加权融合 '''
def cluster_aggregation_process(model, clusters_models: list, clusters_layer_weights: list):
    model_params = get_model_params(model)      # 单个模型model转为params形式
    cluster_model_params = model_list_to_params(clusters_models)  # 所有簇模型转换为params_list

    # 对簇遍历
    for k, cluster_params in enumerate(cluster_model_params):
        new_model = deepcopy(model)                                     # 创建一个空模型
        for layer_weigths in new_model.parameters():
            layer_weigths.data.sub_(layer_weigths.data)

        for idx, layer_weights in enumerate(new_model.parameters()):
            contribution_cluster = cluster_params[idx].data * clusters_layer_weights[k][idx]
            contribution_global = model_params[idx].data * (1 - clusters_layer_weights[k][idx])
            layer_weights.data.add_(contribution_cluster)
            layer_weights.data.add_(contribution_global)

        clusters_models[k] = deepcopy(new_model)

    return clusters_models


''' (Contrast)只分组不分层-全局模型与簇模型整体-a加权融合 '''
def cluster_aggregation_process_without_layer_weights(model, clusters_models: list, local_weight):
    model_params = get_model_params(model)      # 单个模型model转为params形式
    cluster_model_params = model_list_to_params(clusters_models)  # 所有簇模型转换为params_list

    # 对簇遍历
    for k, cluster_params in enumerate(cluster_model_params):
        new_model = deepcopy(model)                                     # 创建一个空模型
        for layer_weigths in new_model.parameters():
            layer_weigths.data.sub_(layer_weigths.data)

        for idx, layer_weights in enumerate(new_model.parameters()):
            contribution_cluster = cluster_params[idx].data * local_weight
            contribution_global = model_params[idx].data * (1 - local_weight)
            layer_weights.data.add_(contribution_cluster)
            layer_weights.data.add_(contribution_global)

        clusters_models[k] = deepcopy(new_model)

    return clusters_models
