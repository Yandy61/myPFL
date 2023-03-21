from .FedProx import *
from scipy.cluster.hierarchy import linkage
from py_func.clustering import sample_clients, get_clusters_with_alg2, get_matrix_similarity_from_grads

def FedCS(
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    iter_FP=0,
    decay=1.0,
    metric_period=1,
    mu=0.0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    sampling = "clustered_2"
    sim_type = "cosine"

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

    # INITILIZATION OF THE GRADIENT HISTORY AS A LIST OF 0

    if sampling == "clustered_1":
        from py_func.clustering import get_clusters_with_alg1

        distri_clusters = get_clusters_with_alg1(n_sampled, weights)

    elif sampling == "clustered_2":
        from py_func.clustering import get_gradients

        gradients = get_gradients(model, [model] * K)

    starttime = datetime.now()  # 记录训练开始时时间

    ''' <<<<<<<<<<<<<<<<<<<<<<<<变量初始化------------------------ '''

    print("========>>> 初始化完成")
    

    ''' ------------------------完整训练>>>>>>>>>>>>>>>>>>>>>>>> '''
    # 全局轮次循环
    for i in range(n_iter):

        i_time = datetime.now()     # 记录当前轮次开始时间

        previous_global_model = deepcopy(model)

        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        if i < iter_FP:
            np.random.seed(i)
            sampled_clients = np.random.choice(
                K, size=n_sampled, replace=True, p=weights
            )

            for k in sampled_clients:
            
                client_index = '{:>{width}}'.format(k, width=2)
                nowtime = datetime.now()    # 记录当前时间
                t0 = str(nowtime - i_time)[2:7]
                print(
                    f"\r====>>> {i+1}/{n_iter}:  Iter - {t0} - {client_index}", end = "")
                
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

                # SAVE THE LOCAL MODEL TRAINED
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model))

                sampled_clients_for_grad.append(k)

        else:
            if sampling == "clustered_2":

                # GET THE CLIENTS' SIMILARITY MATRIX
                sim_matrix = get_matrix_similarity_from_grads(
                    gradients, distance_type=sim_type
                )

                # GET THE DENDROGRAM TREE ASSOCIATED
                linkage_matrix = linkage(sim_matrix, "ward")

                distri_clusters = get_clusters_with_alg2(
                    linkage_matrix, n_sampled, weights
                )

            for k in sample_clients(distri_clusters):
            
                client_index = '{:>{width}}'.format(k, width=2)
                nowtime = datetime.now()    # 记录当前时间
                t0 = str(nowtime - i_time)[2:7]
                print(
                    f"\r====>>> {i+1}/{n_iter}:  Iter - {t0} - {client_index}", end = "")

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

                # SAVE THE LOCAL MODEL TRAINED
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model))

                sampled_clients_for_grad.append(k)

        # CREATE THE NEW GLOBAL MODEL AND SAVE IT
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, weights_list=[1 / n_sampled] * n_sampled
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

        # UPDATE THE HISTORY OF LATEST GRADIENT
        if sampling == "clustered_2":
            gradients_i = get_gradients(
                previous_global_model, clients_models
            )
            for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
                gradients[idx] = gradient

        lr *= decay
        
        # 全局平均loss/acc
        save_pkl(server_loss_hist, "server_loss", file_name)
        save_pkl(server_acc_hist, "server_acc", file_name)


    # 训练结束时存储实验数据
    save_pkl(server_loss_hist, "server_loss", file_name)
    save_pkl(server_acc_hist, "server_acc", file_name)

    # 存储最终模型
    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist