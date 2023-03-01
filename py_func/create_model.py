import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, layer_1, layer_2):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(784, layer_1)
        self.fc3 = nn.Linear(layer_1, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = self.fc3(x)
        return x


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 卷积层
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 50, kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(200, 50)
        self.fc2 = nn.Linear(50, 10)  # 全连接层

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #drop不改变维度随机删去一些数据
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 200)  # view将数据变为200维的数据
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim = 1)


# class CNN_CIFAR(torch.nn.Module):
#   """Model Used by the paper introducing FedAvg"""
#   def __init__(self):
#        super(CNN_CIFAR, self).__init__()
#        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32, kernel_size=(3,3))
#        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
#        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
#
#        self.fc1 = nn.Linear(4*4*64, 64)
#        self.fc2 = nn.Linear(64, 10)
#
#   def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.max_pool2d(x, 2, 2)
#
#        x = F.relu(self.conv2(x))
#        x = F.max_pool2d(x, 2, 2)
#
#        x=self.conv3(x)
#        x = x.view(-1, 4*4*64)
#
#        x = F.relu(self.fc1(x))
#
#        x = self.fc2(x)
#        return x


class CNN_CIFAR_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR_dropout, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3)
        )

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


class AlexNet_Cifar10(torch.nn.Module):

    def __init__(self):
        super(AlexNet_Cifar10, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv3 = nn.Conv2d(
            in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.dropout(x)

        x = F.relu(self.conv4(x))
        x = self.dropout(x)

        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class AlexNet_Cifar100(torch.nn.Module):

    def __init__(self):
        super(AlexNet_Cifar100, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv3 = nn.Conv2d(
            in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 100)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.dropout(x)

        x = F.relu(self.conv4(x))
        x = self.dropout(x)

        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def load_model(dataset, seed):

    torch.manual_seed(seed)

    if dataset == "MNIST_Net":
        model = MNISTNet()
        print("MNISTNet")
    elif dataset == "MNIST_NN":
        model = NN(50, 10)
    elif dataset == "CIFAR10_CNN":
        #        model = CNN_CIFAR()
        model = CNN_CIFAR_dropout()
    elif dataset == "AlexCIFAR100":
        model = AlexNet_Cifar100()
        print("AlexNet_Cifar100")
    elif dataset == "AlexCIFAR10":
        model = AlexNet_Cifar10()
        print("AlexNet_Cifar10")
    else:
        print("Model para Error!")
        import sys
        sys.exit()

    return model
