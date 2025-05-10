import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFactory(nn.Module):
    def __init__(self, in_channels, filters, dropout_rate=None, fun='relu'):
        super(ConvFactory, self).__init__()

        self.activation_func = fun
        self.norm = nn.BatchNorm1d(in_channels)

        if self.activation_func == 'elu':
            # self.norm = None  # ELU 不使用 BatchNorm1d
            self.activation = nn.ELU()

        elif self.activation_func == 'relu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.ReLU()
        elif self.activation_func == 'prelu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.PReLU()

        elif self.activation_func == 'leaky_relu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.LeakyReLU()

        elif self.activation_func == 'selu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.SELU()

        elif self.activation_func == 'gelu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.GELU()

        elif self.activation_func == 'celu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.CELU()

        elif self.activation_func == 'softplus':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.Softplus()

        elif self.activation_func == 'tanh':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.Tanh()

        elif self.activation_func == 'sigmoid':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.Sigmoid()

        elif self.activation_func == 'softsign':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.Softsign()

        elif self.activation_func == 'swish':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.SiLU()

        # 定义卷积层，kernel_size=3，padding='same'相当于padding=1
        self.conv = nn.Conv1d(in_channels, filters, kernel_size=3, padding=1, bias=False)

        # Dropout层（如果有）
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None

        # 使用ELU激活函数

    def forward(self, x):

        ###################################
        x = self.norm(x)  # 如果有 BatchNorm，先应用它
        x = self.activation(x)  # 然后应用激活函数


        # 2. 卷积层
        x = self.conv(x)
        # 3. Dropout层（如果有）
        if self.dropout:
            x = self.dropout(x)

        return x


# L2 正则化（权重衰减）是通过优化器来控制的。
# 因此在 PyTorch 中，L2 正则化通常通过优化器（例如 torch.optim.Adam）的 weight_decay 参数进行设置。


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=None, fun='relu'):
        super(TransitionLayer, self).__init__()

        self.activation_func = fun
        self.norm = nn.BatchNorm1d(in_channels)

        if self.activation_func == 'elu':
            # self.norm = None  # ELU 不使用 BatchNorm1d
            self.activation = nn.ELU()

        elif self.activation_func == 'relu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.ReLU()
        elif self.activation_func == 'prelu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.PReLU()

        elif self.activation_func == 'leaky_relu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.LeakyReLU()

        elif self.activation_func == 'selu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.SELU()

        elif self.activation_func == 'gelu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.GELU()

        elif self.activation_func == 'celu':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.CELU()

        elif self.activation_func == 'softplus':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.Softplus()

        elif self.activation_func == 'tanh':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.Tanh()

        elif self.activation_func == 'sigmoid':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.Sigmoid()

        elif self.activation_func == 'softsign':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.Softsign()

        elif self.activation_func == 'swish':
            # self.norm = nn.BatchNorm1d(in_channels)
            self.activation = nn.SiLU()

        # 激活函数（ELU）
        self.elu = nn.ELU()

        # 1x1卷积
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

        # Dropout层（如果设置了dropout_rate）
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        # 平均池化层
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.norm(x)  # 如果有 BatchNorm，先应用它
        x = self.activation(x)
        # 激活函数

        # 1x1卷积
        x = self.conv1d(x)

        # 如果有Dropout层，则应用Dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # 平均池化
        x = self.pool(x)
        # print(x.shape)

        return x


# 定义DenseBlock
class DenseBlock(nn.Module):
    def __init__(self, in_channels, layers, growth_rate, dropout_rate=None, fun='elu'):
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.layers = layers
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate
        self.conv_factory_list = nn.ModuleList(
            [ConvFactory(self.in_channels + i * growth_rate, self.growth_rate, dropout_rate, fun)
             for i in range(layers)])

    def forward(self, x):
        # 初始化特征图列表
        list_feature_map = [x]

        # 每层进行卷积操作并拼接特征图
        for i in range(self.layers):
            # conv_factory相当于在每一层的卷积操作
            x = self.conv_factory_list[i](x)
            list_feature_map.append(x)

            # 拼接所有的特征图，逐渐增加通道数
            x = torch.cat(list_feature_map, dim=1)  # 沿着通道维度拼接
            # self.in_channels = x.size(1)

        return x


class SqueezeExcitationLayer(nn.Module):
    def __init__(self, in_channels, ratio=0.25):
        super(SqueezeExcitationLayer, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio

        # 1. 使用全局平均池化，将空间维度压缩
        self.pool = nn.AdaptiveAvgPool1d(1)  # 等效于GlobalAveragePooling1D

        # 2. 第一层全连接，压缩通道数
        self.fc1 = nn.Linear(in_channels, int(in_channels * ratio), bias=False)

        # 3. 第二层全连接，恢复通道数
        self.fc2 = nn.Linear(int(in_channels * ratio), in_channels, bias=False)

        # 4. Sigmoid激活函数用于输出通道权重
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. 全局平均池化
        squeeze = self.pool(x)  # 输出形状：[batch_size, in_channels, 1]

        # squeeze维度变换为(batch_size, in_channels)以适应全连接层
        squeeze = squeeze.view(squeeze.size(0), -1)  # 将 [batch_size, in_channels, 1] 转为 [batch_size, in_channels]

        # 2. 第一层全连接，压缩通道数
        excitation = self.relu(self.fc1(squeeze))

        # 3. 第二层全连接，恢复通道数
        excitation = self.fc2(excitation)

        # 4. Sigmoid激活，生成通道加权系数
        excitation = self.sigmoid(excitation).view(-1, self.in_channels, 1)  # 恢复为 [batch_size, out_channels, 1]
        # print(excitation.shape)
        # print(excitation)

        # print(x.shape)
        # 5. 通道加权
        return x * excitation  # 逐通道相乘









class DenseBlockModel_SE(nn.Module):
    def __init__(self, denseblocks=4, in_1=21, in_2=21, layers=2, filters=192,
                 growth_rate=64, dropout_rate=0, ratio=0.25, fun='prelu'):
        super(DenseBlockModel_SE, self).__init__()

        self.denseblocks = denseblocks
        self.filters = filters
        self.growth_rate = growth_rate
        self.layers = layers
        self.dropout_rate = dropout_rate

        # 输入卷积层
        self.conv1d_1 = nn.Conv1d(in_channels=in_1, out_channels=filters, kernel_size=3, padding=1, bias=False)
        self.conv1d_2 = nn.Conv1d(in_channels=in_2, out_channels=filters, kernel_size=3, padding=1, bias=False)

        # DenseBlock 和 Transition 层
        self.denseblocks_1 = nn.ModuleList(
            [DenseBlock(filters + i * growth_rate * layers, layers, growth_rate, dropout_rate, fun)
             for i in range(denseblocks)])
        self.transitions_1 = nn.ModuleList(
            [TransitionLayer(filters + (i + 1) * growth_rate * layers, filters + (i + 1) * growth_rate * layers,
                             dropout_rate, fun)
             for i in range(denseblocks - 1)])

        self.denseblocks_2 = nn.ModuleList(
            [DenseBlock(filters + i * growth_rate * layers, layers, growth_rate, dropout_rate, fun)
             for i in range(denseblocks)])
        self.transitions_2 = nn.ModuleList(
            [TransitionLayer(filters + (i + 1) * growth_rate * layers, filters + (i + 1) * growth_rate * layers,
                             dropout_rate, fun)
             for i in range(denseblocks - 1)])

        # Squeeze-Excitation 层

        self.se = SqueezeExcitationLayer(2*(filters + denseblocks * growth_rate * layers), ratio)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(2 * (filters + denseblocks * growth_rate * layers), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs, xn):
        def process_module(input, conv, denseblocks, transitions):
            x = conv(input)
            for i in range(self.denseblocks - 1):
                x = denseblocks[i](x)
                x = transitions[i](x)

            x = denseblocks[-1](x)  # 最后一个 DenseBlock
            x = F.elu(x)

            return x

        # 模块 1
        xs = xs.permute(0, 2, 1)
        x_1 = process_module(xs, self.conv1d_1, self.denseblocks_1, self.transitions_1)
        # print('x_1',x_1.shape)
        # # 模块 2
        xn = xn.permute(0, 2, 1)
        x_2 = process_module(xn, self.conv1d_2, self.denseblocks_2, self.transitions_2)

        # # 拼接三个模块的输出
        x = torch.cat([x_1, x_2], dim=-2)

        x = self.se(x)
        x = self.global_avg_pool(x).squeeze(-1)

        # 全连接层输出
        x = self.sigmoid(self.fc(x))
        return x


