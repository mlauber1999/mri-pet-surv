import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(fil_num)
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 1)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()

    def forward(self, X):
        X = self.bn1(X)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        return out

class _MLP_Surv(nn.Module):
    def __init__(self, in_size, drop_rate, fil_num,
                 output_shape=1):
        super(_MLP_Surv, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(fil_num)
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, output_shape)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self, X):
        X = self.bn1(X)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        out = self.sig(out)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type=='leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class _CNN(nn.Module):
    "Encoder that encodes Scan to vector"
    def __init__(self, in_size, drop_rate, fil_num=10,
                 out_channels=1):
        super().__init__()
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        self.conv1 = nn.Conv3d(in_size, fil_num, 5, 4, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 2*fil_num, 5, 4, 0, bias=False)
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 5, 4, 0, bias=False)
        self.bn3 = nn.BatchNorm3d(4*fil_num)
        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)

        self.feature_length = 4*fil_num
        self.in_size = in_size
        self.l1 = nn.Linear(self.feature_length, 25)
        self.l2 = nn.Linear(25, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.LeakyReLU()
        '''
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()
        self.num = fil_num

    def forward(self, x):
        x = self.dr(self.conva(self.bn1(self.conv1(x))))
        x = self.dr(self.conva(self.bn2(self.conv2(x))))
        x = self.dr(self.conva(self.bn3(self.conv3(x))))
        x = x.squeeze()
        x = self.l1a(self.l1(x))
        x = self.l2a(self.l2(x))
        return x