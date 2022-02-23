import torch
import torch.nn as nn
import copy
import sys

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, BN=True, relu_type='leaky'):
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

# define the discriminator
class Vanila_CNN_Lite(nn.Module):
    def __init__(self, fil_num, drop_rate):
        super(Vanila_CNN_Lite, self).__init__()
        self.block1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block2 = ConvLayer(fil_num, 2*fil_num, 0.1, (4, 1, 0), (2, 2, 0))
        self.block3 = ConvLayer(2*fil_num, 4*fil_num, 0.1, (3, 1, 0), (2, 2, 0))
        self.block4 = ConvLayer(4*fil_num, 8*fil_num, 0.1, (3, 1, 0), (2, 1, 0))
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(8*fil_num*5*7*5, 30),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(30, 2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

# define the generator
class _netG(nn.Module):
    def __init__(self, num):
        super(_netG, self).__init__()
        # option1: all transposed conv layers
        # self.conv1 = nn.ConvTranspose3d(1, 2*num, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv2 = nn.ConvTranspose3d(2*num, num, kernel_size=2, stride=2, padding=1, bias=False)
        # self.conv3 = nn.ConvTranspose3d(num, num//2, kernel_size=2, stride=2, padding=1, bias=False)
        # self.conv4 = nn.ConvTranspose3d(num//2, 1, kernel_size=2, stride=2, padding=5, bias=False)

        #option2: transposed conv layer + conv layer
        # self.conv1 = nn.Conv3d(1, 2*num, kernel_size=5, stride=3, padding=2, bias=False)
        # self.conv2 = nn.ConvTranspose3d(2*num, num, kernel_size=2, stride=2, padding=1, bias=False)
        # self.conv3 = nn.ConvTranspose3d(num, num//2, kernel_size=2, stride=2, padding=6, bias=False)
        # self.conv4 = nn.ConvTranspose3d(num//2, 1, kernel_size=1, stride=1, padding=1, bias=False)

        #option3: all conv layers
        self.conv1 = nn.Conv3d(1, 2*num, kernel_size=(5,5,5), stride=(1,1,1), padding=(2,2,2), bias=False)
        self.conv2 = nn.Conv3d(2*num, num, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=False)
        self.conv3 = nn.Conv3d(num, num//2, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=False)
        self.conv4 = nn.Conv3d(num//2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)

        self.relu = nn.ReLU()
        self.BN1 = nn.BatchNorm3d(2*num)
        self.BN2 = nn.BatchNorm3d(num)
        self.BN3 = nn.BatchNorm3d(num//2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.tanh(x)
        return x

# define the discriminator
class _netD(nn.Module):
    def __init__(self, num):
        super(_netD, self).__init__()
        self.conv1 = nn.Conv3d(1, num, 4, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(num)
        self.conv2 = nn.Conv3d(num, 2*num, 4, 2, 0, bias=False)
        self.bn2 = nn.BatchNorm3d(2*num)
        self.conv3 = nn.Conv3d(2*num, 4*num, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(4*num)
        self.conv4 = nn.Conv3d(4*num, 8*num, 3, 2, 0, bias=False)
        self.bn4 = nn.BatchNorm3d(8*num)
        self.conv5 = nn.Conv3d(8*num, 1, 2, 1, 0, bias=False)

        # self.bny = nn.BatchNorm1d(50*50*50)
        # self.fc1 = nn.Linear(50*50*50, 136) #remove Bias?
        # self.fc2 = nn.Linear(136*2, 1)
        # self.fc2 = nn.Linear(136, 1)

        self.lr = nn.LeakyReLU()
        self.sg = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lr(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lr(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lr(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lr(x)

        # x = x.view(x.shape[0], -1)
        # y = y.view(y.shape[0], -1)
        # y = self.bny(y)
        # y = self.fc1(y)
        # y = self.lr(y)
        # x = torch.cat((x, y), 1)
        # x = self.fc2(x)

        x = self.conv5(x)
        # x = self.sg(x)# comment out for W-loss
        x = x.view(-1)
        return x

class _FCNt(nn.Module):
    def __init__(self, num, p, dim=1):
        super(_FCNt, self).__init__()
        self.conv1 = nn.Conv3d(dim, num, 4, 1, 0, bias=False)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        self.conv1a = nn.Conv3d(num, num, 2, 1, 0, bias=False)
        self.mp1 = nn.MaxPool3d(2, 1, 0)
        self.bn1 = nn.BatchNorm3d(num)
        self.bn1a = nn.BatchNorm3d(num)
        # self.conv2 = nn.Conv3d(num, 2*num, 4, 2, 0, bias=False)
        self.conv2 = nn.Conv3d(num, 2*num, 4, 1, 0, bias=False)
        self.conv2a = nn.Conv3d(2*num, 2*num, 2, 2, 0, bias=False)
        self.mp2 = nn.MaxPool3d(2, 2, 0)
        self.bn2 = nn.BatchNorm3d(2*num)
        self.bn2a = nn.BatchNorm3d(2*num)
        self.conv3 = nn.Conv3d(2*num, 4*num, 3, 1, 0, bias=False)
        self.conv3a = nn.Conv3d(4*num, 4*num, 2, 2, 0, bias=False)
        self.mp3 = nn.MaxPool3d(2, 2, 0)
        self.bn3 = nn.BatchNorm3d(4*num)
        self.bn3a = nn.BatchNorm3d(4*num)
        # self.conv4 = nn.Conv3d(4*num, 8*num, 2, 1, 0, bias=False)
        self.conv4 = nn.Conv3d(4*num, 8*num, 3, 1, 0, bias=False)
        self.conv4a = nn.Conv3d(8*num, 8*num, 2, 1, 0, bias=False)
        self.mp4 = nn.MaxPool3d(2, 1, 0)
        self.bn4 = nn.BatchNorm3d(8*num)
        self.bn4a = nn.BatchNorm3d(8*num)
        self.skip = nn.Conv3d(2*num, 2*num, 2, 2, 0, bias=False)
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        # self.conv1 = nn.Conv3d(dim, num, 4, 1, 0, bias=False)
        # self.mp1 = nn.MaxPool3d(2, 1, 0)
        # self.bn1 = nn.BatchNorm3d(num)
        # self.conv2 = nn.Conv3d(num, 2*num, 4, 1, 0, bias=False)
        # self.mp2 = nn.MaxPool3d(2, 2, 0)
        # self.bn2 = nn.BatchNorm3d(2*num)
        # self.conv3 = nn.Conv3d(2*num, 4*num, 3, 1, 0, bias=False)
        # self.mp3 = nn.MaxPool3d(2, 2, 0)
        # self.bn3 = nn.BatchNorm3d(4*num)
        # self.conv4 = nn.Conv3d(4*num, 8*num, 3, 1, 0, bias=False)
        # self.mp4 = nn.MaxPool3d(2, 1, 0)
        # self.bn4 = nn.BatchNorm3d(8*num)

        self.relu = nn.LeakyReLU()
        self.dr = nn.Dropout(0.1)

        self.dr2 = nn.Dropout(p)
        self.l1 = nn.Linear(8*num*6*6*6, 30)
        self.l2 = nn.Linear(30, 2)

        self.feature_length = 8*num*6*6*6
        self.num = num

    def forward(self, x, stage='train'):
        # print('inp shape', x.shape)
        # x = self.dr(self.relu(self.bn1(self.mp1(self.conv1(x)))))
        # x = self.dr(self.relu(self.bn1(self.conv1(x))))
        # # print('out shape1', x.shape)
        # x = self.dr(self.relu(self.conv1a(x)))
        # # print('out shape', x.shape)
        # x = self.dr(self.relu(self.bn2(self.conv2(x))))
        # y = self.skip(x)
        # # print('out shape2', x.shape)
        # x = self.dr(self.relu(self.conv2a(x))) + y
        # # print('out shape', x.shape)
        # # print(y.shape)
        # x = self.dr(self.relu(self.bn3(self.conv3(x))))
        # # print('out shape3', x.shape)
        # x = self.dr(self.relu(self.conv3a(x)))
        # # print('out shape', x.shape)
        # x = self.dr(self.relu(self.bn4(self.conv4(x))))
        # # print('out shape4', x.shape)
        # x = self.dr(self.relu(self.conv4a(x)))
        # # print('outa shape', x.shape)
        # sys.exit()
        # print('final shape', x.shape)
        print('outa shape', x.shape)
        x = self.dr(self.relu(self.bn1(self.mp1(self.conv1(x)))))
        print('outa shape', x.shape)
        x = self.dr(self.relu(self.bn2(self.mp2(self.conv2(x)))))
        print('outa shape', x.shape)
        x = self.dr(self.relu(self.bn3(self.mp3(self.conv3(x)))))
        print('outa shape', x.shape)
        x = self.dr(self.relu(self.bn4(self.mp4(self.conv4(x)))))
        print('outa shape', x.shape)
        sys.exit()

        if stage != 'inference':
            x = x.view(-1, self.feature_length)
        x = self.relu(self.l1(self.dr2(x)))
        x = self.l2(self.dr2(x))
        return x

    def dense_to_conv(self):
        fcn = copy.deepcopy(self)
        A = fcn.l1.weight.view(30, 8*self.num, 6, 6, 6)
        B = fcn.l2.weight.view(2, 30, 1, 1, 1)
        C = fcn.l1.bias
        D = fcn.l2.bias
        fcn.l1 = nn.Conv3d(160, 30, 6, 1, 0).cuda()
        fcn.l2 = nn.Conv3d(30, 2, 1, 1, 0).cuda()
        fcn.l1.weight = nn.Parameter(A)
        fcn.l2.weight = nn.Parameter(B)
        fcn.l1.bias = nn.Parameter(C)
        fcn.l2.bias = nn.Parameter(D)
        return fcn


class _FCN(nn.Module):
    def __init__(self, num, p, dim=1):
        super(_FCN, self).__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        self.conv1 = nn.Conv3d(dim, num, 4, 1, 0, bias=False)
        self.mp1 = nn.MaxPool3d(2, 1, 0)
        self.bn1 = nn.BatchNorm3d(num)
        self.conv2 = nn.Conv3d(num, 2*num, 4, 1, 0, bias=False)
        self.mp2 = nn.MaxPool3d(2, 2, 0)
        self.bn2 = nn.BatchNorm3d(2*num)
        self.conv3 = nn.Conv3d(2*num, 4*num, 3, 1, 0, bias=False)
        self.mp3 = nn.MaxPool3d(2, 2, 0)
        self.bn3 = nn.BatchNorm3d(4*num)
        self.conv4 = nn.Conv3d(4*num, 8*num, 3, 1, 0, bias=False)
        self.mp4 = nn.MaxPool3d(2, 1, 0)
        self.bn4 = nn.BatchNorm3d(8*num)

        self.relu = nn.LeakyReLU()
        self.dr = nn.Dropout(0.1)

        self.dr2 = nn.Dropout(p)
        self.l1 = nn.Linear(8*num*6*6*6, 30)
        self.l2 = nn.Linear(30, 4)

        self.feature_length = 8*num*6*6*6
        self.num = num

    def forward(self, x, stage='train'):
        x = self.dr(self.relu(self.bn1(self.mp1(self.conv1(x)))))
        x = self.dr(self.relu(self.bn2(self.mp2(self.conv2(x)))))
        x = self.dr(self.relu(self.bn3(self.mp3(self.conv3(x)))))
        x = self.dr(self.relu(self.bn4(self.mp4(self.conv4(x)))))

        if stage != 'inference':
            x = x.view(-1, self.feature_length)
        x = self.relu(self.l1(self.dr2(x)))
        x = self.l2(self.dr2(x))
        return x

    def dense_to_conv(self):
        fcn = copy.deepcopy(self)
        A = fcn.l1.weight.view(30, 8*self.num, 6, 6, 6)
        B = fcn.l2.weight.view(4, 30, 1, 1, 1)
        C = fcn.l1.bias
        D = fcn.l2.bias
        fcn.l1 = nn.Conv3d(160, 30, 6, 1, 0).cuda()
        fcn.l2 = nn.Conv3d(30, 4, 1, 1, 0).cuda()
        fcn.l1.weight = nn.Parameter(A)
        fcn.l2.weight = nn.Parameter(B)
        fcn.l1.bias = nn.Parameter(C)
        fcn.l2.bias = nn.Parameter(D)
        return fcn

class _MLP(nn.Module):
    "MLP that only use DPMs from fcn"
    def __init__(self, in_size, drop_rate, fil_num):
        super(_MLP, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(fil_num)
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, 4)
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


if __name__ == "__main__":
    print('models.py')
    model = Vanila_CNN_Lite(10, 0.5).cuda()
    input = torch.Tensor(10, 1, 181, 217, 181).cuda()
    output = model(input)
    print(output.shape)
