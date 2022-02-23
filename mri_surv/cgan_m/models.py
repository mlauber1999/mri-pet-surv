import torch
import torch.nn as nn
import copy
import sys
import matplotlib.pyplot as plt

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
    def __init__(self, num, p, dim=1, out=4):
        super(_FCN, self).__init__()
        self.out = out
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
        self.l2 = nn.Linear(30, out)
        '''
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

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
        x = self.l2(self.dr2(x)) # no activation here for regression (does not bound)
        # x = self.ac(x)
        return x

    def dense_to_conv(self):
        fcn = copy.deepcopy(self)
        A = fcn.l1.weight.view(30, 8*self.num, 6, 6, 6)
        B = fcn.l2.weight.view(self.out, 30, 1, 1, 1)
        C = fcn.l1.bias
        D = fcn.l2.bias
        fcn.l1 = nn.Conv3d(8*self.num, 30, 6, 1, 0).cuda()
        fcn.l2 = nn.Conv3d(30, self.out, 1, 1, 0).cuda()
        fcn.l1.weight = nn.Parameter(A)
        fcn.l2.weight = nn.Parameter(B)
        fcn.l1.bias = nn.Parameter(C)
        fcn.l2.bias = nn.Parameter(D)
        return fcn

class _MLP(nn.Module):
    "MLP that only use DPMs from fcn"
    def __init__(self, in_size, drop_rate, fil_num, out=4):
        super(_MLP, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(fil_num)
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, out)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()
        '''
        RELU & Softmax & Sigmoid: not learning
        LeakyReLU: learning, not similar to without it
        GELU: best currently

        '''
        # self.ac2 = nn.GELU()
        '''
        RELU & Softmax & Sigmoid: not learning
        LeakyReLU: learning, not similar to without it
        GELU: best currently

        '''
        # self.ac2 = nn.GELU()
        # self.ac2 = nn.()

    def forward(self, X):
        X = self.bn1(X)
        out = self.do1(X)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.ac1(out)
        out = self.do2(out)
        out = self.fc2(out)
        # out = self.ac2(out)
        return out

class _Encoder(nn.Module):
    "Encoder that encodes Scan to vector"
    def __init__(self, drop_rate, fil_num=64, in_channels=1, out_channels=1):
        super().__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        self.conv1 = nn.Conv3d(in_channels, fil_num, 4, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 2*fil_num, 4, 2, 0, bias=False)
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 4, 2, 0, bias=False)
        self.bn3 = nn.BatchNorm3d(4*fil_num)
        self.conv4 = nn.Conv3d(4*fil_num, 8*fil_num, 4, 2, 0, bias=False)
        self.bn4 = nn.BatchNorm3d(8*fil_num)

        # (D−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # self.conv5 = nn.ConvTranspose3d(fil_num, in_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))
        # self.bn5 = nn.BatchNorm3d(in_channels)

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)

        self.feature_length = 8*fil_num*5*7*5
        self.l1 = nn.Linear(self.feature_length, 100)
        self.l2 = nn.Linear(100, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.Sigmoid()
        '''
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        # print('target', x.shape)
        x = self.dr(self.conva(self.bn1(self.conv1(x))))
        # print('input', x.shape)
        # x = self.dr(self.conva(self.bn5(self.conv5(x))))
        # print('output', x.shape)
        # sys.exit()
        x = self.dr(self.conva(self.bn2(self.conv2(x))))
        x = self.dr(self.conva(self.bn3(self.conv3(x))))
        x = self.dr(self.conva(self.bn4(self.conv4(x))))

        x = x.view(-1, self.feature_length)
        x = self.l1a(self.l1(x))
        x = self.l2a(self.l2(x))
        # x = self.ac(x)
        return x


    # def __init__(self, in_size, drop_rate, fil_num, out):
    #     super().__init__()
    #     self.en_fc1 = nn.Linear(in_size, fil_num)
    #     self.en_fc2 = nn.Linear(fil_num, out)
    #     self.en_bn1 = nn.BatchNorm1d(fil_num)
    #     self.en_bn2 = nn.BatchNorm1d(out)
    #     self.en_do1 = nn.Dropout(drop_rate)
    #     self.en_do2 = nn.Dropout(drop_rate)
    #     self.en_ac1 = nn.ReLU()
    #     # self.en_ac1 = nn.LeakyReLU()
    #     self.en_ac2 = nn.Sigmoid()
    #
    # def forward(self, X):
    #     # out = self.en_do1(out)
    #     out = self.en_fc1(X)
    #     out = self.en_bn1(out)
    #     out = self.en_ac1(out)
    #
    #     # out = self.en_do2(out)
    #     out = self.en_fc2(out)
    #     out = self.en_bn2(out)
    #     out = self.en_ac2(out)
    #
    #     return out

class _Decoder(nn.Module):
    "Decoder that decodes vector to Scan"
    def __init__(self, drop_rate, fil_num, in_channels=1, out_channels=1):
        super().__init__()
        # ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.conv1 = nn.ConvTranspose3d(8*fil_num, 4*fil_num, 4, 2, 0, bias=False, output_padding=(1,0,1))
        self.bn1 = nn.BatchNorm3d(4*fil_num)
        self.conv2 = nn.ConvTranspose3d(4*fil_num, 2*fil_num, 4, 2, 0, bias=False, output_padding=(0,0,0))
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.ConvTranspose3d(2*fil_num, fil_num, 4, 2, 0, bias=False, output_padding=(1,1,1))
        self.bn3 = nn.BatchNorm3d(fil_num)
        self.conv4 = nn.ConvTranspose3d(fil_num, out_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))
        self.bn4 = nn.BatchNorm3d(out_channels)

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)

        self.feature_length = 8*fil_num*5*7*5
        self.l1 = nn.Linear(in_channels, 100)
        self.l2 = nn.Linear(100, self.feature_length)
        self.l1a = nn.Sigmoid()
        self.l2a = nn.LeakyReLU()
        ''' TO TRY
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        x = self.l1a(self.l1(x))
        x = self.l2a(self.l2(x))
        x = x.view(-1, 8*self.num,5,7,5)

        x = self.dr(self.conva(self.bn1(self.conv1(x))))
        x = self.dr(self.conva(self.bn2(self.conv2(x))))
        x = self.dr(self.conva(self.bn3(self.conv3(x))))
        x = self.dr(self.conva(self.bn4(self.conv4(x))))

        return x

    # def __init__(self, in_size, drop_rate, fil_num, out):
    #     super().__init__()
    #     self.de_fc1 = nn.Linear(in_size, fil_num)
    #     self.de_fc2 = nn.Linear(fil_num, out)
    #     self.de_bn1 = nn.BatchNorm1d(fil_num)
    #     self.de_bn2 = nn.BatchNorm1d(out)
    #     self.de_do1 = nn.Dropout(drop_rate)
    #     self.de_do2 = nn.Dropout(drop_rate)
    #     self.de_ac1 = nn.ReLU()
    #     # self.de_ac1 = nn.LeakyReLU()
    #     self.de_ac2 = nn.Sigmoid()
    #     '''
    #     RELU & Softmax & Sigmoid: not learning
    #     LeakyReLU: learning, not similar to without it
    #     GELU: best currently
    #     '''
    #
    # def forward(self, X):
    #     # out = self.de_do1(out)
    #     out = self.de_fc1(X)
    #     out = self.de_bn1(out)
    #     out = self.de_ac1(out)
    #
    #     # out = self.de_do2(out)
    #     out = self.de_fc2(out)
    #     out = self.de_bn2(out)
    #     out = self.de_ac2(out)
    #
    #     return out

class _CNN(nn.Module):
    "Encoder that encodes Scan to vector"
    def __init__(self, drop_rate, fil_num=64, in_channels=1, out_channels=1):
        super().__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        self.conv1 = nn.Conv3d(in_channels, fil_num, 5, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 8*fil_num, 5, 2, 0, bias=False)
        self.bn2 = nn.BatchNorm3d(8*fil_num)
        # self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 5, 4, 0, bias=False)
        # self.bn3 = nn.BatchNorm3d(4*fil_num)
        # self.conv4 = nn.Conv3d(4*fil_num, 8*fil_num, 4, 3, 0, bias=False)
        # self.bn4 = nn.BatchNorm3d(8*fil_num)

        # (D−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # self.conv5 = nn.ConvTranspose3d(fil_num, in_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))
        # self.bn5 = nn.BatchNorm3d(in_channels)

        self.mp = nn.MaxPool3d(2)
        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)

        # self.feature_length = 2*fil_num*10*13*9 # for larger input
        # self.feature_length = 8*fil_num*7*8*7
        self.feature_length = 8*fil_num*6*8*6
        self.l1 = nn.Linear(self.feature_length, 25)
        self.l2 = nn.Linear(25, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.Sigmoid()
        # self.l2a = nn.Tanh()
        '''
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        # x = self.dr(self.mp(self.conva(self.bn1(self.conv1(x)))))
        # x = self.dr(self.mp(self.conva(self.bn2(self.conv2(x)))))
        x = self.mp(self.conva(self.bn1(self.conv1(x))))
        x = self.mp(self.conva(self.bn2(self.conv2(x))))
        # x = self.dr(self.conva(self.bn3(self.conv3(x))))
        # x = self.dr(self.conva(self.bn4(self.conv4(x))))
        # print(x.shape)
        # sys.exit()


        x = x.view(-1, self.feature_length)
        x = self.l1a(self.l1(x))
        x = self.l2(x)
        # x = self.l2a(self.l2(x))
        # x = self.ac(x)
        return x


if __name__ == "__main__":
    print('models.py')

    #  use gpu if available
    batch_size = 512
    epochs = 20
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    encoder = _Encoder(in_size=784, drop_rate=.5, out=128, fil_num=128).to(device)
    decoder = _Decoder(in_size=128, drop_rate=.5, out=784, fil_num=128).to(device)
    # model = AE(input_shape=784).to(device)


    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    import torch.optim as optim
    import torchvision

    optimizerE = optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizerD = optim.Adam(decoder.parameters(), lr=learning_rate)

    # mean-squared error loss
    criterion = nn.MSELoss()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loss_imp = 0.0
    loss_tot = 0.0
    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            # compute reconstructions
            # outputs = model(batch_features)
            # print(batch_features.shape)
            # sys.exit()
            vector = encoder(batch_features)
            outputs = decoder(vector)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizerE.step()
            optimizerD.step()

            vector = encoder(batch_features)
            outputs = decoder(vector)
            loss2 = criterion(outputs, batch_features)
            if loss2 < train_loss:
                loss_imp += 1
            loss_tot += 1

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss), 'loss improved: %.2f' % (loss_imp/loss_tot))

    test_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    test_examples = None

    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, 784).to(device)
            reconstruction = decoder(encoder(test_examples))
            break
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].cpu().numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].cpu().numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        plt.savefig("AE.png")
