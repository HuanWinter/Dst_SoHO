import torch
import torch.nn as nn
from .basic_layers import ResidualBlock
from .attention_module import AttentionModule_stage1, AttentionModule_stage2,\
    AttentionModule_stage3, AttentionModule_stage0
import torch.nn.functional as F


class CNN_second_try(nn.Module):

    def __init__(self, dropout, n_var):
        super().__init__()

        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding_mode='same',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding_mode='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding_mode='same')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn4 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(64*3*3, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.LeakyReLU()
        self.MaxPool2d = nn.MaxPool2d(4)

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out


class CNN_pre(nn.Module):

    def __init__(self, dropout, n_var, radius):
        super().__init__()

        self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding_mode='same',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding_mode='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding_mode='same')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn4 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(64*8*8, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      #nn.Linear(256, 64),
                      #nn.ReLU(),
                      #nn.Dropout(dropout),
                      nn.Linear(256, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool_pre(x)
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out


class CNN_batch(nn.Module):

    def __init__(self, dropout, n_var, radius):
        super().__init__()

        self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding_mode='same',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding_mode='same')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn4 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(64*6*6, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool_pre(x)
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        # x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out

class CNN_batch_multi(nn.Module):

    def __init__(self, dropout, n_var, radius, outputs):
        super().__init__()

        self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding_mode='zeros',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding_mode='zeros')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode='zeros')
        self.bn4 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(64*6*6, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, outputs),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool_pre(x)
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        # x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return F.sigmoid(out)


class CNN_batch_multi_2h(nn.Module):

    def __init__(self, dropout, n_var, radius):
        super().__init__()

        self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding_mode='same',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding_mode='same')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn4 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(64*6*6, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 12),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool_pre(x)
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        # x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return F.sigmoid(out)


class CNN_batch_small(nn.Module):

    def __init__(self, dropout, n_var, radius):
        super().__init__()

        self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(3)
        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding_mode='same',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding_mode='same')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn4 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(64*1*1, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool_pre(x)
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        # x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out


class CNN_FE_jannis(nn.Module):
    "This CNN is just a first test"
    def __init__(self, n_var, w, h, p_dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_var, 4 * n_var, 3, 1, padding=1,
                      padding_mode='zeros', groups=n_var),
            nn.ReLU(),
            nn.BatchNorm2d(4 * n_var),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * n_var, 16 * n_var, 3, 1, padding=1,
                      padding_mode='zeros', groups=4*n_var),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.BatchNorm2d(16 * n_var),
            nn.Flatten(),
            nn.Linear(w * h * 16 * n_var // 16, 128),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(128, 1))

    def forward(self, X):
        return self.net(X)


class CNN_batch_group(nn.Module):

    def __init__(self, dropout, n_var, radius):
        super().__init__()

        self.MaxPool_pre = nn.MaxPool2d(int(radius/32))
        self.MaxPool2d = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=2 * n_var,
                kernel_size=3,
                stride=1,
                groups=n_var,
                padding_mode='zeros',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=2 * n_var,
                out_channels=4 * n_var,
                kernel_size=1,
                groups=2*n_var,
                stride=1,
                padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                groups=32,
                stride=1,
                padding_mode='zeros')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                groups=64,
                stride=1,
                padding_mode='same')
        self.bn4 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(64*8*8, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # self.conv1.we
        # x = self.MaxPool_pre(x)
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        x = self.MaxPool2d(self.ReLU(self.conv2(x)))
        #x = self.MaxPool2d(self.ReLU(self.conv3(x)))
        #x = self.MaxPool2d(x)
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        #x = self.MaxPool2d(self.ReLU(self.conv4(x)))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out
    
class CNN_group(nn.Module):
    "This CNN is just a first test"
    def __init__(self, n_var, w, h, p_dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_var, 4 * n_var, 3, 1, padding=1,
                      padding_mode='zeros', groups=n_var),
            nn.ReLU(),
            nn.BatchNorm2d(4 * n_var),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * n_var, 8 * n_var, 3, 1, padding=1,
                      padding_mode='zeros', groups=4*n_var),
            nn.ReLU(),
            nn.BatchNorm2d(8 * n_var),
            nn.MaxPool2d(2),
            nn.Conv2d(8 * n_var, 16 * n_var, 3, 1, padding=1,
                      padding_mode='zeros', groups=8*n_var),
            nn.ReLU(),
            nn.BatchNorm2d(16 * n_var),
            nn.Flatten(),
            nn.Linear(w * h * 16 * n_var // 64, 256),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(64, 1))

    def forward(self, X):
        return self.net(X)

    
    
class Simple_MLP(nn.Module):

    def __init__(self, dropout, n_var):
        super().__init__()

        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(n_var, 1024),
                      nn.PReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(1024, 64),
                      nn.PReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      #nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
    
        out = self.out(x)
        return out
    
class Simple_MLP2(nn.Module):

    def __init__(self, dropout, n_var):
        super().__init__()

        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(n_var, 16),
                      nn.PReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(16, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.ReLU()
    
    def forward(self, x):
    
        out = self.out(x)
        return out

class CNN_origin(nn.Module):

    def __init__(self, dropout, n_var):
        super().__init__()

        self.conv1 = nn.Conv2d(
                in_channels=n_var,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding_mode='same',
                )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding_mode='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding_mode='same')

        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding_mode='same')
        self.bn4 = nn.BatchNorm2d(128)
        # an affine operation: y = Wx + b
        self.dropout = dropout
        self.out = nn.Sequential(
                      nn.Linear(64*1*1, 256),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(256, 64),
                      nn.ReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(64, 1),
                      # nn.LogSoftmax(dim=1),
                      )
        self.ReLU = nn.LeakyReLU()
        self.MaxPool2d = nn.MaxPool2d(5)

    def forward(self, x):
        # self.conv1.we
        x = self.MaxPool2d(self.ReLU(self.bn1(self.conv1(x))))
        x = self.MaxPool2d(self.ReLU(self.bn2(self.conv2(x))))
        x = self.MaxPool2d(self.ReLU(self.bn3(self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU((self.conv3(x))))
        # x = self.MaxPool2d(self.ReLU(self.bn4(self.conv4(x))))
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out


class CNN_first_try(nn.Module):
    "This CNN is just a first test"
    def __init__(self, n_var, w, h, p_dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_var, 2 * n_var, 3, 1, padding=1, padding_mode='same'),
            nn.ReLU(),
            nn.Conv2d(2 * n_var, 4 * n_var, 3, 1, padding=1,
                      padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm2d(4 * n_var),
            nn.Flatten(),
            nn.Dropout(p=p_dropout),
            nn.Linear(w * h * 4 * n_var // 4, 128),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(128, 1))

    def forward(self, X):
        return self.net(X)


class ResidualAttentionModel_andong_pre(nn.Module):
    # for input size 64
    def __init__(self, n_var):
        super(ResidualAttentionModel_andong_pre, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_var, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage0(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(4096, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        import ipdb; ipdb.set_trace()
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        # out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        # out = self.attention_module3_2(out)
        # out = self.attention_module3_3(out)
        # out = self.residual_block4(out)
        # out = self.residual_block5(out)
        # out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResidualAttentionModel_andong_origin(nn.Module):
    # for input size 200
    def __init__(self, n_var):
        super(ResidualAttentionModel_andong_origin, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_var, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        # out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        # out = self.attention_module3_2(out)
        # out = self.attention_module3_3(out)
        # out = self.residual_block4(out)
        # out = self.residual_block5(out)
        # out = self.residual_block6(out)
        # out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    
class AutoEncoder(nn.Module):
    
    def __init__(self, code_size, n_sample, channels, radius):
        super().__init__()
        self.code_size = code_size
        self.n_sample = n_sample
        
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(channels, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(13 * 13 * 20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)
        
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, channels*radius**2)
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = code.view([images.shape[0], -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code
    
    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out.view([self.n_sample, channels, radius, radius])
        return out
    
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
