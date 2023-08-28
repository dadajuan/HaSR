import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

class Fusion_z_gen(nn.Module):
    def __init__(self):
        super(Fusion_z_gen, self).__init__()

        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.bn1 = nn.BatchNorm1d(num_features=512, momentum=0.9)
        self.bn2 = nn.BatchNorm1d(num_features=256, momentum=0.9)
        self.bn3 = nn.BatchNorm1d(num_features=128, momentum=0.9)

    def forward(self, en_lr, en_t):
        fusion_z = torch.cat((en_lr, en_t), 1)
        print(fusion_z.shape)
        x = self.linear1(fusion_z)  # [8,256]
        x = self.relu(self.bn1(x))

        x1 = self.linear2(x)
        x1 = self.relu(self.bn2(x1))

        x2 = self.linear3(x1)
        x2 = self.relu(self.bn3(x2))
        return x2

class Fusion_z_dis(nn.Module):

    def __init__(self):
        super(Fusion_z_dis, self).__init__()
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm1d(num_features=64, momentum=0.9)
        self.bn2 = nn.BatchNorm1d(num_features=32, momentum=0.9)
        self.bn3 = nn.BatchNorm1d(num_features=1, momentum=0.9)

    def forward(self, x):
        x1 = self.linear1(x)  # [8,64]
        x1 = self.relu(self.bn1(x1))
        x2 = self.linear2(x1)
        x2 = self.relu(self.bn2(x2))
        x3 = self.linear3(x2)
        x3 = self.relu(self.bn3(x3)) #是否还需要加这个？
        x3_logit = nn.Sigmoid()(x3)   #普通的gan网络是需要需要使用sigmoid层的,
        return x3_logit




class mapping_lr(nn.Module):
    def __init__(self, encoder_lr_dim=512):
        super(mapping_lr, self).__init__()
        self.net1_mapping_lr = nn.Sequential(
                nn.Linear(in_features=encoder_lr_dim, out_features=256),
                nn.BatchNorm1d(num_features=256, momentum=0.9),
                nn.LeakyReLU(0.2, inplace=True)
        )
        self.net2_mapping_lr = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(num_features=128, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, encoder_lr):
        x = self.net1_mapping_lr(encoder_lr)
        x = self.net2_mapping_lr(x)
        return x


class mapping_t(nn.Module):
    def __init__(self, encoder_t_dim=512):
        super(mapping_t, self).__init__()
        self.net1_mapping_t = nn.Sequential(
                nn.Linear(in_features=encoder_t_dim, out_features=256),
                nn.BatchNorm1d(num_features=256, momentum=0.9),
                nn.LeakyReLU(0.2, inplace=True)
        )
        self.net2_mapping_t = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(num_features=128, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, encoder_t):
        x1 = self.net1_mapping_t(encoder_t)
        x2 = self.net2_mapping_t(x1)
        return x2



class classifier_common(nn.Module):  #模态内的鉴别损失
    def __init__(self):
        super(classifier_common, self).__init__()
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 9)
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        return x2

def main():

    tmp1 = torch.rand(8, 512)
    tmp2 = torch.rand(8, 512)
    maper_lr = mapping_lr()
    maper_t = mapping_t()
    lr_z = maper_lr(tmp1)
    t_z = maper_t(tmp2)

    c = classifier_common()
    lr_z_c = c(lr_z)
    t_z_c = c(t_z)

    print('lr_z.shape:', lr_z.shape)
    print('t_.shape:', t_z.shape)
    print('lr_z_c.shape:', t_z_c.shape)
    print('t_z_c.shape:', t_z_c.shape)
if __name__ == "__main__":
    main()


