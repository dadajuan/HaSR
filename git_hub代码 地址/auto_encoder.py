import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

class Decoder_HR_gen(nn.Module):
    def __init__(self):
        super(Decoder_HR_gen, self).__init__()

        self.liner1 = nn.Linear(128, 1024)
        self.linear2 = nn.Linear(1024, 512*8*8)
        self.conv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, padding=2,
                                        stride=2, output_padding=1, bias=False)
        self.conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, padding=2,
                                        stride=2, output_padding=1, bias=False)
        self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, padding=2,
                                        stride=2, output_padding=1, bias=False)
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, padding=2,
                                        stride=2, output_padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.bn0 = nn.BatchNorm1d(num_features=1024, momentum=0.9)
        self.bn0_1 = nn.BatchNorm1d(num_features=512*8*8, momentum=0.9)
        self.bn1 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.bn2 = nn.BatchNorm2d(num_features=128, momentum=0.9)
        self.bn3 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.bn4 = nn.BatchNorm2d(num_features=3, momentum=0.9)

    def forward(self, x):
        x = x.clone()
        x1 = self.liner1(x)
        x1 = self.relu(self.bn0(x1))
        x2 = self.linear2(x1)
        x2 = self.relu(self.bn0_1(x2))
        x2 = x2.view(-1, 512, 8, 8)
        x3 = self.conv1(x2)  # 256*16*16
        x3 = self.relu(self.bn1(x3))
        #print(f'x3.shape{x3.shape}')
        x4 = self.conv2(x3)
        x4 = self.relu(self.bn2(x4))  # 128*32*32
        #print(f'x4.shape{x4.shape}')
        x5 = self.conv3(x4)
        x5 = self.relu(self.bn3(x5))  # 64*64*64
        #print(f'x5.shape{x5.shape}')
        x6 = self.conv4(x5)
        x6 = self.relu(self.bn4(x6))  # 3*128*128
        #print(f'x6.shape{x6.shape}')
        return x6
class Decoder_HR_dis(nn.Module):
    def __init__(self):
        super(Decoder_HR_dis, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, padding=2, stride=2,  bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=2, stride=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=2, stride=2, bias=False)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=2, stride=2, bias=False)

        self.linear1 = nn.Linear(41472, 1024)
        self.linear2 = nn.Linear(1024, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.bn2 = nn.BatchNorm2d(num_features=128, momentum=0.9)
        self.bn3 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.bn4 = nn.BatchNorm2d(num_features=512, momentum=0.9)
    def forward(self, x):
        x1 = self.conv1(x)
        #x1 = self.relu(self.bn1(x1))
        x1 = self.relu(self.bn1(x1))

        x2 = self.conv2(x1)
        x2 = self.relu(self.bn2(x2))

        x3 = self.conv3(x2)
        x3 = self.relu(self.bn3(x3))

        x4 = self.conv4(x3)
        x4 = self.relu(self.bn4(x4))
        #print(x4.shape)

        x4 = x4.view(len(x4), -1)
        #print(x4.shape)
        x5 = self.linear1(x4)
        x6 = self.linear2(x5)
        # x6_logit = torch.sigmoid(x6) #wgan的话不需要
        return x1, x2, x3, x4, x5, x6

class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False, t=False):
        ten = self.conv(ten)
        ten_out = ten
        ten = self.bn(ten)
        ten = F.relu(ten, False)  #考虑换成lrelu
        if out:
            return ten, ten_out
        return ten


class Encoder_HR(nn.Module):
    def __init__(self, channel_in=3, hr_size=128):
        super(Encoder_HR, self).__init__()
        self.size = channel_in
        layers = []  # include net1 net2 net3 net4

        for i in range(4):
            if i == 0:
                layers.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2
        # final shape B x 256 x 8 x 8
        self.conv = nn.Sequential(*layers)  # include net1 net2 net3 net4

        self.net5 = nn.Sequential(nn.Linear(in_features=8 * 8 * self.size, out_features=1024, bias=False),
                                  nn.BatchNorm1d(num_features=1024, momentum=0.9),
                                  nn.ReLU(True))
        self.l_out = nn.Linear(in_features=1024, out_features=hr_size)
        self.l_var = nn.Linear(in_features=hr_size, out_features=9)

    def forward(self, ten):
        ten = self.conv(ten)  # 512*8*8
        #print(ten.shape)
        ten = ten.view(len(ten), -1)
        ten = self.net5(ten)
        out = self.l_out(ten)  # 128
        out_logit = self.l_var(out)  # 9
        #out_logit = nn.Softmax(out_logit)
        return out, out_logit

    def __call__(self, *args, **kwargs):
        return super(Encoder_HR, self).__call__(*args, **kwargs)


class Encoder_HR_VGG(nn.Module):
    def __init__(self, channel_in=3, z_size=128):
        super(Encoder_HR_VGG, self).__init__()
        self.size = channel_in
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.l_out = nn.Linear(in_features=1000, out_features=z_size)

        self.l_var = nn.Linear(in_features=z_size, out_features=9)

    def forward(self, ten):
        ten = self.vgg16(ten)
        out = self.l_out(ten)
        out_logit = self.l_var(out)
        #out_logit = nn.Softmax(out_logit)
        return out, out_logit


class Encoder_LR_VGG(nn.Module):
    def __init__(self, channel_in=3, z_size=512):
        super(Encoder_LR_VGG, self).__init__()
        self.size = channel_in
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.l_out = nn.Linear(in_features=1000, out_features=z_size)
        self.l_var = nn.Linear(in_features=z_size, out_features=9)

    def forward(self, ten):
        ten = self.vgg16(ten)
        out = self.l_out(ten)
        out_logit = self.l_var(out)
        #out_logit = nn.Softmax(out_logit)
        return out, out_logit

class Encoder_Tac(nn.Module):
    def __init__(self, channel_in=3, z_size=512):
        super(Encoder_Tac, self).__init__()
        self.size = channel_in
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        # print(self.densenet121)

        self.l_out = nn.Linear(in_features=1000, out_features=z_size)
        self.l_var = nn.Linear(in_features=z_size, out_features=9)

    def forward(self, ten):
        ten = self.densenet121(ten)
        out = self.l_out(ten)
        out_logit = self.l_var(out)
        #out_logit = nn.Softmax(out_logit)
        return out, out_logit

# def main():
#
#     tmp = torch.rand(8, 3, 128, 128)
#
#     encoder = Encoder_Tac()
#     tmp_1, tmp_2 = encoder(tmp)
#     print(tmp_1.shape)
#
#     # tmp_2 = torch.rand(8, 128)
#     # generator = Decoder_HR_gen()
#     # img_gen = generator(tmp_2)
#     # dis = Decoder_HR_dis()
#     # x1, x2, x3, x4, x5, x6 = dis(tmp)
#     # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, x6.shape)
#     # print(img_gen.shape)
#
#     #print(hr_z.shape, hr_z_logit.shape, hr_z_logit)
#
# if __name__ == "__main__":
#     main()
class Encoder_gen(nn.Module):

    def __init__(self):
        super(Encoder_gen, self).__init__()
        self.encoder = Encoder_HR_VGG()
        self.decoder = Decoder_HR_gen()

    def forward(self, input):
        encoder_hr, _ = self.encoder(input)
        hr_gen = self.decoder(encoder_hr)

        return hr_gen

path12 = 'D:/Desktop/12.11responce/path_xxxx'
path13 = 'D:/Desktop/12.11responce/path_yyyy'

def main():

    tmp = torch.rand(8, 3, 128, 128)
    encoder_hr = Encoder_HR()
    a, b = encoder_hr(tmp)
    print(a.shape, b.shape)
    exit()

    encoder_gen = Encoder_gen()
    print(encoder_gen)
    # net1 = Encoder_HR_VGG()
    # net2 = Decoder_HR_gen()
    # encoder_gen2 = Encoder_gen2(net1, net2)
    # print(encoder_gen2)
    out = encoder_gen(tmp)
    print(out.shape)
    torch.save(encoder_gen.encoder.state_dict(), path12)
    torch.save(encoder_gen.decoder.state_dict(), path13)


    #print(hr_z.shape, hr_z_logit.shape, hr_z_logit)

if __name__ == "__main__":
    main()







