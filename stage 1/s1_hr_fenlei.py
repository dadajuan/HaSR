import argparse
import numpy as np
import cv2
import torch

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from model import VAEGAN_HR

from torch import autograd
from torch.autograd import Variable
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torchvision import transforms
from auto_encoder import Decoder_HR_gen, Decoder_HR_dis, Encoder_HR_VGG, Encoder_LR_VGG
from my_dataset import MyDataSet, MyDataSet_tac_densenet
from utils import read_split_data, plot_data_loader_image, read_split_data_tac_densenet
from piqa import PSNR
from piqa import SSIM
from metric import psnr_oneimage, image_normal, sumprescise
import val_s1_1_hr_fenlei as val
from torch.nn import init
import torch.nn as nn

root = "D:/Desktop/12.11responce/LMT/NoFlash/Training"
root1 = "D:/Desktop/12.11responce/LMT/NoFlash/Training_LR"
root2 = "D:/Desktop/12.11responce/LMT/Tapping_stft-32-8(128-128)"

path1 = 'D:/Desktop/12.11responce/path_encoder_hr'
path4 = 'D:/Desktop/12.11responce/path_encoder_lr'
path5 = 'D:/Desktop/12.11responce/path_decoder_t'

def train():
    n_epochs = 3000

    lr = 0.0001
    batch_size = 8
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
    train_images_lr_path, train_images_lr_label, val_images_lr_path, val_images_lr_label = read_split_data(root1)
    train_tactile_path, train_tactile_label, val_tactile_path, val_tactile_label = read_split_data_tac_densenet(root2)

        # '''这两行是新的形势下如何读取数据，因为只需要输入一个就行了'''
        # # train_images_lr_path, train_images_lr_label, val_images_lr_path, val_images_lr_label = 0, 0, 0, 0
        # # train_tactile_path, train_tactile_label, val_tactile_path, val_tactile_label = 0, 0, 0, 0
        # #writer = SummaryWriter('./path/to/log')
        # log_writer = SummaryWriter('./path/to/logs')
    log_writer = SummaryWriter('./path/to/logs')

    train_data_set = MyDataSet_tac_densenet(images_path=train_images_path,
                               images_lr_path=train_images_lr_path,
                               tactile_path=train_tactile_path,
                               images_class=train_images_label,
                               transform=None)

    val_data_set = MyDataSet_tac_densenet(images_path=val_images_path,
                             images_lr_path=val_images_lr_path,
                             tactile_path=val_tactile_path,
                             images_class=val_images_label,
                             transform=None)

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=train_data_set.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             collate_fn=val_data_set.collate_fn)

    encoder = Encoder_HR_VGG().cuda()  #没问题
    encoder_optimizer = Adam(params=encoder.parameters(), lr=lr * 10, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                             amsgrad=False)
    step_index = 0

    for i in range(n_epochs):
        for j, (data_batch, imgs_lr, tacs, target_batch, filepath) in enumerate(train_loader):
            #train()
            data_target = Variable(target_batch, requires_grad=False).float().cuda() #标签
            # data_in = Variable(data_batch, requires_grad=False).float()/255
            # data_in = image_normal(data_in)

            '''输入不再进行归一化'''
            data_in_hr = Variable(data_batch, requires_grad=False).float()
            data_in_hr = image_normal(data_in_hr)

            data_in_hr = data_in_hr.cuda() #[8,3,128,128]
            aux_label_batch = Variable(target_batch, requires_grad=False).long().cuda()
            one_hot_class = F.one_hot(aux_label_batch).float()

            #train encoder
            for name, param in encoder.named_parameters():
                if "vgg16" in name:  # densenet121
                    param.requires_grad = False

            encoder_optimizer.zero_grad()

            encoder_hr_z, encoder_hr_z_logit = encoder(data_in_hr) #[8,9]
            celoss = nn.CrossEntropyLoss()
            loss_encoder = celoss(encoder_hr_z_logit, data_target.long())
            loss_encoder.backward(retain_graph=True)
            encoder_optimizer.step()
            acc_ = sumprescise(encoder_hr_z_logit, data_target)
            acc = acc_[0]/acc_[1]
            print('epoch[{0}/{1}],bathchid[{2}/{3}],loss_encoder:{4},acc-hr:{5}'.format
                  (i, n_epochs, j, len(train_loader), loss_encoder.mean(), acc))
            '''训练集结束后,在验证集进行一次测试'''
        val.run(encoder, val_loader, log_writer, step_index)
        log_writer.add_scalar('loss_encoder', loss_encoder, step_index)
        log_writer.add_scalar('acc_hr', acc, step_index)
        step_index = step_index + 1

        if i % 10 == 0:
            torch.save(encoder.state_dict(), path1)

    log_writer.close()
if __name__ == '__main__':
    train()
