import argparse
import numpy as np
import cv2
import torch
from my_dataset import MyDataSet, MyDataSet_tac_densenet
from utils import read_split_data, plot_data_loader_image, read_split_data_tac_densenet
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from torch.autograd import Variable
from torch.optim import RMSprop, Adam, SGD
from torchvision import transforms
from my_dataset import MyDataSet
from auto_encoder import Decoder_HR_gen, Decoder_HR_dis, Encoder_HR_VGG, Encoder_Tac, Encoder_LR_VGG
from s4_fusion_network import mapping_lr, mapping_t, classifier_common
from utils import read_split_data, plot_data_loader_image

from metric import psnr_oneimage, image_normal, sumprescise, TripletLoss
import val_mapping_lr_t as val
from torch.nn import init
import torch.nn as nn
'''stage2, step1'''

root = "D:/Desktop/12.11responce/LMT/NoFlash/Training"
root1 = "D:/Desktop/12.11responce/LMT/NoFlash/Training_LR"
#root2 = "D:/Desktop/12.11responce/LMT/Tapping/Training"

root2 = "D:/Desktop/12.11responce/LMT/Tapping_stft-32-8(128-128)"

model_path_encoder_lr = 'D:/Desktop/12.11responce/path_encoder_lr'
model_path_encoder_t = 'D:/Desktop/12.11responce/path_encoder_t'

path1 = 'D:/Desktop/12.11responce/path_maper_lr'
path2 = 'D:/Desktop/12.11responce/path_maper_t'
path3 = 'D:/Desktop/12.11responce/path_classifier'

def train():
    n_epochs = 3000
    batch_size = 8

    lr = 0.0001
    decay_lr = 0.75
    margin = 0.35
    equilibrium = 0.68
    decay_mse = 1.0
    decay_margin = 1.0
    decay_equilibrium = 1.0
    lambda_mse = 1e-3
    lambda_gen_mse = 1e-2

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
    train_images_lr_path, train_images_lr_label, val_images_lr_path, val_images_lr_label = read_split_data(root1)
    train_tactile_path, train_tactile_label, val_tactile_path, val_tactile_label = read_split_data_tac_densenet(root2)
    #writer = SummaryWriter('./path/to/log')
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

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
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

    encoder_lr = Encoder_LR_VGG()
    encoder_t = Encoder_Tac()

    encoder_lr.load_state_dict(torch.load(model_path_encoder_lr))
    encoder_t.load_state_dict(torch.load(model_path_encoder_t))

    encoder_lr.eval()
    encoder_t.eval()

    encoder_lr.cuda()
    encoder_t.cuda()

    maper_lr = mapping_lr().cuda()
    maper_t = mapping_t().cuda()
    classifier = classifier_common().cuda()

    maper_lr_optimizer = Adam(params=maper_lr.parameters(), lr=lr*10, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    maper_t_optimizer = Adam(params=maper_t.parameters(), lr=lr*10, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    classifier_optimizer = Adam(params=classifier.parameters(), lr=lr * 10, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    step_index = 0
    for i in range(n_epochs):

        for j, (data_batch, imgs_lr, tacs, target_batch, filepath) in enumerate(train_loader):

            #train()
            data_target = Variable(target_batch, requires_grad=False).float().cuda() #标签

            data_in = Variable(data_batch, requires_grad=False).float()/255
            data_in = data_in.cuda() #[8,3,128,128]
            aux_label_batch = Variable(target_batch, requires_grad=False).long().cuda()
            one_hot_class = F.one_hot(aux_label_batch).float()

            data_in_lr = Variable(imgs_lr, requires_grad=False).float()
            data_in_lr = data_in_lr.cuda() #[8,3,128,128]

            data_in_t = Variable(tacs, requires_grad=False).float()
            data_in_t = data_in_t.cuda()  # [8,3,128,128]


            encoder_lr_z, _ = encoder_lr(data_in_lr)  # [bsz, 512]
            encoder_t_z, _ = encoder_t(data_in_t)     # [bsz, 512]

            #train mapping
            for p in classifier.parameters():  # reset requires_grad
                p.requires_grad = False

            maper_lr_optimizer.zero_grad()
            maper_t_optimizer.zero_grad()

            encoder_lr_map = maper_lr(encoder_lr_z)  # [bsz, 128]
            encoder_t_map = maper_t(encoder_t_z)     # [bsz, 128]

            encoder_lr_map_logit = classifier(encoder_lr_map)  # [b,9]
            encoder_t_map_logit = classifier(encoder_t_map)    # [b,9]
            '''三元组损失'''
            loss_triplet = TripletLoss()
            loss_cor_l = loss_triplet(encoder_lr_map, encoder_t_map, data_target)
            loss_cor_t = loss_triplet(encoder_t_map, encoder_lr_map, data_target)
            """分类损失"""
            celoss = nn.CrossEntropyLoss()
            loss_fenlei_l = celoss(encoder_lr_map_logit, data_target.long())
            loss_fenlei_t = celoss(encoder_t_map_logit, data_target.long())
            loss_cor = loss_cor_l + loss_cor_t
            loss_fenlei = loss_fenlei_l + loss_fenlei_t
            loss_sum = loss_cor + loss_fenlei

            loss_sum.backward(retain_graph=True)
            maper_lr_optimizer.step()
            maper_t_optimizer.step()

            #train classifier
            for p in classifier.parameters():  # reset requires_grad
                p.requires_grad = True

            classifier_optimizer.zero_grad()

            encoder_lr_map = maper_lr(encoder_lr_z)  # [bsz, 128]
            encoder_t_map = maper_t(encoder_t_z)  # [bsz, 128]

            encoder_lr_map_logit = classifier(encoder_lr_map)  # [b,9]
            encoder_t_map_logit = classifier(encoder_t_map)  # [b,9]

            celoss = nn.CrossEntropyLoss()
            loss_fenlei_l = celoss(encoder_lr_map_logit, data_target.long())
            loss_fenlei_t = celoss(encoder_t_map_logit, data_target.long())
            loss_fenlei = loss_fenlei_l + loss_fenlei_t
            loss_fenlei.backward()
            classifier_optimizer.step()
            acc_ll = sumprescise(encoder_lr_map_logit, data_target)
            acc_l = acc_ll[0]/acc_ll[1]
            acc_tt = sumprescise(encoder_t_map_logit, data_target)
            acc_t = acc_tt[0] / acc_tt[1]

            print('train---epoch[{0}/{1}],bathchid[{2}/{3}],loss_cor:{4},loss_cor_l:{5},  loss_cor_t:{6},loss_fenlei:{7},acc_l:{8},acc_t{9}'.format
                  (i, n_epochs, j, len(train_loader), loss_cor, loss_cor_l, loss_cor_t, loss_fenlei, acc_l, acc_t))

        val.run(maper_lr, maper_t, classifier, val_loader, log_writer, step_index)
        log_writer.add_scalar('train_loss_sum', loss_sum, step_index)
        log_writer.add_scalar('train_loss_cor', loss_cor, step_index)
        log_writer.add_scalar('train_loss_fenlei', loss_fenlei, step_index)
        log_writer.add_scalar('acc_lr', acc_l, step_index)
        log_writer.add_scalar('acc_lr', acc_t, step_index)

        step_index = step_index + 1
        if i % 2 == 0:
            torch.save(maper_lr.state_dict(),   path1)
            torch.save(maper_t.state_dict(),    path2)
            torch.save(classifier.state_dict(), path3)
    log_writer.close()

if __name__ == '__main__':
    train()
