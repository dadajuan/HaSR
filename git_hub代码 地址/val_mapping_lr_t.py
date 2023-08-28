import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from metric import psnr_oneimage, image_normal, sumprescise, TripletLoss
from auto_encoder import Decoder_HR_gen, Decoder_HR_dis, Encoder_HR_VGG, Encoder_Tac, Encoder_LR_VGG

model_path_encoder_lr = 'D:/Desktop/12.11responce/path_encoder_lr'
model_path_encoder_t = 'D:/Desktop/12.11responce/path_encoder_t'
from torch import nn

encoder_lr = Encoder_LR_VGG()
encoder_t = Encoder_Tac()

encoder_lr.load_state_dict(torch.load(model_path_encoder_lr))
encoder_t.load_state_dict(torch.load(model_path_encoder_t))
encoder_lr.eval()
encoder_t.eval()
encoder_lr.cuda()
encoder_t.cuda()

def run(maper_lr, maper_t, classifier, val_loader, log_writer, step_index):

    for j, (data_batch, imgs_lr, tacs, target_batch, filepath) in enumerate(val_loader):
        #eval()
        data_target = Variable(target_batch, requires_grad=False).float().cuda()  # 标签

        data_in = Variable(data_batch, requires_grad=False).float() / 255
        data_in = data_in.cuda()  # [8,3,128,128]
        aux_label_batch = Variable(target_batch, requires_grad=False).long().cuda()
        one_hot_class = F.one_hot(aux_label_batch).float()

        data_in_lr = Variable(imgs_lr, requires_grad=False).float()
        data_in_lr = data_in_lr.cuda()  # [8,3,128,128]

        data_in_t = Variable(tacs, requires_grad=False).float()
        data_in_t = data_in_t.cuda()  # [8,3,128,128]


        encoder_lr_z, _ = encoder_lr(data_in_lr)  # [bsz, 512]
        encoder_t_z,  _ = encoder_t(data_in_t)  # [bsz, 512]

        encoder_lr_map = maper_lr(encoder_lr_z)  # [bsz, 128]
        encoder_t_map = maper_t(encoder_t_z)  # [bsz, 128]

        encoder_lr_map_logit = classifier(encoder_lr_map)  # [b,9]
        encoder_t_map_logit = classifier(encoder_t_map)  # [b,9]

        loss_triplet = TripletLoss()
        loss_cor_l = loss_triplet(encoder_lr_map, encoder_t_map, data_target)
        loss_cor_t = loss_triplet(encoder_t_map, encoder_lr_map, data_target)
        celoss = nn.CrossEntropyLoss()
        loss_fenlei_l = celoss(encoder_lr_map_logit, data_target.long())
        loss_fenlei_t = celoss(encoder_t_map_logit, data_target.long())

        loss_cor = loss_cor_l + loss_cor_t
        loss_fenlei = loss_fenlei_l + loss_fenlei_t
        loss_sum = loss_cor + loss_fenlei
        acc_ll = sumprescise(encoder_lr_map_logit, data_target)
        acc_l = acc_ll[0] / acc_ll[1]
        acc_tt = sumprescise(encoder_t_map_logit, data_target)
        acc_t = acc_tt[0] / acc_tt[1]
        print('val----bath_id[{0}/{1}],loss_sum:{8},loss_cor:{2},loss_cor_l:{3}, loss_cor_t:{4},loss_fenlei:{5},acc_l:{6},acc_t{7}'.format
            (j, len(val_loader), loss_cor, loss_cor_l, loss_cor_t, loss_fenlei, acc_l, acc_t, loss_sum))


    log_writer.add_scalar('val_loss_sum', loss_sum, step_index)
    log_writer.add_scalar('val_loss_cor', loss_cor, step_index)
    log_writer.add_scalar('val_loss_fenlei', loss_fenlei, step_index)

if __name__ == '__main__':
    run()