import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from metric import psnr_oneimage, image_normal, sumprescise
from piqa import PSNR
from piqa import SSIM

def run(encoder,val_loader, log_writer, step_index):
    for j, (data_batch, imgs_lr, tacs, target_batch, filepath) in enumerate(val_loader):
        #train()

        data_target = Variable(target_batch, requires_grad=False).float().cuda() #标签

        data_in_hr = Variable(data_batch, requires_grad=False).float()/255
        data_in_hr = data_in_hr.cuda() #[8,3,128,128]

        #train encoder
        encoder_hr_z, encoder_hr_z_logit = encoder(data_in_hr)
        loss_encoder = F.cross_entropy(encoder_hr_z_logit, data_target.long())
        acc_ = sumprescise(encoder_hr_z_logit, data_target)
        acc = acc_[0] / acc_[1]
        print('val-encoder-hr  bathchid[{0}/{1}],loss_encoder:{2},acc_encoder-hr:{3} '.format( j, len(val_loader), loss_encoder.mean(), acc))

    log_writer.add_scalar('val_loss_encoder', loss_encoder, step_index)
    log_writer.add_scalar('val_acc_encoder', acc, step_index)
if __name__ == '__main__':
    run()