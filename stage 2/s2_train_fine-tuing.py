'''stage2, step3'''
import argparse
import numpy as np
import cv2
import torch

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from torch.autograd import Variable
from torch.optim import RMSprop, Adam, SGD
from torchvision import transforms
from my_dataset import MyDataSet, MyDataSet_new_liao
from s4_fusion_network import mapping_lr, mapping_t, classifier_common
from utils import read_split_data, plot_data_loader_image
from auto_encoder import Decoder_HR_gen, Decoder_HR_dis, Encoder_HR, Encoder_HR_VGG
from metric import psnr_oneimage, image_normal, sumprescise, TripletLoss
import val_split as val
from torch.nn import init
import torch.nn as nn
from s3_channel_model import channel
from torch import autograd
from piqa import PSNR
from piqa import SSIM
from metric import psnr_oneimage, image_normal, sumprescise
import val_split as val
from torch.nn import init
import torch.nn as nn


'''stage2, step1'''
root = "D:/Desktop/12.11responce/LMT/NoFlash/Training"
root1 = "D:/Desktop/12.11responce/LMT/NoFlash/Training_LR"
root2 = "D:/Desktop/12.11responce/LMT/Tapping/Training"

model_path_encoder_hr = 'D:/Desktop/12.11responce/path_encoder'
model_path_gen_hr_stage1 = 'D:/Desktop/12.11responce/path_decoder'
model_path_dis_hr_stage1 = 'D:/Desktop/12.11responce/path_discriminator'


model_path_encoder_lr = 'D:/Desktop/12.11responce/path_encoder_lr'
model_path_encoder_t = 'D:/Desktop/12.11responce/path_encoder_t'
model_path_maper_lr = 'D:/Desktop/12.11responce/path_maper_lr'
model_path_maper_t = 'D:/Desktop/12.11responce/path_maper_t'
model_path_fusion_generator = 'D:/Desktop/12.11responce/path_fusion_generator'  # 保存融合网络的生成器的地址


path1 = 'D:/Desktop/12.11responce/path_gen_fine_tuning'
path2 = 'D:/Desktop/12.11responce/path_dis_fine_tuning'
def gradient_penalty(D, xr, xf):
    t = torch.rand(len(xr), 1, 1, 1).cuda()
    t = t.expand_as(xr)
    mid = t*xr + ((1-t)*xf)
    mid.requires_grad_()
    pred = D(mid)[-1]

    grads = autograd.grad(
        outputs=pred, inputs=mid, grad_outputs=torch.ones_like(pred),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp




def train():
    n_epochs = 3000
    lr = 0.0001
    lambda_gen_mse = 1e-2

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
    train_images_lr_path, train_images_lr_label, val_images_lr_path, val_images_lr_label = 0, 0, 0, 0
    train_tactile_path, train_tactile_label, val_tactile_path, val_tactile_label = 0, 0, 0, 0
    #writer = SummaryWriter('./path/to/log')
    log_writer = SummaryWriter('./path/to/logs')
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(128),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(128),
                                   # transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet_new_liao(images_path=train_images_path,
                               images_lr_path=train_images_lr_path,
                               tactile_path=train_tactile_path,
                               images_class=train_images_label,
                               transform=None)
    val_data_set = MyDataSet_new_liao(images_path=val_images_path,
                             images_lr_path=val_images_lr_path,
                             tactile_path=val_tactile_path,
                             images_class=val_images_label,
                             transform=None)

    psnr = PSNR()
    ssim = SSIM()
    batch_size = 8
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
    encoder_lr = torch.load(model_path_encoder_lr)
    encoder_t = torch.load(model_path_encoder_t)
    maper_lr = torch.load(model_path_maper_lr)
    maper_t = torch.load(model_path_maper_t)
    fusion_G = torch.load(model_path_fusion_generator)

    encoder_lr.eval()
    encoder_t.eval()
    maper_lr.eval()
    maper_t.eval()
    fusion_G.eval()

    gen_hr = Decoder_HR_gen().cuda()
    dis_hr = Decoder_HR_dis().cuda()
    '''加载预训练的模型以供微调'''
    pretrained_dict_gen_hr = torch.load(model_path_gen_hr_stage1)
    gen_hr.load_state_dict(pretrained_dict_gen_hr)
    '''加载预训练的鉴别器模型以供微调'''
    pretrained_dict_dis_hr = torch.load(model_path_dis_hr_stage1)
    dis_hr.load_state_dict(pretrained_dict_dis_hr)

    generator_optimizer = RMSprop(params=gen_hr.parameters(), lr=lr * 15, alpha=0.9, eps=1e-8, weight_decay=0,
                                  momentum=0, centered=False)
    discriminator_optimizer = RMSprop(params=dis_hr.parameters(), lr=lr * 8, alpha=0.9, eps=1e-8, weight_decay=0,
                                      momentum=0, centered=False)

    step_index = 0
    psnr_all = []
    for i in range(n_epochs):
        for j, (data_batch, imgs_lr, tacs, target_batch, filepath) in enumerate(train_loader):
            # train()
            data_target = Variable(target_batch, requires_grad=False).float().cuda()  # 标签
            # data_in = Variable(data_batch, requires_grad=False).float()/255
            # data_in = image_normal(data_in)
            '''输入不再进行归一化'''
            data_in = Variable(data_batch, requires_grad=False).float()
            data_in = image_normal(data_in)

            data_in = data_in.cuda()  # [8,3,128,128]
            aux_label_batch = Variable(target_batch, requires_grad=False).long().cuda()
            one_hot_class = F.one_hot(aux_label_batch).float()

            data_in_lr = Variable(imgs_lr, requires_grad=False).float()
            data_in_lr = data_in_lr.cuda()  # [8,3,128,128]

            data_in_t = Variable(tacs, requires_grad=False).float()
            data_in_t = data_in_t.cuda()  # [8,3,128,128]

            '''编码网络'''
            encoder_lr_z, _ = encoder_lr(data_in_lr)  # [bsz, 512]
            encoder_t_z, _ = encoder_t(data_in_t)  # [bsz, 512]

            '''映射网络'''
            encoder_lr_map = maper_lr(encoder_lr_z)  # [bsz, 128]
            encoder_t_map = maper_t(encoder_t_z)  # [bsz, 128]

            '''通过信道模型'''
            channel__ = channel(channel_type='awagn', channel_snr=5)
            encoder_lr_map_noise = channel__(encoder_lr_map)  # [bsz, 128]
            encoder_t_map_noise = channel__(encoder_t_map)  # [bsz, 128]

            fusion_z = fusion_G(encoder_lr_map_noise, encoder_t_map_noise) #  [bsz, 128]

            # 加载之前的模型作为预训练的起点


            # 更新鉴别器
            for p in dis_hr.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for _ in range(5):
                # train_discrimianator

                hr_gen = gen_hr(fusion_z).detach()

                hr_gen_out_1, hr_gen_out_2, hr_gen_out_3, hr_gen_out_4, hr_gen_out_5, hr_gen_out = dis_hr(hr_gen)
                hr_real_out_1, hr_real_out_2, hr_real_out_3, hr_real_out_4, hr_real_out_5, hr_real_out = dis_hr(
                    data_in)
                gp = gradient_penalty(dis_hr, data_in, hr_gen.detach())
                loss_dis = -torch.mean(hr_real_out) + torch.mean(hr_gen_out) + 100 * gp
                print('DIS LOSS', 'hr_real_out:', torch.mean(hr_real_out), 'hr_gen_out:', torch.mean(hr_gen_out), 'gp:',
                      gp)

                discriminator_optimizer.zero_grad()
                loss_dis.backward()
                discriminator_optimizer.step()
                for p in dis_hr.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # train generator
            for p in dis_hr.parameters():
                p.requires_grad = False

            hr_gen = gen_hr(fusion_z)
            hr_gen = image_normal(hr_gen)
            hr_gen_out_1, hr_gen_out_2, hr_gen_out_3, hr_gen_out_4, hr_gen_out_5, hr_gen_out = dis_hr(hr_gen)


            # print('hr_gen_out.shape:', hr_gen_out.shape, 'hr_gen_max:', np.max(hr_gen_ceshi), 'hr_gen_min:', np.min(hr_gen_ceshi))
            loss_mse = torch.nn.MSELoss(reduction='none')  # [bsz,3,128,128]
            # [8, 3, 128, 128]
            loss_gen_2_mse = loss_mse(hr_gen, data_in)  # 这个损失好像不对，一个是归一化的一个没有归一化的？？？
            loss_gen_2_mse = torch.sum(loss_gen_2_mse, dim=[1, 2, 3], keepdim=False).unsqueeze(-1)  # 把后面的几维相加，并保持第一维不变
            # print('hr_gen_out:', hr_gen_out, 'loss_gen_2_mse:', loss_gen_2_mse)
            loss_gen = -torch.mean(hr_gen_out) + lambda_gen_mse * torch.mean(loss_gen_2_mse)  # [8,1]+[8,1]
            print('hr_gen_out:', torch.mean(hr_gen_out), 'loss_gen_2_mse:', torch.mean(loss_gen_2_mse))
            # loss_gen = torch.mean(loss_gen)

            generator_optimizer.zero_grad()
            loss_gen.backward()
            generator_optimizer.step()
            hr_gen_save = hr_gen  # 不进行归一化，直接去生成（0-255)之间的。
            # 保存生成的图像：
            for k in range(len(hr_gen_save)):
                img1 = hr_gen_save[k, :, :, :].to("cpu").detach().numpy()
                img1 = img1.transpose(1, 2, 0)
                # print('img1.max', np.max(img1))
                #  print('img1.min', np.min(img1))
                filename = f"hr_gen/{filepath[k].split('.')[0]}_hr_gen.jpg"
                cv2.imwrite(filename, img1)
                img2 = np.array(data_batch[k, :, :, :])
                img2 = img2.transpose(1, 2, 0)  # [0,255]中间
                # print('img2.max', np.max(img2))
                # print('img2.min', np.min(img2))
                filename = f"hr_gen/{filepath[k]}"
                cv2.imwrite(filename, img2)

                psnr_k = psnr_oneimage(img1, img2)  # 计算单张图像的psnr
                psnr_all.append(psnr_k)

            # print(hr_gen.shape, type(hr_gen), torch.max(hr_gen), torch.min(hr_gen))
            # print(data_in.shape, type(data_in), torch.max(data_in), torch.min(data_in))

            hr_gen_psnr = image_normal(hr_gen)
            data_in_psnr = image_normal(data_in)

            psnr_mean = psnr(hr_gen_psnr, data_in_psnr)
            ssim_mean = ssim(hr_gen_psnr.to('cpu'), data_in_psnr.to('cpu'))
            print(
                'epoch[{0}/{1}],bathchid[{2}/{3}] loss_gen:{4},loss_dis:{5},psnr_mean:{6},psnr_max{7},ssim_mean{8}'.format
                (i, n_epochs, j, len(train_loader), loss_gen.mean(), loss_dis.mean(), psnr_mean,
                 max(psnr_all), ssim_mean))

        #val.run(encoder, generator, discriminator, val_loader, log_writer, step_index)

        log_writer.add_scalar('loss_gen', loss_gen, step_index)
        log_writer.add_scalar('loss_dis', loss_dis, step_index)

        step_index = step_index + 1
        if i / 20 == 0:
            torch.save(gen_hr, path1)
            torch.save(dis_hr, path2)

    log_writer.close()

    # step_index += 1

if __name__ == '__main__':
    train()


