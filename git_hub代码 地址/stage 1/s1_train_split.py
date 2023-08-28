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
from auto_encoder import Decoder_HR_gen, Decoder_HR_dis, Encoder_HR, Encoder_HR_VGG
from my_dataset import MyDataSet, MyDataSet_tac_densenet
from utils import read_split_data, plot_data_loader_image, read_split_data_tac_densenet
from piqa import PSNR
from piqa import SSIM
from metric import psnr_oneimage, image_normal, sumprescise
import val_split as val
from torch.nn import init
import torch.nn as nn

'''stage 1'''

root = "D:/Desktop/12.11responce/LMT/NoFlash/Training"
root1 = "D:/Desktop/12.11responce/LMT/NoFlash/Training_LR"
#root2 = "D:/Desktop/12.11responce/LMT/Tapping/Training"
root2 = "D:/Desktop/12.11responce/LMT/Tapping_stft-32-8(128-128)"

path1 = 'D:/Desktop/12.11responce/path_encoder'
path2 = 'D:/Desktop/12.11responce/path_decoder'
path3 = 'D:/Desktop/12.11responce/path_discriminator'


def weigth_init(net):
    '''初始化网络'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight.data, std=1e-3)
            init.constant_(m.bias.data, 0)

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

    #z_size = 128
    #lr = 3e-4

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
    '''这两行是新的形势下如何读取数据，因为只需要输入一个就行了'''
    # train_images_lr_path, train_images_lr_label, val_images_lr_path, val_images_lr_label = 0, 0, 0, 0
    # train_tactile_path, train_tactile_label, val_tactile_path, val_tactile_label = 0, 0, 0, 0
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


    encoder = Encoder_HR_VGG().cuda()  #没问题

    generator = Decoder_HR_gen().cuda()
    discriminator = Decoder_HR_dis().cuda()
    #weigth_init(encoder())
    #weigth_init(generator())
   # weigth_init(discriminator())
    encoder_optimizer = Adam(params=encoder.parameters(), lr=lr*10, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    generator_optimizer = RMSprop(params=generator.parameters(), lr=lr*15, alpha=0.9, eps=1e-8, weight_decay=0,
                                  momentum=0, centered=False)
    discriminator_optimizer = RMSprop(params=discriminator.parameters(), lr=lr*8, alpha=0.9, eps=1e-8, weight_decay=0,
                                      momentum=0, centered=False)

    step_index = 0
    psnr_all = []
    for i in range(n_epochs):
        for j, (data_batch, imgs_lr, tacs, target_batch, filepath) in enumerate(train_loader):
            #train()
            data_target = Variable(target_batch, requires_grad=False).float().cuda() #标签
            # data_in = Variable(data_batch, requires_grad=False).float()/255
            # data_in = image_normal(data_in)
            '''输入不再进行归一化'''
            data_in = Variable(data_batch, requires_grad=False).float()
            data_in_hr = image_normal(data_in)
            data_in = data_in.cuda() #[8,3,128,128]

            aux_label_batch = Variable(target_batch, requires_grad=False).long().cuda()
            one_hot_class = F.one_hot(aux_label_batch).float()

            #train encoder
            for name, param in encoder.named_parameters():
                if "vgg16" in name:  # densenet121
                    param.requires_grad = False

            encoder_optimizer.zero_grad()
            encoder_hr_z, encoder_hr_z_logit = encoder(data_in) #[8,9]
            #print('encdor_hr_z_logit.shape:', encoder_hr_z_logit.shape)
            celoss = nn.CrossEntropyLoss()
            loss_encoder = celoss(encoder_hr_z_logit, data_target.long())
            loss_encoder.backward(retain_graph=True)
            encoder_optimizer.step()
            acc_ = sumprescise(encoder_hr_z_logit, data_target)
            acc = acc_[0]/acc_[1]

            # 更新鉴别器
            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for _ in range(5):
                # train_discrimianator
                encoder_hr_z, encoder_hr_z_logit = encoder(data_in)
                hr_gen = generator(encoder_hr_z).detach()
                #hr_gen = image_normal(hr_gen)
                hr_gen_out_1, hr_gen_out_2, hr_gen_out_3, hr_gen_out_4, hr_gen_out_5, hr_gen_out = discriminator(hr_gen)
                hr_real_out_1, hr_real_out_2, hr_real_out_3, hr_real_out_4, hr_real_out_5, hr_real_out = discriminator(data_in)
                gp = gradient_penalty(discriminator, data_in, hr_gen.detach())
                loss_dis = -torch.mean(hr_real_out) + torch.mean(hr_gen_out) + 100 * gp
                #print('DIS LOSS', 'hr_real_out:', torch.mean(hr_real_out), 'hr_gen_out:', torch.mean(hr_gen_out), 'gp:', gp)

                discriminator_optimizer.zero_grad()
                loss_dis.backward()
                discriminator_optimizer.step()
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # train generator,法1整个联合训练这个是
            for p in discriminator.parameters():
                p.requires_grad = False
            encoder_hr_z, encoder_hr_z_logit = encoder(data_in)
            hr_gen = generator(encoder_hr_z)
            hr_gen = image_normal(hr_gen)
            hr_gen_out_1, hr_gen_out_2, hr_gen_out_3, hr_gen_out_4, hr_gen_out_5, hr_gen_out = discriminator(hr_gen)

            # hr_gen_ceshi = hr_gen
            # hr_gen_ceshi = hr_gen_ceshi.cpu().detach().numpy()
            # print('hr_gen_out.shape:', hr_gen_out.shape, 'hr_gen_max:', np.max(hr_gen_ceshi), 'hr_gen_min:', np.min(hr_gen_ceshi))
            loss_mse = torch.nn.MSELoss(reduction='none')  # [bsz,3,128,128]
            #[8, 3, 128, 128]
            loss_gen_2_mse = loss_mse(hr_gen, data_in)  # 这个损失好像不对，一个是归一化的一个没有归一化的？？？
            loss_gen_2_mse = torch.sum(loss_gen_2_mse, dim=[1, 2, 3], keepdim=False).unsqueeze(-1)  # 把后面的几维相加，并保持第一维不变
            #print('hr_gen_out:', hr_gen_out, 'loss_gen_2_mse:', loss_gen_2_mse)
            loss_gen = -torch.mean(hr_gen_out) + lambda_gen_mse * torch.mean(loss_gen_2_mse)  # [8,1]+[8,1]
            #print('hr_gen_out:', torch.mean(hr_gen_out), 'loss_gen_2_mse:', torch.mean(loss_gen_2_mse))
            #loss_gen = torch.mean(loss_gen)

            generator_optimizer.zero_grad()
            loss_gen.backward()
            generator_optimizer.step()

            '''分开训练生成器的方法'''
            # for p in discriminator.parameters():
            #     p.requires_grad = False
            # encoder_hr_z, encoder_hr_z_logit = encoder(data_in)
            #
            # #train 生成器使用传统的gan损失；
            # generator_optimizer.zero_grad()
            # hr_gen = generator(encoder_hr_z)
            # #hr_gen = image_normal(hr_gen)
            # hr_gen_out_1, hr_gen_out_2, hr_gen_out_3, hr_gen_out_4, hr_gen_out_5, hr_gen_out = discriminator(hr_gen)
            # loss_1 = -torch.mean(hr_gen_out)
            # loss_1.backward(retain_graph=True)
            # generator_optimizer.step()
            # # train gan使用mse损失
            #
            # # hr_gen_ceshi = hr_gen
            # # hr_gen_ceshi = hr_gen_ceshi.cpu().detach().numpy()
            # # print('hr_gen_out.shape:', hr_gen_out.shape, 'hr_gen_max:', np.max(hr_gen_ceshi), 'hr_gen_min:', np.min(hr_gen_ceshi))
            # loss_mse = torch.nn.MSELoss(reduction='none')  # [bsz,3,128,128]
            # # [8, 3, 128, 128]
            # hr_gen = generator(encoder_hr_z)
            # #hr_gen = image_normal(hr_gen)
            #
            # loss_gen_2_mse = loss_mse(hr_gen, data_in)  # 这个损失好像不对，一个是归一化的一个没有归一化的？？？
            # loss_gen_2_mse = torch.sum(loss_gen_2_mse, dim=[1, 2, 3], keepdim=False).unsqueeze(-1)  # 把后面的几维相加，并保持第一维不变
            # # print('hr_gen_out:', hr_gen_out, 'loss_gen_2_mse:', loss_gen_2_mse)
            # loss_2 = torch.mean(loss_gen_2_mse)
            # generator_optimizer.zero_grad()
            # loss_2.backward()
            # generator_optimizer.step()
            #
            # loss_gen = -torch.mean(hr_gen_out) + lambda_gen_mse * torch.mean(loss_gen_2_mse)  # [8,1]+[8,1]
            # print('GEN LOSS', 'loss_gen:', loss_gen, 'hr_gen_out:', torch.mean(hr_gen_out), 'loss_gen_2_mse:', torch.mean(loss_gen_2_mse))


           #hr_gen_save = hr_gen*255
            hr_gen_save = hr_gen  #不进行归一化，直接去生成（0-255)之间的。
            #保存生成的图像：
            for k in range(len(hr_gen_save)):
                img1 = hr_gen_save[k, :, :, :].to("cpu").detach().numpy()
                img1 = img1.transpose(1, 2, 0)
               # print('img1.max', np.max(img1))
              #  print('img1.min', np.min(img1))
                filename = f"hr_gen/{filepath[k].split('.')[0]}_hr_gen.jpg"
                cv2.imwrite(filename, img1)
                img2 = np.array(data_batch[k, :, :, :])
                img2 = img2.transpose(1, 2, 0) #[0,255]中间
                #print('img2.max', np.max(img2))
                #print('img2.min', np.min(img2))
                filename = f"hr_gen/{filepath[k]}"
                cv2.imwrite(filename, img2)

                psnr_k = psnr_oneimage(img1, img2) #计算单张图像的psnr
                psnr_all.append(psnr_k)

            #print(hr_gen.shape, type(hr_gen), torch.max(hr_gen), torch.min(hr_gen))
            #print(data_in.shape, type(data_in), torch.max(data_in), torch.min(data_in))

            hr_gen_psnr = image_normal(hr_gen)
            data_in_psnr = image_normal(data_in)

            psnr_mean = psnr(hr_gen_psnr, data_in_psnr)
            ssim_mean = ssim(hr_gen_psnr.to('cpu'), data_in_psnr.to('cpu'))
            print('epoch[{0}/{1}],bathchid[{2}/{3}],loss_encoder:{4},acc:{10}, loss_gen:{5},loss_dis:{6},psnr_mean:{7},psnr_max{8},ssim_mean{9}'.format
                  (i, n_epochs, j, len(train_loader), loss_encoder.mean(), loss_gen.mean(), loss_dis.mean(), psnr_mean, max(psnr_all), ssim_mean, acc))

        val.run(encoder, generator, discriminator, val_loader, log_writer, step_index)
        log_writer.add_scalar('loss_encoder', loss_encoder, step_index)
        log_writer.add_scalar('loss_gen', loss_gen, step_index)
        log_writer.add_scalar('loss_dis', loss_dis, step_index)
        log_writer.add_scalar('acc_h', acc, step_index)
        step_index = step_index + 1
        if i%20 == 0:
            torch.save(encoder.state_dict(), path1)
            torch.save(generator.state_dict(), path2)
            torch.save(discriminator.state_dict(), path3)


        margin *= decay_margin
        equilibrium *= decay_equilibrium
        # margin non puo essere piu alto di equilibrium
        if margin > equilibrium:
            equilibrium = margin
        lambda_mse *= decay_mse
        if lambda_mse > 1:
            lambda_mse = 1
    log_writer.close()

        #step_index += 1
if __name__ == '__main__':
    train()