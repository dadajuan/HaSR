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


root = "LMT/LMT/NoFlash/Training"
root1 = "LMT/LMT/NoFlash/Training_LR"
root2 = "LMT/LMT/Tapping/Training"
def run(encoder,generator,discriminator,val_loader,log_writer,step_index):
    lambda_gen_mse = 1e-8
    # log_writer = SummaryWriter('./path/to/logs1')
    # step_index = 0
    psnr_val_all = []
    psnr = PSNR()
    ssim = SSIM()
    psnr_val_all = []
    for j, (data_batch, imgs_lr, tacs, target_batch, filepath) in enumerate(val_loader):
        #train()
        data_target = Variable(target_batch, requires_grad=False).float().cuda() #标签
        data_in = Variable(data_batch, requires_grad=False).float()/255
        data_in = data_in.cuda() #[8,3,128,128]

        #train encoder
        encoder_hr_z, encoder_hr_z_logit = encoder(data_in)
        loss_encoder = F.cross_entropy(encoder_hr_z_logit, data_target.long())
        acc = sumprescise(encoder_hr_z_logit, data_target)


        encoder_hr_z, encoder_hr_z_logit = encoder(data_in)
        hr_gen = generator(encoder_hr_z)
        hr_gen_out = discriminator(hr_gen)[-1]
        loss_mse = torch.nn.MSELoss(reduction='none')  #[bsz,3,128,128]
        loss_gen_2_mse = loss_mse(hr_gen, data_in)
        loss_gen_2_mse = torch.sum(loss_gen_2_mse, dim=[1, 2, 3], keepdim=False).unsqueeze(-1)  # 把后面的几维相加，并保持第一维不变
        loss_gen = -torch.mean(hr_gen_out) + lambda_gen_mse * torch.mean(loss_gen_2_mse)  #[8,1]+[8,1]

        hr_gen = generator(encoder_hr_z)
        hr_gen_out = discriminator(hr_gen)[-1]
        hr_real_out = discriminator(data_in)[-1]
        loss_dis = -torch.mean(hr_real_out) + torch.mean(hr_gen_out)

        #保存生成的图像：
        #hr_gen_save = image_normal(hr_gen) * 255

        hr_gen_save = hr_gen
        # 保存生成的图像：
        for k in range(len(hr_gen_save)):
            img1 = hr_gen_save[k, :, :, :].to("cpu").detach().numpy()
            img1 = img1.transpose(1, 2, 0)
            #print('img1.max', np.max(img1))
            #print('img1.min', np.min(img1))
            filename = f"hr_gen/val_{filepath[k].split('.')[0]}_hr_gen.jpg"
            cv2.imwrite(filename, img1)
            img2 = np.array(data_batch[k, :, :, :])
            img2 = img2.transpose(1, 2, 0)  # [0,255]中间
            filename = f"hr_gen/val_{filepath[k]}"
            cv2.imwrite(filename, img2)
            psnr_k = psnr_oneimage(img1, img2 )  # 计算单张图像的psnr
            psnr_val_all.append(psnr_k)

        hr_gen_psnr = image_normal(hr_gen)
        data_in_psnr = image_normal(data_in)
        psnr_mean = psnr(hr_gen_psnr, data_in_psnr)
        ssim_mean = ssim(hr_gen_psnr.to('cpu'), data_in_psnr.to('cpu'))

        print('val  bathchid[{0}/{1}],loss_encoder:{2},acc_encoder:{8} loss_gen:{3},loss_dis:{4}, psnr_mean:{5},psnr_max:{6}, ssim:{7}'.format( j, len(val_loader), loss_encoder.mean(),
                                                                loss_gen.mean(), loss_dis.mean(), psnr_mean,max(psnr_val_all), ssim_mean,acc))

    log_writer.add_scalar('val_loss_dis', loss_encoder, step_index)
    log_writer.add_scalar('val_loss_encoder', loss_gen, step_index)
    log_writer.add_scalar('val_loss_gen', loss_dis, step_index)
    #log_writer.add_scalar('val_acc_encoder', acc, step_index)
if __name__ == '__main__':
    run()