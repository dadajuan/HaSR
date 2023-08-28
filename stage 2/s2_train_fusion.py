
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from torch.autograd import Variable
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torchvision import transforms
from auto_encoder import Decoder_HR_gen, Decoder_HR_dis, Encoder_HR, Encoder_HR_VGG
from my_dataset import MyDataSet, MyDataSet_new_liao, MyDataSet_tac_densenet
from utils import read_split_data, plot_data_loader_image, read_split_data_tac_densenet
from auto_encoder import Decoder_HR_gen, Decoder_HR_dis, Encoder_HR_VGG, Encoder_Tac, Encoder_LR_VGG

from s4_fusion_network import Fusion_z_gen, Fusion_z_dis, mapping_lr, mapping_t, classifier_common
import val_fusion as val
from s3_channel_model import channel

'''stage2, step2'''

root = "D:/Desktop/12.11responce/LMT/NoFlash/Training"
root1 = "D:/Desktop/12.11responce/LMT/NoFlash/Training_LR"
#root2 = "D:/Desktop/12.11responce/LMT/Tapping/Training"
root2 = "D:/Desktop/12.11responce/LMT/Tapping_stft-32-8(128-128)"

model_path_encoder_hr = 'D:/Desktop/12.11responce/path_encoder_hr'   # 高分编码
model_path_encoder_lr = 'D:/Desktop/12.11responce/path_encoder_lr'   # 低分编码
model_path_encoder_t = 'D:/Desktop/12.11responce/path_encoder_t'    # 触觉编码
model_path_mapping_lr = 'D:/Desktop/12.11responce/path_maper_lr'      # 低分映射
model_path_mapping_t = 'D:/Desktop/12.11responce/path_maper_t'        # 触觉映射


path1 = 'D:/Desktop/12.11responce/path_fusion_generator'  # 保存融合网络的生成器的地址
path2 = 'D:/Desktop/12.11responce/path_fusion_discriminator'  # 保存融合网络的鉴别器的地址


def train():
    n_epochs = 3000
    lr = 0.0001

    '''使用我自己写的时候'''
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
    train_images_lr_path, train_images_lr_label, val_images_lr_path, val_images_lr_label = read_split_data(root1)
    train_tactile_path, train_tactile_label, val_tactile_path, val_tactile_label = read_split_data_tac_densenet(root2)

    '''使用小廖新写的时候'''
    #train_images_lr_path, train_images_lr_label, val_images_lr_path, val_images_lr_label = 0, 0, 0, 0
    #train_tactile_path, train_tactile_label, val_tactile_path, val_tactile_label = 0, 0, 0, 0

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

    encoder_hr = Encoder_HR_VGG()
    encoder_lr = Encoder_LR_VGG()
    encoder_t = Encoder_Tac()
    maper_lr = mapping_lr()
    maper_t = mapping_t()

    encoder_hr.load_state_dict(torch.load(model_path_encoder_hr))
    encoder_lr.load_state_dict(torch.load(model_path_encoder_lr))
    encoder_t.load_state_dict(torch.load(model_path_encoder_t))

    maper_lr.load_state_dict(torch.load(model_path_mapping_lr))
    maper_t.load_state_dict(torch.load(model_path_mapping_t))

    encoder_hr.eval()
    encoder_lr.eval()
    encoder_t.eval()
    maper_lr.eval()
    maper_t.eval()

    encoder_hr.cuda()
    encoder_lr.cuda()
    encoder_t.cuda()
    maper_lr.cuda()
    maper_t.cuda()


    fusion_G = Fusion_z_gen().cuda()
    fusion_D = Fusion_z_dis().cuda()


    fusion_G_optimizer = Adam(params=fusion_G.parameters(), lr=lr*10, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    fusion_D_optimizer = Adam(params=fusion_D.parameters(), lr=lr * 10, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    cuda = True if torch.cuda.is_available() else False


    step_index = 0
    for i in range(n_epochs):

        for j, (data_batch, imgs_lr, tacs, target_batch, filepath) in enumerate(train_loader):

            #train()

            data_in = Variable(data_batch, requires_grad=False).float()/255
            data_in = data_in.cuda() #[8,3,128,128]

            data_in_lr = Variable(imgs_lr, requires_grad=False).float()
            data_in_lr = data_in_lr.cuda()  # [8,3,128,128]

            data_in_t = Variable(tacs, requires_grad=False).float()
            data_in_t = data_in_t.cuda()  # [8,3,128,128]

            '''编码网络'''
            encoder_hr_z, _ = encoder_hr(data_in)  # 第二项其实没用了，只需要第一项作为真实的正样本 , 128维
            encoder_lr_z, _ = encoder_lr(data_in_lr)  # 512
            encoder_t_z, _ = encoder_t(data_in_t)  # 512
            '''映射网络'''
            encoder_lr_map = maper_lr(encoder_lr_z)  # 128
            encoder_t_map = maper_t(encoder_t_z)  # 128

            '''通过信道模型'''
            # channel__ = channel(channel_type='awagn', channel_snr=5)
            # encoder_lr_map_noise = channel__(encoder_lr_map)  # [bsz, 128]
            # encoder_t_map_noise = channel__(encoder_t_map)   # [bsz, 128]

            # valid = Variable(Tensor(imgs_lr.size(0), 1).fill_(1.0), requires_grad=False)
            # fake = Variable(Tensor(imgs_lr.size(0), 1).fill_(0.0), requires_grad=False)

            valid = torch.ones(encoder_hr_z.shape[0], 1).cuda()
            fake = torch.zeros(encoder_hr_z.shape[0], 1).cuda()

            # train generator
            for p in fusion_D.parameters():
                p.requires_grad = False

            fusion_G.zero_grad()
            '''无噪声的'''
            fusion_z = fusion_G(encoder_lr_map, encoder_t_map)
            '''加噪声的'''
            #fusion_z = fusion_G(encoder_lr_map_noise, encoder_t_map_noise)
            semantic_loss = torch.nn.MSELoss()
            semantic_loss2 = semantic_loss(fusion_z, encoder_hr_z)
            adversarial_loss = torch.nn.BCELoss()
            g_loss = adversarial_loss(fusion_D(fusion_z), valid) + semantic_loss2

            g_loss.backward(retain_graph=True)
            fusion_G_optimizer.step()

            # 更新鉴别器
            for p in fusion_D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train_discrimianator
            fusion_D_optimizer.zero_grad()
            fusion_z = fusion_G(encoder_lr_map, encoder_t_map)
            #fusion_z = fusion_G(encoder_lr_map_noise, encoder_t_map_noise)

            fusion_z_out = fusion_D(fusion_z)
            fusion_real_out = fusion_D(encoder_hr_z)
            adversarial_loss = torch.nn.BCELoss()
            real_loss = adversarial_loss(fusion_real_out, valid)
            fake_loss = adversarial_loss(fusion_z_out, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            fusion_D_optimizer.step()

            print('epoch[{0}/{1}],bathchid[{2}/{3}],loss_gen:{4},loss_dis:{5}'.format(i, n_epochs, j, len(train_loader), g_loss, d_loss))


        val.run(encoder_hr, encoder_lr, encoder_t, maper_lr, maper_t,val_loader, log_writer, step_index)
        #val.run(encoder, generator, discriminator, val_loader, log_writer, step_index)
        log_writer.add_scalar('g_loss', g_loss, step_index)
        log_writer.add_scalar('d_loss', d_loss, step_index)

        step_index = step_index + 1
        if i%10 == 0:
            torch.save(fusion_G.state_dict(), path1)
            torch.save(fusion_D.state_dict(), path2)

    log_writer.close()

if __name__ == '__main__':
    train()
