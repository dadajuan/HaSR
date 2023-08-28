
import torch
from torch.autograd import Variable
from s3_channel_model import channel
from s4_fusion_network import Fusion_z_gen, Fusion_z_dis, mapping_lr, mapping_t, classifier_common

def run (encoder_hr, encoder_lr, encoder_t, maper_lr, maper_t,val_loader, log_writer, step_index):

    for j, (data_batch, imgs_lr, tacs, target_batch, filepath) in enumerate(val_loader):
        #train()
        #data_target = Variable(target_batch, requires_grad=False).float().cuda() #标签
        data_in = Variable(data_batch, requires_grad=False).float()/255  # 高分
        data_in = data_in.cuda() #[8,3,128,128]

        data_in_lr = Variable(imgs_lr, requires_grad=False).float()
        data_in_lr = data_in_lr.cuda()  # [8,3,128,128]

        data_in_t = Variable(tacs, requires_grad=False).float()
        data_in_t = data_in_t.cuda()  # [8,3,128,128]

        #加载高分,低分,触觉的编码网络
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
        # encoder_t_map_noise = channel__(encoder_t_map)  # [bsz, 128]

        '''生成融合网络'''
        valid = torch.ones(encoder_hr_z.shape[0], 1).cuda()
        fake = torch.zeros(encoder_hr_z.shape[0], 1).cuda()

        fusion_G = Fusion_z_gen().cuda()
        fusion_D = Fusion_z_dis().cuda()

        fusion_z = fusion_G(encoder_lr_map, encoder_t_map)
        #fusion_z = fusion_G(encoder_lr_map_noise, encoder_t_map_noise)
        semantic_loss = torch.nn.MSELoss()
        semantic_loss2 = semantic_loss(fusion_z, encoder_hr_z)
        adversarial_loss = torch.nn.BCELoss()

        g_loss = adversarial_loss(fusion_D(fusion_z), valid) + semantic_loss2

        fusion_z_out = fusion_D(fusion_z)
        fusion_real_out = fusion_D(encoder_hr_z)
        adversarial_loss = torch.nn.BCELoss()
        real_loss = adversarial_loss(fusion_real_out, valid)
        fake_loss = adversarial_loss(fusion_z_out, fake)
        d_loss = (real_loss + fake_loss) / 2


        print('val  bathchid[{0}/{1}],loss_gen:{2}, loss_dis:{3}'.format( j, len(val_loader), g_loss.mean(), d_loss.mean()))

    log_writer.add_scalar('val_loss_gen', g_loss, step_index)
    log_writer.add_scalar('val_loss_dis', d_loss, step_index)

if __name__ == '__main__':
    run()