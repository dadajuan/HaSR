import torch
import numpy as np

import torch.nn as nn

############################信道######################################
def real_awgn(x, stddev):
    #awgan = torch.randn(x.shape, 0, stddev, dtype = torch.float32)
    awgan = np.random.normal(0, stddev, size=x.shape)
    x = x.detach().numpy()
    y = x + awgan
    return y

def fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        h = torch.complex(
            torch.from_numpy(np.random.normal(0, 1 / np.sqrt(2), size=[x.shape[0], 1])),
            torch.from_numpy(np.random.normal(0, 1 / np.sqrt(2), size=[x.shape[0], 1])))

    # additive white gaussian noise
    awgn = torch.complex(
        torch.from_numpy(np.random.normal(0, 1 / np.sqrt(2), size=x.shape)),
        torch.from_numpy(np.random.normal(0, 1 / np.sqrt(2), size=x.shape)))
    return (h * x + stddev * awgn), h  # awgn的均值是0，方差是stddev/np.sqrt（2）

def phase_invariant_fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise. Also assumes that phase shift
    introduced by the fading channel is known at the receiver, making
    the model equivalent to a real slow fading channel.

    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        n1 = np.random.normal(0, 1 / np.sqrt(2), size=[x.shape[0], 1])
        n2 = np.random.normal(0, 1 / np.sqrt(2), size=[x.shape[0], 1])
        n1 = torch.from_numpy(n1)
        n2 = torch.from_numpy(n2)
        h = torch.sqrt(torch.square(n1) + torch.square(n2))

    # additive white gaussian noise
    awgn = np.random.normal(0, stddev / np.sqrt(2), size=x.shape)

    return (h * x + awgn), h

class channel(nn.Module):
    def __init__(self, channel_type, channel_snr):
        super(channel, self).__init__()
        self.channel_type = channel_type
        self.channel_snr = channel_snr


    def forward(self, z):

        noise_stddev = np.sqrt(10 ** (-self.channel_snr / 10))
        prev_h = None
        # 加性高斯白噪声
        if self.channel_type =='awagn':
            dim_z = z.shape[1]  #int型数据
            dim_z = torch.tensor(dim_z)
           # print(dim_z.type)
            ########## PyTorch Version 2 ################
            z_norm_th = torch.nn.functional.normalize(z, p=2, dim=1)  # 保留的位数与tf并不太一样
            #z_in = torch.sqrt(dim_z.to(torch.float32)) * z_norm_th  # 为啥乘以前面这个还？
            z_in = torch.sqrt(dim_z) * z_norm_th  # 为啥乘以前面这个还？

            print(type(z_in))
            z_out = real_awgn(z_in, noise_stddev)
            h = torch.ones_like(z_in)   # h just makes sense on fading channels

        elif self.channel_type == "fading":
            dim_z = z.shape[1]//2
            dim_z = torch.tensor(dim_z)
            z_in = torch.complex(z[:, :dim_z], z[:, dim_z:])
            z_norm = torch.sum(
                torch.real(z_in * torch.conj(z_in)), keepdim=True, dim=1
            )
            z_in = z_in * torch.complex(
                torch.sqrt(dim_z / z_norm), torch.tensor(0.0)
            )
            z_out, h = fading(z_in, noise_stddev, prev_h)
            # convert back to real
            z_out = torch.cat([torch.real(z_out), torch.imag(z_out)], 1)

        elif self.channel_type == "fading-real":

            dim_z = z.shape[1]//2
            dim_z = torch.tensor(dim_z)
            z_norm_th = torch.nn.functional.normalize(z, p=2, dim=1)
            z_in = torch.sqrt(dim_z) * z_norm_th
            z_out, h = phase_invariant_fading(z_in, noise_stddev, prev_h)
        else:
            raise Exception("This option shouldn't be an option!")

        #avg_power = torch.mean(torch.real(z_in * torch.conj(z_in)))  # 计算所有元素的平均
        #return z_out, avg_power, h
        return z_out

def main():
    tmp = torch.rand(8, 128)
    channel_model = channel(channel_type='awagn', channel_snr=5)
    tmp_n = channel_model(tmp)
    print('tmp_n:', tmp_n, 'tmp_n.shape:', tmp_n.shape)
   # print('tmp_n-tmp:', tmp_n-tmp)

if __name__ == '__main__':
    main()





