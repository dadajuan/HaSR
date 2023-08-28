from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

class MyDataSet_normal(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
    #该数据集下所有样本的个数
    def __len__(self):
        return len(self.images_path) #计算该列表的元素个数

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    #静态方法
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)  #image变成（8,3,224,224）
        labels = torch.as_tensor(labels)     #labels转化为tensot（
        return images, labels
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_lr_path: list, tactile_path: list,  images_class: list, transform=None):
        self.images_path = images_path
        self.images_lr_path = images_lr_path
        self.tactile_path = tactile_path
        self.images_class = images_class
        self.transform = transform
    #该数据集下所有样本的个数
    def __len__(self):
        return len(self.images_path) #计算该列表的元素个数

    def __getitem__(self, item):
        #查看高分、触觉、低分是否匹配，读取之后的数据集的前面的名字和后面的名字是不是一致的都；
        if self.images_path[item].split("\\")[-1] != self.images_lr_path[item].split("\\")[-1]:
            print("!!!!存在不匹配!!!!")
        if self.images_path[item].split("\\")[-1].split("_")[0] != self.tactile_path[item].split("\\")[-1].split("_")[0] or \
                self.images_path[item].split("\\")[-1].split("Image_")[1].split("_")[0] != self.tactile_path[item].split("\\")[-1].split("Z_")[1].split("n")[1].split(".")[0]:
            print("!!!!存在不匹配!!!!")
        # img = Image.open(self.images_path[item])
        # img_lr = Image.open(self.images_lr_path[item])
        img = cv2.imread(self.images_path[item])
        img= img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img_lr = cv2.imread(self.images_lr_path[item])
        img_lr = img_lr.transpose(2, 0, 1)
        img_lr = torch.from_numpy(img_lr)
        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        tac = open(self.tactile_path[item]).readlines()
        tac = list(map(float, tac)) #把list里面的每个str类型的元素转化成float，然后再拼接成list
        tac = torch.Tensor(tac)
        if self.transform is not None:
            img = self.transform(img)
        if self.transform is not None:
            img_lr = self.transform(img_lr)
        if self.transform is not None:
            tac = self.transform(tac)
        file=self.images_path[item].split("\\")[-1]
        #print(file)
        return img, img_lr, tac,  label, file

    @staticmethod
    #静态方法
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        #print((batch))
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, imgs_lr, tacs, labels, filepath = tuple(zip(*batch))

        images = torch.stack(images, dim=0)  #image变成（8,3,224,224）
        imgs_lr = torch.stack(imgs_lr, dim=0)
        tacs = torch.stack(tacs, dim=0)
        labels = torch.as_tensor(labels)     #labels转化为tensor
        return images, imgs_lr, tacs, labels, filepath
class MyDataSet_test(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_lr_path: list, tactile_path: list,  images_class: list, transform=None):
        self.images_path = images_path
        self.images_lr_path = images_lr_path
        self.tactile_path = tactile_path
        self.images_class = images_class
        self.transform = transform
    #该数据集下所有样本的个数
    def __len__(self):
        return len(self.images_path) #计算该列表的元素个数

    def __getitem__(self, item):
        #查看高分、触觉、低分是否匹配，读取之后的数据集的前面的名字和后面的名字是不是一致的都；
        if self.images_path[item].split("\\")[-1] != self.images_lr_path[item].split("\\")[-1]:
            print("!!!!存在不匹配!!!!")
        if self.images_path[item].split("\\")[-1].split("_")[0] != self.tactile_path[item].split("\\")[-1].split("_")[0] or \
                self.images_path[item].split("\\")[-1].split("Image_")[1].split("_")[0] != self.tactile_path[item].split("\\")[-1].split("Z_")[1].split("est")[1].split(".")[0]:
            print("!!!!存在不匹配!!!!")
        # img = Image.open(self.images_path[item])
        # img_lr = Image.open(self.images_lr_path[item])
        img = cv2.imread(self.images_path[item])
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img_lr = cv2.imread(self.images_lr_path[item])
        img_lr = img_lr.transpose(2, 0, 1)
        img_lr = torch.from_numpy(img_lr)
        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        tac = open(self.tactile_path[item]).readlines()
        tac = list(map(float, tac)) #把list里面的每个str类型的元素转化成float，然后再拼接成list
        tac = torch.Tensor(tac)
        if self.transform is not None:
            img = self.transform(img)
        if self.transform is not None:
            img_lr = self.transform(img_lr)
        if self.transform is not None:
            tac = self.transform(tac)
        file=self.images_path[item].split("\\")[-1]
        #print(file)
        return img, img_lr, tac,  label, file

    @staticmethod
    #静态方法
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        #print((batch))
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, imgs_lr, tacs, labels, filepath = tuple(zip(*batch))

        images = torch.stack(images, dim=0)  #image变成（8,3,224,224）
        imgs_lr = torch.stack(imgs_lr, dim=0)
        tacs = torch.stack(tacs, dim=0)
        labels = torch.as_tensor(labels)     #labels转化为tensor
        return images, imgs_lr, tacs, labels, filepath

class MyDataSet_new_liao(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_lr_path: list, tactile_path: list,  images_class: list, transform=None):
        self.images_path = images_path
        self.images_lr_path = images_lr_path
        self.tactile_path = tactile_path
        self.images_class = images_class
        self.transform = transform
    #该数据集下所有样本的个数
    def __len__(self):
        return len(self.images_path) #计算该列表的元素个数

    def __getitem__(self, item):

        imgfile = self.images_path[item].split("\\")[-1]
        img = cv2.imread(self.images_path[item])
        img = img.transpose(2, 0, 1)
        # cv2.imshow("ds",img.transpose(1, 2, 0))
        # cv2.waitKey(0)
        img = torch.from_numpy(img)
        g = self.images_path[item].split('\\')[1]
        lrfile=self.images_path[item].split("Training")[0]+"Training_LR"+self.images_path[item].split("Training")[1]
        #taclrfile=f"LMT/LMT/Tapping/Training/{g}/"+imgfile.split("Image")[0]+f"Tapping_Z_train{imgfile.split('Image_')[1].split('_')[0]}.jpg"
        img_lr = cv2.imread(lrfile)
        img_lr = img_lr.transpose(2, 0, 1)
        img_lr = torch.from_numpy(img_lr)
        # tacfile = f"LMT/LMT/Tapping/Training/{g}/" + imgfile.split("Image")[0] + f"Tapping_Z_train{imgfile.split('Image_')[1].split('_')[0]}.jpg"
        tacfile = f"D:/Desktop/12.11responce/LMT/Training_stft/{g}/" + imgfile.split("Image")[0] + f"Tapping_Z_train{imgfile.split('Image_')[1].split('_')[0]}.jpg"

        # f是format的形式的意思。

        tac = cv2.imread(tacfile)
        tac = tac.transpose(2, 0, 1)
        tac = torch.from_numpy(tac)

        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        g = self.images_path[item].split('\\')[1]
        '''触觉使用的是txt的情况'''
        # tacfile=f"D:/Desktop/12.11responce/LMT/Tapping/Training/{g}/"+imgfile.split("Image")[0]+f"Tapping_Z_train{imgfile.split('Image_')[1].split('_')[0]}.txt"
        # tac = open(tacfile).readlines()
        # tac = list(map(float, tac)) #把list里面的每个str类型的元素转化成float，然后再拼接成list
        # tac = torch.Tensor(tac)
        if self.transform is not None:
            img = self.transform(img)
        if self.transform is not None:
            img_lr = self.transform(img_lr)
        if self.transform is not None:
            tac = self.transform(tac)
        file = self.images_path[item].split("\\")[-1]
        #print(file)
        return img, img_lr, tac,  label, file

    @staticmethod
    #静态方法
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        #print((batch))
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, imgs_lr, tacs, labels, filepath = tuple(zip(*batch))

        images = torch.stack(images, dim=0)  #image变成（8,3,224,224）
        imgs_lr = torch.stack(imgs_lr, dim=0)
        tacs = torch.stack(tacs, dim=0)
        labels = torch.as_tensor(labels)     #labels转化为tensor
        return images, imgs_lr, tacs, labels, filepath
class MyDataSet_tac_densenet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_lr_path: list, tactile_path: list,  images_class: list, transform=None):
        self.images_path = images_path
        self.images_lr_path = images_lr_path
        self.tactile_path = tactile_path
        self.images_class = images_class
        self.transform = transform
    #该数据集下所有样本的个数
    def __len__(self):
        return len(self.images_path) #计算该列表的元素个数

    def __getitem__(self, item):
        #查看高分、触觉、低分是否匹配，读取之后的数据集的前面的名字和后面的名字是不是一致的都；
        if self.images_path[item].split("\\")[-1] != self.images_lr_path[item].split("\\")[-1]:
            print("!!!!存在不匹配!!!!")
        if self.images_path[item].split("\\")[-1].split("_")[0] != self.tactile_path[item].split("\\")[-1].split("_")[0] or \
                self.images_path[item].split("\\")[-1].split("Image_")[1].split("_")[0] != self.tactile_path[item].split("\\")[-1].split("Z_")[1].split("n")[1].split(".")[0]:
            print("!!!!存在不匹配!!!!")
        # img = Image.open(self.images_path[item])
        # img_lr = Image.open(self.images_lr_path[item])
        img = cv2.imread(self.images_path[item])
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        img_lr = cv2.imread(self.images_lr_path[item])
        img_lr = img_lr.transpose(2, 0, 1)
        img_lr = torch.from_numpy(img_lr)

        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        label = self.images_class[item]
        '''当触觉信号使用的stft变换之后的图像时候'''
        img_tac = cv2.imread(self.tactile_path[item])
        img_tac = img_tac.transpose(2, 0, 1)
        tac = torch.from_numpy(img_tac)

        '''当触觉信号是使用512维的序列的时候的数据'''
        # tac = open(self.tactile_path[item]).readlines()
        # tac = list(map(float, tac)) #把list里面的每个str类型的元素转化成float，然后再拼接成list
        # tac = torch.Tensor(tac)

        if self.transform is not None:
            img = self.transform(img)
        if self.transform is not None:
            img_lr = self.transform(img_lr)
        if self.transform is not None:
            tac = self.transform(tac)
        file = self.images_path[item].split("\\")[-1]
        #print(file)
        return img, img_lr, tac,  label, file

    @staticmethod
    #静态方法
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        #print((batch))
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, imgs_lr, tacs, labels, filepath = tuple(zip(*batch))

        images = torch.stack(images, dim=0)  #image变成（8,3,224,224）
        imgs_lr = torch.stack(imgs_lr, dim=0)
        tacs = torch.stack(tacs, dim=0)
        labels = torch.as_tensor(labels)     #labels转化为tensor
        return images, imgs_lr, tacs, labels, filepath