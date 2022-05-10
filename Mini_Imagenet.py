import csv
import os
from PIL import Image
import numpy as np
import paddle
from paddle.io import Dataset, Sampler
from paddle.vision import transforms as pdtransforms
from PIL import ImageEnhance
np.random.seed(10)
paddle.seed(10)
transformtypedict = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness,
                         Color=ImageEnhance.Color)


class ImageJitter(object):
    """
    作者用的数据增强之一
    """
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = paddle.rand([len(self.transforms)])
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out


class classifyDataSet(Dataset):
    def __init__(self, csvname='train.csv', imagepath='./images',csvpath='./datafile'):
        super().__init__()
        self.imagepath = imagepath
        self.csvpath = os.path.join(csvpath,csvname)

        self.transfer = pdtransforms.Compose([
            pdtransforms.RandomResizedCrop(84),
            ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
            pdtransforms.RandomHorizontalFlip(),
            pdtransforms.ToTensor(),
            pdtransforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])])
        with open(self.csvpath) as f:
            self.allfiles = [row for row in csv.reader(f)]
        self.allfiles = np.array(self.allfiles)
        # 把label转化为0~63对应的数字
        self.allfiles[:, 1] = np.arange(64).repeat(600)

        # 拿取所有的文件名,
        self.filelist = self.allfiles[:, 0].reshape(-1)
        # 所有的label
        self.label = self.allfiles.reshape((64, 600, 2))[:, :, 1].reshape(-1).astype(np.int64)


    def __len__(self):
        return self.filelist.shape[0]

    def __getitem__(self, item):
        # 此时的filelist和label是刚好一对一的，用产生的随机的item来读取对应的文件
        self.imagetensor = self.transfer(Image.open(os.path.join(self.imagepath, self.filelist[item])))
        return self.imagetensor, self.label[item]


class metaDataSet(Dataset):
    def __init__(self, part='train', imagepath='/images',csvpath='datafile'):
        super().__init__()

        self.imagefile = imagepath
        self.csvpath = os.path.join(csvpath, part+'.csv')  # 随机shuffle后的数据集
        if part == 'train':
            self.classnum = 64
        elif part == 'val':
            self.classnum = 16
        elif part == 'test':
            self.classnum = 20
        else:
            raise Exception('请输入正确的part! train , test or val')

        self.transfer = pdtransforms.Compose([pdtransforms.Resize((int(84 * 1.15), int(84 * 1.15))),
                                                pdtransforms.CenterCrop((84, 84)),
                                                pdtransforms.ToTensor(),
                                                pdtransforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])
        with open(self.csvpath) as f:
            self.allfiles = [row for row in csv.reader(f)]
        self.allfiles = np.array(self.allfiles)
        # 把label转化为数字
        self.allfiles[:, 1] = np.arange(self.classnum).repeat(600)

    def __len__(self):
        return self.allfiles.shape[0]

    def __getitem__(self, item):
        self.imagetensor = self.transfer(
            Image.open(os.path.join(self.imagefile, self.allfiles[item][0])).convert('RGB'))
        return self.imagetensor, self.allfiles[item][1].astype(np.int32)


class BatchSampler(Sampler):
    def __init__(self, part, Nway, Kshot, Qquarry, len=200, ep_per_batch=4):
        """
        Sampler的初始化函数，本类用于生成一个Sampler来传入dataloader，从而产生我需要的index
        :param Nway:身为小样本的Nway数
        :param Kshot:每way需要多少个训练样本Kshot张，默认前Kshot为训练
        :param Qquarry:每way需要多少个测试样本Qquarry张，默认前Kshot为训练
        :param len:最大迭代次数
        :param ep_per_batch:每回返回多少个metatask任务，如5way1shot15quarry共5*16=80为一个metatask任务
        """
        super().__init__()
        # classnum代表类别总数
        if part == 'train':
            self.classnum = 64
        elif part == 'val':
            self.classnum = 16
        elif part == 'test':
            self.classnum = 20
        else:
            raise Exception('请输入正确的part! train , test or val')

        self.labelidx = np.arange(self.classnum * 600).reshape((self.classnum, 600))
        self.Nway = Nway
        self.Kshot = Kshot
        self.Qquarry = Qquarry
        self.len = len
        self.ep_per_batch = ep_per_batch

    def __len__(self):
        return self.len

    def __iter__(self):
        for _ in range(self.len):
            metabatch = [None] * self.ep_per_batch
            for i in range(self.ep_per_batch):
                batch = [None] * self.Nway
                # 无放回的取Nway个task
                clssmp = np.random.choice(np.arange(self.classnum), self.Nway, False)
                for j, cls in enumerate(clssmp):
                    support = np.random.choice(self.labelidx[cls][:500], self.Kshot, False)
                    quarry = np.random.choice(self.labelidx[cls][500:], self.Qquarry, False)
                    # 把support和quarry进行拼接,得到这个任务的support和quarry下标集合
                    batch[j] = np.concatenate((support, quarry))
                metabatch[i] = batch
            yield np.array(metabatch).reshape(-1).squeeze()

