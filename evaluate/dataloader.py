#encoding=utf-8
from torch.utils import data
import torch
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pre_img_root, label_root):
        print("pre_img_root is {}".format(pre_img_root))
        print("label_gt_root is {}".format(label_root))
        # 这个排序有时候会有bug 不要乱改图片名称
        self.image_path = list(map(lambda x: os.path.join(pre_img_root, x), sorted(os.listdir(pre_img_root))))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), sorted(f for f in os.listdir(label_root) if  f.endswith('.png') and (f.find("edge") == -1))))
        if len(self.image_path) <= 0 :
            print("please check datasource path setting ! No image can be readed in ".format(pre_img_root))
            exit()
        if len(self.label_path) <= 0:
            print("please check datasource path setting ! No image can be readed in ".format(label_root))
            exit()

    def filter_files(self):
        print("images num is ", len(self.image_path))
        print("gts num is ", len(self.label_path))
        assert len(self.image_path) == len(self.label_path)

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        # print("get image path is {}".format(self.image_path[item]))
        # print("get gt path is {}".format(self.label_path[item]))
        # print(self.image_path[item], self.label_path[item])
        # 返回Image 图片名称以便于出现问题数据方便定位图片
        name = (self.image_path[item].split('/')[-1]).split(".")[0]
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt ,name

    def __len__(self):
        return len(self.image_path)


class validate_dataset():
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if  f.endswith('.png') and (f.find("edge") == -1)]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')