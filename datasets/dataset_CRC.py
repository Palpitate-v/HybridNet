import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

# 对图像和标签进行随机旋转和翻转操作
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)  # 生成从零到三的随机数
    image = np.rot90(image, k)  # 进行k次90度旋转
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()  # 用于沿特定轴翻转图像和标签。0：行  1：列
    label = np.flip(label, axis=axis).copy()
    return image, label


# 对图像和标签进行随机旋转
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# 椒盐噪声
def SaltAndPepper(image, noiseProbability):
    if image.ndim != 2:
        raise ValueError("Input image is not a 2D array.")

        # 行数 (height) 与列数 (width)
    rows, cols = image.shape
    # 根据概率计算噪声点的数量
    num_noise_pixels = int(noiseProbability * rows * cols)

    # 随机生成噪声点的坐标
    noise_ind = [np.random.randint(0, rows, size=num_noise_pixels),
                 np.random.randint(0, cols, size=num_noise_pixels)]

    # 在noise_image图像的对应坐标处添加椒盐噪声
    salty_peppered_image = image.copy()
    for i, (row, col) in enumerate(zip(*noise_ind)):
        # 对应二维坐标的添加，每一个噪声点盐(0)或胡椒(255)，根据随机数决定
        if np.random.random() < 0.5:
            salty_peppered_image[row][col] = 0
        else:
            salty_peppered_image[row][col] = 255

    return salty_peppered_image


def adjust_brightness(image, factor):
    # Apply brightness adjustment to all channels if the image is 2D
    image = image * factor
    return np.clip(image, 0, 255).astype('uint8')  # 确保值在0-255范围内，并转换为uint8类型


def darker(image, percetage):
    return adjust_brightness(image, 1 - percetage)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.dark_num=0
        self.noise=0

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)  # 旋转
        elif random.random() < 0.5:
            image, label = random_rotate(image, label)  # 翻转

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:  # 使用双线性插值法调整图像尺寸至所需输出尺寸。
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # Linear
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # Nearst
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(
            0)  # 将调整后的图像转换为 PyTorch 张量，并增加一个新的维度(batch)，以符合 PyTorch 对图像数据的批次维度要求。
        label = torch.from_numpy(label.astype(np.float32))  # 将标签转换为 Py
        sample = {'image': image, 'label': label.long()}
        return sample


class RandomGenerator_test(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']  # 从传入的样本中获取图像和标签数据。

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() < 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:  # 使用双线性插值法调整图像尺寸至所需输出尺寸。
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # Linear
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # Nearst
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(
            0)  # 将调整后的图像转换为 PyTorch 张量，并增加一个新的维度(batch)，以符合 PyTorch 对图像数据的批次维度要求。
        label = torch.from_numpy(label.astype(np.float32))  # 将标签转换为 Py
        sample = {'image': image, 'label': label.long()}
        return sample


class CRC_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()  # 获取样本名列表
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')  # 访问 -> 拼接路径
            data = np.load(data_path)  # 加载npz格式的数据
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npz".format(vol_name)
            data = np.load(filepath)
            image, label = data['image'], data['label']

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
