import numpy as np
import json
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class TensorDataset(Dataset):
    def __init__(self, data_path, usage, input_size):
        self.dataset_path = data_path
        self.usage = usage
        self.input_size = input_size
        self.Y = list()
        self.pics = list()
        for label, folder_name in enumerate(os.listdir(self.dataset_path)):
            folder_path = os.path.join(self.dataset_path, folder_name)
            for pic_name in os.listdir(folder_path):
                self.pics.append(os.path.join(folder_path, pic_name))
                self.Y.append(label)

        # 测试代码用
        # self.pics = self.pics[:500]
        # self.Y = self.Y[:500]

        np.random.seed(2016)
        n_examples = len(self.pics)
        n_train = n_examples * 0.8
        train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
        val_idx = list(set(range(0, n_examples)) - set(train_idx))

        self.pics, self.Y = np.array(self.pics), np.array(self.Y)
        self.X_train, self.Y_train = self.pics[train_idx], self.Y[train_idx]
        self.X_val, self.Y_val = self.pics[val_idx], self.Y[val_idx]

        with open('mean_std.json', 'r') as f:
            mean_std = json.load(f)

        means = mean_std['means']
        stdevs = mean_std['stdevs']
        print(means, stdevs)
        normalize = transforms.Normalize(mean=means, std=stdevs)

        self.transform = transforms.Compose(
            [transforms.CenterCrop(self.input_size), transforms.ToTensor(), normalize, ])

    #  重写后支持通过索引来使用第i个数据的样本 dataset[i]
    def __getitem__(self, index):
        if self.usage == 'train':
            x = Image.open(self.X_train[index])
            y = self.Y_train[index]
        elif self.usage == 'val':
            x = Image.open(self.X_val[index])
            y = self.Y_val[index]
        # width, height, _ = np.shape(x)
        x = self.transform(x).view(3, self.input_size, self.input_size)
        return x, y

    def __len__(self):
        if self.usage == 'train':
            return len(self.X_train)
        elif self.usage == 'val':
            return len(self.X_val)

