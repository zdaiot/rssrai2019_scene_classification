import os
from PIL import Image
import numpy as np
import torch
import json

X, Y = list(), list()
dataset_path = './datasets/train'
for label, folder_name in enumerate(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder_name)
    for pic_name in os.listdir(folder_path):
        # 方式一读取
        # temp = Image.open(os.path.join(folder_path, pic_name))
        # x = temp.copy()
        # temp.close()

        # 方式二读取,因为下面计算方法要使用到numpy.所以这样,
        f = open(os.path.join(folder_path, pic_name), 'rb')
        x = Image.open(f)
        x = np.array(x.convert('RGB'), dtype=np.uint8)
        # x = x / 255.0

        X.append(x)  # 注意不能用 cv2, 否则的话,transforms.CenterCrop(224)报错
        Y.append(label)

        # 测试代码用
        # X = X[:2000]
        # Y = Y[:2000]
# 所有的计算数据维度均是(num_samples,rows,cols,channels)

X = np.array(X) # .astype(np.float32),去掉,否则特别耗费内存
#%%
# 方法一,这样计算stdevs是错的
# X = torch.from_numpy(X).float()
# means = X.mean(dim=1).mean(dim=1).mean(dim=0) / 255.0
# stdevs = X.std(dim=1).std(dim=1).std(dim=0) / 255.0
# print(means, stdevs)

# 方法二，输入numpy，因为transforms.ToTensor()具有归一化功能，所以除以255,内存消耗完报错
# X = X.numpy()
# means = np.mean(X, axis=(0, 1, 2)) / 255.0
# stdevs = np.std(X, axis=(0, 1, 2)) / 255.0
# print(means, stdevs)

# 方法三,可用,内存耗费完，输入numpy
# [0.36801731246914998, 0.3809775213997274, 0.34358175712749295]
# [0.20352853997255294, 0.18543288302554292, 0.18488177291766691]
means = []
stdevs = []
for i in range(3):
    pixels = X[:, :, :, i].ravel()
    means.append(np.round(np.mean(pixels)/255.0,2))
    stdevs.append(np.round(np.std(pixels)/255.0,2))
print(means, stdevs)
mean_std = {'means':means, 'stdevs':stdevs}
f = open("mean_std.json", "w")
json.dump(mean_std, f)
f.close()

