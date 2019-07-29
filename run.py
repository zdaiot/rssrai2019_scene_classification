# -*-coding:utf-8 -*-
import random
import argparse
import os
import json
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import models.imagenet as customized_models
import torchvision.models as models
from solver import run, load_model, submit
from pprint import pprint
import pickle

# Models
# name中若为小写且不以‘——’开头，则对其进行升序排列，
# callable功能为判断返回对象是否可调用（即某种功能）。
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

model_names = default_model_names + customized_models_names
print('All optional models:', model_names)

use_paras = False
if use_paras:
    with open('./checkpoint/' + "params.json", 'r', encoding='utf-8') as json_file:
        state = json.load(json_file)
        json_file.close()
else:
    # Parse arguments
    parser = argparse.ArgumentParser(description='PyTorch Rssrai Training')

    # Datasets
    parser.add_argument('--train_path', help='path to train dataset', default='./datasets/train', type=str)
    parser.add_argument('--val_path', help='path to val dataset', default='./datasets/val', type=str)
    parser.add_argument('--test_path', help='path to test dataset', default='./datasets/test', type=str)
    parser.add_argument('-s', '--image_size', help='the size of the model input', default=224, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Optimization options
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train_batch', default=4, type=int, metavar='N',
                        help='train batchsize (default: 32)')
    parser.add_argument('--val_batch', default=4, type=int, metavar='N',
                        help='val batchsize (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--drop', '--dropout', default=0.5, type=float,
                        metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 40],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnext50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--num-classes', type=int, help='the number of classes', default=5) # TODO
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
    parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')  # dest是存储的变量
    # Device options
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='For example 0,1 to run on two GPUs')

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    pprint(state)

    if not os.path.exists(state['checkpoint']):
        os.makedirs(state['checkpoint'])

    with open(state['checkpoint'] + '/params.json', 'w') as json_file:
        json.dump(vars(args), json_file, ensure_ascii=False)
    json_file.close()

# cuda设置
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True
if len(state['gpu_id']) > 1:
    gpu_id = list(map(int, state['gpu_id'].split(',')))
else:
    gpu_id = None
os.environ['CUDA_VISIBLE_DEVICES'] = state['gpu_id']

# Random seed
if state['manualSeed'] is None:
    state['manualSeed'] = random.randint(1, 10000)
random.seed(state['manualSeed'])
torch.manual_seed(state['manualSeed'])
if use_cuda:
    torch.cuda.manual_seed_all(state['manualSeed'])

# 读取数据
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# train_data = TensorDataset(state['data'], 'train', state['size'])
# val_data = TensorDataset(state['data'], 'val', state['size'])

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean,std=std)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(state['image_size']),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # 将图片转换为Tensor,归一化至[0,1]
    normalize
])
val_transform = transforms.Compose([
    transforms.RandomResizedCrop(state['image_size']),
    transforms.ToTensor(), # 将图片转换为Tensor,归一化至[0,1]
    normalize
])

train_data = ImageFolder(state['train_path'], transform=train_transform)
val_data = ImageFolder(state['val_path'], transform=val_transform)

# 存储文件夹和类标的映射关系
output = open(state['checkpoint'] + '/label_encode.pkl', 'wb')
pickle.dump(train_data.class_to_idx, output)
assert train_data.class_to_idx == val_data.class_to_idx
output.close()
print(train_data.class_to_idx, val_data.class_to_idx)

train_loader = DataLoader(
    train_data,
    batch_size=state['train_batch'], shuffle=True,
    num_workers=state['workers'], pin_memory=True)

val_loader = DataLoader(
    val_data,
    batch_size=state['val_batch'], shuffle=True,
    num_workers=state['workers'], pin_memory=True)

if __name__ == '__main__':
    model = load_model(state, use_cuda)
    run(state, model, train_loader, val_loader, use_cuda)
    submit(model, state, use_cuda, mean, std)
