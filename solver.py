import os, shutil
import torch
from utils import AverageMeter
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import time
import models.imagenet as customized_models
import torchvision.models as models
from utils import summary
from logger import Logger, savefig
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pickle


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, state):
    if epoch in state['schedule']:
        state['lr'] *= state['gamma']
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_val(data_iterator, model, criterion, optimizer, use_cuda, usage):
    if usage == 'train':
        tqdm_iterator = tqdm(data_iterator)
        model.train()
    elif usage == 'val':
        tqdm_iterator = tqdm(data_iterator)
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for inputs, targets in tqdm_iterator:
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if usage == 'train':
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        info = '(Usage:{usage} | Data: {data:.3f}s | Batch: {bt:.3f}s |  Loss: {loss:.4f} | top1: {top1: .4f} | top5: ' \
               '{top5: .4f}'.format(
            usage=usage,
            data=data_time.val,
            bt=batch_time.val,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )

        tqdm_iterator.set_description(info)

    return losses.avg, top1.avg


def load_model(state, use_cuda):
    # create model
    if state['pretrained']:
        print("=> using pre-trained model '{}'".format(state['arch']))
        model = models.__dict__[state['arch']](pretrained=True)
    elif state['arch'].startswith('resnext') or state['arch'].startswith('se_resnext'):
        print("=> creating model '{}'".format(state['arch']))
        model = customized_models.__dict__[state['arch']](
            baseWidth=state['base_width'],
            cardinality=state['cardinality'],
            num_class=state['num_classes']
        )
    else:
        print("=> creating model '{}'".format(state['arch']))
        model = models.__dict__[state['arch']](num_classes=state['num_classes'])

    if use_cuda:
        if state['arch'].startswith('alexnet') or state['arch'].startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # 输出网络信息
    fout = open(os.path.join(state['checkpoint'], 'out.txt'), 'w')
    summary(model, (3, state['image_size'], state['image_size']), print_fn=lambda x: fout.write(x + '\n'))
    num_params = 'Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    print(num_params)
    fout.write(num_params + '\n')
    fout.flush()
    fout.close()
    return model


def run(state, model, train_loader, val_loader, use_cuda):
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=state['momentum'], weight_decay=state['weight_decay'])

    # 杂项初始化
    best_acc = 0
    title = '' + state['arch']

    # 开始训练
    if state['resume']:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(state['resume']), 'Error: no checkpoint directory found!'
        state['checkpoint'] = os.path.dirname(state['resume'])
        checkpoint = torch.load(state['resume'])
        best_acc = checkpoint['best_acc']
        state['start_epoch'] = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(state['checkpoint'], 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(state['checkpoint'], 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    for epoch in range(state['start_epoch'], state['epochs']):
        adjust_learning_rate(optimizer, epoch, state)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, state['epochs'], state['lr']))
        train_loss, train_acc = train_val(train_loader, model, criterion, optimizer, use_cuda, 'train')
        test_loss, test_acc = train_val(val_loader, model, criterion, optimizer, use_cuda, 'val')

        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=state['checkpoint'])

    logger.close()
    logger.plot()
    savefig(os.path.join(state['checkpoint'], 'log.eps'))

    print('Best acc:')
    print(best_acc)


def create_label_decoder():
    label_decoder = dict()
    txt_path = './datasets/ClsName2id.txt'
    f = open(txt_path, 'r', encoding='UTF-8')
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split(':')
        label_decoder[tmp[0]] = eval(tmp[-1])
    return label_decoder


def submit(model, state, use_cuda, mean, std):
    pkl_file = open(state['checkpoint'] + '/label_encode.pkl', 'rb')
    label_encoder = pickle.load(pkl_file)
    label_encoder = {v: k for k, v in label_encoder.items()}
    label_decoder = create_label_decoder()
    checkpoint = torch.load('checkpoint/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    image_names = os.listdir(state['test_path'])
    result = open('classification.txt', 'w')

    def image_transform(image, image_size, mean, std):
        resize = transforms.Resize(image_size)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean, std)
        transform_compose = transforms.Compose([resize, to_tensor, normalize])
        return transform_compose(image)

    with torch.no_grad():
        for image_name in tqdm(image_names):
            img_path = os.path.join(state['test_path'], image_name)
            img = Image.open(img_path).convert('RGB')
            img = image_transform(img, state['image_size'], mean, std)
            img = torch.unsqueeze(img, dim=0)
            if use_cuda:
                img = img.float().cuda()
            else:
                img = img.float()
            pred = torch.softmax(model(img), dim=1)
            pred = torch.squeeze(pred, dim=0)
            pred = pred.detach().cpu().numpy()
            label_pred = np.argmax(pred)
            label_ = label_encoder[label_pred]
            label_true = label_decoder[label_]
            result.write(image_name + ' ' + str(label_true) + '\n')
            result.flush()
    result.close()


if __name__ == '__main__':
    label_decoder = create_label_decoder()
    print(label_decoder)