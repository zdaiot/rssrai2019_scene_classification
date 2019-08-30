Code for [遥感图像场景分类](http://rscup.bjxintong.com.cn/#/theme/1)

## Requirements
- Pytorch
- Python3

## Support Model
- 'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'googlenet', 'inception_v3', 'mobilenet_v2', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'. These model can use pre-training weights.
- 'resnext101', 'resnext152', 'resnext18', 'resnext50', 'se_resnext101', 'se_resnext152', 'se_resnext18', 'se_resnext50'. These model cannot use pre-training weights.

## run
First, Download datasets form https://pan.baidu.com/s/1pw8SmeZ23VRSLXquefiRTA 提取码: hms2
- Unzip train.zip and put in `./datasets`
- Unzip val.zip and put in `./datasets`
- Unzip test.zip and put in `./datasets`
- Copy ClsName2id.txt to `./datasets`

delete `./datasets/train/居民区/residential-area_02938.jpg`, because it is not a image file.

For example
```
├─ClsName2id.txt
├─test
├─train
│  ├─公园
│  ├─旱地
│  ├─河流
│  ├─海滩
│  └─湖泊
└─val
    ├─公园
    ├─旱地
    ├─河流
    ├─海滩
    └─湖泊
```

run
```
python main.py --arch=se_resnext152
```