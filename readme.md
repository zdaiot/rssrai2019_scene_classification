Code for [遥感图像场景分类](http://rscup.bjxintong.com.cn/#/theme/1)

## Requirements
- Pytorch
- Python3

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
python main.py
```