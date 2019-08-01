'''
OSError: Could not load libjpeg-turbo library
https://github.com/ajkxyz/jpeg4py     
sudo apt-get install libturbojpeg
'''
import imghdr
import os
import struct
import cv2
from PIL import Image, ImageSequence 
import uuid
import numpy as np
import multiprocessing as mp
import shutil
import jpeg4py as jpeg
import shutil
 
type_dict = {
    'FFD8FF':'jpg','89504E47':'png','47494638':'gif','49492A00':'tif',
    '424D':'bmp','41433130':'dwg','38425053':'psd','7B5C727466':'rtf','3C3F786D6C':'xml',
    '68746D6C3E':'html','44656C69766572792D646174653A':'eml','CFAD12FEC5FD746F':'dbx','2142444E':'pst',
    'D0CF11E0':'doc/xls','5374616E64617264204A':'mdb','FF575043':'wpd','252150532D41646F6265':'ps/eps',
    '255044462D312E':'pdf','AC9EBD8F':'qdf','E3828596':'pwl','504B0304':'zip',
    '52617221':'rar','57415645':'wav','41564920':'avi','2E7261FD':'ram',
    '2E524D46':'rm','000001BA':'mpg','000001B3':'mpg','6D6F6F76':'mov','3026B2758E66CF11':'asf','4D546864':'mid'
}


def check_remove_broken(img_path):
    '''检查jpeg图片是否损坏

    Args: img_path： 图片路径
    
    Return：如果损坏，则返回True，否则返回False
    '''
    try:
        x = jpeg.JPEG(img_path).decode() # 只能检测jpg，对于png图片同样报错
        # x = cv2.imread(img_path)          #　报warning，
        return False
    except Exception:
        print('Decoding error:' , img_path)
        tmp = img_path.split('/')
        tmp[1] = tmp[1]+'old'
        path = [''.join(x) for x in tmp[:-1]]
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.move(img_path, os.path.join(path, tmp[-1]))
        print('moving pic {}'.format(img_path))
        # os.remove(img_path)
        return True
 
def get_filetype(filename):
    '''检查图片的类型

    Args: filename: 图片路径
    
    Return：图片类型
    '''
    file = open(filename,'rb')
    ftype = 'unknown'
 
    # 转成16进制字符串
    def bytes2hex(bytes):
        num = len(bytes)
        hexstr = u""
        for i in range(num):
            t = u"%x" % bytes[i]
            if len(t) % 2:
                hexstr += u"0"
            hexstr += t
        return hexstr.upper()

    for k,v in type_dict.items():
        num_bytes = int(len(k)/2) # k为16进制，每一位用4bit存储，而下面的无符号char为8bit，除以2得到应该读取多少个char
        file.seek(0)
        # 'B'为无符号char，struct.unpack按照num_bytes个char的形式读取二进制
        hbytes = struct.unpack('B'*num_bytes, file.read(num_bytes))
        code = bytes2hex(hbytes)
        if code == k:
            ftype =  v
            break
 
    file.close()
    return ftype


def modify_image_formate(image_name, origin_format, dir_format='.jpg'):
    '''修改图片的存储格式

    Args: 
        origin_format:图片的正确格式
        image_name: 待修改的图片的存储路径
        dir_format:　目标格式
    
    Return：None
    '''
    if origin_format == 'png' or origin_format == 'bmp':
        image = cv2.imread(image_name)
        (filename, extension) = os.path.splitext(image_name) # 区别文件名(含路径)与后缀名
        dir_image_name = filename + dir_format # 可以按照format形式保存图片
        os.remove(image_name)
        cv2.imwrite(dir_image_name, image)

    elif origin_format == 'gif':
        # using Image
        im = Image.open(image_name)
        iter = ImageSequence.Iterator(im) # GIF图片流的迭代器
        for frame in iter:
            # frame.save("./frame%d.png" % index)
            frame = frame.convert("RGB")
            frame = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)
            (filepath, tempfilename) = os.path.split(image_name) # 区别路径与文件名
            cv2.imwrite(os.path.join('./', uuid.uuid4().hex, dir_format), frame) # filepath
        delete_path = os.path.join('./delete', image_name.split('/')[-1])
        shutil.move(image_name, delete_path)  

        # # using opencv
        # gif = cv2.VideoCapture(image_name)
        # success, frame = gif.read()
        # while(success):
        #     (filepath, tempfilename) = os.path.split(image_name)
        #     cv2.imwrite(os.path.join(filepath, uuid.uuid4().hex, dir_format), frame)
        #     success, frame = gif.read()
        # gif.release()
        # delete_path = os.path.join('./delete', image_name.split('/')[-1])
        # shutil.move(image_name, delete_path)  
    elif origin_format == 'unknown':
        os.remove(image_name)
        print("Unknown format, delete it!!!")

# 全局参数
dir_format = "jpg"
def run(image_full_name):
    '''修改图片为特定的存储格式，之所以将dir_format放置到外面，是为了方便多线程调用

    Args: 
        image_full_name: 图片的路径
        dir_format:　目标格式
    
    Return：None
    '''
    image_type = get_filetype(image_full_name)

    # 图片存储格式正确时，跳过当前图片，否则修改图片存储格式
    if image_type is dir_format:
        pass
    else:
        print("Modifing {}, it's right format is: {}.".format(image_full_name, image_type))
        modify_image_formate(image_full_name, origin_format=image_type, dir_format='.jpg')

def get_image_list(image_path, sub_dir_exit):
    '''读取特定目录下的所有文件

    Args: 
        image_path: 数据存放的路径
        sub_dir_exit: 目录下是否存在二级目录
    
    Return：所有文件的路径
    '''
    if sub_dir_exit:
        sub_dirs = os.listdir(image_path)
    else:
        sub_dirs = image_path
    img_list = []
    for sub_dir in sub_dirs:
        print("------------{}----------".format(sub_dir))
        if sub_dir_exit:
            image_names = os.listdir(os.path.join(image_path, sub_dir))
        else:
            image_names = sub_dir

        for image_name in image_names:
            if sub_dir_exit:
                image_full_name = os.path.join(image_path, sub_dir, image_name)
            else:
                image_full_name = os.path.join(sub_dir, image_name)
            
            img_list.append(image_full_name)
    return img_list

if __name__ == "__main__":
    image_path = "./datasets/train"
    sub_dir_exit = True
    img_list = get_image_list(image_path, sub_dir_exit)
    pool = mp.Pool()

    pool.map(run, img_list)
    print('Convert Done!')
    img_list = get_image_list(image_path, sub_dir_exit)

    # res = pool.map(check_remove_broken, img_list)


        