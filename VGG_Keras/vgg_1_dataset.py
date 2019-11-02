'''
data/
    train/
        veh/
            1.jpg
            2.jpg
            ...
        noveh/
            1.jpg
            2.jpg
            ...
    validation/
        veh/
            1.jpg
            2.jpg
            ...
        noveh/
            1.jpg
            2.jpg
            ...
'''
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split

'''
数据生成器制作
1. 路径+样本数
2. 样本尺度
3. ImageDataGenerator 函数生成数据对象实例，由该实例的成员函数 flow_from_directory 产生数据生成器 tian_generator、balidation_generator
4. 最后该balidation_generator和tian_generator放入模型的fit_generator中进行实现。
'''
TrainDataDir = './data/train'
img_width = img_height = 224
train_sample_size = 6000
Batch_size = 300
def tx_gen(DataDir=TrainDataDir, batch_size=Batch_size, train=True):
    # 1. ImageDataGenerator ClassModel  # train_data = ImageDataGenerator()
    train_data = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    # 2. flow_from_directory in ImageDataGenerator ClassModel
    train_generator = train_data.flow_from_directory(
        DataDir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,    #class_mode='binary'
        shuffle = False
    )    

'''

PetImages/
    train/
        Dog/
            1.jpg
            2.jpg
            ...
        Cat/
            1.jpg
            2.jpg
            ...
    test/
        Dog/
            1.jpg
            2.jpg
            ...
        Cat/
            1.jpg
            2.jpg
            ...

'''

from tqdm import tqdm      # display Tqdm 是一个快速，可扩展的Python进度条
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle 

TRAIN_DATA_DIR = r'./PetImages/train'
img_width = img_height = 224  
# one-hot 编码
def label_data(label): # word_label = img.split('.')[-3]
    if label == 'cat': 
        return [1,0]
    elif label == 'dog': 
        return [0,1]
def create_train_dataset(path=TRAIN_DATA_DIR):
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DATA_DIR)): # os.listdir突然不灵了，不能返回子层文件检索,只返回['cat','dog']   
        path = os.path.join(TRAIN_DATA_DIR, img) # 多加此行 返回 */cat/ 和 */dog/
        for path_list in os.listdir(path):
            label = label_data(img)
            path_list = os.path.join(path,path_list)
            src = cv2.imread(path_list)            
            if src is None:
                continue
            src = cv2.resize(src, (img_width, img_height))     #img
            training_data.append([np.array(src), label])
        pass
    pass
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # reference：keras中文手册: https://keras.io/zh/models/model/    
    # --------dataset--------
    Initdata = False # 如果存在数据npy不需要重新生成
    if Initdata == True:
        training_data = create_train_dataset()
    else:
        training_data = np.load('train_data.npy', allow_pickle = True) # 权限allow_pickle = True
    
    # spilt data 
    train, val = train_test_split(training_data, test_size = 0.25)
    X_train = np.array([i[0] for i in train]).reshape(-1,img_width,img_height,1)
    Y_train = np.array([i[1] for i in train])
    X_val = np.array([i[0] for i in val]).reshape(-1,img_width,img_height,1)
    Y_val = np.array([i[1] for i in val])
    # --------model--------
               
    
