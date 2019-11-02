from __future__ import absolute_import, division, print_function
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
            label = label_data(img) # 
            path_list = os.path.join(path, path_list)
            src = cv2.imread(path_list)      #cv2.IMREAD_GRAYSCALE      
            if src is None:
                continue
            src = cv2.resize(src, (img_width, img_height))     #img
            training_data.append([np.array(src), label])
        pass
    pass
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


from keras.layers import *
from keras.models import *
from keras.optimizers import *
def build_vgg(input_shape=(img_width, img_height, 3), num_classes=2):
    model = Sequential()
    # Block 1, 2层
    model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same', input_shape=input_shape))
    model.add(Conv2D(64, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 2, 2层
    model.add(Conv2D(128, 3, 3, activation='relu',border_mode='same'))
    model.add(Conv2D(128, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 3, 3层
    model.add(Conv2D(256, 3, 3, activation='relu',border_mode='same'))
    model.add(Conv2D(256, 3, 3, activation='relu',border_mode='same'))
    model.add(Conv2D(256, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 4, 3层
    model.add(Conv2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(Conv2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(Conv2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 5, 3层
    model.add(Conv2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(Conv2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(Conv2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Classification block, 全连接3层
    # 扁平层-化为一行，行接行
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
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
    X_train = np.array([i[0] for i in train]).reshape(-1,img_width,img_height,3)
    Y_train = np.array([i[1] for i in train])
    X_val = np.array([i[0] for i in val]).reshape(-1,img_width,img_height,3)
    Y_val = np.array([i[1] for i in val])
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        shear_range=0.3,    #错切变换，效果就是让所有点的x坐标(或者y坐标)保持不变，而对应的y坐标(或者x坐标)则按比例发生平移
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True  # randomly flip images
        )  
    

    # --------model--------
    # 
    epochs = 1
    batch_size = 64
    #optimizer = Adam(0.0002)
    optimizer = RMSprop(lr=1e-4)

    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_loss', 
        patience=4, 
        verbose=1, 
        factor=0.5, 
        min_lr=5e-8
        )

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=6,
        min_delta=0.0002, 
        verbose=1, 
        mode='auto'
        )   

    filepath="./weights/weights.best.hdf5"
    if not os.path.exists('./weights'):
            os.mkdir('./weights')
    checkpoint = ModelCheckpoint(
        filepath, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='auto'
        )

    # 模型结构
    model = build_vgg()
    # 配置器
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    print(X_train.shape[0])
    
    model_his = model.fit_generator(
        datagen.flow(X_train,Y_train, batch_size=batch_size),
        epochs = epochs, 
        validation_data = (X_val,Y_val), 
        shuffle=True, 
        verbose = 1, 
        steps_per_epoch=X_train.shape[0] // batch_size, 
        callbacks=[learning_rate_reduction, early_stopping, checkpoint]
        )
    model.save(filepath)
    # draw training ——》》》model_his

    pass

'''
# 问题解决 1 ：
AttributeError: module ‘tensorboard' has no attribute 'lazy'（谷歌出来的结果是Make sure that you don't have tb-nightly installed in your env.）
我采用pip uninstall tb-nightly指令发现并没有安装
'''