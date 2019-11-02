from keras.layers import *
from keras.models import *
from keras.optimizers import *
height = width = 224
def build_vgg(input_shape=(height, width, 3), num_classes=1000):
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
if __name__ == "__main__":
    # reference：keras中文手册: https://keras.io/zh/models/model/    
    # 模型结构
    model = build_vgg()
    # 配置器
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002), metrics=['accuracy'])
    # 模型打印    
    print(model.summary())
    pass
