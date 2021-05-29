import numpy as np

from tqdm import tqdm
# from scipy import misc
import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from skimage.transform import resize

# random.seed
np.random.seed(2021)
tf.random.set_seed(2021)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# MobileNet的原始设计是224*224的输入，而fashion mnist的图像只有28*28，差别比较大。
# 虽然说直接输入也不会报错，这里还是把图像放大了两倍，变成了56*56，以免丢失细节信息，
# 当然可以放得更大，但效果没有明显提升，浪费计算量；

height, width = 56, 56

# 改变fashion-MNIST的格式
input_image = Input(shape=(height, width))
# MobileNet必须要三通道图像输入，把图像复制三次，确保能正确馈送至网络中
input_image_ = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, 3), 3, 3))(input_image)

# 调用MobileNet网络
base_model = MobileNet(input_tensor=input_image_, include_top=False, pooling='avg')
output = Dropout(0.5)(base_model.output)
# 预测模型
predict = Dense(10, activation='softmax')(output)

model = Model(inputs=input_image, outputs=predict)

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 将变换后的fashion-MNIST集变换回来，要将模型与数据进行拟合！
X_train = X_train.reshape((-1, 28, 28))
X_train = np.array([resize(x, output_shape=(height, width)).astype(float) for x in tqdm(iter(X_train))]) / 255.

X_test = X_test.reshape((-1, 28, 28))
X_test = np.array([resize(x, output_shape=(height, width)).astype(float) for x in tqdm(iter(X_test))]) / 255.


# 数据扩增
# 随机左右翻转
def random_reverse(x):
    if np.random.random() > 0.5:
        return x[:, ::-1]
    else:
        return x


def data_generator(X, Y, batch_size=100):
    while True:
        idxs = np.random.permutation(len(X))
        X = X[idxs]
        Y = Y[idxs]
        p, q = [], []
        for i in range(len(X)):
            p.append(random_reverse(X[i]))
            q.append(Y[i])
            if len(p) == batch_size:
                yield np.array(p), np.array(q)
                p, q = [], []
        if p:
            yield np.array(p), np.array(q)
            p, q = [], []


model.fit(data_generator(X_train, y_train), steps_per_epoch=600, epochs=50,
          validation_data=data_generator(X_test, y_test), validation_steps=100)
