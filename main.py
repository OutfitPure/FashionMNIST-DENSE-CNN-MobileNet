import tensorflow as tf

from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint

# 加载数据
fashion_mnist = datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("train_images shape:", train_images.shape, "train_labels shape:", train_labels.shape)

# Print training set shape - note there are 60,000 training data of image size of 28x28, 60,000 train labels)
# 打印训练集结构-请注意，有60,000个训练数据，图像尺寸为28x28，60,000个训练标签）
print("train_images shape:", train_images.shape, "train_labels shape:", train_labels.shape)

# Print the number of training and test datasets
# 打印训练和测试数据集的数量
print(train_images.shape[0], 'train set')
print(test_images.shape[0], 'test set')

# Define the text labels
# 定义文字标签
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",  # index 1
                        "Pullover",  # index 2
                        "Dress",  # index 3
                        "Coat",  # index 4
                        "Sandal",  # index 5
                        "Shirt",  # index 6
                        "Sneaker",  # index 7
                        "Bag",  # index 8
                        "Ankle boot"]  # index 9

# Image index, you can pick any number between 0 and 59,999
# 图片索引，您可以选择0到59,999之间的任意数字
img_index = 5

# train_labels contains the labels, ranging from 0 to 9
# train_labels包含标签，范围从0到9
label_index = train_labels[img_index]

# Print the label, for example 2 Pullover
# 打印标签，例如2套头衫
print("y = " + str(label_index) + " " + (fashion_mnist_labels[label_index]))

# # Show one of the images from the training dataset
# 显示训练数据集中的图像之一
# plt.imshow(train_images[img_index])

# Data normalization     数据归一化
# Normalize the data dimensions so that they are of approximately the same scale.   归一化数据维度，以使它们具有大致相同的比例。
# 将像素的值标准化至0到1的区间内。
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
print("Number of train data - " + str(len(train_images)))
print("Number of test data - " + str(len(test_images)))

# Split the data into train/validation/test data sets
# Training data - used for training the model
# Validation data - used for tuning the hyperparameters and evaluate the models
# Test data - used to test the model after the model has gone through initial vetting by the validation set.
# 将数据分为训练/验证/测试数据集
# 训练数据-用于训练模型
# 验证数据-用于调整超参数和评估模型
# 测试数据-用于在模型经过验证集的初始审核之后对模型进行测试。

# Further break training data into train / validation sets
# (# put 5000 into validation set and keep remaining 55,000 for train)
# 进一步将训练数据分解为训练/验证集（将5000个输入验证集，并保留剩余的55,000进行训练）
(train_images, images_valid) = train_images[5000:], train_images[:5000]
(train_labels, labels_valid) = train_labels[5000:], train_labels[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
# 将输入数据从（28，28）调整为（28，28，1)
w, h = 28, 28
train_images = train_images.reshape(train_images.shape[0], w, h, 1)
images_valid = images_valid.reshape(images_valid.shape[0], w, h, 1)
test_images = test_images.reshape(test_images.shape[0], w, h, 1)

# One-hot encode the labels
# 独热编码标签
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
labels_valid = tf.keras.utils.to_categorical(labels_valid, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Print training set shape
# 打印训练集结构
print("train_images shape:", train_images.shape, "train_labels shape:", train_labels.shape)

# Print the number of training, validation, and test datasets
# 打印训练，验证和测试数据集的数量
print(train_images.shape[0], 'train set')
print(images_valid.shape[0], 'validation set')
print(test_images.shape[0], 'test set')

# Create the model architecture                 # There are two APIs for defining a model in Keras:
# 创建模型架构                                   # 在Keras中有两个用于定义模型的API:
# Sequential model API                       # 顺序模型API
# Functional API                            # 功能性API
# We are using the Sequential model API.   # 我们正在使用顺序模型API。
#
# In defining the model we will be using some of these Keras APIs:
# 在定义模型时，我们将使用以下一些Keras API：
#
# Conv2D()  - create a convolutional layer  - 创建卷积层
# Pooling() - create a pooling layer        - 创建一个池化层
# Dropout() - apply drop out                - 应用退出

model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
# 必须在神经网络的第一层中定义输入形状
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()

# Compile the model                  # 编译模型
# Configure the learning process with compile() API before training the model. It receives three arguments:
# 在训练模型之前，请使用compile（）API配置学习过程。 它收到三个参数：
# An optimizer                      # 优化器
# A loss function                   # 损失函数
# A list of metrics                 # 指标列表

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model                               # 训练模型
# Now let's train the model with fit() API.     # 现在，让我们使用fit（）API训练模型。
#
# We use the ModelCheckpoint API to save the model after every epoch.
# Set "save_best_only = True" to save only when the validation accuracy improves.
# 在每个时期之后，我们使用ModelCheckpoint API来保存模型。 设置“ save_best_only = True”仅在验证准确性提高时保存。
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
model.fit(train_images,
          train_labels,
          batch_size=64,
          epochs=10,
          validation_data=(images_valid, labels_valid),
          callbacks=[checkpointer])

# Load the weights with the best validation accuracy  # 以最佳验证精度加载模型
model.load_weights('model.weights.best.hdf5')

# Test Accuracy  # 测试精度
# Evaluate the model on test set
score = model.evaluate(test_images, test_labels, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])
