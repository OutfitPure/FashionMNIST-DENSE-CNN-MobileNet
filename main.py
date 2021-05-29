# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 从 TensorFlow 中导入和加载 Fashion MNIST 数据：
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 训练集结构
print("训练集结构", train_images.shape)
# 训练集标签个数
print("训练集标签个数", len(train_labels))
# 每个标签都是一个 0 到 9 之间的整数：
print("每个标签都是一个 0 到 9 之间的整数：", train_labels)
# 测试集中有 10,000 个图像。同样，每个图像都由 28x28 个像素表示：
print("测试集图像结构", test_images.shape)
# 测试集包含 10,000 个图像标签：
print("测试集图像标签的数目", len(test_labels))

# 预处理数据
# 在训练网络之前，必须对数据进行预处理。如果您检查训练集中的第一个图像，您会看到像素值处于 0 到 255 之间：
plt.figure()
plt.rcParams["font.family"] = "SimHei"
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.title("数据处理图")
# plt.show()

# 将这些值缩小至 0 到 1 之间，然后将其馈送到神经网络模型。为此，请将这些值除以 255。请务必以相同的方式对训练集和测试集进行预处理：
train_images = train_images / 255.0
test_images = test_images / 255.0

# 为了验证数据的格式是否正确，以及您是否已准备好构建和训练网络，让我们显示训练集中的前 25 个图像，并在每个图像下方显示类名称。
plt.figure(figsize=(10, 10))
plt.rcParams["font.family"] = "SimHei"
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 构建模型
"""
构建神经网络需要先配置模型的层，然后再编译模型。
"""
# 设置层
"""
神经网络的基本组成部分是层。层会从向其馈送的数据中提取表示形式。
大多数深度学习都包括将简单的层链接在一起。大多数层（如 tf.keras.layers.Dense）都具有在训练期间才会学习的参数。
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])


model.summary()
"""
该网络的第一层 tf.keras.layers.Flatten 将图像格式从二维数组（28 x 28 像素）转换成一维数组（28 x 28 = 784 像素）。
将该层视为图像中未堆叠的像素行并将其排列起来。该层没有要学习的参数，它只会重新格式化数据。
展平像素后，网络会包括两个 tf.keras.layers.Dense 层的序列。
它们是密集连接或全连接神经层。
第一个 Dense 层有 128 个节点（或神经元）。第二个（也是最后一个）层会返回一个长度为 10 的 logits 数组→→对数数组，是非标准化的→→可通过softmax层输出后变线性。
每个节点都包含一个得分，用来表示当前图像属于 10 个类中的哪一类。
"""

# 编译模型
"""
在准备对模型进行训练之前，还需要再对其进行一些设置。以下内容是在模型的编译步骤中添加的：

损失函数 - 用于测量模型在训练期间的准确率。您会希望最小化此函数，以便将模型“引导”到正确的方向上。
优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
指标 - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 训练模型
"""
训练神经网络模型需要执行以下步骤：

将训练数据馈送给模型。在本例中，训练数据位于 train_images 和 train_labels 数组中。
模型学习将图像和标签关联起来。
要求模型对测试集（在本例中为 test_images 数组）进行预测。
验证预测是否与 test_labels 数组中的标签相匹配。
"""
# 向模型馈送数据
"""开始训练，调用 model.fit 方法，这样命名是因为该方法会将模型与训练数据进行“拟合”："""
model.fit(train_images, train_labels, epochs=10)
"""在模型训练期间，会显示损失和准确率指标。此模型在训练数据上的准确率达到了 0.91（或 91%）左右。"""
# 评估准确率
"""比较模型在测试数据集上的表现："""
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
model.summary()
print('\nTest accuracy:', test_acc)

#
# # 以下内容新增加了一个层 softmax层，进行预测，与最后的精度无关。
# # 进行预测
# """在模型经过训练后，您可以使用它对一些图像进行预测。模型具有线性输出，即 logits。您可以附加一个 softmax 层，将 logits 转换成更容易理解的概率。"""
# print("将 logits 转换成更容易理解的概率")
# probability_model = tf.keras.Sequential([model,
#                                          tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_images)
#
# """在上例中，模型预测了测试集中每个图像的标签。我们来看看第一个预测结果："""
# print("预测的第一个结果为：", predictions[0])
#
# """预测结果是一个包含 10 个数字的数组。它们代表模型对 10 种不同服装中每种服装的“置信度”。您可以看到哪个标签的置信度值最大："""
# print("转化为置信度为", np.argmax(predictions[0]))
#
# print("该模型非常确信这个图像是短靴，或 class_names[9]。通过检查测试标签发现这个分类是正确的：")
# print("第一个图像的实际标签为：", test_labels[0])
#
#
# # 您可以将其绘制成图表，看看模型对于全部 10 个类的预测。
# def plot_image(i, predictions_array, true_label, img):
#     predictions_array, true_label, img = predictions_array, true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'
#
#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          class_names[true_label]),
#                color=color)
#
#
# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array, true_label[i]
#     plt.grid(False)
#     plt.xticks(range(10))
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#
#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')
#
#
# # 验证预测结果
# """在模型经过训练后，您可以使用它对一些图像进行预测。
# 我们来看看第 0 个图像、预测结果和预测数组。正确的预测标签为蓝色，错误的预测标签为红色。数字表示预测标签的百分比（总计为 100）。"""
# i = 0
# plt.figure(figsize=(6, 3))
# plt.rcParams["font.family"] = "SimHei"
# plt.subplot(1, 2, 1)
# plt.title("预测测试集中第1个图片")
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i], test_labels)
# plt.title("预测正确")
# plt.show()
#
# i = 12
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.title("预测测试集中第12张图片")
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i], test_labels)
# plt.title("预测出错")
# plt.show()
#
# # 用模型的预测绘制几张图像。      即使置信度很高，模型也可能出错。
# # 绘制前X张测试图像，它们的预测标签和真实标签。
# # 将正确的预测颜色设置为蓝色，将不正确的预测颜色设置为红色。
# num_rows = 5
# num_cols = 3
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions[i], test_labels, test_images)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()
#
# # 使用训练好的模型
# # 最后，使用训练好的模型对单个图像进行预测。
#
# # 从测试数据集中获取图像。
# img = test_images[1]
# print("图片大小为", img.shape)
#
# # tf.keras 模型经过了优化，可同时对一个批或一组样本进行预测。因此，即便您只使用一个图像，您也需要将其添加到列表中：
# # Add the image to a batch where it's the only member.将图像添加到它是唯一成员的批处理中。
# img = (np.expand_dims(img, 0))
# print("图片大小为", img.shape)
#
# # 现在预测这个图像的正确标签：
# predictions_single = probability_model.predict(img)
# print("图片格式", predictions_single)
#
# plot_value_array(1, predictions_single[0], test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
# plt.title("置信度柱形图")
# plt.show()
# # keras.Model.predict 会返回一组列表，每个列表对应一批数据中的每个图像。在批次中获取对我们（唯一）图像的预测：
# print(np.argmax(predictions_single[0]))
