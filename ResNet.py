import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers
import numpy as np
from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
print('\nTensorflow GPU installed: ' + str(tf.test.is_built_with_cuda()))
print('Is Tensorflow using GPU: \n' + str(tf.test.is_gpu_available()))


class BasicBlock(layers.Layer):
    # 残差模块
    def __init__(self, filter_num, kernel_size, strides=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积单元
        self.conv1 = layers.Conv1D(filter_num, kernel_size, strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        # 第二个卷积单元
        self.conv2 = layers.Conv1D(filter_num, kernel_size, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')

        if strides != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv1D(filter_num, 1, strides=strides))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # 通过identity模块
        identity = self.downsample(inputs)
        # 2条路径输出直接相加
        output = layers.add([out, identity])
        output = tf.nn.relu(output)  # 激活函数
        return output


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=4):
        # layer_dims:list[2,2,2,2,2,2]
        super(ResNet, self).__init__()

        self.stem = Sequential([layers.Conv1D(16, kernel_size=3, strides=1),
                                layers.BatchNormalization(),
                                layers.Activation('relu')
                                ])
        self.layer1 = self.build_resblock(16, layer_dims[0])  # 512
        self.layer2 = self.build_resblock(32, layer_dims[1], kernel_size=5, strides=4)  # 128
        self.layer3 = self.build_resblock(64, layer_dims[2], kernel_size=5, strides=4)  # 32
        self.layer4 = self.build_resblock(128, layer_dims[3], strides=2)  # 16
        self.layer5 = self.build_resblock(256, layer_dims[4], strides=2)  # 8
        self.layer6 = self.build_resblock(512, layer_dims[5], strides=2)  # 4

        self.avgpool = layers.GlobalAveragePooling1D()  # 512大小的向量： 512*1
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, kernel_size=3, strides=1):
        # 辅助函数，堆叠filter_num个BasicBlock
        res_blocks = Sequential()
        # 只有第一个BasicBlock的步长可能不为1，实现下采样
        res_blocks.add(BasicBlock(filter_num, kernel_size, strides))

        for _ in range(1, blocks):  # 其他BasicBlock步长都为1
            res_blocks.add(BasicBlock(filter_num, kernel_size, strides=1))

        return res_blocks
        
x_train = np.loadtxt(r'/content/drive/My Drive/Data/x_train').reshape(-1, 512, 1).astype(np.float32)
y_train = np.loadtxt(r'/content/drive/My Drive/Data/y_train').astype(np.int32)
x_test = np.loadtxt(r'/content/drive/My Drive/Data/x_test').reshape(-1, 512, 1).astype(np.float32)
y_test = np.loadtxt(r'/content/drive/My Drive/Data/y_test').astype(np.int32)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(512)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(512)
# sample = next(iter(train_db))
# print(sample)

model = ResNet([2,2,2,2,2,2])
model.build(input_shape=(512,512,1))
# conv_net.summary()
# fc_net.summary()
optimizer = optimizers.Adam(lr=1e-3)

train_loss = []
test_acc = []
acc_max = 0
for epoch in range(500):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # [b,512,1]=>[b,4]
            logits = model(x, training=True)

            y_onehot = tf.one_hot(y, depth=4)
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))
    train_loss.append(loss)
    total_num = 0
    total_correct = 0
    for x, y in test_db:
        logits = model(x)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)

    acc = total_correct / total_num
    test_acc.append(acc)
    print(epoch, 'acc:', acc)
    if acc > acc_max:
        acc_max = acc
        model.save_weights(r'ResNet/weights.ckpt')
