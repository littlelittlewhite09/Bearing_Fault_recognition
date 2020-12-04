import tensorflow as tf # 2.X版本
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers
import numpy as np

# 测试GPU，是否使用GPU
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
print('\nTensorflow GPU installed: ' + str(tf.test.is_built_with_cuda()))
print('Is Tensorflow using GPU: \n' + str(tf.test.is_gpu_available()))



# 一维卷积的输入形式：[b,L,C]
conv_layers = [  # 5 units of conv + max pooling
    # unit 1
    layers.Conv1D(16, kernel_size=3, padding='same', activation=tf.nn.relu),
    # layers.Dropout(0.5),
    layers.Conv1D(16, kernel_size=3, padding='same', activation=tf.nn.relu),
    # layers.Dropout(0.5),
    layers.MaxPool1D(pool_size=4, strides=4, padding='same'),  # [b,128,32]

    # unit 2
    layers.Conv1D(32, kernel_size=3, padding='same', activation=tf.nn.relu),
    # layers.Dropout(0.5),
    layers.Conv1D(32, kernel_size=3, padding='same', activation=tf.nn.relu),
    # layers.Dropout(0.5),
    layers.MaxPool1D(pool_size=4, strides=4, padding='same'), # [b,32,64]

    # unit 3
    layers.Conv1D(64, kernel_size=3, padding='same', activation=tf.nn.relu), # 1
    # layers.Dropout(0.5),
    layers.Conv1D(64, kernel_size=3, padding='same', activation=tf.nn.relu), # 1
    # layers.Dropout(0.5),
    layers.MaxPool1D(pool_size=2, strides=2, padding='same'),#[b,16,128]

    # unit 4
    layers.Conv1D(128, kernel_size=3, padding='same', activation=tf.nn.relu),# 1
    # layers.Dropout(0.5),
    layers.Conv1D(128, kernel_size=3, padding='same', activation=tf.nn.relu),# 1
    # layers.Dropout(0.5),
    layers.MaxPool1D(pool_size=2, strides=2, padding='same'),#[b,8,512]

    # unit 5
    layers.Conv1D(256, kernel_size=3, padding='same', activation=tf.nn.relu),# 1
    # layers.Dropout(0.5),
    layers.Conv1D(256, kernel_size=3, padding='same', activation=tf.nn.relu),# 1
    # layers.Dropout(0.5),
    layers.MaxPool1D(pool_size=2, strides=2, padding='same'), # [b,4,512]

    # unit 6
    layers.Conv1D(512, kernel_size=3, padding='same', activation=tf.nn.relu),
    layers.Conv1D(512, kernel_size=3, padding='same', activation=tf.nn.relu),
    layers.MaxPool1D(pool_size=2, strides=2, padding='same'),  # [b,2,512]

    # # unit 7
    layers.Conv1D(512, kernel_size=3, padding='same', activation=tf.nn.relu),
    layers.Conv1D(512, kernel_size=3, padding='same', activation=tf.nn.relu),
    layers.MaxPool1D(pool_size=2, strides=2, padding='same')  # [b,1,512]
    ]

fc_net = Sequential([
    layers.Dense(128, activation=tf.nn.relu),
    # layers.Dropout(0.5),
    layers.Dense(64,activation=tf.nn.relu),
    # layers.Dropout(0.5),
    layers.Dense(32,activation=tf.nn.relu),
    layers.Dense(4, activation=None)
])

x_train = np.loadtxt(r'/content/drive/My Drive/Data/x_train_std').reshape(-1,512,1).astype(np.float32)
y_train = np.loadtxt(r'/content/drive/My Drive/Data/y_train').astype(np.int32)
x_test = np.loadtxt(r'/content/drive/My Drive/Data/x_test_std').reshape(-1,512,1).astype(np.float32)
y_test = np.loadtxt(r'/content/drive/My Drive/Data/y_test').astype(np.int32)
train_db= tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(512)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(512)
# sample = next(iter(train_db))
# print(sample)


conv_net = Sequential(conv_layers)
conv_net.build(input_shape=[4, 512, 1])
fc_net.build(input_shape=[4,512])
# conv_net.summary()
# fc_net.summary()
optimizer = optimizers.Adam(lr=1e-3)

variables = conv_net.trainable_variables + fc_net.trainable_variables
train_loss = []
test_acc = []
acc_max = 0
for epoch in range(200):
    for step, (x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = conv_net(x,training=True)
            out = tf.reshape(out,[-1,512])
            logits = fc_net(out,training=True)
            y_onehot = tf.one_hot(y, depth=4)
            loss = tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))
    train_loss.append(loss)
    total_num = 0
    total_correct = 0
    for x,y in test_db:

        out = conv_net(x)
        out = tf.reshape(out, [-1, 512])
        logits = fc_net(out)
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
      conv_net.save_weights('conv/weights.ckpt')
      fc_net.save_weights('fc/weights.ckpt')
