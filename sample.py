import scipy.io as sio
import os
import numpy as np


def sample(path, label, numbers=1000):
    files = os.listdir(path)
    X = np.arange(512)
    for file in files:
        data = sio.loadmat(os.path.join(path, file))
        name = file[:-4]
        if len(name) > 2:
            head = 'X' + name + '_DE_time'
        else:
            head = 'X0' + name + '_DE_time'
        data = data[head].reshape(-1)
        stride = int((len(data) - 512) / (numbers - 1))
        i = 0
        while i < len(data):
            j = i + 512
            if j > len(data):
                break
            x = data[i:j]
            X = np.row_stack([X, x])
            i = i + stride
    X = np.delete(X, 0, axis=0)
    y = np.empty(len(X))
    y.fill(label)
    return X, y


if __name__ == '__main__':
    # normal:4000, 1000/file, label:0
    # inner: 4000, 250/file, label:1
    # roll:4000, 250/file, label:2
    # outer:4000, 142/file, label:3
    path_normal = r'D:\BaiduNetdiskDownload\西储大学轴承数据中心网站\Normal Baseline Data'
    path_inner = r'D:\BaiduNetdiskDownload\西储大学轴承数据中心网站\12k Drive End Bearing Fault Data\内圈故障'
    path_roll = r'D:\BaiduNetdiskDownload\西储大学轴承数据中心网站\12k Drive End Bearing Fault Data\滚动体故障'
    path_outer = r'D:\BaiduNetdiskDownload\西储大学轴承数据中心网站\12k Drive End Bearing Fault Data\外圈故障'
    x_noraml, y_normal = sample(path_normal, label=0)
    x_inner, y_inner = sample(path_inner, label=1, numbers=250)
    x_roll, y_roll = sample(path_roll, label=2, numbers=250)
    x_outer, y_outer = sample(path_outer, label=3, numbers=143)
    print(x_noraml.shape, y_normal.shape)
    print(x_inner.shape, y_inner.shape)
    print(x_roll.shape, y_roll.shape)
    print(x_outer.shape, y_outer.shape)

    np.savetxt(r'Samples\x_normal', x_noraml)
    np.savetxt(r'Samples\y_normal', y_normal)

    np.savetxt(r'Samples\x_inner', x_inner)
    np.savetxt(r'Samples\y_inner', y_inner)

    np.savetxt(r'Samples\x_roll', x_roll)
    np.savetxt(r'Samples\y_roll', y_roll)

    np.savetxt(r'Samples\x_outer', x_outer)
    np.savetxt(r'Samples\y_outer', y_outer)
