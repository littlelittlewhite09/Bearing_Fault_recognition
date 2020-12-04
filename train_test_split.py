import numpy as np
from sklearn.model_selection import train_test_split

x_noraml = np.loadtxt(r'Samples\x_normal')
y_normal = np.loadtxt(r'Samples\y_normal')

x_inner = np.loadtxt(r'Samples\x_inner')
y_inner = np.loadtxt(r'Samples\y_inner')

x_roll = np.loadtxt(r'Samples\x_roll')
y_roll = np.loadtxt(r'Samples\y_roll')

x_outer = np.loadtxt(r'Samples\x_outer')
y_outer = np.loadtxt(r'Samples\y_outer')

x = x_noraml
x = np.row_stack([x, x_inner])
x = np.row_stack([x, x_roll])
x = np.row_stack([x, x_outer])
np.savetxt(r'Data\x', x)

y = np.append(y_normal, y_inner)
y = np.append(y, y_roll)
y = np.append(y, y_outer)
np.savetxt(r'Data\y', y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=66)
np.savetxt(r'Data\x_train', x_train)
np.savetxt(r'Data\y_train', y_train)
np.savetxt(r'Data\x_test', x_test)
np.savetxt(r'Data\y_test', y_test)

# 训练集归一化
x_train_max = np.max(x_train)
x_train_min = np.min(x_train)
x_train_std = (x_train - x_train_min) / (x_train_max - x_train_min)
x_train_std = x_train_std.astype(np.float32)
np.savetxt(r'Data\x_train_std', x_train_std)
# 测试集归一化
x_test_std = (x_test - x_train_min) / (x_train_max - x_train_min)
x_test_std = x_test_std.astype(np.float32)
np.savetxt(r'Data\x_test_std', x_test_std)