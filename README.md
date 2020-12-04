# Bearing_Fault_recognition

## 1 Resource of data

本文试验数据为开源的西储大学轴承数据，使用了以12KHZ采样率得到的驱动端故障振动数据与轴承正常振动数据。原数据集中，轴承的故障类型一共有3种，包括内圈故障、滚动体故障和外圈故障。所有故障类型均为试验前人为制造，利用电火花加工在轴承不同部位(内圈、滚动体和外圈)制造4种不同半径的凹坑，分别是0.007、0.0014、0.021和0.028，单位：mils。电机转速与轴承载荷是成对设置的，0hp对应1797rmp，1hp对应1772rmp，2hp对应1750rmp，3hp对应1730rmp，共四种(hp:马力，rmp:每分钟转)。

## 2 Data sampling

在进行轴承振动数据学习之前，需先对振动数据进行采样。在所有的工况中，最低的转速为1730rmp，以12KHZ的采样频率进行采样时，转轴转一圈时，将会采到约416（60/1730*12000=416）个振动加速度数据，即此时一个数据周期为416。根据数据周期长度来确定单个样本数据的时间跨度。一般而言，选择连续截取512个数据点作为单一样本的时间跨度。

另一方面，为了尽可能获得更多的数据样本，考虑重叠采样，如下图所示。其中，Stride表示两次相邻的采样之间的间隔步长。

![1](https://github.com/littlelittlewhite09/Bearing_Fault_recognition/raw/main/Screenshots/1.png)                          

## 3 VGG architecture

采用的网络参考了VGG13网络，如图所示。该网络的输入为时序长度512的振动加速度信号，时序信号先通过7个Conv-Conv-Pooling单元，再通过4层全连接层，最后由softmax层得到预测结果。7个单元都是由两个卷积层和一个最大池化层组成。每个单元的卷积核均采用31大小，通过padding填充，使得每次卷积操作都不改变序列长度；池化窗口大小与窗口移动步长相关联：当步长为4时，序列长度因为最大池化下采样会变为原来的四分之一，为了尽可能保留特征映射信息，选择41大小的池化窗口。当步长为2时窗口为21也是同理。

![2](https://github.com/littlelittlewhite09/Bearing_Fault_recognition/raw/main/Screenshots/2.png) 

## 4 ResNet architecture

### 4.1 ResBlock

每层卷积层后面还添加一个批量归一化层（BatchNormalization），目的是为了获得更加平滑的优化地形，以提高优化效率，除此以外它也是一种正则化方法，有助于提高网络的泛化能力。通过两层卷积层和批量归一化层之后，得到特征变换后的输出，与最开始的输入  相加得到最终输出  。

![3](https://github.com/littlelittlewhite09/Bearing_Fault_recognition/raw/main/Screenshots/1.png) 
将上述残差块堆叠起来，形成深度残差网络，如图7所示（批量归一化层并入了前面的卷积层，这里省略）。振动信号先通过一层简单的卷积层，该卷积层有16个3*1的卷积核，步长为1，不改变时序信号的长度；接着依次通过12个残差块，当步长为n（n！=1）时，时序信号的长度则变为原来的1/n；最后再依次通过全局平均池化层和全连接层，得到最终的预测结果。

![4](https://github.com/littlelittlewhite09/Bearing_Fault_recognition/raw/main/Screenshots/1.png) 
