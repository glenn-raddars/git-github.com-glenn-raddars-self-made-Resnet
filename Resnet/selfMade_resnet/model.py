import torch
import torch.nn as nn

#定义基本残差结构,18~36层
class BasicBlock(nn.Module):
    expansion = 1#这个参数用来定义输入通道和输出通道的关系，到50层往后会变成4倍

    def __init__(self, in_channel, out_channel, stride = 1, downsample=None):
        """downsample是自定义下采样参数,用来在每一层的残差结构的第一层降维"""
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size = 3, 
        stride=stride, padding=1, bias=False)#对于18-36层输入输出的每张图片，张量形状是不会变的，w,h经过网络后不变
        self.bn1 = nn.BatchNorm2d(out_channel)#放在Relu之前吗，用来缩小参数，不会影响通道数
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3,
        stride = 1, padding=1,bias = False)#也是一样，不仅张量不变，而且输入输出通道也不会变，但是对于50往上的层数就不一定了。
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self,x):
        """前向传播函数"""
        identity = x#定义千层网络输出特征
        if self.downsample is not None:
            identity = self.downsample(x)

        #下面开始一步一步执行网络结构
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #二层残差
        out = self.conv2(out)
        out = self.bn2(out)

        #合并浅层，深层特征值
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """50-152层resnet残差基本结构"""
    expansion = 4#每个残差结构最后一层输出是初始输入通道数的4倍
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1
        ,stride=1,bias=False)#这个卷积也不会改变每张图片的张量，w，h不变
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3
        ,padding=1,stride=1,bias=False)#也不会改变每张图片的张量，w,h不变
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1
        ,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,x):
        """前向传播函数"""
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        #一层一层执行残差结构
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out

#定义Resnet
class Resnet(nn.Module):
    """resnet网络"""
    def __init__(self,block,block_num,class_num=1000,include_top = True):#解释参数，block残差结构基本单元的类型，分为18-34层和50-152层
        #block_num是一个列表，表示了每一层残差结构包含了多少残差结构基本单元，class_num是最终分类的个数
        super(Resnet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64#无论是18-34还是50-152在经过第一层卷积和maxpooling之后都会变成64，所以输入通道数就是64
        
        #接下来定义第一层卷积层
        self.conv1 = nn.Conv2d(3, out_channels = self.in_channel, kernel_size=7,
        stride=2,padding=3,bias=False)#输入就是RGB图像3个通道，输出就是其它层的输入通道数64，然后卷积核大小是7，为了使输入图像的w,h减半，padding设为3
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(stride=2,kernel_size=3,padding=1)#最大池化层,最后输出图片的张量维度计算公式跟conv2d一样，这里也是w,h缩减为原来的一半

        #后面紧跟的是第二层，第三层，第四层，第五层
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)

        #最后一层，自适应平均池化，全连接层
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))#意思就是最后输出的每个通道上的图片的张量最后都变成了(1,1)，也就是一个点
            self.fc = nn.Linear(512*block.expansion, class_num)#最后全连接层的输入就是第五层残差结构输出的通道数，因为图片变成一个点了，就是512*1*1=512

        #如果要使用迁移学习，对卷基层进行一个初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")



    def _make_layer(self,block, channel, block_num, stride=1):#用来构造一层完整的残差层，用残差基本单元,channel是每个残差基本单元的第一层的输出通道数
        #其它参数的意义跟前面相同
        downsample = None#下采样函数
        if stride!=1 or self.in_channel != channel*block.expansion:#当步长不唯一或者输入通道数不等于输出通道数的时候，就是50-152层，需要生成下采样函数，这样才能相加
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel*block.expansion, kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )#Sequential会将网络结构按顺序打包最后会按顺序执行

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))#对于后面的每一层残差结构的第一层，都会使图像的张量减半，但是之后都是不变的，所以这一层带stride，后面的不带
        self.in_channel = channel*block.expansion

        for _ in range(1,block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)#将layers解包成非关键字参数，然后用Sequential组合成网络

    def forward(self,x):#正向传播函数
        #conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #conv2-conv5
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #conv6
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x,1)#将512*1*变成512*1
            fc = self.fc.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            x = fc(x)#分类

        return x

#定义34层resnet
def resnet34(class_num=1000, include_top=True):
    return Resnet(BasicBlock, block_num=[3,4,6,3], class_num=class_num, include_top=include_top)

#定义101层resnet
def resnet101(class_num=1000, include_top=True):
    return Resnet(Bottleneck, block_num=[3,4,23,3], include_top=include_top, class_num=class_num)

            


        



   
                                                         

