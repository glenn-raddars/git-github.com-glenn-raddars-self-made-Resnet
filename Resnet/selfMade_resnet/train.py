import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from model import resnet34,resnet101
import torchvision.models.resnet

def main():
    #define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #定义图像预处理器,这里训练resnet
    data_transform = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.Grayscale(3),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))])



    #下载训练数据集
    train_dataset = datasets.MNIST(root = '../Resnet/dataset', train=True, transform=data_transform, download=True)
    test_dataset = datasets.MNIST(root = '../Resnet/dataset', train=False, transform=data_transform, download=True)

    #数据集的加载
    train_loader = DataLoader(dataset = train_dataset,batch_size=64,shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset = test_dataset,batch_size=64,shuffle=False, drop_last=False)

    #实例化网络
    net = resnet34()#默认分类数是1000
    net = net.to(device)
    model_weight_path = '../Resnet/pre_model/resnet34-b627a593.pth'#官方给的预训练权重，我们主要是做迁移学习，因此是在官方权重的基础上进行训练
    net.load_state_dict(torch.load(model_weight_path),strict=False)
    #然后把全连接层的输出通道数修改为10
    inchannel = net.fc.in_features
    net.fc = nn.Linear(inchannel, 10)

    #损失函数
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0001)

    #最佳命中率
    best_acc = 0.0
    #te length of test_loader
    val_num = len(test_loader)
    #模型存储路径
    save_path = '../Resnet/FinalModel/resnet34.pth'

    for epoch in range(100):
        #train
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader,start=0):
            image, label = data
            optimizer.zero_grad()

            output = net(image.to(device))
            loss = loss_function(output, label.to(device))

            loss.backward()
            optimizer.step()

            #一轮下来的总损失
            running_loss += loss.item()
            #训练进度
            rate = (step + 1)/len(train_loader)

            a = "*"*int(rate*50)
            b = "."*int((1-rate)*50)
            print("\rtarin loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss))

        #验证，并保存最优模型的参数
        net.eval()
        acc = 0.0
        with torch.no_grad():
            for image, label in test_loader:
                output = net(image.to(device))
                pridect_y = torch.max(output,dim=1)[1]#返回最大值的索引
                acc += (pridect_y == label.to(device)).sum().item()

            accurate_test = acc / val_num
            if accurate_test > best_acc:
                best_acc = accurate_test
                torch.save(net.state_dict(), save_path)

            print('[epoch %d] train_loss: %.3f test_accuracy: %.3f'%
                    (epoch + 1, running_loss / step, acc / val_num))


#开始训练
if __name__ == "__main__":
    main()