import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 原始CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 4, 6, 28, 28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):       
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ResNet18
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), padding=0)
        self.batch_normal = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normal(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        if self.is_downsample:
            self.down_sample = DownSample(in_channels, out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.is_downsample:
            x = self.down_sample(x)
        out = self.relu2(out + x)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.residual_layer1 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1)
        )
        self.residual_layer2 = nn.Sequential(
            BasicBlock(64, 128, 2 ,True),
            BasicBlock(128,128, 1)
        )
        self.residual_layer3 = nn.Sequential(
            BasicBlock(128, 256, 2 ,True),
            BasicBlock(256, 256, 1)
        )
        self.residual_layer4 = nn.Sequential(
            BasicBlock(256, 512, 2 ,True),
            BasicBlock(512, 512, 1)
        )
        self.fc = nn.Linear(512, 10)

    def forward(self, x):                # 3*32*32
        x = self.conv1(x)                # 64*32*32
        x = self.residual_layer1(x)      # 64*32*32
        x = self.residual_layer2(x)      # 128*16*16
        x = self.residual_layer3(x)      # 256*8*8
        x = self.residual_layer4(x)      # 512*4*4
        x =  F.avg_pool2d(x, 4)          # 512*1*1
        x = x.view(x.size(0), -1)        # 512
        x = self.fc(x)
        return x

# DenseNet
class Bottleneck(nn.Module):
    def __init__(self, in_channels, k):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, 4 * k, kernel_size=1,stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * k)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d( 4 * k, k, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = torch.cat([out, x], 1)
        return out
    
class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, k):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            if i == 0:
                self.bottlenecks = nn.Sequential(
                    Bottleneck(in_channels+i*k, k)
                )
            else:
                bottleneck = Bottleneck(in_channels+i*k, k)
                self.bottlenecks.add_module("bottleneck%d" % (i+1), bottleneck)
    
    def forward(self,x):
        return self.bottlenecks(x)
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
        
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    def __init__(self, k=12, block_config=(3, 3, 3), init_channels=24, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
 
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, init_channels, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
 
        in_channels = init_channels
        for i, num_layers in enumerate(block_config):
            denseblock = DenseBlock(num_layers, in_channels, k)
            if i == 0:
                self.denseblocks = nn.Sequential(
                    denseblock
                )
            else:
                self.denseblocks.add_module("denseblock%d" % (i + 1), denseblock)
            in_channels += num_layers * k
            if i != len(block_config) - 1:
                transition = Transition(in_channels, int(in_channels*reduction))
                self.denseblocks.add_module("transition%d" % (i + 1), transition)
                in_channels = int(in_channels * reduction)
 
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_channels, num_classes)
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.denseblocks(out)
        out = self.bn(out)
        out = self.relu(out)
        out = F.avg_pool2d(out,4).view(out.size(0), -1)
        out = self.fc(out)
        return out

# SE-ResNet18
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class SEResNet18(nn.Module):
    def __init__(self):
        super(SEResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1_block1 = BasicBlock(64, 64, stride=1)
        self.layer1_block2 = BasicBlock(64, 64, stride=1)
        self.se1 = SEBlock(64)

        self.layer2_block1 = BasicBlock(64, 128, stride=2, is_downsample=True)
        self.layer2_block2 = BasicBlock(128, 128, stride=1)
        self.se2 = SEBlock(128)

        self.layer3_block1 = BasicBlock(128, 256, stride=2, is_downsample=True)
        self.layer3_block2 = BasicBlock(256, 256, stride=1)
        self.se3 = SEBlock(256)

        self.layer4_block1 = BasicBlock(256, 512, stride=2, is_downsample=True)
        self.layer4_block2 = BasicBlock(512, 512, stride=1)
        self.se4 = SEBlock(512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        x = self.se1(x)

        x = self.layer2_block1(x)
        x = self.layer2_block2(x)
        x = self.se2(x)

        x = self.layer3_block1(x)
        x = self.layer3_block2(x)
        x = self.se3(x)

        x = self.layer4_block1(x)
        x = self.layer4_block2(x)
        x = self.se4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Res2Net18
class Bottle2neck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=4, base_width=26, is_downsample=False):
        super(Bottle2neck, self).__init__()
        assert out_channels % scale == 0, "out_channels must be divisible by scale"
        self.scale = scale
        self.width = out_channels // scale
        self.is_downsample = is_downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, self.width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width * scale)

        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(self.width) for _ in range(scale - 1)
        ])

        self.conv3 = nn.Conv2d(self.width * scale, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if is_downsample or in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) if stride == 2 else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.chunk(out, self.scale, dim=1)

        outputs = []
        for i in range(self.scale):
            if i == 0:
                out_i = spx[i]
            else:
                out_i = spx[i] + outputs[i - 1]
                out_i = self.relu(self.bns[i - 1](self.convs[i - 1](out_i)))
            outputs.append(out_i)

        out = torch.cat(outputs, dim=1)

        if self.pool is not None:
            out = self.pool(out)

        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)

class Res2Net18(nn.Module):
    def __init__(self, num_classes=10):
        super(Res2Net18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            Bottle2neck(64, 64, stride=1, is_downsample=False),
            Bottle2neck(64, 64, stride=1, is_downsample=False)
        )
        self.layer2 = nn.Sequential(
            Bottle2neck(64, 128, stride=2, is_downsample=True),
            Bottle2neck(128, 128, stride=1, is_downsample=False)
        )
        self.layer3 = nn.Sequential(
            Bottle2neck(128, 256, stride=2, is_downsample=True),
            Bottle2neck(256, 256, stride=1, is_downsample=False)
        )
        self.layer4 = nn.Sequential(
            Bottle2neck(256, 512, stride=2, is_downsample=True),
            Bottle2neck(512, 512, stride=1, is_downsample=False)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train(epoch, log_interval=200):
    # Set model to training mode
    net.train()
    
    
    running_loss = 0.0
    for batch_idx, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), running_loss/log_interval))
            running_loss = 0.0

def validate(loss_vector, accuracy_vector):
    net.eval()
    val_loss, correct = 0, 0

    
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            # calculate outputs by running images through the network
            outputs = net(data)
            val_loss += criterion(outputs, target).data.item()
            # the class with the highest energy is what we choose as prediction
            pred = outputs.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            
        val_loss /= len(testloader)
        loss_vector.append(val_loss)
        
        accuracy = 100. * correct.to(torch.float32) / len(testloader.dataset)
        accuracy_vector.append(accuracy)
        
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(testloader.dataset), accuracy))


if __name__ == "__main__":

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=12)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=12)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    epochs = 20
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------- Origin CNN --------------------------------------------------------


    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    origin_lossv, origin_accv = [], []

    best_acc = 0.0
    no_improve_count = 0
    early_stop_patience = 3

    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(origin_lossv, origin_accv)

        # early stopping (3 epochs without improvement)
        current_acc = origin_accv[-1]
        if current_acc > best_acc:
            best_acc = current_acc
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} consecutive epochs)")
            break



    # --------------------------------------------------------------- ResNet18 --------------------------------------------------------

    net = ResNet18().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    resnet18_lossv, resnet18_accv = [], []

    best_acc = 0.0
    no_improve_count = 0
    early_stop_patience = 3

    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(resnet18_lossv, resnet18_accv)   

        # early stopping (3 epochs without improvement)
        current_acc = resnet18_accv[-1]
        if current_acc > best_acc:
            best_acc = current_acc
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} consecutive epochs)")
            break

    # --------------------------------------------------------------- DenseNet --------------------------------------------------------


    net = DenseNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    densenet_lossv, densenet_accv = [], []

    best_acc = 0.0
    no_improve_count = 0
    early_stop_patience = 3

    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(densenet_lossv, densenet_accv)   

        # early stopping (3 epochs without improvement)
        current_acc = densenet_accv[-1]
        if current_acc > best_acc:
            best_acc = current_acc
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} consecutive epochs)")
            break

    # --------------------------------------------------------------- SE-ResNet18 --------------------------------------------------------

    net = SEResNet18().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    se_resnet18_lossv, se_resnet18_accv = [], []

    best_acc = 0.0
    no_improve_count = 0
    early_stop_patience = 3

    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(se_resnet18_lossv, se_resnet18_accv)   

        # early stopping (3 epochs without improvement)
        current_acc = se_resnet18_accv[-1]
        if current_acc > best_acc:
            best_acc = current_acc
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} consecutive epochs)")
            break

    # plt.figure(figsize=(8, 6))  
    # plt.plot(np.arange(1, len(se_resnet18_lossv)+1), se_resnet18_lossv)  
    # plt.title('Validation Loss')  
    # plt.legend()  
    # plt.savefig('se-resnet18_validation_loss.png')

    # plt.figure(figsize=(8, 6))  
    # plt.plot(np.arange(1, len(se_resnet18_accv)+1), se_resnet18_accv)  
    # plt.title('Validation Accuracy')  
    # plt.legend()  
    # plt.savefig('se-resnet18_validation_accuracy.png')

    # --------------------------------------------------------------- Res2Net18 --------------------------------------------------------

    net = Res2Net18().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    res2net18_lossv, res2net18_accv = [], []
    best_acc = 0.0
    no_improve_count = 0
    early_stop_patience = 3

    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(res2net18_lossv, res2net18_accv)   

        # early stopping (3 epochs without improvement)
        current_acc = res2net18_accv[-1]
        if current_acc > best_acc:
            best_acc = current_acc
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} consecutive epochs)")
            break

    
    # plt.figure(figsize=(8, 6))  
    # plt.plot(np.arange(1, len(res2net18_lossv)+1), res2net18_lossv)  
    # plt.title('Validation Loss')  
    # plt.legend()  
    # plt.savefig('res2net18_validation_loss.png')

    # plt.figure(figsize=(8, 6))  
    # plt.plot(np.arange(1, len(res2net18_accv)+1), res2net18_accv)  
    # plt.title('Validation Accuracy')  
    # plt.legend()  
    # plt.savefig('res2net18_validation_accuracy.png') 

    plt.figure(figsize=(8, 6))  
    plt.plot(np.arange(1, epochs+1), origin_lossv, label='Origin')  
    plt.plot(np.arange(1, epochs+1), resnet18_lossv, label='ResNet18') 
    plt.plot(np.arange(1, epochs+1), densenet_lossv, label='DenseNet')  
    plt.plot(np.arange(1, epochs+1), se_resnet18_lossv, label='SE-ResNet18')  
    plt.title('Validation Loss')  
    plt.legend()  
    plt.savefig('overall_validation_loss.png')

    plt.figure(figsize=(8, 6))  
    plt.plot(np.arange(1, epochs+1), origin_accv, label='Origin')  
    plt.plot(np.arange(1, epochs+1), resnet18_accv, label='ResNet18') 
    plt.plot(np.arange(1, epochs+1), densenet_accv, label='DenseNet')  
    plt.plot(np.arange(1, epochs+1), se_resnet18_accv, label='SE-ResNet18')  
    plt.title('Validation Accuracy')  
    plt.legend()  
    plt.savefig('overall_validation_accuracy.png') 