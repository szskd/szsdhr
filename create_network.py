import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        #nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0)
        #in_channels(输入维度),out_channels(输出维度)
        self.conv1=nn.Conv2d(1,6,3)
        self.conv2=nn.Conv2d(6,16,3)
        self.fc1=nn.Linear(16*6*6,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        # max pooling over a (2,2) window
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # if the size is a square ,only specify a single number
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(X))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return  num_features

net=Net()
print(net)

params=list(net.parameters())
print(params[0].size())   #conv1's.weight
#torch.Size([6,1,3,3])