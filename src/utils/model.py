import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, h, num_classes, dims):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.dims = dims
        self.layers.append(nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1))
        self.dropout = nn.Dropout(p=0.3)
        for i in range(1, len(dims)):
            self.layers.append(nn.Conv2d(dims[i - 1], dims[i], kernel_size=3, stride=1, padding=1))
            h //= 2
        h = h//2
        #print(h)
        self.fc1 = nn.Linear(dims[-1]*h*h, 512)
        self.fc2 = nn.Linear(512, 10)
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        for i in range(len(self.dims)):
            x = nn.functional.relu(self.layers[i](x))
            x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        #x = nn.functional.softmax(self.fc3(x),dim = 1)
        x = self.fc3(x)
        return x