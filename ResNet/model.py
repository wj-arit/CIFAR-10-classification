import torch
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_connection = nn.Sequential()
        self.weight_initialize()
    def weight_initialize(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output += self.skip_connection(x)
        output = self.relu(output)

        return output

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,in_channel,bottleneck_channel,stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel,bottleneck_channel,kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channel)
        self.conv2 = nn.Conv2d(bottleneck_channel,bottleneck_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channel)
        self.conv3 = nn.Conv2d(bottleneck_channel,bottleneck_channel*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channel != bottleneck_channel * self.expansion:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channel, bottleneck_channel*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(bottleneck_channel*self.expansion)
            )
        else:
            self.skip_connection = nn.Sequential()
        self.weight_initialize()

    def weight_initialize(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.skip_connection(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,depth):
        super().__init__()
        self.depth = depth
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        block = ResidualBlock if depth < 50 else BottleNeck
        if depth < 50:
            n = int((depth - 2) / 6)
        else:
            n = int((depth - 2) / 9)
        self.relu = nn.ReLU(inplace=True)
        self.expansion = block.expansion if hasattr(block, 'expansion') else 1
        self.layer1 = self.make_layer(block, 16, 16, n, stride=1)
        self.layer2 = self.make_layer(block, 16 * self.expansion, 32, n, stride=2)
        self.layer3 = self.make_layer(block, 32 * self.expansion, 64, n, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        final_channels = 64 * self.expansion
        self.fc1 = nn.Linear(final_channels, 10)
        self.weight_initialize()
    def weight_initialize(self):
        nn.init.kaiming_normal_(self.conv1.weight,mode='fan_out',nonlinearity='relu')
        nn.init.constant_(self.bn1.weight,1)
        nn.init.constant_(self.bn1.bias,0)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def make_layer(self,block,in_channels,out_channels,num_blocks,stride):
        layers = []
        # first block: apply stride
        layers.append(block(in_channels,out_channels,stride))
        in_channels = out_channels * self.expansion
        for _ in range(1,num_blocks):
            layers.append(block(in_channels,out_channels,stride=1))
        return nn.Sequential(*layers)

    def forward(self,x):

        output = self.bn1(self.conv1(x))
        output = self.relu(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = torch.flatten(self.avg_pool(output),start_dim=1)
        output = self.fc1(output)
        return output

if __name__ == '__main__':
    model = ResNet(20)
    print(model)