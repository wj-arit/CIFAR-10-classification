import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


# trainset, dataset
train_transform = transforms.Compose([
    transforms.RandomCrop(32,4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_DS = datasets.CIFAR10('../data',train=True,download=False,transform=train_transform)
test_DS = datasets.CIFAR10('../data',train=False,download=False,transform=test_transform)


# testdataLoader
BATCH_SIZE = 64
test_DL = torch.utils.data.DataLoader(test_DS,batch_size=BATCH_SIZE,shuffle=True)
if __name__ == '__main__':
    print(train_DS.data.shape)
    print(train_DS.classes)
    print(train_DS.class_to_idx)