import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# train_dataset, test_dataset
transform = transforms.ToTensor()
train_DS = datasets.CIFAR10('../data',train=True,download=False,transform=transform)
test_DS = datasets.CIFAR10('../data',train=False,download=False,transform=transform)

# data_load
BATCH_SIZE = 32
train_DL = torch.utils.data.DataLoader(train_DS,batch_size = BATCH_SIZE, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS,batch_size = BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    # checking datasets
    # print(train_DS.data.shape)
    # print(test_DS.data.shape)
    # plt.imshow(test_DS.data[0])
    # plt.show()
    # print(test_DS.class_to_idx)
    # print(test_DS.targets[0])

    # checking data_loader
    print(type(train_DL))
    images, labels = next(iter(test_DL))
    print(images[0].shape)
    plt.imshow(images[0].permute(1,2,0))
    plt.show()
    print(labels[0])
