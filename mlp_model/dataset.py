import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


# train setting + weak_augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32,4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #convert np as tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)) # cifar-10 normalization
])

# validation setting
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])

# test setting
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])

full_train_dataset = datasets.CIFAR10('/home/user/CIFAR-10-classification/data',download=False,train=True,transform=train_transform)
full_test_dataset = datasets.CIFAR10('/home/user/CIFAR-10-classification/data',download=False,train=False,transform=test_transform)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

# 스플릿 비율 고정
torch.manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset,(train_size,val_size))

# set batch size
train_batch_size = 64
val_batch_size = 1000
test_batch_size = 1000
# loader correspond to dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(full_test_dataset, batch_size=test_batch_size, shuffle=False)

if __name__ == '__main__':
    print(len(full_train_dataset))
    print(train_size, val_size)
    print(type(full_test_dataset),type(train_dataset))
    x1,y1 = train_dataset[0]
    x2,y2 = val_dataset[0]
    print(x1.shape)
    print(full_train_dataset.class_to_idx)
    print(y1)
    plt.imshow(x1.permute(1,2,0))
    plt.show()
