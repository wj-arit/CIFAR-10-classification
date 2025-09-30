import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()

# 훈련 데이터셋 (50,000장)
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform
)

# 테스트 데이터셋 (10,000장)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform
)
img_train = torch.stack([img.flatten() for img, _ in trainset])
label_train = torch.tensor([label for _, label in trainset])
if __name__ == '__main__':


    print(len(testset), len(trainset))
    print(type(testset))
    print(testset[0][0].shape)
    print(testset[0][1])
    img_train = torch.stack([img.flatten() for img, _ in trainset])
    label_train = torch.tensor([label for _, label in trainset])
    print(img_train.dim())
    print(label_train.dim())
