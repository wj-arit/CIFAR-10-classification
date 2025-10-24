import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

transform = transforms.ToTensor()
train_DS = datasets.CIFAR10(root='../data',train=True, download=False,transform=transform)
test_DS = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)
print(test_DS.classes)
print(test_DS.class_to_idx)
print(train_DS.data.shape)
#print(train_DS.targets) # label about train dataset
#plt.imshow(train_DS.data[0])
#plt.show()

BATCH_SIZE = 32
train_DL = torch.utils.data.DataLoader(train_DS,batch_size=BATCH_SIZE,shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS,batch_size=BATCH_SIZE,shuffle=True)

x_batch, y_batch = next(iter(test_DL))
print(type(train_DS.data))
print(type(x_batch))
print(x_batch.shape)
print(y_batch.shape) # labels
plt.imshow(x_batch[0].permute(1,2,0)) # permute는 reshape처럼 메모리 재배치 아니라 메모리 그대로 두고 차원 순서 스위칭
plt.show()
print(x_batch[0].dtype)
print(y_batch[0].dtype)

#ToTensor의 역할
#1. tensor로 바꿔준다.
#2. 개체행열로 바꿔준다
#3. 0~1 사이로 바꿔준다.