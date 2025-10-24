import torch
from torch import optim
import torch.nn as nn
from model import MLP_CIFAR10
from datasets import train_DL
from train import Train

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
LR = 0.001
EPOCH = 200

criterion = nn.CrossEntropyLoss()
model = MLP_CIFAR10()
optimizer = optim.Adam(model.parameters(),lr=LR,)

loss_history = Train(model,train_DL,criterion,optimizer,device=DEVICE, epoch=EPOCH)

torch.save(model.state_dict(),'mlp_cifar10.pt')
