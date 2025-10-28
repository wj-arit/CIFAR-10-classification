import torch
import torch.nn as nn
from sipbuild.generator.outputs import output_api


class MLP_CIFAR10(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(512,10)
        #self.fc4 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(256,10)
        #self.fc5 = nn.Linear(128, 64)
        #self.fc5 = nn.Linear(128, 10)
        #self.fc6 = nn.Linear(64,32)
        #self.fc7 = nn.Linear(32,10)
        self.relu = nn.ReLU()

    def _init_weights(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)
        #nn.init.zeros_(self.fc5.bias)
        #nn.init.zeros_(self.fc6.bias)
        #nn.init.zeros_(self.fc7.bias)

    def forward(self,input_data: torch.Tensor)->float:
        input_data = input_data.flatten(start_dim = 1)
        input_data = self.relu(self.fc1(input_data))
        input_data = self.relu(self.fc2(input_data))
        #output_data = self.fc3(input_data)
        input_data = self.relu(self.fc3(input_data))
        #output_data = self.fc3(input_data)
        #input_data = self.relu(self.fc4(input_data))
        output_data = self.fc4(input_data)
        #input_data = self.relu(self.fc5(input_data))
        #output_data = self.fc5(input_data)
        #input_data = self.relu(self.fc6(input_data))
        #output_data = self.fc7(input_data)

        return output_data