import torch
import torch.nn as nn
class MlpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,10)
        self.relu = nn.ReLU()
        # initialize weight and bias
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self,image):
        image = torch.flatten(image,start_dim=1)
        input_data = self.relu(self.fc1(image))
        hidden_1 = self.relu(self.fc2(input_data))
        hidden_2 = self.relu(self.fc3(hidden_1))
        output = self.fc4(hidden_2)

        return output

