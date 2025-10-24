import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(3072,1024),
                                    nn.ReLU(),
                                    nn.Linear(1024,512),
                                    nn.ReLU(),
                                    nn.Linear(512,256),
                                    nn.ReLU(),
                                    nn.Linear(256,128),
                                    nn.ReLU(),
                                    nn.Linear(128,56),
                                    nn.ReLU(),
                                    nn.Linear(56,28),
                                    nn.ReLU(),
                                    nn.Linear(28,10),
                                    )
    def foward(self,x):
        x =self.linear(x)