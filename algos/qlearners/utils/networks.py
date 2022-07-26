import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self,nUnits,nObs,nClasses,activation=nn.LeakyReLU()):
        super().__init__()
        self.activation = activation
        self.hidden_layers = [nn.Linear(nObs,nUnits[0])]
        for i in range(1,len(nUnits)):
            self.hidden_layers.append(nn.Linear(nUnits[i-1],nUnits[i]))

        print(self.hidden_layers)
        self.out = nn.Linear(nUnits[-1],nClasses)

    def forward(self, x):
        x = x.clone()
        for i in range(len(self.hidden_layers)):
            x = self.activation(self.hidden_layers[i](x))
        return self.out(x)

class DuelNetwork(nn.Module):
    def __init__(self,nUnits,nObs,nClasses,activation=nn.LeakyReLU()):
        super().__init__()
        self.activation = activation
        self.hidden_layers = [nn.Linear(nObs,nUnits[0])]
        self.nClasses = nClasses
        for i in range(1,len(nUnits)):
            self.hidden_layers.append(nn.Linear(nUnits[i-1],nUnits[i]))

        self.value = nn.Sequential( 
            nn.Linear(nUnits[-1],32),
            activation,
            nn.Linear(32,1)
        )

        self.advantage = nn.Sequential( 
            nn.Linear(nUnits[-1],32),
            activation,
            nn.Linear(32,nClasses)
        )

    def forward(self, x):
        x = x.clone()
        for i in range(len(self.hidden_layers)):
            x = self.activation(self.hidden_layers[i](x))

        value = self.value(x)
        advantage = self.advantage(x)

        q = value + advantage - advantage.mean()
        return q
