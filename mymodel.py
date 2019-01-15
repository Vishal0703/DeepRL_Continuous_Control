import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp


class ActorNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, h1_size = 128, h2_size = 128):
        
        
        super(ActorNetwork, self).__init__()
#         mp.set_start_method('spawn', force = True)
#         self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, action_size)
        
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x
    
    
class CriticNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, hs1_size = 128, ha1_size = 32, h2_size = 64):
        
        
        super(CriticNetwork, self).__init__()
#         mp.set_start_method('spawn', force = True)
#         self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, hs1_size)
        self.fca1 = nn.Linear(action_size, ha1_size)
        self.fc2 = nn.Linear(hs1_size + ha1_size, h2_size) 
        self.fc3 = nn.Linear(h2_size, action_size)
        
        
    def forward(self,x,a):
        x = F.relu(self.fcs1(x))
        a = F.relu(self.fca1(a))
        y = torch.cat((x,a), dim = 1)
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        return y