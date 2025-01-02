import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

#first we are going to be making the critic NN.
class criticNetwork(nn.Module):
    def __init__(self, in_dims, learning_rate, fc1_units = 256 , fc2_units = 256, 
                 no_actions = 2, name = "critic", chk_file = "tmp/sac"):
        super(criticNetwork, self).__init__()
        self.in_dims = in_dims
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.name = name
        self.beta = learning_rate
        self.num_actions = no_actions
        self.chk_file = chk_file
        self.epsilon = 1e-8
        if not os.path.exists(self.chk_file):
            print("The root directory does not exist")

        try:
            self.temp_file = os.path.join(self.chk_file, self.name+'_sac')
        except Exception as e:
            print("The file paths coould not be joined")
            return None
        
        self.layer1 = nn.Linear(self.in_dims[0] + self.num_actions, self.fc1_units)
        self.layer2 = nn.Linear(self.fc1_units, self.fc2_units)
        self.fin = nn.Linear(self.fc2_units, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = self.beta)
        if torch.backends.mps.is_available():
            print("mps was found !!!")
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, state, actions):
        x = self.layer1(torch.cat([state, actions], dim = 1))
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)

        x = self.fin(x)
        return x
    
    def save_chkpt(self):
        try:
            torch.save(self.state_dict(), self.temp_file)
        except Exception as e:
            print("The checkpoint could not be saved")
            return None
        
    def load_chkpt(self, state_dict):
        self.load_state_dict(torch.load(self.temp_file))

class valueNet(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_units = 256, fc2_units = 256,
                 name = "value", chkpnt_file = "tmp/sac"):
        super(valueNet, self).__init__()
        self.beta = learning_rate
        self.input_dims = input_dims
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.name = name
        self.chkpnt_file= chkpnt_file

        if not os.path.exists(self.chkpnt_file):
            print("This file path does not exist")
        
        try:
            self.full_path = os.path.join(self.chkpnt_file, self.name+'_sac')
        except Exception as e:
            print("The file could not be concatenated")
            return None
        
        self.layer1 = nn.Linear(*self.input_dims, self.fc1_units)
        self.layer2 = nn.Linear(self.fc1_units, self.fc2_units)
        self.fin = nn.Linear(self.fc2_units, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = self.beta)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)

        out = self.fin(x)
        return out
    
    def save_chkpnt(self):
        try:
            torch.save(self.state_dict(), self.full_path)
        except Exception as e:
            print("The checkpoint could not be saved")
            return None
        
    def load_chkpnt(self):
        self.load_state_dict(torch.load(self.full_path))

class ActorNet(nn.Module):
    '''this is a NN that is going to be used to return the mean and standard deviations
        of distribuitons of all the actions that are possible for a state.
        So, it takes in a state and then returns the means ans std devs of the possible actions.'''
    def __init__(self,learning_rate,max_factor,in_dims, fc1_units = 256, fc2_units = 256, name = "actor",
                 n_actions = 2, chkpnt_file = "tmp/sac"):
        super(ActorNet, self).__init__()
        self.alpha = learning_rate
        self.in_dims = in_dims
        self.max_factor = max_factor
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.name = name
        self.n_actions = n_actions 
        self.epsilon = 1e-8
        self.chkpnt_file = chkpnt_file
        self.reparam_noise = 1e-6

        if not os.path.exists(self.chkpnt_file):
            print("The path was not found")

        try:
            self.new_path = os.path.join(self.chkpnt_file, self.name+'_sac')
        except Exception as e:
            print("Error in file path concat")
            return None
    
        self.layer1 = nn.Linear(*self.in_dims, self.fc1_units)
        self.layer2 = nn.Linear(self.fc1_units, self.fc2_units)
        self.mean = nn.Linear(self.fc2_units, n_actions)
        self.std = nn.Linear(self.fc2_units, n_actions)


        self.optimizer = optim.Adam(self.parameters(), lr = self.alpha)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')


    def forward(self,data):
        x = self.layer1(data)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)

        mean = self.mean(x)
        std = self.std(x)

        std = torch.clamp(std, min = self.reparam_noise, max = 1)
        return mean, std 
        
    def sample_data(self,state, reparam = True):
        #this function will be used to sample the state actions pairs
        mean, std = self.forward(state)
        dists = Normal(mean, std)

        if reparam == True:
            actions = dists.rsample()
        else:
            actions = dists.sample()

        action = torch.tanh(actions)*torch.tensor(self.max_factor).to(self.device)
        log_probs = dists.log_prob(actions)
        log_probs -= torch.log(1-actions.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(dim = 1, keepdims = True)

        return action, log_probs
        
    def save_model(self):
        try:
            torch.save(self.state_dict(), self.new_path)
        except Exception as e:
            print("the checkpoint could not be saved")
            return None

    def load_model(self):
        self.load_state_dict(torch.load(self.new_path))

        
