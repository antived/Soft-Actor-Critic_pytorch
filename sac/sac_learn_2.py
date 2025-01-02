import torch
import os
import torch.nn.functional as F
import numpy as np
from buffer_mem import buffer
from networks import criticNetwork,valueNet, ActorNet


class Agent():
    def __init__(self, alpha = 0.0003, beta = 0.0003,tau = 0.005,env = None,
                    batch_size= 256,fc1_units = 256,max_size = 1000000,
                    fc2_units = 256,n_actions = 2,input_dims = [8], gamma = 0.99,max_factor = 2):
        self.alpha =  alpha
        self.beta = beta
        self.gamma = 0.99
        self.max_factor = max_factor
        self.tau = tau
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.reward_factor = max_factor
        self.scale = max_factor
        self.epsilon = 1e-8
        self.memory = buffer(input_size = input_dims,n_actions = n_actions, max_cap = max_size)
        self.actor = ActorNet(learning_rate = self.alpha, max_factor=env.action_space.high,in_dims = self.input_dims,
                              name = 'Actor', n_actions = n_actions)
        self.critic1= criticNetwork(in_dims = input_dims, learning_rate = self.beta, no_actions= n_actions, name = "critic1")
        self.critic2 = criticNetwork(in_dims = input_dims, learning_rate = self.beta, no_actions = n_actions, name = "critic2")
        self.value = valueNet(learning_rate = self.alpha, input_dims= input_dims, name = "value")   
        self.target_value = valueNet(learning_rate = self.alpha, input_dims = input_dims, name = "target_val")

    def get_actions(self, state):
        #we have to use the actor network to get the tensor with the possible actions.
        #the state must be sent to the gpu

        in_states = torch.tensor([state],dtype = torch.float32).to(self.actor.device)
        actions,_ = self.actor.sample_data(in_states, reparam = False)

        #now the actions tensor is on the gpu, in the computational graph so we must extract it from it
        actions_fin = actions.cpu().detach().numpy()[0]
        return actions_fin
    
    def save_data(self, state, action, reward, next_state, terminal):
        self.memory.store_trans(state,action,next_state,reward,done = terminal)

    def save_model(self):
        print("....SAVING THE MODEL.....")
        self.actor.save_model()
        self.critic1.save_chkpt()
        self.critic2.save_chkpt()
        self.value.save_chkpnt()

    def load_models(self):
        print("....LOADING THE MODELS....")
        self.actor.load_model()
        self.critic1.load_chkpt()
        self.critic2.load_chkpt()
        self.value.load_chkpnt()

    def update_net_params(self,tau = None):
        '''the entire idea behind using a seperate target value function for the means of getting the value of the
            V(s)(i+1) is that, in a case in which we decide not to do that, while trying to update the value of the target for the critic network, we will 
            be using the value of the V(s)(i+1) from the newtwork that has just been updated. This will then be used to fine tune the Q value that will again 
            be used to fine turn the value network. this lead to updated that are very fast and also, is forming a feedback loop. So, the updat for the target Q 
             value must be made using the old parameter of the value network. This makes sure that both the value net and the critic nets get updated at the same
            rates. '''
        
        #this function is for the means of updating the learnable params of the target_value_net

        #the first time, we want the value net and the target_net to give exactly the same values.
        if tau == None:
            tau = 1
        else:
            tau = self.tau

        value_learnable_params = self.value.state_dict()
        target_value_learnable_params = self.target_value.state_dict()
        for name in value_learnable_params:
            value_learnable_params[name] = tau * value_learnable_params[name].clone() + (1-tau)*target_value_learnable_params[name].clone()
            #imp to use the .clone() here since otherwise the value_learnable_params etc can get modified.

        self.target_value.load_state_dict(value_learnable_params)


    def log_probs_q_val(self,states):
        actions, log_probs = self.actor.sample_data(states, True)
        q_val_1 = self.critic1.forward(states,actions).view(-1)
        q_val_2 = self.critic2.forward(states,actions).view(-1)
        q_final_val = torch.min(q_val_1,q_val_2).clone()
        log_probs = log_probs.view(-1)
        return log_probs, q_final_val
    

    def networks_learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        state,action,reward,new_state,done = self.memory.sample_state(self.batch_size)

        reward = torch.tensor(reward,dtype = torch.float32).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state = torch.tensor(state,dtype = torch.float32).to(self.actor.device)
        state_ = torch.tensor(new_state,dtype = torch.float32).to(self.actor.device)
        action = torch.tensor(action,dtype = torch.float32).to(self.actor.device)

        value = self.value.forward(state).view(-1)
        value_ = self.value.forward(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_data(state, reparam= False)
        log_probs = log_probs.view(-1)
        q1_val = self.critic1.forward(state,actions)
        q2_val = self.critic2.forward(state,actions)
        q_fin_val = torch.min(q1_val, q2_val)
        q_fin_val = q_fin_val.view(-1)

        self.value.optimizer.zero_grad()
        val_tar = q_fin_val - log_probs
        #print(log_probs)
        loss = 0.5 * F.mse_loss(value,val_tar)
        loss.backward(retain_graph= True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_data(state,reparam= True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state,actions)
        q2_new_policy = self.critic1.forward(state,actions)
        q_val = torch.min(q1_new_policy, q2_new_policy)
        q_val = q_val.view(-1)

        actor_loss = log_probs - q_val
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*(value_).clone().detach()
        q1_old = self.critic1.forward(state,action).view(-1)
        q2_old = self.critic2.forward(state,action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old,q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old,q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_net_params()  
