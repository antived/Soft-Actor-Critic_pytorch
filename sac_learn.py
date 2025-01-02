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
        q_final_val = torch.min(q_val_1,q_val_2)
        log_probs = log_probs.view(-1)
        return log_probs, q_final_val
    

    def networks_learn(self):
        #this function is going to be used for the means of training the networks.
        '''the main aim of training the critic and the value networks is for the means of training the policy/actor network.
            This function if for the means of training on one sampled mini-batch.'''
        
        if self.memory.mem_counter < self.batch_size:
            return
    
        #else: when the buffer is full, we can now sample a batch with the size of batch_size.
        states,actions,rewards,states_,dones = self.memory.sample_state(self.batch_size)
        
        states = torch.tensor(states, dtype = torch.float32).to(self.actor.device)
        action = torch.tensor(actions, dtype = torch.float32).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype = torch.float32).to(self.actor.device)
        states_ = torch.tensor(states_, dtype =torch.float32).to(self.actor.device)
        dones = torch.tensor(dones).to(self.actor.device)

        values = self.value.forward(states).view(-1)
        values_ = self.value.forward(states_).view(-1)
        values_[dones] = 0.0 #values_ * (~dones)

        '''the update for the Q is : target(i)(s,a) = reward + gamma*(V(s)(i+1))-alpha*log_prob).
        for the s vals that dont have s(i+1) we set V to 0.'''
        #first we will train the value network and then the policy and then finally the critic networks. finally, we will update the 
        #params of the target_value network.

        #training the value net. loss is the mse loss of V(s) - expected(Q(s,a)-log(pi(a|s)))
        actions,log_pro = self.actor.sample_data(states,False)
        q_val_1 = self.critic1.forward(state=states,actions= actions).view(-1)
        q_val_2 = self.critic2.forward(state= states,actions = actions).view(-1)
        q_not_overestim = torch.min(q_val_1, q_val_2)
        #we have to get the log probs and the actions also.
        log_pro = log_pro.view(-1)
        #print("Q_not_overstim:", q_not_overestim)
        #print("log_pro", log_pro)
        '''this can directly be used since same number of elements, Q(s,a) is state-action pair,
            and also in the case of the log_probs, its log(pi(a|s)), so again its state-actions pairs.'''
        self.value.optimizer.zero_grad()
        q_new_exp = q_not_overestim - log_pro
        loss_exp = 0.5 * F.mse_loss(values, q_new_exp)
        loss_exp.backward(retain_graph=True) #retain_graph had been removed
        self.value.optimizer.step()  #this trains the value network on the mini batch.

        #now we can train the critic since we have got the updated V(s)(i) function value.
        
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_target = self.reward_factor*rewards + self.gamma*(values_).clone().detach()
        q1_old = self.critic1.forward(states,actions).view(-1)
        q2_old = self.critic2.forward(states,actions).view(-1)
        loss_val1 = 0.5*F.mse_loss(q1_old, q_target)
        loss_val2 = 0.5*F.mse_loss(q2_old,q_target)
        loss_val_fin = loss_val1 + loss_val2
        loss_val_fin.backward(retain_graph=True)
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        #now finally, we can train the policy.
        log_probs_fin, q_vals_fin = self.log_probs_q_val(states)
        train_exp = log_probs_fin - q_vals_fin
        train_exp = torch.mean(train_exp)
        loss_new = train_exp
        self.actor.optimizer.zero_grad()
        loss_new.backward() #retain_graph had been removed.
        self.actor.optimizer.step()

        self.update_net_params()
