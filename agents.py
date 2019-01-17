import numpy as np
import random
from collections import namedtuple, deque
from multiprocessing import Lock

from mymodel import ActorNetwork, CriticNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUF_SIZE = 1000000
N = 5
TAU = 1e-3
batch_size = 256
GAMMA = 0.99
ALPHA = 1e-4
BETA = 1e-4
TD_EPSILON = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, seed):
        self.actor_network = ActorNetwork(state_size, action_size, seed).to(device)
        self.critic_network = CriticNetwork(state_size, action_size, seed).to(device)
        
    def ret_act(self,states):
        return(self.actor_network(states))
    
    def ret_qval(self,states, actions):
        return(self.critic_network(states, actions))
        
        
class Learner():
    def __init__(self, state_size, action_size, seed):
        
        self.actor_network = ActorNetwork(state_size, action_size, seed).to(device)
        self.actor_target_network = ActorNetwork(state_size, action_size, seed).to(device)
        for target_param, local_param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
                    target_param.data.copy_(local_param.data)
        
        self.critic_network = CriticNetwork(state_size, action_size, seed).to(device)
        self.critic_target_network = CriticNetwork(state_size, action_size, seed).to(device)
        for target_param, local_param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
                    target_param.data.copy_(local_param.data)
        
        self.rollout_length = N
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.replaymem = ReplayBuffer(BUF_SIZE, self.rollout_length)
        
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=ALPHA)
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr=BETA)
        self.tau = TAU
        
    def learn(self):
        st_init, st_final, action, done, reward, priority, indices = self.replaymem.sample(self.batch_size)  #sampling from 
                                                                                                          # minibatch
        priority = np.reshape(priority, (priority.shape[0],1))
        root_priority = np.sqrt(priority)    # squar rooting priorities, since we are using mse loss and 
        #print(root_priority.shape)             # i am multiplying both the target and predicted values with 
                                                # priorities beforehand, so ((a-b)^2)/k = (a/p - b/p)^2 
                                                # where p = sqrt(k)
        
        tensor_rp = torch.from_numpy(root_priority).float().to(device)
        q_initialst = ((self.critic_network(st_init, action))/tensor_rp).float().to(device)  # qvalues of (s(i),a(i))

        action_finalst = self.actor_target_network(st_final)                # action_values of final state(s(i+N))
        q_finalst = self.critic_target_network(st_final, action_finalst).detach().cpu().numpy()   # q_value of final (s(i+N),a_final)
        #print(done.shape)
        done = np.reshape(done, (done.shape[0],1))
        
        q_finalst = (1-done)*q_finalst
        #print(q_finalst.shape)
        
        disc = 1
        gamma = self.gamma
        g = []
        for _ in range(self.rollout_length):
            g.append(disc)
            disc *= gamma
            
        g = np.array(g)
        
        disc_reward = g*reward                                           # discounted reward
        
        added_reward = np.reshape(np.sum(disc_reward, axis = 1), (self.batch_size,1))
        #print(added_reward.shape)
        #print(q_finalst.shape)
        
        y_val = (added_reward + disc*gamma*q_finalst)/root_priority  # target value
        
        y_val = torch.from_numpy(y_val).float().to(device)
        #print(y_val.shape)
        #print(q_initialst.shape)
        td_error = ((torch.abs(y_val - q_initialst))*tensor_rp + TD_EPSILON).detach().cpu().numpy()
        td_error = np.reshape(td_error, (self.batch_size,))
        
        for i in range(self.batch_size):
            #self.replaymem.add(td_error[i], (st_init[i], st_final[i], action[i], done[i], reward[i]))
            self.replaymem.update(indices, td_error)
        
        critic_loss = (F.mse_loss(q_initialst, y_val))/BUF_SIZE   # mse loss between target and predicted value
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_network.parameters(), 1)  # gradient clipping
        self.optimizer_critic.step()
        
        
        action_init = self.actor_network(st_init)          # action values as predicted by actor(policy) network
        actor_loss = -self.critic_network(st_init, action_init).mean()   # q_values of (s(i), action_value) by critic network
                                                                  # negative sign because of gradient ascent
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        self.soft_update(self.actor_network, self.actor_target_network, self.tau)
        self.soft_update(self.critic_network, self.critic_target_network, self.tau)
        
    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        
        
class ReplayBuffer():
    def __init__(self, BUF_SIZE,N):
        self.N = N
        self.queue = deque(maxlen = BUF_SIZE)
        
    def add(self, priority, exp):
        #lock.acquire()
        self.queue.append([priority, exp])
        #lock.release()
    def update(self, indices, td_error):
        for i in range(len(indices)):
            self.queue[indices[i]][0] = td_error[i]
        
    def sample(self, batch_size, rollout_length = N):
        
        #lock.acquire()
        w = []
        for e in self.queue:
            w.append(e[0])
                  
        #print("yo")
#         print(w[0])
        w = np.array(w)
        w /= np.sum(w)
#         print(self.queue[0])
#         print(batch_size)
        
        indices = np.random.choice(np.arange(len(self.queue)), size = batch_size, replace = False, p = w)
        
        exp = []
        for i in indices:
            exp.append(self.queue[i])
        
#         print(exp[0])
#         indices = np.sort(indices)[::-1]
        
#         print(indices[0])
#         for i in indices:
#             self.queue.remove(self.queue[i])
        
            
        #lock.release()
        
        st_init = []
        st_final = []
        action = []
        done = []
        reward = []
        priority = []
        for [pr,x] in exp:
            priority.append(pr)
            st_init.append(x[0])
            st_final.append(x[1])
            action.append(x[2].detach())
            done.append(x[3])
            reward.append(x[4])
            
#         print(st_init[0])
#         print(st_final[0])
#         print(action[0])
#         print(done[0])
#         print(reward[0])
#         print(priority[0])
            
        st_init = torch.from_numpy(np.vstack(st_init)).float().to(device)
        st_final = torch.from_numpy(np.vstack(st_final)).float().to(device)
        action = torch.from_numpy(np.vstack(action)).float().to(device)
        done = np.array(done)
        reward = np.array(reward)
        priority = np.array(priority)
        priority = priority/np.sum(priority)
        
        return st_init, st_final, action, done, reward, priority, indices
    def __len__(self):
        return len(self.queue)
        
        
        
        
        
        
        

