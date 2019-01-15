import numpy as np
import random
from collections import namedtuple, deque
from multiprocessing import Lock

from mymodel import ActorNetwork, CriticNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUF_SIZE = 10000
N = 5
TAU = 1e-3
batch_size = 256
GAMMA = 0.99
ALPHA = 1e-4
BETA = 1e-4
TD_EPSILON = 1e-3

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
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
        
    def learn(self,lock):
        st_init, st_final, action, done, reward, priority = self.replaymem.sample(self.batch_size, lock)  #sampling from 
                                                                                                          # minibatch
        
        root_priority = torch.sqrt(priority)    # squar rooting priorities, since we are using mse loss and 
                                                # i am multiplying both the target and predicted values with 
                                                # priorities beforehand, so ((a-b)^2)/k = (a/p - b/p)^2 
                                                # where p = sqrt(k)
                    
        q_initialst = (self.critic_network(st_init, action))/root_priority  # qvalues of (s(i),a(i))

        action_finalst = self.actor_target_network(st_final)                # action_values of final state(s(i+N))
        q_finalst = self.critic_target_network(st_final, action_finalst)    # q_value of final (s(i+N), afinal)
        q_finalst = q_finalst*(1-done)
        
        disc = 1
        gamma = self.gamma
        g = []
        for _ in range(self.rollout_length):
            g.append(disc)
            disc *= gamma
            
        g = np.array(g)
        
        disc_reward = g*reward                                           # discounted reward
        
        y_val = (torch.sum(disc_reward, dim = 1) + q_finalst)/root_priority  # target value
        
        td_error = torch.abs(y_val - q_initialst)*root_priority + TD_EPSILON
        
        for i in range(self.batch_size):
            self.replaymem.add(td_error[i], (st_init[i], st_final[i], action[i], done[i], reward[i]))
        
        critic_loss = (F.mse_loss(q_initialst, y_val))/BUF_SIZE   # mse loss between target and predicted value
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_network.parameters(), 1)  # gradient clipping
        self.optimizer_critic.step()
        
        
        action_init = self.actor_network(st_init)          # action values as predicted by actor(policy) network
        actor_loss = -self.critic_network(st_init, action_init)   # q_values of (s(i), action_value) by critic network
                                                                  # negative sign because of gradient ascent
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        soft_update(actor_network, actor_target_network, self.tau)
        soft_update(critic_network, critic_target_network, self.tau)
        
    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        
        
class ReplayBuffer():
    def __init__(self, BUF_SIZE,N):
        self.N = N
        self.queue = deque(maxlen = BUF_SIZE)
        
    def add(self, priority, exp, lock):
        lock.acquire()
        self.queue.append((priority, exp))
        lock.release()
        
    def sample(self, batch_size, lock, rollout_length = N):
        
        lock.acquire()
        w = []
        for e in self.queue:
            w.append(e[0])
            
        exp = random.choices(self.queue, weights = w, k = batch_size)
        
        for e in exp:
            self.queue.remove(e)
            
        lock.release()
        
        st_init = []
        st_final = []
        action = []
        done = []
        reward = []
        priority = []
        for (pr,x) in exp:
            priority.append(pr)
            st_init.append(x[0])
            st_final.append(x[1])
            action.append(x[2])
            done.append(x[3])
            reward.append(x[4])
            
        st_init = torch.from_numpy(np.array(st_init)).float().to(device)
        st_final = torch.from_numpy(np.array(st_final)).float().to(device)
        action = torch.from_numpy(np.array(action)).float().to(device)
        done = torch.from_numpy(np.array(done)).float().to(device)
        reward = torch.from_numpy(np.array(reward)).float().to(device)
        priority = torch.from_numpy(np.array(priority)).float().to(device)
        priority = priority/torch.mean(priority)
        
        return st_init, st_final, action, done, reward, priority
    def __len__(self):
        return len(self.queue)
        
        
        
        
        
        
        

