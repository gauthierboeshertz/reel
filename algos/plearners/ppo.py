import numpy as np
from .utils import LinearAnnealer,ExponentialAnnealer
import tqdm
import torch 
import torch.nn as nn
import wandb
from progress.bar import Bar
from array2gif import write_gif
import copy
from .utils import set_seed
from .utils import save_rewards_meanvar_plot,get_logger,MLP,ReplayMemory
import logging
import time 
from torch.distributions.categorical import Categorical
import torch.nn.functional as F 
import random 
from gym.spaces import Box, Discrete
from IPython import embed
import math
# https://spinningup.openai.com/en/latest/algorithms/ppo.html

class PPOExperience:
    def __init__(self,s,a,r,done,logp):
        self.s = s
        self.a = a
        self.r = r
        self.done = done
        self.logp = logp


class PPOMem:
    def __init__(self,N,batch_size):
        self.max_size = N
        self.batch_size = batch_size
        self.memory = []

    def add_exp(self,s,a,r,done,logp):
        
        r = torch.tensor(r).unsqueeze(0)
        done = torch.tensor(done).unsqueeze(0)
        if len(self.memory) == self.max_size:
            self.memory.pop(0)

        self.memory.append(PPOExperience(s,a,r,done,logp))


class PPO:
    def __init__(self, env, config):
        for k, v in config.items():
            setattr(self, k, v)
        print(config)
        self.env = env
        self.categorical_env = (type(self.env.action_space) == Discrete)
        self.config = copy.deepcopy(config)
        self.reset(self.seed)


    def reset(self, seed):
        self.seed = seed
        set_seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape

        obs_size = self.env.observation_space.n if 'n' in self.env.observation_space.__dict__ else self.env.observation_space._shape[0]
        if self.categorical_env:
            self.policy = MLP(self.nUnits, obs_size,self.env.action_space.n,activation= torch.nn.ELU).to(self.device)
        else:
            self.policy = MLP(self.nUnits, obs_size,self.env.action_space.shape[0]).to(self.device)
        self.value = MLP(self.nUnits, obs_size,1,activation= torch.nn.ELU).to(self.device)
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)

    def get_policy(self,s):
        if self.categorical_env:
            return Categorical(logits=self.policy(s))
        else:
            std = torch.exp(self.log_std)
            return Normal(self.policy(s), std)


    def get_action_logp(self,s):
        with torch.no_grad():
            pi = self.get_policy(s)
            a = pi.sample()
            logp = pi.log_prob(a).numpy()
            v = self.value(s).squeeze().numpy()
            a = a.numpy()
            return a, logp

    def process_epoch_mem(self, ep_mem):

        with torch.no_grad():
            rewards = torch.stack([exp.r for exp in ep_mem.memory])
            states = torch.stack([exp.s for exp in ep_mem.memory]).squeeze()
            actions = torch.stack([exp.a for exp in ep_mem.memory])

            for i in range(rewards.shape[0]-2, -1, -1):
                rewards[i] += rewards[i+1] * self.discount

            values = self.value(states)
            advantages = rewards - values
            logp = torch.stack([exp.logp for exp in ep_mem.memory])

            traj_data = {}
            traj_data["states"] = states.squeeze()
            traj_data["actions"] = actions.squeeze()
            traj_data["old_logp"] = logp.squeeze()
            traj_data["advantages"] = advantages.squeeze()
            traj_data["rewards"] = rewards.squeeze()
            return traj_data

    def get_policy_loss(self, states, actions, advantages, old_logp):
        new_logp = self.get_policy(states).log_prob(actions)
        new_old = torch.exp(new_logp - old_logp)
        policy_loss = -(torch.min(new_old * advantages, torch.clamp(new_old, 1-self.clip_ratio, 1+self.clip_ratio) * advantages)).mean()
        return policy_loss

    def update_nets(self,data):

        # Train policy with multiple steps of gradient descent
        policy_losses = []
        for _ in range(self.num_policy_epochs):
            all_batch_indices = torch.randperm(data["states"].shape[0])
            num_batch = math.ceil(all_batch_indices.shape[0]/self.batch_size)
            for i in range(num_batch):
                self.policy_optimizer.zero_grad()
                batch_indices = all_batch_indices[i*self.batch_size: (i+1)*self.batch_size]
                loss_pi = self.get_policy_loss(data['states'][batch_indices], data['actions'][batch_indices], data['advantages'][batch_indices], data['old_logp'][batch_indices])
                loss_pi.backward()
                policy_losses.append(float(loss_pi.item()))
                self.policy_optimizer.step()

        val_losses = []
        # Value function learning
        for _ in range(self.num_value_epochs):
            all_batch_indices = torch.randperm(data["states"].shape[0])
            num_batch = math.ceil(all_batch_indices.shape[0]/self.batch_size)
            for i in range(num_batch):
                self.value_optimizer.zero_grad()
                batch_indices = all_batch_indices[i*self.batch_size: (i+1)*self.batch_size]
                loss_v = ((self.value(data['states'][batch_indices]).squeeze() - data['rewards'][batch_indices])**2).mean()
                loss_v.backward()
                val_losses.append(float(loss_v.item()))
                self.value_optimizer.step()

        return np.mean(policy_losses), np.mean(val_losses)


    def train(self):
        env = self.env
        obs_dim = self.env.observation_space.shape
        act_dim = env.action_space.shape
        steps_per_epoch = 8000

        bar = Bar('{}'.format('Training'), max=self.nepisodes)
        self.logger = get_logger("PPO",self.env.spec.name)
        
        ep_rewards = []
        episode_rewards = []
        eval_rewards = []
        
        for ep in range(self.nepisodes):
            policy_trajs = dict()
            while True:
                state = env.reset()
                ep_reward = 0 
                replaymem = PPOMem(10000,1)
                for _ in range(self.max_steps):
                    action, logp = self.get_action_logp(torch.as_tensor(state, dtype=torch.float32))
                    next_state, reward, done, _ = env.step(action)
                    ep_reward += reward
                    replaymem.add_exp(torch.tensor(state).float(),torch.tensor(action).float(),reward,int(done),torch.tensor(logp).float())     
                    state = next_state
                    if done:
                        break
                ep_rewards.append(ep_reward)
                eval_rewards.append(ep_reward)
                traj = self.process_epoch_mem(replaymem)
                if len(policy_trajs.keys()) == 0:
                    policy_trajs = traj
                else:
                    for k in policy_trajs.keys():
                        policy_trajs[k] = torch.cat((policy_trajs[k],traj[k]),dim=0)
                
                if policy_trajs["states"].shape[0] > steps_per_epoch:
                    break
                    
            pol_loss, val_loss = self.update_nets(policy_trajs)
            
            
            ep_reward  = np.mean(ep_rewards)


            if self.use_wandb:
                wandb.log({"episode_reward": ep_reward,'eval_reward_mean':last_eval_mean,'eval_reward_std':last_eval_std})
            
            episode_rewards.append(ep_reward)
            ep_info =  'Episode '+str(ep)+' reward: ' + str(ep_reward) +"  pol_loss: " + str(pol_loss.item())+" val_loss: "+str(val_loss.item()) #+ ' Mean r over last 20 episodes :' + str(np.mean(episode_rewards[-20:]).item())+' last eval mean,std  ' +str(last_eval_mean)+' '+str(last_eval_std)
            self.logger.info(ep_info)
            if "cart" in self.env.spec.name.lower() and np.mean(episode_rewards[-20:]).item() > 480:
                print("Solved cartpole exiting early")
                bar.finish() 
                return eval_rewards, np.mean(episode_rewards[-30:]).item()

            
            Bar.suffix = ep_info
            bar.next()
            
        bar.finish()            
        
        return eval_rewards, np.mean(episode_rewards[-30:]).item()


    def show_results(self):
        self.evaluate(save_gif=True)

    def evaluate(self,save_gif = False):
        state = self.env.reset(seed=self.seed)
        total_reward = 0
        frames = []
        for t in range(1,self.max_steps):
            
            action, logp = self.get_action_logp(torch.as_tensor(state, dtype=torch.float32))
            new_state, reward, done, info = self.env.step(action.item()) 
            if save_gif:
                img = self.env.render(mode="rgb_array")
                frames.append(img)

            total_reward += reward
            state = new_state
            if done :
                break         
        
        if save_gif:
            write_gif([np.transpose(f, axes=[2,0, 1]) for f in frames], 'gifs/vpg_'+self.env.spec.name+'.gif', fps=30)
        if self.use_wandb:
            wandb.log({"loss": total_reward})
        return total_reward
