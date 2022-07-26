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


# With spinning up help ;)


class VPG:
    def __init__(self, env, config):
        for k, v in config.items():
            setattr(self, k, v)
        print(config)
        self.env = env
        self.config = copy.deepcopy(config)
        self.reset(self.seed)


    def reset(self, seed):
        self.seed = seed
        set_seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        obs_size = self.env.observation_space.n if 'n' in self.env.observation_space.__dict__ else self.env.observation_space._shape[0]
        self.policy = MLP(self.nUnits, obs_size,self.env.action_space.n).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)


    def get_policy(self,s):
        return Categorical(logits=self.policy(s))


    def get_action(self,s):
        return self.get_policy(s).sample()
        
        
    def update_net(self,ep_mem):
        
        def r_to_go(rewards):
            return torch.cumsum(rewards.flip(dims=[0]),dim=0).flip(dims=[0])

        s = torch.stack([exp.s for exp in ep_mem.memory])
        a = torch.stack([exp.a for exp in ep_mem.memory])
        r = torch.stack([exp.r for exp in ep_mem.memory])
        r = r_to_go(r)
        self.optimizer.zero_grad()
        ep_loss = -(self.get_policy(s).log_prob(a) * r).mean()
        ep_loss.backward()
        self.optimizer.step()
        return ep_loss


    def train(self):
        bar = Bar('{}'.format('Training'), max=self.nepisodes)
        self.logger = get_logger("VPG",self.env.spec.name)
        
        episode_rewards = []
        eval_rewards = []
        
        n_experience = 0
        last_eval_mean = 0
        last_eval_std = 0
        step = 0

        for ep in (range(self.nepisodes)):
            self.policy.train()
            replaymem = ReplayMemory(10000,1)
            state = self.env.reset(seed=self.seed)
            ep_reward = 0
            for t in range(1,self.max_steps):
            
                action = self.get_action(torch.tensor(state).unsqueeze(0).float())
                new_state, reward, done, info = self.env.step(action.item())
                
                ep_reward += reward
                replaymem.add_exp(torch.tensor(state).unsqueeze(0).float(),action,reward,torch.tensor(new_state).unsqueeze(0).float(),int(done))   
                
                state = new_state

                step += 1
                if done: 
                    break

            self.update_net(replaymem)
            

            if self.num_eval_episodes > 0 and ((ep % self.eval_freq )==0):
                temp_eval_rewards = []
                for _ in range(self.num_eval_episodes):
                    temp_eval_rewards.append(self.evaluate())
                last_eval_mean = np.mean(temp_eval_rewards)
                last_eval_std = np.std(temp_eval_rewards)
                eval_rewards.append(temp_eval_rewards)

            if self.use_wandb:
                wandb.log({"episode_reward": ep_reward,'eval_reward_mean':last_eval_mean,'eval_reward_std':last_eval_std})
            
            episode_rewards.append(ep_reward)
            ep_info =  ('Episode '+str(ep)+' reward: ' + str(ep_reward) + ' Mean r over last 20 episodes :' + str(np.mean(episode_rewards[-20:]).item())+' last eval mean,std  ' +str(last_eval_mean)+' '+str(last_eval_std))

            if "cart" in self.env.spec.name.lower() and np.mean(episode_rewards[-20:]).item() > 480:
                print("Solved cartpole exiting early")
                bar.finish() 
                self.logger.info(ep_info)
                return eval_rewards, np.mean(episode_rewards[-30:]).item()

            self.logger.info( ep_info)
            Bar.suffix = ep_info
            bar.next()
        
        bar.finish()            
        
        return eval_rewards, np.mean(episode_rewards[-30:]).item()


    def show_results(self):
        self.evaluate(save_gif=True)

    def evaluate(self,save_gif = False):
        self.policy.eval()
        state = self.env.reset(seed=self.seed)
        total_reward = 0
        frames = []
        for t in range(1,self.max_steps):
            
            action = self.get_action(torch.tensor(state).unsqueeze(0).float())
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
