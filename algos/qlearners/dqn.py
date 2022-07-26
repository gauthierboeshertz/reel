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
from .utils import save_rewards_meanvar_plot,get_logger,MLP, ReplayMemory
import logging
import time 
        
class DQN:
    def __init__(self, env, config):
        for k, v in config.items():
            setattr(self, k, v)
        print(config)
        self.env = env
        self.config = copy.deepcopy(config)
        self.reset(self.seed)

    def _get_e_action(self,state,e):
        if np.random.rand() < e:
            return torch.tensor(self.env.action_space.sample()).unsqueeze(0)
        else:
            with torch.no_grad():
                return torch.argmax(self.q(state)).unsqueeze(0)

    def reset(self, seed):
        self.seed = seed
        set_seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        self.epsilon = np.array([self.config['epsilon']])
        self.e_annealer = LinearAnnealer(self.epsilon,self.config['min_e'],max_step=self.config['e_max_step'])
        obs_size = self.env.observation_space.n if 'n' in self.env.observation_space.__dict__ else self.env.observation_space._shape[0]
        self.q = MLP(self.nUnits, obs_size,self.env.action_space.n).to(self.device)
        self.replaymem = ReplayMemory(self.memorysize,self.batch_size)
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

    def train(self):
        bar = Bar('{}'.format('Training'), max=self.nepisodes)
        self.logger = get_logger("DQN",self.env.spec.name)
        self.q.train()
        episode_rewards = []
        eval_rewards = []
        
        n_experience = 0
        last_eval_mean = 0
        last_eval_std = 0
        step = 0
        for ep in (range(self.nepisodes)):
            
            state = self.env.reset(seed=self.seed)
            ep_reward = 0
            for t in range(1,self.max_steps):
                
                action = self._get_e_action(torch.tensor(state).unsqueeze(0).float(),self.epsilon)
                new_state, reward, done, info = self.env.step(action[0].item())
                
                ep_reward += reward
                self.replaymem.add_exp(torch.tensor(state).unsqueeze(0).float(),action,reward,torch.tensor(new_state).unsqueeze(0).float(),int(done))   
                
                if step > self.start_training_step and (step%self.update_freq) == 0:
                    batch = self.replaymem.sample_minibatch()
                    self.optimizer.zero_grad()
                    qs = self.q(batch['s'])
                    qs = qs.gather(1, batch['a'].view(-1,1)).view(-1)
                    qss = self.q(batch['ss']).detach().max(axis=1)[0]
                    y = batch['r'] + (1 - batch['d'].float())*(self.gamma*qss)

                    loss = self.mse(qs,y)
                    loss.backward()
                    for param in self.q.parameters():
                        param.grad.data.clamp_(-1, 1)

                    self.optimizer.step()   

                state = new_state

                step += 1
                if done: 
                    break
            self.e_annealer.step()


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
        state = self.env.reset(seed=self.seed)
        total_reward = 0
        frames = []
        for t in range(1,self.max_steps):
            
            action = self._get_e_action(torch.tensor(state).unsqueeze(0).float(),0.05)
            new_state, reward, done, info = self.env.step(action[0].item()) 
            if save_gif:
                img = self.env.render(mode="rgb_array")
                frames.append(img)

            total_reward += reward
            state = new_state
            if done :
                break         
        
        if save_gif:
            write_gif([np.transpose(f, axes=[2,0, 1]) for f in frames], 'gifs/dqn_'+self.env.spec.name+'.gif', fps=30)
        if self.use_wandb:
            wandb.log({"loss": total_reward})
        return total_reward
