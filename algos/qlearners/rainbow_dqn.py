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
from .utils import save_rewards_meanvar_plot
from .utils import MLP, DuelNetwork
from .utils import ReplayMemory

# Implementation of 
# Rainbow: Combining Improvements in Deep Reinforcement Learning
# prioritized : https://arxiv.org/pdf/1511.05952.pdf
# Not for atari, only simple envs


        
class RainbowDQN:
    def __init__(self, env, config):
        for k, v in config.items():
            setattr(self, k, v)
        print(config)
        self.env = env
        self.config = copy.deepcopy(config)
        self.reset(self.seed)
        self.mse = torch.nn.MSELoss()
        

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
        if not self.config['duel_net']:
            self.q = MLP(self.nUnits, obs_size,self.env.action_space.n).to(self.device)
        else:
            self.q = DuelNetwork(self.nUnits, obs_size,self.env.action_space.n).to(self.device)
        
        self.old_q = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        self.replaymem = ReplayMemory(self.memorysize,self.batch_size)


    def train(self):
        bar = Bar('{}'.format('Training'), max=self.nepisodes)
        self.q.train()
        episode_rewards = []
        eval_rewards = []
        
        n_experience = 0
        last_eval_mean = 0
        last_eval_std = 0
        for ep in (range(self.nepisodes)):
            
            state = self.env.reset()
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
                    # DDQN
                    old_qss = self.old_q(batch['ss']).detach()
                    qss_argmax = self.q(batch['ss']).detach().argmax(axis=1)
                    old_qss_label = old_qss.gather(1, qss_argmax.view(-1,1)).view(-1)
                    y = batch['r'] + (1 - batch['d'].float())*(self.gamma*old_qss_label)

                    loss = self.mse(qs,y)
                    loss.backward()
                    for param in self.q.parameters():
                        param.grad.data.clamp_(-1, 1)

                    self.optimizer.step()   
                    if (n_experience % self.ddqn_update_freq) == 0:
                        self.old_q = copy.deepcopy(self.q)

                state = new_state

                n_experience += 1
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
                wandb.log({"episode_reward": ep_reward})
            episode_rewards.append(ep_reward)
            
            Bar.suffix = ('Episode '+str(ep)+' reward: ' + str(ep_reward) + ' Mean reward over last 20 episodes :' + str(np.mean(episode_rewards[-20:]).item())+' last eval mean,std  '+str(last_eval_mean)+' '+str(last_eval_std))#+' eval reward '+str(eval_reward))
            bar.next()
        
        bar.finish()
        if self.num_eval_episodes > 0 :
            save_rewards_meanvar_plot(np.array(eval_rewards),'plots/dqn_meanvar_'+self.env.spec.name+'.jpg')

    def show_results(self):
        self.evaluate(save_gif=True)

    def evaluate(self,save_gif = False):
        state = self.env.reset()
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
