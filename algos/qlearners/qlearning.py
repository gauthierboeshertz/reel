import numpy as np
import tqdm
from .utils import set_seed

class QLearning:
    def __init__(self, env, config):
        self.env = env
        set_seed(config['seed'])
        for k, v in config.items():
            setattr(self, k, v)
        self.q = np.zeros((self.env.observation_space.n,self.env.action_space.n))

    def _get_e_action(self,state,e):
        if np.random.rand() < e:
            return self.env.action_space.sample()
        else:
            action_sorted = np.argsort(self.q[state])[::-1]
            for act in action_sorted:
                if self.env.action_space.contains(act):
                    return act

    def train(self):
        
        for _ in tqdm.tqdm(range(self.nrepeats)):
            state = self.env.reset(seed=self.seed)
            for t in range(self.max_steps):
                action = self._get_e_action(state,e=self.epsilon)
                new_state, reward, done, info = self.env.step(action)
                self.q[state,action] = (1 - self.alpha)*self.q[state,action] + self.alpha*(self.gamma*np.max(self.q[new_state]) + reward)
                state = new_state
                if done:
                    break
    
    def show_results(self):
        print('Q TABLE')
        print(self.q)
        print('Test')
        total_reward = 0
        state = self.env.reset(seed=self.seed)
        self.env.render()
        for t in range(self.max_steps):
            action = self._get_e_action(state,e=0)
            new_state, reward, done, info = self.env.step(action)
            total_reward += reward
            state = new_state
            self.env.render()
            if done:
                break
        print('Total reward',total_reward)