
import numpy as np
import torch

class Experience:
    def __init__(self,s,a,r,ss,done):
        self.s = s
        self.a = a
        self.r = r
        self.ss = ss
        self.done = done

class ReplayMemory:
    def __init__(self,N,batch_size):
        self.max_size = N
        self.batch_size = batch_size
        self.memory = []

    def add_exp(self,s,a,r,ss,done):
        
        r = torch.tensor(r).unsqueeze(0)
        done = torch.tensor(done).unsqueeze(0)
        if len(self.memory) == self.max_size:
            self.memory.pop(0)

        self.memory.append(Experience(s,a,r,ss,done))

    def sample_minibatch(self, prioritized=False):
        if not prioritized:
            batch = np.random.choice(self.memory,self.batch_size)
        batch_dict = {}

        states = batch[0].s
        actions = batch[0].a
        rewards = batch[0].r
        sstates = batch[0].ss
        dones = batch[0].done
        for exp in batch[1:]:
            states = torch.cat((states,exp.s),dim=0)
            actions = torch.cat((actions,exp.a),dim=0)
            rewards = torch.cat((rewards,exp.r),dim=0)
            sstates = torch.cat((sstates,exp.ss),dim=0)
            dones = torch.cat((dones,exp.done),dim=0)

        batch_dict['s'] = states
        batch_dict['a'] = actions
        batch_dict['r'] = rewards
        batch_dict['ss'] = sstates
        batch_dict['d'] = dones
        return batch_dict
        