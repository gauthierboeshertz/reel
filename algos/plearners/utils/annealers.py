import numpy as np 

class LinearAnnealer:

    def __init__(self,e,min_e,max_step=-1):
        self.e = e
        self.max_step = max_step
        self.cur_step = 1
        self.values = np.linspace(self.e,min_e,num=max_step)
        
    def step(self):

        if self.cur_step >= self.max_step:
            return

        self.e[0] =  self.values[self.cur_step]
        self.cur_step += 1


class ExponentialAnnealer:
    
    def __init__(self,e,min_e,factor):
        self.e = e
        self.factor = factor
        self.min_e = min_e
        
    def step(self):

        if self.e >= self.min_e:
            return
        self.e[0] =  self.e[0] * self.factor
        print(self.e)

if __name__ == '__main__':
    eps = np.array([0.8])
    anl = LinearAnnealer(eps,min_e=0.1,max_step=10)
    print(anl.values.shape)
    for _ in range(10):
        print('eps ',eps)
        anl.step()