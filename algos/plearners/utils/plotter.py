import matplotlib.pyplot as plt
import numpy as np
from  datetime import datetime

def save_rewards_meanvar_plot(rewards,algo_name, env_name):
    # rewards is shape (num_evals, num_test_per_eval)
    filename = "plots/"+algo_name +"_"+ env_name +"_"+ datetime.now().isoformat(timespec='seconds')+'.jpg'
    x = np.arange(rewards.shape[0])
    means = np.mean(rewards,axis=1)
    stds = np.std(rewards,axis=1)
    plt.plot(x,means, 'k', color='#CC4F1B')
    plt.fill_between(x,means-stds, means+stds,
        alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    
    plt.savefig(filename)

