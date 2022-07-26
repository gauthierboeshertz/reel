import gym
from algos import *
import argparse
import yaml
import numpy as np
import random
import wandb
import copy
from algos.qlearners.utils import save_rewards_meanvar_plot
import numpy as np


def get_env(env_name):
    if args.env == 'cliff':
        env = 'CliffWalking-v0'

    if args.env == 'taxi':
        env = 'Taxi-v3'

    if args.env == 'pend':
        env = 'Pendulum-v1'

    if args.env == 'cart':
        env = 'CartPole-v1'

    if args.env == 'dcar':
        env = 'MountainCar-v0'

    if args.env == 'lunar':
        env = 'LunarLander-v2'

    return gym.make(env)

def run_seeds(algo,seeds):

    all_evals = []
    for seed in seeds:
        print("Training on seed ",seed)
        algo.reset(seed)
        eval_rewards, _ = algo.train()
        all_evals.append(eval_rewards)

    all_evals = np.array(all_evals)
    return np.mean(all_evals, axis=0)

def main(args):
    
    env = get_env(args.env)

    algo_class = eval(args.algo)
    config = yaml.safe_load(open("configs/"+ args.algo+".yaml", 'r'))
    if config['use_wandb']:
        wandb.init(project=args.algo, entity="igautthepower")
        wandb.config = config

    algo = algo_class(env,config)

    if args.nseeds > 1:
        print("Running on ",args.nseeds," seeds")
        eval_rewards = run_seeds(algo, list(range(args.nseeds)))
    else:
        eval_rewards,_ = algo.train()

    save_rewards_meanvar_plot(np.array(eval_rewards),args.algo,args.env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='cliff')
    parser.add_argument('--algo', type=str, default='qlearning')
    parser.add_argument('--nseeds', type=int, default=1)

    args = parser.parse_args()
    main(args)
