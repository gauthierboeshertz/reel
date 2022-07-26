import gym
from algos import *
import argparse
import yaml
import numpy as np
import random
import wandb
import optuna
import logging
import time 

logger = logging.getLogger()

logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler("log" + str(time.time())+ ".log", mode="w"))

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


def main(args):
    
    if args.env == 'cliff':
        env_name = 'CliffWalking-v0'

    if args.env == 'taxi':
        env_name = 'Taxi-v3'

    if args.env == 'cart':
        env_name = 'CartPole-v1'

    if args.env == 'dcar':
        env_name = 'MountainCar-v0'

    

    if args.algo == 'dqn':
        config = yaml.safe_load(open("configs/dqn.yaml", 'r'))
        if config['use_wandb']:
            wandb.init(project="dqn", entity="igautthepower")
            wandb.config = config

        def optuna_trial(trial):
            
            config_trial = {
                "lr": trial.suggest_float("lr", 1e-5, 2e-2, log=True),
            }
            config.update(config_trial)

            env_name = 'CartPole-v1'
            env = gym.make(env_name)
            algo = DQN(env,config)
            return algo.train()

    if args.algo == 'rainbow':
        config = yaml.safe_load(open("configs/rainbow_dqn.yaml", 'r'))
        if config['use_wandb']:
            wandb.init(project="rainbow_dqn", entity="igautthepower")
            wandb.config = config
        

        def optuna_trial(trial):
            
            config_trial = {
                "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=False),
            }
            config.update(config_trial)

            env = gym.make(env_name)
            algo = RainbowDQN(env,config)
            return algo.train()

    print('Training')
    study = optuna.create_study(directions=["maximize"])
    study.optimize(optuna_trial, n_trials=30, timeout=None,n_jobs=5)

    #algo.show_results()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='cliff')
    parser.add_argument('--algo', type=str, default='qlearning')

    args = parser.parse_args()
    main(args)
