import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO, A2C
import os

from snake_env import SnakeEnv

import time


models_dir = f"models/{int(time.time())}"
log_dir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = SnakeEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, device='cuda', tensorboard_log=log_dir) # try cuda

Timesteps = 10000
for i in range(1, 11):
    model.learn(total_timesteps=Timesteps, reset_num_timesteps=False, tb_log_name="PPO") # log in logs/times/A2C
    model.save(f"{models_dir}/{Timesteps * i}") # save model in models/times/10000.zip

env.close()