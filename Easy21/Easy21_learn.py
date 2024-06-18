from stable_baselines3 import A2C, PPO
from Easy21_env import Easy21
import os
import time

import warnings
warnings.filterwarnings("ignore")


# fit the model
models_dir = f"models/{int(time.time())}" # save model
log_dir = f"logs/{int(time.time())}" # save log

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# create env
env = Easy21()
env.reset()

# select model (algorithm)
model = A2C("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log=log_dir)

Timesteps = 10000
for i in range(1, 16):
    model.learn(total_timesteps=Timesteps, reset_num_timesteps=False, tb_log_name="A2C") # log in logs/times/A2C
    model.save(f"{models_dir}/{Timesteps * i}") # save model in models/times/10000.zip

env.close()
