import gym

import warnings # normal
warnings.filterwarnings("ignore")

from stable_baselines3 import A2C
import os # sava and load model

# log somethings
models_dir = "models/A2C" # model_direction, save model
log_dir = "logs" # log training stuff

if not os.path.exists(models_dir): # if not exists repeated model directory
    os.makedirs(models_dir)

if not os.path.exists(log_dir): # if not exists repeated log directory
    os.makedirs(log_dir) # create log file

# select environment
env = gym.make("LunarLander-v2", render_mode=None) # lander demo human???
env.reset()

# select model (algorithm)
model = A2C("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log=log_dir) # try cuda, log my training stuff with log directory

# fit several times (train)
Timesteps = 10000 # max interaction for each model.learn()
for i in range(1, 6): # 5 times
    model.learn(total_timesteps=Timesteps, reset_num_timesteps=False, tb_log_name="A2C") # totally fit 5*10000 times
    model.save(f"{models_dir}/{Timesteps * i}") # save model with path

env.close()