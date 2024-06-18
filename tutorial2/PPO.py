import gym

import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
import os

models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = gym.make("LunarLander-v2", render_mode=None) # lander demo human???
env.reset()

model = PPO("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log=log_dir) # try cuda

Timesteps = 10000
for i in range(1, 6):
    model.learn(total_timesteps=Timesteps, reset_num_timesteps=False, tb_log_name="PPO") # ?
    model.save(f"{models_dir}/{Timesteps * i}") # ?

# episodes = 10
# for ep in range(episodes):
#     obs, info = env.reset() # just observation
#     done = False
#     while not done:
#         env.render()
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, ismax, info = env.step(action)

env.close()



