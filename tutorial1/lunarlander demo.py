import gym

import warnings
warnings.filterwarnings("ignore")

# a demo
# env = gym.make("LunarLander-v2") # lander demo
# env.reset()
# print(env.action_space.sample())
# print(env.observation_space.shape)
# print(env.observation_space.sample())
# env.close()

# demo2
# env = gym.make("LunarLander-v2", render_mode="human") # lander demo
# env.reset()
#
# for step in range(200):
#     env.render()
#     action = env.action_space.sample() # random action
#     obs, reward, done, is_max, info = env.step(action)
#     print(reward)
#
# env.close()

# demo3 test algorithm
from stable_baselines3 import A2C

# select environment
env = gym.make("LunarLander-v2", render_mode="human") # lander demo human???
env.reset()

# select algorithm
model = A2C("MlpPolicy", env, verbose=1, device='cpu') # try cuda

# fit the model
model.learn(total_timesteps=10000) # fit the model with max timestep

# in this case, each step is per second (maybe), once we reach 10000 steps, then it ends
# but our method A2C says, it will update model in 5 timesteps

# evaluate model after fitting
episodes = 10 # repeat 10 times
for ep in range(episodes):
    obs, info = env.reset() # just observation of agent
    done = False # not done
    while not done:
        env.render()
        action, _states = model.predict(obs, deterministic=True) # use model to return action
        obs, reward, done, ismax, info = env.step(action) # collection after each step

env.close()

# demo4 with file save and load
# from stable_baselines3 import PPO
# import os
#
# models_dir = "models/PPO"
# log_dir = "logs"
#
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)
#
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
#
# env = gym.make("LunarLander-v2", render_mode="human") # lander demo human???
# env.reset()
#
# model = PPO("MlpPolicy", env, verbose=1, device='cuda') # try cuda
#
# model.learn(total_timesteps=10000) # ?
#
# episodes = 10
# for ep in range(episodes):
#     obs, info = env.reset() # just observation
#     done = False
#     while not done:
#         env.render()
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, ismax, info = env.step(action)
#
# env.close()
