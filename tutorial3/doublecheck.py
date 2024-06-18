from snake_env import SnakeEnv


env = SnakeEnv()
episode = 5

for ep in range(episode):
    done = False
    env.reset() # reset reward

    while not done: # while not eat itself or reach the boundary

        # confirm action
        random_action = env.action_space.sample()
        print("action", random_action) # random

        # step
        obs, reward, done, truncated, info = env.step(random_action)
        print("rd", reward) # print reward after each step