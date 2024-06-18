from Easy21_env import Easy21

env = Easy21()

episode = 10
for ep in range(episode):
    print("this is", ep + 1, "rounds")
    done = False
    obs, info = env.reset() # reset reward
    print("initial obs", obs)

    while not done: # while not eat itself or reach the boundary

        # confirm action
        random_action = env.action_space.sample()
        print("action", random_action) # random

        # step
        obs, reward, done, truncated, info = env.step(random_action)
        print("obs", obs, "rd", reward) # print reward after each step