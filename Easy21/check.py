from stable_baselines3.common.env_checker import check_env
from Easy21_env import Easy21

env = Easy21()
# It will check your custom environment and output additional warnings if needed
check_env(env)