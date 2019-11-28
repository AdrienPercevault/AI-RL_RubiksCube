import gym
import random
import numpy as np
import tensorflow as tf
import time
import gym.spaces
import gym_Rubiks_Cube

from colorama import init
init(autoreset=True)

env = gym.make("RubiksCube2x2-v0")
env.reset()

while 1:
    env.render()
    time.sleep(0.5)
    action = env.action_space.sample()
    env.step(action)
    state, reward, done, info = env.step(action)
    print(state,"#", reward,"#", done,"#", info)