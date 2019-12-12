from cube_model import CubeModel
from cube_solver import OpenAICubeSolver 
from cube_solver import gymcube_state_to_puzzle_cube 

# from cube_solver import CubeSolver
# from puzzle_cube import PuzzleCube

import gym_Rubiks_Cube
import gym.spaces
import os

### ------------------------------------------------ ###
# load the neural network from a file of saved weights #
### ------------------------------------------------ ###
cm = CubeModel()
cm.load_from_config("../example/checkpoint_model_v1.0.5-r_gen034.h5")

### ------------------------------------- ###
# create a new puzzle cube and randomize it #
### ------------------------------------- ###
# pc = PuzzleCube()
# pc = pc.scramble(8)
# print(pc)

### --------------------------------------- ###
# initialize cube from OpenAI Gym environment #
### --------------------------------------- ###
env = gym.make("RubiksCube-v0")
env.reset()
os.system("clear")

# scramble cube between x and y actions 
env.setScramble(10, 20, False)
env.scramble()

# print render
env.render()

# get actual state
action = env.action_space.sample()
state, reward, done, info = env.step(action)
puzzle_state = gymcube_state_to_puzzle_cube(state)
print("********** Puzzle_state ********** {}".format(puzzle_state))
print("********** State ********** {}".format(state))


### ------------------------------------------------------------------------ ###
# use Monte Carlo tree search with the loaded neural network to solve the cube #
### ------------------------------------------------------------------------ ###
# Puzzle Cube
# s = CubeSolver(pc, cm)
# s.solve(steps=1600)
# print(s.solution())

# Gym Cube
s = OpenAICubeSolver(puzzle_state, cm)
s.solve(steps=1000)
print(s.solution())

# verify that this solution works
for action in s.solution():
    state, reward, done, info = env.step(action)

env.render()

assert done