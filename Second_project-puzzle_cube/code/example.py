from cube_model import CubeModel
from cube_solver import OpenAICubeSolver 
from cube_solver import gymcube_state_to_puzzle_cube, CubeSolver

# from cube_solver import CubeSolver
from puzzle_cube import PuzzleCube

import gym_Rubiks_Cube
import gym.spaces
import os

### ------------------------------------------------ ###
# load the neural network from a file of saved weights #
### ------------------------------------------------ ###
cm = CubeModel()
cm.load_from_config("../example/checkpoint_model_v1.0.5-r_gen034.h5")

### ------------------------------ ###
# create a new cube and randomize it #
### ------------------------------ ###
# initialize puzzle cube
pc = PuzzleCube()
pc = pc.scramble(0)
print(pc.__str__())

# initialize OpenAI Gym environment 
env = gym.make("RubiksCube-v0")
env.reset()
os.system("clear")

# scramble cube between x and y actions 
env.setScramble(0, 0, False)
# env.scramble()
env.reset()

# get actual state
state = env.getstate()

# print render
env.render()

puzzle_state = gymcube_state_to_puzzle_cube(state)
print("********** Puzzle_state ********** \n{}".format(puzzle_state.__str__()))
print("********** state ********** \n{}".format(state))

### ------------------------------------------------------------------------ ###
# use Monte Carlo tree search with the loaded neural network to solve the cube #
### ------------------------------------------------------------------------ ###
# Puzzle Cube
# s = CubeSolver(pc, cm)
# s.solve(steps=1600)
# print(s.solution())

# # Gym Cube
s = OpenAICubeSolver(puzzle_state, cm)
s.solve(steps=1000)
print(s.solution())

# # verify that this solution works
# for action in s.solution():
#     state, reward, done, info = env.step(action)

# env.render()

# assert done