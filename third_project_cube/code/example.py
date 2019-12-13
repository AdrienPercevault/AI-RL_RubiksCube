from cube_model import CubeModel
from cube_solver import OpenAICubeSolver 
from cube_solver import gymcube_state_to_puzzle_cube 
from puzzle_cube import PuzzleCube
from termcolor import colored
# from cube_solver import CubeSolver
# from puzzle_cube import PuzzleCube

import gym_Rubiks_Cube
import gym.spaces
import os


def print_cube(cube_array):
    dico = {0: colored('██', "red"), 1:colored('██', "yellow"), 2:colored("██", "green"), 3:colored("██", "white"), 4:colored("██", "magenta"), 5:colored("██", "blue")}          
    l1 = [9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38]
    l2 = [12, 13, 14, 21, 22, 23, 30, 31, 32, 39, 40, 41]
    l3 = [15, 16, 17, 24, 25, 26 ,33, 34, 35, 42, 43, 44]
    str_format = \
"""         {} {} {}
         {} {} {}
         {} {} {}
""".format(*(dico[c] for c in cube_array[0][:9]))
    
    str_format += \
"""{} {} {} {} {} {} {} {} {} {} {} {}
""".format(*(dico[cube_array[0][i]] for i in l1))

    str_format += \
"""{} {} {} {} {} {} {} {} {} {} {} {}
""".format(*(dico[cube_array[0][i]] for i in l2))

    str_format += \
"""{} {} {} {} {} {} {} {} {} {} {} {}
""".format(*(dico[cube_array[0][i]] for i in l3))
    
    str_format += \
"""         {} {} {}
         {} {} {}
         {} {} {}""".format(*(dico[c] for c in cube_array[0][45:]))
    
    print(str_format)

### ------------------------------------------------ ###
# load the neural network from a file of saved weights #
### ------------------------------------------------ ###
# cm = CubeModel()
# cm.load_from_config("../example/checkpoint_model_v1.0.5-r_gen034.h5")

### ------------------------------------- ###
# create a new puzzle cube and randomize it #
### ------------------------------------- ###
# pc = PuzzleCube()
# pc = pc.scramble(8)
# print(pc)

### --------------------------------------- ###
# initialize cube from OpenAI Gym environment #
### --------------------------------------- ###
# env = gym.make("RubiksCube-v0")
# env.reset()
# os.system("clear")

# scramble cube between x and y actions 
# env.setScramble(10, 20, False)
# env.scramble()

# print render
# env.render()

# get actual state
# action = env.action_space.sample()
# state, reward, done, info = env.step(action)
# print("********** State ********** {}".format(state))
# puzzle_state = gymcube_state_to_puzzle_cube(state)
# print("********** Puzzle_state ********** {}".format(puzzle_state))

### ------------------------------------------------------------------------ ###
# use Monte Carlo tree search with the loaded neural network to solve the cube #
### ------------------------------------------------------------------------ ###
# Puzzle Cube
s = PuzzleCube()
print(s._inner_cube._cube_array)
print_cube(s._inner_cube._cube_array)