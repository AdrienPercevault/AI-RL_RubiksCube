from cube_model import CubeModel
from cube_solver import OpenAICubeSolver 
from puzzle_cube import PuzzleCube
from termcolor import colored
from cube_solver import CubeSolver
from puzzle_cube import PuzzleCube

import os
import time


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
    str_format+='\n'
    str_format += \
"""{} {} {} {} {} {} {} {} {} {} {} {}
""".format(*(dico[cube_array[0][i]] for i in l1))

    str_format += \
"""{} {} {} {} {} {} {} {} {} {} {} {}
""".format(*(dico[cube_array[0][i]] for i in l2))

    str_format += \
"""{} {} {} {} {} {} {} {} {} {} {} {}
""".format(*(dico[cube_array[0][i]] for i in l3))
    str_format+='\n'
    str_format += \
"""         {} {} {}
         {} {} {}
         {} {} {}""".format(*(dico[c] for c in cube_array[0][45:]))

    print('\n')
    print(str_format)
    print('\n')


### ------------------------------------------------ ###
# Load the neural network from a file of saved weights #
### ------------------------------------------------ ###
cm = CubeModel()
cm.load_from_config("../example/checkpoint_model_v1.0.5-r_gen034.h5")

### ------------------------------------- ###
# Create a new puzzle cube and randomize it #
### ------------------------------------- ###
os.system("clear")
pc = PuzzleCube()
pc = pc.scramble(8)
inititial_state = pc._inner_cube._cube_array
print(f'State cube init : {inititial_state}')

### ------------------ ###
# Calculate the solution #
### ------------------ ###
s = CubeSolver(pc, cm)
s.solve(steps=1600)
print(f'Solution : {s.solution()} ({len(s.solution())} moves)')

### ----------- ###
# Resolve de cube #
### ----------- ###
for move in s.solution():
    pc = pc.move(move)
    current_state = pc._inner_cube._cube_array
    print_cube(current_state)
    time.sleep(0.5)
