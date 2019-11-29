import gym
import random
import numpy as np
import tensorflow as tf
import time
import datetime
import gym.spaces
import gym_Rubiks_Cube
import os

def generateCube(env):
    # Mélange entre 1 et 5 coup aléatoires
    env.setScramble(1,1,False)
    env.reset()
    env.scramble()

# param : l'env
# return : une liste de coup à effectuer pour résoudre le cube
def methodeRandom(env):
    actions = []
    rew = 0
    while rew == 0:
        action = env.action_space.sample()
        _, rew, _, _ = env.step(action)
        actions.append(action)
    return actions # Solution

def testFunction(function_to_test, env, nb_cube):
    # ret : liste of (nb_move, time in second)
    ret = []
    index_cube = 0
    while index_cube != nb_cube:
        index_cube += 1
        generateCube(env)

        beginHour = time.time()
        actions = function_to_test(env) 
        end_hour = time.time()

        print("\tCube n°", index_cube, "Found a solution in ", len(actions), "moves(s).", " Encovered in ", str(datetime.timedelta(seconds=(end_hour - beginHour))))
        ret.append((len(actions), (end_hour - beginHour)))
    return ret


# ***************----Debut du code----*********************
env = gym.make("RubiksCube2x2-v0")
env.reset()
os.system("clear")
nombre_cube = 4

print("Méthode ramdom :")
ret = testFunction(methodeRandom,env, nombre_cube)
print(f'Results:\nOn {nombre_cube} cubes, average number of moves : {sum([r[0] for r in ret])/nombre_cube}\nAverage time : {str(datetime.timedelta(seconds=sum([r[1] for r in ret])/nombre_cube))}')

#while 1:
#    env.render()
#    time.sleep(0.5)
#    action = env.action_space.sample()
#    env.step(action)
#    state, reward, done, info = env.step(action)
#    print(state,"#", reward,"#", done,"#", info)

# Output example :
#
# Méthode ramdom :
#         Cube n° 1 Found a solution in  1 moves(s).  Encovered in  0:00:00.000204
#         Cube n° 2 Found a solution in  10790317 moves(s).  Encovered in  0:10:07.804373
#         Cube n° 3 Found a solution in  17327 moves(s).  Encovered in  0:00:00.983742
#         Cube n° 4 Found a solution in  2438881 moves(s).  Encovered in  0:02:17.804241
# Results:
# On 4 cubes, average number of moves : 3311631.5
# Average time : 0:03:06.648140