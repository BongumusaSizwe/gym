#!/usr/bin/env python
import numpy as np
import sys, gym, time
from skimage.color import rgb2gray
from PIL import Image
import os
from os import walk


#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#
#This will save the game states in a PIL file, along with the actions the keyboard agent took.
#


env = gym.make('LunarLander-v2' if len(sys.argv)<2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
	##Use arrow keys
    #print(key)
    if key == 65362: #UP
    	key = 49
    if key == 65363: #Right
    	key= 50
    if key == 65361: #Left
    	key = 51
    if key == 65364: #Down
    	key = 52
    	
    a = int( key - ord('0') )
    
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

state_array = []
action_array = []
def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        # Action a was taken
        obser, r, done, info = env.step(a)

        
        if a != 0:
            # Save observation to state array, only if action was taken
            gray_obser = rgb2gray(obser)
            state_array.append(gray_obser)
            action_array.append(a)

        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break

#At the end, add this states to a file
#f = open("state-action.txt", "a")

actions = np.array(action_array)
states = np.array(state_array)

# Save file
def path_name(env):
    directory = 'datasets/'
    env_name = env.unwrapped.spec.id.lower()
    path = os.path.join(directory, env_name)
    dir_exist = os.path.isdir(path)
    dataset_name = env_name

    if not dir_exist:
        os.mkdir(path)

    count = 1
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            count += 1
    #Create and save dataset
    loc = dataset_name.find('-')
    filepath = dataset_name[:loc]+str(count)
    
    return filepath

filepath = path_name(env)
np.savez_compressed('RoadRunner.npz', states = states, actions = actions)

print("Data saved as", filepath)