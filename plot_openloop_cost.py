
import numpy as np

import json
import pathlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


name = "playground_mpc_2"


# frames = [0, 5, 49]
# colors = ['r','b','g']

# fig = plt.figure(figsize=plt.figaspect(8/11))
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(1, 1, 1)
handles =[]
# ax_twin = ax.twinx()

pink    = '#EC6779'
green   = '#288737'
blue    = '#4678a8'
yellow  = '#CBBA4E'
cyan    = '#6BCCeC'
magenta = '#A83676'


end_frame = 0
while 1:
    if pathlib.Path("experiments/" + name +"/mpc/"+str(end_frame)+".json").exists():
        end_frame +=1
    else:
        break


print(end_frame)
taken_actions = []

data = json.load(open("experiments/" + name +"/mpc/"+str(0)+".json"))
ax.plot(data['total_cost'], c =pink, alpha = 1, linewidth= 2)

for frame in range(end_frame):
    data = json.load(open("experiments/" + name +"/mpc/"+str(frame)+".json"))
    taken_actions.append( data['actions'][0] )

agent = json.load(open("experiments/" + name +"/OL_GT/agent_data.json"))


import torch

state = torch.tensor(agent['true_states'])

actions = torch.tensor(taken_actions)
max_len = actions.shape[0]
print(max_len)
print(state.shape)

pos = state[:max_len, 0:3]
v   = state[:max_len, 3:6]
R = state[:max_len, 6:15].reshape((-1,3,3))
omega = state[:max_len, 15:]


import types

print(pos.shape)
print(v.shape)
print(R.shape)
print(omega.shape)
print(actions.shape)

def monkey_patch_get_everything(self):
    return (pos, v, None, R, omega, None, actions)
    # pos, vel, accel, rot_matrix, omega, angular_accel, actions = self.calc_everything()

from quad_plot import System, get_nerf
# from load_nerf import get_nerf

renderer = get_nerf('configs/playground.txt')
cfg = {"T_final": 2,
        "steps": 30,
        "lr": 0.002,
        "epochs_init": 2500,
        "fade_out_epoch": 0,
        "fade_out_sharpness": 10,
        "epochs_update": 250,
        }

traj = System(renderer, torch.zeros(18), torch.zeros(18) , cfg)

# traj.calc_everything = monkey_patch_get_everything
traj.calc_everything = types.MethodType(monkey_patch_get_everything, traj)

taken_cost, colision_cost = traj.get_state_cost()
taken_cost = taken_cost.detach().numpy()

ax.plot(taken_cost, c ="k", alpha = 1, linewidth=4)

ax.set_ylabel("Cost", fontsize=20)

ax.set_xlabel("Trajectory time", fontsize=20)

handles.append(  Line2D([0], [0], label='Cost of open loop trajectory', color='k', linewidth=4)  )
handles.append(  Line2D([0], [0], label='Cost of intial plan', color=pink, linewidth=2)  )

# handles.extend([line1, line2])
plt.legend(handles=handles, prop={"size":16})

# ax.legend()

plt.show()




