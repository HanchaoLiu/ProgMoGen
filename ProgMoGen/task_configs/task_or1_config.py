import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: walking on a terrain. One foot touches the ground at any time.
'''

# TODO: requires token
TEXT_PROMPT="a man is walking."
TEXT_TOKEN=None
LENGTH=100
DEMO_NUM=1


# optimizer params
lr=0.005
iterations=80
decay_steps=None



def get_terrain(length, device):
    terrain_1 = np.linspace(0,0.5,num=length//2)
    terrain_2 = np.linspace(0.5,0,num=length-length//2)
    terrain   = np.concatenate([terrain_1, terrain_2],0)
    assert len(terrain)==length
    x = th.FloatTensor(terrain).to(device)
    return x 


def walk_on_terrain(joints, terrain, length):
    foot_traj_1 = keyframe_by_length(dimY(get_joint(joints, left_foot)),length).squeeze()
    foot_traj_2 = keyframe_by_length(dimY(get_joint(joints, right_foot)),length).squeeze()

    loss_1 = (foot_traj_1 - terrain)**2
    loss_2 = (foot_traj_2 - terrain)**2 
    loss = operation_or(loss_1, loss_2)

    return loss 



def f_loss(self, sample, sample_0, steps):

    length = self.length 
    joints = self.sample_to_joints(sample)
    terrain = get_terrain(length, sample.device)

    loss = walk_on_terrain(joints, terrain, length)
    loss = loss.mean()
    
    return loss 


def f_eval(self, sample, sample_0):

    length = self.length 
    joints = self.sample_to_joints(sample)
    terrain = get_terrain(length, sample.device)

    loss = walk_on_terrain(joints, terrain, length)
    loss = loss.mean().sqrt()
    
    return loss 







