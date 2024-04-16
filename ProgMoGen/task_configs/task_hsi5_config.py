import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: walking in a narrow gap.
'''
# This is a challenging case, so run multiple samples to get good result.

TEXT_PROMPT="a man walks forwards"
TEXT_TOKEN=None
LENGTH=150
DEMO_NUM=5

# optimizer params
lr=0.05
iterations=100
decay_steps=None




def f_loss(self, sample, sample_0, step):

    joints = self.sample_to_joints(sample)

    # narrow gap
    # there is a body_width of 0.1
    body_width=0.1
    s=0.2 - body_width

    loss_1 = less_than( dimX(keyframe_by_length(joints,self.length)), s ).mean()
    loss_2 = greater_than( dimX(keyframe_by_length(joints,self.length)), -s ).mean()
    loss_margin = (loss_1 + loss_2) / 2 

    # avoid collision (optional)
    body_width=0.2
    loss_collision = less_than( body_width**2 - dist_to_point(dimXZ(get_joint(joints, left_wrist)), dimXZ(get_joint(joints, pelvis)))**2, 0 ).mean() + \
                     less_than( body_width**2 - dist_to_point(dimXZ(get_joint(joints, right_wrist)), dimXZ(get_joint(joints, pelvis)))**2, 0 ).mean()
    loss_collision = loss_collision / 2

    loss = loss_margin + loss_collision
    # loss = loss_margin

    return loss 


def f_eval(self, sample, sample_0):
    '''
    joints: [1,22,3,t]
    '''
    joints = self.sample_to_joints(sample)

    # narrow gap
    # there is a body_width of 0.1
    body_width=0.1
    s=0.2 - body_width

    loss_1 = less_than( dimX(keyframe_by_length(joints,self.length)), s ).mean()
    loss_2 = greater_than( dimX(keyframe_by_length(joints,self.length)), -s ).mean()
    loss_margin = (loss_1 + loss_2) / 2 
    return loss_margin
    

