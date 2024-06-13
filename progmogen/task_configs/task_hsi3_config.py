import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 


from atomic_lib.math_utils import * 

'''
task: walking inside a square. (-1 < x < 1, -1 < z < 1)
'''

# TODO: requires token
TEXT_PROMPT="drunk walking animation turning around."
TEXT_TOKEN=None
LENGTH=196
DEMO_NUM=4


# optimizer params
lr=0.005
iterations=50
decay_steps=None


def loss_limited_space(joints):
    '''
    joints: [1,22,3,t]
    '''
    loss_1 = less_than(    dimX(joints), 1.0 ).mean()
    loss_2 = greater_than( dimX(joints), -1.0 ).mean()
    loss_3 = less_than(    dimZ(joints), 1.0 ).mean()
    loss_4 = greater_than( dimZ(joints), -1.0 ).mean()
    loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4 
    return loss 


def f_loss(self, sample, sample_0, step):
    joints = self.sample_to_joints(sample)
    loss = loss_limited_space(joints)
    return loss 


def f_eval(self, sample, sample_0):
    joints = self.sample_to_joints(sample)
    loss = loss_limited_space(joints)
    return loss



