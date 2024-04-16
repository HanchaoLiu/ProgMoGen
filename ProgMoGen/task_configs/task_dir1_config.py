import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: direction of left_elbow and right_elbow assigned for all frames.
'''

# TODO: requires token
TEXT_PROMPT="a man is walking."
TEXT_TOKEN=None
LENGTH=100
DEMO_NUM=1


# optimizer params
lr=0.005
iterations=100
decay_steps=None


def f_loss(self, sample, sample_0, steps):

    length = self.length 
    joints = self.sample_to_joints(sample)
    
    d1 = th.FloatTensor([1.0, 0.0, 1.0]).to(joints.device)
    d2 = th.FloatTensor([-1.0, 0.0, 1.0]).to(joints.device)
    loss_1 = loss_directional(joints, left_elbow, d1)
    loss_2 = loss_directional(joints, right_elbow, d2)

    loss = (loss_1 + loss_2)/2
    loss = loss.mean()
    
    return loss 


def f_eval(self, sample, sample_0):

    length = self.length 
    joints = self.sample_to_joints(sample)
    
    d1 = th.FloatTensor([1.0, 0.0, 1.0]).to(joints.device)
    d2 = th.FloatTensor([-1.0, 0.0, 1.0]).to(joints.device)
    loss_1 = loss_directional(joints, left_elbow, d1)
    loss_2 = loss_directional(joints, right_elbow, d2)

    loss = (loss_1 + loss_2)/2
    loss = loss.mean()
    
    return loss 







