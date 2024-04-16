import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: avoiding an overhead barrier (2 < z < 3)
'''
TEXT_PROMPT="a person is walking."
TEXT_TOKEN=None
LENGTH=100
DEMO_NUM=1


# optimizer params
lr=0.05
iterations=60
decay_steps=None



def f_loss(self, sample, sample_0, step):
    joints = self.sample_to_joints(sample)
    loss_confined_space = loss_ifelse_wholebody(joints, self.length)

    loss_reg = equal( sample[:,0:4,:,:self.length], sample_0[:,0:4,:,:self.length] )
    loss = loss_confined_space + loss_reg 
    return loss 


def f_eval(self, sample, sample_0):
    joints = self.sample_to_joints(sample)
    loss = loss_ifelse_wholebody(joints, self.length)
    return loss



def loss_ifelse_wholebody(sample, length):
    # length = model_kwargs['y']['lengths'].item()
    sample = sample[:,:,:, :length]

    h0=0.8
    h1=1.2
    h2=1.4

    # loss head 
    x_pos = dimZ( get_joint(sample, head) ).reshape(-1)
    selected_idx = (x_pos > 2.0) & (x_pos < 3.0)
    selected_idx2= (x_pos < 1.0) | (x_pos > 4.0)
    loss_head = less_than( keyframe_list(dimY(get_joint(sample, head)), selected_idx), h0 ).mean() + \
                greater_than( keyframe_list(dimY(get_joint(sample, head)), selected_idx2), h2 ).mean()
    loss_head = loss_head / 2

    # loss_head = less_than( keyframe_list(dimY(get_joint(sample, head)), selected_idx), h0 ).mean()

    # loss spine
    x_pos = dimZ( get_joint(sample, spine3) ).reshape(-1)
    selected_idx = (x_pos > 2.0) & (x_pos < 3.0)
    loss_spine3 = less_than( keyframe_list(dimY(get_joint(sample, spine3)), selected_idx), h1 ).mean()

    x_pos = dimZ( get_joint(sample, spine2) ).reshape(-1)
    selected_idx = (x_pos > 2.0) & (x_pos < 3.0)
    loss_spine2 = less_than( keyframe_list(dimY(get_joint(sample, spine2)), selected_idx), h1 ).mean()

    loss = (loss_head + loss_spine3 + loss_spine2) / 3
    
    return loss


