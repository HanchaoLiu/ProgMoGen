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
    # loss = loss_limited_space(joints)
    motion = tensor_to_gpt_form(joints)
    loss = compute_total_error(motion)
    if isinstance(loss,float):
        loss = torch.tensor(loss).float().to(sample.device)
        loss.requires_grad=True
    return loss 


def f_eval(self, sample, sample_0):
    joints = self.sample_to_joints(sample)
    # loss = loss_limited_space(joints)
    motion = tensor_to_gpt_form(joints)
    loss = compute_total_error(motion)
    if isinstance(loss,float):
        loss = torch.tensor(loss).float().to(sample.device)
    return loss


# written by GPT
def compute_total_error(motion):
    total_error = 0.0 
    for frame in motion:
        for joint in frame.values(): 
            x = joint['x']
            z = joint['z']
            if x < -1 or x > 1:
                error = max(abs(x) - 1, 0)
                total_error += error
            if z < -1 or z > 1:
                error = max(abs(z) - 1, 0)
                total_error += error
    return total_error

