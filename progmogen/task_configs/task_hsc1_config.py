import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: left hand touching head.
'''


TEXT_PROMPT='a person walks casually forward.'
TEXT_TOKEN=None
LENGTH=100
DEMO_NUM=3


# optimizer params
lr=0.005
iterations=50
decay_steps=None



def loss_handheadcontact(joints):
    '''
    joints: [1,22,3,t]
    '''
    assert joints.shape[0]==1
    d0=0.1
    hand=left_wrist
    joints_hand = get_joint(joints, hand)
    joints_head = get_joint(joints, head)
    d = dist_to_point(joints_hand, joints_head)
    loss = equal_L1(d**2, d0**2)
    return loss


def f_loss(self, sample, sample_0, step):
    joints = self.sample_to_joints(sample)

    return loss_handheadcontact(joints)
    

def f_eval(self, sample, sample_0):
    joints = self.sample_to_joints(sample)
    return loss_handheadcontact(joints)
    






