import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: overhead barrier.
'''


# optimizer params

lr=0.005
iterations=100
decay_steps=None







def loss_overhead_barrier(joints, length):
    t_all = length - 1
    t_0 = 0
    t_middle = length//2

    loss_head = greater_than( keyframe(dimY(get_joint(joints, head)), t_0), 1.5 ) + \
                greater_than( keyframe(dimY(get_joint(joints, head)), t_all), 1.5 ) + \
                less_than( keyframe(dimY(get_joint(joints, head)), t_middle), 0.5 )
    loss_head = loss_head / 3

    # because of the motion prior, the feet will actually become closed to the ground.
    # avoiding floating feet
    loss_foot = less_than( keyframe(dimY(get_joint(joints, left_foot)), t_middle), 0.0 ) + \
                less_than( keyframe(dimY(get_joint(joints, right_foot)), t_middle), 0.0 )
    loss_foot = loss_foot / 2
    
    loss = loss_foot + loss_head
    return loss 



def f_loss(self, sample, sample_0, step):
    '''
    joints: [1,22,3,t]
    '''
    joints = self.sample_to_joints(sample)
    assert joints.shape[0]==1

    length = self.length 
    loss = loss_overhead_barrier(joints, length)
    
    return loss


def f_eval(self, sample, sample_0):
    '''
    joints: [1,22,3,t]
    '''
    joints = self.sample_to_joints(sample)
    assert joints.shape[0]==1

    length = self.length 
    loss = loss_overhead_barrier(joints, length)
    
    return loss






