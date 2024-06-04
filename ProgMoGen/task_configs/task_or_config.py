import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: walking on an inclined plane. (slope=0.2)
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





def walk_on_inclined_plane(joints, length):

    left_foot_traj = keyframe_by_length(get_joint(joints, left_foot),length)
    right_foot_traj = keyframe_by_length(get_joint(joints, right_foot),length)

    left_foot_y = dimY(left_foot_traj).squeeze()
    left_foot_z = dimZ(left_foot_traj).squeeze()

    right_foot_y = dimY(right_foot_traj).squeeze()
    right_foot_z = dimZ(right_foot_traj).squeeze()

    # slope
    k = 0.2

    # left foot on plane OR right foot on plane.
    loss_1 = (left_foot_y - left_foot_z*k)**2
    loss_2 = (right_foot_y - right_foot_z*k)**2 
    loss = operation_or(loss_1, loss_2)
    loss = loss.mean()

    # add regularization on trajectory. Avoid standing still.
    h_min = 0.4
    loss_height = greater_than(left_foot_y[-1],h_min).mean()+greater_than(right_foot_y[-1],h_min).mean()

    # both feet should not be under the plane.
    loss_1_above = greater_than(left_foot_y - left_foot_z*k, 0.0)
    loss_2_above = greater_than(right_foot_y - right_foot_z*k, 0.0) 
    loss_above_plane = loss_1_above + loss_2_above
    loss_above_plane = loss_above_plane.mean()

    loss_total = loss + loss_height + loss_above_plane

    print(f"loss_total={loss_total:.5f}, loss={loss:.5f}, loss_height={loss_height:.5f}, loss_above={loss_above_plane:.5f}")
    

    return loss_total 



def f_loss(self, sample, sample_0, steps):

    length = self.length 
    joints = self.sample_to_joints(sample)

    loss = walk_on_inclined_plane(joints, length)
    
    return loss 


def f_eval(self, sample, sample_0):

    length = self.length 
    joints = self.sample_to_joints(sample)

    loss = walk_on_inclined_plane(joints, length)
    loss = loss.sqrt()
    
    return loss 







