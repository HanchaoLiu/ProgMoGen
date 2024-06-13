import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: direction of left_elbow assigned for all frames.
'''


TEXT_PROMPT="a man is walking."
TEXT_TOKEN=None
LENGTH=100
DEMO_NUM=5


# optimizer params
lr=0.005
iterations=100
decay_steps=None


def f_loss(self, sample, sample_0, steps):

    length = self.length 
    joints = self.sample_to_joints(sample)
    
    d1 = th.FloatTensor([1.0, 0.0, 1.0]).to(joints.device)
    loss_1 = loss_directional(joints, left_elbow, d1)
    

    # loss_reg on right arm (optional).
    right_body_joint_list = [right_collar, right_shoulder, right_elbow, right_wrist]
    right_body_idx_list = []
    for ij in right_body_joint_list:
        right_body_idx_list += self.parse_joint_idx_list(ij)
    loss_reg = equal(sample[:,right_body_idx_list,:,:], sample_0[:,right_body_idx_list,:,:])


    loss_dir = loss_1.mean()

    loss = loss_dir + loss_reg * 0.1

    print(f"loss_total={loss:.5f}, loss_dir={loss_dir:.5f}, loss_reg={loss_reg:.5f}")
    
    return loss 


def f_eval(self, sample, sample_0):

    length = self.length 
    joints = self.sample_to_joints(sample)
    
    d1 = th.FloatTensor([1.0, 0.0, 1.0]).to(joints.device)
    loss_1 = loss_directional(joints, left_elbow, d1)
    

    loss = loss_1.mean()

    return loss 







