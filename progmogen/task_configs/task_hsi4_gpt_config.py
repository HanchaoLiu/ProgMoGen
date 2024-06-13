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
    # loss_confined_space = loss_ifelse_wholebody(joints, self.length)
    # loss_reg on trajectory. Avoid standing still.
    loss_reg = equal( sample[:,0:4,:,:self.length], sample_0[:,0:4,:,:self.length] )

    joints = joints[:,:,:, :self.length]
    motion = tensor_to_gpt_form(joints)
    barrier_start, barrier_end, barrier_height, body_width = 2.0, 3.0, 1.3, 0.2
    loss = compute_total_error(motion, barrier_start, barrier_end, barrier_height, body_width)
    loss = loss / self.length
    if isinstance(loss,float):
        loss = torch.tensor(loss).float().to(sample.device)
        loss.requires_grad=True

    loss = loss + loss_reg
    return loss 

    

def f_eval(self, sample, sample_0):
    joints = self.sample_to_joints(sample)

    joints = joints[:,:,:, :self.length]
    motion = tensor_to_gpt_form(joints)
    barrier_start, barrier_end, barrier_height, body_width = 2.0, 3.0, 1.3, 0.2
    loss = compute_total_error(motion, barrier_start, barrier_end, barrier_height, body_width)
    loss = loss / self.length
    if isinstance(loss,float):
        loss = torch.tensor(loss).float().to(sample.device)
    return loss



# written by GPT, change spine to spine2
# def compute_total_error(motion, barrier_start, barrier_end, barrier_height, body_width): 
#     total_error = 0.0
#     for frame in motion:
#         head_height = frame['head']['y'] + body_width 
#         spine_height = frame['spine2']['y'] + body_width 
#         head_distance = frame['head']['z'] 
#         spine_distance = frame['spine2']['z']
#         if barrier_start <= head_distance <= barrier_end: 
#             head_error = max(head_height - barrier_height, 0) 
#             total_error += head_error
#         if barrier_start <= spine_distance <= barrier_end: 
#             spine_error = max(spine_height - barrier_height, 0) 
#             total_error += spine_error
#     return total_error



def compute_total_error(motion, barrier_start, barrier_end, barrier_height, body_width): 
    total_error = 0.0
    for frame in motion:
        head_height = frame['head']['y'] + body_width 
        spine_height = frame['spine3']['y'] + body_width 
        head_distance = frame['head']['z'] 
        spine_distance = frame['spine3']['z']
        if barrier_start <= head_distance <= barrier_end: 
            head_error = max(head_height - barrier_height, 0) 
            total_error += head_error
        if barrier_start <= spine_distance <= barrier_end: 
            spine_error = max(spine_height - barrier_height, 0) 
            total_error += spine_error*0.5

        spine_height_2 = frame['spine2']['y'] + body_width 
        spine_distance_2 = frame['spine2']['z']
        if barrier_start <= spine_distance_2 <= barrier_end: 
            spine_error_2 = max(spine_height_2 - barrier_height, 0) 
            total_error += spine_error_2*0.5

    return total_error


