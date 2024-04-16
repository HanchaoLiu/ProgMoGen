import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import *


'''
task: moving object from A to B.
'''


TEXT_PROMPT="a person picks an object from a table and move it to another table."
TEXT_TOKEN=None
LENGTH=176
DEMO_NUM=10
TARGET=[0, 0.5, 0.2]+[2.0, 0.5, 0.2]


# optimizer params
lr=0.05
iterations=90
decay_steps=60
decay_steps=None


def get_target_pred_from_pos(joints, length, joint_control):
    # torch.Size([1, 22, 3, 196])
    # print(joints_ref.shape)
    t_all = length - 1
    t_0 = 0

    p0 = keyframe(get_joint(joints, joint_control),t_0).squeeze()
    p1 = keyframe(get_joint(joints, joint_control), t_all).squeeze()
    return [p0,p1]


def f_loss(self, sample, sample_0, it, target):
    joints = self.sample_to_joints(sample)
    length = self.length

    joint_control = left_wrist
    p0_pred, p1_pred = get_target_pred_from_pos(joints, length, joint_control)
    p0_gt, p1_gt     = target[0,:3], target[0,3:]
    loss = equal_sum(p0_pred, p0_gt) + equal_sum(p1_pred, p1_gt)
    loss = loss / 2
    return loss 


def f_eval(self, sample, sample_0, target):
    # get loss 
    joints = self.sample_to_joints(sample)
    length = self.length

    joint_control = left_wrist
    p0_pred, p1_pred = get_target_pred_from_pos(joints, length, joint_control)
    p0_gt, p1_gt     = target[0,:3], target[0,3:]
    loss = equal_sum(p0_pred, p0_gt) + equal_sum(p1_pred, p1_gt)
    loss = loss / 2
    loss_ret_val = loss.sqrt()
    return loss_ret_val







# TEXT_PROMPT_LIST = [
#     'a man picks something up, then puts it back.',
#     'a person picks an object from a table and move it to another table.',
#     'a person moves an object from a place to another place.',
#     'a man picks something up puts it to another place while walking.',
#     'a man picks something up, then walk and puts it back.',
#     'a person picks an object from a table, and then walk and move it to another table.',
#     'a person moves an object from a place to another place while walking.',
#     'a person picks an object, and carry it to another place.'
#     ]*4
# LENGTH_LIST=[176]*32
# # model_kwargs['y']['tokens'] = [None]*32
# # (n,6)
# TARGET_LIST = [
#     [[0, 0.5, 0.2], [1.0, 0.5, 0.2]],
#     [[0, 0.5, 0.2], [2.0, 0.5, 0.2]],
#     [[0, 0.5, 0.2], [3.0, 0.5, 0.2]],
#     [[0, 0.5, 0.2], [5.0, 0.5, 0.2]],

#     [[0, 0.2, 0.2], [1.0, 0.5, 0.2]],
#     [[0, 0.5, 0.2], [2.0, 0.2, 0.2]],
#     [[0, 0.5, 0.2], [3.0, 0.8, 0.2]],
#     [[0, 0.8, 0.2], [4.0, 0.5, 0.2]],
# ]*4