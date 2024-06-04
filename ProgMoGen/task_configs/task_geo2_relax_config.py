import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_root_rot_pos, reverse_pose

from atomic_lib.relax_geometry import construct_line, calc_RT_from_two_lines, apply_RT_on_joints
from atomic_lib.math_utils import * 

'''
task: walking straight on a balance beam.
'''


TEXT_PROMPT='a person is walking in a straight line.'
TEXT_TOKEN=None
LENGTH=192
DEMO_NUM=1
# point+dir
TARGET=[0.5 , 0.1 , 0.4]+[0.4, 0.0, 1.0]


# optimizer params
lr=0.05
epoch_relax=2
iterations=50



def get_lr_schedule(self, i_k):
    if i_k in [0]:
        base_lr = 0.01
    else:
        base_lr = 0.005
    return base_lr


def f_loss(self, sample, sample_0, it, line_params):

    self.joint_idx_list = [left_foot, right_foot]

    joints_traj = self.get_global_traj_for_joints_list(sample, self.joint_idx_list)
    loss = loss_distance_3d_point_line(joints_traj, line_params)
    return loss


def f_eval(self, sample, sample_0, line_params, XZ_offset=False):

    self.joint_idx_list = [left_foot, right_foot]

    if not XZ_offset:
        joints_traj = self.get_global_traj_for_joints_list(sample, self.joint_idx_list)
    else:
        joints_traj = self.get_global_traj_for_joints_list_with_relax(sample, self.joint_idx_list)

    loss = distance_3d_point_line(joints_traj, line_params)
    return loss


def update_goal(self, sample, target, target_relaxed, i_k):
    # return target_relaxed.
    # update_relaxed_goal()
    # (seqlen, 3)

    self.joint_idx_list = [left_foot, right_foot]

    joint_traj = self.get_global_traj_for_joints_list(sample, self.joint_idx_list)
    line_params = fit_3d_line(joint_traj)
    # self.line_params (1,6)
    return line_params


def transform_sample(self, sample_ret, target_relaxed, target):
    
    # transform sample_ret -> joints -> (R,T) -> joints_new -> sample_ret_new 
    line_relax  = construct_line(target_relaxed[0,:3], target_relaxed[0, 3:])

    # target_list = th.FloatTensor(target_list).to(device)
    target_list = target
    line_target = construct_line(target_list[0, :3], target_list[0, 3:])
    R,translation = calc_RT_from_two_lines(line_relax, line_target)

    joints_src = self.sample_to_joints(sample_ret)
    # torch.Size([1, 22, 3, 196])
    joints_dst = apply_RT_on_joints(joints_src, R,translation)

    sample_ret_dst = self.joints_to_sample_with_XZ_offset(joints_dst)

    return sample_ret_dst















