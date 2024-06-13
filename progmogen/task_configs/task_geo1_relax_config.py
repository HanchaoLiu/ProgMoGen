import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_root_rot_pos, reverse_pose

from atomic_lib.relax_geometry import construct_plane, calc_RT_from_two_planes, apply_RT_on_joints

from atomic_lib.math_utils import * 

'''
task: right hand on a plane.
'''


TEXT_PROMPT='a person walks rather slowly'
TEXT_TOKEN=None
LENGTH=96
DEMO_NUM=1
# point+normal
TARGET=[-1.0 , 0.0 , 0.0]+[1.0, 0, 0.25]


# optimizer params
lr=0.05
epoch_relax=5
iterations=20



def get_lr_schedule(self, i_k):
    if i_k in [0]:
        base_lr = 1e-2
    else:
        base_lr = 1e-3
    return base_lr


def f_loss(self, sample, sample_0, it, plane_params):

    joint_idx=right_wrist
    joints = self.sample_to_joints(sample)

    # torch.Size([1, 22, 3, 196])
    assert joints.shape[0]==1
    # (seqlen, 3)
    joint_traj = get_joint(joints, joint_idx).squeeze()
    joint_traj = joint_traj.permute(1,0).contiguous()
    loss_plane = loss_dist_to_plane(joint_traj, plane_params).mean()

    return loss_plane
    

def f_eval(self, sample, sample_0, plane_params, XZ_offset=False):

    if plane_params.shape[1]==6:
        plane_target = construct_plane(plane_params[0, :3], plane_params[0, 3:])
        plane_target_4d_params = plane_point_normal_form_to_params_4d(plane_target)
        plane_params = plane_target_4d_params

    joint_idx=right_wrist

    if XZ_offset==False:
        joints=self.sample_to_joints(sample)
    else:
        joints=self.sample_to_joints_with_XZ_offset(sample)

    joints = get_joint(joints, joint_idx)
    loss_plane = dist_to_plane(joints, plane_params)
    return loss_plane
    

def update_goal(self, sample, target, target_relaxed, i_k):
    # return target_relaxed.
    # update_relaxed_goal()
    # (seqlen, 3)
    joint_idx=right_wrist
    # (seqlen, 3)   
    joint_traj = self.get_global_traj_for_joints(sample, joint_idx)

    # get plane_params
    plane_params = fit_yPlane(joint_traj)
    return plane_params


def transform_sample(self, sample_ret, target_relaxed, target):
    # apply transform back 

    plane_params = target_relaxed
    target_list = target

    plane_pn     = plane_params_4d_to_point_normal_form(plane_params)
    plane_relax  = construct_plane(plane_pn[0,:3], plane_pn[0, 3:])

    # target_list = th.FloatTensor(target_list).to(device)
    plane_target = construct_plane(target_list[0, :3], target_list[0, 3:])
    R,translation = calc_RT_from_two_planes(plane_relax, plane_target)

    joints_src = self.sample_to_joints(sample_ret)
    # torch.Size([1, 22, 3, 196])
    joints_dst = apply_RT_on_joints(joints_src, R,translation)
    
    # 0404: add last frame 
    joints_dst = torch.cat([joints_dst, joints_dst[:,:,:,-1:]],3)
    sample_ret_dst = self.joints_to_sample_with_XZ_offset(joints_dst)

    # derive 4 points on a plane for later visualization.
    if False:
        plane_target_4d_params = plane_point_normal_form_to_params_4d(plane_target)

        joints_zdim = joints_dst[0,:,2,:self.length]
        z_min, z_max = joints_zdim.min().item(), joints_zdim.max().item()
        plane_4points = get_4points_on_plane(plane_target_4d_params.data.cpu().numpy(),
                            z_list=[z_min, z_max], y_list=[0,2])
        print(plane_4points.shape)
        print("plane_4points = ")
        print(plane_4points.tolist())

        plane_p = plane_target[0,:3].data.cpu().numpy()
        plane_n = plane_target[0,3:].data.cpu().numpy()
        check_plane = ((plane_4points - plane_4points) * plane_n).sum(axis=1)
        print("check_plane = ", check_plane)

    return sample_ret_dst


