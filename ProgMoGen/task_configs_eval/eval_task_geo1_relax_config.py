import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_root_rot_pos, reverse_pose
# from data_loaders.humanml.scripts.motion_process import recover_from_rot, recover_from_rot_with_skeleton

from atomic_lib.relax_geometry import construct_plane, calc_RT_from_two_planes, apply_RT_on_joints

from atomic_lib.math_utils import * 

'''
task: left hand on a plane.
'''



DEMO_NUM=32


# optimizer params
lr=0.05
epoch_relax=5
iterations=20


joint_control=left_wrist

# def get_lr_schedule(self, i_k):
#     if i_k in [0]:
#         base_lr = 1e-2
#     else:
#         base_lr = 1e-3
#     return base_lr

# def get_lr_schedule(self, i_k):
#     if i_k in [0]:
#         base_lr = 1e-2
#     else:
#         base_lr = 5e-3
#     return base_lr

def get_lr_schedule(self, i_k):
    if i_k in [0]:
        base_lr = 1e-2
    elif i_k in [1,2]:
        base_lr = 5e-3
    else:
        base_lr = 1e-3
    return base_lr



def f_sample_random_plane(self, r_range=3, seed=0):


    rng = np.random.default_rng(seed)
    r = r_range*rng.random()

    theta = rng.random()*np.pi*2 

    x0 = r*np.cos(theta)+1e-5
    z0 = r*np.sin(theta)

    A = 1
    B = 0
    C = z0 / x0 
    D = (-x0*x0 - z0*z0)/x0 

    plane = th.FloatTensor([[A,B,C,D]])
    print("plane=", plane)
    return plane 



def f_loss(self, sample, sample_0, it, plane_params):

    joint_idx=joint_control
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

    joint_idx=joint_control

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
    joint_idx=joint_control
    # (seqlen, 3)   
    joint_traj = self.get_global_traj_for_joints(sample, joint_idx)

    # get plane_params
    plane_params = fit_yPlane(joint_traj)
    return plane_params


def transform_sample(self, sample_ret, target_relaxed, target):
    # apply transform back 

    # update target relaxed again using sample_ret
    if True:
        target_relaxed = self.update_goal(sample_ret, None, None, None)

    plane_params = target_relaxed
    target_list = target

    plane_pn     = plane_params_4d_to_point_normal_form(plane_params)
    plane_relax  = construct_plane(plane_pn[0,:3], plane_pn[0, 3:])

    # target_list = th.FloatTensor(target_list).to(device)
    target_list  = plane_params_4d_to_point_normal_form(target_list)
    plane_target = construct_plane(target_list[0, :3], target_list[0, 3:])
    R,translation = calc_RT_from_two_planes(plane_relax, plane_target)

    joints_src = self.sample_to_joints(sample_ret)
    # torch.Size([1, 22, 3, 196])
    # joints_dst = apply_RT(joints_src, R,T)
    joints_dst = apply_RT_on_joints(joints_src, R,translation)
    
    # 0404: add last frame 
    joints_dst = torch.cat([joints_dst, joints_dst[:,:,:,-1:]],3)
    sample_ret_dst = self.joints_to_sample_with_XZ_offset(joints_dst)


    if True:
        loss_new = self.f_eval(sample_ret_dst, None, target, XZ_offset=True).data.cpu().numpy()
        loss_old = self.f_eval(sample_ret, None, target_relaxed, XZ_offset=False).data.cpu().numpy()
        print("--->loss_old=", loss_old.mean(), "loss_new=",loss_new.mean())

        joints_dst_recon = self.sample_to_joints_with_XZ_offset(sample_ret_dst)
        print(joints_dst.shape, joints_dst_recon.shape)
        diff = np.abs(joints_dst[:,:,:,:196].data.cpu().numpy() - joints_dst_recon.data.cpu().numpy()  )
        assert np.abs(loss_new.mean()-loss_old.mean()) < 1e-5
    
    return sample_ret_dst





