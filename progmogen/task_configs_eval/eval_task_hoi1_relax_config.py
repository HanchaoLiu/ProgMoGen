import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 
from atomic_lib.relax_geometry import construct_plane, calc_RT_from_two_planes, apply_RT_on_joints



'''
task: move object from A to B.
'''

DEMO_NUM=32



# optimizer params
lr=0.02
epoch_relax=3
iterations=30



def get_lr_schedule(self, i_k):
    if i_k in [0,1]:
        base_lr=0.02
    elif i_k in [2]:
        base_lr=0.002
    return base_lr




def get_target_pred_from_pos(joints, length, joint_control):
    # torch.Size([1, 22, 3, 196])
    # print(joints_ref.shape)
    t_all = length - 1
    t_0 = 0

    p0 = keyframe(get_joint(joints, joint_control),t_0).squeeze()
    p1 = keyframe(get_joint(joints, joint_control), t_all).squeeze()
    return [p0,p1]


def f_loss(self, sample, sample_0, it, target):
    # torch.Size([1, 22, 3, 196])
    joints = self.sample_to_joints(sample)
    assert joints.shape[2]==3
    length = self.length
    joint_control = left_wrist
    p0_pred, p1_pred = get_target_pred_from_pos(joints, length, joint_control)
    p0_gt, p1_gt     = target[0,:3], target[0,3:]
    loss = equal_sum(p0_pred, p0_gt) + equal_sum(p1_pred, p1_gt)
    loss = loss / 2
    return loss 


def f_eval(self, sample, sample_0, target, XZ_offset=False):
    # get loss 
    if XZ_offset==False:
        joints = self.sample_to_joints(sample)
    else:
        joints = self.sample_to_joints_with_XZ_offset(sample)

    assert joints.shape[2]==3
    length = self.length
    joint_control = left_wrist
    p0_pred, p1_pred = get_target_pred_from_pos(joints, length, joint_control)
    p0_gt, p1_gt     = target[0,:3], target[0,3:]
    loss = equal_sum(p0_pred, p0_gt) + equal_sum(p1_pred, p1_gt)
    loss = loss / 2
    loss_ret_val = loss.sqrt()
    return loss_ret_val





def update_goal(self, sample, target, target_relaxed, i_k):
    # return target_relaxed.
    # update_relaxed_goal()

    length = self.length
    joint_control = left_wrist
    
    p0_pred, p1_pred = get_target_pred_from_pos(self.sample_to_joints(sample), length, joint_control)
    # if i_k==0:
    #     self.target_gt_list_relax = self.target_gt_list 
    # else:
    #     self.target_gt_list_relax = get_xz_constraint(p0_gt, p1_gt, p0_pred, p1_pred)
    # return target_gt_list_relax

    if i_k==0:
        target_gt_list_relax = target.clone() 
    else:
        p0_gt, p1_gt = target_relaxed[0,:3], target_relaxed[0,3:]
        target_gt_list_relax = get_xz_constraint(p0_gt, p1_gt, p0_pred, p1_pred)
        target_gt_list_relax = th.cat([ target_gt_list_relax[0].reshape(1,-1), 
                                           target_gt_list_relax[1].reshape(1,-1) ], 1)
    return target_gt_list_relax



def transform_sample(self, sample_ret, target_relaxed, target):

    joints = self.sample_to_joints(sample_ret)

    target_gt_list_relax = [target_relaxed[0][:3], target_relaxed[0][3:]] 
    target_gt_list       = [target[0][:3], target[0][3:]]

    # find transformation from self.target_gt_list_relax to self.target_gt_list 
    R,translation = calc_RT_between_goals(target_gt_list_relax, target_gt_list)

    check_trans(target_gt_list_relax, target_gt_list, R,translation)

    # apply transformation on joints 
    do_relax=True 
    if do_relax:
        joints_dst = apply_RT_on_joints(joints, R,translation)
    else:
        joints_dst = joints

    # 0404: add last frame 
    joints_dst = torch.cat([joints_dst, joints_dst[:,:,:,-1:]],3)

    # joints to sample 
    sample_ret_dst = self.joints_to_sample_with_XZ_offset(joints_dst)


    return sample_ret_dst








def get_xz_constraint(p0_gt_0, p1_gt_0, p0_pred_0, p1_pred_0):
    '''
    p0_gt, p1_gt
    p0_pred, p1_pred
    '''
    p0_gt = p0_gt_0.clone()
    p1_gt = p1_gt_0.clone()
    p0_pred = p0_pred_0.clone()
    p1_pred = p1_pred_0.clone()
    p0_gt[1] = 0
    p1_gt[1] = 0
    p0_pred[1] = 0
    p1_pred[1] = 0

    center_pred = (p0_pred + p1_pred) / 2
    dir_p0 = p0_pred - center_pred
    dir_p1 = p1_pred - center_pred
    len_p0 = th.linalg.norm(dir_p0)
    len_p1 = th.linalg.norm(dir_p1)
    dir_p0 = dir_p0 / len_p0
    dir_p1 = dir_p1 / len_p1

    center_gt = (p0_gt + p1_gt) /2 
    len_p0_gt = th.linalg.norm( p0_gt - center_gt )
    len_p1_gt = th.linalg.norm( p1_gt - center_gt )
    
    p0_gt_relax = center_gt + len_p0_gt * dir_p0
    p1_gt_relax = center_gt + len_p1_gt * dir_p1 
    
    p0_gt_relax[1] = p0_gt_0[1]
    p1_gt_relax[1] = p1_gt_0[1]
    # self.target_gt_list_relax = []

    assert th.allclose( th.linalg.norm(p0_gt_0-p1_gt_0), th.linalg.norm(p0_gt_relax-p1_gt_relax) )

    target_gt_list_relax = [p0_gt_relax, p1_gt_relax]
    print("old=", p0_gt_0, p1_gt_0, "->new=", p0_gt_relax, p1_gt_relax)
    return target_gt_list_relax



def calc_RT_between_goals(target_gt_list_relax, target_gt_list):
    '''
    args:
        target_gt_list_relax = [ (x1,y1), (x2,y2) ] in tensor
        target_gt_list = [ (x1,y1), (x2,y2) ] in tensor
    return:
        R = (3,3), T=(3,1) torch.tensor
    '''

    device = target_gt_list_relax[0].device

    x1    = target_gt_list_relax[0][0]
    z1    = target_gt_list_relax[0][2]
    x1_gt = target_gt_list[0][0]
    z1_gt = target_gt_list[0][2]

    x2    = target_gt_list_relax[1][0]
    z2    = target_gt_list_relax[1][2]
    x2_gt = target_gt_list[1][0]
    z2_gt = target_gt_list[1][2]

    A_mat = [[x1, -z1, 1, 0],
         [z1,  x1, 0, 1],
         [x2, -z2, 1, 0],
         [z2,  x2, 0, 1]]
    b_mat = [x1_gt, z1_gt, x2_gt, z2_gt]
    A_mat = th.FloatTensor(A_mat).reshape(4,4)
    b_mat = th.FloatTensor(b_mat).reshape(4,1)
    sol   = th.linalg.solve(A_mat, b_mat)
    a,b,c,d = sol.reshape(-1)

    R = th.zeros((3,3))
    translation = th.zeros((3,1))

    R[0,0] = a 
    R[0,2] = -b 
    R[1,1] = 1.0
    R[2,0] = b 
    R[2,2] = a 

    translation[0,0] = c 
    translation[2,0] = d 

    R = R.to(device)
    translation = translation.to(device)
    return R, translation




def check_trans(target_gt_list_relax, target_gt_list, R,translation):
    p1_relax, p2_relax = target_gt_list_relax
    p1, p2 = target_gt_list 

    p1_relax = p1_relax.reshape(3,1)
    p2_relax = p2_relax.reshape(3,1)
    p1 = p1.reshape(3,1)
    p2 = p2.reshape(3,1)

    p1_new = th.matmul(R, p1_relax) + translation
    p2_new = th.matmul(R, p2_relax) + translation

    # print("p1_diff = ", p1_new.reshape(-1), p1.reshape(-1)  )
    # print("p2_diff = ", p2_new.reshape(-1), p2.reshape(-1)  )










