import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import *

'''
task: carrying a large ball.
'''


TEXT_PROMPT='a person walks forward rather slowly.'
TEXT_TOKEN='sos/OTHER_a/DET_person/NOUN_walk/VERB_forward/ADV_rather/ADV_slowly/ADV_eos/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER_unk/OTHER'
LENGTH=96
# DEMO_NUM=9
DEMO_NUM=1


# optimizer params
lr=0.05
iterations=60
decay_steps=None






def f_loss(self, sample, sample_0, it):
    length = self.length 

    def loss_two_hands(sample, margin):
        joints = self.sample_to_joints(sample)
        dd = dist_to_point( get_joint(joints, left_wrist), get_joint(joints, right_wrist) )
        d_margin = th.ones_like(dd)*margin
        loss = equal(dd**2, d_margin**2)
        return loss 

    def loss_hand_chest(sample, margin):
        '''
        consider distance on the horizontal(XZ) plane.
        '''
        joints = self.sample_to_joints(sample)
        hand_center = mid_point(get_joint(joints, left_wrist), get_joint(joints, right_wrist))
        chest = get_joint(joints, spine3)
        # d2 = dist_to_point( dimZ(hand_center), dimZ(chest) )
        d2 = dist_to_point( dimXZ(hand_center), dimXZ(chest) )
        loss = greater_than(d2**2 - margin**2, 0.0)
        loss = loss.mean()
        return loss  

    # def loss_hand_chest(sample, margin):
    #     '''
    #     consider two points (spine3, pevlis) on the spine
    #     '''
    #     joints = self.sample_to_joints(sample)
    #     hand_center = mid_point(get_joint(joints, left_wrist), get_joint(joints, right_wrist))
        
    #     chest = get_joint(joints, spine3)
    #     d2 = dist_to_point( hand_center, chest )
    #     loss_1 = greater_than(d2**2 - margin**2, 0.0).mean()

    #     chest = get_joint(joints, pelvis)
    #     d2 = dist_to_point( hand_center, chest )
    #     loss_2 = greater_than(d2**2 - margin**2, 0.0).mean()

    #     loss = loss_1 + loss_2
    #     return loss  

    # avoid arm crossing.
    def loss_two_hand_side(sample):
        local_pose = self.do_inv_norm(sample)
        j20 = self.parse_joint_idx_list(left_wrist)
        j21 = self.parse_joint_idx_list(right_wrist)
        loss_j20 = greater_than(local_pose[:,j20[0],:,:], 0.0).mean()
        loss_j21 = less_than(   local_pose[:,j21[0],:,:], 0.0).mean()
        loss = loss_j20 + loss_j21
        return loss 

    # regularization term on [r_velocity, l_velocity]. Avoid standing still.
    loss_root = equal(sample[:,:3,:,:], sample_0[:,:3,:,:])

    loss_two_hands      = loss_two_hands(sample, margin=0.4)
    loss_hand_chest     = loss_hand_chest(sample, margin=0.4)
    loss_two_hands_side = loss_two_hand_side(sample)

    if it <=10:
        loss = loss_two_hands + loss_two_hands_side + loss_root 
    else:
        loss = loss_two_hands + loss_two_hands_side + loss_hand_chest + loss_root

    # if it <=10:
    #     loss = loss_two_hands  + loss_root 
    # else:
    #     loss = loss_two_hands  + loss_hand_chest + loss_root
    return loss 


def f_eval(self, sample, sample_0):
    length = self.length 

    def loss_two_hands(sample, margin):
        joints = self.sample_to_joints(sample)
        dd = dist_to_point( get_joint(joints, left_wrist), get_joint(joints, right_wrist) )
        d_margin = th.ones_like(dd)*margin
        # loss = equal(dd**2, d_margin**2)
        loss = equal_L1(dd, d_margin)
        return loss 

    loss_two_hands  = loss_two_hands(sample, margin=0.4)
    return loss_two_hands

