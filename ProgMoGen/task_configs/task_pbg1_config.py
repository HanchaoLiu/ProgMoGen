import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: stand on one foot and keep balanced.
'''

# TEXT_PROMPT='person balances on left leg then with arms up high has arms fully extended keeping balance on left leg.'
TEXT_PROMPT='person balances on a leg then with arms up high has arms fully extended keeping balance on a leg.'
TEXT_TOKEN=None
LENGTH=192
DEMO_NUM=3


# optimizer params
lr=0.02
iterations=100
decay_steps=None



def f_loss(self, sample, sample_0, steps):
    
    length = self.length 
    # # torch.Size([1, 22, 3, 196])
    joints = self.sample_to_joints(sample)

    control_joint = right_foot
    control_joint_traj = keyframe_by_length( get_joint(joints, control_joint), self.length )
    center_of_mass_traj= keyframe_by_length( get_center_of_mass(joints), self.length )
    loss_mass_ctr = dist_to_point( dimXZ(control_joint_traj), dimXZ(center_of_mass_traj) )**2
    # loss_mass_ctr = dist_to_point_squared( dimXZ(control_joint_traj), dimXZ(center_of_mass_traj) )
    loss_mass_ctr = loss_mass_ctr.mean()

    loss_base = equal( get_joint(joints, control_joint), keyframe(get_joint(joints, control_joint),0) )

    # loss reg on one foot
    lower_body_joint_list = [pelvis, right_hip, right_knee, right_ankle, right_foot,
                                     left_hip, left_knee, left_ankle, left_foot]
    lower_body_idx_list = []
    for ij in lower_body_joint_list:
        lower_body_idx_list += self.parse_joint_idx_list(ij)
    loss_reg = equal(sample[:,lower_body_idx_list,:,:], sample_0[:,lower_body_idx_list,:,:])

    loss = loss_mass_ctr + loss_reg*0.1 + loss_base*10
    return loss 


def f_eval(self, sample, sample_0):
    joints = self.sample_to_joints(sample)

    control_joint = right_foot
    control_joint_traj = keyframe_by_length( get_joint(joints, control_joint), self.length )
    center_of_mass_traj= keyframe_by_length( get_center_of_mass(joints), self.length )
    loss_mass_ctr = dist_to_point( dimXZ(control_joint_traj), dimXZ(center_of_mass_traj) ) #**2
    # loss_mass_ctr = dist_to_point_squared( dimXZ(control_joint_traj), dimXZ(center_of_mass_traj) )
    loss_mass_ctr = loss_mass_ctr.mean()
    return loss_mass_ctr 


