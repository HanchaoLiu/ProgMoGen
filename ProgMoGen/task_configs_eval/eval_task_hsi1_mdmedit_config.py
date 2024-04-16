import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 
import copy 



from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain

from config_data import SKEL_JOINTS_TEMPLATE_FILE_NAME

from atomic_lib.math_utils import * 

'''
task: head height constraint on the first, middle and last frames.
'''




def f_loss(self, sample, joints_pos_frame0):
    '''
    '''
    assert sample.shape[0]==1

    def loss_head_from_rot(sample, h0, t, frame0):
        #  torch.Size([1, 22, 3, 196])
        joints=self.sample_to_joints_from_rot(sample, frame0)
        # head_idx = 15
        # h_pred = joints[0,head_idx,1,t]
        # loss = ((h_pred - h0)**2).mean()
        h_pred = keyframe(dimY(get_joint(joints, head)), t)
        loss = equal(h_pred, h0)
        return loss  

    t_all = self.length - 1
    t_0 = 0
    t_middle = self.length //2

    loss =      loss_head_from_rot(sample, h0=self.h_gt_list[1], t=t_middle, frame0=joints_pos_frame0)*2 + \
                loss_head_from_rot(sample, h0=self.h_gt_list[0], t=t_0, frame0=joints_pos_frame0) + \
                loss_head_from_rot(sample, h0=self.h_gt_list[2], t=t_all, frame0=joints_pos_frame0)
    return loss


def f_eval(self, sample, joints_pos_frame0):
    '''
    '''
    assert sample.shape[0]==1

    def loss_head_from_rot(sample, h0, t, frame0):
        #  torch.Size([1, 22, 3, 196])
        joints=self.sample_to_joints_from_rot(sample, frame0)
        # head_idx = 15
        # h_pred = joints[0,head_idx,1,t]
        # loss = ((h_pred - h0)**2).mean()
        h_pred = keyframe(dimY(get_joint(joints, head)), t)
        loss = equal(h_pred, h0)
        return loss  

    t_all = self.length - 1
    t_0 = 0
    t_middle = self.length //2

    loss =      loss_head_from_rot(sample, h0=self.h_gt_list[1], t=t_middle, frame0=joints_pos_frame0)*2 + \
                loss_head_from_rot(sample, h0=self.h_gt_list[0], t=t_0, frame0=joints_pos_frame0) + \
                loss_head_from_rot(sample, h0=self.h_gt_list[2], t=t_all, frame0=joints_pos_frame0)
    return loss


def f_eval_pos(self, sample):
    '''
    '''
    assert sample.shape[0]==1

    def loss_head(sample, h0, t):
        #  torch.Size([1, 22, 3, 196])
        joints = self.sample_to_joints(sample)
        # head_idx = 15
        # h_pred = joints[0,head_idx,1,t]
        # loss = ((h_pred - h0)**2).mean()
        h_pred = keyframe(dimY(get_joint(joints, head)), t)
        loss = equal(h_pred, h0)
        return loss  

    t_all = self.length - 1
    t_0 = 0
    t_middle = self.length //2

    loss = loss_head(sample, h0=self.h_gt_list[1], t=t_middle)*2 + \
                loss_head(sample, h0=self.h_gt_list[0], t=t_0) + \
                loss_head(sample, h0=self.h_gt_list[2], t=t_all)
    return loss








def load_frame0_template(device):
    x = np.load(SKEL_JOINTS_TEMPLATE_FILE_NAME, allow_pickle=True).item()
    motion = x["motion"][0:1,:,:,0]
    motion = th.FloatTensor(motion).to(device)
    return motion

def load_skeleton(device):
    n_raw_offsets = th.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain
    skeleton = Skeleton(n_raw_offsets, kinematic_chain, device)
    return skeleton





def ddim_sample_loop_opt_fn(
    self,
    model,
    shape,
    noise=None,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    eta=0.0,
    skip_timesteps=0,
    init_image=None,
    randomize_class=False,
    cond_fn_with_grad=False,
    dump_steps=None,
    const_noise=False,
    ref_motions=None
):
    """
    Generate samples from the model using DDIM.

    Same usage as p_sample_loop().
    """

    device = next(model.parameters()).device
    shape0 = [1, 263, 1, 196]

    model_kwargs_nopaint_each = copy.deepcopy(model_kwargs)
    model_kwargs_nopaint_each['y'].pop("inpainted_motion")
    model_kwargs_nopaint_each['y'].pop("inpainting_mask")

    assert ("inpainted_motion" not in model_kwargs_nopaint_each['y'].keys()) and ("inpainting_mask" not in model_kwargs_nopaint_each['y'].keys())

    # run without inpainting
    sample_each = self.ddim_sample_loop_opt_fn_base(
        model,
        shape0,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs_nopaint_each,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        # init_image=motion.to(dist_util.dev()),
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
    )

    model_kwargs_each = model_kwargs

    # inpainted_motion
    sample_inpaint = self.get_head_height_ref_batch_gt(sample_each, model_kwargs_nopaint_each)
    model_kwargs_each['y']['inpainted_motion'] = sample_inpaint
    
    # inpainting_mask
    a = model_kwargs_each['y']['inpainting_mask'] * 0
    # head 
    # joint_idx = 15
    joint_idx = head
    joint_idx_params_list = self.parse_joint_idx_list(joint_idx)
    # print("joint_idx_params_list=", joint_idx_params_list)
    # torch.Size([32, 263, 1, 196])
    # root trajectory = 4
    a[:,[0,1,2],:,:] = 1 
    # y 
    for ki, length in enumerate(model_kwargs_each['y']['lengths']):
        t_all = length - 1
        t_0 = 0
        t_middle = length//2
        # a[:,joint_idx_params_list[1],:,:] = 1
        a[ki,joint_idx_params_list[1],:,[t_0, t_middle, t_all]] = 1
    model_kwargs_each['y']['inpainting_mask'] = a 


    # denoise again.
    sample_each = self.ddim_sample_loop_opt_fn_base(
        model,
        shape0,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs_each,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        # init_image=motion.to(dist_util.dev()),
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
    )

    # (1,3,22,196)
    # get loss 
    pred_res_ret = sample_each.detach().clone()


    self.skel_joints_template = load_frame0_template(device)
    joints_pos_frame0 = self.skel_joints_template
    self.skeleton = load_skeleton(device)
    loss_head = self.f_eval(pred_res_ret, joints_pos_frame0)
    # loss_head = self.f_eval(pred_res_ret)
    self.loss_ret_val = loss_head.data.cpu().numpy()
    print("loss_ret_val=", self.loss_ret_val)

    return pred_res_ret








