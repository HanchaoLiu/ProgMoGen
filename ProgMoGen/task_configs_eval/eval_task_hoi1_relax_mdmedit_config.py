import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 
from atomic_lib.relax_geometry import construct_plane, calc_RT_from_two_planes, apply_RT_on_joints


from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from config_data import SKEL_JOINTS_TEMPLATE_FILE_NAME


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


# def f_eval(self, sample, sample_0, target, XZ_offset=False):
#     # get loss 
#     if XZ_offset==False:
#         joints = self.sample_to_joints(sample)
#     else:
#         joints = self.sample_to_joints_with_XZ_offset(sample)

#     assert joints.shape[2]==3
#     length = self.length
#     joint_control = left_wrist
#     p0_pred, p1_pred = get_target_pred_from_pos(joints, length, joint_control)
#     p0_gt, p1_gt     = target[0,:3], target[0,3:]
#     loss = equal_sum(p0_pred, p0_gt) + equal_sum(p1_pred, p1_gt)
#     loss = loss / 2
#     loss_ret_val = loss.sqrt()
#     return loss_ret_val


def f_eval(self, sample, sample_0, target, frame0, XZ_offset=False):


    if XZ_offset==False:
        # joints=self.sample_to_joints(sample)
        joints=self.sample_to_joints_from_rot(sample, frame0)
    else:
        joints=self.sample_to_joints_with_XZ_offset(sample)

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



def transform_sample(self, sample_ret, target_relaxed, target, frame0):

    # joints = self.sample_to_joints(sample_ret)

    RECOVER_TYPE="rot" 
    if RECOVER_TYPE=="pos":
        joints_src = self.sample_to_joints(sample_ret)
    elif RECOVER_TYPE=="rot":
        joints_src = self.sample_to_joints_from_rot(sample_ret, frame0)
    else:
        raise ValueError()

    target_gt_list_relax = [target_relaxed[0][:3], target_relaxed[0][3:]] 
    target_gt_list       = [target[0][:3], target[0][3:]]

    # find transformation from self.target_gt_list_relax to self.target_gt_list 
    R,translation = calc_RT_between_goals(target_gt_list_relax, target_gt_list)

    check_trans(target_gt_list_relax, target_gt_list, R,translation)

    # apply transformation on joints 
    do_relax=True 
    if do_relax:
        joints_dst = apply_RT_on_joints(joints_src, R,translation)
    else:
        joints_dst = joints_src

    # 0404: add last frame 
    joints_dst = torch.cat([joints_dst, joints_dst[:,:,:,-1:]],3)

    # joints to sample 
    sample_ret_dst = self.joints_to_sample_with_XZ_offset(joints_dst)

    if True:
        loss_new = self.f_eval(sample_ret_dst, None, target, frame0, XZ_offset=True).data.cpu().numpy()
        loss_old = self.f_eval(sample_ret, None, target_relaxed, frame0, XZ_offset=False).data.cpu().numpy()
        print("--->loss_old=", loss_old.mean(), "loss_new=",loss_new.mean())

        joints_dst_recon = self.sample_to_joints_with_XZ_offset(sample_ret_dst)
        print(joints_dst.shape, joints_dst_recon.shape)
        diff = np.abs(joints_dst[:,:,:,:196].data.cpu().numpy() - joints_dst_recon.data.cpu().numpy()  )

        if RECOVER_TYPE=="rot":
            assert np.abs(loss_new.mean()-loss_old.mean()) < 1e-5


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








def ddim_sample_loop_opt_fn_goal_relaxed(
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
    if dump_steps is not None:
        raise NotImplementedError()
    if const_noise == True:
        raise NotImplementedError()
    
    import time 
    t0 = time.time()
    
    ##################################################################################
    # phase1: generate a sample
    ##################################################################################
    final = None
    res_list=[]
    self.n_noise=0

    # print("model.device=", model.device)
    device = next(model.parameters()).device

    

    # rng = np.random.default_rng(10)
    rng = np.random.default_rng(self.np_seed)
    # print(f"using np_seed in diffusion = {self.np_seed}")
    noise_list_npy = [rng.standard_normal(size=shape) for _ in range(self.num_timesteps)]
    noise_init_npy = rng.standard_normal(size=shape)
    noise_list = [th.FloatTensor(a).to(device) for a in noise_list_npy]
    noise_init = th.FloatTensor(noise_init_npy).to(device)
    
    assert noise is None 

    # load dataset std 
    self.load_inv_normalization_data(device)
    # load templates
    self.skel_joints_template = load_frame0_template(device)
    self.skeleton = load_skeleton(device)
    joints_pos_frame0 = self.skel_joints_template

    noise_init.requires_grad=False
    eta=0.0
    assert eta==0.0

    self.length = model_kwargs['y']['lengths'].item()

    # (1, 6)
    # np.ndarray
    # target_list = model_kwargs['y']['target_list']

    # target, target_relaxed
    # (1,6)
    target = self.target_gt
    target_relaxed = None

    #### start relax
    import copy 
    model_kwargs_nopaint = copy.deepcopy(model_kwargs)
    model_kwargs_nopaint["y"].pop("inpainted_motion")
    model_kwargs_nopaint["y"].pop("inpainting_mask")
    assert ("inpainted_motion" not in model_kwargs_nopaint["y"].keys()) and ("inpainting_mask" not in model_kwargs_nopaint["y"].keys())
    pred_res, res_list, pred_x0_list = self.f_forward_return_middle_list(model, shape, noise_list, noise_init, init_image, model_kwargs_nopaint, eta=eta, progress=False)
    pred_res_0 = pred_res.detach().clone()


    # target_relaxed = self.update_goal(pred_res)
    i_k = 0
    target_relaxed = self.update_goal(pred_res, target, target_relaxed, i_k)

    joint_control = left_wrist

    ##################################################################################
    # phase1: prepare inpainting signals
    ##################################################################################
    def get_hoipick_move_ref_gt(pred_res, length, target_relaxed):


        joint_idx = joint_control
        joint_idx_params_list = self.parse_joint_idx_list(joint_idx)
        
        pred_res_ret = pred_res.detach().clone()
        seq_len = pred_res_ret.shape[-1]

        t_all = length - 1
        t_0 = 0
        t_middle = length//2
        
        # (seqlen,3)
        fill_data = np.zeros((seq_len,3)).astype(np.float32)
        projected_curve = th.FloatTensor(fill_data)
        

        projected_curve[t_0, :]      = target_relaxed[0,:3]
        projected_curve[t_all, :]    = target_relaxed[0,3:6]

        # (196,3)
        local_pose_this = self.reverse_to_local_pose_wrapper(pred_res_ret, projected_curve, joint_idx=joint_idx)
        # ->[1, 3, 1, 196]
        pred_res_inv = self.do_inv_norm(pred_res_ret)
        # torch.Size([1, 263, 1, 196])

        pred_res_inv[:,joint_idx_params_list,:,:] = local_pose_this.t()[None,:,None,:]
        pred_res_recovered = self.do_norm(pred_res_inv)
        
        return pred_res_recovered



    # get trajectory for each (in batch form)
    sample_inpaint = get_hoipick_move_ref_gt(pred_res, self.length, target_relaxed)

    model_kwargs['y']['inpainted_motion'] = sample_inpaint
    a = model_kwargs['y']['inpainting_mask'] * 0

    # head 
    joint_idx = joint_control
    joint_idx_params_list = self.parse_joint_idx_list(joint_idx)
    print("joint_idx_params_list=", joint_idx_params_list)
    # torch.Size([32, 263, 1, 196])

    # root trajectory = 4
    a[:,[0,1,2],:,:] = 1 
    # y 
    t_st  = 0
    t_end = self.length-1
    a[:,joint_idx_params_list,:,t_st] = 1
    a[:,joint_idx_params_list,:,t_end] = 1
    model_kwargs['y']['inpainting_mask'] = a 


    ##################################################################################
    # phase2: inpainting inpainting signals
    ##################################################################################
    final = None
    res_list=[]
    self.n_noise=0

    # print("model.device=", model.device)
    device = next(model.parameters()).device
    

    # rng = np.random.default_rng(10)
    rng = np.random.default_rng(self.np_seed)
    # print(f"using np_seed in diffusion = {self.np_seed}")
    noise_list_npy = [rng.standard_normal(size=shape) for _ in range(self.num_timesteps)]
    noise_init_npy = rng.standard_normal(size=shape)
    noise_list = [th.FloatTensor(a).to(device) for a in noise_list_npy]
    noise_init = th.FloatTensor(noise_init_npy).to(device)
    
    assert noise is None 

    # load dataset std 
    self.load_inv_normalization_data(device)
    # load templates
    self.skel_joints_template = load_frame0_template(device)
    self.skeleton = load_skeleton(device)
    joints_pos_frame0 = self.skel_joints_template

    noise_init.requires_grad=False
    eta=0.0
    assert eta==0.0


    self.length = model_kwargs['y']['lengths'].item()

    if False:
        model_kwargs["y"].pop("inpainted_motion")
        model_kwargs["y"].pop("inpainting_mask")
        assert ("inpainted_motion" not in model_kwargs["y"].keys()) and ("inpainting_mask" not in model_kwargs["y"].keys())

    pred_res, res_list, pred_x0_list = self.f_forward_return_middle_list(model, shape, noise_list, noise_init, init_image, model_kwargs, eta=eta, progress=False)
    pred_res_ret = pred_res.detach().clone()


    # eval on target_relaxed
    d = self.f_eval(pred_res_ret, pred_res_0, target_relaxed, joints_pos_frame0)
    print("[last] d.mean() = ", d.mean(), d.max(), d.min())


    # transform
    pred_res_ret_dst = self.transform_sample(pred_res_ret, target_relaxed, target, joints_pos_frame0)

    # line_target = construct_line(target[0, :3], target[0, 3:])
    self.loss_ret_val = self.f_eval(pred_res_ret_dst, pred_res_0, target, joints_pos_frame0, XZ_offset=True).data.cpu().numpy()

    print("->[final loss_ret_val for joints_transformed] = ", self.loss_ret_val.mean())

    # fix the problem of seqlen-1.
    return pred_res_ret_dst
    






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

