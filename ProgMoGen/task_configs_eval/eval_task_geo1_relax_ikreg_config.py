import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_root_rot_pos, reverse_pose
# from data_loaders.humanml.scripts.motion_process import recover_from_rot, recover_from_rot_with_skeleton

from atomic_lib.relax_geometry import construct_plane, calc_RT_from_two_planes, apply_RT_on_joints

from atomic_lib.math_utils import * 

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from config_data import SKEL_JOINTS_TEMPLATE_FILE_NAME

'''
task: left hand on a plane.
'''


DEMO_NUM=32


lr=1e-5
epoch_relax=1
iterations=100


joint_control=left_wrist


def get_lr_schedule(self, i_k):
    # if i_k in [0]:
    #     base_lr = 1e-2
    # elif i_k in [1,2]:
    #     base_lr = 5e-3
    # else:
    #     base_lr = 1e-3
    # return base_lr
    # base_lr = 1e-2
    base_lr = 2e-2
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


def f_loss(self, sample, sample_0, it, plane_params, frame0):

    joint_idx=joint_control
    # joints = self.sample_to_joints(sample)
    joints = self.sample_to_joints_from_rot(sample, frame0)

    # torch.Size([1, 22, 3, 196])
    assert joints.shape[0]==1
    # (seqlen, 3)
    joint_traj = get_joint(joints, joint_idx).squeeze()
    joint_traj = joint_traj.permute(1,0).contiguous()
    loss_plane = loss_dist_to_plane(joint_traj, plane_params).mean()


    joints_num=22
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    loss_reg = F.mse_loss(sample[:,0:end_indx,:,1:],sample[:,0:end_indx,:,:-1])
    loss = loss_plane + loss_reg

    return loss
    

def f_eval(self, sample, sample_0, plane_params, frame0, XZ_offset=False):

    if plane_params.shape[1]==6:
        plane_target = construct_plane(plane_params[0, :3], plane_params[0, 3:])
        plane_target_4d_params = plane_point_normal_form_to_params_4d(plane_target)
        plane_params = plane_target_4d_params

    joint_idx=joint_control

    if XZ_offset==False:
        # joints=self.sample_to_joints(sample)
        joints=self.sample_to_joints_from_rot(sample, frame0)
    else:
        joints=self.sample_to_joints_with_XZ_offset(sample)

    joints = get_joint(joints, joint_idx)
    loss_plane = dist_to_plane(joints, plane_params)
    return loss_plane
    





def update_goal(self, sample, target, target_relaxed, i_k, frame_0):
    # return target_relaxed.
    # update_relaxed_goal()
    # (seqlen, 3)
    joint_idx=joint_control
    # (seqlen, 3)   
    # joint_traj = self.get_global_traj_for_joints(sample, joint_idx)
    joint_traj = self.get_global_traj_for_joints_from_rot(sample, joint_idx, frame_0)

    # get plane_params
    plane_params = fit_yPlane(joint_traj)
    return plane_params


def transform_sample(self, sample_ret, target_relaxed, target, frame0):
    # apply transform back 

    # update target relaxed again using sample_ret
    if True:
        target_relaxed = self.update_goal(sample_ret, None, None, None, frame0)

    plane_params = target_relaxed
    target_list = target

    plane_pn     = plane_params_4d_to_point_normal_form(plane_params)
    plane_relax  = construct_plane(plane_pn[0,:3], plane_pn[0, 3:])

    # target_list = th.FloatTensor(target_list).to(device)
    target_list  = plane_params_4d_to_point_normal_form(target_list)
    plane_target = construct_plane(target_list[0, :3], target_list[0, 3:])
    R,translation = calc_RT_from_two_planes(plane_relax, plane_target)

    # joints_src = self.sample_to_joints(sample_ret)
    joints_src = self.sample_to_joints_from_rot(sample_ret, frame0)
    # torch.Size([1, 22, 3, 196])
    # joints_dst = apply_RT(joints_src, R,T)
    joints_dst = apply_RT_on_joints(joints_src, R,translation)
    
    # 0404: add last frame 
    joints_dst = torch.cat([joints_dst, joints_dst[:,:,:,-1:]],3)
    sample_ret_dst = self.joints_to_sample_with_XZ_offset(joints_dst)


    if True:
        loss_new = self.f_eval(sample_ret_dst, None, target, frame0, XZ_offset=True).data.cpu().numpy()
        loss_old = self.f_eval(sample_ret, None, target_relaxed, frame0, XZ_offset=False).data.cpu().numpy()
        print("--->loss_old=", loss_old.mean(), "loss_new=",loss_new.mean())

        joints_dst_recon = self.sample_to_joints_with_XZ_offset(sample_ret_dst)
        print(joints_dst.shape, joints_dst_recon.shape)
        diff = np.abs(joints_dst[:,:,:,:196].data.cpu().numpy() - joints_dst_recon.data.cpu().numpy()  )
        assert np.abs(loss_new.mean()-loss_old.mean()) < 1e-5
    

    return sample_ret_dst









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
    

    self.length = model_kwargs['y']['lengths'].item()

    # (1, 6)
    # np.ndarray
    # target_list = model_kwargs['y']['target_list']

    # target, target_relaxed
    # (1,6)
    target = self.target_gt
    target_relaxed = None

    # base_lr = self.lr 
    epoch_relax = self.epoch_relax
    max_it  = self.iterations
    print(f"epoch_relax={epoch_relax}, max_it={max_it}")

    for i_k, k in enumerate(range(epoch_relax)):
        
        #### start relax
        pred_res, res_list, pred_x0_list = self.f_forward_return_middle_list(model, shape, noise_list, noise_init, init_image, model_kwargs, eta=eta, progress=False)
        if i_k==0:
            pred_res_0 = pred_res.detach().clone()

        # target_relaxed = self.update_goal(pred_res)
        target_relaxed = self.update_goal(pred_res, target, target_relaxed, i_k, joints_pos_frame0)
        print("shape check:", target.shape, target_relaxed.shape)

        pred_res_ret = pred_res.detach().clone()
        # del pred_res, res_list, pred_x0_list
        
        # evaluate error w.r.t relaxed goal.
        d = self.f_eval(pred_res_ret, pred_res_0, target_relaxed, joints_pos_frame0)
        print("[first] d.mean() = ", d.mean(), d.max(), d.min())
        #### end relax

        # noise_init.requires_grad=True
        noise_init.requires_grad=False

        # if i_k in [0]:
        #     base_lr = 0.01
        # else:
        #     base_lr = 0.005

        base_lr = self.get_lr_schedule(i_k)
        

        # optimizer = optim.Adam([noise_init], base_lr)
        pred_res.requires_grad=True
        optimizer = optim.Adam([pred_res], base_lr)

        for it in range(max_it):

            self.adjust_learning_rate(optimizer,base_lr,it,step=max_it)
            lr_curr = self.get_optimizer_lr(optimizer)

            optimizer.zero_grad()

            # pred_res, res_list, pred_x0_list = self.f_forward_return_middle_list(model, shape, noise_list, noise_init, init_image, model_kwargs, eta=eta, progress=False)
            pred_res_ret = pred_res.detach().clone()
            
            # # torch.Size([1, 22, 3, 196])
            loss = self.f_loss(pred_res, pred_res_0, it, target_relaxed, joints_pos_frame0)
            loss_eval = self.f_eval(pred_res, pred_res_0, target_relaxed, joints_pos_frame0).mean()

            print(f"{it}, lr={lr_curr:.4f}, loss={loss.item():.6f}, loss_eval={loss_eval.item():.6f}")
            
            loss.backward()
            optimizer.step()

            pred_res_ret = pred_res.detach().clone()

            # del pred_res, res_list, pred_x0_list
        
        # noise_init.requires_grad=False

        d = self.f_eval(pred_res_ret, pred_res_0, target_relaxed, joints_pos_frame0)
        print("[last] d.mean() = ", d.mean(), d.max(), d.min())

    self.loss_str = f"mean:{d.mean().item():.3f},max:{d.max().item():.3f}"
    t_end = time.time()
    t_elapsed = t_end - t0
    print(f"t = {t_elapsed:.1f}")
    self.loss_ret_val = d.data.cpu().numpy()

    pred_res_ret_dst = self.transform_sample(pred_res_ret, target_relaxed, target, joints_pos_frame0)

    # line_target = construct_line(target[0, :3], target[0, 3:])
    self.loss_ret_val = self.f_eval(pred_res_ret_dst, pred_res_0, target, joints_pos_frame0, XZ_offset=True).data.cpu().numpy()

    print("->[final loss_ret_val for joints_transformed] = ", self.loss_ret_val.mean())

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





