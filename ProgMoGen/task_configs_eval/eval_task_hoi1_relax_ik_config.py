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
epoch_relax=1
iterations=80



def get_lr_schedule(self, i_k):
    # if i_k in [0,1]:
    #     base_lr=0.02
    # elif i_k in [2]:
    #     base_lr=0.002
    base_lr=0.1
    return base_lr




def get_target_pred_from_pos(joints, length, joint_control):
    # torch.Size([1, 22, 3, 196])
    # print(joints_ref.shape)
    t_all = length - 1
    t_0 = 0

    p0 = keyframe(get_joint(joints, joint_control),t_0).squeeze()
    p1 = keyframe(get_joint(joints, joint_control), t_all).squeeze()
    return [p0,p1]



def f_loss(self, sample, sample_0, it, target, frame0):

    # joints = self.sample_to_joints(sample)
    joints = self.sample_to_joints_from_rot(sample, frame0)

    length = self.length
    joint_control = left_wrist
    p0_pred, p1_pred = get_target_pred_from_pos(joints, length, joint_control)
    p0_gt, p1_gt     = target[0,:3], target[0,3:]
    loss = equal_sum(p0_pred, p0_gt) + equal_sum(p1_pred, p1_gt)
    loss = loss / 2

    return loss 

    
def f_eval(self, sample, sample_0, target, frame0, XZ_offset=False):

    if XZ_offset==False:
        # joints=self.sample_to_joints(sample)
        joints=self.sample_to_joints_from_rot(sample, frame0)
    else:
        joints=self.sample_to_joints_with_XZ_offset(sample)

    length = self.length
    joint_control = left_wrist
    p0_pred, p1_pred = get_target_pred_from_pos(joints, length, joint_control)
    p0_gt, p1_gt     = target[0,:3], target[0,3:]
    loss = equal_sum(p0_pred, p0_gt) + equal_sum(p1_pred, p1_gt)
    loss = loss / 2
    loss_ret_val = loss.sqrt()
    return loss_ret_val 






def update_goal(self, sample, target, target_relaxed, i_k, frame0):
    # return target_relaxed.
    # update_relaxed_goal()

    length = self.length
    joint_control = left_wrist
    
    p0_pred, p1_pred = get_target_pred_from_pos(self.sample_to_joints_from_rot(sample,frame0), length, joint_control)
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



def transform_sample(self, sample_ret, target_relaxed, target, frame_0):

    joints = self.sample_to_joints_from_rot(sample_ret, frame_0)

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

        print('pred_res.requires_grad=', pred_res.requires_grad)


        target_relaxed = self.update_goal(pred_res, target, target_relaxed, i_k, joints_pos_frame0)
        print("shape check:", target.shape, target_relaxed.shape)

        pred_res_ret = pred_res.detach().clone()
        # del pred_res, res_list, pred_x0_list
        
        # evaluate error w.r.t relaxed goal.
        d = self.f_eval(pred_res_ret, pred_res_0, target_relaxed, joints_pos_frame0)
        print("[first] d.mean() = ", d.mean(), d.max(), d.min())
        #### end relax

        noise_init.requires_grad=False


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

            # print(f"{it}, lr={lr_curr:.4f}, loss={loss.item():.6f}, loss_eval={loss_eval.item():.6f}")
            
            loss.backward()
            optimizer.step()

            pred_res_ret = pred_res.detach().clone()

        
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





