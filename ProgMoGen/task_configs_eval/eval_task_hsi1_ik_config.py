import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 



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
    if False:
        noise_list = [th.randn(*shape, device=device) for _ in range(self.num_timesteps)]
        noise_init = th.randn(*shape, device=device)  

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

    noise_init.requires_grad=False
    # assert eta==1.0
    eta=0.0
    assert eta==0.0

    joints_pos_frame0 = self.skel_joints_template

    # get first
    pred_res, res_list, pred_x0_list = self.f_forward_return_middle_list(model, shape, noise_list, noise_init, init_image, model_kwargs, eta=eta, progress=False)
    pred_res_0=pred_res.detach().clone()
    pred_res.requires_grad=True

    base_lr = 1e-2
    max_it = 100
    optimizer = optim.Adam([pred_res], base_lr)
    for it in range(max_it):

        self.adjust_learning_rate(optimizer,base_lr,it,step=max_it)
        lr_curr = self.get_optimizer_lr(optimizer)

        optimizer.zero_grad()

        loss_head = self.f_loss(pred_res, joints_pos_frame0)
        loss = loss_head
        # print(f"{it}, lr={lr_curr:.4f}, loss={loss.item():.6f}, loss_head={loss_head.item():.6f}")
        
        loss.backward()
        optimizer.step()
        # del pred_res, res_list, pred_x0_list
        
    loss = self.f_loss(pred_res, joints_pos_frame0)
    print("[last] loss = ", loss)
        
    self.loss_str = f"loss={loss.item():.4f}"

    pred_res_ret = pred_res.detach().clone()
            
    t_end = time.time()
    t_elapsed = t_end - t0
    print(f"t = {t_elapsed:.1f}")

    # get loss 
    loss_head = self.f_eval(pred_res_ret, joints_pos_frame0)
    self.loss_ret_val = loss_head.data.cpu().numpy()

    return pred_res_ret






