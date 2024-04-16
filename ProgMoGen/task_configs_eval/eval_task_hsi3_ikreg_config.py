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
task: inside a square.
'''

# TODO: requires token
# TEXT_PROMPT="drunk walking animation turning around."
# LENGTH=196
# DEMO_NUM=4


# optimizer params
# lr=0.005
# iterations=50
# decay_steps=None


def loss_limited_space(joints):
    '''
    joints: [1,22,3,t]
    '''
    loss_1 = less_than(    dimX(joints), 1.0 ).mean()
    loss_2 = greater_than( dimX(joints), -1.0 ).mean()
    loss_3 = less_than(    dimZ(joints), 1.0 ).mean()
    loss_4 = greater_than( dimZ(joints), -1.0 ).mean()
    loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4 
    return loss 


def f_loss(self, sample, frame0):
    # joints = self.sample_to_joints(sample)
    joints = self.sample_to_joints_from_rot(sample, frame0)
    loss = loss_limited_space(joints)

    joints_num=22
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    loss_reg = F.mse_loss(sample[:,0:end_indx,:,1:],sample[:,0:end_indx,:,:-1])
    loss = loss + loss_reg

    return loss 


def f_eval(self, sample, frame0):
    # joints = self.sample_to_joints(sample)
    joints = self.sample_to_joints_from_rot(sample, frame0)
    loss = loss_limited_space(joints)
    return loss






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
    max_it = 120
    optimizer = optim.Adam([pred_res], base_lr)
    for it in range(max_it):

        self.adjust_learning_rate(optimizer,base_lr,it,step=int(max_it/6*5))
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





