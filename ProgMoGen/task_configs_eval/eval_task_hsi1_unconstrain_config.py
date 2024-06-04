import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: head height constraint on the first, middle and last frames.
'''




def f_loss(self, sample):
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


def f_eval(self, sample):
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

    noise_init.requires_grad=False
    # 
    eta=0.0
    

    # fit plane first 
    pred_res, res_list, pred_x0_list = self.f_forward_return_middle_list(model, shape, noise_list, noise_init, init_image, model_kwargs, eta=eta, progress=False)
    # (seqlen, 3)
    pred_res_ret = pred_res.detach().clone()
    del pred_res, res_list, pred_x0_list

    self.length = model_kwargs['y']['lengths'].item()

        
    loss = self.f_loss(pred_res_ret)

    print(f"loss={loss.item():.6f}")
        

    t_end = time.time()
    t_elapsed = t_end - t0
    print(f"t = {t_elapsed:.1f}")

    
    loss_head = self.f_eval(pred_res_ret)
    self.loss_ret_val = loss_head.data.cpu().numpy()

    return pred_res_ret


