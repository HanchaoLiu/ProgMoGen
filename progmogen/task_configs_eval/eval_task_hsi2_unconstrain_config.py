import torch as th 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import time 

from atomic_lib.math_utils import * 

'''
task: overhead barrier.
'''



def loss_overhead_barrier(joints, length):
    t_all = length - 1
    t_0 = 0
    t_middle = length//2

    loss_head = greater_than( keyframe(dimY(get_joint(joints, head)), t_0), 1.5 ) + \
                greater_than( keyframe(dimY(get_joint(joints, head)), t_all), 1.5 ) + \
                less_than( keyframe(dimY(get_joint(joints, head)), t_middle), 0.5 )
    loss_head = loss_head / 3

    # because of the motion prior, the feet will actually become closed to the ground.
    # avoiding floating feet
    loss_foot = less_than( keyframe(dimY(get_joint(joints, left_foot)), t_middle), 0.0 ) + \
                less_than( keyframe(dimY(get_joint(joints, right_foot)), t_middle), 0.0 )
    loss_foot = loss_foot / 2
    
    loss = loss_foot + loss_head
    return loss 



def f_loss(self, sample, sample_0, step):
    '''
    joints: [1,22,3,t]
    '''
    joints = self.sample_to_joints(sample)
    assert joints.shape[0]==1

    length = self.length 
    loss = loss_overhead_barrier(joints, length)
    
    return loss


def f_eval(self, sample, sample_0):
    '''
    joints: [1,22,3,t]
    '''
    joints = self.sample_to_joints(sample)
    assert joints.shape[0]==1

    length = self.length 
    loss = loss_overhead_barrier(joints, length)
    
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


    noise_init.requires_grad=False
    # 
    eta=0.0
    


    # get first
    pred_res, res_list, pred_x0_list = self.f_forward_return_middle_list(model, shape, noise_list, noise_init, init_image, model_kwargs, eta=eta, progress=False)
    pred_res_0=pred_res.detach().clone()

    pred_res_ret = pred_res.detach().clone()
            
    t_end = time.time()
    t_elapsed = t_end - t0
    print(f"t = {t_elapsed:.1f}")

    # get loss 
    loss_head = self.f_eval(pred_res_ret, pred_res_0)
    self.loss_ret_val = loss_head.data.cpu().numpy()

    return pred_res_ret

