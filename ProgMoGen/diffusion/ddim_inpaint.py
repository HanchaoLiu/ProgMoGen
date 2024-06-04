
from diffusion.respace import SpacedDiffusion
from .gaussian_diffusion_v2 import _extract_into_tensor

import numpy as np 

import torch as th
import os,sys 
from IPython import embed 
from torch import optim

import torch.nn.functional as F

from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_root_rot_pos, reverse_pose
from data_loaders.humanml.scripts.motion_process import recover_from_rot, recover_from_rot_with_skeleton

from config_data import ROOT_DIR

class InpaintingGaussianDiffusion(SpacedDiffusion):
    def q_sample(self, x_start, t, noise=None, model_kwargs=None):
        """
        overrides q_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        bs, feat, _, frames = noise.shape
        # noise *= 1. - model_kwargs['y']['inpainting_mask']
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
            )
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        overrides p_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        # with th.no_grad():
        if True:
            

            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )

        # In [3]: out['mean'].requires_grad
        # Out[3]: True
        # In [4]: out['variance'].requires_grad
        # Out[4]: False
        # In [5]: out['log_variance'].requires_grad
        # Out[5]: False
        # In [6]: out['pred_xstart'].requires_grad
        # Out[6]: True
            
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)
        # noise *= 1. - model_kwargs['y']['inpainting_mask']
        # print("noise=", noise.shape)
        # set requires_grad_

        # if t[0]==self.num_timesteps-1:
        print("t=", t)
        

        

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "noise": noise}
    


    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """

        
        

        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )



        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        self.n_noise = self.n_noise + 1

        # print("t=", t)
        # if t[0]==90:
        if t[0]==99:
            noise.requires_grad_()


        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        # !!!!! # no noise when t == 0
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"], "noise": noise}



    def adjust_learning_rate(self, optimizer,base_lr,epoch,step):
        lr = base_lr * (
                0.1 ** np.sum(epoch >= np.array(step)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_optimizer_lr(self, optimizer):
        lr = optimizer.param_groups[0]['lr']
        return lr 

    def ddim_sample_loop_progressive_opt(
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
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
            self.n_noise = self.n_noise + 1

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img, model_kwargs=model_kwargs)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            # with th.no_grad():
            if True:
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]




    def f_forward_return_middle_list(self, model, shape, noise_list, noise_init, init_image, model_kwargs, 
        clip_denoised = False, denoised_fn=None, progress=True, 
        eta=0.0,
        skip_timesteps=0,
        randomize_class=False,
        cond_fn_with_grad=False,
        cond_fn=None
        ):

        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        final = None

        res_list=[]
        predxstart_list=[]

        # print("model.device=", model.device)
        device = next(model.parameters()).device
        
        noise=None
        assert noise is None 

        for sample in self.ddim_sample_loop_progressive_opt_known_noise(
            model,
            shape,
            noise=noise,
            noise_list = noise_list , 
            noise_init = noise_init ,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample

            res_list.append( [sample["sample"].detach(), sample["pred_xstart"].detach(),  sample["noise"].detach()] )
            predxstart_list.append(sample["pred_xstart"])


        return final["sample"], res_list, predxstart_list
        # return res_list



    def load_inv_normalization_data(self, device):
        # model.nfeats = 1
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        root_dir = ROOT_DIR
        mean_file = os.path.join(root_dir, "dataset/HumanML3D/Mean.npy")
        std_file  = os.path.join(root_dir, "dataset/HumanML3D/Std.npy")

        d_mean = np.load(mean_file)
        d_std  = np.load(std_file)
        d_mean = th.FloatTensor(d_mean).to(device)
        d_std  = th.FloatTensor(d_std).to(device)
        self.d_mean = d_mean[None,:,None,None]
        self.d_std  = d_std[None,:,None,None]


    def do_inv_norm(self, data):
        # model.nfeats = 1
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)

        assert data.shape[1]==self.d_mean.shape[1] and data.shape[2]==self.d_mean.shape[2]
        assert data.shape[1]==self.d_std.shape[1] and data.shape[2]==self.d_std.shape[2]

        return data * self.d_std + self.d_mean
        

    def do_norm(self, data):
        assert data.shape[1]==self.d_mean.shape[1] and data.shape[2]==self.d_mean.shape[2]
        assert data.shape[1]==self.d_std.shape[1] and data.shape[2]==self.d_std.shape[2]
        return (data - self.d_mean) / self.d_std
    


    # predefined noise 
    def ddim_sample_loop_progressive_opt_known_noise(
        self,
        model,
        shape,
        noise_list,
        noise_init,
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
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))


        
        img = noise_init

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img, model_kwargs=model_kwargs)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        
        # noise_list = [th.randn(*shape, device=device) for _ in range(self.num_timesteps)]

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            # with th.no_grad():
            if True:
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample_known_noise
                out = sample_fn(
                    model,
                    img,
                    t,
                    noise_list, 
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]



    def ddim_sample_known_noise(
        self,
        model,
        x,
        t,
        noise_list, 
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """

        

        assert clip_denoised==False
        

        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        

        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.

        # noise = noise_list[t]
        t0 = t[0].item()
        noise = noise_list[t0]
        

        self.n_noise = self.n_noise + 1


        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        # !!!!! # no noise when t == 0
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"], "noise": noise}







    def sample_to_joints(self, pred):
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        x_pred = pred
        x_pred = self.do_inv_norm(x_pred)
        
        sample = recover_from_ric(x_pred.permute(0,2,3,1).contiguous(), 22)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        # torch.Size([1, 22, 3, 196])
        assert sample.shape[0]==1
        return sample
    
    def sample_to_joints_from_rot(self, pred, skel_joints):
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        # skel_joints: (bs, 22, 3)
        x_pred = pred
        x_pred = self.do_inv_norm(x_pred)
        
        sample = recover_from_rot_with_skeleton(x_pred.permute(0,2,3,1).contiguous(), 22, self.skeleton, skel_joints)
        
        sample = sample.view(-1, *sample.shape[:]).permute(0, 2, 3, 1)

        # torch.Size([1, 22, 3, 196])
        assert sample.shape[0]==1
        return sample
    
    def get_joint_traj(self, sample, joint_idx):
        '''
        return (seqlen,3)
        '''
        # torch.Size([1, 22, 3, 196])
        assert sample.shape[0]==1
        res = sample[0,joint_idx,:,:]
        res = res.permute(1,0).contiguous()
        # (seqlen, 3)
        return res 


    def get_global_traj_for_joints_list(self, pred, joint_idx_list):
        joints = self.sample_to_joints(pred)
        # torch.Size([1, 22, 3, 196])
        assert joints.shape[0]==1 
        res = []
        for joint_idx in joint_idx_list:
            res_each = joints[0,joint_idx,:,:self.length].t()
            res.append( res_each )
        res = th.cat(res, 0)
        return res 

    def get_global_traj_for_joints(self, pred, joint_idx):
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        x_pred = pred
        x_pred = self.do_inv_norm(x_pred)
        
        sample = recover_from_ric(x_pred.permute(0,2,3,1).contiguous(), 22)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        # torch.Size([1, 22, 3, 196])
        assert sample.shape[0]==1
        res = sample[0,joint_idx,:,:]
        res = res.permute(1,0).contiguous()
        # (seqlen, 3)
        return res


    def parse_joint_idx_list(self,idx):
        st = 4 + (idx-1)*3
        ed = st + 3 
        return list(range(st, ed))



    def write_to_file(self, file_name, string):
        with open(file_name, "a+") as f:
            f.write(string+"\n")




    def ddim_sample_loop_opt_fn_base(
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
        

        pred_res, res_list, pred_x0_list = self.f_forward_return_middle_list(model, shape, noise_list, noise_init, init_image, model_kwargs, eta=eta, progress=False)
        pred_res_ret = pred_res.detach().clone()
        del pred_res, res_list, pred_x0_list

        return pred_res_ret



    def reverse_to_local_pose_wrapper(self, data, new_global_pos, joint_idx):
        '''
        data: # torch.Size([1, 263, 1, 196])
        new_global_pos: (seqlen, 3)
        return torch.Size([1, 263, 1, 196])
        '''
        #  torch.Size([1, 263, 1, 196]) -> torch.Size([1, 22, 3, 196])
        joints = self.sample_to_joints(data)

        joints[0,joint_idx,:,:] = new_global_pos.t()

        # data: [bs, 1, 196, 263]
        # joints: torch.Size([1, 196, 22, 3])
        # output_pose: # torch.Size([1, 196, 21, 3])
        data = self.do_inv_norm(data)
        data = data.permute(0,2,3,1).contiguous()
        joints_new = joints.permute(0,3,1,2).contiguous()
        output_pose = reverse_pose(data, 22, joints_new)
        # (196,3)
        local_pose_this = output_pose[0,:,joint_idx-1,:]
        return local_pose_this




    def get_head_height_ref_batch_gt(self, pred_res, model_kwargs):
        res=[]
        length_list=model_kwargs["y"]["lengths"]
        bs = pred_res.shape[0]
        for i in range(bs):
            length = length_list[i]
            sample_each = self.get_head_height_ref_gt(pred_res[i:i+1], length)
            res.append(sample_each)
        res = th.cat(res,0)
        assert res.shape==pred_res.shape 
        return res 


    def get_head_height_ref_gt(self, pred_res, length):
        # loss_0 
        # loss_0 = self.loss_head_my(pred_res, length)
        # print(f"loss_0(head)={loss_0.item():.4f}")

        joint_idx = 15
        joint_idx_params_list = self.parse_joint_idx_list(joint_idx)
        
        pred_res_ret = pred_res.detach().clone()
        seq_len = pred_res_ret.shape[-1]

        t_all = length - 1
        t_0 = 0
        t_middle = length//2

        # (seqlen,3)
        fill_data = np.zeros((seq_len,3)).astype(np.float32)
        projected_curve = th.FloatTensor(fill_data)

        projected_curve[t_0, 1]      = self.h_gt_list[0]
        projected_curve[t_middle, 1] = self.h_gt_list[1]
        projected_curve[t_all, 1]    = self.h_gt_list[2]

        # (196,3)
        local_pose_this = self.reverse_to_local_pose_wrapper(pred_res_ret, projected_curve, joint_idx=joint_idx)
        # ->[1, 3, 1, 196]
        pred_res_inv = self.do_inv_norm(pred_res_ret)
        # torch.Size([1, 263, 1, 196])
        
        pred_res_inv[:,joint_idx_params_list,:,:] = local_pose_this.t()[None,:,None,:]
        pred_res_recovered = self.do_norm(pred_res_inv)
        
        return pred_res_recovered

