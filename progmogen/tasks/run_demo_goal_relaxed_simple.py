
import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../task_configs"))


from diffusion.respace import SpacedDiffusion
from utils.parser_util import evaluation_inpainting_parser, evaluation_inpainting_parser_add_args
from utils.fixseed import fixseed
from datetime import datetime
# from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader, get_mdm_loader_debug  # get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
# from utils.model_util import load_model_blending_and_diffusion
from utils.model_util_v2 import load_model_blending_and_diffusion

from data_loaders.humanml_utils import get_inpainting_mask

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import wrap_model

import os,sys 
import copy
from IPython import embed 

from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.scripts.motion_process import recover_from_rot, recover_from_rot_with_skeleton

from data_loaders.humanml.utils.word_vectorizer import WordVectorizer

from torch.utils.data._utils.collate import default_collate


import types

from config_data import MODEL_PATH, ABS_BASE_PATH, ROOT_DIR, SKEL_JOINTS_TEMPLATE_FILE_NAME

torch.multiprocessing.set_sharing_strategy('file_system')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod



####################################################################
# args
####################################################################

def f_add_args(parser):
    parser.add_argument("--use_ddim_tag", default=0, type=int, help="")
    parser.add_argument("--save_tag", default='', type=str, help="")
    parser.add_argument("--mask_type", default='', type=str, help="")
    parser.add_argument("--save_fig_dir", default='', type=str, help="")
    parser.add_argument("--ret_type", default='', type=str, help="from pos/rot")
    parser.add_argument("--text_split", default='', type=str, help="test text split for evaluation")
    parser.add_argument("--num_samples_limit", default=0, type=int, help="")

    parser.add_argument("--task_config", default='none', type=str, help="task configs")
    return parser



####################################################################
# evaluation
####################################################################

def renorm(data, dataset):
    # gen_loader.dataset.mean_for_eval
    # motion_gt =  torch.Size([32, 196, 263]) 
    mean = gen_loader.dataset.mean[None,None,:]
    std  = gen_loader.dataset.std[None,None,:]
    mean_for_eval = gen_loader.dataset.mean_for_eval[None,None,:]
    std_for_eval  = gen_loader.dataset.std_for_eval[None,None,:]
    data = data * std + mean 
    data = (data - mean_for_eval)/std_for_eval
    return data 


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)





####################################################################
# get gt / generated motions
####################################################################

def get_gen_motion(args, model, dataloader, num_samples_limit, scale, init_motion_type):

    clip_denoised = False  # FIXME - hardcoded
    # self.max_motion_length = max_motion_length
    # sample_fn = (
    #     diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
    # )
    # real_num_batches = len(dataloader)
    # if num_samples_limit is not None:
    #     real_num_batches = num_samples_limit // dataloader.batch_size + 1
    # print('real_num_batches', real_num_batches)
    batch_size = 32
    real_num_batches = num_samples_limit // batch_size + 1
    print('real_num_batches', real_num_batches)

    generated_motion = []
    loss_list = []
    length_list = []
    text_list = []
    constraint_list = []

    caption_list = []
    tokens_list =  []
    cap_len_list = []


    model.eval()

    for v in model.parameters():
        v.requires_grad=False

    # here has a grad!!!
    # with torch.no_grad():

    # for _ in range(real_num_batches//len(dataloader)):
    for _ in range(1):
        # for i, (motion, model_kwargs) in enumerate(dataloader):
        for i in range(1):

            model_kwargs = {'y': {}}
            motion = torch.zeros(batch_size,263,1,196).float()

            # print("len(generated_motion)=", len(generated_motion))

            if num_samples_limit is not None and len(generated_motion) >= real_num_batches:
                break

            


            # text_prompt = import_class(f"{args.task_config}.TEXT_PROMPT")
            # motion_length = import_class(f"{args.task_config}.LENGTH")
            # model_kwargs['y']['text'] = [text_prompt]*32
            # model_kwargs['y']['tokens'] = [None]*32
            # model_kwargs['y']['lengths'] = torch.LongTensor([motion_length]*32)

            # here form of TARGET_LIST = [ [[a1,a2,a3], [b1,b2,b3] ], [] ]
            # TARGET_LIST = [
            #        [[0, 0.8, 0.2], [4.0, 0.5, 0.2]],
            #    ]*4
            # can be: start_position, end_position (HOI-1)
            # line params of point, direction (GEO-2)
            # plane params of point, normal (GEO-1)


            # text_prompt_list = import_class(f"{args.task_config}.TEXT_PROMPT_LIST")
            # motion_length_list = import_class(f"{args.task_config}.LENGTH_LIST")
            # model_kwargs['y']['text']    = text_prompt_list
            # model_kwargs['y']['tokens']  = [None]*32
            # model_kwargs['y']['lengths'] = torch.LongTensor(motion_length_list)

            # target_list_ref = import_class(f"{args.task_config}.TARGET_LIST")
            # target_list_ref = [ np.array(a[0]+a[1]) for a in target_list_ref ]
            # target_list_ref = np.stack(target_list_ref, 0)
            # print(target_list_ref.shape)
            # model_kwargs['y']['target_list'] = target_list_ref

            text_prompt = import_class(f"{args.task_config}.TEXT_PROMPT")
            motion_length = import_class(f"{args.task_config}.LENGTH")
            text_token = import_class(f"{args.task_config}.TEXT_TOKEN")
            
            model_kwargs['y']['text'] = [text_prompt]*32
            model_kwargs['y']['tokens'] = [text_token]*32
            model_kwargs['y']['lengths'] = torch.LongTensor([motion_length]*32)

            tokens = [(t.split('_') if t is not None else [] ) for t in model_kwargs['y']['tokens']]

            # (6,)
            target_list_ref = import_class(f"{args.task_config}.TARGET")
            target_list_ref = [ target_list_ref ]*32
            target_list_ref = np.stack(target_list_ref, 0)
            print(target_list_ref.shape)
            model_kwargs['y']['target_list'] = target_list_ref


            # add CFG scale to batch
            if scale != 1.:
                model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                        device=dist_util.dev()) * scale
            

            

            model_kwargs['y']['inpainted_motion'] = motion.to(dist_util.dev())
            model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, motion.shape)).float().to(dist_util.dev())

            repeat_times=1
            for t in range(repeat_times):

                diffusion.load_inv_normalization_data(dist_util.dev())

                f_loss = import_class(f"{args.task_config}.f_loss")
                f_eval = import_class(f"{args.task_config}.f_eval")
                diffusion.f_loss = types.MethodType(f_loss, diffusion)
                diffusion.f_eval = types.MethodType(f_eval, diffusion)

                update_goal = import_class(f"{args.task_config}.update_goal")
                diffusion.update_goal = types.MethodType(update_goal, diffusion)
                transform_sample = import_class(f"{args.task_config}.transform_sample")
                diffusion.transform_sample = types.MethodType(transform_sample, diffusion)

                get_lr_schedule = import_class(f"{args.task_config}.get_lr_schedule")
                diffusion.get_lr_schedule = types.MethodType(get_lr_schedule, diffusion)
                
                
                sample_fn = diffusion.ddim_sample_loop_opt_fn_goal_relaxed

                sample = []
                loss = []
                constraint = []
                lengths = model_kwargs['y']['lengths']
                bs = motion.shape[0]

                demo_num = import_class(f"{args.task_config}.DEMO_NUM")
                for ii in range(bs):
                    print(ii,bs)

                    diffusion.np_seed = np.random.randint(0,1000)+1
                    model_kwargs_each = get_slice_model_kwargs(model_kwargs, ii)

                    motion_each = model_kwargs_each['y']['inpainted_motion']
                    length_each = model_kwargs_each['y']['lengths'].item()

                    diffusion.length = length_each



                    # diffusion.lr = import_class(f"{args.task_config}.lr")
                    diffusion.iterations = import_class(f"{args.task_config}.iterations")
                    diffusion.epoch_relax = import_class(f"{args.task_config}.epoch_relax")

                    # get target 
                    target_gt_list = model_kwargs_each['y']['target_list']
                    # target_a, target_b = target_gt_list[0][0:3], target_gt_list[0][3:6]
                    # target_gt_list = [torch.FloatTensor(target_a).to(dist_util.dev()),
                    #                     torch.FloatTensor(target_b).to(dist_util.dev())]
                    target_gt = torch.FloatTensor(target_gt_list).to(dist_util.dev())
                    diffusion.target_gt = target_gt

                    sample_each = sample_fn(
                        model,
                        motion[ii:ii+1].shape,
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
                    sample.append(sample_each)
                    loss.append(diffusion.loss_ret_val)

                    constraint_1d=target_gt_list.reshape(-1).tolist()
                    constraint.append(constraint_1d)

                    print("-->loss_ret_val = ", diffusion.loss_ret_val.mean())

                    if (demo_num is not None) and len(sample)>=demo_num:
                        break

                sample = torch.cat(sample, 0)
                # assert sample.shape==motion.shape
                lengths = model_kwargs['y']['lengths']
                texts = model_kwargs['y']['text']
                loss = np.array(loss)
                constraint = np.array(constraint)

                if (demo_num is not None):
                    sample = sample[:demo_num]
                    lengths = lengths[:demo_num]
                    texts = texts[:demo_num]

            generated_motion.append(sample.data.cpu().detach())
            length_list.append( lengths.data.cpu().detach() )
            text_list += texts
            
            # caption = model_kwargs['y']['text']
            # tokens = tokens
            # cap_len = [len(tokens[bs_i]) for bs_i in range(len(tokens))]
            if demo_num is not None:
                caption = model_kwargs['y']['text'][:demo_num]
                tokens = tokens[:demo_num]
                cap_len = [len(tokens[bs_i]) for bs_i in range(len(tokens))][:demo_num]
            else:
                caption = model_kwargs['y']['text']
                tokens = tokens
                cap_len = [len(tokens[bs_i]) for bs_i in range(len(tokens))]

            caption_list += caption
            tokens_list += tokens 
            cap_len_list += cap_len 

            loss_list.append(loss)
            constraint_list.append(constraint)


    generated_motion = torch.cat(generated_motion, 0)
    length_list = torch.cat(length_list, 0)
    print("final len(generated_motion)=", len(generated_motion), generated_motion.shape)
    assert len(length_list) == len(text_list )
    loss_list = np.concatenate(loss_list, 0)
    constraint_list = np.concatenate(constraint_list, 0)
    assert len(loss_list) == len(length_list)
    assert len(constraint_list) == len(length_list)
    # return generated_motion, length_list, text_list 
    
    assert len(caption_list) == len(length_list)
    assert len(tokens_list)  == len(length_list)
    assert len(cap_len_list) == len(length_list)

    # return [generated_motion, loss_list, constraint_list], length_list, text_list 
    return [generated_motion, loss_list, constraint_list], length_list, text_list, [caption_list, tokens_list, cap_len_list] 



def get_slice_model_kwargs(model_kwargs, i):
    res={}
    res['y']={}
    for k,v in model_kwargs['y'].items():
        res['y'][k] = v[i:i+1]
    return res 



####################################################################
# transform and geometric loss
####################################################################

class DataTransform(object):
    def __init__(self, device='cpu') -> None:
        self.load_inv_normalization_data(device)

        n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        kinematic_chain = t2m_kinematic_chain
        self.skeleton = Skeleton(n_raw_offsets, kinematic_chain, device)

        
        self.skel_joints_template = self.load_frame0_template(device)

    def load_frame0_template(self, device):
        
        skel_joints_template_file_name = SKEL_JOINTS_TEMPLATE_FILE_NAME
        x = np.load(skel_joints_template_file_name, allow_pickle=True).item()
        motion = x["motion"][0:1,:,:,0]
        motion = torch.FloatTensor(motion).to(device)
        print("frame0 = ", motion.shape)
        # exit(0)
        return motion

    def load_inv_normalization_data(self, device):
        # model.nfeats = 1
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        root_dir = ROOT_DIR
        mean_file = os.path.join(root_dir, "dataset/HumanML3D/Mean.npy")
        std_file  = os.path.join(root_dir, "dataset/HumanML3D/Std.npy")

        d_mean = np.load(mean_file)
        d_std  = np.load(std_file)
        d_mean = torch.FloatTensor(d_mean).to(device)
        d_std  = torch.FloatTensor(d_std).to(device)
        self.d_mean = d_mean[None,:,None,None]
        self.d_std  = d_std[None,:,None,None]

        # t2m_for_eval 
        mean_file_for_eval = os.path.join(root_dir, "dataset/t2m_mean.npy")
        std_file_for_eval  = os.path.join(root_dir, "dataset/t2m_std.npy")

        d_mean = np.load(mean_file_for_eval)
        d_std  = np.load(std_file_for_eval)
        d_mean = torch.FloatTensor(d_mean).to(device)
        d_std  = torch.FloatTensor(d_std).to(device)
        self.d_mean_for_eval = d_mean[None,:,None,None]
        self.d_std_for_eval  = d_std[None,:,None,None]


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
    

    def do_inv_norm_t2m_eval(self, data):
        # model.nfeats = 1
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)

        assert data.shape[1]==self.d_mean_for_eval.shape[1] and data.shape[2]==self.d_mean_for_eval.shape[2]
        assert data.shape[1]==self.d_std_for_eval.shape[1] and data.shape[2]==self.d_std_for_eval.shape[2]

        return data * self.d_std_for_eval + self.d_mean_for_eval
        

    def do_norm_t2m_eval(self, data):
        assert data.shape[1]==self.d_mean_for_eval.shape[1] and data.shape[2]==self.d_mean_for_eval.shape[2]
        assert data.shape[1]==self.d_std_for_eval.shape[1] and data.shape[2]==self.d_std_for_eval.shape[2]
        return (data - self.d_mean_for_eval) / self.d_std_for_eval

    def transform_motion_gt_to_sample(self, motion_gt):
        '''
        motion: torch.Size([128, 196, 263]) -> (n,t,v), c=1
        to sample: torch.Size([2, 263, 1, 196])
        '''
        # (bs, 196, 263, 1)
        motion_gt = motion_gt[:,:,:,None]
        motion_gt = motion_gt.permute(0,2,3,1).contiguous()
        motion_gt = self.do_inv_norm_t2m_eval(motion_gt)
        motion_gt = self.do_norm(motion_gt)
        return motion_gt


    def sample_to_joints(self, sample):
        # # torch.Size([1, 22, 3, 196])
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        x_pred = sample
        x_pred = self.do_inv_norm(x_pred)
        
        sample = recover_from_ric(x_pred.permute(0,2,3,1).contiguous(), 22)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        # torch.Size([1, 22, 3, 196])
        # assert sample.shape[0]==1
        return sample
    
    def sample_to_joints_from_rot(self, pred):
        bs = pred.shape[0]
        res = []
        for i in range(bs):
            x = self.sample_to_joints_from_rot_each(pred[i:i+1])
            res.append(x)
        res = torch.cat(res,0)
        return res 
    
    def sample_to_joints_from_rot_each(self, pred):
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        # skel_joints: (bs, 22, 3)
        x_pred = pred
        x_pred = self.do_inv_norm(x_pred)
        
        
        skel_joints = self.skel_joints_template
        bs = pred.shape[0]
        new_shape = (bs, skel_joints.shape[1], skel_joints.shape[2])
        skel_joints = skel_joints.expand(new_shape)

        
        sample = recover_from_rot_with_skeleton(x_pred.permute(0,2,3,1).contiguous(), 22, self.skeleton, skel_joints)
        
        sample = sample.view(-1, *sample.shape[:]).permute(0, 2, 3, 1)

        # torch.Size([1, 22, 3, 196])
        assert sample.shape[0]==1
        return sample
    
    def nvct2ntvc(self, joints):
        # torch.Size([1, 22, 3, 196])
        # (seq_len, joints_num, 3)
        joints = joints.permute(0,3,1,2).contiguous()
        return joints

    # relax 
    def get_XZ_offset(self, joints):
        '''
        first frame, joint idx 0, only XZ dim.
        joints : torch.Size([1, 22, 3, 196])
        return : joints_dst_XZ = (1, 2)
        '''
        joints_XZ = joints[:,0,[0,2],0]
        return joints_XZ


    def add_XZ_offset(self, joints0, offset):
        '''
        joints : torch.Size([1, 22, 3, 196])
        offset : joints_dst_XZ = (1, 2)
        return 
            joints: torch.Size([1, 22, 3, 196])
        '''
        joints = joints0.clone()
        joints[:,:,0:1,:] += offset[:,0:1][:,None,:,None]
        joints[:,:,2:3,:] += offset[:,1:2][:,None,:,None]
        return joints


    def sample_to_joints_with_XZ_offset(self, pred):
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        # # torch.Size([1, 22, 3, 196])
        # [1, 263, 1, 195]
        assert pred.shape[1]==265
        x_pred = pred[:,:263, :, :]
        XZ_offset = pred[:,263:, 0, 0]

        x_pred = self.do_inv_norm(x_pred)
        
        # torch.Size([2, 263, 1, 3])
        
        sample = recover_from_ric(x_pred.permute(0,2,3,1).contiguous(), 22)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        sample = self.add_XZ_offset(sample, XZ_offset)

        # torch.Size([1, 22, 3, 196])
        # assert sample.shape[0]==1
        return sample


def get_loss_stat(loss_list):
    loss_mean = loss_list.mean()
    return loss_mean




####################################################################
# save
####################################################################

def save_to_npy_with_motion_gen(out_path, all_motions, all_text, all_lengths, fid, motion_gen, loss,
                                constraint):
    '''
    all in np.ndarray
    all_motions:   # [bs, njoints, 3, seqlen], .e.g (1, 22, 3, 196)
    all_text:     list 
    all_lengths:  np.ndarray of int
    '''
    # npy_path = os.path.join(out_path, 'results.npy')
    npy_path = out_path
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': 1, 'num_repetitions': 1,
             'fid': np.array([fid]),
             'motion_gen': motion_gen,
             'loss': loss,
             'constraint': constraint})

class Gen_loader(object):
    def __init__(self):
        self.dataset = None


if __name__ == '__main__':
    # args_list = evaluation_inpainting_parser()
    args_list = evaluation_inpainting_parser_add_args(f_add_args)
    args = args_list[0]
    fixseed(args.seed)

    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    if args.use_ddim_tag==1:
        use_ddim_tag=True
    elif args.use_ddim_tag==0:
        use_ddim_tag=False 
    else:
        raise ValueError()
    

    # set mask_type 
    mask_type = args.mask_type 
    assert mask_type in ['root_horizontal', 'left_wrist']
    args_list[0].inpainting_mask = mask_type
    args.inpainting_mask         = mask_type

    

    id_str = ''

    log_file = os.path.join(os.path.dirname(args.model_path), f'debug_ddim{int(use_ddim_tag)}_{args.save_tag}_{id_str}'+'_eval_humanml_{}_{}'.format(name, niter))


    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    if args.inpainting_mask != '':
        log_file += f'_mask_{args.inpainting_mask}'
    log_file += f'_{args.eval_mode}'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')
    if os.path.exists(log_file):
        os.remove(log_file)
    assert args.overwrite or not os.path.exists(log_file), "Log file already exists!"

    
    # replication_times = replication_times if args.replication_times is None else args.replication_times


    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = args.text_split

    # gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, load_mode='gt')
    # gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, load_mode='eval')
    # num_actions = gen_loader.dataset.num_actions
    gen_loader = Gen_loader()


    logger.log("Creating model and diffusion...")
    from diffusion.ddim_relax import InpaintingGaussianDiffusion
    DiffusionClass =  InpaintingGaussianDiffusion if args.filter_noise else SpacedDiffusion
    model, diffusion = load_model_blending_and_diffusion(args_list, gen_loader, dist_util.dev(), DiffusionClass=DiffusionClass)

    
    data_transform = DataTransform(device='cpu')


    replication_times=1
    fid_all_list = []
    res_list=[]
    loss_all_list = []
    for ii in range(replication_times):

        num_samples_limit = args.num_samples_limit

        # generate motions
        motion_gen_all, length_gen, texts_gen ,(caption_list, tokens_list, cap_len_list)  = get_gen_motion(args, model, gen_loader, num_samples_limit, args.guidance_param,
                                    init_motion_type=None)
        
        motion_gen, loss_head_gen, constraint_gen = motion_gen_all
        print("constraint_gen.shape = ", constraint_gen.shape)
        
        if args.ret_type=="pos":
            # motion_gen_joints = data_transform.sample_to_joints(motion_gen)
            motion_gen_joints = data_transform.sample_to_joints_with_XZ_offset(motion_gen)
            motion_gen = motion_gen[:,:263,:,:]
        elif args.ret_type=="rot":
            print("from rot!")
            motion_gen_joints = data_transform.sample_to_joints_from_rot(motion_gen)
        else:
            raise ValueError()
        motion_gen_joints_copy = motion_gen_joints.detach().clone()
        print(f"--> motion_gen = {motion_gen.shape}, motion_gen_joints = {motion_gen_joints.shape}")

        
        # save result
        os.makedirs(args.save_fig_dir, exist_ok=True)
        do_evaluation = False 
        if not do_evaluation:
            # save_to_npy_with_motion_gen
            save_npy_path = os.path.join(args.save_fig_dir, "gen.npy")
            fid = None
            motion_gen=None
            save_to_npy_with_motion_gen(save_npy_path, 
                        all_motions=motion_gen_joints_copy.data.cpu().numpy(), 
                        all_text=list(texts_gen),
                        all_lengths = length_gen.data.cpu().numpy(),
                        fid = fid,
                        motion_gen = motion_gen,
                        loss = loss_head_gen, 
                        constraint = constraint_gen 
            )
            
        

