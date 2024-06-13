
import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../task_configs_eval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../task_configs_eval_others"))


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

from config_data import EVAL_SAMPLE32_FILE_NAME, EVAL_HOI1_FILE_NAME

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
    parser.add_argument("--eval_task", default='none', type=str, help="task configs")

    parser.add_argument("--diffusion_type", default='none', type=str, help="ddim, ddim_inpaint")
    return parser



####################################################################
# evaluation
####################################################################


def eval_stat_each(groundtruth_loader, eval_wrapper, device):
    print('========== Evaluating FID ==========')
    gt_motion_embeddings = []
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            # _, _, _, sent_lens, motions, m_lens, _, _ = batch
            motions , m_lens = batch
            motions = motions.to(device)
            m_lens  = m_lens.to(device)
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)
    return gt_mu, gt_cov
    


def evaluate_matching_score_my(eval_wrapper, gen_loader):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})

    
    motion_loaders = {"gen_loader": gen_loader}

    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        # print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        # print(line, file=file, flush=True)
    # return match_score_dict, R_precision_dict, activation_dict
    return R_precision_dict



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



def get_motion_embeddings(groundtruth_loader, eval_wrapper, device):
    print('========== Evaluating Motion embeddings ==========')
    gt_motion_embeddings = []
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            # _, _, _, sent_lens, motions, m_lens, _, _ = batch
            motions , m_lens = batch
            motions = motions.to(device)
            m_lens  = m_lens.to(device)
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    # gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)
    # return gt_mu, gt_cov
    return gt_motion_embeddings



def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)




####################################################################
# evaluation dataloader
####################################################################

class Motion_and_length_dataset(torch.utils.data.Dataset):
    def __init__(self, motion_list, length_list):
        self.motion_list = motion_list 
        self.length_list = length_list 

    def __getitem__(self, idx):
        return self.motion_list[idx], self.length_list[idx]
    
    def __len__(self):
        return len(self.motion_list)
    

class Motion_all_dataset(torch.utils.data.Dataset):

    def __init__(self, motion_list, length_list, caption_list, tokens_list, cap_len_list):
        self.motion_list = motion_list 
        self.length_list = length_list 
        self.caption_list = caption_list
        self.tokens_list = tokens_list
        self.cap_len_list = cap_len_list

        abs_base_path = ABS_BASE_PATH
        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
    
    def __len__(self):
        return len(self.motion_list)

    def __getitem__(self, item):
        # data = self.generated_motion[item]
        # motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        # sent_len = data['cap_len']

        motion = self.motion_list[item]
        m_length = self.length_list[item]
        caption = self.caption_list[item]
        tokens = self.tokens_list[item]
        sent_len = self.cap_len_list[item]

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # if m_length < self.opt.max_motion_length:
        # max_motion_length = 196
        # if m_length < max_motion_length:
        #     motion = np.concatenate([motion,
        #                              np.zeros((max_motion_length - m_length, motion.shape[1]))
        #                              ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []



####################################################################
# get gt / generated motions
####################################################################

def get_gt_motion(args, dataloader, num_samples_limit):
    real_num_batches = len(dataloader)
    if num_samples_limit is not None:
        real_num_batches = num_samples_limit // dataloader.batch_size + 1
    print('real_num_batches', real_num_batches)

    generated_motion = []
    m_lens_list = []
    text_list = []

    # for idx, batch in enumerate(groundtruth_loader):
    #         _, _, _, sent_lens, motions, m_lens, _, _ = batch
    #         motion_embeddings = eval_wrapper.get_motion_embeddings(
    #             motions=motions,
    #             m_lens=m_lens
    #         )
    
    # for _ in range(real_num_batches//len(dataloader)):
    # if True:
    for _ in range(1):
        # for i, batch in tqdm(enumerate(dataloader)):
        for i, batch in enumerate(dataloader):
            # _, _, _, sent_lens, motions, m_lens, _, _ = batch
            _, _, text_tuple, sent_lens, motions, m_lens, _, _ = batch
            
            motion = motions
            # print("len(generated_motion)=", len(generated_motion))
            if num_samples_limit is not None and len(generated_motion) >= real_num_batches:
                break
            generated_motion.append(motion.data.cpu().detach())
            m_lens_list.append(m_lens.data.cpu().detach())
            text_list += list(text_tuple)
    generated_motion = torch.cat(generated_motion, 0)
    m_lens_list = torch.cat(m_lens_list, 0)
    assert len(m_lens_list)==len(text_list)
    return generated_motion, m_lens_list, text_list


def get_gen_motion(args, model, dataloader, num_samples_limit, scale, init_motion_type):

    clip_denoised = False  # FIXME - hardcoded
    # self.max_motion_length = max_motion_length
    # sample_fn = (
    #     diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
    # )
    real_num_batches = len(dataloader)
    if num_samples_limit is not None:
        real_num_batches = num_samples_limit // dataloader.batch_size + 1
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


    # for _ in range(real_num_batches//len(dataloader)):
    for _ in range(1):
        for i, (motion, model_kwargs) in enumerate(dataloader):

            # print("len(generated_motion)=", len(generated_motion))

            if num_samples_limit is not None and len(generated_motion) >= real_num_batches:
                break
            

            # eval_task="geo1"
            # eval_task="hoi1"
            eval_task = args.eval_task
            assert eval_task in ["hoi1", "geo1"]

            if eval_task=="geo1":
                ref_n32_data = np.load(EVAL_SAMPLE32_FILE_NAME, allow_pickle=True)
                ref_text_prompt_list = [each_sample[0] for each_sample in ref_n32_data]
                ref_tokens_list      = [each_sample[1] for each_sample in ref_n32_data]
                ref_length_list      = [int(each_sample[2]) for each_sample in ref_n32_data]

                model_kwargs['y']['text']    = ref_text_prompt_list
                model_kwargs['y']['tokens']  = ref_tokens_list
                model_kwargs['y']['lengths'] = torch.LongTensor(ref_length_list)

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

            elif eval_task=="hoi1":
                ref_n32_data = np.load(EVAL_HOI1_FILE_NAME, allow_pickle=True)
                ref_text_prompt_list = [each_sample[0] for each_sample in ref_n32_data]
                ref_tokens_list      = [each_sample[1] for each_sample in ref_n32_data]
                ref_length_list      = [int(each_sample[2]) for each_sample in ref_n32_data]

                model_kwargs['y']['text']    = ref_text_prompt_list
                model_kwargs['y']['tokens']  = ref_tokens_list
                model_kwargs['y']['lengths'] = torch.LongTensor(ref_length_list)

                # no tokens
                tokens = [[] for t in model_kwargs['y']['tokens']]
            else:
                raise ValueError()
            

            if eval_task=="hoi1":
                # (6,)
                target_list_ref      = [each_sample[3] for each_sample in ref_n32_data]
                target_list_ref = np.array(target_list_ref)
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
                
                # only for geo1
                if eval_task=="geo1":
                    f_sample_random_plane = import_class(f"{args.task_config}.f_sample_random_plane")
                    diffusion.f_sample_random_plane = types.MethodType(f_sample_random_plane, diffusion)
                
                # sample_fn = diffusion.ddim_sample_loop_opt_fn_goal_relaxed
                ddim_sample_loop_opt_fn_goal_relaxed = import_class(f"{args.task_config}.ddim_sample_loop_opt_fn_goal_relaxed")
                diffusion.ddim_sample_loop_opt_fn_goal_relaxed = types.MethodType(ddim_sample_loop_opt_fn_goal_relaxed, diffusion)
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

                    if eval_task=="hoi1":
                        # get target 
                        target_gt_list = model_kwargs_each['y']['target_list']
                        target_gt = torch.FloatTensor(target_gt_list).to(dist_util.dev())
                        diffusion.target_gt = target_gt
                    elif eval_task=="geo1":
                        # only for geo1
                        target_gt_list = diffusion.f_sample_random_plane(r_range=3, seed=diffusion.np_seed)
                        target_gt = torch.FloatTensor(target_gt_list).to(dist_util.dev())
                        diffusion.target_gt = target_gt
                    else:
                        raise ValueError()

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

                    print("-->loss_ret_val each = ", diffusion.loss_ret_val.mean())

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


def save_to_npy(out_path, all_motions, all_text, all_lengths, fid):
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
             'fid': np.array([fid])})
    

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

    gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, load_mode='gt')
    gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, load_mode='eval')
    num_actions = gen_loader.dataset.num_actions



    logger.log("Creating model and diffusion...")
    # from diffusion.ddim_relax import InpaintingGaussianDiffusion
    assert args.diffusion_type in ["ddim", "ddim_inpaint"]
    if args.diffusion_type == "ddim":
        from diffusion.ddim_relax import InpaintingGaussianDiffusion
    elif args.diffusion_type == "ddim_inpaint":
        from diffusion.ddim_relax_inpaint import InpaintingGaussianDiffusion
    else:
        raise ValueError()
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


        # change shape for generated motions.
        motion_gen = motion_gen.squeeze(2).permute(0,2,1).contiguous()
        print('motion_gen = ', motion_gen.shape, length_gen.shape)
        motion_gen = renorm(motion_gen, gen_loader.dataset)


        # get gt motions
        # motion_gt, length_gt, texts_gt = get_gt_motion(args, gt_loader, num_samples_limit)
        # print('motion_gt = ', motion_gt.shape, 'length_gt = ', length_gt.shape)

        
        # save result
        os.makedirs(args.save_fig_dir, exist_ok=True)
        do_evaluation = False 
        if not do_evaluation:
            # save_to_npy_with_motion_gen
            save_npy_path = os.path.join(args.save_fig_dir, "gen.npy")
            fid = None
            save_to_npy_with_motion_gen(save_npy_path, 
                        all_motions=motion_gen_joints_copy.data.cpu().numpy(), 
                        all_text=list(texts_gen),
                        all_lengths = length_gen.data.cpu().numpy(),
                        fid = fid,
                        motion_gen = motion_gen.data.cpu().numpy(),
                        loss = loss_head_gen, 
                        constraint = constraint_gen 
            )
            
        
