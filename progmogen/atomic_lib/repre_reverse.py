import os,sys 
from os.path import join as pjoin
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from data_loaders.humanml.common.skeleton import Skeleton
import numpy as np
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.utils.paramUtil import *

from data_loaders.humanml.utils.utils import *
from data_loaders.humanml.scripts.motion_process import recover_from_ric

import torch
from tqdm import tqdm

from config_data import MODEL_PATH, ABS_BASE_PATH, ROOT_DIR, SKEL_JOINTS_TEMPLATE_FILE_NAME



####################################################################
# transform and geometric loss
####################################################################

class DataTransform(object):
    def __init__(self, device='cpu') -> None:
        self.load_inv_normalization_data(device)

        n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        kinematic_chain = t2m_kinematic_chain
        self.skeleton = Skeleton(n_raw_offsets, kinematic_chain, device)

        # self.skel_joints_template = self.load_frame0_template(device)

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


    def joints_to_sample(self, joints_raw):
        # args torch.tensor
        # # torch.Size([1, 22, 3, 196])
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        # seqlen -> seqlen-1
        joints = joints_raw.clone()
        assert joints.shape[0]==1
        joints = joints[0]
        joints = joints.permute(2,0,1).contiguous()

        # (195, 263)
        pred_estimate = joints2sample(joints.numpy())
        pred_estimate = torch.FloatTensor(pred_estimate)

        # torch.Size([195, 263])
        pred_estimate = pred_estimate.permute(1,0).contiguous()
        # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
        pred_estimate = pred_estimate[None,:,None,:]
        
        pred_estimate = self.do_norm(pred_estimate)
        return pred_estimate



# joints2sample 
def joints2sample(positions):
    # receive parameters: tvc
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22

    feet_thre=0.002

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]


    '''New ground truth positions'''
    global_positions = positions.copy()


    """ Get Foot Contacts """
    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        # feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        # feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(float)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)


    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(dataset.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    # return data, global_positions, positions, l_velocity
    return data 



def joints_to_sample_cpu(joints_raw):
    # args torch.tensor
    # # torch.Size([1, 22, 3, 196])
    # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
    # seqlen -> seqlen-1
    joints = joints_raw.clone()
    assert joints.shape[0]==1
    joints = joints[0]
    joints = joints.permute(2,0,1).contiguous()

    # (195, 263)
    pred_estimate = joints2sample(joints.numpy())
    pred_estimate = torch.FloatTensor(pred_estimate)

    # torch.Size([195, 263])
    pred_estimate = pred_estimate.permute(1,0).contiguous()
    # shape = (args.batch_size, model.njoints, model.nfeats, max_frames)
    pred_estimate = pred_estimate[None,:,None,:]

    # fix
    pred_estimate = repair_dst(pred_estimate)


    # pred_estimate = self.do_norm(pred_estimate)
    return pred_estimate



def repair_dst(pred_estimate_torch):
    # TODO: r_velocity (first dimension of 263) not continuous over time.
    # change sign if necessary
    # only support cases, a1,a2,a3.  a1 and a3 are correct, and a2 is incorrect.
    device = pred_estimate_torch.device 
    label_changed=False

    check_thresh=4

    pred_estimate = pred_estimate_torch.data.cpu().numpy()
    assert pred_estimate.shape[0]==1
    length = pred_estimate.shape[3]
    for t in range(2,length-2):
        a_pred = pred_estimate[0,0,0,t-1].item()
        a      = pred_estimate[0,0,0,t].item()
        a_succ = pred_estimate[0,0,0,t+1].item()

        a_changed = -a
        if a_pred*a_succ>0 and abs(a_pred-a_succ)<check_thresh:
            if abs(a_pred-a_changed) + abs(a_succ-a_changed) < abs(a_pred-a) + abs(a_succ-a):
                if False:
                    # print(f"START{t}:abs({a_pred}-{a_changed}) + abs({a_succ}-{a_changed}) < abs({a_pred}-{a}) + abs({a_succ}-{a})")
                    print(f"START{t}: [{a_pred},{a},{a_succ}] ->  [{a_pred},{a_changed},{a_succ}]")
                    print(f"END{t}:so change {pred_estimate[0,0,0,t]} to {a_changed}")
                pred_estimate[0,0,0,t] = a_changed
                label_changed=True 

    pred_estimate_torch_ret = torch.FloatTensor(pred_estimate).to(device)
    return pred_estimate_torch_ret





'''
NOTE: because recover_from_ric, recover_root_pos always recover root trajectory starting at XZ origin (0,0)
so we have to add another offset parameter 263+2=265 so that the generated motion can start at any position.
'''
def main():
    pass


if __name__ == "__main__":
    main()






'''
For Text2Motion Dataset
'''
'''
if __name__ == "__main__":
    example_id = "000021"
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22
    # ds_num = 8
    data_dir = '../dataset/pose_data_raw/joints/'
    save_dir1 = '../dataset/pose_data_raw/new_joints/'
    save_dir2 = '../dataset/pose_data_raw/new_joint_vecs/'

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Get offsets of target skeleton
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    # print(tgt_offsets)

    source_list = os.listdir(data_dir)
    frame_num = 0
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        try:
            dataset, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
            rec_ric_data = recover_from_ric(torch.from_numpy(dataset).unsqueeze(0).float(), joints_num)
            np.save(pjoin(save_dir1, source_file), rec_ric_data.squeeze().numpy())
            np.save(pjoin(save_dir2, source_file), dataset)
            frame_num += dataset.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 20 / 60))
'''

"""
kit
    example_id = "03950_gt"
    # Lower legs
    l_idx1, l_idx2 = 17, 18
    # Right/Left foot
    fid_r, fid_l = [14, 15], [19, 20]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [11, 16, 5, 8]
    # l_hip, r_hip
    r_hip, l_hip = 11, 16
    joints_num = 21
    # ds_num = 8
    data_dir = '../dataset/kit_mocap_dataset/joints/'
    save_dir1 = '../dataset/kit_mocap_dataset/new_joints/'
    save_dir2 = '../dataset/kit_mocap_dataset/new_joint_vecs/'

    n_raw_offsets = torch.from_numpy(kit_raw_offsets)
    kinematic_chain = kit_kinematic_chain

    '''Get offsets of target skeleton'''
    example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    # print(tgt_offsets)

    source_list = os.listdir(data_dir)
    frame_num = 0
    '''Read source dataset'''
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num]
        try:
            name = ''.join(source_file[:-7].split('_')) + '.npy'
            data, ground_positions, positions, l_velocity = process_file(source_data, 0.05)
            rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
            if np.isnan(rec_ric_data.numpy()).any():
                print(source_file)
                continue
            np.save(pjoin(save_dir1, name), rec_ric_data.squeeze().numpy())
            np.save(pjoin(save_dir2, name), data)
            frame_num += data.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 12.5 / 60))
"""