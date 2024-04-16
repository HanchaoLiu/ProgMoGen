import os,sys
import numpy as np 
import torch 
from metrics import calculate_skating_ratio, calculate_foot_contact_ratio

def get_motion(input_joint):
    joint_data = np.load(input_joint, allow_pickle=True).item()
    # print(joint_data.keys())
    lengths = joint_data["lengths"][0]
    motion = joint_data["motion"][:,:,:,:lengths]
    return motion


def get_contact_stat(input_joint_reg):
    
    x = np.load(input_joint_reg, allow_pickle=True).item()["motion"]
    lengths = np.load(input_joint_reg, allow_pickle=True).item()["lengths"]

    bs = len(x)
    x = torch.FloatTensor(x)
    res=[]
    for i in range(bs):
        skate_ratio = calculate_foot_contact_ratio(x[i:i+1,:,:,:lengths[i]])
        # print(skate_ratio)
        # print(skate_ratio.shape)
        res.append(skate_ratio.item())
    res = np.array(res).round(4)
    # print("foot contact ratio = ", res.mean(), res)

    print("foot contact ratio = ", res.mean())


def get_skate_stat(input_joint_reg):
    
    x = np.load(input_joint_reg, allow_pickle=True).item()["motion"]
    lengths = np.load(input_joint_reg, allow_pickle=True).item()["lengths"]

    bs = len(x)
    x = torch.FloatTensor(x)
    res=[]
    for i in range(bs):
        skate_ratio, _ = calculate_skating_ratio(x[i:i+1,:,:,:lengths[i]])
        # print(skate_ratio)
        # print(skate_ratio.shape)
        res.append(skate_ratio.item())
    res = np.array(res).round(4)
    # print("skate ratio = ", res.mean(), res)
    print("skate ratio = ", res.mean())


def get_jittor_stat(npy_file_name, order=1, stat_type="mean"):
    # print("="*80)
    # print(npy_file_name)
    assert stat_type in ["mean", "max"]
    sample_list, length_list = read_all_sample(npy_file_name)
    if stat_type=="mean":
        x = get_jittor_mean(sample_list, length_list, "all_frame", order=order)
    elif stat_type=="max":
        x = get_jittor_max(sample_list, length_list, "all_sample", order=order)
    else:
        raise ValueError()
    x = np.round(np.array(x),4)
    print(f"jittor_{stat_type} [order={order}]= {x:.4f}")


def read_all_sample(npy_file_name):
    data_npy = np.load(npy_file_name, allow_pickle=True).item()

    sample_list = data_npy["motion"]
    length_list = data_npy["lengths"]
    return sample_list, length_list


def get_order_stat(x, order=None):
    '''
    (t,v,c)
    '''
    
    a = np.diff(x, n=order, axis=0)
    a_val = np.linalg.norm(a, axis=2).reshape(-1)
    return a_val


def get_jittor_mean(motion_list, length_list, ret_type, order):
    bs = len(motion_list)
    res = []
    for i in range(bs):
        length = length_list[i]
        # (1,22,3,t)
        motion = motion_list[i,:,:,:length]
        # (t,v,c)
        motion = np.transpose(motion, [2,0,1])
        
        jittor_val = get_order_stat(motion,order=order)
        res.append(jittor_val)

    val_res=[]
    if ret_type=="all_sample":
        for a in res:
            val = np.sqrt((a**2).mean())
            val_res.append(val)
            ret_val = np.mean(val_res)
    elif ret_type=="all_frame":
        val_res = np.concatenate(res,0)
        ret_val = np.sqrt( (val_res**2).mean() )
    else:
        raise ValueError()

    return ret_val


def get_jittor_max(motion_list, length_list, ret_type, order):
    bs = len(motion_list)
    res = []
    for i in range(bs):
        length = length_list[i]
        # (1,22,3,t)
        motion = motion_list[i,:,:,:length]
        # (t,v,c)
        motion = np.transpose(motion, [2,0,1])
        
        jittor_val = get_order_stat(motion,order=order)
        res.append(jittor_val)

    val_res=[]
    if ret_type=="all_sample":
        for a in res:
            # val = np.sqrt((a**2).mean())
            # print("a=", a.shape)
            val = np.abs(a).max()
            val_res.append(val)
            ret_val = np.mean(val_res)
        # exit(0)
    # elif ret_type=="all_frame":
    #     val_res = np.concatenate(res,0)
    #     ret_val = np.sqrt( (val_res**2).mean() )
    else:
        raise ValueError()

    return ret_val

