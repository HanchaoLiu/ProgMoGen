import os,sys
import numpy as np 
from IPython import embed 
import argparse

import torch 
import torch.nn.functional as F 

from metrics2 import get_skate_stat, get_jittor_stat



def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', default='', help='input_path')
    return parser.parse_args()


def read_loss_by_length(npy_file_name):
    data_npy = np.load(npy_file_name, allow_pickle=True).item()
    # print(data_npy.keys())
    sample_list = data_npy["motion"]
    loss_list   = data_npy["loss"]
    length_list = data_npy["lengths"]
    # (32, 1, 1, 195)
    loss = loss_list.mean()

    n = len(sample_list)
    res =[]
    for i in range(n):
        loss_each   = loss_list[i,0,0,:]
        length_each = length_list[i]
        res.append( loss_each[:length_each].mean() )
    loss_val_by_length = np.mean(res)

    print(f"loss_val_by_length = {loss_val_by_length:.5f}")



def read_loss(npy_file_name):
    data_npy = np.load(npy_file_name, allow_pickle=True).item()
    # print(data_npy.keys())
    sample_list = data_npy["motion"]
    loss_list   = data_npy["loss"]
    loss = loss_list.mean()
    # print("loss_list.shape = ", loss_list.shape)
    # print(f"sample_list.shape = {sample_list.shape}")
    print(f"constraint_error = {loss:.5f}")

    # print(loss_list.mean(axis=3).reshape(-1))


def get_trajectory_length(file_name):
    data_npy = np.load(file_name, allow_pickle=True).item()
    sample_list = data_npy["motion"]
    # (32, 22, 3, 195)
    print(sample_list.shape)

    bs = sample_list.shape[0]
    res = []
    for i in range(bs):
        # (2,195) -> (195,2)
        x = sample_list[i,0,[0,2],:].T

        # x = x[[0,20,40,60,80,100,120,140,160,180],:]
        x = x[[0,180],:]
        # print(x)

        # (194,2)
        diff = x[1:,:] - x[:-1,:]
        diff = np.linalg.norm(diff, axis=1)
        # (194,)
        traj_len = np.sum(diff)
        res.append(traj_len)
    res = np.array(res)
    traj_len_mean = res.mean()
    print("traj_len_each = ", res)
    print("traj_len_mean = ", traj_len_mean)
    print("short_traj_ratio = ", (res<0.1).mean())



def main():
    args = get_parser()
    file_name = args.input_path
    print("="*80)
    print("->",file_name)
    # get_contact_stat(file_name)
    get_skate_stat(file_name)
    get_jittor_stat(file_name, order=2, stat_type="max")

    read_loss(file_name)

    # get_trajectory_length(file_name)




if __name__ == "__main__":
    main()