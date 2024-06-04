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




def read_loss(npy_file_name):
    data_npy = np.load(npy_file_name, allow_pickle=True).item()
    # print(data_npy.keys())
    sample_list = data_npy["motion"]
    loss_list   = data_npy["loss"]
    loss = loss_list.mean()
    # print("loss_list.shape = ", loss_list.shape)
    # print(f"sample_list.shape = {sample_list.shape}")
    print(f"constraint_error = {loss:.5f}")
    return loss_list



def main():

    args = get_parser()
    file_name = args.input_path
    print("="*80)
    print("->",file_name)
    # get_contact_stat(file_name)
    get_skate_stat(file_name)
    get_jittor_stat(file_name, order=2, stat_type="max")

    read_loss(file_name)





if __name__ == "__main__":
    main()
