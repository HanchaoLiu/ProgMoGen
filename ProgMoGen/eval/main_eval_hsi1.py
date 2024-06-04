import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import numpy as np 
from IPython import embed 
import argparse

from metrics2 import get_skate_stat, get_jittor_stat

import torch 

from config_data import EVAL_HSI1_FILE_NAME


def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', default='', help='input_path')
    return parser.parse_args()


def read_all_sample(npy_file_name):
    data_npy = np.load(npy_file_name, allow_pickle=True).item()

    sample_list = data_npy["motion"]
    length_list = data_npy["lengths"]
    return sample_list, length_list


def get_y(sample, t, tag):
    if tag=="head":
        return sample[0,15,1,t]
    elif tag=="foot":
        # return (sample[0,10,1,t]+sample[0,11,1,t])/2.0
        return min(sample[0,10,1,t],sample[0,11,1,t])
    else:
        raise ValueError()


def get_head_height_loss(npy_file_name, constraint):

    sample_list, length_list = read_all_sample(npy_file_name)
    print(sample_list.shape, len(length_list))

    bs = len(length_list)
    res=[]
    head_all = []
    for i in range(bs):
        # (bs, 22, 3, 196)
        sample = sample_list[i:i+1]
        length = length_list[i]
        t_st = 0
        t_mid = length//2
        t_ed = length-1

        y0_head = get_y(sample, t_st,  'head')
        y1_head = get_y(sample, t_mid, 'head')
        y2_head = get_y(sample, t_ed,  'head')

        head_all.append( [y0_head, y1_head, y2_head] )

    head_all = np.array(head_all)
    # print(constraint.shape, head_all.shape)

    rmse = np.sqrt( ((head_all - constraint)**2).mean() )
    rmae = np.abs(head_all-constraint).mean()
    # print(f"rmse = {rmse:.4f}, mae = {rmae:.6f}")
    print(f"constraint_error(mae) = {rmae:.6f}")

    

    threshold = 0.05 
    err = np.abs(head_all - constraint)
    # considered sucess if error for each frame falls within the threshold.
    success_idx_list = np.all(err < threshold,axis=1)
    success_error_rate = 1-success_idx_list.mean()
    print(f"unsuccess_rate = {success_error_rate:.4f}")

    
def main():
    
    args = get_parser()
    file_name = args.input_path
    
    input_constraint_file = EVAL_HSI1_FILE_NAME

    print("="*80)
    print("->",file_name)
    # get_contact_stat(input_joint_reg)
    get_skate_stat(file_name)
    get_jittor_stat(file_name, order=2, stat_type="max")

    constraint = np.load(input_constraint_file, allow_pickle=True) #.item()['constraint'] #[:64]
    constraint = np.array([each[3] for each in constraint])
    get_head_height_loss(file_name, constraint)




if __name__ == "__main__":
    main()
