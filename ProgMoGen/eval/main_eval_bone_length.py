import os,sys
import numpy as np 
from IPython import embed 
import argparse



def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', default='', help='input_path')
    return parser.parse_args()



def load_motion(file_name):
    '''
    (544, 22, 3, 196)
    '''
    x = np.load(file_name, allow_pickle=True).item()["motion"]
    lengths = np.load(file_name, allow_pickle=True).item()["lengths"]
    return x, lengths 

def get_bone_list():
    bone_list = [
        (21,19),(19,17),(17,14),(14,9),
        (9,13),(13,16),(16,18),(18,20),
        (15,12),(12,9),(9,6),(6,3),(3,0),
        (0,1),(1,4),(4,7),(7,10),
        (0,2),(2,5),(5,8),(8,11)
    ]
    return bone_list

def get_bone_length_each(x, bone_list):
    '''
    x: (22, 3, length)
    bone_list: (n_bones, 2)
    return (196, 21)
    '''
    bone_list = np.array(bone_list)
    bone_idx_1 = bone_list[:,0]
    bone_idx_2 = bone_list[:,1]

    lengths = x.shape[2]
    res = []
    for i in range(lengths):
        skeleton = x[:,:,i]
        joint1 = skeleton[bone_idx_1,:]
        joint2 = skeleton[bone_idx_2,:]
        joint_d = np.linalg.norm(joint1-joint2, axis=1)
        # print(joint_d.shape)
        # print(joint_d)
        res.append(joint_d)
    res = np.stack(res,0)
    return res 
    

def get_bone_length_all(x_batch, bone_list):
    '''
    x: (22, 3, length)
    bone_list: (n_bones, 2)
    return (bs, 196, 21)
    '''
    res_all = []
    for x in x_batch:
        res = get_bone_length_each(x, bone_list)
        res_all.append(res)
    res_all = np.stack(res_all,0)
    return res_all


def fetch_keyframes(motions, lengths):
    '''
    (544, 22, 3, 196)
    '''
    bs = motions.shape[0]
    motions_new = []
    for i in range(bs):
        length = lengths[i]
        t0 = 0 
        t1 = length//2 
        t2 = length - 1
        x = motions[i:i+1, :, :, [t0,t1,t2]]
        motions_new.append(x)
    motions_new = np.concatenate(motions_new, 0)
    return motions_new


def calc_valid_ratio(x_bone, bone_idx, bone_length, threshold):
    assert x_bone.shape[0]==1
    x_bone = x_bone[0]
    # (n,)
    x_bone = x_bone[:, bone_idx]

    upper = bone_length + threshold
    lower = bone_length - threshold
    s = np.logical_and(x_bone < upper, x_bone > lower)
    ratio = 1 - s.mean()
    return ratio


def main():

    args = get_parser()
    file_name = args.input_path

    bone_list = get_bone_list()

    motions, lengths = load_motion(file_name)

    # motions.shape =  (544, 22, 3, 196)
    print("motions.shape = ", motions.shape)

    motions = fetch_keyframes(motions, lengths)
    print("new motions.shape = ", motions.shape)
    
    x_bone = get_bone_length_all(motions, bone_list)

    # (bs, length, 21)
    print("x_bone.shape=", x_bone.shape)
    x_bone = x_bone.reshape((1, x_bone.shape[0]*x_bone.shape[1], x_bone.shape[2]  ))

    print("x_bone.shape=", x_bone.shape)
    print('-'*80)
    print('ratio of invalid bones:')
    # x_bone.shape= (1, 1632, 21)
    ratio = calc_valid_ratio(x_bone, bone_idx=8, bone_length=0.08, threshold=0.025)
    print(f"invalid ratio = {ratio:.4f}")
    








if __name__ == "__main__":
    main()