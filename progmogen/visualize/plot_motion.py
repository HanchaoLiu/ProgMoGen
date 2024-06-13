import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

import numpy as np 
import argparse

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script_v2 import explicit_plot_3d_motion, explicit_plot_3d_motion_debug



def get_motion(gen_data, idx):
    # dict_keys(['motion', 'text', 'lengths', 'num_samples', 'num_repetitions', 'fid', 'motion_gen', 'loss', 'constraint'])
    # (1, 22, 3, 196)
    motion = gen_data['motion']
    text = gen_data['text']
    lengths = gen_data['lengths']

    motion_each = motion[idx]
    text_each = text[idx]
    length = lengths[idx]

    motion_each = np.transpose(motion_each, [2,0,1])
    motion_each = motion_each[:length,:,:]
    return motion_each, text_each


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='')
    parser.add_argument("--output_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--selected_idx", type=int, required=True, help='')
    params = parser.parse_args()

    input_gen_path = params.input_path
    idx = params.selected_idx
    save_path = params.output_path

    kinematic_tree = paramUtil.t2m_kinematic_chain
    dataset = "humanml"
    fps = 12.5

    # # (seq_len, joints_num, 3)
    gen_data = np.load(input_gen_path, allow_pickle=True).item()
    joints, title = get_motion(gen_data, idx)
    print("joints.shape = ", joints.shape)
    print("title = ", title)
    print("save to", save_path)

    explicit_plot_3d_motion_debug(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=fps, radius=3, vis_mode="default")

    


if __name__ == "__main__":
    main()