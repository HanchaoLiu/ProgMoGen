import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

import argparse
import os
# from visualize import vis_utils
from visualize import vis_utils_v2_hand
import shutil
from tqdm import tqdm

# input_path: *.npy 
# output_dir: mkdir name, {.npy, obj_dir}
# save all the .obj files to obj_dir
import os,sys
import numpy as np 
# from trimesh import Trimesh
from IPython import embed 

import time 

class Saver(object):
    def __init__(self, x) -> None:
        self.vertices = x['vertices']
        self.faces = x['faces'] 
        self.real_num_frames = self.vertices.shape[-1]
        print(f"vert = {self.vertices.shape}, faces = {self.faces.shape}, real_num_frames = {self.real_num_frames}")
    
    def get_vertices(self, frame_i):
        # return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()
        return self.vertices[:, :, frame_i].squeeze().tolist()

    def get_trimesh(self, frame_i):
        return Trimesh(vertices=self.get_vertices(frame_i),
                        faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    parser.add_argument("--bs", type=int, default=-1, help='')
    parser.add_argument("--selected_idx", type=int, nargs="+", default=-1, help='')
    params = parser.parse_args()

    assert params.input_path.endswith('.npy')
    # parsed_name = os.path.basename(params.input_path).replace('.gif', '').replace('sample', '').replace('rep', '')
    # sample_i, rep_i = [int(e) for e in parsed_name.split('_')]
    # sample_i, rep_i = 0,0 
    # npy_path = os.path.join(os.path.dirname(params.input_path), 'results.npy')
    npy_path = params.input_path
    assert os.path.exists(npy_path)

    

    data = np.load(npy_path,allow_pickle=True).item()["motion"]
    bs = data.shape[0]
    if params.bs!=-1:
        bs = params.bs 
        run_list = list(range(bs))

    selected_idx_list = params.selected_idx 
    # print(selected_idx_list)
    run_list = selected_idx_list
    print("run_list = ", run_list)

    print(f"motion.shape = {data.shape}")
    

    # out_npy_path = params.input_path.replace('.npy', '_smpl_params.npy')
    
    results_dir = params.input_path.replace('.npy', '_smpl')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    rep_i=0
    # for sample_i in range(bs):
    for sample_i in run_list:
        t0=time.time()
        out_npy_path = os.path.join(results_dir, f'gen{sample_i}_smpl_params_fixhand.npy')
        print(sample_i, out_npy_path)
        npy2obj = vis_utils_v2_hand.npy2obj(npy_path, sample_i, rep_i,
                                    device=params.device, cuda=params.cuda)
        print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
        # npy2obj.save_npy_numpy(out_npy_path)
        npy2obj.save_npy_numpy_with_joints(out_npy_path)
        
        t_elapsed=time.time()-t0 
        print(f"running time for sample_{sample_i} = {t_elapsed} s")

    if False:
        print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
        # for frame_i in tqdm(range(npy2obj.real_num_frames)):
        #     npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)
        
        x = np.load(out_npy_path, allow_pickle=True).item()
        saver = Saver(x)
        for frame_i in (range(saver.real_num_frames)):
            if frame_i%5!=0:
                continue
            save_path = os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i))
            print("save to", save_path)
            saver.save_obj(save_path, frame_i)

    