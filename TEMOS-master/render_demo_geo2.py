import os
import sys

import numpy as np 
print("np.version=", np.__version__)

try:
    import bpy
    sys.path.append(os.path.dirname(bpy.data.filepath))
except ImportError:
    raise ImportError("Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender.")

import temos.launch.blender
import temos.launch.prepare  # noqa
import logging
import hydra
from omegaconf import DictConfig



logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="render")
def _render_cli(cfg: DictConfig):
    return render_cli(cfg)


def extend_paths(path, keyids, *, onesample=True, number_of_samples=1):
    if not onesample:
        template_path = str(path / "KEYID_INDEX.npy")
        paths = [template_path.replace("INDEX", str(index)) for index in range(number_of_samples)]
    else:
        paths = [str(path / "KEYID.npy")]

    all_paths = []
    for path in paths:
        all_paths.extend([path.replace("KEYID", keyid) for keyid in keyids])
    return all_paths


def load_data_from_npy_file(file_name):
    x = np.load(file_name, allow_pickle=True).item()
    # print(x.keys())

    # motion = x['motion']
    # print(motion.shape)
    # print(x["vertices"].shape)

    vertices=x["vertices"]
    # (t,v,c)
    vertices=np.transpose(vertices, [2,0,1])

    # (3, 96) -> (96,3)
    root_trans = x["root_translation"].T
    # root_trans[:,1]=0.0
    vertices = vertices - root_trans[:,None,:]
    

    # vertices = vertices.numpy()
    print('vertices.shape = ', vertices.shape)
    vertices = vertices[:,:,[0,2,1]]
    # vertices[:,:,0] *= -1
    # vertices[:,:,1] *= -1 

    root_trans = root_trans[:,[0,2,1]]
    return vertices, root_trans


def load_data_from_npy_file_for_real(file_name):
    x = np.load(file_name, allow_pickle=True).item()
    # print(x.keys())

    # motion = x['motion']
    # print(motion.shape)
    # print(x["vertices"].shape)
    vertices =x["vertices"]
    joints = x["joints"]
    motion = x["motion"]

    vertices = np.transpose(vertices, [2,0,1])
    joints = np.transpose(joints, [2,0,1])
    motion = np.transpose(motion, [2,0,1])

    # print("->")
    # print(vertices.shape)
    # print(joints.shape)
    # print(motion.shape)

    # vert_mean = vertices.mean(axis=1)
    # joint_mean = joints.mean(axis=1)

    # np.set_printoptions(3,suppress=True)
    # print("->vert")
    # print(vert_mean)
    # print("->joint")
    # print(joint_mean)

    smpl_pred_root_traj = joints[:,0,:]
    original_root_traj = motion[:,-1,:3]

    vertices_new = vertices - smpl_pred_root_traj[:,None,:] + original_root_traj[:,None,:]
    # vertices_new = vertices - smpl_pred_root_traj[:,None,:] #+ original_root_traj[:,None,:]
    # vertices_new = joints - smpl_pred_root_traj[:,None,:] + original_root_traj[:,None,:]

    # print(smpl_pred_root_traj)
    # print(original_root_traj)
    vertices = vertices_new
    vertices = vertices[:,:,[0,2,1]]
    # vertices[:,:,0] *= -1

    return vertices, original_root_traj



def magnify_root_traj_only(data, original_root_traj):

    # data[:,:,0] *= -1
    # data_joint[:,:,0] *= -1 

    original_root_traj = original_root_traj[:,[0,2,1]]
    # set y to zero.
    original_root_traj[:,2]=0.0
    data = data + original_root_traj[:,None,:]
    # data_joint = data_joint + original_root_traj[:,None,:]

    # data[:,:,1] *= -1
    # data_joint[:,:,1] *= -1 

    return data


def magnify_root_traj(data,data_joint, original_root_traj):

    # data[:,:,0] *= -1
    # data_joint[:,:,0] *= -1 

    original_root_traj = original_root_traj[:,[0,2,1]]
    # set y to zero.
    original_root_traj[:,2]=0.0
    data = data + original_root_traj[:,None,:]
    data_joint = data_joint + original_root_traj[:,None,:]

    # data[:,:,1] *= -1
    # data_joint[:,:,1] *= -1 

    return data, data_joint



def do_reverse(data, data_joint):
    data[:,:,1] *= -1
    data_joint[:,:,1] *= -1 

    return data, data_joint


def load_data_from_npy_file_joint(file_name, idx):
    x = np.load(file_name, allow_pickle=True).item()
    # print(x.keys())

    # motion = x['motion']
    # print(motion.shape)
    # print(x["vertices"].shape)

    vertices=x["motion"]
    # print("motion.shape=", vertices.shape)
    vertices = vertices[idx]
    # print("motion.shape=", vertices.shape)
    vertices=np.transpose(vertices, [2,0,1])

    length = x['lengths'][idx]

    vertices = vertices[:length]
    # vertices = vertices.numpy()
    # print('vertices.shape = ', vertices.shape)
    vertices = vertices[:,:,[0,2,1]]
    # vertices[:,:,0] *= -1
    # vertices[:,:,1] *= -1
    return vertices


def joint_result_merge_with_root_trans(joint, root_trans, data):
    '''
    joint: (t,v,c)
    root_trans: (t,c)
    '''

    # set y to 0.0
    # root_trans[:,2]=0.0

    joint[:,:,1] *= -1
    joint = joint + root_trans[:,None,:]
    # joint = joint - data[...,2].min()

    data_2 = data.copy()
    data_2[:,:,1] *= -1 
    joint = joint - data_2[...,1].min()

    joint[:,:,1] *= -1
    return joint 



def prepare_meshes_for_npy(data, canonicalize=True, always_on_floor=False):
    if canonicalize:
        print("No canonicalization for now")

    # fix axis
    data[..., 1] = - data[..., 1]
    data[..., 0] = - data[..., 0]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data





def render_cli(cfg: DictConfig) -> None:
    if cfg.npy is None:
        if cfg.folder is None or cfg.split is None:
            raise ValueError("You should either use npy=XXX.npy, or folder=XXX and split=XXX")
        # only them can be rendered for now
        if not cfg.infolder:
            jointstype = cfg.jointstype
            assert ("mmm" in jointstype) or jointstype == "vertices"

        from temos.data.utils import get_split_keyids
        from pathlib import Path
        from evaluate import get_samples_folder
        from sample import cfg_mean_nsamples_resolution, get_path
        keyids = get_split_keyids(path=Path(cfg.path.datasets)/ "kit-splits", split=cfg.split)

        onesample = cfg_mean_nsamples_resolution(cfg)
        if not cfg.infolder:
            model_samples, amass, jointstype = get_samples_folder(cfg.folder,
                                                                  jointstype=cfg.jointstype)
            path = get_path(model_samples, amass, cfg.gender, cfg.split, onesample, cfg.mean, cfg.fact)
        else:
            path = Path(cfg.folder)

        paths = extend_paths(path, keyids, onesample=onesample, number_of_samples=cfg.number_of_samples)
    else:
        paths = [cfg.npy]

    # path_joint = cfg.npy_joint
    # path_joint_idx = cfg.npy_joint_idx

    # from temos.render.blender import render
    # from temos.render.blender import render_base
    # from temos.render.blender import render_base_rawheight
    # from temos.render.blender import render_base_rawheight_mp4
    # from temos.render.blender import render_base_rawheight_footline
    from temos.render.blender import render_base_rawheight_footline_2
    from temos.render.video import Video
    import numpy as np

    path_joint = cfg.npy_joint
    path_joint_idx = cfg.npy_joint_idx


    init = True
    for path in paths:
        try:
            # data = np.load(path)
            # data = load_data_from_npy_file(path)
            data, original_root_traj = load_data_from_npy_file_for_real(path)
            # data_joint = load_data_from_npy_file_joint(path_joint, int(path_joint_idx))
            # data, data_joint = magnify_root_traj(data,data_joint, original_root_traj)
            data_joint = load_data_from_npy_file_joint(path_joint, int(path_joint_idx))

            # magnify traj
            # data = magnify_root_traj_only(data, original_root_traj)

            data[:,:,1] *= -1

            data_joint[:,:,1] *= -1

            print('data=',data.shape)

            # (192,22,3)
            # print("data_joint.shape = ", data_joint.shape)
            # print(data_joint[:,10,:])
            # print(data_joint[:,11,:])
            

            # exit(0)
            # data_joint = joint_result_merge_with_root_trans(data_joint, root_trans, data)

        except FileNotFoundError:
            logger.info(f"{path} not found")
            continue

        # cfg.mode="video"
        # cfg.mode="sequence"
        # cfg.denoising=False
        cfg.denoising=True
        cfg.downsample=True

        if cfg.mode == "video":
            frames_folder = path.replace(".npy", "_frames")
        else:
            frames_folder = path.replace(".npy", ".png")

        if cfg.mode == "video":
            cfg.vid_ext="mp4"
            vid_path = path.replace(".npy", f".{cfg.vid_ext}")
            if os.path.exists(vid_path):
                continue

        out = render_base_rawheight_footline_2(data, frames_folder,
                     denoising=cfg.denoising,
                     oldrender=cfg.oldrender,
                     res=cfg.res,
                     canonicalize=cfg.canonicalize,
                     exact_frame=cfg.exact_frame,
                     num=cfg.num, mode=cfg.mode,
                     faces_path=cfg.faces_path,
                     downsample=cfg.downsample,
                     always_on_floor=cfg.always_on_floor,
                     init=init,
                     gt=cfg.gt, add_geometry=data_joint)

        init = False

        if cfg.mode == "video":
            if cfg.downsample:
                video = Video(frames_folder, fps=12.5, res=cfg.res)
            else:
                video = Video(frames_folder, fps=100.0, res=cfg.res)

            video.save(out_path=vid_path)
            logger.info(vid_path)

        else:
            logger.info(f"Frame generated at: {out}")



if __name__ == '__main__':
    _render_cli()
