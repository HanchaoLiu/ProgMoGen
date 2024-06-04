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


def load_data_from_npy_file_for_real(file_name):
    x = np.load(file_name, allow_pickle=True).item()
    # print(x.keys())

    vertices =x["vertices"]
    joints = x["joints"]
    motion = x["motion"]

    vertices = np.transpose(vertices, [2,0,1])
    joints = np.transpose(joints, [2,0,1])
    motion = np.transpose(motion, [2,0,1])

    smpl_pred_root_traj = joints[:,0,:]
    original_root_traj = motion[:,-1,:3]

    vertices_new = vertices - smpl_pred_root_traj[:,None,:] + original_root_traj[:,None,:]

    vertices = vertices_new
    vertices = vertices[:,:,[0,2,1]]
    # vertices[:,:,0] *= -1

    return vertices, original_root_traj


def load_data_from_npy_file_joint(file_name, idx):
    x = np.load(file_name, allow_pickle=True).item()
    
    vertices=x["motion"]
    vertices = vertices[idx]
    vertices=np.transpose(vertices, [2,0,1])

    length = x['lengths'][idx]

    vertices = vertices[:length]
    vertices = vertices[:,:,[0,2,1]]
    # vertices[:,:,0] *= -1
    # vertices[:,:,1] *= -1
    return vertices


def do_reverse(data, data_joint):
    data[:,:,1] *= -1
    data_joint[:,:,1] *= -1 

    return data, data_joint


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


def set_plane_params():
    # NOTE: write 4 points on the desired plane here.
    
    plane = [[-0.9649365283548832, 0.0, -0.14025388658046722], 
             [-0.9649365283548832, 2.0, -0.14025388658046722], 
             [-2.138232469558716, 2.0, 4.552929878234863], 
             [-2.138232469558716, 0.0, 4.552929878234863]]
    plane = np.array(plane)
    return plane


def reorder_plane_pts(x):
    x = x[:,[0,2,1]]
    x[:,1]*=-1
    return x 


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

    path_joint = cfg.npy_joint
    path_joint_idx = cfg.npy_joint_idx

    from temos.render.blender import render_plane
    from temos.render.video import Video
    import numpy as np

    init = True
    for path in paths:
        try:
            data, original_root_traj = load_data_from_npy_file_for_real(path)
            data_joint = load_data_from_npy_file_joint(path_joint, int(path_joint_idx))
            # data, data_joint = magnify_root_traj(data,data_joint, original_root_traj)
            data, data_joint = do_reverse(data, data_joint)

            # # plane parameters are hard-coded here.
            plane_4points = set_plane_params()
            plane_4points = reorder_plane_pts(plane_4points)
            print('plane_4joints=\n',plane_4points)
            print('data=',data.shape, 'data_joint=', data_joint.shape)
        except FileNotFoundError:
            logger.info(f"{path} not found")
            continue

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

        out = render_plane(data, frames_folder,
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
                     gt=cfg.gt,
                     add_geometry=plane_4points)

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
