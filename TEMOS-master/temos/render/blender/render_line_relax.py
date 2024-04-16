import bpy
import os
import sys
import numpy as np
import math

from .scene import setup_scene  # noqa
from .floor import show_traj, plot_floor, get_trajectory, plot_floor_for_line_relax
from .vertices import prepare_vertices
from .tools import load_numpy_vertices_into_blender, delete_objs, mesh_detect
from .camera import Camera
# from .camera_plane import Camera
from .sampler import get_frameidx

from .materials import body_material
from .floor import show_traj_3D

def prune_begin_end(data, perc):
    to_remove = int(len(data)*perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    # add resolution 
    # 1280, 1024
    bpy.context.scene.render.resolution_x = 320 
    bpy.context.scene.render.resolution_y = 256
    print("resolution = ", bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y) 
    bpy.ops.render.render(use_viewport=True, write_still=True)


def render_current_frame_lowres(path):
    bpy.context.scene.render.filepath = path
    # add resolution 
    # 1280, 1024
    bpy.context.scene.render.resolution_x = 320 
    bpy.context.scene.render.resolution_y = 256
    print("resolution = ", bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y) 
    bpy.ops.render.render(use_viewport=True, write_still=True)


def render_current_frame_highres(path):
    bpy.context.scene.render.filepath = path
    # add resolution 
    # 1280, 1024
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 1024
    print("resolution = ", bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y) 
    bpy.ops.render.render(use_viewport=True, write_still=True)


def cylinder_between(x1, y1, z1, x2, y2, z2, r):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1    
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    bpy.ops.mesh.primitive_cylinder_add(
        radius = r, 
        depth = dist,
        location = (dx/2 + x1, dy/2 + y1, dz/2 + z1)   
    ) 

    phi = math.atan2(dy, dx) 
    theta = math.acos(dz/dist) 

    bpy.context.object.rotation_euler[1] = theta 
    bpy.context.object.rotation_euler[2] = phi 
    bpy.context.object.active_material = body_material(*(1.0, 0.5, 0.5, 0.3))


def add_cube(x1, y1, z1, x2, y2, z2):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1    
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(dx/2 + x1, dy/2 + y1, dz/2 + z1), scale=(0.05, dist, 0.02))

    phi = math.atan2(dz, dy)
    theta = -math.asin(dx/dist) 
    
    bpy.context.object.rotation_mode='ZYX'
    bpy.context.object.rotation_euler[0] = phi
    bpy.context.object.rotation_euler[1] = 0
    bpy.context.object.rotation_euler[2] = theta
    bpy.context.object.active_material = body_material(*(1.0, 0.5, 0.5, 0.3))
    


def fit_3d_line(data):
    '''
    data: (seqlen,3)
    return: line_params (1,6) [0:3]=p, [3:6]=n
    '''
    datamean = data.mean(axis=0) 
    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)
    vec_d = vv[0]
    # assert np.allclose(th.norm(vec_d).item(),1.0,atol=1e-4)
    # print(datamean.shape, vec_d.shape)
    line_params = np.concatenate([datamean, vec_d], 0).reshape(1,-1)
    return line_params


def plot_footline(data_joint):
    
    length=70
    left_foot = data_joint[:length,10,:]
    right_foot = data_joint[:length,11,:]
    foot_traj = np.concatenate([left_foot, right_foot],0)

    line_params = fit_3d_line(foot_traj)
    p = line_params[0,:3]
    d = line_params[0,3:]
    d = d / np.linalg.norm(d)

    p1 = p + d*5 
    p2 = p - d*6

    point_st = p1 
    point_ed = p2 
    
    print('point_st = ', point_st)
    add_cube(x1=point_st[0], y1=point_st[1], z1=point_st[2], x2=point_ed[0], y2=point_ed[1], z2=point_ed[2])

    point_st = p + d*1.8
    cylinder_between(x1=point_st[0], y1=point_st[1], z1=0.0, x2=point_st[0], y2=point_st[1], z2=point_st[2], r=0.025)
    point_ed = p - d*2.6
    cylinder_between(x1=point_ed[0], y1=point_ed[1], z1=0.0, x2=point_ed[0], y2=point_ed[1], z2=point_ed[2]-0.03, r=0.025)


def plot_line_relax(data_joint):

    length=70
    left_foot = data_joint[:length,10,:]
    right_foot = data_joint[:length,11,:]
    foot_traj = np.concatenate([left_foot, right_foot],0)

    # line_params = fit_3d_line(foot_traj)
    # line_params = np.array([[0.5, -0.4, 0.3, 0.2, -1, 0.1]])

    if True:
        # task -> blender order = [0,-2,1]
        line_params_task = [[0.5 , 0.1 , 0.4], [0.4, 0.0, 1.0]]
        p_task, d_task = line_params_task
        p_task_blender = [p_task[0], -p_task[2], p_task[1]]
        d_task_blender = [d_task[0], -d_task[2], d_task[1]]
        line_params_blender = p_task_blender + d_task_blender
        line_params = np.array([line_params_blender])

    print("---->line_params = ", line_params)

    p = line_params[0,:3]
    d = line_params[0,3:]
    d = d / np.linalg.norm(d)

    # define two points.
    # point_st = p + d*0.5
    point_st = p + d*1.5
    point_ed = p - d*5.5
    print('point_st = ', point_st)
    print('point_ed = ', point_ed)

    cylinder_between(x1=point_st[0], y1=point_st[1], z1=-0.1, x2=point_st[0], y2=point_st[1], z2=point_st[2], r=0.025)
    cylinder_between(x1=point_ed[0], y1=point_ed[1], z1=-0.1, x2=point_ed[0], y2=point_ed[1], z2=point_ed[2], r=0.025)

    if False:
        line_seg = np.stack([point_st, point_ed], 0)
        show_traj_3D(line_seg)
    
    add_cube(x1=point_st[0], y1=point_st[1], z1=point_st[2], x2=point_ed[0], y2=point_ed[1], z2=point_ed[2])
    

def render_line_relax(npydata, frames_folder, *, mode, faces_path, gt=False,
           exact_frame=None, num=8, downsample=True,
           canonicalize=True, always_on_floor=False, denoising=True,
           oldrender=True,
           res="high", init=True, add_geometry=None):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res, denoising=denoising, oldrender=oldrender)

    is_mesh = mesh_detect(npydata)

    # Put everything in this folder
    if mode == "video":
        if always_on_floor:
            frames_folder += "_of"
        os.makedirs(frames_folder, exist_ok=True)
        # if it is a mesh, it is already downsampled
        if downsample and not is_mesh:
            npydata = npydata[::8]
    elif mode == "sequence":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        # img_path = f"{img_name}{ext}"
        img_path = f"{img_name}{ext}"

    elif mode == "frame":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        # img_path = f"{img_name}_{exact_frame}{ext}"
        img_path = f"{img_name}_{exact_frame}{ext}"

    # remove X% of begining and end
    # as it is almost always static
    # in this part
    if mode == "sequence":
        perc = 0.0
        # perc=0.0
        npydata = prune_begin_end(npydata, perc)

    if is_mesh:
        # from .meshes import Meshes
        from .meshes_noshift_y import Meshes
        data = Meshes(npydata, gt=gt, mode=mode,
                      faces_path=faces_path,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor)
        
    else:
        from .joints import Joints
        data = Joints(npydata, gt=gt, mode=mode,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor)

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    # show_traj(data.trajectory)

    data_joint = add_geometry

    # plot_footline(data_joint)
    plot_line_relax(data_joint)

    # Create a floor
    plot_floor_for_line_relax(data.data, big_plane=False)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh)

    frameidx = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidx)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_obj_names = []

    print("frameidx=", frameidx)
    # frameidx=[0, 27, 55, 82, 109, 136, 164, 191]

    # render_seq=False 
    if mode == "sequence":
        # frameidx=[a*13 for a in range(8)]
        frameidx=[a*13 for a in range(6)]
        frameidx[-2]=52-4
        nframes_to_render = len(frameidx)

    for index, frameidx in enumerate(frameidx):


        print(f"--> rendering [{index}/{nframes_to_render}]")

        if mode == "sequence":
            frac = index / (nframes_to_render-1)
            mat = data.get_sequence_mat(frac)
        else:
            mat = data.mat
            camera.update(data.get_root(frameidx))

        islast = index == (nframes_to_render-1)

        objname = data.load_in_blender(frameidx, mat)
        name = f"{str(index).zfill(4)}"

        if mode == "video":
            path = os.path.join(frames_folder, f"frame_{name}.png")
        else:
            path = img_path

        if mode == "sequence":
            imported_obj_names.append(objname)
        elif mode == "frame":
            camera.update(data.get_root(frameidx))

        # if mode != "sequence" or islast:
        #     render_current_frame(path)
        #     delete_objs(objname)

        if mode != "sequence" or islast:
            if mode=="sequence":
                render_current_frame_highres(path)
                delete_objs(objname)
            elif mode=="video":
                render_current_frame_lowres(path)
                # render_current_frame_highres(path)
                delete_objs(objname)
            else:
                raise ValueError()

    # bpy.ops.wm.save_as_mainfile(filepath="/Users/mathis/TEMOS_github/male_line_test.blend")
    # exit()

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    if mode == "video":
        return frames_folder
    else:
        return img_path
