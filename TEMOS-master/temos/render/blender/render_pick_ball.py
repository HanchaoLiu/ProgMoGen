import bpy
import os
import sys
import numpy as np
import math

from .scene import setup_scene  # noqa
from .floor import show_traj, plot_floor, get_trajectory, plot_floor_for_pick_ball
from .vertices import prepare_vertices
from .tools import load_numpy_vertices_into_blender, delete_objs, mesh_detect
from .camera import Camera
from .sampler import get_frameidx

from .materials import body_material


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


def set_camera():
    camera = bpy.data.objects['Camera']
    camera.location.y = -7.5


def render_pick_ball(npydata, frames_folder, *, mode, faces_path, gt=False,
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
        # perc = 0.2
        perc=0.0
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

    # Create a floor
    plot_floor_for_pick_ball(data.data, big_plane=False)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh)

    # set_camera()
    frameidx = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)
    
    # hold_ball_massctr_sequence = False
    # if hold_ball_massctr_sequence:
    #     frameidx = [0,12,24,36,48,59]
    # frameidx = [0,40,80,120,160]

    nframes_to_render = len(frameidx)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_obj_names = []

    for index, frameidx in enumerate(frameidx):

        # every frame of the video requires add the wall plane.
        # nothing here
        print(f"--> rendering [{index}/{nframes_to_render}]")

        if mode == "sequence":
            frac = index / (nframes_to_render-1)
            mat = data.get_sequence_mat(frac)
        else:
            mat = data.mat
            camera.update(data.get_root(frameidx))

        islast = index == (nframes_to_render-1)
        
        print('mat = ', type(mat), mat)

        objname = data.load_in_blender(frameidx, mat)
        name = f"{str(index).zfill(4)}"

        # draw ball in the hand.
        joints = add_geometry
        joints_this_frame = joints[frameidx,:,:]
        left_hand = joints_this_frame[21]
        right_hand = joints_this_frame[20]
        print("hand coord = ", left_hand, right_hand)
        sphere_center = right_hand
        print("sphere_center = ", sphere_center, frameidx)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.05, location=(sphere_center[0], sphere_center[1], sphere_center[2]))   
        print(bpy.data.objects.keys())
        for k in bpy.data.objects.keys():
            if k.startswith("Icosphere"):
                # obj = bpy.data.objects["Icosphere"]
                print("ball obj name = ", k)
                obj = bpy.data.objects[k]
                obj.active_material = body_material(*(1.0, 0.5, 0.5, 0.3))

        # draw start and end goals
        point_a = [0.0, -0.2, 0.5]
        sphere_center = point_a
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=0.02, location=(sphere_center[0], sphere_center[1], sphere_center[2]))   

        point_b = [2.0, -0.2, 0.5]
        sphere_center = point_b
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=0.02, location=(sphere_center[0], sphere_center[1], sphere_center[2]))   
        
        print(bpy.data.objects.keys())
        for k in bpy.data.objects.keys():
            # if k.startswith("Icosphere"):
            if k in ["Icosphere.001", "Icosphere.002"]:
                # obj = bpy.data.objects["Icosphere"]
                print("ball obj name for point a and b = ", k)
                obj = bpy.data.objects[k]
                import matplotlib
                cmap = matplotlib.cm.get_cmap('Oranges')
                rgbcolor = cmap(0.5)
                obj.active_material = body_material(*(rgbcolor[0], rgbcolor[1], rgbcolor[2], 1.0))


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
                print("delete obj = ", objname)
                delete_objs(objname)

                ball_obj_list = [k for k in bpy.data.objects.keys() if k.startswith("Icosphere")]
                delete_objs(ball_obj_list)
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
