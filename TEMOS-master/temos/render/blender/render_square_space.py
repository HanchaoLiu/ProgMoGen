import bpy
import os
import sys
import numpy as np
import math

from .scene import setup_scene  # noqa
from .floor import show_traj, plot_floor, get_trajectory, plot_floor_for_square_space
from .vertices import prepare_vertices
from .tools import load_numpy_vertices_into_blender, delete_objs, mesh_detect
from .camera import Camera
# from .camera_plane import Camera
from .sampler import get_frameidx

from .materials import body_material, body_material_transparent, body_material_transparent_with_alpha

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


def set_camera():
    camera = bpy.data.objects['Camera']
    # camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh)

    # set camera
    # camera.location.x = 10
    # # front 
    # camera.location.y = -6
    # # height
    # camera.location.z = 4.2
    # print(camera.rotation_euler)
    # # camera.rotation_euler.x=63.59/180*3.14
    # # up,down view
    # camera.rotation_euler.x=70/180*3.14
    # camera.rotation_euler.y=0.0
    # # camera.rotation_euler.z=46.7/180*3.14
    # # left/right view
    # camera.rotation_euler.z=60/180*3.14
    camera.data.lens = 120


def render_square_space(npydata, frames_folder, *, mode, faces_path, gt=False,
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

    # set_camera()
    # Show the trajectory
    # show_traj(data.trajectory)

    # y1=-2
    # z_height=1.3
    z_height=0.8
    
    # Create a floor
    plot_floor_for_square_space(data.data, big_plane=False)

    print(bpy.data.objects.keys())

    # plot four walls 
    z_height = 0.8
    z_height_2 = 0.4

    # (x -1)
    location = (-1,0,z_height)
    scale = (1, 1, z_height)
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1,1,1), rotation=(0.0, np.pi/2, 0.0))
    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "Plane_1"
    obj.data.name = "Plane_1"
    obj.active_material = body_material(*(1.0, 0.5, 0.5, 0.3))

    # (x 1)
    location = (1,0,z_height_2)
    scale = (1, 1, z_height_2)
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1,1,1), rotation=(0.0, np.pi/2, 0.0))
    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                            constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                            proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                            use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "Plane_2"
    obj.data.name = "Plane_2"
    obj.active_material = body_material_transparent_with_alpha(*(1.0, 0.5, 0.5, 0.3))
    

    # (y -1)
    location = (0,-1,z_height_2)
    scale = (1, 1, z_height_2)
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1,1,1), rotation=(np.pi/2, 0.0, 0.0))
    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                            constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                            proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                            use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "Plane_3"
    obj.data.name = "Plane_3"
    obj.active_material = body_material_transparent_with_alpha(*(1.0, 0.5, 0.5, 0.3))


    # (y 1)
    location = (0,1,z_height)
    scale = (1, 1, z_height)
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1,1,1), rotation=(np.pi/2, 0.0, 0.0))
    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                            constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                            proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                            use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "Plane_4"
    obj.data.name = "Plane_4"
    obj.active_material = body_material(*(1.0, 0.5, 0.5, 0.3))
    

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



    for index, frameidx in enumerate(frameidx):


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
