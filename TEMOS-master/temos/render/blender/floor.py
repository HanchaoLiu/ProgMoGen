import bpy
from .materials import floor_mat


def get_trajectory(data, is_mesh):
    if is_mesh:
        # mean of the vertices
        trajectory = data[:, :, [0, 1]].mean(1)
    else:
        # get the root joint
        trajectory = data[:, 0, [0, 1]]
    return trajectory


def show_traj(coords):
    # create the Curve Datablock
    curveData = bpy.data.curves.new('myCurve', type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 2

    # map coords to spline
    polyline = curveData.splines.new('POLY')
    polyline.points.add(len(coords)-1)
    for i, coord in enumerate(coords):
        x, y = coord
        polyline.points[i].co = (x, y, 0.001, 1)

    # create Object
    curveOB = bpy.data.objects.new('myCurve', curveData)
    curveData.bevel_depth = 0.01

    bpy.context.collection.objects.link(curveOB)


def show_traj_3D(coords):
    # create the Curve Datablock
    curveData = bpy.data.curves.new('myCurve', type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 2

    # map coords to spline
    polyline = curveData.splines.new('POLY')
    polyline.points.add(len(coords)-1)
    for i, coord in enumerate(coords):
        x, y, z = coord
        polyline.points[i].co = (x, y, z, 1)

    # create Object
    curveOB = bpy.data.objects.new('myCurve', curveData)
    # curveData.bevel_depth = 0.01
    curveData.bevel_depth = 0.03

    bpy.context.collection.objects.link(curveOB)
    

def plot_floor(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    scale = ((maxx - minx)/2, (maxy - miny)/2, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))


def plot_floor_for_plane(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    location = ((maxx + minx)/2, (maxy + miny)/2, 0.10)
    scale = ((maxx - minx)/2*4.0, (maxy - miny)/2*1.5, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))



def plot_floor_for_line_relax(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    location = ((maxx + minx)/2, (maxy + miny)/2, -0.1)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    scale = ((maxx - minx)/2*4.0, (maxy - miny)/2*1.5, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))


def plot_floor_for_velocity(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    location = ((maxx + minx)/2, (maxy + miny)/2, 0.06)
    # location = ((maxx + minx)/2, (maxy + miny)/2, 0.0)

    scale = ((maxx - minx)/2*1.5, (maxy - miny)/2*1.5, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
        # obj.active_material = floor_mat(color=(0.15, 0.15, 0.15, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))




def plot_floor_for_or(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    location = ((maxx + minx)/2, (maxy + miny)/2, 0.0)

    scale = ((maxx - minx)/2*2, (maxy - miny)/2*1.5, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
        # obj.active_material = floor_mat(color=(0.15, 0.15, 0.15, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))


def plot_floor_for_pick_ball(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    scale = ((maxx - minx)/2*2, (maxy - miny)/2*4, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))



def plot_floor_for_hold_ball(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0
    # minz = 0.05

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    location = ((maxx + minx)/2, (maxy + miny)/2, 0.05)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    scale = ((maxx - minx)/2*4, (maxy - miny)/2*1.5, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))




def plot_floor_for_head_hand_contact(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    location = ((maxx + minx)/2, (maxy + miny)/2, 0.05)
    # scale = ((maxx - minx)/2*5, (maxy - miny)/2*3, 1)
    scale = ((maxx - minx)/2*3, (maxy - miny)/2*3, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))




def plot_floor_for_square_space(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    # scale = ((maxx - minx)/2*2.0, (maxy - miny)/2, 1)

    location = (0.0, 0.0, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    # scale = (1.0, 1.0, 1)
    # scale = (1.5, 1.5, 1)
    scale = (2.0, 2.0, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))




def plot_floor_for_overhead_barrier(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    scale = ((maxx - minx)/2*2.0, (maxy - miny)/2, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))




def plot_floor_for_narrow_gap(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    scale = ((maxx - minx)/2*12.0, (maxy - miny)/2*2.5, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))




def plot_floor_for_center_of_mass(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    location = ((maxx + minx)/2, (maxy + miny)/2, 0-0.1)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    scale = ((maxx - minx)/2*1.5, (maxy - miny)/2, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))








def plot_floor_my(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    scale = ((maxx - minx)/2*1.5, (maxy - miny)/2, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))














def plot_floor_my_for_confined_space(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    # scale = ((maxx - minx)/2*2.0, (maxy - miny)/2, 1)

    location = (0.0, 0.0, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    scale = (1.0, 1.0, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))







def plot_floor_my_for_footline(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    scale = ((maxx - minx)/2*4.0, (maxy - miny)/2*1.5, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))














def plot_floor_my_for_headheight(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    # location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    scale = ((maxx - minx)/2*4, (maxy - miny)/2*2, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        # obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))



