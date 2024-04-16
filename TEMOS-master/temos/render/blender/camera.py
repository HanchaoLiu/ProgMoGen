import bpy


class Camera:
    def __init__(self, *, first_root, mode, is_mesh):
        camera = bpy.data.objects['Camera']

        ## initial position
        camera.location.x = 7.36
        camera.location.y = -6.93
        if is_mesh:
            # camera.location.z = 5.45
            camera.location.z = 5.6
        else:
            camera.location.z = 5.2

        # here add new 
        if False:
            
            if False:
                # view from front 
                # left/right
                camera.location.x = 1
                # front 
                camera.location.y = -7
                # height
                camera.location.z = 2.2
                print(camera.rotation_euler)
                # camera.rotation_euler.x=63.59/180*3.14
                # up,down view
                camera.rotation_euler.x=80/180*3.14
                camera.rotation_euler.y=0.0
                # camera.rotation_euler.z=46.7/180*3.14
                # left/right view
                camera.rotation_euler.z=10/180*3.14

            if True:
                # view from side 
                # left/right
                camera.location.x = 10
                # front 
                camera.location.y = -0
                # height
                camera.location.z = 4.2
                print(camera.rotation_euler)
                # camera.rotation_euler.x=63.59/180*3.14
                # up,down view
                camera.rotation_euler.x=70/180*3.14
                camera.rotation_euler.y=0.0
                # camera.rotation_euler.z=46.7/180*3.14
                # left/right view
                camera.rotation_euler.z=90/180*3.14

        # wider point of view
        if mode == "sequence":
            if is_mesh:
                # camera.data.lens = 100
                camera.data.lens = 55
            else:
                camera.data.lens = 85
        elif mode == "frame":
            if is_mesh:
                camera.data.lens = 130
            else:
                camera.data.lens = 140
        elif mode == "video":
            if is_mesh:
                camera.data.lens = 110
            else:
                camera.data.lens = 140

        # camera.location.x += 0.75

        self.mode = mode
        self.camera = camera

        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]

        self._root = first_root

    def update(self, newroot):
        delta_root = newroot - self._root

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]

        self._root = newroot
