import numpy as np

from .materials import body_material

# green
# GT_SMPL = body_material(0.009, 0.214, 0.029)
GT_SMPL = body_material(0.035, 0.415, 0.122)

# blue
# GEN_SMPL = body_material(0.022, 0.129, 0.439)
# Blues => cmap(0.87)
GEN_SMPL = body_material(0.035, 0.322, 0.615)


class Meshes:
    def __init__(self, data, *, gt, mode, faces_path, canonicalize, always_on_floor, oldrender=True, **kwargs):
        data = prepare_meshes(data, canonicalize=canonicalize, always_on_floor=always_on_floor)

        self.faces = np.load(faces_path)
        self.data = data
        self.mode = mode
        self.oldrender = oldrender

        self.N = len(data)
        self.trajectory = data[:, :, [0, 1]].mean(1)

        if gt:
            self.mat = GT_SMPL
        else:
            self.mat = GEN_SMPL

    def get_sequence_mat(self, frac):
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Blues')
        # begin = 0.60
        # end = 0.90
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end-begin)*frac)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        return mat


    def get_sequence_mat_alpha(self, frac, alpha):
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Blues')
        # begin = 0.60
        # end = 0.90
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end-begin)*frac)
        from .materials import body_material_transparent_with_alpha
        list_all = rgbcolor + (alpha,)
        mat = body_material_transparent_with_alpha(*list_all)
        return mat
    
    def get_sequence_mat_orange(self, frac):
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Oranges')
        # begin = 0.60
        # end = 0.90
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end-begin)*frac)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        return mat

    def get_root(self, index):
        return self.data[index].mean(0)

    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index, mat):
        vertices = self.data[index]
        faces = self.faces
        name = f"{str(index).zfill(4)}"

        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)

        return name


    def load_in_blender_mesh(self, vertices, faces, mat):
        # vertices = self.data[index]
        # faces = self.faces
        # name = f"{str(index).zfill(4)}"
        name = "table"

        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)

        return name


    def __len__(self):
        return self.N


def prepare_meshes(data, canonicalize=True, always_on_floor=False):
    if canonicalize:
        print("No canonicalization for now")

    # fix axis
    # data[..., 1] = - data[..., 1]
    # data[..., 0] = - data[..., 0]

    if False:
        # Remove the floor
        data[..., 2] -= data[..., 2].min()

        # Put all the body on the floor
        if always_on_floor:
            data[..., 2] -= data[..., 2].min(1)[:, None]

    return data
