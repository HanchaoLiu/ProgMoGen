import os,sys
import numpy as np 
import trimesh 
from IPython import embed 
from trimesh import Trimesh 


class Saver(object):
    def __init__(self, x) -> None:
        self.vertices = x['vertices']
        self.faces = x['faces'] 
    
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



def main():
    input_name = sys.argv[1]
    output_name = sys.argv[2]

    x=np.load(input_name, allow_pickle=True).item()
    # dict_keys(['motion', 'thetas', 'root_translation', 'faces', 'vertices', 'text', 'length'])

    vert = x['vertices']
    face = x['faces']

    saver = Saver(x)
    frame_i = 50
    save_path = output_name.replace(".obj", f"_frame{frame_i}.obj")
    print("save to ", save_path)
    saver.save_obj(save_path, frame_i)






if __name__ == "__main__":
    main()