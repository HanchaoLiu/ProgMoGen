import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
import numpy as np
import torch 
from atomic_lib.geometry_utils import axis_angle_to_matrix



#################################################################
# line
#################################################################
def construct_line(p, d):
    p = p.reshape((1,3))
    d = d.reshape((1,3))
    d = d / torch.norm(d.reshape(-1))
    line = torch.cat([p, d], 1)
    return line 


def get_axis_angle_from_two_lines(line1, line2):
    dir1 = line1[:, 3:]
    dir2 = line2[:, 3:]
    axis = torch.cross(dir1, dir2)
    axis = axis / torch.norm(axis, dim=1)
    cos_angle = (dir1*dir2).sum(dim=1) / ( torch.norm(dir1, dim=1) * torch.norm(dir2, dim=1) )
    angle = torch.arccos(cos_angle)
    # print('axis  = ', axis)
    # print('angle = ', angle)
    axis_angle = axis * angle[:,None]
    # print(axis_angle)
    return axis_angle


def get_translation_from_two_lines(line1, line2, rot_mat_1):
    # rot_mat_1=(3,3)
    # should be p2-R*p1
    # return (1,3)
    pts1 = line1[:, :3]
    pts2 = line2[:, :3]
    trans = pts2.t() - torch.matmul(rot_mat_1[0], pts1.t())
    return trans.t()


def calc_RT_from_two_lines(line1, line2):
    # (1,3) -> (1,3,3)
    axis_angle = get_axis_angle_from_two_lines(line1, line2)
    # print("axis_angle = ", axis_angle)

    # rot_mat_1: (1,3,3)
    rot_mat_1 = axis_angle_to_matrix(axis_angle)
    # print("rot_mat_1 = ")
    # print(rot_mat_1)
    # print("det = ", torch.linalg.det(rot_mat_1[0]))

    translation = get_translation_from_two_lines(line1, line2, rot_mat_1)
    # print("translation = ", translation)
    translation = translation.reshape(3,1)

    # return R=(3,3), T=(3,1)
    return rot_mat_1[0], translation


#################################################################
# plane
#################################################################
def construct_plane(p, d):
    return construct_line(p, d)

def calc_RT_from_two_planes(plane1, plane2):
    return calc_RT_from_two_lines(plane1, plane2)


#################################################################
# transform
#################################################################
def apply_RT(point_list, R, T):
    '''
    point_list.shape = (N, 3)
    '''
    assert (len(point_list.shape)==2) and (point_list.shape[-1]==3)

    point_list_res = torch.matmul(R, point_list.t()) + T
    point_list_res = point_list_res.t()
    return point_list_res


def apply_RT_on_joints(joints_src, R,translation):
    '''
    joints_src = torch.Size([1, 22, 3, 196])
    R: (3,3)
    T: (3,1)
    '''
    shape0 = joints_src.shape
    N,V,C,T = joints_src.shape 
    # (C,N,V,T)
    joints_src = joints_src.permute(2,0,1,3).contiguous().reshape(C, N*V*T )
    joints_dst = torch.matmul(R, joints_src) + translation 
    joints_dst = joints_dst.reshape(C,N,V,T).permute(1,2,0,3).contiguous()
    assert joints_dst.shape==shape0
    return joints_dst 

#################################################################
# others
#################################################################
def get_p_by_length_list(line, length_list):
    # (1,3)
    point = line[:, :3]
    # (1,3)
    direction = line[:, 3:]

    length_list = torch.FloatTensor(length_list).reshape((-1,1))
    s = point + direction * length_list
    return s 


def gather_points(p_list):
    return torch.stack(p_list, 0)



def test_relax_line():
    # a line is represented as 6 parameters. (point, direction)
    # line1: (1,6), [0,:3]=point, [0,3:]=direction
    # line2: (1,6), [0,:3]=point, [0,3:]=direction
    # find transformation [R,t] so that R*p1+t = p2
    # where p1, p2 are corresponding points on line1 and line2.

    # line:  [point, direction], 
    # plane: [point, normal]
    # start/end points: [start_point, end_point]

    p1 = [1,-1,-1]
    d1 = [1,1,2]

    p2 = [2,5,1]
    d2 = [2,1,-1]

    p1, d1 = torch.FloatTensor(p1), torch.FloatTensor(d1)
    p2, d2 = torch.FloatTensor(p2), torch.FloatTensor(d2)

    line1 = construct_line(p1, d1)
    line2 = construct_line(p2, d2)

    print("line1 = ", line1)
    print("line2 = ", line2)

    R, T = calc_RT_from_two_lines(line1, line2)
    print("R,T=")
    print(R)
    print(T)


    length_list = [0,1,5,8,-1]
    point_list_src = get_p_by_length_list(line1, length_list)
    point_list_dst = get_p_by_length_list(line2, length_list)
    print("point_list_src=\n", point_list_src)
    print("point_list_dst=\n", point_list_dst)
    print("point_list_src.shape=", point_list_src.shape)
    print("point_list_dst.shape=", point_list_dst.shape)


    point_list_src_transformed = apply_RT(point_list_src, R, T)
    print("point_list_src_transformed=\n", point_list_src_transformed)
    print("point_list_dst=\n", point_list_dst)

    diff = point_list_src_transformed - point_list_dst
    print("diff=\n", diff)
    print("max element = ", torch.abs(diff).max())




def main():
    test_relax_line()



if __name__ == "__main__":
    main()