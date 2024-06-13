import numpy as np 
import torch 
import torch.nn.functional as F 



#####################################################
# joint index
#####################################################
pelvis = 0
left_hip = 1
right_hip = 2
spine1 = 3
left_knee = 4
right_knee = 5
spine2 = 6
left_ankle = 7
right_ankle = 8
spine3 = 9
left_foot = 10
right_foot = 11
neck = 12
left_collar = 13
right_collar = 14
head = 15
left_shoulder = 16
right_shoulder = 17
left_elbow = 18
right_elbow = 19
left_wrist = 20
right_wrist = 21

# HML_JOINT_NAMES = [
#     'pelvis',
#     'left_hip',
#     'right_hip',
#     'spine1',
#     'left_knee',
#     'right_knee',
#     'spine2',
#     'left_ankle',
#     'right_ankle',
#     'spine3',
#     'left_foot',
#     'right_foot',
#     'neck',
#     'left_collar',
#     'right_collar',
#     'head',
#     'left_shoulder',
#     'right_shoulder',
#     'left_elbow',
#     'right_elbow',
#     'left_wrist',
#     'right_wrist',
# ]

def get_joint(joints, joint_idx):
    return joints[:,joint_idx:joint_idx+1,:,:]





#####################################################
# dimension selection
#####################################################
def dimX(joints):
    # torch.Size([1, 22, 3, 196])
    return joints[:,:,0:1,:]

def dimY(joints):
    # torch.Size([1, 22, 3, 196])
    return joints[:,:,1:2,:]

def dimZ(joints):
    # torch.Size([1, 22, 3, 196])
    return joints[:,:,2:3,:]


def dimXZ(joints):
    # torch.Size([1, 22, 3, 196])
    return joints[:,:,[0,2],:]


#####################################################
# absolute position constraint
#####################################################
# def sample_to_joints(sample)


#####################################################
# high-order velocity constraint
#####################################################
def get_velocity(joints):
    # joints_ref = self.sample_to_joints(motion)
    joints_vel = joints[:,:,:,1:] - joints[:,:,:,:-1]
    return joints_vel


#####################################################
# relative distance constraint
#####################################################
def dist_to_point(joints_1, joints_2):
    '''
    joints: [1,22,3,t]
    '''
    dist = ((joints_1 - joints_2)**2).sum(dim=2,keepdims=True).sqrt()
    return dist 


def dist_to_point_squared(joints_1, joints_2):
    '''
    joints: [1,22,3,t]
    '''
    dist = ((joints_1 - joints_2)**2).sum(dim=2,keepdims=True)#.sqrt()
    return dist 


#####################################################
# keyframe constraint
#####################################################
def keyframe(joints, t):
    # torch.Size([1, 22, 3, 196])
    return joints[:,:,:,t:t+1]

def keyframe_list(joints, t_list):
    # torch.Size([1, 22, 3, 196])
    return joints[:,:,:,t_list]

def keyframe_by_length(joints, length):
    # torch.Size([1, 22, 3, 196])
    return joints[:,:,:,:length]


#####################################################
# geometric constraint
#####################################################
def distance_3d_point_line(data, line_params):
    '''
    data: (seqlen,3)
    return: line_params (1,6) [0:3]=p, [3:6]=d (direction vector)
    '''
    th = torch
    p = line_params[:,:3]
    d = line_params[:, 3:]
    # print("d_norm = ", d.norm())
    x = th.cross(data-p, d, dim=1).norm(dim=1)
    d_norm = d.reshape(-1).norm() 
    return x / d_norm


def loss_distance_3d_point_line(data, line_params):
    '''
    data: (seqlen,3)
    return: line_params (1,6) [0:3]=p, [3:6]=d (direction vector)
    '''
    th = torch
    p = line_params[:,:3]
    d = line_params[:, 3:]
    x = th.cross(data-p, d, dim=1)
    x = (x**2).sum(dim=1)
    return x.mean()


def fit_3d_line(data):
    '''
    data: (seqlen,3)
    return: line_params (1,6) [0:3]=p, [3:6]=n
    '''
    th = torch
    datamean = data.mean(dim=0) 
    # Do an SVD on the mean-centered data.
    uu, dd, vv = th.linalg.svd(data - datamean)
    vec_d = vv[0]
    assert np.allclose(th.norm(vec_d).item(),1.0,atol=1e-4)
    # print(datamean.shape, vec_d.shape)
    line_params = th.cat([datamean, vec_d], 0).reshape(1,-1)
    return line_params



def dist_to_plane(data, plane_params):
    '''
    data: torch.Size([1, 22->1, 3, 196])
    return: plane_params (1,4) 
    '''
    th= torch 
    # torch.Size([1, 22, 3, 196])
    # sample = sample[:, joint_list, :, :]
    joints = data
    joints = joints.permute(0,1,3,2).contiguous()
    # torch.Size([1, 22, 196, 3])

    one_col = th.ones(joints.shape[:-1]+(1,)).float().to(joints.device)
    joints_homo = th.cat([joints, one_col], 3)
    plane_params = plane_params.reshape(1,1,1,4)
    # d = (plane_params * joints_homo).sum(dim=3).abs() / ((plane_params**2).sum().sqrt())
    d = (plane_params * joints_homo).sum(dim=3).abs() / ((plane_params[:,:,:,:3]**2).sum().sqrt()) 
    return d


def loss_dist_to_plane(data, plane_params):
    '''
    data: (seqlen,3)
    return: plane_params (1,4) 
    '''
    # torch.Size([1, 22, 3, 196])
    # (seqlen, 3)
    th= torch 
    assert plane_params.shape[0]==1 and plane_params.shape[1]==4
    one_col = th.ones(data.shape[0], 1).to(data.device)
    data_homo = th.cat([data, one_col], 1)

    d = (((plane_params * data_homo).sum(dim=1))**2)
    return d


def fit_yPlane(joint_pos):
    '''
    joint_pos: (seqlen, 3), a detached tensor
    return [1,0,C,d]: Ax+By+Cz+d=0
    so that min |F_line(joint_pos)-0|
    '''
    th = torch
    # b=[x_i,...]^T
    # A=[[z_i, 1], ...]^T
    b = joint_pos[:,0:1]
    A = th.cat([ joint_pos[:,2:3], th.ones_like(b) ], 1)
    A_pinv = th.inverse(th.matmul(A.t(),A))
    # btA = th.matmul(b.t(), A)
    btA = th.matmul(A.t(), b)
    # (c,d)
    x_res = -th.matmul(A_pinv, btA)
    x_res = x_res.reshape(-1)
    plane_params = th.zeros(1,4).to(x_res.device)
    plane_params[0,0]=1.0 
    plane_params[0,1]=0.0
    plane_params[0,2]=x_res[0]
    plane_params[0,3]=x_res[1]
    return plane_params


def plane_params_4d_to_point_normal_form(plane_params):
    # plane_params: (1,4)
    # return (1,6) in (point+normal)
    th = torch
    A = plane_params[0,0]
    B = plane_params[0,1]
    C = plane_params[0,2]
    D = plane_params[0,3]

    pn = th.zeros(1,6).to(plane_params.device)
    # point = (-D,0,0), normal = (1,0,C)
    pn[0,0] = -D/A
    pn[0,1] = 0
    pn[0,2] = 0
    pn[0,3] = A
    pn[0,4] = 0.0
    pn[0,5] = C
    return pn 


def plane_point_normal_form_to_params_4d(point_normal):
    # (1,6) in (point+normal)
    # return plane_params: (1,4)
    th = torch
    plane_params = th.zeros(1,4).to(point_normal.device)
    p1 = point_normal[0,0]
    p2 = point_normal[0,1]
    p3 = point_normal[0,2]
    n1 = point_normal[0,3]
    n2 = point_normal[0,4]
    n3 = point_normal[0,5]
    
    # A=n1, B=n2, C=n3, D = -(n1*p1+n2*p2+n3*p3)
    plane_params[0,0] = n1 
    plane_params[0,1] = n2 
    plane_params[0,2] = n3 
    plane_params[0,3] = -(n1*p1+n2*p2+n3*p3)

    return plane_params


#####################################################
# center of mass constraint
#####################################################
def get_center_of_mass(joints):
    '''
    torch.Size([1, 22, 3, 196])
    each item: ([joint1,joint2], weight)
    return (seqlen,3) --> mass center trajectory.
    '''
    body_weight_list = [
        ([0,1], 7.5 ),
        ([1,4], 11.255),
        ([4,7], 5.05),
        ([7,10], 1.38),
        ([0,2], 7.5),
        ([2,5], 11.255),
        ([5,8], 5.05 ),
        ([8,11], 1.38),
        ([0,3], 6),
        ([3,6], 6.65),
        ([6,9], 6.5 ),
        ([9,12],3),
        ([12,15],5),
        ([9,13],3),
        ([13,16],3),
        ([16,18],3.075),
        ([9,14],3),
        ([14,17],3),
        ([17,19],3.075),
        ([19,21],(1.72+0.575)  ),
        ([18,20],(1.72+0.575)  ),
        # ([19,21],1.72+0.575)
        # ([18,20],1.72+0.575),
    ]

    s=0.0

    weight_list = [i[1] for i in body_weight_list]
    joint_list = np.array([i[0] for i in body_weight_list])
    joint_list_1 = joint_list[:,0].tolist()
    joint_list_2 = joint_list[:,1].tolist()

    joints1 = joints[:,joint_list_1,:,:]
    joints2 = joints[:,joint_list_2,:,:]
    weight_list_expand = torch.tensor(weight_list).to(joints.device)
    weight_list_expand = weight_list_expand/weight_list_expand.sum()
    weight_list_expand = weight_list_expand.reshape(1,-1,1,1)
    # ([1, n_bones, 3, 196])
    joints_mean = (joints1+joints2)/2.0

    # weight_list_expand = ([1, n_bones, 3, 196])

    # ([1,1,3,196])
    mass_ctr = (weight_list_expand * joints_mean).sum(dim=1,keepdims=True)
    return mass_ctr
    # assert mass_ctr.shape[0]==1 and mass_ctr.shape[1]==1
    # mass_ctr = mass_ctr[0,0]
    # mass_ctr = mass_ctr.permute(1,0).contiguous()
    # return mass_ctr



def get_center_of_mass_fixhead(joints):
    '''
    torch.Size([1, 22, 3, 196])
    each item: ([joint1,joint2], weight)
    return (seqlen,3) --> mass center trajectory.
    '''
    body_weight_list = [
        ([0,1], 7.5 ),
        ([1,4], 11.255),
        ([4,7], 5.05),
        ([7,10], 1.38),
        ([0,2], 7.5),
        ([2,5], 11.255),
        ([5,8], 5.05 ),
        ([8,11], 1.38),
        ([0,3], 6),
        ([3,6], 6.65),
        ([6,9], 6.5 ),
        ([9,12],3),
        # ([12,15],5),
        ([12,15],0),
        ([9,13],3),
        ([13,16],3),
        ([16,18],3.075),
        ([9,14],3),
        ([14,17],3),
        ([17,19],3.075),
        ([19,21],(1.72+0.575)  ),
        ([18,20],(1.72+0.575)  ),
        # ([19,21],1.72+0.575)
        # ([18,20],1.72+0.575),
    ]

    s=0.0

    weight_list = [i[1] for i in body_weight_list]
    joint_list = np.array([i[0] for i in body_weight_list])
    joint_list_1 = joint_list[:,0].tolist()
    joint_list_2 = joint_list[:,1].tolist()

    joints1 = joints[:,joint_list_1,:,:]
    joints2 = joints[:,joint_list_2,:,:]
    weight_list_expand = torch.tensor(weight_list).to(joints.device)
    weight_list_expand = weight_list_expand/weight_list_expand.sum()
    weight_list_expand = weight_list_expand.reshape(1,-1,1,1)
    # ([1, n_bones, 3, 196])
    joints_mean = (joints1+joints2)/2.0

    # weight_list_expand = ([1, n_bones, 3, 196])

    # ([1,1,3,196])
    mass_ctr = (weight_list_expand * joints_mean).sum(dim=1,keepdims=True)
    # assert mass_ctr.shape[0]==1 and mass_ctr.shape[1]==1
    # mass_ctr = mass_ctr[0,0]
    # mass_ctr = mass_ctr.permute(1,0).contiguous()
    return mass_ctr


#####################################################
# directional constraint
#####################################################
def loss_directional(joints, j, d):
    '''
    joints: torch.Size([1, 22, 3, 196])
    d: (3,)
    '''
    assert j!=0
    parents_list = [0,0,0,0,1,2,3,4,5,6,7,8,
                    9,9,9,12,13,14,16,17,18,19]
    joint = get_joint(joints, j)
    joint_par = get_joint(joints, parents_list[j])
    d_pred = joint - joint_par
    # (3,length)
    d_pred = d_pred.squeeze(dim=1).squeeze(dim=0)
    # (3,1)
    d_gt   = d.unsqueeze(dim=1)

    # (3,length)
    d_pred_normed = d_pred / d_pred.detach().norm(dim=0)
    d_gt = d_gt / d_gt.detach().norm(dim=0)

    loss = ((d_pred_normed - d_gt)**2).sum(dim=0)
    # (length,)
    return loss 


#####################################################
# logical operation
#####################################################
def less_than(joints, margin):
    return F.relu( joints-margin )


def greater_than(joints, margin):
    return F.relu( margin-joints )


def equal(pred, gt):
    '''
    shape=(bs, joint, dim, t)
    '''
    # loss_reg = F.mse_loss(pred, gt )
    loss = ((pred-gt)**2).mean()
    return loss


def equal_L1(pred, gt):
    loss = (pred-gt).abs().mean()
    return loss 


def equal_sum(v0_pred, v0_gt):
    '''
    v0_pred: (3,)
    v0_gt: (3,)
    '''
    d2 = ((v0_pred - v0_gt)**2).sum()
    return d2 


def operation_or(joints1, joints2):
    return torch.minimum(joints1, joints2)




#####################################################
# utils
#####################################################

def mid_point(joints_1, joints_2):
    return (joints_1+joints_2)/2.0


def tensor_to_gpt_form(joints):
    '''
    args:
        joints: [1,22,3,t]
    return:
        motion
    [{'joint1': {'x':x, 'y':y, 'z':z},
      'joint2': {'x':x, 'y':y, 'z':z},
    },frame2, frame_n]
    '''
    HML_JOINT_NAMES = [
        'pelvis',
        'left_hip',
        'right_hip',
        'spine1',
        'left_knee',
        'right_knee',
        'spine2',
        'left_ankle',
        'right_ankle',
        'spine3',
        'left_foot',
        'right_foot',
        'neck',
        'left_collar',
        'right_collar',
        'head',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
    ]

    motion=[]
    n_frames = joints.shape[3]
    n_joints = joints.shape[1]
    for t in range(n_frames):
        frame = {}
        for j in range(n_joints):
            x = joints[:, j, 0, t]
            y = joints[:, j, 1, t]
            z = joints[:, j, 2, t]
            frame[ HML_JOINT_NAMES[j] ] = {'x': x, 'y': y, 'z': z}
        motion.append(frame)
    return motion 
    







