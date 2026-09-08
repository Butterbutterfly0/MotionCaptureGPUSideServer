import torch
from utils import transforms
from torch import Tensor
from typing import Tuple

def rollout_global_motion(root_r:Tensor, root_v:Tensor, init_trans:Tensor|None=None)->Tuple[Tensor,Tensor]:
    """
    Args:
        root_r(Tensor):[B,F+1,6]根朝向序列
        root_v(Tensor):[B,F,3]根局部坐标系下的速度
        init_trans(Tensor):世界原点的平移偏置
    Returns:
        root(Tensor):[B,F,3,3]逐帧的朝向
        trans(Tensor):[B,F,3]逐帧的世界位置
    """
    # 获取Batch,Frame
    b, f = root_v.shape[:2]
    # 将6D坐标转换为旋转矩阵
    root = transforms.rotation_6d_to_matrix(root_r[:])
    # 对root_v应用旋转矩阵进行旋转到全局坐标系下
    vel_world = (root[:, :-1] @ root_v.unsqueeze(-1)).squeeze(-1)
    # 对速度进行累计
    trans = torch.cumsum(vel_world, dim=1)
    
    if init_trans is not None: trans = trans + init_trans
    return root[:, 1:], trans

def compute_camera_motion(output, root_c_d6d, root_w, trans, pred_cam):
    root_c = transforms.rotation_6d_to_matrix(root_c_d6d)  # Root orient in cam coord
    cam_R = root_c @ root_w.mT
    pelvis_cam = output.full_cam.view_as(pred_cam)
    pelvis_world = (cam_R.mT @ pelvis_cam.unsqueeze(-1)).squeeze(-1)
    cam_T_world = pelvis_world - trans
    cam_T = (cam_R @ cam_T_world.unsqueeze(-1)).squeeze(-1)
    
    return cam_R, cam_T

def compute_camera_pose(root_c_d6d, root_w):
    root_c = transforms.rotation_6d_to_matrix(root_c_d6d)  # Root orient in cam coord
    cam_R = root_c @ root_w.mT
    return cam_R


def reset_root_velocity(smpl, output, stationary, pred_ori, pred_vel, thr=0.7):
    b, f = pred_vel.shape[:2]
    
    stationary_mask = (stationary.clone().detach() > thr).unsqueeze(-1).float()
    poses_root = transforms.rotation_6d_to_matrix(pred_ori.clone().detach())
    vel_world = (poses_root[:, 1:] @ pred_vel.clone().detach().unsqueeze(-1)).squeeze(-1)
    
    output = smpl.get_output(body_pose=output.body_pose.clone().detach(),
                             global_orient=poses_root[:, 1:].reshape(-1, 1, 3, 3),
                             betas=output.betas.clone().detach(),
                             pose2rot=False)
    feet = output.feet.reshape(b, f, 4, 3)
    feet_vel = feet[:, 1:] - feet[:, :-1] + vel_world[:, 1:].unsqueeze(-2)
    feet_vel = torch.cat((torch.zeros_like(feet_vel[:, :1]), feet_vel), dim=1)
    
    stationary_vel = feet_vel * stationary_mask
    del_vel = stationary_vel.sum(dim=2) / ((stationary_vel != 0).sum(dim=2) + 1e-4)
    vel_world_update = vel_world - del_vel
    
    vel_root = (poses_root[:, 1:].mT @ vel_world_update.unsqueeze(-1)).squeeze(-1)
    
    return vel_root
