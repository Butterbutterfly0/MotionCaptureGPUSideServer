from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
from torch import nn,Tensor
from configs import constants as _C
from .utils import rollout_global_motion
from utils.transforms import axis_angle_to_matrix
from typing import List,Tuple,Dict

class Regressor(nn.Module):
    """
    RNN解码头,被(MotionEncoder/TrajectoryDecoder/MotionDecoder/TrajectoryRefiner)公用
    实际上使用LSTM逐帧递归解码

    """
    def __init__(self, in_dim:int, hid_dim:int, out_dims:List[int], init_dim:int, layer:str='LSTM', n_layers:int=2):
        """
        构造器
        Args:
            in_dim(int):当前帧特征维数
            hid_dim(int):RNN隐状态宽度
            out_dims(List[int]):输出头列表
            init_dim(int):条件输入总宽度
            layer(str):RNN类型
            n_layers(int):堆叠深度
        """
        super().__init__()
        self.n_outs = len(out_dims)

        # 使用字符串进行网络类型路由，注册RNN
        self.rnn = getattr(nn, layer.upper())(
            in_dim + init_dim, hid_dim, n_layers, 
            bidirectional=False, batch_first=True, dropout=0.3)


        for i, out_dim in enumerate(out_dims):
            # 动态注册多个Linear头
            setattr(self, 'declayer%d'%i, nn.Linear(hid_dim, out_dim))
            # 近零初始化解码头
            nn.init.xavier_uniform_(getattr(self, 'declayer%d'%i).weight, gain=0.01)

    def forward(self, x:Tensor, inits:List[Tensor], h0:Tensor)->Tuple[List[Tensor],Tensor,Tensor]:
        """
        前向传递函数
        每次将输入特征与所有条件拼接在一起作为RNN的输入,同时产生的h0循环地用在网络中
        RNN的隐输出通过多个线性层输出为多个解码头的输出
        Args:
            x(Tensor):当前输入特征
            inits(List[Tensor]):条件列表
            h0(Tensor):隐状态
        """
        # 将初始状态和输入拼接作为真正输出
        xc = torch.cat([x, *inits], dim=-1)
        xc, h0 = self.rnn(xc, h0)

        # 对每个解码头进行求输出
        preds = []
        for j in range(self.n_outs):
            out = getattr(self, 'declayer%d'%j)(xc)
            preds.append(out)
        # 返回预测输出，隐状态，更新后的末态
        return preds, xc, h0
    
    
class NeuralInitialization(nn.Module):
    """
    解决的是RNN的初始隐状态从哪来的问题?
    该模块通过将第一帧的单帧回归结果用一个MLP学习成一个状态向量,喂给LSTM
    用一个MLP装下所有层的h和c
    """
    def __init__(self, in_dim:int, hid_dim:int, layer:str, n_layers:int):
        """
        构造器
        Args:
            in_dim(int):输入特征维度
            hid_dim(int):隐藏层特征维度
            layer(str):字符路由网络类型
            n_layers(int):层数
        """
        super().__init__()

        out_dim = hid_dim
        self.n_layers = n_layers
        self.num_inits = int(layer.upper() == 'LSTM') + 1 # LSTM有两个初始向量h,c
        out_dim *= self.num_inits * n_layers

        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim * self.n_layers)
        self.linear3 = nn.Linear(hid_dim * self.n_layers, out_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x:Tensor)->Tensor|Tuple[Tensor]:
        """
        前向传递函数
        Args:
            x(Tensor):[B,1,in_dim]每个序列一条首帧向量
        """
        b = x.shape[0]

        out = self.linear3(self.relu2(self.linear2(self.relu1(self.linear1(x)))))
        out = out.view(b, self.num_inits, self.n_layers, -1).permute(1, 2, 0, 3).contiguous()
        # 先将输出调整为[Batch, num_inits,layers,dim]的形状
        # 再重排为[num_inits, layers, Batch,dim]的形状
        # 仅用permute仅会调整引用的顺序，使用contiguous()理顺内存内的布局

        if self.num_inits == 2:
            return tuple([_ for _ in out])
        return out[0]


class Integrator(nn.Module):
    """
    将每帧图像特征注入纯运动上下文的小MLP
    """
    def __init__(self, in_channel:int, out_channel:int, hid_channel:int=1024):
        """
        构造器
        Args:
            in_channel(int):dim_feat(特征维度)+dim_context(上下文维度)
            out_channel(int):dim_context(上下文维度)
            hid_channel(int):隐藏维度
        """
        super().__init__()
        
        self.layer1 = nn.Linear(in_channel, hid_channel)
        self.relu1 = nn.ReLU()
        self.dr1 = nn.Dropout(0.1)
        
        self.layer2 = nn.Linear(hid_channel, hid_channel)
        self.relu2 = nn.ReLU()
        self.dr2 = nn.Dropout(0.1)
        
        self.layer3 = nn.Linear(hid_channel, out_channel)
        
        
    def forward(self, x:Tensor, feat:Tensor):
        """
        前向传递函数
        Args:
            x(Tensor):
            feat(Tensor):
        """
        res = x # 残差
        mask = (feat != 0).all(dim=-1).all(dim=-1)
        # 主要是训练时有的数据，只有运动数据没有图像数据，对于这种数据用全零张量代替
        
        out = torch.cat((x, feat), dim=-1) # 拼接特征维度和上下文维度
        out = self.layer1(out)
        out = self.relu1(out)
        out = self.dr1(out)
        
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.dr2(out)
        
        out = self.layer3(out)
        out[mask] = out[mask] + res[mask]
        
        return out


class MotionEncoder(nn.Module):
    """
    输入一段2D关键点序列,输出:时序平滑的3D关节点+每帧一个563维的运动上下文
    """
    def __init__(self, 
                 in_dim:int, 
                 d_embed:int,
                 pose_dr:float,
                 rnn_type:str,
                 n_layers:int,
                 n_joints:int):
        super().__init__()
        """
        Args:
            in_dim(int):输入特征的维度
            d_embed(int):
            pose_dr(float):特征dropout率
            rnn_type(str):rnn网络的类型
            n_layers(int):堆叠深度
            n_joints(int):关节数量
        """
        
        self.n_joints = n_joints
        
        self.embed_layer = nn.Linear(in_dim, d_embed)
        self.pos_drop = nn.Dropout(pose_dr)
        
        # Keypoints initializer
        self.neural_init = NeuralInitialization(n_joints * 3 + in_dim, d_embed, rnn_type, n_layers)
        
        # 3d keypoints regressor
        self.regressor = Regressor(
            d_embed, d_embed, [n_joints * 3], n_joints * 3, rnn_type, n_layers)
        
    def forward(self, x:Tensor, init:Tensor)->Tuple[Tensor,Tensor]:
        """ 
        Forward pass of motion encoder.
        Args:
            x(Tensor):[B,F,37],每帧17个归一化2D关节点+3个附加标量
            init(Tensor):[B,1,88]仅首帧:kp3d(51)+kp2d(37)
        """
        # 嵌入
        self.b, self.f = x.shape[:2]
        x = self.embed_layer(x.reshape(self.b, self.f, -1))
        x = self.pos_drop(x)
        # 启动
        h0 = self.neural_init(init)
        pred_list = [init[..., :self.n_joints * 3]]
        motion_context_list = []
        # 逐帧自回归循环
        for i in range(self.f):
            (pred_kp3d, ), motion_context, h0 = self.regressor(x[:, [i]], pred_list[-1:], h0)
            motion_context_list.append(motion_context)
            pred_list.append(pred_kp3d)
        # 汇总拼接
        pred_kp3d = torch.cat(pred_list[1:], dim=1).view(self.b, self.f, -1, 3)
        motion_context = torch.cat(motion_context_list, dim=1)
        
        # Merge 3D keypoints with motion context
        motion_context = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        return pred_kp3d, motion_context


class TrajectoryDecoder(nn.Module):
    """
    解码人在世界里的运动轨迹
    逐帧输出根的全局朝向和根的速度-合成粗轨迹
    """
    def __init__(self, 
                 d_embed:int,
                 rnn_type:str,
                 n_layers:int):
        super().__init__()
        
        # Trajectory regressor
        self.regressor = Regressor(
            d_embed, d_embed, [3, 6], 12, rnn_type, n_layers, )
        # [3,6] [根速度,根朝向]
        
    def forward(self, x:Tensor, root:Tensor, cam_a:Tensor, h0:Tensor|None=None)->Tuple[Tensor,Tensor]:
        """ 
        输出预测的所有帧的相机角速度和6D位姿
        Args:
            x(Tensor):未注入图像特征的motion_context
            root(Tensor):首帧根朝向,第一帧的全局朝向
            cam_a(Tensor):相机角速度:第i帧相机相对于第i+1的旋转r6d

        """
        
        b, f = x.shape[:2]
        pred_root_list, pred_vel_list = [root[:, :1]], []
        
        for i in range(f):
            # Global coordinate estimation
            (pred_rootv, pred_rootr), _, h0 = self.regressor(
                x[:, [i]], [pred_root_list[-1], cam_a[:, [i]]], h0)
            
            pred_root_list.append(pred_rootr)
            pred_vel_list.append(pred_rootv)
        
        pred_root = torch.cat(pred_root_list, dim=1).view(b, f + 1, -1)
        pred_vel = torch.cat(pred_vel_list, dim=1).view(b, f, -1)
        
        return pred_root, pred_vel
        

class MotionDecoder(nn.Module):
    """
    将运动上下文逐帧解码成SMPL参数四件套---pose(24x6 r6d), shape, cam, contact
    """
    def __init__(self, 
                 d_embed:int,
                 rnn_type:str,
                 n_layers:int):
        super().__init__()
        
        self.n_pose = 24
        
        # SMPL pose initialization
        self.neural_init = NeuralInitialization(len(_C.BMODEL.MAIN_JOINTS) * 6, d_embed, rnn_type, n_layers)
        
        # 3d keypoints regressor
        self.regressor = Regressor(
            d_embed, d_embed, [self.n_pose * 6, 10, 3, 4], self.n_pose * 6, rnn_type, n_layers)
        
    def forward(self, x:Tensor, init:Tensor)->Tuple[Tensor,Tensor,Tensor,Tensor]:
        """ Forward pass of motion decoder.
        """
        # 初始起点
        b, f = x.shape[:2]
        
        h0 = self.neural_init(init[:, :, _C.BMODEL.MAIN_JOINTS].reshape(b, 1, -1))
        
        # Recursive prediction of SMPL parameters
        pred_pose_list = [init.reshape(b, 1, -1)]
        pred_shape_list, pred_cam_list, pred_contact_list = [], [], []
        # 自循环解码
        for i in range(f):
            # Camera coordinate estimation
            (pred_pose, pred_shape, pred_cam, pred_contact), _, h0 = self.regressor(x[:, [i]], pred_pose_list[-1:], h0)
            pred_pose_list.append(pred_pose)
            pred_shape_list.append(pred_shape)
            pred_cam_list.append(pred_cam)
            pred_contact_list.append(pred_contact)
        # 合并输出
        pred_pose = torch.cat(pred_pose_list[1:], dim=1).view(b, f, -1)
        pred_shape = torch.cat(pred_shape_list, dim=1).view(b, f, -1)
        pred_cam = torch.cat(pred_cam_list, dim=1).view(b, f, -1)
        pred_contact = torch.cat(pred_contact_list, dim=1).view(b, f, -1)
        
        return pred_pose, pred_shape, pred_cam, pred_contact


class TrajectoryRefiner(nn.Module):
    """"
    给粗轨迹打防滑补丁
    """
    def __init__(self, 
                 d_embed:int,
                 d_hidden:int, 
                 rnn_type:str,
                 n_layers:int):
        super().__init__()
        
        d_input = d_embed + 12
        self.refiner = Regressor(
            d_input, d_hidden, [6, 3], 9, rnn_type, n_layers)

    def forward(self, context:Tensor, pred_vel:Tensor, output:Dict, cam_angvel:Tensor, return_y_up:Tensor)->Dict:
        b, f = context.shape[:2]
        # 梯度隔离
        pred_root = output['poses_root_r6d'].clone().detach()
        feet = output['feet'].clone().detach()
        contact = output['contact'].clone().detach()

        
        feet_vel = torch.cat((torch.zeros_like(feet[:, :1]), feet[:, 1:] - feet[:, :-1]), dim=1) * 30   # Normalize to 30 times
        feet = (feet_vel * contact.unsqueeze(-1)).reshape(b, f, -1)  # Velocity input
        inpt_feat = torch.cat([context, feet], dim=-1)
        
        (delta_root, delta_vel), _, _ = self.refiner(inpt_feat, [pred_root[:, 1:], pred_vel], h0=None)
        pred_root[:, 1:] = pred_root[:, 1:] + delta_root
        pred_vel = pred_vel + delta_vel

        # root_world, trans_world = rollout_global_motion(pred_root, pred_vel)
        
        # if return_y_up:
        #     yup2ydown = axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float().to(root_world.device)
        #     root_world = yup2ydown.mT @ root_world
        #     trans_world = (yup2ydown.mT @ trans_world.unsqueeze(-1)).squeeze(-1)
            
        output.update({
            'poses_root_r6d_refined': pred_root,
            'vel_root_refined': pred_vel,
            # 'poses_root_world': root_world,
            # 'trans_world': trans_world,
        })
        
        return output
