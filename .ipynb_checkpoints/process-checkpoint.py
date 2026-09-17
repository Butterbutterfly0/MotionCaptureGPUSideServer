import joblib
import numpy as np
import cv2
from scipy.spatial.transform import Rotation,Slerp

import pyrender
import trimesh

def interp_rot_batch(R_prev, R_next, ts):
    r0 = Rotation.from_matrix(R_prev)
    r1 = Rotation.from_matrix(R_next)
    key_rots = Rotation.concatenate([r0, r1])
    slerp = Slerp([0,1], key_rots)
    ts = (np.arange(1, ts + 1)) / (ts + 1)
    return slerp(ts).as_matrix()

def interp_linear_batch(R_prev,R_next,n):
    r0 = np.asarray(R_prev, dtype=float)
    r1 = np.asarray(R_next, dtype=float)
    ts = np.arange(1,n+1)/(n+1)       # (23,)
    out = (1 - ts)[:, None] * r0 + ts[:, None] * r1
    return out

def LBS_forward(frames, template):
    N = len(frames['frame_id'])
    betas = frames['betas'][0]
    # 体型混合
    v_shaped = template['v_template'] + np.einsum('vck,k->vc', template['shapedirs'], betas)
    # 关节回归
    J = template['J_regressor']@v_shaped
    print(betas.shape,v_shaped.shape)
    parents = template['kintree_table'][0].astype(np.int64)
    parents[0] = -1
    print(parents.shape)
    res_verts = np.empty((N,6890,3))
    rel_j = J.copy()
    rel_j[1:] = J[1:] - J[parents[1:]]
    # 
    for k,i in enumerate(frames['frame_id']):
        print(i,'blend')
        # 姿态blend shape
        global_orient = frames['poses_root_cam'][i,0]
        body_pose = frames['poses_body'][i]
        rot_mats = np.concatenate([global_orient[None], body_pose],axis=0)
        pose_feature = (rot_mats[1:] - np.eye(3)).reshape(-1)
        v_posed = v_shaped + np.einsum('vcp,p->vc',template['posedirs'],pose_feature)
        # FK
        T = np.zeros((24,4,4))
        T[:,3,3] = 1
        T[:,:3,:3] = rot_mats
        T[:,:3,3] = rel_j
        A = np.empty((24,4,4))
        for j in range(24):
            A[j] = T[j] if parents[j] == -1 else A[parents[j]] @ T[j]
        # LBS
        v_posed = v_posed[None] - J[:, None, :]
        v_posed_homo = np.concatenate([v_posed, np.ones((24,6890,1))], axis=2)
        transformed = np.einsum('jab,jvb->jva',A, v_posed_homo)
        verts = np.einsum('jva,vj->va',transformed, template['weights'])[:,:3]
        offset = (A[1, :3, 3] + A[2, :3, 3]) / 2.0   # 标准SMPL关节1、2 = 左右髋, 取中点
        verts = verts - offset                        # 减去髋部中点(WHAM 的做法)
        verts=verts+frames['trans_cam'][i]
        # res_verts = np.concatenate([res_verts,verts[None]],axis=0)
        res_verts[k] = verts
    return res_verts

def converts_verts_from_world_to_camera(frame_num,verts_world,slam_results):
    verts_cam = []
    def slam_to_world2cam(slam_i):
        t_cw = slam_i[:3]
        R_cw = Rotation.from_quat(slam_i[3:]).as_matrix()
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        return R_wc, t_wc
    
    for i in range(frame_num):
        R_wc, t_wc = slam_to_world2cam(slam_results[i])
        p_cam = (R_wc @ verts_world[i].T).T + t_wc
        verts_cam.append(np.array(p_cam))
    return np.stack(verts_cam,axis=0)

def load_wham_results(filepath:str):
    return joblib.load(filepath)

def get_video_total_frame(videopath:str):
    cap = cv2.VideoCapture(videopath)
    ret = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return ret

def merge_wham_results(results):
    whole = {}
    for index in results:
        print('-'*10+str(index)+'th'+'-'*10)
        chunk = results[index]
        # poses_body [N, 23, 3, 3] 23个身体关节的旋转
        # poses_root_cam [N, 1, 3, 3] 根关节在相机系的旋转
        # betas [N, 10] 体型参数
        # verts_cam [N, 6890, 3] 相机系下的网络顶点
        # poses_root_world [N, 3, 3] 根关节在世界系的旋转
        # trans_world [N, 3]  根关节在世界坐标系的平移
        # frame_id [N] 帧号
        for k,v in chunk.items():
            if whole.get(k,None) is None:
                whole[k] = v
            else:
                whole[k] = np.concatenate([whole[k], v], axis=0)
            # print(k,':',v.shape)
    return whole

def interp_wham_results(whole,total_frame_count):
    # 对缺失的帧进行插值
    missed_frames = []
    expected = 0

    for frame_id in whole['frame_id']:
        if frame_id < expected:
            continue

        if frame_id > expected:
            missed_frames.append([expected,frame_id-1])
        expected = frame_id + 1

    if expected < total_frame_count:
        missed_frames.append([expected,total_frame_count-1])

    missed_frames.reverse()
    for missed_frame in missed_frames:
        front_frame_id = missed_frame[0]-1
        back_frame_id = missed_frame[1]+1

        front_frame = None
        if front_frame_id >= 0:
            mask = whole['frame_id'] == front_frame_id
            front_frame = {'poses_body':whole['poses_body'][mask],
                        'poses_root_cam':whole['poses_root_cam'][mask],
                        'betas':whole['betas'][mask],
                        'verts_cam':whole['verts_cam'][mask],
                        'poses_root_world':whole['poses_root_world'][mask],
                            'trans_world':whole['trans_world'][mask],
                            'frame_id':whole['frame_id'][mask],
                            'trans_cam':whole['trans_cam'][mask]}
        back_frame = None
        if back_frame_id < total_frame_count:
                    mask = whole['frame_id'] == back_frame_id
                    back_frame = {'poses_body':whole['poses_body'][mask],
                                'poses_root_cam':whole['poses_root_cam'][mask],
                                'betas':whole['betas'][mask],
                                'verts_cam':whole['verts_cam'][mask],
                                'poses_root_world':whole['poses_root_world'][mask],
                                    'trans_world':whole['trans_world'][mask],
                                    'frame_id':whole['frame_id'][mask],
                                    'trans_cam':whole['trans_cam'][mask]}

        # 头/尾缺帧: 只有一边有锚点。缺的那一边就拿对侧锚点填上, 这样
        # slerp/线性插值两端同值, 自动退化成"保持", 不用另写一套分支。
        # front_frame_id/back_frame_id 本来就是对的: 头缺口算出 -1, 尾缺口算出
        # total_frame_count, 所以 interp_num 和 arange(front+1, back) 原样成立。
        if front_frame is None and back_frame is None:
            continue                      # 整个序列一帧都没有, 无从补起
        if front_frame is None:           # 从头就缺
            front_frame = back_frame
        if back_frame is None:            # 到尾都缺
            back_frame = front_frame

        if front_frame is not None and back_frame is not None:
            interp = {}
            interp_num = back_frame_id - front_frame_id - 1
            # poses_body
            poses_body = np.empty((interp_num,0,3,3))
            for i in range(23):
                R_prev = front_frame['poses_body'][0,i]
                R_next = back_frame['poses_body'][0,i]
                mats = interp_rot_batch(R_prev,R_next,interp_num)
                mats = mats[:,None,:,:]
                poses_body = np.concatenate([poses_body, mats], axis=1)
            # print(poses_body.shape)
            # poses_root_cam
            R_prev = front_frame['poses_root_cam'][0,0]
            R_next = back_frame['poses_root_cam'][0,0]
            poses_root_cam = interp_rot_batch(R_prev, R_next, interp_num)[:,None,:,:]
            # print(poses_root_cam.shape)
            # betas
            R_prev = front_frame['betas'][0]
            R_next = back_frame['betas'][0]
            betas = interp_linear_batch(R_prev, R_next, interp_num)
            # print(betas.shape)
            # verts_cam
            verts_cam = np.empty((interp_num,0,3))
            for i in range(6890):
                R_prev = front_frame['verts_cam'][0,i]
                R_next = back_frame['verts_cam'][0,i]
                mats = interp_linear_batch(R_prev, R_next,interp_num)[:,None,:]
                verts_cam = np.concatenate([verts_cam,mats],axis=1)
            # print(verts_cam.shape)
            # poses_root_world
            R_prev = front_frame['poses_root_world'][0]
            R_next = back_frame['poses_root_world'][0]
            poses_root_world = interp_rot_batch(R_prev, R_next, interp_num)
            # print(poses_root_world.shape)
            # trans_world
            R_prev = front_frame['trans_world'][0]
            R_next = back_frame['trans_world'][0]
            trans_world = interp_linear_batch(R_prev, R_next, interp_num)
            # trans_cam
            R_prev = front_frame['trans_cam'][0]
            R_next = back_frame['trans_cam'][0]
            trans_cam = interp_linear_batch(R_prev, R_next, interp_num)

            # frame_id
            frame_id = np.arange(front_frame_id+1,back_frame_id)

            hit = np.where(whole['frame_id'] == front_frame_id)[0]        # 位置索引
            insert_at = hit[0] + 1 if len(hit) else 0    # 头缺口没有前锚点 -> 插最前面
            new = {
                'poses_body':       poses_body,          # (n,23,3,3)
                'poses_root_cam':   poses_root_cam,      # (n,1,3,3)
                'betas':            betas,               # (n,10)
                'verts_cam':        verts_cam,           # (n,6890,3)
                'poses_root_world': poses_root_world,    # (n,3,3)
                'trans_world':      trans_world,         # (n,3)
                'trans_cam':        trans_cam,
                'frame_id':         frame_id,            # (n,)
            }

            for k in whole:                              # 关键:一次性全拼,永不遗漏
                whole[k] = np.concatenate(
                    [whole[k][:insert_at], new[k], whole[k][insert_at:]],
                    axis=0
                )

if __name__ == '__main__':
    result = load_wham_results('wham_results.pth')
    whole = merge_wham_results(result)
    total_frame_count = get_video_total_frame('origin.mp4')
    interp_enable = True

    if interp_enable:
        interp_wham_results(whole)

    template = np.load('SMPL_FEMALE_np.npz')
    # J_regressor_prior
    # f
    # J_regressor
    # kintree_table
    # J
    # weights_prior
    # weights
    # vert_sym_idxs
    # posedirs
    # pose_training_info
    # bs_style
    # v_template
    # shapedirs
    # bs_type
    verts_cam = LBS_forward(whole, template)
    print('LBS Done')
    verts_cam[:,:,1] *= -1  
    verts_cam[:,:,2] *= -1  
    faces = template['f']
    cap = cv2.VideoCapture('origin.mp4')
    writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
    r = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)
    fx = fy = 1468.60478
    cx,cy = 640.0, 360.0
    camera = pyrender.IntrinsicsCamera(fx,fy,cx,cy)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0],          # 背景透明
                       ambient_light=[0.3, 0.3, 0.3]) # 环境光，避免背光面全黑
    scene.add(camera)
    node =None
    for i in range(verts_cam.shape[0]):
        print(i,'render')
        ok, frame = cap.read()
        if not ok:
            break
        if node is not None:
            scene.remove_node(node)
        mesh = trimesh.Trimesh(vertices=verts_cam[i], faces=faces)
        node = scene.add(pyrender.Mesh.from_trimesh(mesh))
        color, depth = r.render(scene,flags=pyrender.RenderFlags.RGBA)
        rgb = color[:, :, :3][:, :, ::-1].copy()   # RGB -> BGR
        alpha = color[:, :, 3:4].astype(np.float32) / 255.0   # (H,W,1)
        frame_f = frame.astype(np.float32)
        blended = rgb.astype(np.float32) * alpha + frame_f * (1.0 - alpha)
        blended = blended.astype(np.uint8)
        writer.write(blended)
    writer.release()
    cap.release()
