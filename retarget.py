# -*- coding: utf-8 -*-
"""SMPL(WHAM) -> VRM 姿态重定向, 以及给 Blender 用的欧拉角转换

硬依赖只有 numpy。GLB 用 stdlib 解析, mathutils 只在 BoneFrames 里按需导入,
所以 **Blender 里和普通 python 里都能 import**:

    SmplRig / VrmRig / Retargeter     两边都能用
    BoneFrames                        只有 Blender 里能用(要 armature + mathutils)

旋转和**根位移**都做。根位移不走骨架 FK, 而是直接从相机系的 trans_cam 搬过来,
除以参考帧消掉绝对位置, 再换两次基底。见 Retargeter.root_translation。

原理: 两套骨架的 rest 都是 T-pose, 局部旋转可以直接搬运, 只需换坐标基底。
    W_t[j]   SMPL 关节 j 在相机系下的世界旋转 (由 FK 得到)
    S_t[j] = W_t[0]^T @ W_t[j]        关节 j 相对骨盆, 在骨盆系里表示
Roxy 节点 n (对应 SMPL 关节 m, 父节点对应关节 p) 的局部旋转是
    R[n] = C @ ( S_t[p]^T @ S_t[m] ) @ C^T

用累积量 S 而不是 SMPL 自己的局部旋转, 好处是没有对应节点的 SMPL 关节
(例如 spine2) 的旋转会被子关节自动带入, 不会丢。

用法(两个进程, 各跑一边):
    # --- 批量那边 ---
    rig = Retargeter(SmplRig('SMPL_FEMALE_np.npz'), VrmRig('Roxy.vrm'))
    rig.set_reference(poses_root_cam[0, 0])
    R = rig.retarget(poses_root_cam[f, 0], poses_body[f])     # {节点索引: 3x3}

    # --- Blender 那边 ---
    bf = BoneFrames(arm, VrmRig('Roxy.vrm'))                  # 建一次
    eulers = bf.euler(R)                                      # {Blender骨名: (rx,ry,rz)}
    # 键是 Blender 骨名(Root_M/Shoulder_L/...), 值直接喂 rotation_euler

不想拆两个进程也行 —— 两边都用 numpy, 在 Blender 里一口气跑完即可。

自检: python retarget.py [SMPL.npz] [Roxy.vrm] [wham_results.pth]
"""
import json, struct
import numpy as np


# ================================================================ 常量
# SMPL 24 关节 -> VRM humanoid 键。None 表示该关节没有对应骨骼,
# 它的旋转由子关节的累积量自动带入。6(spine2) 必须留空:
# 若 6 和 9 都指向 Chest_M, 两者会互相覆盖。
smpl2vrm = {
    0: 'hips',
    1: 'leftUpperLeg',
    2: 'rightUpperLeg',
    3: 'spine',
    4: 'leftLowerLeg',
    5: 'rightLowerLeg',
    6: None,
    7: 'leftFoot',
    8: 'rightFoot',
    9: 'Chest_M',
    10: 'leftToes',
    11: 'rightToes',
    12: 'neck',
    13: 'leftShoulder',
    14: 'rightShoulder',
    15: 'head',
    16: 'leftUpperArm',
    17: 'rightUpperArm',
    18: 'leftLowerArm',
    19: 'rightLowerArm',
    20: 'leftHand',
    21: 'rightHand',
    22: None,
    23: None,
}

# SMPL 模板系 (+X左 +Y上 +Z前) -> glTF/VRM 系 (+X右 +Y上 +Z后)
# 这是 180° 绕 Y 的**真旋转**, det(C) = +1, 所以 C M C^T 仍然落在 SO(3) 里,
# 不会把旋转变成镜像。
C = np.diag([-1.0, 1.0, -1.0])

SMPL_NJ = 24          # SMPL 关节数

# Roxy / SMPL 身高比。roxy_retarget.py 里由四条肢体链长度平均得到, 已随 BVH 验证过。
SCALE = 0.8055

# glTF 全局系 -> Blender armature 系。用 177 根同名骨骼的 rest 位置做 Kabsch 拟合,
# 残差 0.0000 m, det = +1。直观理解: glTF 是 Y-up, Blender 这边 Roxy 沿 +Z 站立、
# 面朝 -Y, 差分正好是这个 180° 旋转。做**位移**和做**全局位置**用的是同一个 S
# (因为拟合出的平移项 t = 0)。
S_BLENDER = np.array([[-1., 0., 0.],
                      [0., 0., 1.],
                      [0., 1., 0.]])


# ================================================================ 通用
def _get(obj, key, default=None):
    """容忍对象/字典两种取法, 并把 None 归一成 default"""
    if obj is None:
        return default
    v = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    return default if v is None else v


def _q2m(q):
    """glTF 四元数 (x,y,z,w) -> 3x3"""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_glb(path):
    """纯 stdlib 解析 GLB, 返回 JSON 块 (不依赖 pygltflib)"""
    raw = open(path, 'rb').read()
    assert raw[:4] == b'glTF', '不是 GLB 文件: %s' % path
    off, js = 12, None
    while off < len(raw):
        clen, ctype = struct.unpack('<II', raw[off:off + 8])
        if ctype == 0x4E4F534A:                       # 'JSON'
            js = json.loads(raw[off + 8: off + 8 + clen])
        off += 8 + clen
    return js


# ================================================================ SMPL 骨架
class SmplRig:
    """SMPL 骨架: 静止姿态 + 前向运动学; npz 只读一次"""

    def __init__(self, smpl_path):
        d = np.load(smpl_path)
        J = d['J_regressor'] @ d['v_template']      # (24,3) T-pose 关节位置
        kpar = d['kintree_table'][0].astype(np.int64)
        kpar[0] = -1
        self.kpar = kpar
        t_rest = np.empty((SMPL_NJ, 3))
        t_rest[0] = J[0]
        t_rest[1:] = J[1:] - J[kpar[1:]]            # 相对父关节的偏移
        self.t_rest = t_rest

    def fk(self, root_rot, body_pose):
        """一帧 -> (24,4,4) 全局变换; G[j][:3,3] 就是关节 j 的位置"""
        Rj = np.empty((SMPL_NJ, 3, 3))
        Rj[0] = root_rot
        Rj[1:] = body_pose
        L = np.zeros((SMPL_NJ, 4, 4))
        L[:, 3, 3] = 1.0
        L[:, :3, :3] = Rj
        L[:, :3, 3] = self.t_rest        # 平移写第 4 列; 写成 [:3,:3] 会把旋转覆盖掉
        G = np.empty((SMPL_NJ, 4, 4))
        for j in range(SMPL_NJ):
            G[j] = L[j] if self.kpar[j] < 0 else G[self.kpar[j]] @ L[j]
        return G


def smpl_fk(smpl_path, root_rot, body_pose):
    """保持原来的函数签名, 单次调用用"""
    return SmplRig(smpl_path).fk(root_rot, body_pose)


# ================================================================ VRM 骨架
class VrmRig:
    """Roxy.vrm 的节点树 + humanoid 映射; 整个文件只解析一次"""

    def __init__(self, vrm_file):
        gltf = load_glb(vrm_file)
        self.nodes = gltf['nodes']

        ext = gltf.get('extensions') or {}
        hb = _get(_get(_get(ext, 'VRM') or {}, 'humanoid') or {}, 'humanBones') or []
        self.b2n = {_get(b, 'bone'): _get(b, 'node') for b in hb}

        parent = {}
        for i, nd in enumerate(self.nodes):
            for c in _get(nd, 'children') or ():
                parent[c] = i
        self.parent = parent

        self.rest_t = {i: np.array(_get(nd, 'translation', (0., 0., 0.)), float)
                       for i, nd in enumerate(self.nodes)}

        by_name = {}
        for i, nd in enumerate(self.nodes):
            by_name.setdefault(_get(nd, 'name'), []).append(i)
        self.by_name = by_name

        # 各节点的 rest 局部旋转 L_j、rest 全局旋转 G_j, 以及父先子后的遍历序
        self.Lloc = np.stack([_q2m(nd['rotation']) if 'rotation' in nd else np.eye(3)
                              for nd in self.nodes])
        n = len(self.nodes)
        self.Grest = np.empty((n, 3, 3))
        self.order, seen = [], set()

        def walk(i):
            if i in seen:
                return
            seen.add(i)
            self.order.append(i)
            p = parent.get(i)
            self.Grest[i] = self.Lloc[i] if p is None else self.Grest[p] @ self.Lloc[i]
            for c in _get(self.nodes[i], 'children') or ():
                walk(c)

        for i in range(n):
            walk(i)

    def N(self, target):
        """目标骨骼名 -> 节点索引。有些骨骼(如 Chest_M)不在 humanoid 里, 按节点名找"""
        if target in self.b2n:
            return self.b2n[target]
        if target in self.by_name:
            return self.by_name[target][0]
        raise KeyError('%r 既不在 humanoid 映射里, 也不是任何节点的名字' % (target,))

    def name(self, idx):
        """节点索引 -> 名字 (Blender 里用的就是这个名字)"""
        return _get(self.nodes[idx], 'name')


# ================================================================ 重定向
class Retargeter:
    """把 SMPL 的一帧姿态搬到 VRM 节点上"""

    def __init__(self, smpl_rig, vrm_rig):
        self.smpl = smpl_rig
        self.vrm = vrm_rig
        self.root_ref = np.eye(3)
        self.ref_trans = None                    # 参考帧的 trans_cam, 根位移要用
        self.hips_node = vrm_rig.N('hips')
        self.rest_hips = np.asarray(vrm_rig.rest_t[self.hips_node], float)
        # 节点索引 -> SMPL 关节, 供"在 Roxy 层级里往上找最近的已映射祖先"用
        self.node2j = {vrm_rig.N(t): j for j, t in smpl2vrm.items() if t}

    def set_reference(self, root_rot, root_trans=None):
        """定基准帧: 传第 0 帧的根旋转, 角色从这个朝向起手。
        要用根位移的话还得把第 0 帧的 trans_cam 一起传进来。"""
        self.root_ref = np.asarray(root_rot, float)
        self.ref_trans = None if root_trans is None else np.asarray(root_trans, float)
        return self

    def root_translation(self, trans_cam, scale=None):
        """相机系骨盆位置 -> hips 节点的 glTF 平移 (绝对量, 含 rest 偏移)

        先减参考帧, 把绝对位置变成位移; 再 root_ref^T 转进模板系(消掉人的朝向),
        再乘身高比、用 C 转进 glTF 系; 最后加回 hips 的 rest 平移。

        这里得到的是 hips 节点的 node.translation —— 在 glTF 里那是**父节点坐标系**
        下的量, 且不受 hips 自身旋转影响(glTF 是 T·R·S)。Roxy 里 hips 的父节点
        DeformationSystem 的 rest 旋转是单位阵, 所以它同时也等于全局位移。
        """
        if self.ref_trans is None:
            raise RuntimeError('root_translation 需要参考帧位移; '
                               '先 set_reference(root_rot, trans_cam[0])')
        s = SCALE if scale is None else scale
        d = np.asarray(trans_cam, float) - self.ref_trans
        return self.rest_hips + C @ (s * (self.root_ref.T @ d))

    def retarget(self, root_rot, body_pose, root_ref=None):
        """一帧 -> {节点索引: 局部旋转 3x3}, glTF 系"""
        ref = self.root_ref if root_ref is None else np.asarray(root_ref, float)
        W = self.smpl.fk(root_rot, body_pose)[:, :3, :3]
        S = np.einsum('ij,kjl->kil', W[0].T, W)      # S[j] = W0^T Wj

        R = {}
        for j, tgt in smpl2vrm.items():
            if tgt is None or j == 0:
                continue                             # 关节 0 是根, 单独处理
            n = self.vrm.N(tgt)
            p = self.vrm.parent.get(n)               # 在 Roxy 层级里往上找父节点
            while p is not None and p not in self.node2j:
                p = self.vrm.parent.get(p)
            base = S[self.node2j[p]] if p is not None else np.eye(3)
            R[n] = C @ (base.T @ S[j]) @ C.T         # 累积式: 跳过的关节自动吸收

        # 根: 相对参考帧的朝向。
        # 必须写在循环之后(或者循环里跳过 j == 0): 循环的第一轮就是 j=0, 此时
        # 往上找父节点会走出根 -> base 取单位阵, 而 S[0] = W0^T W0 恰好也是单位阵,
        # 于是 R[hips] 被覆盖成 I —— 角色的转向就全丢了, 只剩原地平移。
        R[self.vrm.N('hips')] = C @ (ref.T @ W[0]) @ C.T
        return R

    def retarget_named(self, root_rot, body_pose, root_ref=None):
        """同上, 但键是 humanoid 名(hips / spine / ...)而不是节点索引"""
        R = self.retarget(root_rot, body_pose, root_ref)
        return {tgt: R[self.vrm.N(tgt)]
                for tgt in smpl2vrm.values() if tgt is not None}


# ================================================================ Blender 转换
class BoneFrames:
    """retarget() 的 glTF 局部旋转 -> Blender 骨骼局部欧拉角

    为什么不能直接 .to_euler():
        返回的 3x3 在 glTF 节点局部系里(glTF 对节点轴没有约定), 而
        rotation_euler 在 Blender 骨骼局部系里(局部 Y 永远沿骨头)。两者差一个
        换基, 这个换基是**常量** —— 两个系刚性长在同一根骨上, 姿态后不变。

    正确形式:
        β_j = T_j^-1 @ K_p @ S @ l_j @ S^T @ K_j^-1  ->  .to_euler('XYZ')
        T_j = b_p^-1 @ b_j            常量, Blender rest (bone.matrix_local)
        K_n = b_n^-1 @ S @ G_n @ S^T  常量, glTF 节点静止系 -> Blender 骨静止系
        S   = S_BLENDER, glTF 全局系 -> armature 系
        l_j = 就是 retarget() 返回的那个
    静止时 l_j = L_j, 这个式子精确给出单位阵(手推 + 实测都验过)。

    踩过的坑: 原先用 basis_j = T_j^-1 @ Q_p^-1 @ T_j @ Q_j, Q_p 是逐帧的父全局
    旋转。它**保旋转角**, 所以只比角的自检(差 0.1°)是发现不了的 —— 但换基的轴
    会被搬歪: 左肘 123° 的弯曲贴到骨头上只剩 33°, 因为轴落到了骨头长轴附近,
    弯曲变成了自转。而且父链还会把根旋转的误差漏给没被点亮的骨。

    用法:
        bf = BoneFrames(arm, vrm_rig)          # 建一次
        eulers = bf.euler(R)                   # {Blender骨名: (rx,ry,rz) 弧度}
    """

    def __init__(self, armature, vrm_rig):
        from mathutils import Matrix           # 只有 Blender 有, 按需导入
        self._Matrix = Matrix
        self.vrm = vrm_rig
        self.rest = {b.name: np.array(b.matrix_local.to_3x3())
                     for b in armature.data.bones}
        self.parent_name = {b.name: (b.parent.name if b.parent else None)
                            for b in armature.data.bones}
        self.K = self._offsets()

    def _offsets(self):
        """K_n = b_n^-1 @ S @ G_n @ S^T —— glTF 节点静止系到 Blender 骨骼静止系的
        常量偏移。两个系刚性长在同一根骨上, 所以姿态后这个偏移不变。"""
        out = {}
        for nm, bm in self.rest.items():
            i = self.vrm.by_name.get(nm, [None])[0]
            if i is None:
                continue
            G = np.asarray(self.vrm.Grest[i], float)
            out[nm] = np.linalg.inv(bm) @ S_BLENDER @ G @ S_BLENDER.T
        return out

    def basis(self, locals_by_node):
        """{节点索引: 3x3} -> {Blender 骨名: 3x3 骨骼局部旋转}"""
        vrm = self.vrm
        Sinv = S_BLENDER.T                          # S 是旋转, 转置即逆
        out = {}
        for i, l in locals_by_node.items():
            nm = _get(vrm.nodes[i], 'name')
            if nm not in self.rest or nm not in self.K:
                continue
            pn = self.parent_name[nm]
            T = np.eye(3) if pn is None else np.linalg.inv(self.rest[pn]) @ self.rest[nm]
            Kp = np.eye(3) if (pn is None or pn not in self.K) else self.K[pn]
            m = Kp @ S_BLENDER @ np.asarray(l, float) @ Sinv @ np.linalg.inv(self.K[nm])
            out[nm] = np.linalg.inv(T) @ m
        return out

    def euler(self, locals_by_node):
        """{节点索引: 3x3} -> {Blender 骨名: (rx,ry,rz) 弧度}"""
        out = {}
        for nm, m in self.basis(locals_by_node).items():
            e = self._Matrix(m.tolist()).to_euler('XYZ')
            out[nm] = (e.x, e.y, e.z)
        return out

    def root_location(self, gltf_trans):
        """hips 节点的 glTF 平移 -> pose_bone.location 的三元组

        pose_bone.location 活在骨骼自己的 rest 系里(局部 Y 沿骨头), 和 glTF 的
        父节点系差两步:
            v_armature = S_BLENDER @ (gltf_trans - rest_gltf)     位移换系
            location   = b_hips.rot^T @ v_armature                进骨头 rest 系
        第二行实测: location=e_x 时 armature 位移恰好是 bone.matrix_local 的第一列。
        """
        i = self.vrm.N('hips')
        nm = self.vrm.name(i)
        if nm not in self.rest:
            raise KeyError('hips 节点 %r 在 Blender 骨架里没有同名骨骼' % (nm,))
        d = np.asarray(gltf_trans, float) - np.asarray(self.vrm.rest_t[i], float)
        v = S_BLENDER @ d
        return tuple(self.rest[nm].T @ v)


# ================================================================ 自检
def _angle_deg(M):
    """3x3 旋转矩阵的旋转角, 度"""
    return float(np.degrees(np.arccos(np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0))))


if __name__ == '__main__':
    import sys
    try:
        import joblib
    except ImportError:
        sys.exit('自检需要 joblib (Blender 的 python 没有); 这条只在普通 python 里跑')

    smpl_path = sys.argv[1] if len(sys.argv) > 1 else 'SMPL_FEMALE_np.npz'
    vrm_file  = sys.argv[2] if len(sys.argv) > 2 else 'Roxy.vrm'
    wham_file = sys.argv[3] if len(sys.argv) > 3 else 'wham_results.pth'

    chunks = joblib.load(wham_file)
    whole = {}
    for k in sorted(chunks):
        for kk, vv in chunks[np.int64(k)].items():
            whole[kk] = vv if kk not in whole else np.concatenate([whole[kk], vv], axis=0)
    root = whole['poses_root_cam'][:, 0]             # (T,3,3)
    body = whole['poses_body']                       # (T,23,3,3)

    rig = Retargeter(SmplRig(smpl_path), VrmRig(vrm_file)).set_reference(root[0])

    # 找根朝向偏离第 0 帧最多的一帧 —— 也就是人转得最厉害的时候
    cos = (np.einsum('tij,kj->tik', root, root[0]).trace(axis1=1, axis2=2) - 1) / 2
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    f = int(np.argmax(ang))
    print('%d 帧; 根朝向偏离第 0 帧最多的是第 %d 帧, %.1f°' % (len(root), f, ang[f]))

    hips = rig.vrm.N('hips')
    ref_R = rig.retarget(root[0], body[0])
    f_R = rig.retarget(root[f], body[f])
    print('帧 0      hips 局部旋转角 = %7.3f°   (参考帧, 应当接近 0)' % _angle_deg(ref_R[hips]))
    print('帧 %-5d hips 局部旋转角 = %7.3f°   (应当接近 %.1f°)' % (f, _angle_deg(f_R[hips]), ang[f]))

    # 关节相对角: 换帧后应当有变化, 但不该整体跟着根一起转
    for name in ('leftUpperArm', 'rightUpperArm', 'spine'):
        n = rig.vrm.N(name)
        print('  %-14s 帧0 %7.2f°   帧%-5d %7.2f°'
              % (name, _angle_deg(ref_R[n]), f, _angle_deg(f_R[n])))
