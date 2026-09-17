import bpy

from typing import Tuple
import math
from mathutils import Vector, Euler
import os

def import_vrm(vrm_path:str):
    '''
        对环境进行重置,加载vrm模型,获取模型的骨架和姿态骨骼并返回
    '''
    # 删除当前.blender场景中的所有对象
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    # 导入vrm模型
    bpy.ops.import_scene.vrm(filepath=vrm_path)
    # 获取导入模型的骨架
    armature = None
    pose_bone = None
    try:
        armature = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
        # 获取姿态骨架
        pose_bone = armature.pose.bones
    except IndexError as e:
        raise ValueError("VRM model Invaild")
    
    return armature, pose_bone

def get_bone_name(armature):
    bones = [ bone for bone in armature.data.bones ]
    return bones

def add_cam_to_scene():
    '''
        向场景中添加一个相机并进行绑定，并返回相机的引用
    '''
    camera_data = bpy.data.cameras.new('Cam')
    cam = bpy.data.objects.new('Cam', camera_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam

def cam_fit(cam, far = 2.4, down = 0.1):
    meshes = [ o for o in bpy.data.objects if o.type == 'MESH']
    pts = [ m.matrix_world @ Vector(c) for m in meshes for c in m.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    ctr, size = (lo + hi) / 2, max(hi[i]-lo[i] for i in range(3))
    cam.location = ctr + Vector((0, -size*far, size*down))
    cam.rotation_euler = (ctr - cam.location).normalized().to_track_quat('-Z','Y').to_euler()


def add_light_to_scene(Sun_ang:Tuple[float,float,float], Sun_intensity:float,
                Fill_ang:Tuple[float, float, float], Fill_intensity:float):
    for name, rot, en in (  ('Sun', [math.radians(ang) for ang in Sun_ang], Sun_intensity),
                            ('Fill', [math.radians(ang) for ang in Fill_ang], Fill_intensity) ):
        ld = bpy.data.lights.new(name, 'SUN')
        ld.energy = en
        lo_ = bpy.data.objects.new(name, ld)
        # print(rot)
        lo_.rotation_euler = rot
        bpy.context.scene.collection.objects.link(lo_)

def config_render_single_picture(width:int, height:int, filepath:str, device:str = 'CPU', format:str = 'PNG'):
    sc = bpy.context.scene
    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.image_settings.file_format = format
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.filepath = filepath
    sc.render.engine = 'CYCLES'
    sc.cycles.device = device
    sc.cycles.samples = 48
    sc.view_settings.view_transform = 'Standard'

def config_render_video(nFrame, fps, width, height,output_dir='.',save_name='output.png',device='CPU'):
    sc = bpy.context.scene
    sc.frame_start = 1
    sc.frame_end = nFrame
    sc.render.fps = fps
    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.resolution_percentage = 100
    # sc.render.film_transparent = True
    sc.render.use_persistent_data = True
    sc.render.image_settings.media_type = 'VIDEO'
    sc.render.image_settings.file_format = 'FFMPEG'
    sc.render.image_settings.color_mode = 'RGB'

    sc.render.ffmpeg.format = 'MPEG4'
    sc.render.ffmpeg.codec = 'H264'
    sc.render.ffmpeg.constant_rate_factor = 'HIGH'
    
    os.makedirs(output_dir, exist_ok=True)
    sc.render.filepath = os.path.join(output_dir,save_name)

    sc.render.engine = 'CYCLES'
    sc.cycles.device = device
    sc.cycles.samples = 16
    sc.cycles.use_denoising = False
    sc.view_settings.view_transform = 'Standard'


def run_single_image():
    bpy.ops.render.render(write_still=True)

def run_video():
    bpy.ops.render.render(animation=True)

V2B = {
    'hips': 'Root_M', 'spine': 'Spine1_M', 'Chest_M': 'Chest_M',
    'neck': 'Neck_M', 'head': 'Head_M',
    'leftShoulder': 'Scapula_L', 'leftUpperArm': 'Shoulder_L',
    'leftLowerArm': 'Elbow_L',   'leftHand': 'Wrist_L',
    'rightShoulder': 'Scapula_R', 'rightUpperArm': 'Shoulder_R',
    'rightLowerArm': 'Elbow_R',   'rightHand': 'Wrist_R',
    'leftUpperLeg': 'Hip_L',  'leftLowerLeg': 'Knee_L',
    'leftFoot': 'Ankle_L',    'leftToes': 'Toes_L',
    'rightUpperLeg': 'Hip_R', 'rightLowerLeg': 'Knee_R',
    'rightFoot': 'Ankle_R',   'rightToes': 'Toes_R',
}

def change_pose(pose_bone, pose):
    for key, euler in pose.items():
        name = V2B[key]
        b = pose_bone.get(name)
        if b is None:
            continue
        b.rotation_mode = 'XYZ'
        b.rotation_euler = Euler(euler,'XYZ')
    bpy.context.view_layer.update()

def change_pose_video(armature, pose_bone, poses):
    for key in V2B.values():
        b = pose_bone.get(key)
        if b: b.rotation_mode = 'XYZ'
    for f, pose in enumerate(poses):
        for bone, euler in pose.items():
            b = pose_bone.get(bone)
            if b is None:
                continue
            b.rotation_euler = Euler(euler, 'XYZ')
            b.keyframe_insert(data_path='rotation_euler',frame=f+1)
    def iter_fcurves(action):
        if hasattr(action, 'fcurves'):
            yield from action.fcurves
            return
        for layer in getattr(action, 'layers', ()):
            for strip in getattr(layer, 'strips', ()):
                for cb in getattr(strip, 'channelbags', ()):
                    yield from cb.fcurves
    for fc in iter_fcurves(armature.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
