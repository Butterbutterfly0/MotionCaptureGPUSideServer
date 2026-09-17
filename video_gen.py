import sys
import os
# sys.path.insert(0, '/home/root/.local/lib/python3.13/site-packages')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from process import load_wham_results,get_video_total_frame,merge_wham_results,interp_wham_results
from retarget import VrmRig, SmplRig, Retargeter,BoneFrames
from blender_module import import_vrm, add_cam_to_scene, \
        cam_fit, add_light_to_scene, config_render_video, run_video, change_pose_video
import argparse

if __name__ == '__main__':
    # 1. 提取 '--' 之后的参数
    argv = sys.argv
    if "--" not in argv:
        print("未检测到自定义参数。")
        exit()
    argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--task_id', type=str)
    args = parser.parse_args(argv)
    print()

    result_path = os.path.join( os.path.abspath(args.data_dir), 'wham_results.pth')
    video_path = os.path.abspath(args.video_path)

    frame_count = get_video_total_frame(video_path)
    result = load_wham_results(result_path)
    whole = merge_wham_results(result)
    interp_wham_results(whole, frame_count)
    vrm = VrmRig('Roxy.vrm')
    smpl = SmplRig('SMPL_FEMALE_np.npz')
    retargeter = Retargeter(smpl, vrm)
    retargeter.set_reference(whole['poses_root_cam'][0,0])
    final_pose = []
    arm, posebone =import_vrm('Roxy.vrm')
    add_light_to_scene((50,0,30),3.0,(60,0,-120),1.5)
    cam = add_cam_to_scene()
    cam_fit(cam,3.5)
    bf = BoneFrames(arm, vrm)

    for i in range(frame_count):
        R = bf.euler(retargeter.retarget(whole['poses_root_cam'][:, 0][i],
  whole['poses_body'][i]))
        final_pose.append(R)
    
    OUTPUT_DIR = 'tmp/outputs'
    output_name = args.task_id+'_anime.mp4'
    config_render_video(frame_count,30,720,480,OUTPUT_DIR,output_name,'GPU')
    change_pose_video(arm, posebone,final_pose)
    run_video()