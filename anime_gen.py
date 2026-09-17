import subprocess

cmd = ['blender', '--background', '--python', 'video_gen.py', '--', '--data_dir', 'data', '--video_path', 'output.mp4', '--task_id', 'task']

DATA_DIR_INDEX = cmd.index("data")
VIDEO_PATH_INDEX = cmd.index("output.mp4")
TASK_ID_INDEX = cmd.index("task")

def anime_video_gen(data_dir, video_path, task_id):
    cmd_copy = cmd.copy()
    cmd_copy[DATA_DIR_INDEX] = data_dir
    cmd_copy[VIDEO_PATH_INDEX] = video_path
    cmd_copy[TASK_ID_INDEX] = task_id
    subprocess.run(cmd_copy)
    
if __name__ == '__main__':
    anime_video_gen("tmp/outputs/73019135-244d-4946-8791-cdd3b85cc570/","tmp/73019135-244d-4946-8791-cdd3b85cc570.mp4","73019135-244d-4946-8791-cdd3b85cc570")