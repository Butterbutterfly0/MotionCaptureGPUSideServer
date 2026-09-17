"""把 WHAM 当作模块使用。

WHAM 不是按包结构写的：它内部全是顶层绝对导入（`from configs.config import ...`、
`from lib.models import ...`），设计上是「cd 进 WHAM 目录当脚本跑」。
所以这里做两件事：

1. 把 WHAM 根目录注入 sys.path —— 这样它的内部导入才能解析。只注入一次。
2. 提供进程内单例 —— 权重约 5.8GB，必须加载一次常驻显存，不能每个请求重来。

用法：
    from wham_service import run
    results, tracking, slam = run("/abs/path/video.mov", output_dir="/abs/out")
"""

import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WHAM_DIR = PROJECT_ROOT / "WHAM"


def _ensure_sys_path():
    """把 WHAM 根目录放到 sys.path 最前面。

    注意：这会让 `configs` / `lib` / `dataset` 这些通用顶层包名解析到 WHAM 那边。
    本项目自身没有同名模块，暂时无冲突；若以后引入同名的第三方包需留意。

    这里刻意用 insert(0) 且只做一次，避免出现 WHAM 被加载两份的情况
    （一份 WHAM.configs、一份 configs，模块级状态不共享，极难排查）。
    """
    p = str(WHAM_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_sys_path()

# 必须在 sys.path 注入之后才能导入 —— WHAM 的内部导入依赖它
from wham_api import WHAM_API  # noqa: E402


_model = None
_model_lock = threading.Lock()


def load_model():
    """加载 WHAM 模型，进程内单例。

    FastAPI 的 BackgroundTasks 跑在线程池里，所以这里的锁是必要的 ——
    否则并发的头几个请求会各自加载一遍权重，直接把显存撑爆。
    首次调用耗时以分钟计，建议在服务启动时预热（见 server.py）。
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:          # 双重检查：拿到锁后再确认一次
                _model = WHAM_API()
    return _model


def run(video_path, output_dir=None, run_global=True, visualize=False):
    """跑一次 WHAM。

    Args:
        video_path: 视频绝对路径。WHAM 内部用 cv2 读，相对路径会按当前 CWD 解析，
                    这里强制转成绝对路径，避免调用方的 CWD 影响结果。
        output_dir: 中间结果（tracking/slam/wham_results.pth）的输出目录。
                    传 None 时 WHAM 会落到自己目录下的 output/demo。
        run_global: 是否跑 SLAM 估计全局坐标。DPVO 没装好时 WHAM 会自动降级。
        visualize: 是否渲染可视化视频（走 pytorch3d，较慢）。

    Returns:
        (results, tracking_results, slam_results)
        results: dict[track_id] -> poses_body / betas / verts_cam / trans_world ...
    """
    return load_model()(
        str(Path(video_path).resolve()),
        output_dir=str(Path(output_dir).resolve()) if output_dir else None,
        run_global=run_global,
        visualize=visualize,
    )
