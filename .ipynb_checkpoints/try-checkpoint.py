from pathlib import Path

from wham_service import run, WHAM_DIR

results, tracking, slam = run(
    WHAM_DIR / "examples" / "IMG_9732.mov",
    output_dir=Path("tmp/outputs/smoke"),
    visualize=True
)
