"""运行独立 STL 查看器。"""
import subprocess
import sys


def main():
    from pathlib import Path
    viewer_path = Path(__file__).resolve().parents[1] / "viewer.py"
    subprocess.run([sys.executable, str(viewer_path)])


if __name__ == "__main__":
    main()
