import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minicpm_robot_track.precompute import main


if __name__ == "__main__":
    main()
