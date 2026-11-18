"""pose_core: camera -> pose -> floor position. Entry point: capture.CameraPoseSource."""
import os

# Avoids a common Windows camera failure (MSMF hardware transforms). Harmless elsewhere.
# Runs whenever pose_core is imported, so import pose_core BEFORE cv2 in scripts.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

from pose_core.blazer_matcher import (
    BlazerMatchResult,
    BlazerPoseTemplate,
    evaluate_blazer_pose,
    load_blazer_catalog,
    render_reference_card,
    sample_game_poses,
    save_blazer_catalog,
)
from pose_core.blazer_indexer import BlazerIndexer
