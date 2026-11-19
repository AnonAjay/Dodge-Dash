"""pose_core: camera -> pose -> floor position. Entry point: capture.CameraPoseSource."""
import os

# Avoids a common Windows camera failure (MSMF hardware transforms). Harmless elsewhere.
# Runs whenever pose_core is imported, so import pose_core BEFORE cv2 in scripts.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
