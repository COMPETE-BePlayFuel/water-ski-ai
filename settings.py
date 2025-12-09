from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
MULTIPLE_VIDEOS_2 = 'Multiple Videos (Scoring)'
MULTIPLE_VIDEOS = 'Multiple Videos Data Comparison'
DEEP_ANALYSIS = 'Deep AI Detections'

SOURCES_LIST = [IMAGE, VIDEO, MULTIPLE_VIDEOS_2, MULTIPLE_VIDEOS, DEEP_ANALYSIS]


# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / '2024-02-14 (4).png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / '2024-02-14 (4).png'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    #'video_2': VIDEO_DIR / 'file.mp4',
    'video_1': VIDEO_DIR / 'Ski_Vid_test.mp4',

}

# Multiple Videos config
MULTIPLE_VIDEOS_DIR = ROOT / 'videos'
MULTIPLE_VIDEOS_DICT = {
    'video_4': VIDEO_DIR / 'file1.mp4',
    'video_2': VIDEO_DIR / 'file1.mp4',
    'video_3': VIDEO_DIR / 'file1.mp4',
    'video_1': VIDEO_DIR / '0326.mp4',
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'
DETECTION_MODEL = MODEL_DIR / 'yolov8n-pose.pt'


#SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n.pt'

