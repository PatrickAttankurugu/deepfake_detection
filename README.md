# DeepFake Detection

A streamlined tool for detecting deepfake videos using deep learning.

## Setup
1. Create a virtual environment and activate it
2. Install requirements: `pip install -r requirements.txt`
3. Download model weights and place them in the `weights/` directory

## Usage
Run the script on a single video:
```bash
python predict_folder.py --video-path path/to/video.mp4
```

The script will generate:
1. A prediction report (prediction_report.txt)
2. Detailed logs (deepfake_detection.log)
```





