# DeepFake Detection

A Django package for detecting deepfake videos using deep learning.

## Installation

1. Install the package:
   ```bash
   pip install deepfake-detector
   ```

2. Add to INSTALLED_APPS:
   ```python
   INSTALLED_APPS = [
       ...
       'deepfake_detector',
   ]
   ```

3. Configure settings in settings.py:
   ```python
   DEEPFAKE_DETECTOR = {
       'WEIGHTS_DIR': 'models',
       'LOG_FILE': 'deepfake_detection.log',
       'GENERATE_REPORTS': True
   }
   ```

4. Download model weights:
   ```bash
   python manage.py download_weights
   ```

## Usage
Run the script on a single video:
```bash
python predict_folder.py --video-path path/to/video.mp4
```

The script will generate:
1. A prediction report (prediction_report.txt)
2. Detailed logs (deepfake_detection.log)
```





