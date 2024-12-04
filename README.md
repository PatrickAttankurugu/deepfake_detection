# DeepFake Detection

A Django package for detecting deepfake videos using deep learning. This package uses an ensemble of EfficientNet models trained on large-scale deepfake datasets.

## Features
- High accuracy deepfake detection using multiple EfficientNet models
- Easy django integration
- REST API endpoint for video analysis
- Detailed analysis reports and logging
- Support for various video formats

## Installation

### For Production Use
1. Install the package:

bash
pip install deepfake-detector

2. Add to INSTALLED_APPS in settings.py:

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

4. Include the URLs in your project's urls.py:

```python
urlpatterns = [
    ...
    path('api/deepfake/', include('deepfake_detector.examples.urls')),
]
```

### Internal
Install directly from this repo:

```bash
pip install git+https://github.com/PatrickAttankurugu/deepfake_detection.git@django-package
```

## Usage

### As a Django App
1. Download model weights:

```bash
python download_models.py
```

2. Send POST requests to the API endpoint:

```python
import requests

url = 'http://your-domain/api/deepfake/analyze/'
files = {'video': open('path/to/video.mp4', 'rb')}
response = requests.post(url, files=files)
result = response.json()
```

### As a Python Package

```python
from deepfake_detector import DeepFakeDetector

detector = DeepFakeDetector(weights_dir="models")
result = detector.detect("path/to/video.mp4", generate_report=True)

print(f"Is Fake: {result['is_fake']}")
print(f"Confidence: {result['confidence']:.1f}%")
```

## Output
The detector generates:
1. A prediction report (prediction_report.txt) containing:
   - Analysis timestamp
   - Video details
   - Detection verdict
   - Confidence score
   - Processing time
2. Detailed logs (deepfake_detection.log)

## Requirements
- Python >= 3.8
- Django >= 3.2.0
- PyTorch >= 1.7.0
- Other dependencies listed in requirements.txt

## Testing
Run the test suite:

```bash
python -m deepfake_detector.tests.test_detector
```

## License
Proprietary - For internal use only

## Support
Please contact the development team.

