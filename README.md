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
pip install git+https://github.com/PatrickAttankurugu/deepfake_detection.git@django-package

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
2. Run this script to download the model weights

```python
import os
import gdown

def download_weights():
    weights_dir = 'models'
    os.makedirs(weights_dir, exist_ok=True)
    
    weights = {
        'final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36': 'https://drive.google.com/uc?id=1Q8EDSx1jOFx4SGv90YkEVeVnksADjHcm',
        'final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19': 'https://drive.google.com/uc?id=1ypnKmX7NvNfo6RYcOWZehEDQEHQScs1O',
        'final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31': 'https://drive.google.com/uc?id=1M_VRMvLjC3WLgMjH9eIszC5x7wbSG1YR'
    }

    for name, url in weights.items():
        output = os.path.join(weights_dir, name)
        if not os.path.exists(output):
            print(f'Downloading {name}...')
            gdown.download(url, output, quiet=False)

if __name__ == "__main__":
    download_weights()
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

