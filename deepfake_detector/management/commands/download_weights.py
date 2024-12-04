# deepfake_detector/management/commands/download_weights.py
from django.core.management.base import BaseCommand
import os
import gdown

class Command(BaseCommand):
    help = 'Downloads required model weights'

    def handle(self, *args, **kwargs):
        weights_dir = 'models'
        os.makedirs(weights_dir, exist_ok=True)
        
        # Model weights with their Google Drive file IDs
        weights = {
            'final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36': 'https://drive.google.com/uc?id=1Q8EDSx1jOFx4SGv90YkEVeVnksADjHcm',
            'final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19': 'https://drive.google.com/uc?id=1ypnKmX7NvNfo6RYcOWZehEDQEHQScs1O',
            'final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31': 'https://drive.google.com/uc?id=1M_VRMvLjC3WLgMjH9eIszC5x7wbSG1YR'
        }

        for name, url in weights.items():
            output = os.path.join(weights_dir, name)
            if not os.path.exists(output):
                self.stdout.write(f'Downloading {name}...')
                gdown.download(url, output, quiet=False)