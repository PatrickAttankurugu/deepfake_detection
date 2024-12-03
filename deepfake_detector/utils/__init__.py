from typing import Dict, Any
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

def get_detector_settings() -> Dict[str, Any]:
    """
    Get DeepFake detector settings from Django settings.
    
    Returns:
        Dict containing configuration settings
    
    Raises:
        ImproperlyConfigured: If required settings are missing
    """
    default_settings = {
        'WEIGHTS_DIR': 'models',
        'LOG_DIR': 'logs',
        'GENERATE_REPORTS': False,
        'REPORT_DIR': 'reports'
    }
    
    user_settings = getattr(settings, 'DEEPFAKE_DETECTOR', {})
    
    if not user_settings.get('WEIGHTS_DIR'):
        raise ImproperlyConfigured(
            "DEEPFAKE_DETECTOR['WEIGHTS_DIR'] is required in settings.py"
        )
    
    default_settings.update(user_settings)
    return default_settings 
