from django.apps import AppConfig

class DeepfakeDetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'deepfake_detector'
    verbose_name = 'DeepFake Detection'

    def ready(self):
        # Perform any initialization here
        pass
