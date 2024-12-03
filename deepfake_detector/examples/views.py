from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from deepfake_detector import DeepFakeDetector
import os

@csrf_exempt
def analyze_video(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
        
    video_file = request.FILES.get('video')
    if not video_file:
        return JsonResponse({'error': 'No video file provided'}, status=400)
        
    # Save uploaded file temporarily
    temp_path = f'temp_{video_file.name}'
    with open(temp_path, 'wb+') as destination:
        for chunk in video_file.chunks():
            destination.write(chunk)
            
    try:
        detector = DeepFakeDetector()
        result = detector.detect(temp_path)
        return JsonResponse(result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path) 