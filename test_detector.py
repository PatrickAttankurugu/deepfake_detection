import os
from deepfake_detector import DeepFakeDetector

def test_detection():
    # Initialize detector
    detector = DeepFakeDetector(weights_dir="models")
    
    # Test on a video
    video_path = "videos/f1.mp4"
    
    if not os.path.exists(video_path):
        print(f"Please place a test video at: {video_path}")
        return
        
    try:
        result = detector.detect(video_path, generate_report=True)
        
        print("\nDeepFake Detection Results:")
        print("-" * 30)
        print(f"Is Fake: {result['is_fake']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Processing Time: {result['processing_time']:.1f} seconds")
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")

if __name__ == "__main__":
    test_detection() 