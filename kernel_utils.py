import os

import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.augmentations.functional import image_compression
from facenet_pytorch.models.mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor

from torchvision.transforms import Normalize

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)


class VideoReader:
    """Helper class for reading one or more frames from a video file."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset video reader state"""
        if hasattr(self, 'video') and self.video is not None:
            self.video.release()
        self.video = None

    def read_frames(self, video_path, num_frames, jitter=0):
        """Read frames from video with proper error handling"""
        try:
            # Set OpenCV to avoid using threads
            cv2.setNumThreads(0)
            
            # Open video file
            self.reset()
            self.video = cv2.VideoCapture(video_path)
            if not self.video.isOpened():
                print(f"Could not open video: {video_path}")
                return []

            # Get total frames
            total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                print(f"No frames in video: {video_path}")
                return []

            # Read frames
            frames = []
            for _ in range(min(num_frames, total_frames)):
                success, frame = self.video.read()
                if not success:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            return frames

        except Exception as e:
            print(f"Error reading video {video_path}: {str(e)}")
            return []

        finally:
            self.reset()


class FaceExtractor:
    def __init__(self, video_read_fn):
        self.video_read_fn = video_read_fn
        self.detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device="cpu")

    def process_video(self, video_path):
        frames = self.video_read_fn(video_path)
        if len(frames) == 0:
            return []
            
        boxes = []
        for frame in frames:
            frame_boxes = self.detector.detect(Image.fromarray(frame))[0]
            if frame_boxes is not None:
                boxes.append({"faces": [crop_face(frame, box) for box in frame_boxes]})
            else:
                boxes.append({"faces": []})
        return boxes

def crop_face(frame, box):
    box = [int(b) for b in box]  # Convert to regular Python int
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


def confident_strategy(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    # 11 frames are detected as fakes with high probability
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)

strategy = confident_strategy


def put_to_center(img, input_size):
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h:start_h + img.shape[0], start_w: start_w + img.shape[1], :] = img
    return image


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


def predict_on_video(face_extractor, video_path, batch_size, input_size, models, strategy=np.mean,
                    apply_compression=False):
    try:
        faces = face_extractor.process_video(video_path)
        if len(faces) > 0:
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = put_to_center(resized_face, input_size)
                    if apply_compression:
                        resized_face = image_compression(resized_face, quality=90, image_type=".jpg")
                    if n + 1 < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        pass
            if n > 0:
                x = torch.tensor(x).float()
                x = x.permute((0, 3, 1, 2))
                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)
                with torch.no_grad():
                    preds = []
                    for model in models:
                        y_pred = model(x[:n])
                        y_pred = torch.sigmoid(y_pred.squeeze())
                        bpred = y_pred[:n].numpy()
                        preds.append(strategy(bpred))
                    return np.mean(preds)
        return 0.5
    except Exception as e:
        print(f"Prediction error on video {video_path}: {str(e)}")
        return 0.5


def predict_on_video_set(face_extractor, input_size, models, strategy, frames_per_video, videos, num_workers, test_dir):
    predictions = []
    for video in videos:
        try:
            full_path = os.path.join(test_dir, video)
            pred = predict_on_video(
                face_extractor=face_extractor,
                video_path=full_path,
                batch_size=frames_per_video * 4,  # Increased batch size for better detection
                input_size=input_size,
                models=models,
                strategy=strategy
            )
            predictions.append(pred)
        except Exception as e:
            print(f"Error processing {video}: {str(e)}")
            predictions.append(0.5)  # Default prediction for failed videos
            
    return predictions