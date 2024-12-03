import os
import re
import time
import logging
import torch
from typing import Dict, List, Optional, Union
import numpy as np

from .video_processor import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from .model import DeepFakeClassifier
from ..utils.logging_config import setup_logging

class DeepFakeDetector:
    """
    A class for detecting deepfake videos using multiple EfficientNet models.
    
    Attributes:
        weights_dir (str): Directory containing model weights
        logger (Logger): Logger instance for tracking operations
        frames_per_video (int): Number of frames to process per video
        input_size (int): Input size for the neural network
    """

    DEFAULT_WEIGHTS: List[str] = [
        "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
        "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
        "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31"
    ]

    def __init__(self, weights_dir: str = "models", logger: Optional[logging.Logger] = None) -> None:
        self.weights_dir = weights_dir
        self.logger = logger or setup_logging()
        self.models = self._load_models()
        self.frames_per_video = 32
        self.input_size = 380
        
    def _load_models(self) -> List[DeepFakeClassifier]:
        models = []
        for weight in self.DEFAULT_WEIGHTS:
            path = os.path.join(self.weights_dir, weight)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            self.logger.info(f"Loading model: {path}")
            model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
            model.eval()
            models.append(model)
        return models

    def detect(self, video_path: str, generate_report: bool = False) -> Dict[str, Union[bool, float]]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        stime = time.time()
        video_reader = VideoReader()
        video_read_fn = lambda x: video_reader.read_frames(x, num_frames=self.frames_per_video)
        face_extractor = FaceExtractor(video_read_fn)

        video_name = os.path.basename(video_path)
        predictions = predict_on_video_set(
            face_extractor=face_extractor,
            input_size=self.input_size,
            models=self.models,
            strategy=confident_strategy,
            frames_per_video=self.frames_per_video,
            videos=[video_name],
            num_workers=1,
            test_dir=os.path.dirname(video_path)
        )

        prediction = predictions[0]
        confidence = abs(prediction - 0.5) * 200
        processing_time = time.time() - stime

        result = {
            "is_fake": prediction > 0.5,
            "confidence": confidence,
            "raw_score": prediction,
            "processing_time": processing_time
        }

        if generate_report:
            self._generate_report(video_name, result)

        return result

    def _generate_report(self, video_name: str, result: Dict[str, Union[bool, float]]) -> None:
        report_path = "prediction_report.txt"
        self.logger.info(f"Saving report to: {report_path}")
        
        with open(report_path, 'w') as f:
            f.write("DeepFake Detection Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video Analyzed: {video_name}\n")
            f.write(f"Processing Time: {result['processing_time']:.1f} seconds\n\n")
            
            f.write("Results:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Verdict: {'FAKE' if result['is_fake'] else 'REAL'}\n")
            f.write(f"Confidence: {result['confidence']:.1f}%\n")
            f.write("\nNote: Confidence levels above 95% indicate very high certainty in the prediction.\n")