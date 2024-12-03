import argparse
import os
import re
import time
import sys
import logging
import traceback

import torch
import pandas as pd
from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from classifiers import DeepFakeClassifier

# Define the model weights to use
DEFAULT_WEIGHTS = [
    "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
    "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
    "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31"
]

def setup_logging():
    log_file = 'deepfake_detection.log'
    try:
        with open(log_file, 'w') as f:
            f.write('')
            
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging initialized successfully")
    except Exception as e:
        print(f"Failed to setup logging: {str(e)}")
        sys.exit(1)

def main():
    try:
        setup_logging()
        logging.info("Starting DeepFake detection script")
        
        parser = argparse.ArgumentParser("Predict single video")
        arg = parser.add_argument
        arg('--video-path', type=str, required=True, help="path to video file")
        arg('--weights-dir', type=str, default="weights", help="path to directory with checkpoints")
        arg('--output', type=str, required=False, help="path to output file", default="prediction_report.txt")
        args = parser.parse_args()
        
        logging.info(f"Arguments parsed successfully: {vars(args)}")

        if not os.path.exists(args.weights_dir):
            raise FileNotFoundError(f"Weights directory not found: {args.weights_dir}")
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video file not found: {args.video_path}")
        
        models = []
        for weight in DEFAULT_WEIGHTS:
            path = os.path.join(args.weights_dir, weight)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            logging.info(f"Loading model: {path}")
            model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
            model.eval()
            del checkpoint
            models.append(model)
            logging.info(f"Successfully loaded model: {path}")

        frames_per_video = 32
        video_reader = VideoReader()
        video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
        face_extractor = FaceExtractor(video_read_fn)
        input_size = 380
        strategy = confident_strategy
        stime = time.time()

        video_name = os.path.basename(args.video_path)
        predictions = predict_on_video_set(
            face_extractor=face_extractor,
            input_size=input_size,
            models=models,
            strategy=strategy,
            frames_per_video=frames_per_video,
            videos=[video_name],
            num_workers=1,
            test_dir=os.path.dirname(args.video_path)
        )

        prediction = predictions[0]
        verdict = "FAKE" if prediction > 0.5 else "REAL"
        confidence = abs(prediction - 0.5) * 200

        report_path = os.path.abspath(args.output)
        logging.info(f"Saving report to: {report_path}")
        
        with open(report_path, 'w') as f:
            f.write("DeepFake Detection Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video Analyzed: {video_name}\n")
            f.write(f"Processing Time: {time.time() - stime:.1f} seconds\n\n")
            
            f.write("Results:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Verdict: {verdict}\n")
            f.write(f"Confidence: {confidence:.1f}%\n")
            f.write("\nNote: Confidence levels above 95% indicate very high certainty in the prediction.\n")

        logging.info(f"Processing completed in {time.time() - stime:.1f} seconds")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()