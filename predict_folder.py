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
from training.zoo.classifiers import DeepFakeClassifier

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
        
        parser = argparse.ArgumentParser("Predict test videos")
        arg = parser.add_argument
        arg('--weights-dir', type=str, default="weights", help="path to directory with checkpoints")
        arg('--models', nargs='+', required=True, help="checkpoint files")
        arg('--test-dir', type=str, required=True, help="path to directory with videos")
        arg('--output', type=str, required=False, help="path to output csv", default="submission.csv")
        args = parser.parse_args()
        
        logging.info(f"Arguments parsed successfully: {vars(args)}")

        if not os.path.exists(args.weights_dir):
            raise FileNotFoundError(f"Weights directory not found: {args.weights_dir}")
        if not os.path.exists(args.test_dir):
            raise FileNotFoundError(f"Test directory not found: {args.test_dir}")
        
        models = []
        model_paths = [os.path.join(args.weights_dir, model) for model in args.models]
        for path in model_paths:
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

        test_videos = sorted([x for x in os.listdir(args.test_dir) if x[-4:] == ".mp4"])
        logging.info(f"Found {len(test_videos)} videos to process")
        
        predictions = predict_on_video_set(
            face_extractor=face_extractor,
            input_size=input_size,
            models=models,
            strategy=strategy,
            frames_per_video=frames_per_video,
            videos=test_videos,
            num_workers=2,
            test_dir=args.test_dir
        )

        submission_df = pd.DataFrame({
            "filename": test_videos,
            "label": predictions,
            "verdict": ["FAKE" if p > 0.5 else "REAL" for p in predictions],
            "confidence": [f"{abs(p - 0.5) * 200:.1f}%" for p in predictions]
        })

        output_path = os.path.abspath(args.output)
        report_path = os.path.abspath(args.output.replace('.csv', '_report.txt'))
        
        logging.info(f"Saving CSV to: {output_path}")
        submission_df.to_csv(output_path, index=False)
        
        logging.info(f"Saving report to: {report_path}")
        with open(report_path, 'w') as f:
            f.write("DeepFake Detection Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Videos Analyzed: {len(test_videos)}\n")
            f.write(f"Processing Time: {time.time() - stime:.1f} seconds\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 40 + "\n")
            
            fakes = submission_df[submission_df.label > 0.5]
            reals = submission_df[submission_df.label <= 0.5]
            
            f.write("\nDetected Fake Videos:\n")
            for _, row in fakes.iterrows():
                f.write(f"• {row['filename']:<20} (Confidence: {row['confidence']})\n")
                
            f.write("\nDetected Real Videos:\n")
            for _, row in reals.iterrows():
                f.write(f"• {row['filename']:<20} (Confidence: {row['confidence']})\n")
                
            f.write("\nNote: Confidence levels above 95% indicate very high certainty in the prediction.\n")

        logging.info(f"Processing completed in {time.time() - stime:.1f} seconds")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()