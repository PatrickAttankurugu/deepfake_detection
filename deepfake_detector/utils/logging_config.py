import logging
import sys
import os

def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'deepfake_detection.log')
    
    logger = logging.getLogger('deepfake_detector')
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger 
