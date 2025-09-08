import os
import logging

def setup_logging(log_dir):
    # create a log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # create a log file
    log_file = os.path.join(log_dir, "log.txt")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def log_training_config(logger, training_configs):
    logger.info("Training Configs:")
    for key, value in training_configs.items():
        logger.info(f"{key}: {value}")

def log_model_config(logger, model_configs):
    logger.info("Model Configs:")
    for key, value in model_configs.items():
        logger.info(f"{key}: {value}")

def log_loss(logger, loss):
    logger.info(f"Loss: {loss}")
    