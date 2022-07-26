import logging
from  datetime import datetime


def get_logger(algo_name, env_name, level='INFO'):
    logger = logging.getLogger()
    logger.handlers = []
    file_log_handler = logging.FileHandler("logs/"+algo_name +"_"+ env_name +"_"+ datetime.now().isoformat(timespec='seconds')+'.log')
    logger.addHandler(file_log_handler)
    logger.setLevel(level)
    return logger

