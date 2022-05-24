#
# Copyright (c) 2022
#

import os
import subprocess
import logging

SERVER_PID = 0

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def set_log_level(level: str):
    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif level == 'INFO':
        logger.setLevel(logging.INFO)
    elif level == 'ERROR':
        logger.setLevel(logging.ERROR)
    elif level == 'WARNING':
        logger.setLevel(logging.WARNING)
    else:
        raise NotImplementedError(f'{level} not supported')

def log_to_file(path: str):
    if os.path.exists(path):
        raise Exception(f'{path} already exists')

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

def run_server():
    global SERVER_PID
    if SERVER_PID:
        raise Exception(f'server[{SERVER_PID}] already runnig')
    SERVER_PID = subprocess.Popen(["python", "prime/server.py"]).pid
    # TODO
    # os.system('python -m server')

def kill_server():
    global SERVER_PID
    if not SERVER_PID:
        raise Exception('server is not running')

    os.kill(SERVER_PID, 9)
