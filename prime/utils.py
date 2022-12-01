#
# Copyright (c) 2022
#

import os
import subprocess
import logging
from typing import Optional


IS_SERVER = False
SERVER_PID = 0

MAX_MESSAGE_LENGTH = 100 * 1024 * 1024

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(filename)16s:%(lineno)4s - %(funcName)20s()] %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

last_filehandler = None

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

    global last_filehandler
    if last_filehandler is not None:
        # Support logging to only one file
        logger.removeHandler(last_filehandler)

    last_filehandler = file_handler
    logger.addHandler(file_handler)

def is_server():
    global IS_SERVER
    IS_SERVER = True

def run_server(port: Optional[int] = None, ci: Optional[str] = None,
               ll: Optional[str] = None):
    if IS_SERVER: return

    global SERVER_PID
    if SERVER_PID:
        raise Exception(f'server[{SERVER_PID}] already runnig')

    port = [ "--port", str(port) ] if port else []
    ci = ["--ci", ci] if ci else []
    ll = ["--ll", ll] if ll else []

    # TODO remove this
    pwd = os.environ['PWD']
    SERVER_PID = subprocess.Popen(["python",
                                   f"{pwd}/prime/server.py"]
                                  + port + ci + ll).pid
    # TODO
    # os.system('python -m server')

def kill_server():
    global SERVER_PID
    if not SERVER_PID:
        raise Exception('server is not running')

    os.kill(SERVER_PID, 9)
    SERVER_PID = 0
