import logging
import os
from datetime import datetime

LOG_FILE =f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs",f"{datetime.now().strftime('%d_%m_%Y')}-logs")
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] - %(filename)s - %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
    
)