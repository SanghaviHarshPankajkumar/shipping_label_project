#import libraries
import numpy as np
import os
import cv2
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.') / '.env'

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

load_dotenv(dotenv_path=env_path)
path = {
        'ROTATED_IMAGE_FOLDER_PATH': str(os.getenv('ROTATED_IMAGE_FOLDER_PATH')),
        }
# Traves rotated images  
for img_name in os.listdir(path["ROTATED_IMAGE_FOLDER_PATH"]):

    # perform ocr 
    file_name = img_name.split(".")[0]
    result = ocr.ocr(os.path.join(path["ROTATED_IMAGE_FOLDER_PATH"],img_name), cls=True)
    ocr_output_paddle = []
    for i in result:
        ocr_output_paddle.append(" ".join([line[1][0] for line in i]))

    #store ocr in OCR_LABEL_DATA folder
    if result is not None:
        try: 
            Path('runs').mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment', 'inference')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment', 'inference', 'ocr_label_data')).mkdir(parents=True, exist_ok=True)
        except OSError as error:  
            print(error)

    with open(os.path.join('runs', 'segment', 'inference', 'ocr_label_data', img_name) +'.txt',"w+") as f:
        f.write("\n".join(ocr_output_paddle))