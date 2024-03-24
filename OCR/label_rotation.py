
import cv2
import os 
from rotation_functions import rotate,hoffman_transformation,pytesseractRotate
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

path = {
        'ENHANCED_IMAGE_FOLDER_PATH': str(os.getenv('ENHANCED_IMAGE_FOLDER_PATH_2')),
        }

import numpy as np

for img_name in os.listdir(path['ENHANCED_IMAGE_FOLDER_PATH']):
    img = cv2.imread(os.path.join(path['ENHANCED_IMAGE_FOLDER_PATH'],img_name))
    original_img = img

    # scale and dialate the image for better result
    img  = cv2.resize(img,None,fx=2.7, fy=3)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.dilate(img,kernel)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, sharpen_kernel)

    #apply hoffman transformation
    rotated_image,angle = hoffman_transformation(img, True)

    original_img  = rotate(original_img,angle)

    # apply tesseract ocd
    rotated_image = pytesseractRotate(rotated_image,original_img,1)


      # save to file
    if rotated_image is not None:
        try: 
            Path('runs').mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment', 'inference')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join('runs', 'segment', 'inference', 'rotated_images')).mkdir(parents=True, exist_ok=True)
        except OSError as error:  
            print(error)
            pass
        
        cv2.imwrite(os.path.join('runs', 'segment', 'inference', 'rotated_images', img_name), rotated_image)