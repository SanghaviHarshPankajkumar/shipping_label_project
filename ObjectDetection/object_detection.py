import cv2
from ultralytics import YOLO 
import os
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
from functions import cropBlackBackground, enhanceImage, generateMask 

path = {
        'SEG_MODEL_PATH': str(os.getenv('SEG_MODEL_PATH')),
        'DET_MODEL_PATH': str(os.getenv('DET_MODEL_PATH')),
        'IMG_DIR_PATH': str(os.getenv('IMG_DIR_PATH')),
        'INFERENCE_FOLDER': str(os.getenv('INFERENCE_FOLDER')),
        }

#import models
seg_model = YOLO(path['SEG_MODEL_PATH'])
det_model = YOLO(path['DET_MODEL_PATH'])

CONF = 0.7  


# do inference for detection and store croped images in folder
for img in os.listdir(path['IMG_DIR_PATH']):
    img_file = cv2.imread(os.path.join(path['IMG_DIR_PATH'],img),0)
    cv2.imwrite(os.path.join(path['IMG_DIR_PATH'],img),img_file)
    
    det_model(os.path.join(path['IMG_DIR_PATH'],img),conf = CONF, save=True , save_crop=True , name=path['INFERENCE_FOLDER'],exist_ok = True)

#do inference for image segmentation and store image in folder
for img in os.listdir(path['IMG_DIR_PATH']):
    img_file = cv2.imread(os.path.join(path['IMG_DIR_PATH'],img),0)
    cv2.imwrite(os.path.join(path['IMG_DIR_PATH'],img),img_file)
    
    result = seg_model(os.path.join(path['IMG_DIR_PATH'],img),conf = CONF,save = True,name = path['INFERENCE_FOLDER'],exist_ok = True)
    
    original_img = cv2.imread(os.path.join(path['IMG_DIR_PATH'],img))
    for res in result:
        crop_img, mask = generateMask(res, original_img)
        
        image = None
        if crop_img is not None:
            # Convert to gray scale image
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            crop_img = cropBlackBackground(crop_img)
            image = enhanceImage(crop_img)
        
        # save to file
        if image is not None:
            try: 
                Path('runs').mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment', 'inference')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment', 'inference', 'crops_seg')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment', 'inference', 'enhanced')).mkdir(parents=True, exist_ok=True)
                Path(os.path.join('runs', 'segment', 'inference', 'masks')).mkdir(parents=True, exist_ok=True)
            except OSError as error:  
                print(error)
                pass
            
            cv2.imwrite(os.path.join('runs', 'segment', 'inference', 'masks', img), mask)
            cv2.imwrite(os.path.join('runs', 'segment', 'inference', 'crops_seg', img), crop_img )
            cv2.imwrite(os.path.join('runs', 'segment', 'inference', 'enhanced', img), image )
